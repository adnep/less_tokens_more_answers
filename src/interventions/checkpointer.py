"""
Sparse KV cache checkpointing for efficient backtracking.

Problem
-------
When we backtrack to position bt, we need the model's KV cache to be in the
state it was at position bt (covering prompt + bt generated tokens).
Without checkpointing, we must re-run a full forward pass on all bt tokens
(O(bt) cost).  With checkpointing, we restore the nearest saved checkpoint
and replay only the short gap since then (O(interval) cost).

Memory
------
The KV cache at position k for Qwen3-4B has shape:
  num_layers × 2 (K,V) × [1, n_kv_heads, prompt_len+k, head_dim]
  = 36 × 2 × [1, 8, prompt_len+k, 128]

In bfloat16 this is ~147 KB × (prompt_len + k).
At k=4096 (prompt_len≈150): ~630 MB per checkpoint.

Storing multiple checkpoints on GPU quickly exhausts VRAM.  The solution is
**CPU offloading**: checkpoints are pinned to CPU RAM (typically ≥256 GB on
Metacentrum), leaving GPU free for the live generation KV cache and model.
A PCIe Gen3 restore (600 MB) takes ~40 ms — negligible vs the alternative of
recomputing thousands of tokens.

If you have plenty of VRAM and want zero transfer latency, set offload_to_cpu=False.

Lifecycle
---------
1. engine calls  checkpointer.should_save(step)  → bool
2. if True:      checkpointer.save(step, out.past_key_values, logits_1d)
3. on backtrack: ckpt = checkpointer.get_best_before(bt)
                 past_kv, logits = checkpointer.restore(ckpt, device)
4. after backtrack: checkpointer.prune_after(bt)   # free discarded-token memory
5. on generate():   checkpointer.reset()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import torch


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """
    Saved model state just BEFORE generating token at `position`.

    past_key_values covers  prompt_tokens + position  tokens.
    logits[vocab_size]  are the predictions for the token at `position`
    (i.e. what the model returns as `out.logits[:, -1, :]` at that step).

    Tensors may be on CPU (if offloaded) or GPU depending on engine config.
    """
    position: int
    past_key_values: Any           # Tuple[Tuple[Tensor, Tensor], ...]
    logits: torch.Tensor           # [vocab_size]


# ── KV clone helpers ───────────────────────────────────────────────────────────

def _clone_kv_cpu(past_key_values) -> Any:
    """Clone KV cache tensors to CPU (pinned for fast DMA transfer back).

    Works for both legacy tuple-of-tuples and DynamicCache (HF 4.36+) because
    DynamicCache.__iter__ yields (key, value) per layer — same iteration shape.
    We always store as plain tuple internally; restored as DynamicCache if needed.
    """
    return tuple(
        tuple(t.detach().cpu() for t in layer)
        for layer in past_key_values
    )


def _clone_kv_gpu(past_key_values) -> Any:
    """Clone KV cache tensors, staying on GPU."""
    return tuple(
        tuple(t.detach().clone() for t in layer)
        for layer in past_key_values
    )


def _kv_to_device(past_key_values, device: str) -> Any:
    """Move (CPU-offloaded) KV cache back to device as a plain tuple."""
    return tuple(
        tuple(t.to(device, non_blocking=True) for t in layer)
        for layer in past_key_values
    )


def _as_dynamic_cache(kv_tuple: Any) -> Any:
    """
    Wrap a saved tuple-of-tuples KV cache back into a DynamicCache object.

    HF transformers 4.36+ requires past_key_values to be a DynamicCache
    subclass — passing a plain tuple causes AttributeError on .get_seq_length().

    Compatibility
    -------------
    HF < 4.36 : plain tuple is fine → returned as-is.
    HF 4.36–4.x : DynamicCache used key_cache / value_cache list attributes.
    HF 5.x     : DynamicCache accepts ddp_cache_data=[(k, v), ...] in __init__.

    We try the newest API first and fall back through older ones.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return kv_tuple   # HF < 4.36

    # ── HF 5.x API: ddp_cache_data constructor ───────────────────────────────
    try:
        return DynamicCache(ddp_cache_data=list(kv_tuple))
    except TypeError:
        pass   # older DynamicCache doesn't have ddp_cache_data

    # ── HF 4.36–4.x API: key_cache / value_cache lists ───────────────────────
    try:
        cache = DynamicCache()
        cache.key_cache   = [layer[0] for layer in kv_tuple]
        cache.value_cache = [layer[1] for layer in kv_tuple]
        cache._seen_tokens = kv_tuple[0][0].shape[-2] if kv_tuple else 0
        return cache
    except Exception:
        pass

    # ── Fallback: return tuple and hope the model handles it ──────────────────
    return kv_tuple


# ── Checkpointer ───────────────────────────────────────────────────────────────

class KVCheckpointer:
    """
    Sparse KV cache store with optional CPU offloading.

    Args:
        interval:        Save a checkpoint every N generated tokens.
                         Smaller = more frequent checkpoints, less recomputation
                         on restore, but more memory.
                         Recommended: 64–256 (match your typical backtrack distance).
        max_checkpoints: Keep at most this many checkpoints (0 = unlimited).
                         When the limit is reached the OLDEST checkpoint is
                         evicted (LRU approximation — oldest = furthest back).
                         Useful when backtrack distances are bounded and you want
                         a hard cap on memory.  0 = keep all.
        offload_to_cpu:  Store checkpoint tensors in CPU RAM instead of GPU VRAM.
                         Strongly recommended for long sequences.  Restore incurs
                         a PCIe transfer (~40 ms for 600 MB) but saves VRAM.
    """

    def __init__(
        self,
        interval: int = 128,
        max_checkpoints: int = 0,
        offload_to_cpu: bool = True,
    ):
        self.interval = interval
        self.max_checkpoints = max_checkpoints
        self.offload_to_cpu = offload_to_cpu
        self._checkpoints: Dict[int, Checkpoint] = {}

    # ── Saving ─────────────────────────────────────────────────────────────────

    def should_save(self, position: int) -> bool:
        """True when this position falls on a checkpoint interval boundary."""
        return position % self.interval == 0

    def save(
        self,
        position: int,
        past_key_values: Any,
        logits_1d: torch.Tensor,
    ) -> None:
        """
        Save model state at `position`.

        Args:
            position:         Generated-token index (0-based from start of generation).
            past_key_values:  Model KV cache from `out.past_key_values` at this step.
                              Covers prompt + position tokens.
            logits_1d:        `out.logits[:, -1, :].squeeze(0)` — predictions for
                              the token AT this position.  Shape [vocab_size].
        """
        if self.offload_to_cpu:
            kv = _clone_kv_cpu(past_key_values)
            lg = logits_1d.detach().cpu()
        else:
            kv = _clone_kv_gpu(past_key_values)
            lg = logits_1d.detach().clone()

        self._checkpoints[position] = Checkpoint(
            position=position,
            past_key_values=kv,
            logits=lg,
        )

        # Evict oldest if over limit
        if self.max_checkpoints > 0 and len(self._checkpoints) > self.max_checkpoints:
            oldest_pos = min(self._checkpoints)
            del self._checkpoints[oldest_pos]

    # ── Restoring ──────────────────────────────────────────────────────────────

    def get_best_before(self, target: int) -> Optional[Checkpoint]:
        """
        Return the checkpoint with the largest position ≤ target.

        Returns None if no checkpoint exists at or before target.
        """
        candidates = {p: c for p, c in self._checkpoints.items() if p <= target}
        if not candidates:
            return None
        return candidates[max(candidates)]

    def restore(
        self,
        checkpoint: Checkpoint,
        device: str,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Return (past_key_values, logits) on `device`, ready for model use.

        If tensors were offloaded to CPU, they are moved to `device` here.
        The checkpoint itself is NOT modified (can be restored again).
        """
        if self.offload_to_cpu:
            past_kv = _kv_to_device(checkpoint.past_key_values, device)
            logits   = checkpoint.logits.to(device)
        else:
            past_kv = checkpoint.past_key_values
            logits   = checkpoint.logits
        # Wrap as DynamicCache if HF 4.36+ requires it (plain tuple → AttributeError)
        return _as_dynamic_cache(past_kv), logits

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def prune_after(self, position: int) -> None:
        """
        Delete all checkpoints with position > `position`.

        Call this after committing to a backtrack to `position`, to release
        memory for the tokens we just discarded.
        """
        to_del = [p for p in list(self._checkpoints) if p > position]
        for p in to_del:
            del self._checkpoints[p]

    def reset(self) -> None:
        """Clear all checkpoints.  Called at the start of each generate() call."""
        self._checkpoints.clear()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def n_checkpoints(self) -> int:
        return len(self._checkpoints)

    def memory_mb(self) -> float:
        """Rough estimate of total tensor bytes across all checkpoints (MB)."""
        total = 0
        for ckpt in self._checkpoints.values():
            for layer in ckpt.past_key_values:
                for t in layer:
                    total += t.nelement() * t.element_size()
            total += ckpt.logits.nelement() * ckpt.logits.element_size()
        return total / 1e6

    def saved_positions(self) -> list:
        return sorted(self._checkpoints.keys())
