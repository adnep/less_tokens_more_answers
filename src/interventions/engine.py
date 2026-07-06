"""
InterventionEngine — token-by-token LLM generation with mid-stream interventions
(backtracking, early-stop, token injection, sampling-parameter changes).

Framework
---------
Uses HuggingFace transformers (AutoModelForCausalLM) directly, NOT vLLM.
vLLM is great for batched offline generation but does not support mid-generation
KV cache manipulation.  Here we drive the generation loop ourselves, which gives
us full control:

  • Access to logits at every step (for metric computation).
  • KV cache is preserved at checkpoint positions, allowing O(interval)
    recomputation on backtrack rather than O(full prefix length).

KV Cache Checkpointing
-----------------------
At every `checkpoint_interval` generated tokens the engine saves a copy of
`(past_key_values, logits)`.  On backtrack to position bt:

  1. Restore the nearest checkpoint at position c ≤ bt.
  2. Replay tokens[c : bt] through the model (at most `checkpoint_interval`
     forward passes).
  3. Continue generation from bt.

If c == bt (the backtrack target falls exactly on a checkpoint) no forward
passes are needed at all — just copy the saved tensors.

Memory: checkpoints are offloaded to CPU RAM by default (`offload_to_cpu=True`).
A PCIe restore for a 600 MB checkpoint takes ~40 ms on typical Metacentrum nodes
— negligible compared to the recompute savings.  Set offload_to_cpu=False if you
have spare VRAM and want zero transfer latency.

Disable checkpointing entirely with checkpoint_interval=0 (falls back to the
original full-prefix recomputation).

Temperature after backtrack
----------------------------
`temperature_boost` is added for `boost_tokens` steps after each backtrack,
making the new continuation less likely to reproduce the discarded tokens.

Usage
-----
    from src.model.qwen3_helper import Qwen3Helper
    from src.interventions.metrics import get_metric
    from src.interventions.strategies import get_strategy
    from src.interventions.engine import InterventionEngine

    helper   = Qwen3Helper(local_path="/path/to/model")
    metric   = get_metric("sc")
    strategy = get_strategy("drop_detect", drop_threshold=0.5)
    engine   = InterventionEngine(
        helper, metric, strategy,
        max_backtracks=5,
        checkpoint_interval=128,   # save every 128 tokens
        offload_to_cpu=True,       # keep VRAM free
    )
    result = engine.generate(prompt_text, answer_type="integer")
    print(result.extracted_answer, result.n_backtracks)
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import List, Optional

from .result import GenerationResult, BacktrackEvent, InterventionEvent
from .metrics import MetricComputer
from .strategies import InterventionStrategy
from .checkpointer import KVCheckpointer


class InterventionEngine:
    """
    Autoregressive generation with pluggable backtracking and KV checkpointing.

    Args:
        helper:               Qwen3Helper (or compatible HF model wrapper).
        metric:               MetricComputer (SC or LP).
        strategy:             InterventionStrategy (decides when/where to backtrack).
        max_tokens:           Hard cap on generated tokens (including re-generated).
        max_backtracks:       Maximum number of backtracks per problem.
        temperature:          Base sampling temperature.
        top_p:                Nucleus sampling parameter.
        temperature_boost:    Extra temperature added for `boost_tokens` steps after
                              each backtrack, to diversify the new continuation.
        boost_tokens:         Number of tokens to apply the temperature boost for.
        checkpoint_interval:  Save a KV cache checkpoint every N generated tokens.
                              On backtrack, restores the nearest checkpoint and
                              replays only the short gap (≤ interval tokens).
                              0 = disable checkpointing (full prefix recompute).
        max_checkpoints:      Keep at most this many checkpoints; 0 = unlimited.
                              Oldest checkpoint is evicted when the limit is hit.
        offload_to_cpu:       Move checkpoint tensors to CPU RAM to save GPU VRAM.
                              True = recommended for long sequences.
        use_chat_template:    If True (default), wrap the prompt in the model's
                              chat template before tokenising.  Required for
                              Qwen3 thinking models to emit <think>...</think>.
        verbose:              If True, stream tokens to stdout as they are
                              generated.  Reasoning tokens (pre-</think>) are
                              printed in grey; answer tokens in normal colour.
                              Backtrack and early-stop events are highlighted.
                              Zero memory overhead — one token decoded at a time.
        seed:                 Optional random seed for reproducibility.
    """

    def __init__(
        self,
        helper,
        metric: MetricComputer,
        strategy: InterventionStrategy,
        max_tokens: int = 8192,
        max_backtracks: int = 5,
        temperature: float = 0.6,
        top_p: float = 0.95,
        temperature_boost: float = 0.2,
        boost_tokens: int = 15,
        checkpoint_interval: int = 128,
        max_checkpoints: int = 0,
        offload_to_cpu: bool = True,
        use_chat_template: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        self.helper = helper
        self.metric = metric
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.max_backtracks = max_backtracks
        self.temperature = temperature
        self.top_p = top_p
        self.temperature_boost = temperature_boost
        self.boost_tokens = boost_tokens
        self.use_chat_template = use_chat_template
        self.verbose = verbose

        # KV checkpointer (disabled when interval == 0)
        self._checkpointer: Optional[KVCheckpointer] = (
            KVCheckpointer(
                interval=checkpoint_interval,
                max_checkpoints=max_checkpoints,
                offload_to_cpu=offload_to_cpu,
            )
            if checkpoint_interval > 0
            else None
        )

        if seed is not None:
            torch.manual_seed(seed)

        # Cache EOS token id for termination
        self._eos_id: int = helper.tokenizer.eos_token_id

    # ── Verbose streaming helpers ─────────────────────────────────────────────

    # ANSI colour codes (fall back gracefully on non-ANSI terminals)
    _GREY   = "\033[90m"   # dim grey  — reasoning tokens
    _YELLOW = "\033[33m"   # yellow    — backtrack / event markers
    _GREEN  = "\033[32m"   # green     — answer tokens (post-</think>)
    _RESET  = "\033[0m"

    def _stream_token(self, token_id: int, past_think_close: bool) -> bool:
        """
        Decode and print one token.  Returns True if </think> was just emitted
        (so the caller can flip the reasoning→answer colour for subsequent tokens).
        """
        text = self.helper.tokenizer.decode([token_id], skip_special_tokens=False)
        just_closed = "</think>" in text

        if just_closed:
            # Print the </think> marker in yellow so it stands out
            pre, _, post = text.partition("</think>")
            print(f"{self._GREY}{pre}{self._RESET}"
                  f"{self._YELLOW}</think>{self._RESET}"
                  f"{self._GREEN}{post}{self._RESET}",
                  end="", flush=True)
        elif past_think_close:
            print(f"{self._GREEN}{text}{self._RESET}", end="", flush=True)
        else:
            print(f"{self._GREY}{text}{self._RESET}", end="", flush=True)

        return just_closed

    def _stream_event(self, msg: str) -> None:
        """Print a highlighted event line (backtrack, early-stop, etc.)."""
        print(f"\n{self._YELLOW}{msg}{self._RESET}\n", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        answer_type: str = "integer",
    ) -> GenerationResult:
        """
        Generate a response to `prompt` with backtracking and KV checkpointing.

        Loop invariant (top of each iteration):
            out.logits[:, -1, :]  — predictions for the token at position `step`
            out.past_key_values   — KV cache covering prompt + step tokens

        Checkpoint saves happen BEFORE sampling the token at `step`, so the
        saved logits are exactly the ones used to sample from that position.
        On restore we use those saved logits directly — zero recomputation.
        """
        # ── Tokenise prompt ──────────────────────────────────────────────────
        if self.use_chat_template:
            prompt_ids = self.helper.tokenize_chat(prompt)
        else:
            prompt_ids = self.helper.tokenize(prompt)

        # ── Warm up KV cache on prompt ───────────────────────────────────────
        out = self.helper.model(prompt_ids, use_cache=True)

        # ── Generation state ─────────────────────────────────────────────────
        metric_history: List[float] = []
        token_history:  List[int]   = []
        backtrack_events: List[BacktrackEvent] = []
        interventions:    List[InterventionEvent] = []
        n_backtracks: int = 0
        total_generated: int = 0
        finish_reason: str = "max_tokens"
        tokens_since_backtrack: int = self.boost_tokens  # No boost initially

        # Mutable sampling params — strategies can change these on the fly
        current_temp:  float = self.temperature
        current_top_p: float = self.top_p

        self.metric.reset()   # no-op for stateless metrics; clears running stats for z_score_sc
        self.strategy.reset()
        if self._checkpointer:
            self._checkpointer.reset()

        # Verbose streaming state
        past_think_close = False   # flips True once </think> is seen
        if self.verbose:
            print(f"{self._YELLOW}{'─'*60}{self._RESET}")
            print(f"{self._GREY}[reasoning]{self._RESET} "
                  f"{self._GREEN}[answer]{self._RESET}  "
                  f"{self._YELLOW}[events]{self._RESET}\n")

        # ── Main generation loop ─────────────────────────────────────────────
        step = 0
        while step < self.max_tokens:
            logits = out.logits[:, -1, :].squeeze(0)  # [V] float

            # ── Maybe save checkpoint (BEFORE sampling this token) ────────────
            if self._checkpointer and self._checkpointer.should_save(step):
                self._checkpointer.save(step, out.past_key_values, logits)

            # ── Sample next token (with optional post-backtrack temp boost) ──
            effective_temp = current_temp + (
                self.temperature_boost
                if tokens_since_backtrack < self.boost_tokens
                else 0.0
            )
            next_token_id  = self._sample(logits, effective_temp, current_top_p)

            # ── Compute metric ────────────────────────────────────────────────
            metric_val = self.metric.compute(logits, next_token_id)

            # ── Record ────────────────────────────────────────────────────────
            metric_history.append(metric_val)
            token_history.append(next_token_id)
            total_generated += 1
            tokens_since_backtrack += 1

            # ── Stream token to stdout (verbose mode) ─────────────────────────
            if self.verbose:
                just_closed = self._stream_token(next_token_id, past_think_close)
                if just_closed:
                    past_think_close = True

            # ── Ask strategy ──────────────────────────────────────────────────
            decision = self.strategy.on_token(
                position=step,
                metric=metric_val,
                metric_history=metric_history,
                token_ids=token_history,
                n_backtracks_so_far=n_backtracks,
            )

            # ── Strategy-guided checkpoint hint ───────────────────────────────
            if (
                self._checkpointer
                and decision.save_checkpoint
                and step not in self._checkpointer._checkpoints
            ):
                self._checkpointer.save(step, out.past_key_values, logits)

            # ── Action 1: Early stop ──────────────────────────────────────────
            if decision.stop_generation:
                interventions.append(InterventionEvent(
                    type="stop", position=step, detail=None,
                    reason=decision.reason,
                ))
                finish_reason = "early_stop"
                if self.verbose:
                    self._stream_event(
                        f"⏹  EARLY STOP at pos {step}  |  {decision.reason}"
                    )
                break

            # ── Action 2: Backtrack ───────────────────────────────────────────
            if decision.should_backtrack and n_backtracks < self.max_backtracks:
                bt = max(0, min(decision.backtrack_to, step))
                if bt >= step:
                    bt = max(0, step - 1)

                backtrack_events.append(BacktrackEvent(
                    trigger_position=step,
                    backtrack_to=bt,
                    trigger_metric=metric_val,
                    restore_metric=metric_history[bt] if bt > 0 else 0.0,
                    reason=decision.reason,
                    n_backtrack=n_backtracks + 1,
                ))
                n_backtracks += 1
                self.strategy.on_backtrack(bt, n_backtracks)

                if self.verbose:
                    self._stream_event(
                        f"↩  BACKTRACK #{n_backtracks}  pos {step} → {bt}"
                        f"  |  metric {metric_val:.4f}"
                        f"  |  {decision.reason}"
                    )
                    # Reset the think-close flag if backtracking before </think>
                    if bt < len(token_history) and past_think_close:
                        full_so_far = self.helper.tokenizer.decode(
                            token_history[:bt], skip_special_tokens=False
                        )
                        past_think_close = "</think>" in full_so_far

                metric_history = metric_history[:bt]
                token_history  = token_history[:bt]

                out = self._restore_to(prompt_ids, token_history, bt)

                if self._checkpointer:
                    self._checkpointer.prune_after(bt)

                step = bt
                tokens_since_backtrack = 0
                continue

            # ── EOS / max_tokens check (before injection / advance) ───────────
            if next_token_id == self._eos_id:
                finish_reason = "eos"
                break

            if step + 1 >= self.max_tokens:
                break

            # ── Action 3: Sampling-param steering ─────────────────────────────
            if decision.set_temperature is not None and decision.set_temperature != current_temp:
                interventions.append(InterventionEvent(
                    type="set_temperature", position=step,
                    detail=(current_temp, decision.set_temperature),
                    reason=decision.reason,
                ))
                current_temp = float(decision.set_temperature)
            if decision.set_top_p is not None and decision.set_top_p != current_top_p:
                interventions.append(InterventionEvent(
                    type="set_top_p", position=step,
                    detail=(current_top_p, decision.set_top_p),
                    reason=decision.reason,
                ))
                current_top_p = float(decision.set_top_p)

            # ── Advance KV cache by the just-sampled token ────────────────────
            out = self.helper.model(
                torch.tensor([[next_token_id]], device=self.helper.device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            step += 1

            # ── Action 4: Steering injection ──────────────────────────────────
            inject_ids = self._resolve_inject(decision)
            if inject_ids:
                step, out, past_think_close = self._inject_tokens(
                    inject_ids, out, step, metric_history, token_history,
                    past_think_close,
                )
                interventions.append(InterventionEvent(
                    type="inject", position=step,
                    detail=(list(inject_ids), decision.inject_text),
                    reason=decision.reason,
                ))
                total_generated += len(inject_ids)
                if step >= self.max_tokens:
                    break

        if n_backtracks >= self.max_backtracks and finish_reason == "max_tokens":
            finish_reason = "max_backtracks"

        # ── Decode and extract answer ─────────────────────────────────────────
        generated_text    = self.helper.tokenizer.decode(token_history, skip_special_tokens=False)
        extracted_answer  = self._extract(generated_text, answer_type)

        if self.verbose:
            print(f"\n{self._YELLOW}{'─'*60}{self._RESET}")
            print(f"  tokens={len(token_history)}  backtracks={n_backtracks}"
                  f"  finish={finish_reason}  answer={extracted_answer!r}")

        return GenerationResult(
            prompt=prompt,
            metric_name=self.metric.name,
            strategy_name=self.strategy.name,
            generated_text=generated_text,
            all_tokens=token_history,
            metric_values=metric_history,
            backtrack_events=backtrack_events,
            interventions=interventions,
            n_backtracks=n_backtracks,
            total_tokens_generated=total_generated,
            extracted_answer=extracted_answer,
            finish_reason=finish_reason,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _restore_to(
        self,
        prompt_ids: torch.Tensor,
        token_history: List[int],  # already truncated to [:bt]
        bt: int,
    ):
        """
        Rebuild model state at position `bt` as cheaply as possible.

        With checkpointing
        ------------------
        1. Find the nearest saved checkpoint at position c ≤ bt.
        2a. If c == bt (exact hit): restore tensors directly → zero forward passes.
             Return a SimpleNamespace with the saved logits and past_key_values,
             matching the shape expected by the main loop.
        2b. If c < bt (partial hit): restore past_kv from checkpoint, run
             forward pass only on token_history[c:bt] (bt-c tokens ≤ interval).
        2c. If no checkpoint (gap too large): full prefix recompute (fallback).

        Without checkpointing
        ---------------------
        Full forward pass on prompt + token_history.  O(bt) attention cost.
        """
        if self._checkpointer:
            ckpt = self._checkpointer.get_best_before(bt)
        else:
            ckpt = None

        if ckpt is not None:
            past_kv, saved_logits = self._checkpointer.restore(ckpt, self.helper.device)
            c = ckpt.position

            if c == bt:
                # ── Exact hit: zero forward passes ──────────────────────────
                # Wrap saved state in a namespace that the loop can use as `out`.
                # out.logits shape [B, T, V] — loop reads [:, -1, :].squeeze(0)
                return SimpleNamespace(
                    logits=saved_logits.unsqueeze(0).unsqueeze(0),  # [1, 1, V]
                    past_key_values=past_kv,
                )

            # ── Partial hit: replay gap tokens ───────────────────────────────
            # token_history[c:bt] are the tokens between checkpoint and target.
            gap_tokens = token_history[c:bt]
            if gap_tokens:
                gap_ids = torch.tensor(
                    [gap_tokens], dtype=torch.long, device=self.helper.device
                )
                return self.helper.model(
                    gap_ids, past_key_values=past_kv, use_cache=True
                )
            else:
                # c == bt already handled above; this branch is unreachable,
                # but if we somehow land here, just use the checkpoint state.
                return SimpleNamespace(
                    logits=saved_logits.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_kv,
                )

        # ── Fallback: full prefix recompute (no checkpoint available) ────────
        restore_ids = self._build_ids(prompt_ids, token_history)
        return self.helper.model(restore_ids, use_cache=True)

    def _resolve_inject(self, decision) -> List[int]:
        """
        Convert a strategy's inject_token_ids / inject_text into a list of ids.
        Returns empty list if the decision didn't request injection.
        """
        if decision.inject_token_ids:
            return list(decision.inject_token_ids)
        if decision.inject_text:
            return self.helper.tokenizer.encode(
                decision.inject_text, add_special_tokens=False
            )
        return []

    def _inject_tokens(
        self,
        inject_ids: List[int],
        out,
        step: int,
        metric_history: List[float],
        token_history: List[int],
        past_think_close: bool = False,
    ):
        """
        Force-feed a sequence of tokens through the model for steering.

        For each injected token id `t`:
          1. Use the current logits (out.logits[:, -1, :]) to compute the
             metric for token `t` AS IF the model had generated it (so the
             trajectory plot still shows confidence at injected positions).
          2. Append `t` to token_history.
          3. Advance the KV cache by feeding `t` to the model.

        Returns (new_step, new_out, past_think_close) — the loop's invariant
        holds again on return: out.logits[:, -1, :] predicts the token AFTER
        the last injected one.  past_think_close is updated if </think> was
        among the injected tokens.
        """
        for tid in inject_ids:
            logits_t = out.logits[:, -1, :].squeeze(0)
            metric_history.append(self.metric.compute(logits_t, tid))
            token_history.append(int(tid))

            if self.verbose:
                just_closed = self._stream_token(tid, past_think_close)
                if just_closed:
                    past_think_close = True

            out = self.helper.model(
                torch.tensor([[int(tid)]], device=self.helper.device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            step += 1
            if step >= self.max_tokens:
                break
        return step, out, past_think_close

    def _build_ids(
        self, prompt_ids: torch.Tensor, token_history: List[int]
    ) -> torch.Tensor:
        """Concatenate prompt_ids with token_history into a [1, L] tensor."""
        if not token_history:
            return prompt_ids
        gen_ids = torch.tensor(
            [token_history], dtype=torch.long, device=self.helper.device
        )
        return torch.cat([prompt_ids, gen_ids], dim=1)

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float
    ) -> int:
        """
        Sample one token with temperature scaling and nucleus (top-p) filtering.

        Args:
            logits:       [vocab_size] raw logits (float32).
            temperature:  Scaling temperature (> 0).
            top_p:        Nucleus probability mass.

        Returns:
            Sampled token id (int).
        """
        if temperature <= 0:
            return int(logits.argmax().item())

        logits = logits.float() / temperature

        # Top-p nucleus filtering
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)

        # Remove tokens that push cumulative probability beyond top_p
        # (keep the first token that pushes it over, then remove the rest)
        remove_mask = (cumsum - sorted_probs) > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices[sampled_idx].item())

    @staticmethod
    def _extract(text: str, answer_type: str) -> Optional[str]:
        """Extract answer from generated text using the shared voting logic."""
        from src.evaluation.voting import extract_answer
        try:
            return extract_answer(text, answer_type)
        except Exception:
            return None
