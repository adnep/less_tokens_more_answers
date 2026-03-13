"""
Logit lens: project intermediate layer hidden states to vocabulary space.

Optimized approach: batch all layers through norm+lm_head in one go
for short sequences (prefix DTR with ~50 tokens), or process layer-by-layer
for longer sequences to save memory.

Important: the hidden_states tuple from Qwen3Helper.get_layer_hidden_states()
has entries [0..N-2] as pre-norm (need norm+lm_head) and entry [N-1] as
post-norm (need lm_head only). Both paths handle this correctly.
"""

import torch
from typing import Tuple

from src.dtr.jsd import jsd_matrix


def compute_jsd_per_layer(
    hidden_states: Tuple[torch.Tensor, ...],
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    token_positions: slice = slice(None),
    batch_layers: bool = True,
) -> torch.Tensor:
    """
    Compute JSD(final_layer || layer_l) for every layer and token position.

    Args:
        hidden_states: tuple of num_layers tensors from Qwen3Helper.get_layer_hidden_states().
            Entries [0..N-2] are pre-norm, entry [N-1] is post-norm.
        norm: the model's final RMSNorm
        lm_head: the model's output projection (unembedding)
        token_positions: slice to select which token positions to analyze
        batch_layers: if True, stack all layers and process in one matmul.
            Faster for short sequences (<100 tokens). Set False for long sequences.

    Returns:
        jsd_tensor: [T_selected, num_layers] JSD values
    """
    num_layers = len(hidden_states)

    if batch_layers:
        return _compute_batched(hidden_states, norm, lm_head,
                                token_positions, num_layers)
    else:
        return _compute_sequential(hidden_states, norm, lm_head,
                                   token_positions, num_layers)


def _get_final_probs(hidden_states, lm_head, token_positions):
    """Get reference probs from final layer (already post-norm, just lm_head)."""
    final_hidden = hidden_states[-1][0, token_positions, :]  # [T, D]
    final_logits = lm_head(final_hidden).float()
    final_probs = torch.softmax(final_logits, dim=-1)  # [T, V]
    del final_logits
    return final_probs


def _compute_batched(hidden_states, norm, lm_head,
                     token_positions, num_layers):
    """
    Stack all layer hidden states and do one big norm+lm_head+softmax.
    Fast for short sequences (T <= ~100 tokens).
    """
    # Final layer probs (post-norm, just lm_head)
    final_probs = _get_final_probs(hidden_states, lm_head, token_positions)
    T_sel = final_probs.shape[0]

    # Stack the pre-norm layers (0..N-2) for batched norm+lm_head
    pre_norm_layers = num_layers - 1
    stacked = torch.stack(
        [hidden_states[i][0, token_positions, :] for i in range(pre_norm_layers)],
        dim=0,
    )  # [L-1, T, D]

    L_pre, _, D = stacked.shape
    flat = stacked.reshape(L_pre * T_sel, D)
    flat_normed = norm(flat)
    flat_logits = lm_head(flat_normed).float()
    flat_probs = torch.softmax(flat_logits, dim=-1)
    pre_norm_probs = flat_probs.reshape(L_pre, T_sel, -1)  # [L-1, T, V]

    del flat, flat_normed, flat_logits, flat_probs, stacked

    # Compute JSD for each layer
    jsd_tensor = torch.zeros(T_sel, num_layers, device=final_probs.device)

    # Layers 0..N-2 (pre-norm, went through norm+lm_head)
    for layer_idx in range(pre_norm_layers):
        jsd_tensor[:, layer_idx] = jsd_matrix(final_probs, pre_norm_probs[layer_idx])

    # Layer N-1 (final layer vs itself) — JSD should be 0
    jsd_tensor[:, num_layers - 1] = 0.0

    del pre_norm_probs, final_probs
    return jsd_tensor


def _compute_sequential(hidden_states, norm, lm_head,
                         token_positions, num_layers):
    """
    Process one layer at a time. Slower but uses less peak memory.
    Use for long sequences.
    """
    # Final layer probs (post-norm, just lm_head)
    final_probs = _get_final_probs(hidden_states, lm_head, token_positions)
    T = final_probs.shape[0]

    jsd_tensor = torch.zeros(T, num_layers, device=final_probs.device)

    # Layers 0..N-2 (pre-norm)
    for layer_idx in range(num_layers - 1):
        hidden = hidden_states[layer_idx][0, token_positions, :]
        normed = norm(hidden)
        logits = lm_head(normed).float()
        probs = torch.softmax(logits, dim=-1)
        jsd_tensor[:, layer_idx] = jsd_matrix(final_probs, probs)
        del normed, logits, probs

    # Layer N-1 vs itself = 0
    jsd_tensor[:, num_layers - 1] = 0.0

    del final_probs
    return jsd_tensor
