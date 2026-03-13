"""
Logit lens: project intermediate layer hidden states to vocabulary space.
Memory-efficient: computes JSD one layer at a time instead of materializing
the full [T, L, V] probability tensor.
"""

import torch
from typing import Dict

from src.dtr.jsd import jsd_matrix


def compute_jsd_per_layer(
    layer_hidden_states: Dict[int, torch.Tensor],
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    token_positions: slice = slice(None),
) -> torch.Tensor:
    """
    Compute JSD(final_layer || layer_l) for every layer and token position.

    Processes one layer at a time to avoid materializing [T, L, V] probs.

    Args:
        layer_hidden_states: dict[layer_idx -> Tensor[1, T, hidden_dim]]
        norm: the model's final RMSNorm
        lm_head: the model's output projection (unembedding)
        token_positions: slice to select which token positions to analyze

    Returns:
        jsd_tensor: [T_selected, L] JSD values, where L = num layers
    """
    num_layers = len(layer_hidden_states)

    # Get final layer probs (reference distribution)
    final_hidden = layer_hidden_states[num_layers - 1][:, token_positions, :]
    final_logits = lm_head(norm(final_hidden)).float()
    final_probs = torch.softmax(final_logits, dim=-1).squeeze(0)  # [T, V]

    T = final_probs.shape[0]
    jsd_tensor = torch.zeros(T, num_layers, device=final_probs.device)

    for layer_idx in range(num_layers):
        hidden = layer_hidden_states[layer_idx][:, token_positions, :]
        logits = lm_head(norm(hidden)).float()
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [T, V]

        jsd_tensor[:, layer_idx] = jsd_matrix(final_probs, probs)

        # Free memory
        del logits, probs

    del final_logits, final_probs
    return jsd_tensor
