"""
Jensen-Shannon Divergence computation.
JSD(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5*(p + q).
Uses log base 2 so JSD is bounded in [0, 1].
"""

import torch

EPS = 1e-10


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute JSD between two probability distributions.

    Args:
        p: [..., V] probability distribution (e.g., final layer)
        q: [..., V] probability distribution (e.g., intermediate layer)

    Returns:
        jsd_values: [...] scalar JSD per position, in [0, 1]
    """
    m = 0.5 * (p + q)
    # KL(p || m) = sum(p * log2(p / m))
    kl_p_m = (p * (torch.log2(p + EPS) - torch.log2(m + EPS))).sum(dim=-1)
    kl_q_m = (q * (torch.log2(q + EPS) - torch.log2(m + EPS))).sum(dim=-1)
    return 0.5 * kl_p_m + 0.5 * kl_q_m


def jsd_matrix(
    final_layer_probs: torch.Tensor, layer_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute JSD between final layer and each intermediate layer for all token positions.

    Args:
        final_layer_probs: [T, V] probability distribution from the final layer
        layer_probs: [T, V] probability distribution from one intermediate layer

    Returns:
        jsd_values: [T] JSD per token position
    """
    return jsd(final_layer_probs, layer_probs)
