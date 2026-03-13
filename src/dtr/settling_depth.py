"""
Settling depth computation.
Finds the earliest layer at which the monotonically non-increasing
JSD envelope drops below the settling threshold gamma.
"""

import torch


def compute_settling_depth(
    jsd_tensor: torch.Tensor, gamma: float = 0.5
) -> torch.Tensor:
    """
    Compute settling depth c_t for each token position.

    c_t = min{l : D_bar_t_l <= gamma}
    where D_bar_t_l = min_{j<=l} D_t_j (cumulative minimum envelope)

    Args:
        jsd_tensor: [T, L] JSD values per token and layer
        gamma: settling threshold (default 0.5 per paper)

    Returns:
        settling_depths: [T] layer index where each token settles.
            If a token never settles (JSD never drops below gamma),
            settling depth is set to L (total number of layers).
    """
    T, L = jsd_tensor.shape

    # Cumulative minimum across layers: D_bar_t_l = min(D_t_0, ..., D_t_l)
    cummin_jsd, _ = torch.cummin(jsd_tensor, dim=1)

    # Find first layer where envelope <= gamma
    settled = cummin_jsd <= gamma  # [T, L] boolean

    # argmax on bool tensor gives first True index; if none True, gives 0
    # We need to distinguish "settled at layer 0" from "never settled"
    any_settled = settled.any(dim=1)  # [T]
    first_settled_layer = settled.float().argmax(dim=1)  # [T]

    # For tokens that never settle, set depth to L
    settling_depths = torch.where(any_settled, first_settled_layer, torch.tensor(L, device=jsd_tensor.device))

    return settling_depths.long()
