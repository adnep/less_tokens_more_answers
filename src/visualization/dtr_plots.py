"""
DTR distribution and settling depth plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def plot_settling_depth_histogram(
    settling_depths: np.ndarray,
    num_layers: int,
    deep_threshold: int,
    title: str = "Settling Depth Distribution",
    save_path: Optional[str] = None,
):
    """Plot histogram of settling depths with deep-thinking threshold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        settling_depths, bins=num_layers, range=(0, num_layers),
        edgecolor="black", alpha=0.7, color="steelblue",
    )
    ax.axvline(deep_threshold, color="red", linestyle="--", linewidth=2,
               label=f"Deep threshold (layer {deep_threshold})")

    deep_frac = (settling_depths >= deep_threshold).mean()
    ax.set_xlabel("Settling depth (layer)")
    ax.set_ylabel("Token count")
    ax.set_title(f"{title} | DTR = {deep_frac:.3f}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dtr_comparison(
    dtr_values: List[Tuple[str, float]],
    title: str = "DTR Across Samples",
    eta: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Bar chart of DTR values per sample, showing the selection threshold.

    Args:
        dtr_values: list of (label, dtr) tuples, sorted by DTR descending
        title: plot title
        eta: fraction of samples to keep
        save_path: if provided, save figure
    """
    labels, values = zip(*dtr_values) if dtr_values else ([], [])
    n = len(values)
    n_keep = max(1, int(n * eta))

    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), 5))
    colors = ["#2196F3" if i < n_keep else "#BDBDBD" for i in range(n)]
    ax.bar(range(n), values, color=colors, edgecolor="black", linewidth=0.5)

    if n > 0:
        ax.axvline(n_keep - 0.5, color="red", linestyle="--",
                   label=f"Selection cutoff (top {eta*100:.0f}%)")

    ax.set_xlabel("Sample (sorted by DTR)")
    ax.set_ylabel("DTR")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
