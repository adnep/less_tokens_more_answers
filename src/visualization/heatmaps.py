"""
JSD heatmap visualization: layers x tokens.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_jsd_heatmap(
    jsd_matrix: np.ndarray,
    token_labels: Optional[List[str]] = None,
    title: str = "JSD(final layer || layer l)",
    settling_depths: Optional[np.ndarray] = None,
    deep_threshold: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 6),
):
    """
    Plot a heatmap of JSD values across tokens and layers.

    Args:
        jsd_matrix: [T, L] array of JSD values
        token_labels: optional list of token strings for x-axis
        title: plot title
        settling_depths: optional [T] array to overlay settling line
        deep_threshold: layer index threshold for deep-thinking regime
        save_path: if provided, save figure to this path
        figsize: figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    T, L = jsd_matrix.shape
    im = ax.imshow(
        jsd_matrix.T, aspect="auto", cmap="viridis", origin="lower",
        vmin=0, vmax=min(1.0, jsd_matrix.max()),
    )
    plt.colorbar(im, ax=ax, label="JSD")

    if settling_depths is not None:
        ax.plot(range(T), settling_depths, color="red", linewidth=1.5,
                label="Settling depth", alpha=0.8)

    if deep_threshold is not None:
        ax.axhline(deep_threshold, color="white", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Deep threshold (layer {deep_threshold})")

    if token_labels and T <= 80:
        ax.set_xticks(range(T))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
    else:
        ax.set_xlabel("Token position")

    ax.set_ylabel("Layer")
    ax.set_title(title)
    if settling_depths is not None or deep_threshold is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
    return fig
