"""
Pluggable metric computers for the backtracking engine.

A MetricComputer turns (logits, token_id) into a scalar signal.
Higher = more confident for all metrics.

Available metrics
-----------------
"sc"          SelfCertaintyMetric   KL(u ‖ p_t) ∈ [0, ∞)   Kang et al. (2025)
"neg_entropy" NegativeEntropyMetric -H(p_t) ∈ (-∞, 0]
"lp"          LPMetric              log p(y_t) ∈ (-∞, 0]

Adding a new metric
-------------------
1. Create src/interventions/metrics/my_metric.py with a class that inherits
   MetricComputer and implements .compute(), .name, .higher_is_better.
2. Import it below and add to _REGISTRY.
3. Done — the engine and CLI pick it up automatically.
"""

from abc import ABC, abstractmethod
import torch


class MetricComputer(ABC):
    """Abstract base class for per-token metrics used during generation."""

    @abstractmethod
    def compute(self, logits: torch.Tensor, token_id: int) -> float:
        """
        Compute the metric for one generated token.

        Args:
            logits:   1-D float tensor of shape [vocab_size] — raw (un-softmaxed)
                      logits for the current position.  Passed as float32.
            token_id: The integer id of the token that was actually sampled.

        Returns:
            Scalar float.  Higher = more confident for all metrics.
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each generate() call.
        Override in stateful metrics (e.g. ZScoreSCMetric) to clear per-generation
        running statistics.  The default implementation is a no-op.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string (e.g. 'sc', 'neg_entropy', 'lp')."""
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """True if a higher metric value signals more confidence / better state."""
        ...


# ── Registry ──────────────────────────────────────────────────────────────────

from .sc  import SelfCertaintyMetric, NegativeEntropyMetric, ZScoreSCMetric  # noqa: E402
from .lp  import LPMetric                                                     # noqa: E402

_REGISTRY: dict = {
    "sc":          SelfCertaintyMetric,
    "neg_entropy": NegativeEntropyMetric,
    "lp":          LPMetric,
    "z_score_sc":  ZScoreSCMetric,
}


def get_metric(name: str, **kwargs) -> MetricComputer:
    """
    Instantiate a metric by name.

    Args:
        name:   One of the keys in _REGISTRY ("sc", "neg_entropy", "lp").
        kwargs: Forwarded to the metric's __init__.

    Returns:
        A MetricComputer instance.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric: {name!r}.  Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "MetricComputer",
    "SelfCertaintyMetric",
    "NegativeEntropyMetric",
    "LPMetric",
    "ZScoreSCMetric",
    "get_metric",
]
