"""
Distributional confidence metrics computed from the model's output logits.

NegativeEntropyMetric  (name="neg_entropy")
    SC_old = -H(p_t) = ∑_v p_t(v) log p_t(v)
    Range: (-∞, 0].  Closer to 0 = more confident.
    Simple to compute; used as a baseline metric.

SelfCertaintyMetric  (name="sc")
    SC = KL(u ‖ p_t) = -log|V| - (1/|V|) ∑_v log p_t(v)
    Kang et al. (2025), Eq. 10.
    Range: [0, ∞).  0 = uniform distribution (maximum uncertainty).
    Higher = more confident (distribution farther from uniform).

ZScoreSCMetric  (name="z_score_sc")
    Online z-score of SC: z(t) = (SC(t) − μ_t) / σ_t
    where μ_t, σ_t are the running mean/std accumulated since the last reset().
    Range: approximately (−∞, +∞); typically (−2, +2) in practice.
    Positive = current token is more certain than the per-generation average.
    Requires reset() at the start of each generate() call; the engine does this.
    During warm-up (< warmup tokens) returns 0.0 to avoid instability.

    Note: after a backtrack the running stats include the discarded tokens.
    This is an accepted approximation — the direction of the signal is unaffected.
"""

import math
import torch
import torch.nn.functional as F
from . import MetricComputer


def _compute_sc(logits: torch.Tensor) -> float:
    """SC = KL(u ‖ p_t) = -log|V| - mean_v(log p_t(v))."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return (-math.log(log_probs.shape[0]) - log_probs.mean()).item()


class NegativeEntropyMetric(MetricComputer):
    """
    Negative entropy: -H(p_t) = ∑_v p_t(v) log p_t(v)

    Range: (-∞, 0].  Closer to 0 = more certain.
    token_id is ignored (whole distribution matters).
    """

    @property
    def name(self) -> str:
        return "neg_entropy"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, token_id: int) -> float:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs).sum()
        return -entropy.item()


class SelfCertaintyMetric(MetricComputer):
    """
    Self-Certainty as defined in Kang et al. (2025), Eq. 10:

        SC(t) = KL(u ‖ p_t) = -log|V| - (1/|V|) ∑_v log p_t(v)

    where u is the uniform distribution over the vocabulary V.

    Range: [0, ∞).  0 = uniform (maximum uncertainty).  Higher = more confident.
    token_id is ignored (whole distribution matters).
    """

    @property
    def name(self) -> str:
        return "sc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, token_id: int) -> float:
        return _compute_sc(logits)


class ZScoreSCMetric(MetricComputer):
    """
    Online z-scored Self-Certainty (name="z_score_sc").

    At each token t computes:
        z(t) = (SC(t) − μ_t) / σ_t

    where μ_t and σ_t are updated via Welford's online algorithm using every
    SC value seen since the last reset() call.

    Args:
        warmup: Number of tokens to accumulate before returning a meaningful
                z-score.  Returns 0.0 during warm-up to avoid instability
                from insufficient statistics.

    Threshold calibration guide (σ units, Qwen3-4B at temp 0.6):
        "Confident" token:   z ≈ +0.5 to +1.5
        "Average" token:     z ≈ 0
        "Uncertain" token:   z ≈ −0.5 to −1.5

    For ThresholdStrategy: trigger when z < −0.2 to −0.4.
    For DropDetectStrategy: drop_threshold = 0.3 to 0.8.
    For wait strategies: wait_threshold = 0.2 to 0.5 (exit_on="high").
    """

    def __init__(self, warmup: int = 50):
        self.warmup = warmup
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0  # sum of squared deviations (Welford)

    def reset(self) -> None:
        """Clear accumulated stats — called at the start of each generate()."""
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    @property
    def name(self) -> str:
        return "z_score_sc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, token_id: int) -> float:
        sc = _compute_sc(logits)

        # Welford's online mean/variance update
        self._count += 1
        delta = sc - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (sc - self._mean)

        if self._count < self.warmup:
            return 0.0

        variance = self._M2 / (self._count - 1) if self._count > 1 else 1.0
        std = max(variance ** 0.5, 1e-6)
        return (sc - self._mean) / std
