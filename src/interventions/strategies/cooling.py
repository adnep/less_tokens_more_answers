"""
Temperature-cooling strategies.

Two flavours:

LinearCoolingStrategy
    Deterministic linear schedule: temperature goes from `start_temp` down to
    `end_temp` linearly between positions `start_token` and `end_token`.
    Useful as a baseline and easy to reason about.

ConfidenceCoolingStrategy
    Adaptive: cool only when the smoothed metric stays in a "confident region"
    (above `confidence_threshold` for `sustained_window` consecutive tokens).
    Re-warm (raise temperature back up) when the metric drops below the
    threshold for `dropout_window` consecutive tokens.

    The intuition: if the model has been confident for a while, sampling
    randomness is mostly contributing noise — make it more deterministic.
    When confidence drops, the model needs more exploration again.

Both strategies emit decision.set_temperature on every step (or only on
change, via internal cache).  The engine applies the new temperature from
the next sampling step on.
"""

from typing import List
import numpy as np

from . import InterventionStrategy, InterventionDecision


# ──────────────────────────────────────────────────────────────────────────────
# LinearCoolingStrategy
# ──────────────────────────────────────────────────────────────────────────────

class LinearCoolingStrategy(InterventionStrategy):
    """
    Cool temperature linearly from `start_temp` to `end_temp`.

    Args:
        start_temp:  Initial temperature (used before `start_token`).
        end_temp:    Final temperature (used after `end_token`).
        start_token: Position at which cooling begins (default 0 = immediate).
        end_token:   Position at which cooling ends and `end_temp` is held.
    """

    def __init__(
        self,
        start_temp: float = 0.6,
        end_temp:   float = 0.2,
        start_token: int = 0,
        end_token:   int = 4000,
    ):
        if end_token <= start_token:
            raise ValueError("end_token must be > start_token")
        self.start_temp  = start_temp
        self.end_temp    = end_temp
        self.start_token = start_token
        self.end_token   = end_token
        self._last_emitted: float = -1.0  # so the first call always emits

    def reset(self) -> None:
        self._last_emitted = -1.0

    @property
    def name(self) -> str:
        return "linear_cooling"

    def describe(self) -> str:
        return (
            f"linear_cooling({self.start_temp:.2f} → {self.end_temp:.2f} "
            f"over [{self.start_token}, {self.end_token}])"
        )

    def _temp_at(self, position: int) -> float:
        if position <= self.start_token:
            return self.start_temp
        if position >= self.end_token:
            return self.end_temp
        progress = (position - self.start_token) / (self.end_token - self.start_token)
        return self.start_temp + (self.end_temp - self.start_temp) * progress

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        new_temp = self._temp_at(position)
        # Emit only on change to keep telemetry clean
        if abs(new_temp - self._last_emitted) > 1e-4:
            self._last_emitted = new_temp
            return InterventionDecision(
                set_temperature=new_temp,
                reason=f"linear cooling at pos {position}: T={new_temp:.3f}",
            )
        return InterventionDecision()


# ──────────────────────────────────────────────────────────────────────────────
# ConfidenceCoolingStrategy
# ──────────────────────────────────────────────────────────────────────────────

class ConfidenceCoolingStrategy(InterventionStrategy):
    """
    Adaptive temperature based on confidence regions.

    State machine:
        EXPLORING  — temperature = warm_temp
                     transition to CONFIDENT after `sustained_window` tokens
                     with smoothed metric ≥ confidence_threshold.
        CONFIDENT — temperature = cool_temp
                     transition back to EXPLORING after `dropout_window`
                     tokens with smoothed metric < confidence_threshold.

    Args:
        confidence_threshold:  Smoothed-metric value above which we're "confident".
                               For SC: -3.0 to -1.0 typical.  For LP: -1.0 to -0.3.
        sustained_window:      Required consecutive confident tokens to cool.
        dropout_window:        Required consecutive non-confident tokens to re-warm.
        smoothing_window:      Rolling-mean window for the smoothed metric.
        warm_temp:             Temperature when EXPLORING (default = engine's base temp).
        cool_temp:             Temperature when CONFIDENT.
        min_position:          Don't change temperature in the first N tokens.
    """

    def __init__(
        self,
        confidence_threshold: float = -2.0,
        sustained_window:     int = 30,
        dropout_window:       int = 15,
        smoothing_window:     int = 30,
        warm_temp:            float = 0.6,
        cool_temp:            float = 0.3,
        min_position:         int = 50,
    ):
        self.confidence_threshold = confidence_threshold
        self.sustained_window  = sustained_window
        self.dropout_window    = dropout_window
        self.smoothing_window  = smoothing_window
        self.warm_temp = warm_temp
        self.cool_temp = cool_temp
        self.min_position = min_position
        self._state: str = "EXPLORING"
        self._consecutive: int = 0
        self._last_emitted_temp: float = -1.0

    def reset(self) -> None:
        self._state = "EXPLORING"
        self._consecutive = 0
        self._last_emitted_temp = -1.0

    @property
    def name(self) -> str:
        return "confidence_cooling"

    def describe(self) -> str:
        return (
            f"confidence_cooling(thr={self.confidence_threshold:.2f}, "
            f"sustained={self.sustained_window}, dropout={self.dropout_window}, "
            f"warm={self.warm_temp}, cool={self.cool_temp})"
        )

    def _smoothed(self, metric_history: List[float]) -> float:
        n = min(self.smoothing_window, len(metric_history))
        return float(np.mean(metric_history[-n:])) if n else 0.0

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        if position < self.min_position:
            return InterventionDecision()

        smoothed = self._smoothed(metric_history)

        if self._state == "EXPLORING":
            if smoothed >= self.confidence_threshold:
                self._consecutive += 1
                if self._consecutive >= self.sustained_window:
                    self._state = "CONFIDENT"
                    self._consecutive = 0
                    return self._maybe_emit(self.cool_temp,
                        f"entered confident region at pos {position} "
                        f"(smoothed={smoothed:.3f} ≥ {self.confidence_threshold})"
                    )
            else:
                self._consecutive = 0
        else:  # CONFIDENT
            if smoothed < self.confidence_threshold:
                self._consecutive += 1
                if self._consecutive >= self.dropout_window:
                    self._state = "EXPLORING"
                    self._consecutive = 0
                    return self._maybe_emit(self.warm_temp,
                        f"left confident region at pos {position} "
                        f"(smoothed={smoothed:.3f} < {self.confidence_threshold})"
                    )
            else:
                self._consecutive = 0

        return InterventionDecision()

    def _maybe_emit(self, new_temp: float, reason: str) -> InterventionDecision:
        """Emit a temperature change only if it differs from the last emitted."""
        if abs(new_temp - self._last_emitted_temp) < 1e-4:
            return InterventionDecision()
        self._last_emitted_temp = new_temp
        return InterventionDecision(set_temperature=new_temp, reason=reason)
