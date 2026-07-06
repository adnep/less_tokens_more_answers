"""
ThresholdStrategy: backtrack when metric drops below a fixed threshold.

Simplest possible trigger: if SC or LP falls below `threshold` for a
sustained window of tokens, rewind `backtrack_n_tokens` and try again.

Good for: quickly testing whether backtracking at all helps.
Weakness: the right threshold is metric/model/dataset specific.
"""

from typing import List
from . import InterventionStrategy, InterventionDecision, _no_intervention


class ThresholdStrategy(InterventionStrategy):
    """
    Trigger a backtrack when the SMOOTHED metric stays below `threshold`.

    Args:
        threshold:          Metric value below which we consider the model
                            "in trouble".  For SC: a value like -8.0 to -12.0
                            (very uncertain).  For LP: -5.0 to -10.0.
        confirm_window:     Number of consecutive below-threshold tokens required
                            before triggering.  1 = trigger immediately on first
                            below-threshold token.  Higher = more conservative.
        backtrack_n_tokens: How many generated tokens to discard.
                            The engine will restore to max(0, trigger_pos - n).
        min_position:       Don't trigger in the first N generated tokens.
                            Avoids spurious early triggers during the model's
                            warm-up into the <think> block.
        cooldown:           Minimum tokens between consecutive backtracks.
                            Prevents rapid repeated triggers on the same spot.
    """

    def __init__(
        self,
        threshold: float = -8.0,
        confirm_window: int = 1,
        backtrack_n_tokens: int = 30,
        min_position: int = 80,
        cooldown: int = 50,
    ):
        self.threshold = threshold
        self.confirm_window = confirm_window
        self.backtrack_n_tokens = backtrack_n_tokens
        self.min_position = min_position
        self.cooldown = cooldown
        self._last_backtrack_at: int = -cooldown
        self._below_count: int = 0  # consecutive below-threshold tokens

    def reset(self) -> None:
        self._last_backtrack_at = -self.cooldown
        self._below_count = 0

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        self._last_backtrack_at = backtrack_to
        self._below_count = 0

    @property
    def name(self) -> str:
        return "threshold"

    def describe(self) -> str:
        return (
            f"threshold(thr={self.threshold}, confirm={self.confirm_window}, "
            f"rewind={self.backtrack_n_tokens}, min_pos={self.min_position}, "
            f"cooldown={self.cooldown})"
        )

    def on_token(
        self,
        position: int,
        metric: float,
        metric_history: List[float],
        token_ids: List[int],
        n_backtracks_so_far: int,
    ) -> InterventionDecision:
        # Skip early tokens
        if position < self.min_position:
            self._below_count = 0
            return _no_intervention()

        # Skip during cooldown
        if position - self._last_backtrack_at < self.cooldown:
            self._below_count = 0
            return _no_intervention()

        # Track consecutive below-threshold tokens
        if metric < self.threshold:
            self._below_count += 1
        else:
            self._below_count = 0

        # Trigger if confirmed
        if self._below_count >= self.confirm_window:
            self._below_count = 0
            backtrack_to = max(0, position - self.backtrack_n_tokens)
            return InterventionDecision(
                should_backtrack=True,
                backtrack_to=backtrack_to,
                reason=(
                    f"metric {metric:.3f} < threshold {self.threshold} "
                    f"for {self.confirm_window} consecutive token(s)"
                ),
            )

        return _no_intervention()
