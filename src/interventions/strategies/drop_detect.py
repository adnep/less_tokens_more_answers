"""
DropDetectStrategy: backtrack on a sharp drop in the smoothed metric.

Instead of an absolute threshold (ThresholdStrategy), this detects a
*relative* change: the smoothed metric fell by more than `drop_threshold`
compared to the smoothed value `window` tokens ago.

This mirrors the event-detection logic in selfcert_trajectory.py and
trace_inspector.py, making it easy to verify with the offline plots.

Two backtrack-target modes
--------------------------
"local_max"
    Find the peak of metric_history in the last `search_window` tokens
    before the drop.  This is the "highest-confidence point we were at
    recently", which is a natural place to restart from.

"fixed_n"
    Always go back exactly `backtrack_n_tokens`.  Simpler but coarser.

Trigger detection
-----------------
We maintain a short running average of the metric (half-window on each
side of the current position).  A trigger fires when:

    smoothed_now - smoothed_prev_window < -drop_threshold

where smoothed_prev_window is the average `window` steps ago.
"""

from typing import List
import numpy as np
from . import InterventionStrategy, InterventionDecision, _no_intervention


class DropDetectStrategy(InterventionStrategy):
    """
    Backtrack when the smoothed metric drops sharply.

    Args:
        drop_threshold:     Minimum smoothed-metric decrease to trigger.
                            Positive number: trigger when smoothed drops by
                            more than this amount in one `window` span.
                            For SC: 0.3–1.0 works well.
                            For LP: 0.5–2.0 may be appropriate.
        window:             Smoothing window (rolling mean half-width).
                            Also the look-back span for drop computation.
        backtrack_mode:     "local_max" or "fixed_n".
        backtrack_n_tokens: Tokens to rewind for "fixed_n" mode.
        search_window:      How far back to search for local max (for
                            "local_max" mode).  Should be ≥ window.
        min_position:       Don't trigger in the first N generated tokens.
        cooldown:           Minimum tokens between consecutive backtracks.
    """

    def __init__(
        self,
        drop_threshold: float = 0.5,
        window: int = 30,
        backtrack_mode: str = "local_max",
        backtrack_n_tokens: int = 40,
        search_window: int = 60,
        min_position: int = 80,
        cooldown: int = 60,
    ):
        if backtrack_mode not in ("local_max", "fixed_n"):
            raise ValueError(
                f"backtrack_mode must be 'local_max' or 'fixed_n', "
                f"got {backtrack_mode!r}"
            )
        self.drop_threshold = drop_threshold
        self.window = window
        self.backtrack_mode = backtrack_mode
        self.backtrack_n_tokens = backtrack_n_tokens
        self.search_window = search_window
        self.min_position = min_position
        self.cooldown = cooldown
        self._last_backtrack_at: int = -cooldown

    def reset(self) -> None:
        self._last_backtrack_at = -self.cooldown

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        self._last_backtrack_at = backtrack_to

    @property
    def name(self) -> str:
        return "drop_detect"

    def describe(self) -> str:
        return (
            f"drop_detect(drop_thr={self.drop_threshold}, window={self.window}, "
            f"mode={self.backtrack_mode}, search_win={self.search_window}, "
            f"min_pos={self.min_position}, cooldown={self.cooldown})"
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _smoothed_at(self, metric_history: List[float], idx: int) -> float:
        """Centred rolling mean around idx with radius window//2."""
        half = self.window // 2
        lo = max(0, idx - half)
        hi = min(len(metric_history), idx + half + 1)
        return float(np.mean(metric_history[lo:hi]))

    def _find_backtrack_to(
        self, metric_history: List[float], trigger_pos: int
    ) -> int:
        """Compute the backtrack target position."""
        if self.backtrack_mode == "fixed_n":
            return max(0, trigger_pos - self.backtrack_n_tokens)

        # "local_max": peak of metric_history in search_window before drop
        search_start = max(0, trigger_pos - self.search_window)
        search_slice = metric_history[search_start:trigger_pos]
        if not search_slice:
            return max(0, trigger_pos - self.backtrack_n_tokens)
        local_max_offset = int(np.argmax(search_slice))
        return search_start + local_max_offset

    # ── main hook ──────────────────────────────────────────────────────────────

    def on_token(
        self,
        position: int,
        metric: float,
        metric_history: List[float],
        token_ids: List[int],
        n_backtracks_so_far: int,
    ) -> InterventionDecision:
        # Need at least window + 1 points to compute a drop
        if position < self.min_position or len(metric_history) < self.window + 1:
            return _no_intervention()

        if position - self._last_backtrack_at < self.cooldown:
            return _no_intervention()

        # Compare smoothed metric now vs window steps ago
        smoothed_now  = self._smoothed_at(metric_history, position)
        smoothed_prev = self._smoothed_at(metric_history, position - self.window)
        delta = smoothed_now - smoothed_prev  # negative = drop

        if delta < -self.drop_threshold:
            backtrack_to = self._find_backtrack_to(metric_history, position)
            return InterventionDecision(
                should_backtrack=True,
                backtrack_to=backtrack_to,
                reason=(
                    f"smoothed drop Δ={delta:.3f} < -{self.drop_threshold} "
                    f"(now={smoothed_now:.3f}, prev={smoothed_prev:.3f})"
                ),
            )

        return _no_intervention()
