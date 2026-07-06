"""
Token-injection (steering) strategies.

When some condition fires, the strategy emits a chunk of text or token ids
that the engine forces through the model.  After injection the model
continues sampling normally, but its KV cache now contains the injected
tokens — they bias subsequent generation just as if the model had written
them itself.

Two flavours:

DropSteeringStrategy
    Inject `steering_text` when a sharp drop is detected in the smoothed
    metric (analogous to DropDetectStrategy but with steering instead of
    backtracking).  Useful for nudging the model out of a uncertain region
    by inserting a phrase like "Wait, let me reconsider this carefully."

ConfidentRegionEndStrategy
    Wait until a confident region has been established (sustained high
    metric), then on the FIRST drop after it, inject a steering phrase and
    optionally lower temperature too.  Mirrors the "drop after confidence"
    pattern the user described.
"""

from typing import List, Optional
import numpy as np

from . import InterventionStrategy, InterventionDecision


# ──────────────────────────────────────────────────────────────────────────────
# DropSteeringStrategy
# ──────────────────────────────────────────────────────────────────────────────

class DropSteeringStrategy(InterventionStrategy):
    """
    On a sharp drop in the smoothed metric, inject a steering phrase.

    Args:
        steering_text:        Text to inject (passed verbatim to the tokenizer).
        drop_threshold:       Smoothed-metric Δ that triggers injection.
        window:               Smoothing window.
        min_position:         Don't trigger before this position.
        cooldown:             Min tokens between consecutive injections.
        also_lower_temp:      If set, also drop temperature to this value when
                              triggering (combined intervention).
        max_injections:       Hard cap on number of injections per generation.
    """

    def __init__(
        self,
        steering_text: str = "\nWait, let me re-examine this carefully.\n",
        drop_threshold: float = 0.5,
        window: int = 30,
        min_position: int = 80,
        cooldown: int = 60,
        also_lower_temp: Optional[float] = None,
        max_injections: int = 3,
    ):
        self.steering_text = steering_text
        self.drop_threshold = drop_threshold
        self.window = window
        self.min_position = min_position
        self.cooldown = cooldown
        self.also_lower_temp = also_lower_temp
        self.max_injections = max_injections
        self._last_injection_at = -cooldown
        self._n_injections = 0

    def reset(self) -> None:
        self._last_injection_at = -self.cooldown
        self._n_injections = 0

    @property
    def name(self) -> str:
        return "drop_steering"

    def describe(self) -> str:
        return (
            f"drop_steering(drop_thr={self.drop_threshold}, window={self.window}, "
            f"text={self.steering_text!r:.40}, max_inject={self.max_injections})"
        )

    def _smoothed_at(self, hist: List[float], idx: int) -> float:
        half = self.window // 2
        lo = max(0, idx - half)
        hi = min(len(hist), idx + half + 1)
        return float(np.mean(hist[lo:hi]))

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        if position < self.min_position or len(metric_history) < self.window + 1:
            return InterventionDecision()
        if position - self._last_injection_at < self.cooldown:
            return InterventionDecision()
        if self._n_injections >= self.max_injections:
            return InterventionDecision()

        sm_now  = self._smoothed_at(metric_history, position)
        sm_prev = self._smoothed_at(metric_history, position - self.window)
        delta = sm_now - sm_prev

        if delta < -self.drop_threshold:
            self._last_injection_at = position
            self._n_injections += 1
            return InterventionDecision(
                inject_text=self.steering_text,
                set_temperature=self.also_lower_temp,
                reason=(
                    f"drop Δ={delta:.3f} < -{self.drop_threshold}: "
                    f"injecting steering text"
                ),
            )
        return InterventionDecision()


# ──────────────────────────────────────────────────────────────────────────────
# ConfidentRegionEndStrategy
# ──────────────────────────────────────────────────────────────────────────────

class ConfidentRegionEndStrategy(InterventionStrategy):
    """
    Wait for a confident region, then act on the FIRST drop afterwards.

    State machine:
        WARMUP        — at start, ignore everything
        WAITING       — looking for the start of a confident region
                       (smoothed metric ≥ threshold for `sustained_window`)
        IN_CONFIDENT  — saw confidence; watching for the next drop
        DONE          — already acted (one shot)

    On the action, the strategy can do any combination of:
      - inject a steering phrase
      - lower the temperature
      - save a checkpoint at the position where confidence ended

    Args:
        confidence_threshold:  Smoothed metric ≥ this counts as "confident".
        sustained_window:      Tokens of confidence before considering us in-region.
        drop_threshold:        Drop magnitude that ends the region.
        smoothing_window:      Rolling mean window.
        steering_text:         Optional text to inject when triggering. None = no inject.
        cool_to_temp:          Optional new temperature when triggering. None = no change.
        max_position:          Stop watching past this position (give up).
        min_position:          Don't start looking for a confident region before this
                               position.  Use this to skip structural humps/dips/spikes
                               that affect both correct and incorrect traces (e.g. the
                               hump at fraction 0.2–0.3 in char_occur / distinct_char,
                               the early spike in aime24, the U-curve in gpqa_diamond).
    """

    def __init__(
        self,
        confidence_threshold: float = -2.0,
        sustained_window:     int = 50,
        drop_threshold:       float = 0.5,
        smoothing_window:     int = 30,
        steering_text:        Optional[str] = None,
        cool_to_temp:         Optional[float] = None,
        max_position:         int = 100000,
        min_position:         int = 0,
    ):
        if steering_text is None and cool_to_temp is None:
            raise ValueError(
                "Specify at least one of steering_text or cool_to_temp "
                "— otherwise this strategy does nothing."
            )
        self.confidence_threshold = confidence_threshold
        self.sustained_window = sustained_window
        self.drop_threshold = drop_threshold
        self.smoothing_window = smoothing_window
        self.steering_text = steering_text
        self.cool_to_temp = cool_to_temp
        self.max_position = max_position
        self.min_position = min_position
        self._state = "WAITING"
        self._consecutive = 0

    def reset(self) -> None:
        self._state = "WAITING"
        self._consecutive = 0

    @property
    def name(self) -> str:
        return "confident_region_end"

    def describe(self) -> str:
        return (
            f"confident_region_end(thr={self.confidence_threshold}, "
            f"sustained={self.sustained_window}, drop_thr={self.drop_threshold}, "
            f"inject={self.steering_text is not None}, "
            f"cool_to={self.cool_to_temp}, min_pos={self.min_position})"
        )

    def _smoothed(self, hist: List[float]) -> float:
        n = min(self.smoothing_window, len(hist))
        return float(np.mean(hist[-n:])) if n else 0.0

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        if position < self.min_position:
            return InterventionDecision()
        if position > self.max_position or self._state == "DONE":
            return InterventionDecision()

        smoothed = self._smoothed(metric_history)

        if self._state == "WAITING":
            if smoothed >= self.confidence_threshold:
                self._consecutive += 1
                if self._consecutive >= self.sustained_window:
                    self._state = "IN_CONFIDENT"
                    self._consecutive = 0
                    # Save a checkpoint here — this is a "good" position
                    return InterventionDecision(
                        save_checkpoint=True,
                        reason=f"entered confident region at pos {position}",
                    )
            else:
                self._consecutive = 0
            return InterventionDecision()

        # IN_CONFIDENT — watch for drop
        if len(metric_history) < self.smoothing_window + 1:
            return InterventionDecision()

        sm_prev = float(np.mean(
            metric_history[-2 * self.smoothing_window : -self.smoothing_window]
        ))
        delta = smoothed - sm_prev

        if delta < -self.drop_threshold:
            self._state = "DONE"
            return InterventionDecision(
                inject_text=self.steering_text,
                set_temperature=self.cool_to_temp,
                reason=(
                    f"end of confident region at pos {position} "
                    f"(Δ={delta:.3f}): triggering steering"
                ),
            )

        return InterventionDecision()
