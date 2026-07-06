"""
ThinkBudgetStrategy — truncate the <think> reasoning block by injecting </think>.

Instead of letting the model reason until it naturally closes <think>, this
strategy forces a </think> injection so the model answers sooner.  It is a
token-saving intervention, not an accuracy-improving one: the trade-off is
fewer tokens spent reasoning in exchange for a possible slight accuracy drop.

Trigger modes
-------------
"fixed"        Inject at exactly max_think_tokens generated tokens.

"drop"         Inject as soon as the smoothed metric drops by drop_threshold.
               max_think_tokens acts as a hard fallback cap so the strategy
               always fires eventually.

"drop_or_fixed" Inject at whichever fires first — the hard cap or the drop.

"wait"         Inject when the model emits a wait token (e.g. "Wait") AND the
               mean metric over wait_window tokens before it satisfies the
               wait_threshold condition.  max_think_tokens is a fallback cap.

"wait_or_drop" Inject on wait condition OR metric drop, whichever fires first.
               max_think_tokens is a fallback cap.

"wait_or_fixed" Inject on wait condition OR hard cap, whichever fires first.

"any"          Inject when ANY trigger fires — hard cap OR metric drop OR wait
               condition, whichever comes first.

The strategy is one-shot per generate() call.  If </think> has already been
produced naturally, the strategy does nothing.
"""

import numpy as np
from typing import List, Optional

from . import InterventionStrategy, InterventionDecision
from ..conditions import WaitTokenCondition

_VALID_TRIGGERS = (
    "fixed", "drop", "drop_or_fixed", "wait", "wait_or_drop", "wait_or_fixed", "any",
)

class ThinkBudgetStrategy(InterventionStrategy):
    """
    Args:
        tokenizer:          Model tokenizer — used once at init to encode
                            </think> so we can detect if it already appeared.
        max_think_tokens:   Hard cap on reasoning tokens.  Injection fires at
                            this position for "fixed" and as a fallback for
                            "drop" / "wait" / "wait_or_drop".  Set very large
                            to effectively disable.
        trigger:            One of "fixed", "drop", "drop_or_fixed", "wait",
                            "wait_or_drop", "wait_or_fixed", "any".
        drop_threshold:     Smoothed-metric decrease that fires the drop trigger.
                            Good starting range: 0.3–0.8 for SC, 0.5–1.5 for LP.
        window:             Smoothing window (tokens) for the drop detector.
        min_position:       Don't inject before this position — avoids cutting
                            the think block before the model has started reasoning.
        think_close_text:   The string to inject.  Default "</think>" matches
                            Qwen3's natural output; adjust if your model uses
                            a different end-of-think marker.
        wait_tokens:        List of strings to watch for as self-correction
                            signals (e.g. ["Wait", " Wait"]).  Only used by
                            "wait*" trigger modes.
        wait_window:        Number of tokens before the wait token to average
                            the metric over for the threshold check.
        wait_threshold:     Metric mean threshold.  Combined with wait_exit_on
                            to decide whether to fire.
        wait_exit_on:       "high" — fire when mean metric > wait_threshold
                            (model was confident, exit before it second-guesses).
                            "low"  — fire when mean metric < wait_threshold
                            (model was already lost, cut it off).
                            "any"  — fire unconditionally on any wait token.
    """

    def __init__(
        self,
        tokenizer,
        max_think_tokens: int = 1024,
        trigger: str = "fixed",
        drop_threshold: float = 0.4,
        window: int = 30,
        min_position: int = 100,
        think_close_text: str = "</think>",
        wait_tokens: Optional[List[str]] = None,
        wait_window: int = 20,
        wait_threshold: float = 0.5,
        wait_exit_on: str = "high",
    ):
        if trigger not in _VALID_TRIGGERS:
            raise ValueError(
                f"trigger must be one of {_VALID_TRIGGERS}, got {trigger!r}"
            )
        if wait_exit_on not in ("high", "low", "any"):
            raise ValueError(
                f"wait_exit_on must be 'high', 'low', or 'any', got {wait_exit_on!r}"
            )
        self.max_think_tokens = max_think_tokens
        self.trigger = trigger
        self.drop_threshold = drop_threshold
        self.window = window
        self.min_position = min_position
        self.think_close_text = think_close_text
        self.wait_window = wait_window
        self.wait_threshold = wait_threshold
        self.wait_exit_on = wait_exit_on

        # Resolve the token ID(s) for think_close_text.
        # Priority: added_tokens_encoder (exact vocabulary entry) → encode fallback.
        # We use inject_token_ids (not inject_text) so the engine bypasses
        # tokenizer.encode() entirely and feeds the correct ID directly.
        added_id = tokenizer.added_tokens_encoder.get(think_close_text)
        if added_id is not None:
            self._inject_ids: list = [added_id]
        else:
            self._inject_ids = tokenizer.encode(think_close_text, add_special_tokens=False)
        self._think_close_ids: set = set(self._inject_ids)

        # Wait condition (only meaningful for "wait*" trigger modes).
        self._wait_cond = WaitTokenCondition(
            tokenizer,
            tokens=wait_tokens,
            window=wait_window,
            threshold=wait_threshold,
            exit_on=wait_exit_on,
        )

        self._fired = False

    def reset(self) -> None:
        self._fired = False

    @property
    def name(self) -> str:
        return "think_budget"

    def describe(self) -> str:
        base = (
            f"think_budget(max={self.max_think_tokens}, trigger={self.trigger!r}, "
            f"drop_thr={self.drop_threshold}, window={self.window})"
        )
        if self.trigger.startswith("wait") or self.trigger == "any":
            base += (
                f"[wait_win={self.wait_window}, "
                f"wait_thr={self.wait_threshold}, exit_on={self.wait_exit_on!r}]"
            )
        return base

    def _already_closed(self, token_ids: List[int]) -> bool:
        return bool(self._think_close_ids.intersection(token_ids))

    def _smoothed_at(self, hist: List[float], idx: int) -> float:
        half = self.window // 2
        lo = max(0, idx - half)
        hi = min(len(hist), idx + half + 1)
        return float(np.mean(hist[lo:hi]))

    def _drop_fired(self, position: int, metric_history: List[float]) -> bool:
        if len(metric_history) < self.window + 1:
            return False
        delta = (
            self._smoothed_at(metric_history, position)
            - self._smoothed_at(metric_history, position - self.window)
        )
        return delta < -self.drop_threshold


    def on_token(
        self,
        position: int,
        metric: float,
        metric_history: List[float],
        token_ids: List[int],
        n_backtracks_so_far: int,
    ) -> InterventionDecision:
        if self._fired or self._already_closed(token_ids):
            self._fired = True
            return InterventionDecision()

        if position < self.min_position:
            return InterventionDecision()

        fire, reason = False, ""

        uses_fixed = "fixed" in self.trigger or self.trigger == "any"
        uses_drop  = "drop" in self.trigger or self.trigger == "any"
        uses_wait  = "wait" in self.trigger or self.trigger == "any"

        # Hard cap (primary trigger for the modes that name it).
        if uses_fixed and position >= self.max_think_tokens:
            fire = True
            reason = f"think budget hit at pos {position} (max={self.max_think_tokens})"

        # Metric drop.
        if not fire and uses_drop and self._drop_fired(position, metric_history):
            sm_now  = self._smoothed_at(metric_history, position)
            sm_prev = self._smoothed_at(metric_history, position - self.window)
            fire = True
            reason = (
                f"metric drop Δ={sm_now - sm_prev:.3f} < -{self.drop_threshold} "
                f"at pos {position}: closing think early"
            )

        # Wait-token condition.
        if not fire and uses_wait:
            wait_reason = self._wait_cond.check(position, metric_history, token_ids)
            if wait_reason:
                fire = True
                reason = f"{wait_reason}: closing think early"

        # Fallback hard cap for modes that don't already use it as a primary trigger.
        if not fire and not uses_fixed and position >= self.max_think_tokens:
            fire = True
            reason = f"fallback cap at pos {position} (max={self.max_think_tokens})"

        if fire:
            self._fired = True
            return InterventionDecision(inject_token_ids=self._inject_ids, reason=reason)

        return InterventionDecision()

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        pass
