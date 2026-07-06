"""
WaitBacktrackStrategy: backtrack when the model emits a self-correction token.

Reasoning models (Qwen3, QwQ, DeepSeek-R1) learn to emit tokens like "Wait"
when they detect a potential error in their own reasoning.  This strategy
treats that signal as a backtracking trigger: when a wait token appears and
the metric over the preceding window satisfies a threshold, we rewind to a
prior position and let the model try again from there.

Backtrack-target modes mirror DropDetectStrategy
-------------------------------------------------
"local_max"   Rewind to the peak of metric_history in the last search_window
              tokens before the wait.  This is the highest-confidence point
              we were at recently — a natural restart position.

"fixed_n"     Always go back exactly backtrack_n_tokens.  Simpler but coarser.

Threshold modes
---------------
"high"   Backtrack when mean metric before the wait is ABOVE threshold.
         The model was confident but is about to second-guess itself —
         trust the pre-wait reasoning and restart from the peak.

"low"    Backtrack when mean metric before the wait is BELOW threshold.
         The model was already uncertain; the wait confirms it — rewind
         to escape the low-confidence stretch.

"any"    Backtrack on every wait token regardless of metric.
"""

from typing import List, Optional

import numpy as np

from . import InterventionStrategy, InterventionDecision, _no_intervention
from ..conditions import WaitTokenCondition


class WaitBacktrackStrategy(InterventionStrategy):
    """
    Backtrack when the model emits a self-correction token (e.g. "Wait").

    Args:
        tokenizer:          Model tokenizer — used at init to resolve token IDs.
        wait_tokens:        Strings to watch for.  Default: ["Wait", " Wait"].
        window:             Tokens before the wait token to average the metric
                            over for the threshold check.
        threshold:          Metric mean threshold (used with exit_on "high"/"low").
        exit_on:            "high", "low", or "any" — see module docstring.
        backtrack_mode:     "local_max" or "fixed_n".
        backtrack_n_tokens: Tokens to rewind for "fixed_n" mode.
        search_window:      How far back to search for local max in "local_max"
                            mode.  Should be ≥ window.
        min_position:       Don't trigger in the first N generated tokens.
        cooldown:           Minimum tokens between consecutive backtracks.
    """

    def __init__(
        self,
        tokenizer,
        wait_tokens: Optional[List[str]] = None,
        window: int = 20,
        threshold: float = 0.5,
        exit_on: str = "high",
        backtrack_mode: str = "local_max",
        backtrack_n_tokens: int = 40,
        search_window: int = 60,
        min_position: int = 80,
        cooldown: int = 60,
    ):
        if backtrack_mode not in ("local_max", "fixed_n"):
            raise ValueError(
                f"backtrack_mode must be 'local_max' or 'fixed_n', got {backtrack_mode!r}"
            )
        self.backtrack_mode = backtrack_mode
        self.backtrack_n_tokens = backtrack_n_tokens
        self.search_window = search_window
        self.min_position = min_position
        self.cooldown = cooldown
        self._last_backtrack_at: int = -cooldown

        self._cond = WaitTokenCondition(
            tokenizer,
            tokens=wait_tokens,
            window=window,
            threshold=threshold,
            exit_on=exit_on,
        )

    def reset(self) -> None:
        self._last_backtrack_at = -self.cooldown
        self._cond.reset()

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        self._last_backtrack_at = backtrack_to

    @property
    def name(self) -> str:
        return "wait_backtrack"

    def describe(self) -> str:
        return (
            f"wait_backtrack({self._cond.describe()}, "
            f"mode={self.backtrack_mode}, search_win={self.search_window}, "
            f"min_pos={self.min_position}, cooldown={self.cooldown})"
        )

    def _find_backtrack_to(self, metric_history: List[float], trigger_pos: int) -> int:
        if self.backtrack_mode == "fixed_n":
            return max(0, trigger_pos - self.backtrack_n_tokens)
        search_start = max(0, trigger_pos - self.search_window)
        search_slice = metric_history[search_start:trigger_pos]
        if not search_slice:
            return max(0, trigger_pos - self.backtrack_n_tokens)
        return search_start + int(np.argmax(search_slice))

    def on_token(
        self,
        position: int,
        metric: float,
        metric_history: List[float],
        token_ids: List[int],
        n_backtracks_so_far: int,
    ) -> InterventionDecision:
        if position < self.min_position:
            return _no_intervention()

        if position - self._last_backtrack_at < self.cooldown:
            return _no_intervention()

        reason = self._cond.check(position, metric_history, token_ids)
        if reason:
            backtrack_to = self._find_backtrack_to(metric_history, position)
            return InterventionDecision(
                should_backtrack=True,
                backtrack_to=backtrack_to,
                reason=reason,
            )

        return _no_intervention()
