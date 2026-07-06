"""
Pluggable trigger conditions for backtracking strategies.

A TriggerCondition is a lightweight signal detector: it inspects the current
token, position, and metric history and returns a reason string if it fires,
or None if it does not.  Conditions carry no action logic — they are passed
to strategies which decide what to do when the condition fires.

Usage inside a strategy
-----------------------
    class MyStrategy(InterventionStrategy):
        def __init__(self, ..., condition=None):
            self._cond = condition

        def on_token(self, position, metric, metric_history, token_ids, ...):
            if self._cond is not None:
                reason = self._cond.check(position, metric_history, token_ids)
                if reason:
                    return InterventionDecision(should_backtrack=True, reason=reason)
            ...

Available conditions
--------------------
WaitTokenCondition  — fires when the model emits a self-correction token
                      (e.g. "Wait") and the prior metric window satisfies a
                      configurable threshold.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


_DEFAULT_WAIT_TOKENS = ["Wait", " Wait"]


class TriggerCondition(ABC):
    """
    Abstract base for trigger conditions.

    Subclasses must implement check().  reset() is optional — override it if
    the condition keeps per-generation state (e.g. a one-shot flag).
    """

    @abstractmethod
    def check(
        self,
        position: int,
        metric_history: List[float],
        token_ids: List[int],
    ) -> Optional[str]:
        """
        Inspect the current generation step.

        Returns a non-empty reason string if the condition fires, None otherwise.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each generate() call. Override if needed."""
        pass


class WaitTokenCondition(TriggerCondition):
    """
    Fires when the most-recently generated token is a self-correction token
    (e.g. "Wait") AND the mean metric over the preceding `window` tokens
    satisfies the threshold condition.

    Args:
        tokenizer:  Model tokenizer — used once at init to resolve token IDs.
        tokens:     Strings to watch for.  Default: ["Wait", " Wait"].
        window:     Number of tokens before the wait token to average the
                    metric over.
        threshold:  Metric mean threshold (used with exit_on "high"/"low").
        exit_on:    "high" — fire when mean metric > threshold
                             (model was confident; exit before it second-guesses).
                    "low"  — fire when mean metric < threshold
                             (model was already uncertain; cut it off).
                    "any"  — fire on any wait token regardless of metric.
    """

    def __init__(
        self,
        tokenizer,
        tokens: Optional[List[str]] = None,
        window: int = 20,
        threshold: float = 0.5,
        exit_on: str = "high",
    ):
        if exit_on not in ("high", "low", "any"):
            raise ValueError(f"exit_on must be 'high', 'low', or 'any', got {exit_on!r}")
        self.window = window
        self.threshold = threshold
        self.exit_on = exit_on

        _toks = tokens if tokens is not None else _DEFAULT_WAIT_TOKENS
        self._wait_ids: set = set()
        for tok in _toks:
            added = tokenizer.added_tokens_encoder.get(tok)
            if added is not None:
                self._wait_ids.add(added)
            else:
                self._wait_ids.update(tokenizer.encode(tok, add_special_tokens=False))

    def check(
        self,
        position: int,
        metric_history: List[float],
        token_ids: List[int],
    ) -> Optional[str]:
        if not token_ids or token_ids[-1] not in self._wait_ids:
            return None

        if self.exit_on == "any":
            return f"wait token at pos {position}"

        lo = max(0, position - self.window)
        window_metrics = metric_history[lo:position]
        if not window_metrics:
            return None

        mean_m = float(np.mean(window_metrics))

        if self.exit_on == "high" and mean_m > self.threshold:
            return (
                f"wait token at pos {position}, "
                f"mean metric={mean_m:.3f} > {self.threshold} (high conf)"
            )
        if self.exit_on == "low" and mean_m < self.threshold:
            return (
                f"wait token at pos {position}, "
                f"mean metric={mean_m:.3f} < {self.threshold} (low conf)"
            )
        return None

    def describe(self) -> str:
        return (
            f"wait_condition(win={self.window}, thr={self.threshold}, "
            f"exit_on={self.exit_on!r})"
        )
