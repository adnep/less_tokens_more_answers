"""
EarlyStopClassifierStrategy — stop generation when an external classifier
says the partial reasoning trace already determines the answer.

The classifier is decoupled from the strategy via the AnswerClassifier
interface (see src/interventions/classifiers/), so you can plug in:

  • SelfPromptedClassifier  — re-prompt same model to extract the answer
  • <your own>              — anything that implements .can_answer()

Cost control
------------
The classifier is invoked at most every `check_every` tokens (default 200)
and only after `min_position` tokens have been generated.  This keeps the
overhead bounded — important for the model-based classifiers.

Optional gating by confidence region
------------------------------------
If `require_confidence_first` is True, the strategy first waits for a
sustained confident region (smoothed metric ≥ confidence_threshold for
sustained_window tokens) BEFORE it starts polling the classifier.  This
matches the user's "first build confidence, then check" workflow.
"""

from typing import List, Optional
import numpy as np

from . import InterventionStrategy, InterventionDecision
from ..classifiers import AnswerClassifier


class EarlyStopClassifierStrategy(InterventionStrategy):
    """
    Args:
        classifier:               AnswerClassifier instance.
        tokenizer:                Tokenizer used to decode token_ids → text
                                  for the classifier.  Pass helper.tokenizer.
        check_every:              Polling cadence in tokens.
        min_position:             Don't poll before this position.
        require_confidence_first: Gate polling on a sustained confident region.
        confidence_threshold:     Smoothed-metric threshold for "confident".
        sustained_window:         Tokens of confidence required before polling.
        smoothing_window:         Rolling-mean window for the confidence test.
    """

    def __init__(
        self,
        classifier: AnswerClassifier,
        tokenizer,
        check_every: int = 200,
        min_position: int = 300,
        require_confidence_first: bool = False,
        confidence_threshold: float = -2.0,
        sustained_window: int = 30,
        smoothing_window: int = 30,
    ):
        self.classifier = classifier
        self.tokenizer  = tokenizer
        self.check_every = check_every
        self.min_position = min_position
        self.require_confidence_first = require_confidence_first
        self.confidence_threshold = confidence_threshold
        self.sustained_window = sustained_window
        self.smoothing_window = smoothing_window
        self._last_check: int = -check_every
        self._confident_seen: bool = not require_confidence_first
        self._consecutive: int = 0

    def reset(self) -> None:
        self._last_check = -self.check_every
        self._confident_seen = not self.require_confidence_first
        self._consecutive = 0

    @property
    def name(self) -> str:
        return f"early_stop({self.classifier.name})"

    def describe(self) -> str:
        return (
            f"early_stop(classifier={self.classifier.name}, "
            f"check_every={self.check_every}, min_pos={self.min_position}, "
            f"gate_on_confidence={self.require_confidence_first})"
        )

    def _smoothed(self, hist: List[float]) -> float:
        n = min(self.smoothing_window, len(hist))
        return float(np.mean(hist[-n:])) if n else 0.0

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        # Track confidence region if gating
        if self.require_confidence_first and not self._confident_seen:
            if self._smoothed(metric_history) >= self.confidence_threshold:
                self._consecutive += 1
                if self._consecutive >= self.sustained_window:
                    self._confident_seen = True
            else:
                self._consecutive = 0

        if position < self.min_position:
            return InterventionDecision()
        if not self._confident_seen:
            return InterventionDecision()
        if position - self._last_check < self.check_every:
            return InterventionDecision()

        self._last_check = position

        # Decode trace and ask the classifier
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        result = self.classifier.can_answer(text)

        if result.can_answer:
            return InterventionDecision(
                stop_generation=True,
                reason=(
                    f"early-stop at pos {position}: "
                    f"classifier says answer={result.predicted_answer!r} "
                    f"(conf={result.confidence:.2f}) — {result.reason}"
                ),
            )
        return InterventionDecision()
