"""
Pluggable "can we answer now?" classifiers used by EarlyStopClassifierStrategy.

A classifier consumes the partial generated text and returns:
    (can_answer, predicted_answer, confidence)

The strategy decides whether to terminate generation based on these.

Available implementations
-------------------------
SelfPromptedClassifier
    Re-feeds the (truncated) reasoning trace to the SAME model with a
    "Based on the reasoning above, the final answer is:" prompt and asks
    it to extract the answer.  Uses the engine's own helper, so no
    additional weights are loaded.  More powerful but more expensive.

Adding a new classifier
-----------------------
1. Create src/interventions/classifiers/my_classifier.py with a class that
   inherits AnswerClassifier and implements .can_answer().
2. Import it below and add to _REGISTRY.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ClassifierResult:
    """Returned by AnswerClassifier.can_answer()."""
    can_answer: bool                  # True = stop generation now
    predicted_answer: Optional[str]   # Extracted answer (if any)
    confidence: float                 # ∈ [0, 1] — strategy may ignore
    reason: str = ""                  # For logging


class AnswerClassifier(ABC):
    """Decide whether the partial reasoning trace already contains the answer."""

    @abstractmethod
    def can_answer(self, generated_text: str) -> ClassifierResult:
        """
        Args:
            generated_text:  All tokens generated so far, decoded.
        Returns:
            ClassifierResult.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict = {}

# SelfPromptedClassifier is imported lazily because it needs torch + a helper
# instance — keep the package importable without GPU until a classifier is
# actually requested.
def _lazy_register_self_prompted():
    from .self_prompted import SelfPromptedClassifier
    _REGISTRY["self_prompted"] = SelfPromptedClassifier
    return SelfPromptedClassifier


def get_classifier(name: str, **kwargs) -> AnswerClassifier:
    if name == "self_prompted" and "self_prompted" not in _REGISTRY:
        _lazy_register_self_prompted()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown classifier: {name!r}.  Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


__all__ = ["AnswerClassifier", "ClassifierResult", "get_classifier"]
