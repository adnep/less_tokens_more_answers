"""
src.interventions — pluggable mid-generation backtracking for LLM reasoning.

Public API
----------
    from src.interventions import InterventionEngine
    from src.interventions.metrics import get_metric
    from src.interventions.strategies import get_strategy

    engine = InterventionEngine(
        helper   = helper,
        metric   = get_metric("sc"),
        strategy = get_strategy("drop_detect", drop_threshold=0.5),
    )
    result = engine.generate(prompt, answer_type="integer")
"""

from .engine import InterventionEngine
from .result import GenerationResult, BacktrackEvent

__all__ = ["InterventionEngine", "GenerationResult", "BacktrackEvent"]
