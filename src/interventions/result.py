"""
Dataclasses for backtracking / intervention generation results.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BacktrackEvent:
    """Records a single backtracking event during generation."""
    trigger_position: int    # Token index (from start of generation) where backtrack triggered
    backtrack_to: int        # Token index we restored to (first token to re-generate)
    trigger_metric: float    # Metric value at the trigger token
    restore_metric: float    # Metric value at the restore point (may be 0.0 if at start)
    reason: str              # Human-readable trigger reason from the strategy
    n_backtrack: int         # Sequential backtrack number (1-indexed)


@dataclass
class InterventionEvent:
    """
    Records a non-backtracking intervention (stop / inject / temperature change).

    type:
      "stop"             — generation terminated by strategy
      "inject"           — extra tokens were injected by strategy (steering)
      "set_temperature"  — sampling temperature was modified
      "set_top_p"        — top-p was modified
    """
    type: str
    position: int            # Token index where the event happened
    detail: Any              # Type-specific payload
                             #   stop:            None
                             #   inject:          (List[int] tokens, str text)
                             #   set_temperature: (old_value, new_value)
                             #   set_top_p:       (old_value, new_value)
    reason: str = ""


@dataclass
class GenerationResult:
    """Full result of one backtracking / intervention generation call."""
    # Input
    prompt: str
    metric_name: str         # "sc" or "lp"
    strategy_name: str       # e.g. "drop_detect", "cooling", "early_stop"

    # Final output (after all backtracks resolved)
    generated_text: str
    all_tokens: List[int]    # Final sequence of generated token IDs
    metric_values: List[float]  # Metric at each final token position

    # Backtracking telemetry
    backtrack_events: List[BacktrackEvent]
    n_backtracks: int
    total_tokens_generated: int  # Includes tokens discarded by backtracking

    # Other interventions (early-stop / inject / temperature changes)
    interventions: List[InterventionEvent] = field(default_factory=list)

    # Evaluation
    extracted_answer: Optional[str] = None
    finish_reason: str = "max_tokens"
    # Possible values:
    #   "eos"             — model emitted EOS token
    #   "max_tokens"      — hit max_tokens cap
    #   "max_backtracks"  — exhausted backtrack budget
    #   "early_stop"      — strategy requested termination
