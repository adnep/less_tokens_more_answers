"""
Pluggable intervention strategies for the generation engine.

An InterventionStrategy is called once per generated token and decides what
intervention (if any) to apply at this step — backtracking, early-stop, token
injection, or a sampling-parameter change.  See InterventionDecision for the
full set of actions a strategy can request.

The engine calls:
  decision = strategy.on_token(position, metric, metric_history, token_ids, n_backtracks)
  # ... if backtracking:
  strategy.on_backtrack(backtrack_to, n_backtracks)
  # ... at start of each generate():
  strategy.reset()

Adding a new strategy
---------------------
1. Create src/interventions/strategies/my_strategy.py with a class that
   inherits InterventionStrategy and implements .on_token() and .name.
2. Import it below and add to _REGISTRY.
3. Done — pass --strategy my_strategy to the CLI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# ── Core dataclass ─────────────────────────────────────────────────────────────

@dataclass
class InterventionDecision:
    """
    Returned by strategy.on_token().  This is a general INTERVENTION decision:
    it can request any combination of backtracking, early-stop, token
    injection, and sampling-parameter changes.

    Multiple actions can be combined in one decision.  The engine processes
    them in this order each step:

        1. save_checkpoint           → save KV cache at current position
        2. stop_generation           → break the loop immediately
        3. should_backtrack          → rewind, restart loop
        4. set_temperature / top_p   → update sampling params for next step
        5. inject_tokens / text      → feed extra tokens through the model

    Attributes
    ----------
    BACKTRACKING
        should_backtrack:    Whether to initiate a backtrack.
        backtrack_to:        Token index (0-based from start of generation) to
                             restore to.  Only meaningful when should_backtrack.

    EARLY STOP
        stop_generation:     If True, terminate generation NOW.  Used by
                             classifier-based or confidence-based strategies that decide the
                             reasoning so far is sufficient to extract an answer.

    STEERING (token injection)
        inject_token_ids:    List of token ids to feed through the model AFTER
                             the current step's token, before next sampling.
        inject_text:         Alternative: text to tokenise and inject.  Cannot
                             be combined with inject_token_ids.

    TEMPERATURE / TOP-P STEERING
        set_temperature:     New sampling temperature, applied from next step on
                             (overrides any previous setting until changed again).
                             None = keep current.
        set_top_p:           Same idea for nucleus top-p.

    ENGINE HINTS
        save_checkpoint:     Save a KV checkpoint at the CURRENT position
                             (regardless of periodic interval).  Useful at
                             "good" positions so future backtracks are free.

    BOOK-KEEPING
        reason:              Human-readable explanation, logged with the event.
    """
    # Backtracking
    should_backtrack: bool = False
    backtrack_to: int = -1

    # Early stop
    stop_generation: bool = False

    # Steering: inject tokens
    inject_token_ids: Optional[List[int]] = None
    inject_text: Optional[str] = None

    # Sampling-param steering
    set_temperature: Optional[float] = None
    set_top_p: Optional[float] = None

    # Engine hints
    save_checkpoint: bool = False

    # Logging
    reason: str = ""


def _no_intervention() -> InterventionDecision:
    return InterventionDecision(should_backtrack=False)


# ── Abstract base ──────────────────────────────────────────────────────────────

class InterventionStrategy(ABC):
    """
    Abstract base class for intervention strategies.

    Subclasses must implement:
      - on_token()
      - name (property)

    Subclasses may optionally override:
      - reset()       — called at the start of each generate() call
      - on_backtrack()— called after the engine commits to a backtrack
    """

    @abstractmethod
    def on_token(
        self,
        position: int,
        metric: float,
        metric_history: List[float],
        token_ids: List[int],
        n_backtracks_so_far: int,
    ) -> InterventionDecision:
        """
        Called after each generated token, BEFORE it is committed.

        Args:
            position:           0-indexed token position in the generated sequence.
                                Equals len(metric_history) - 1.
            metric:             Metric value at this position.
            metric_history:     Full metric history up to and including this position.
            token_ids:          Full token id history up to and including this position.
            n_backtracks_so_far: How many backtracks have already happened.

        Returns:
            InterventionDecision.
        """
        ...

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        """
        Called by the engine after it commits to a backtrack.
        Use this to update cooldown counters, etc.

        Args:
            backtrack_to:  The position we're restoring to.
            n_backtracks:  Total backtracks including this one.
        """
        pass

    def reset(self) -> None:
        """Reset all internal state. Called at the start of each generate() call."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string for this strategy."""
        ...

    def describe(self) -> str:
        """Human-readable description of strategy config."""
        return self.name


# ── Registry ───────────────────────────────────────────────────────────────────

from .threshold      import ThresholdStrategy    # noqa: E402
from .drop_detect    import DropDetectStrategy   # noqa: E402
from .composite      import NoInterventionStrategy, AnyStrategy, AllStrategy  # noqa: E402
from .cooling        import LinearCoolingStrategy, ConfidenceCoolingStrategy  # noqa: E402
from .steering       import DropSteeringStrategy, ConfidentRegionEndStrategy  # noqa: E402
from .think_budget   import ThinkBudgetStrategy                               # noqa: E402
from .wait_backtrack import WaitBacktrackStrategy                             # noqa: E402
from ..conditions    import TriggerCondition, WaitTokenCondition              # noqa: E402
# early_stop is imported lazily because it pulls in the classifiers package
# which may need a tokenizer / model — see _lazy_register_early_stop below.

_REGISTRY: dict = {
    "no_intervention":      NoInterventionStrategy,
    "threshold":            ThresholdStrategy,
    "drop_detect":          DropDetectStrategy,
    "linear_cooling":       LinearCoolingStrategy,
    "confidence_cooling":   ConfidenceCoolingStrategy,
    "drop_steering":        DropSteeringStrategy,
    "confident_region_end": ConfidentRegionEndStrategy,
    "think_budget":         ThinkBudgetStrategy,
    "wait_backtrack":       WaitBacktrackStrategy,
}


def _lazy_register_early_stop():
    from .early_stop import EarlyStopClassifierStrategy
    _REGISTRY["early_stop"] = EarlyStopClassifierStrategy
    return EarlyStopClassifierStrategy


def get_strategy(name: str, **kwargs) -> InterventionStrategy:
    """
    Instantiate a strategy by name.

    Args:
        name:   One of the keys in _REGISTRY.
        kwargs: Forwarded to the strategy's __init__.

    Returns:
        A InterventionStrategy instance.
    """
    if name == "early_stop" and "early_stop" not in _REGISTRY:
        _lazy_register_early_stop()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name!r}.  Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "InterventionDecision",
    "InterventionStrategy",
    "TriggerCondition",
    "WaitTokenCondition",
    "ThresholdStrategy",
    "DropDetectStrategy",
    "NoInterventionStrategy",
    "AnyStrategy",
    "AllStrategy",
    "LinearCoolingStrategy",
    "ConfidenceCoolingStrategy",
    "DropSteeringStrategy",
    "ConfidentRegionEndStrategy",
    "ThinkBudgetStrategy",
    "WaitBacktrackStrategy",
    "get_strategy",
]
