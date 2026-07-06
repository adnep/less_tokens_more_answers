"""
Composite and utility strategies.

NoInterventionStrategy — never intervene (baseline / control condition).
AnyStrategy     — backtrack if ANY sub-strategy votes to.
AllStrategy     — backtrack only if ALL sub-strategies vote to.

These let you compose strategies from the registry without writing new code.
For example, to require BOTH a drop AND a threshold breach:

    strategy = AllStrategy([
        DropDetectStrategy(drop_threshold=0.5),
        ThresholdStrategy(threshold=-9.0),
    ])

Or to trigger on either signal:

    strategy = AnyStrategy([
        DropDetectStrategy(drop_threshold=0.3),
        ThresholdStrategy(threshold=-12.0),
    ])
"""

from typing import List
from . import InterventionStrategy, InterventionDecision, _no_intervention


class NoInterventionStrategy(InterventionStrategy):
    """
    Never intervene.

    Use as the baseline: run this to measure accuracy WITHOUT any
    intervention, then compare against other strategies.
    """

    @property
    def name(self) -> str:
        return "no_intervention"

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        return _no_intervention()


class AnyStrategy(InterventionStrategy):
    """
    Backtrack if at least one sub-strategy says to (OR logic).

    The first sub-strategy that triggers determines the backtrack position.
    All strategies' reset() and on_backtrack() are forwarded.
    """

    def __init__(self, strategies: List[InterventionStrategy]):
        if not strategies:
            raise ValueError("AnyStrategy requires at least one sub-strategy.")
        self.strategies = strategies

    @property
    def name(self) -> str:
        return "any(" + ", ".join(s.name for s in self.strategies) + ")"

    def reset(self) -> None:
        for s in self.strategies:
            s.reset()

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        for s in self.strategies:
            s.on_backtrack(backtrack_to, n_backtracks)

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        for s in self.strategies:
            decision = s.on_token(position, metric, metric_history, token_ids, n_backtracks_so_far)
            if decision.should_backtrack:
                return decision  # first trigger wins
        return _no_intervention()


class AllStrategy(InterventionStrategy):
    """
    Backtrack only if ALL sub-strategies agree (AND logic).

    The backtrack position is taken from the LAST strategy that triggers
    (typically the most specific / aggressive one — adjust ordering as needed).
    """

    def __init__(self, strategies: List[InterventionStrategy]):
        if not strategies:
            raise ValueError("AllStrategy requires at least one sub-strategy.")
        self.strategies = strategies

    @property
    def name(self) -> str:
        return "all(" + ", ".join(s.name for s in self.strategies) + ")"

    def reset(self) -> None:
        for s in self.strategies:
            s.reset()

    def on_backtrack(self, backtrack_to: int, n_backtracks: int) -> None:
        for s in self.strategies:
            s.on_backtrack(backtrack_to, n_backtracks)

    def on_token(self, position, metric, metric_history, token_ids, n_backtracks_so_far):
        decisions = [
            s.on_token(position, metric, metric_history, token_ids, n_backtracks_so_far)
            for s in self.strategies
        ]
        if all(d.should_backtrack for d in decisions):
            # Use the last triggering decision's backtrack position
            triggering = [d for d in decisions if d.should_backtrack]
            last = triggering[-1]
            reasons = " AND ".join(d.reason for d in triggering)
            return InterventionDecision(
                should_backtrack=True,
                backtrack_to=last.backtrack_to,
                reason=reasons,
            )
        return _no_intervention()
