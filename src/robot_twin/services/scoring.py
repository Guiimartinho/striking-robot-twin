"""Scoring: grade the trainee's dodges and guard over a drill. Phase 2.

Aggregates per-strike outcomes into a running session score and a summary. The
record API takes primitives, not the engine's outcome type, so scoring stays
decoupled from the drill FSM.

Safety outcomes are tracked but never rewarded: a vetoed or aborted strike is a
safety event, not a training rep, and is kept out of the dodge rate so the score
reflects only strikes the trainee actually faced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Snapshot of a drill session."""

    episodes: int
    resolved: int  # strikes actually delivered to the trainee (not vetoed/aborted)
    dodges: int
    hits: int  # resolved but not dodged (a non-impacting tag)
    guards_up: int
    aborts: int  # strikes aborted mid-flight by the safety layer
    vetoes: int  # strikes rejected at launch by the safety layer
    feints: int  # telegraphed but withdrawn; not delivered, not scored
    score: float

    @property
    def dodge_rate(self) -> float:
        """Fraction of delivered strikes the trainee dodged."""
        return self.dodges / self.resolved if self.resolved else 0.0


class Scorer:
    """Accumulates drill outcomes into a score and summary."""

    def __init__(
        self, dodge_points: float = 1.0, guard_points: float = 0.5, hit_penalty: float = 0.0
    ) -> None:
        self._dodge_points = dodge_points
        self._guard_points = guard_points
        self._hit_penalty = hit_penalty
        self._episodes = 0
        self._resolved = 0
        self._dodges = 0
        self._hits = 0
        self._guards_up = 0
        self._aborts = 0
        self._vetoes = 0
        self._feints = 0
        self._score = 0.0

    def record(
        self,
        *,
        dodged: bool | None,
        guard_up: bool | None,
        aborted: bool,
        vetoed: bool,
        feinted: bool = False,
    ) -> None:
        """Record one drill episode.

        Args:
            dodged: Whether the trainee dodged, or None if the strike was never
                delivered (vetoed, aborted or feinted).
            guard_up: Whether the guard was up at arrival, or None if not delivered.
            aborted: The strike was aborted mid-flight by the safety layer.
            vetoed: The strike was rejected at launch by the safety layer.
            feinted: The strike was a feint, telegraphed but withdrawn.
        """
        self._episodes += 1
        if vetoed:
            self._vetoes += 1
            return
        if aborted:
            self._aborts += 1
            return
        if feinted:
            self._feints += 1
            return

        self._resolved += 1
        if dodged:
            self._dodges += 1
            self._score += self._dodge_points
        else:
            self._hits += 1
            self._score += self._hit_penalty
        if guard_up:
            self._guards_up += 1
            self._score += self._guard_points

    def summary(self) -> ScoreSummary:
        """Return the current session summary."""
        return ScoreSummary(
            episodes=self._episodes,
            resolved=self._resolved,
            dodges=self._dodges,
            hits=self._hits,
            guards_up=self._guards_up,
            aborts=self._aborts,
            vetoes=self._vetoes,
            feints=self._feints,
            score=self._score,
        )
