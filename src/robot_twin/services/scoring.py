"""Scoring: grade the trainee's dodges and guard over a drill. Phase 2+.

Aggregates per-strike dodge/guard outcomes into a session score and feedback.
"""

from __future__ import annotations


class Scorer:
    """Accumulates drill outcomes into a score. Phase 2+."""

    def record(self, dodged: bool, guarded: bool) -> None:  # noqa: D102
        raise NotImplementedError("Scoring is Phase 2+ (closed loop)")
