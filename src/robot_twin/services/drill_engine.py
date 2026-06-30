"""DrillEngine: the FSM / curriculum that runs a training session. Phase 2+.

Owns difficulty progression (telegraph time, speed, randomisation, feints),
seeded by the video-mined combo grammar. Every command it issues still passes
through the SafetyArbiter; the engine never bypasses safety.
"""

from __future__ import annotations

from enum import Enum


class DrillState(Enum):
    """Coarse FSM states for a drill session. Filled out in Phase 2+."""

    IDLE = "idle"
    TELEGRAPH = "telegraph"
    STRIKE = "strike"
    RECOVER = "recover"
    SCORE = "score"


class DrillEngine:
    """FSM/curriculum driver. Phase 2+."""

    def step(self) -> DrillState:  # noqa: D102
        raise NotImplementedError("DrillEngine is Phase 2+ (closed loop and curriculum)")
