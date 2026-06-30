"""StrikePlanner: turn a chosen target into a StrikeCommand. Phase 1+.

Plans the punch (type, speed, telegraph) toward a selected target. The articular
trajectory is learned in the sim, never copied from video (morphology mismatch).
"""

from __future__ import annotations

from robot_twin.core.types import StrikeCommand, StrikeType, Vec3


def plan_strike(target: Vec3, strike_type: StrikeType, arm_id: int) -> StrikeCommand:  # noqa: D103
    raise NotImplementedError("StrikePlanner is Phase 1+")
