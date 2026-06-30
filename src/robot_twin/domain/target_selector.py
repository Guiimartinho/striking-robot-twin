"""TargetSelector: choose where a strike aims. Phase 1.

Safety-critical contract this module will implement: a strike at head height must
aim at an offset point BEHIND the head, never at the head itself. The SafetyArbiter
is the independent check on top, but the selector must not propose head-contact
targets in the first place. Body targets are opt-in and a later phase.
"""

from __future__ import annotations

from robot_twin.core.types import StrikeType, TraineePose, Vec3


def select_target(pose: TraineePose, strike_type: StrikeType) -> Vec3:  # noqa: D103
    raise NotImplementedError("TargetSelector is Phase 1 (perception and detection)")
