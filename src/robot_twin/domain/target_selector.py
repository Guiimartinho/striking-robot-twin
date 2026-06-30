"""TargetSelector: choose where a strike aims. Phase 2 (body drill).

Safety-critical contract: never propose a head-contact target. For the 1-DOF
Phase 2 arm, which moves only in +y at a fixed (x, z), the only honest aim is a
point on that reachable line, so the selector returns a body target at the arm's
own x and z and a depth in y derived from the trainee's chest. Aiming at the arm
plane (body height) keeps the strike line clear of the head and neck keep-out;
the SafetyArbiter is the independent check on top.

The target x is the arm's x, not the trainee's: the 1-DOF arm cannot track the
trainee sideways. That is the point of the drill: the trainee slips laterally off
the fixed line, and a successful dodge is the glove passing through empty space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_twin.core.types import Keypoint, StrikeType, TraineePose, Vec3


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Parameters for body-target selection, in metres.

    ``body_standoff_m`` is large enough that the glove stops short of the body:
    Phase 2 is a non-contact dodge drill. Per the safety spec, body contact is
    opt-in and only a later phase (padded vest, validated energy envelope), so
    the telegraphed feint comes near the trainee but does not strike them.
    """

    body_standoff_m: float = 0.20  # aim this far in front of the chest centre (no contact)
    default_depth_m: float = 0.5  # fallback strike depth in y if the chest is unseen
    min_confidence: float = 0.5


class TargetSelector:
    """Selects a safe body target on the arm's reachable line."""

    def __init__(self, config: TargetConfig | None = None) -> None:
        self._cfg = config or TargetConfig()

    def select(
        self, pose: TraineePose, arm_base: Vec3, strike_type: StrikeType = StrikeType.JAB
    ) -> Vec3:
        """Choose the aim point for a strike from this arm.

        Args:
            pose: Current trainee keypoints.
            arm_base: World position of the striking arm's base.
            strike_type: Punch family (kept for the contract; the Phase 2 body
                drill treats all punches the same depth-wise).

        Returns:
            A target at the arm's x and z (the reachable plane) and a y depth
            into the trainee's body. Falls back to a default depth when the chest
            keypoint is unavailable, so selection never depends on a dropped point.
        """
        chest_idx = int(Keypoint.CHEST)
        chest_seen = pose.confidence[chest_idx] >= self._cfg.min_confidence and np.all(
            np.isfinite(pose.positions[chest_idx])
        )
        if chest_seen:
            depth_y = float(pose.positions[chest_idx][1]) - self._cfg.body_standoff_m
        else:
            depth_y = arm_base.y + self._cfg.default_depth_m
        return Vec3(arm_base.x, depth_y, arm_base.z)
