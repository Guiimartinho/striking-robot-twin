"""DodgeDetector: decide whether the trainee dodged an incoming strike. Phase 1.

A dodge counts when, by the time the strike arrives, the head has both (a) moved
meaningfully from where it was when the strike launched and (b) ended up clear of
the strike line by a margin. Requiring real displacement separates a genuine slip
from a head that was never on the line to begin with.

The strike line is the segment from the strike origin (arm base) to the aim
point; clearance is the head's shortest distance to that segment. Thresholds keep
a wide margin over pose jitter so injected noise does not flip the verdict, which
is the Phase 1 reliability gate. A non-impacting drill: "not dodged" is a logical
tag (buzzer/LED), never a strike projected onto the head.

Fail-safe: a dropped or low-confidence head keypoint yields UNKNOWN, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_twin.core.geometry import point_to_segment_distance
from robot_twin.core.types import Keypoint, TraineePose, Vec3


class DodgeState(Enum):
    """Outcome of a dodge assessment."""

    DODGED = "dodged"
    NOT_DODGED = "not_dodged"
    UNKNOWN = "unknown"  # head not reliably observed at launch or arrival


@dataclass(frozen=True, slots=True)
class DodgeConfig:
    """Thresholds for a successful dodge, in metres."""

    min_clearance_m: float = 0.15  # head must end at least this far from the line
    min_displacement_m: float = 0.10  # head must have moved at least this much
    min_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class DodgeResult:
    """Result of a dodge assessment, with the measurements behind it."""

    state: DodgeState
    clearance_m: float
    displacement_m: float


class DodgeDetector:
    """Judges whether the head left the strike line in time."""

    def __init__(self, config: DodgeConfig | None = None) -> None:
        self._cfg = config or DodgeConfig()

    def evaluate(
        self,
        strike_origin: Vec3,
        strike_target: Vec3,
        pose_at_launch: TraineePose,
        pose_now: TraineePose,
    ) -> DodgeResult:
        """Assess a dodge from the launch pose and the arrival pose.

        Args:
            strike_origin: Where the strike starts (the arm base), world frame.
            strike_target: The aim point, world frame.
            pose_at_launch: Trainee pose when the strike was launched.
            pose_now: Trainee pose when the strike arrives.

        Returns:
            A DodgeResult. UNKNOWN when the head is dropped or below the
            confidence floor in either pose, or any needed value is non-finite.
        """
        h = int(Keypoint.HEAD)
        if (
            pose_at_launch.confidence[h] < self._cfg.min_confidence
            or pose_now.confidence[h] < self._cfg.min_confidence
        ):
            return DodgeResult(DodgeState.UNKNOWN, clearance_m=0.0, displacement_m=0.0)

        head_launch = pose_at_launch.positions[h]
        head_now = pose_now.positions[h]
        if not np.all(np.isfinite(head_launch)) or not np.all(np.isfinite(head_now)):
            return DodgeResult(DodgeState.UNKNOWN, clearance_m=0.0, displacement_m=0.0)

        clearance = float(
            point_to_segment_distance(
                strike_origin.as_array(), strike_target.as_array(), head_now[None, :]
            )[0]
        )
        displacement = float(np.linalg.norm(head_now - head_launch))

        dodged = (
            clearance >= self._cfg.min_clearance_m and displacement >= self._cfg.min_displacement_m
        )
        state = DodgeState.DODGED if dodged else DodgeState.NOT_DODGED
        return DodgeResult(state=state, clearance_m=clearance, displacement_m=displacement)
