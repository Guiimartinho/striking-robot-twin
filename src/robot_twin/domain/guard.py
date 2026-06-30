"""GuardDetector: assess the trainee's guard from keypoints. Phase 1.

A guard is "up" when both gloves sit near the head: not far below it, not flung
above it, and horizontally close to it (covering the face). The thresholds carry
a wide margin relative to expected pose jitter, so injected noise does not flip
the decision; that margin is the Phase 1 reliability gate.

Fail-safe on missing data: if the head or either hand is dropped or
low-confidence, the guard is reported UNKNOWN rather than guessed. Domain code is
not the safety hot path, so this returns a plain assessment value, not a Result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_twin.core.types import Keypoint, TraineePose


class GuardState(Enum):
    """Outcome of a guard assessment."""

    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"  # head or a hand not reliably observed


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """Thresholds for what counts as a guarding hand, in metres.

    Defaults leave a wide band around the canonical guard pose (hands ~0.12 m
    below and ~0.18 m horizontally from the head) so that pose noise on the order
    of a few centimetres cannot move a clear guard across the boundary.
    """

    max_drop_below_m: float = 0.30  # hand at most this far below the head
    max_raise_above_m: float = 0.20  # hand at most this far above the head
    max_horizontal_m: float = 0.30  # hand within this horizontal radius of head
    min_confidence: float = 0.5
    require_both_hands: bool = True  # full guard needs both gloves up


@dataclass(frozen=True, slots=True)
class GuardAssessment:
    """Result of a guard assessment, with the per-hand detail behind it."""

    state: GuardState
    left_guarding: bool
    right_guarding: bool


class GuardDetector:
    """Classifies a pose as guard UP / DOWN / UNKNOWN."""

    def __init__(self, config: GuardConfig | None = None) -> None:
        self._cfg = config or GuardConfig()

    def _hand_guards(self, head: np.ndarray, hand: np.ndarray) -> bool:
        """Whether a single hand is in a guarding position relative to the head."""
        dz = float(head[2] - hand[2])  # positive when the hand is below the head
        if dz > self._cfg.max_drop_below_m or dz < -self._cfg.max_raise_above_m:
            return False
        horizontal = float(np.hypot(hand[0] - head[0], hand[1] - head[1]))
        return horizontal <= self._cfg.max_horizontal_m

    def assess(self, pose: TraineePose) -> GuardAssessment:
        """Assess the guard from a single pose snapshot.

        Args:
            pose: The trainee keypoints with per-keypoint confidence.

        Returns:
            A GuardAssessment. UNKNOWN when the head or either hand is dropped or
            below the confidence floor, or when any needed keypoint is non-finite.
        """
        needed = (Keypoint.HEAD, Keypoint.L_HAND, Keypoint.R_HAND)
        idx = [int(kp) for kp in needed]
        if np.any(pose.confidence[idx] < self._cfg.min_confidence) or not np.all(
            np.isfinite(pose.positions[idx])
        ):
            return GuardAssessment(GuardState.UNKNOWN, left_guarding=False, right_guarding=False)

        head = pose.positions[int(Keypoint.HEAD)]
        left = self._hand_guards(head, pose.positions[int(Keypoint.L_HAND)])
        right = self._hand_guards(head, pose.positions[int(Keypoint.R_HAND)])

        if self._cfg.require_both_hands:
            up = left and right
        else:
            up = left or right
        return GuardAssessment(
            GuardState.UP if up else GuardState.DOWN,
            left_guarding=left,
            right_guarding=right,
        )
