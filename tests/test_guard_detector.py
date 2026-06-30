"""Phase 1 gate (guard): reliable UP/DOWN classification under injected noise.

Clear guard-up and guard-down poses must keep their verdict across hundreds of
noisy pose draws, and a dropped keypoint must yield UNKNOWN, not a guess.
"""

from __future__ import annotations

import numpy as np

from conftest import FixedObserver, make_pose
from robot_twin.core.types import Keypoint, TraineePose
from robot_twin.domain.guard import GuardDetector, GuardState
from robot_twin.safety.fault_injection import DropoutFault, NoiseFault


def _with_hands_low(pose: TraineePose) -> TraineePose:
    """Same pose but with both gloves dropped to hip height (guard down)."""
    positions = pose.positions.copy()
    positions[int(Keypoint.L_HAND)] = (0.15, 0.70, 1.05)
    positions[int(Keypoint.R_HAND)] = (-0.15, 0.70, 1.05)
    return TraineePose(
        positions=positions, confidence=pose.confidence.copy(), timestamp_s=pose.timestamp_s
    )


class TestGuardClean:
    def test_canonical_pose_is_guard_up(self) -> None:
        # The conftest mannequin has both hands high, near the face.
        assessment = GuardDetector().assess(make_pose())
        assert assessment.state is GuardState.UP
        assert assessment.left_guarding and assessment.right_guarding

    def test_low_hands_is_guard_down(self) -> None:
        assessment = GuardDetector().assess(_with_hands_low(make_pose()))
        assert assessment.state is GuardState.DOWN

    def test_one_hand_down_is_guard_down_when_both_required(self) -> None:
        pose = make_pose()
        positions = pose.positions.copy()
        positions[int(Keypoint.R_HAND)] = (-0.15, 0.70, 1.05)  # drop only the right
        one_down = TraineePose(positions=positions, confidence=pose.confidence, timestamp_s=0.0)
        assessment = GuardDetector().assess(one_down)
        assert assessment.state is GuardState.DOWN
        assert assessment.left_guarding and not assessment.right_guarding


class TestGuardDropout:
    def test_head_dropout_is_unknown(self) -> None:
        faulted = DropoutFault(FixedObserver(make_pose()), dropped=(Keypoint.HEAD,))
        assert GuardDetector().assess(faulted.get_pose()).state is GuardState.UNKNOWN

    def test_hand_dropout_is_unknown(self) -> None:
        faulted = DropoutFault(FixedObserver(make_pose()), dropped=(Keypoint.L_HAND,))
        assert GuardDetector().assess(faulted.get_pose()).state is GuardState.UNKNOWN


class TestGuardNoiseRobustness:
    """The Phase 1 gate: the verdict holds under injected pose noise."""

    def test_guard_up_holds_under_noise(self) -> None:
        rng = np.random.default_rng(11)
        noisy = NoiseFault(FixedObserver(make_pose()), sigma_m=0.02, rng=rng)
        detector = GuardDetector()
        correct = sum(detector.assess(noisy.get_pose()).state is GuardState.UP for _ in range(1000))
        assert correct >= 990  # >= 99% stable

    def test_guard_down_holds_under_noise(self) -> None:
        rng = np.random.default_rng(12)
        noisy = NoiseFault(FixedObserver(_with_hands_low(make_pose())), sigma_m=0.02, rng=rng)
        detector = GuardDetector()
        correct = sum(
            detector.assess(noisy.get_pose()).state is GuardState.DOWN for _ in range(1000)
        )
        assert correct >= 990
