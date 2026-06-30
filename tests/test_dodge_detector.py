"""Phase 1 gate (dodge): reliable DODGED/NOT_DODGED under injected noise.

A clear slip and a clear no-move keep their verdict across hundreds of noisy pose
draws, and a dropped head keypoint yields UNKNOWN.
"""

from __future__ import annotations

import numpy as np

from conftest import FixedObserver, make_pose
from robot_twin.core.types import Keypoint, TraineePose, Vec3
from robot_twin.domain.dodge_detector import DodgeDetector, DodgeState
from robot_twin.safety.fault_injection import DropoutFault, NoiseFault

# A head-aimed strike: origin at the arm base, target at the head keypoint.
_ORIGIN = Vec3(0.0, 0.0, 1.3)
_TARGET = Vec3(0.0, 0.75, 1.62)


def _with_head_at(pose: TraineePose, x: float, y: float, z: float) -> TraineePose:
    positions = pose.positions.copy()
    positions[int(Keypoint.HEAD)] = (x, y, z)
    return TraineePose(
        positions=positions, confidence=pose.confidence.copy(), timestamp_s=pose.timestamp_s
    )


class TestDodgeClean:
    def test_lateral_slip_is_dodged(self) -> None:
        launch = make_pose()  # head on the strike line endpoint
        now = _with_head_at(make_pose(), 0.30, 0.75, 1.62)  # slipped 0.30 m sideways
        result = DodgeDetector().evaluate(_ORIGIN, _TARGET, launch, now)
        assert result.state is DodgeState.DODGED
        assert result.clearance_m >= 0.15
        assert result.displacement_m >= 0.10

    def test_no_movement_is_not_dodged(self) -> None:
        launch = make_pose()
        now = make_pose()  # head did not move; would be tagged (non-impacting)
        result = DodgeDetector().evaluate(_ORIGIN, _TARGET, launch, now)
        assert result.state is DodgeState.NOT_DODGED

    def test_movement_along_line_is_not_dodged(self) -> None:
        # The head ducks a long way but stays on the strike line (no lateral
        # clearance), so it is not a dodge even though it moved a lot.
        launch = make_pose()
        now = _with_head_at(make_pose(), 0.0, 0.375, 1.46)  # midpoint of the line
        result = DodgeDetector().evaluate(_ORIGIN, _TARGET, launch, now)
        assert result.state is DodgeState.NOT_DODGED
        assert result.displacement_m >= 0.10  # it did move
        assert result.clearance_m < 0.15  # but stayed on the line


class TestDodgeDropout:
    def test_head_dropout_now_is_unknown(self) -> None:
        launch = make_pose()
        now = DropoutFault(FixedObserver(make_pose()), dropped=(Keypoint.HEAD,)).get_pose()
        result = DodgeDetector().evaluate(_ORIGIN, _TARGET, launch, now)
        assert result.state is DodgeState.UNKNOWN


class TestDodgeNoiseRobustness:
    """The Phase 1 gate: the verdict holds under injected pose noise."""

    def test_slip_holds_under_noise(self) -> None:
        rng = np.random.default_rng(21)
        launch_obs = NoiseFault(FixedObserver(make_pose()), sigma_m=0.02, rng=rng)
        now_obs = NoiseFault(
            FixedObserver(_with_head_at(make_pose(), 0.30, 0.75, 1.62)), sigma_m=0.02, rng=rng
        )
        detector = DodgeDetector()
        correct = sum(
            detector.evaluate(_ORIGIN, _TARGET, launch_obs.get_pose(), now_obs.get_pose()).state
            is DodgeState.DODGED
            for _ in range(1000)
        )
        assert correct >= 990

    def test_no_move_holds_under_noise(self) -> None:
        rng = np.random.default_rng(22)
        launch_obs = NoiseFault(FixedObserver(make_pose()), sigma_m=0.02, rng=rng)
        now_obs = NoiseFault(FixedObserver(make_pose()), sigma_m=0.02, rng=rng)
        detector = DodgeDetector()
        correct = sum(
            detector.evaluate(_ORIGIN, _TARGET, launch_obs.get_pose(), now_obs.get_pose()).state
            is DodgeState.NOT_DODGED
            for _ in range(1000)
        )
        assert correct >= 990
