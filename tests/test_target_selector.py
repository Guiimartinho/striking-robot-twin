"""Tests for TargetSelector: targets sit on the arm plane and pass the arbiter."""

from __future__ import annotations

import numpy as np

from conftest import make_pose
from robot_twin.core.types import Keypoint, TraineePose, Vec3
from robot_twin.domain.target_selector import TargetConfig, TargetSelector

_ARM_BASE = Vec3(0.0, 0.0, 1.0)


class TestTargetSelector:
    def test_target_on_arm_plane(self) -> None:
        target = TargetSelector().select(make_pose(), _ARM_BASE)
        # The 1-DOF arm cannot aim in x or z: the target must lie on its plane.
        assert target.x == _ARM_BASE.x
        assert target.z == _ARM_BASE.z

    def test_depth_tracks_chest(self) -> None:
        cfg = TargetConfig(body_standoff_m=0.08)
        target = TargetSelector(cfg).select(make_pose(), _ARM_BASE)
        chest_y = make_pose().keypoint(Keypoint.CHEST).y
        assert target.y == chest_y - 0.08

    def test_selected_target_passes_arbiter(self, arbiter) -> None:
        from robot_twin.core.types import StrikeCommand, StrikeType

        target = TargetSelector().select(make_pose(), _ARM_BASE)
        cmd = StrikeCommand(StrikeType.JAB, target, 1.5, 0.4, 0)
        result = arbiter.evaluate(cmd, make_pose(), observer_latency_s=0.02)
        assert result.is_ok

    def test_falls_back_when_chest_dropped(self) -> None:
        pose = make_pose()
        positions = pose.positions.copy()
        confidence = pose.confidence.copy()
        positions[int(Keypoint.CHEST)] = np.nan
        confidence[int(Keypoint.CHEST)] = 0.0
        dropped = TraineePose(positions=positions, confidence=confidence, timestamp_s=0.0)
        target = TargetSelector().select(dropped, _ARM_BASE)
        # Fallback depth, finite and on the arm plane, not a NaN from the chest.
        assert np.isfinite(target.y)
        assert target.x == _ARM_BASE.x and target.z == _ARM_BASE.z
