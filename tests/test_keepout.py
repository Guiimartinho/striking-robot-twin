"""Unit tests for the keep-out math and the basic arbiter decisions.

Pure NumPy: no mujoco. These pin the geometry and the radius formula so a later
change cannot silently weaken the protected volume.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_twin.core.result import ErrorCode
from robot_twin.core.types import StrikeCommand, StrikeType, Vec3
from robot_twin.safety.arbiter import (
    SafetyArbiter,
    SafetyConfig,
    point_to_segment_distance,
)


class TestKeepoutRadius:
    def test_formula_exact(self) -> None:
        cfg = SafetyConfig(
            tracking_error_m=0.05,
            head_v_max_mps=3.0,
            actuator_stop_s=0.05,
            margin_m=0.10,
        )
        arb = SafetyArbiter(config=cfg, geometry=_unit_geometry())
        # R = 0.05 + (0.10 + 0.05) * 3.0 + 0.10 = 0.60 at observer latency 0.10.
        assert arb.compute_keepout_radius(0.10) == pytest.approx(0.60)

    def test_monotonic_non_decreasing_in_latency(self) -> None:
        arb = SafetyArbiter(config=SafetyConfig(), geometry=_unit_geometry())
        latencies = np.linspace(0.0, 0.2, 21)
        radii = [arb.compute_keepout_radius(float(l)) for l in latencies]
        assert all(b >= a for a, b in zip(radii, radii[1:]))

    def test_zero_velocity_bound_removes_latency_term(self) -> None:
        cfg = SafetyConfig(head_v_max_mps=0.0, tracking_error_m=0.05, margin_m=0.10)
        arb = SafetyArbiter(config=cfg, geometry=_unit_geometry())
        # With no head motion the radius is just tracking error + margin.
        assert arb.compute_keepout_radius(0.5) == pytest.approx(0.15)


class TestPointToSegmentDistance:
    def test_point_on_segment_is_zero(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        pts = np.array([[0.5, 0.0, 0.0]])
        assert point_to_segment_distance(p0, p1, pts)[0] == pytest.approx(0.0)

    def test_perpendicular_distance(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        pts = np.array([[0.5, 0.3, 0.0]])
        assert point_to_segment_distance(p0, p1, pts)[0] == pytest.approx(0.3)

    def test_beyond_endpoint_clamps_to_endpoint(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        pts = np.array([[2.0, 0.0, 0.0]])
        # Closest point on the finite segment is the endpoint p1, distance 1.0.
        assert point_to_segment_distance(p0, p1, pts)[0] == pytest.approx(1.0)

    def test_degenerate_segment_is_point_distance(self) -> None:
        p0 = np.array([1.0, 1.0, 1.0])
        pts = np.array([[1.0, 1.0, 2.0]])
        assert point_to_segment_distance(p0, p0, pts)[0] == pytest.approx(1.0)

    def test_vectorised_over_multiple_points(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 1.0, 0.0])
        pts = np.array([[0.1, 0.5, 0.0], [0.0, 2.0, 0.0]])
        out = point_to_segment_distance(p0, p1, pts)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(0.1)
        assert out[1] == pytest.approx(1.0)


class TestArbiterBasicDecisions:
    def test_safe_command_allowed(self, arbiter, pose, safe_command) -> None:
        result = arbiter.evaluate(safe_command, pose, observer_latency_s=0.0)
        assert result.is_ok
        assert result.code is ErrorCode.OK

    def test_head_target_vetoed(self, arbiter, pose, head_command) -> None:
        result = arbiter.evaluate(head_command, pose, observer_latency_s=0.0)
        assert result.is_err
        assert result.code is ErrorCode.KEEPOUT_VIOLATION

    def test_reach_exceeded(self, arbiter, pose) -> None:
        far = StrikeCommand(
            strike_type=StrikeType.CROSS,
            target=Vec3(0.0, 2.0, 1.3),  # 2.0 m out, beyond the 0.9 m reach
            speed_mps=2.0,
            telegraph_s=0.1,
            arm_id=0,
        )
        result = arbiter.evaluate(far, pose, observer_latency_s=0.0)
        assert result.code is ErrorCode.REACH_EXCEEDED

    def test_unknown_arm_id(self, arbiter, pose, safe_command) -> None:
        bad = StrikeCommand(
            strike_type=safe_command.strike_type,
            target=safe_command.target,
            speed_mps=safe_command.speed_mps,
            telegraph_s=safe_command.telegraph_s,
            arm_id=5,  # only arm 0 exists
        )
        result = arbiter.evaluate(bad, pose, observer_latency_s=0.0)
        assert result.code is ErrorCode.INVALID_COMMAND

    def test_force_cap_exceeded(self, arbiter, pose, safe_command) -> None:
        over_cap = np.array([0.0, 200.0])  # 200 N > 80 N default cap
        result = arbiter.evaluate(
            safe_command, pose, observer_latency_s=0.0, joint_state_force=over_cap
        )
        assert result.code is ErrorCode.FORCE_CAP_EXCEEDED

    def test_higher_latency_inflates_radius_and_can_reject(self, geometry, pose) -> None:
        # A target safe at zero latency becomes unsafe once the inflated radius
        # grows past its clearance. This is the core latency-margin behaviour.
        cmd = StrikeCommand(
            strike_type=StrikeType.JAB,
            target=Vec3(0.0, 0.7, 1.0),
            speed_mps=2.0,
            telegraph_s=0.2,
            arm_id=0,
        )
        arb = SafetyArbiter(config=SafetyConfig(), geometry=geometry)
        assert arb.evaluate(cmd, pose, observer_latency_s=0.0).is_ok
        # Push latency just under the ceiling so it inflates R without tripping
        # the separate latency-ceiling check.
        assert arb.evaluate(cmd, pose, observer_latency_s=0.19).is_err


def _unit_geometry():
    from robot_twin.safety.arbiter import RobotGeometry

    return RobotGeometry(arm_bases=(Vec3(0.0, 0.0, 0.0),), reach_max_m=10.0)
