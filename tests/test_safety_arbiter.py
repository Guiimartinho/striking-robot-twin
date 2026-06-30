"""Phase 0 gate: the SafetyArbiter rejects 100% of unsafe commands under faults.

Two layers of test:
  1. Explicit scenarios for each fault family (latency, dropout, lunge, noise),
     so a failure points straight at the cause.
  2. A randomized sweep over commands x fault stacks, checked against an
     independent oracle. The hard assertion is the safety direction: every
     command the oracle deems a violation is rejected by the arbiter. Not one
     unsafe command may slip through.

All pure NumPy. No mujoco, no camera: the fault harness is the adversary.
"""

from __future__ import annotations

import numpy as np

from conftest import FixedObserver, make_pose
from robot_twin.core.result import ErrorCode
from robot_twin.core.types import (
    Keypoint,
    StrikeCommand,
    StrikeType,
    TraineePose,
    Vec3,
)
from robot_twin.safety.arbiter import (
    RobotGeometry,
    SafetyConfig,
    point_to_segment_distance,
)
from robot_twin.safety.fault_injection import (
    DropoutFault,
    LatencyFault,
    LungeFault,
    NoiseFault,
)


# --------------------------------------------------------------------------- #
# Explicit per-fault scenarios.
# --------------------------------------------------------------------------- #
class TestLatencyFault:
    def test_over_ceiling_rejects_any_command(self, arbiter, observer, safe_command) -> None:
        faulted = LatencyFault(observer, extra_latency_s=0.5)  # well past the ceiling
        result = arbiter.evaluate(safe_command, faulted.get_pose(), faulted.latency_s())
        assert result.code is ErrorCode.LATENCY_EXCEEDED

    def test_moderate_latency_inflates_and_rejects(self, arbiter, observer) -> None:
        # A target safe at zero latency, rejected once latency inflates R but
        # stays under the ceiling (so it is keep-out, not the latency ceiling).
        cmd = _jab(Vec3(0.0, 0.7, 1.0))
        assert arbiter.evaluate(cmd, observer.get_pose(), 0.0).is_ok
        faulted = LatencyFault(observer, extra_latency_s=0.19)
        result = arbiter.evaluate(cmd, faulted.get_pose(), faulted.latency_s())
        assert result.code is ErrorCode.KEEPOUT_VIOLATION


class TestDropoutFault:
    def test_head_dropout_fails_safe(self, arbiter, observer, safe_command) -> None:
        faulted = DropoutFault(observer, dropped=(Keypoint.HEAD,))
        result = arbiter.evaluate(safe_command, faulted.get_pose(), faulted.latency_s())
        assert result.code is ErrorCode.KEYPOINT_DROPOUT

    def test_neck_dropout_fails_safe(self, arbiter, observer, safe_command) -> None:
        faulted = DropoutFault(observer, dropped=(Keypoint.NECK,))
        result = arbiter.evaluate(safe_command, faulted.get_pose(), faulted.latency_s())
        assert result.code is ErrorCode.KEYPOINT_DROPOUT

    def test_unprotected_dropout_does_not_block(self, arbiter, observer, safe_command) -> None:
        # Dropping a hand (not protected) must not, by itself, veto a safe shot.
        faulted = DropoutFault(observer, dropped=(Keypoint.L_HAND,))
        result = arbiter.evaluate(safe_command, faulted.get_pose(), faulted.latency_s())
        assert result.is_ok


class TestLungeFault:
    def test_lunge_into_strike_rejects(self, arbiter, observer) -> None:
        # The trainee drives head and neck down/forward onto a target that was
        # clear a moment ago. The arbiter must catch it via the inflated sphere.
        cmd = _jab(Vec3(0.0, 0.7, 1.0))
        assert arbiter.evaluate(cmd, observer.get_pose(), 0.0).is_ok
        lunge = LungeFault(observer, delta=Vec3(0.0, -0.05, -0.45))
        result = arbiter.evaluate(cmd, lunge.get_pose(), lunge.latency_s())
        assert result.code is ErrorCode.KEEPOUT_VIOLATION


class TestNoiseFault:
    def test_noise_never_lets_unsafe_through(self, arbiter) -> None:
        # Under jitter, a straight-at-the-head command must never be approved,
        # across many noise draws.
        rng = np.random.default_rng(7)
        head_cmd = _jab(Vec3(0.0, 0.75, 1.62))
        base = FixedObserver(make_pose(), latency_s=0.0)
        noisy = NoiseFault(base, sigma_m=0.02, rng=rng)
        for _ in range(200):
            result = arbiter.evaluate(head_cmd, noisy.get_pose(), noisy.latency_s())
            assert result.is_err


# --------------------------------------------------------------------------- #
# Randomized sweep against an independent oracle: the 100% gate.
# --------------------------------------------------------------------------- #
def _oracle_violates(
    cmd: StrikeCommand,
    obs_pose: TraineePose,
    latency_s: float,
    force: np.ndarray | None,
    config: SafetyConfig,
    geometry: RobotGeometry,
) -> bool:
    """Independent reference: does ANY safety rule fail for this command?

    Mirrors the arbiter's rules but is written separately so the test is a real
    cross-check, not a restatement of the implementation it guards.
    """
    if not 0 <= cmd.arm_id < geometry.n_arms:
        return True
    if latency_s > config.max_latency_s:
        return True
    base = geometry.arm_bases[cmd.arm_id].as_array()
    target = cmd.target.as_array()
    if float(np.linalg.norm(target - base)) > geometry.reach_max_m:
        return True

    protected = [int(kp) for kp in config.protected_keypoints]
    positions = obs_pose.positions[protected]
    confidence = obs_pose.confidence[protected]
    if not np.all(np.isfinite(positions)) or np.any(confidence < config.min_confidence):
        return True

    radius = (
        config.tracking_error_m
        + (latency_s + config.actuator_stop_s) * config.head_v_max_mps
        + config.margin_m
    )
    distances = point_to_segment_distance(base, target, positions)
    if np.any(distances < radius):
        return True

    if force is not None and force.size > 0 and float(np.max(np.abs(force))) > config.force_cap_n:
        return True
    return False


def test_sweep_rejects_all_violations(arbiter, safety_config, geometry) -> None:
    """The gate. Over a large fault-injected sample, no violation slips through."""
    rng = np.random.default_rng(20240629)
    base = FixedObserver(make_pose(), latency_s=0.02)

    n_trials = 4000
    n_violations = 0
    n_allowed = 0
    leaked = 0  # unsafe commands the arbiter wrongly approved: must stay 0

    for _ in range(n_trials):
        # Random target across and beyond the trainee envelope.
        target = Vec3(
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(0.3, 1.0)),
            float(rng.uniform(0.8, 1.7)),
        )
        cmd = _jab(target)

        # Random fault stack on top of the base observer.
        obs = base
        if rng.random() < 0.4:
            obs = LatencyFault(obs, extra_latency_s=float(rng.uniform(0.0, 0.4)))
        if rng.random() < 0.3:
            kp = Keypoint(int(rng.integers(0, 9)))
            obs = DropoutFault(obs, dropped=(kp,))
        if rng.random() < 0.3:
            obs = LungeFault(
                obs,
                delta=Vec3(
                    float(rng.uniform(-0.2, 0.2)),
                    float(rng.uniform(-0.2, 0.2)),
                    float(rng.uniform(-0.5, 0.1)),
                ),
            )
        if rng.random() < 0.3:
            obs = NoiseFault(obs, sigma_m=0.02, rng=rng)

        # Capture the observed pose and latency ONCE so the oracle and the
        # arbiter judge identical data (noise is redrawn per get_pose call).
        obs_pose = obs.get_pose()
        latency = obs.latency_s()
        force = None
        if rng.random() < 0.2:
            force = np.array([0.0, float(rng.uniform(0.0, 200.0))])

        violates = _oracle_violates(cmd, obs_pose, latency, force, safety_config, geometry)
        result = arbiter.evaluate(cmd, obs_pose, latency, joint_state_force=force)

        if violates:
            n_violations += 1
            if result.is_ok:
                leaked += 1
        else:
            n_allowed += 1
            # Over-rejection would be safe but useless; assert agreement so the
            # arbiter is neither too permissive nor trivially blocking everything.
            assert result.is_ok, f"safe command wrongly vetoed: {result.code} {result.message}"

    assert leaked == 0, f"{leaked} unsafe commands slipped through the arbiter"
    # Guard against a vacuous test: the sample must contain both outcomes.
    assert n_violations > 100, f"too few violations sampled ({n_violations})"
    assert n_allowed > 100, f"too few safe commands sampled ({n_allowed})"


def _jab(target: Vec3) -> StrikeCommand:
    return StrikeCommand(
        strike_type=StrikeType.JAB,
        target=target,
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )
