"""SafetyArbiter: the independent veto on every strike command.

This is not one module among many. It is the design constraint that organises
the architecture, because this machine hits a human. The arbiter decides, per
cycle, whether a commanded strike may proceed, and returns the reason on a
``Result`` (exception-free, like the firmware it stands in for).

The central idea is the latency-inflated keep-out volume. Between "we estimated
the head" and "the actuator actually stopped", the head moves. So the protected
sphere around the head is not static; it grows with the worst-case distance the
head could have travelled in that window:

    R_keepout = tracking_error + latency_total * head_v_max + margin

where ``latency_total = observer_latency + actuator_stop_time``. The observer
latency is read live every cycle (it rises under load), and the actuator stop
time is a fixed mechanical property of the SEA drive.

Checks, in order, all fail-safe (any doubt rejects):
  1. command validity (known arm)
  2. latency ceiling (pose too stale to trust at all)
  3. keypoint availability (a dropped protected keypoint is unknown, so reject)
  4. reach (target beyond the mechanical end-stop)
  5. keep-out (commanded trajectory crosses an inflated protected sphere)
  6. force cap (sensed SEA force over the energy budget)

Trajectory model: Phase 0 approximates the commanded motion as the straight
segment from the arm base to the target and checks that segment against each
protected sphere. A richer swept-volume model (hooks curve) is future work; the
straight segment plus the inflated radius is the documented Phase 0 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_twin.core.geometry import point_to_segment_distance
from robot_twin.core.result import ErrorCode, Result
from robot_twin.core.types import (
    FloatArray,
    Keypoint,
    StrikeCommand,
    TraineePose,
    Vec3,
)

# Re-exported for backward compatibility: callers and tests historically import
# point_to_segment_distance from this module. The implementation now lives in
# core.geometry so domain code can share it without depending on safety.
__all__ = ["RobotGeometry", "SafetyArbiter", "SafetyConfig", "point_to_segment_distance"]


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Tunable safety limits. Conservative defaults; tighten, never loosen.

    Units are SI. ``head_v_max_mps`` is a worst-case bound on head speed, not a
    measurement: bounding it (rather than tracking it) keeps the keep-out margin
    safe even when pose velocity is noisy or briefly wrong.
    """

    tracking_error_m: float = 0.05  # pose estimator position error budget
    head_v_max_mps: float = 3.0  # worst-case head speed (a fast slip/duck)
    actuator_stop_s: float = 0.05  # time for the SEA drive to halt after a veto
    margin_m: float = 0.10  # fixed safety pad on top of the computed radius
    max_latency_s: float = 0.20  # hard ceiling: above this, pose is untrusted
    min_confidence: float = 0.5  # below this a keypoint counts as dropped out
    force_cap_n: float = 80.0  # sensed SEA force ceiling (energy budget proxy)
    protected_keypoints: tuple[Keypoint, ...] = (Keypoint.HEAD, Keypoint.NECK)

    def __post_init__(self) -> None:
        for name in (
            "tracking_error_m",
            "head_v_max_mps",
            "actuator_stop_s",
            "margin_m",
            "max_latency_s",
            "force_cap_n",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if not self.protected_keypoints:
            raise ValueError("at least one protected keypoint is required")


@dataclass(frozen=True, slots=True)
class RobotGeometry:
    """Fixed mechanical layout the arbiter reasons about.

    ``arm_bases`` is indexed by ``arm_id``: each striker arm originates at a
    fixed point on the frame. ``reach_max_m`` is the mechanical end-stop: the
    actuator physically cannot extend a target beyond this from its base,
    independent of any command.
    """

    arm_bases: tuple[Vec3, ...]
    reach_max_m: float

    def __post_init__(self) -> None:
        if not self.arm_bases:
            raise ValueError("at least one arm base is required")
        if self.reach_max_m <= 0.0:
            raise ValueError(f"reach_max_m must be > 0, got {self.reach_max_m}")

    @property
    def n_arms(self) -> int:
        return len(self.arm_bases)


@dataclass
class SafetyArbiter:
    """Vetoes commands that violate keep-out, reach, latency or force limits.

    Holds no mutable physics state: it is a pure decision function over the data
    handed to it each cycle (command, pose, observed latency, joint state). That
    purity is what lets it be exhaustively fault-injected in tests and, later,
    ported to the STM32 as the primary safety channel.
    """

    config: SafetyConfig
    geometry: RobotGeometry
    # Cached array views, computed once, since geometry is fixed for a run.
    _arm_base_arrays: tuple[FloatArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._arm_base_arrays = tuple(b.as_array() for b in self.geometry.arm_bases)

    def compute_keepout_radius(self, observer_latency_s: float) -> float:
        """R_keepout for the given observed latency.

        Monotonically non-decreasing in latency: the slower we see, the wider we
        keep clear. ``latency_total`` adds the fixed actuator stop time to the
        live observer latency.
        """
        latency_total = observer_latency_s + self.config.actuator_stop_s
        return (
            self.config.tracking_error_m
            + latency_total * self.config.head_v_max_mps
            + self.config.margin_m
        )

    def evaluate(
        self,
        cmd: StrikeCommand,
        pose: TraineePose,
        observer_latency_s: float,
        joint_state_force: FloatArray | None = None,
    ) -> Result[StrikeCommand]:
        """Approve or veto a strike command. Never raises.

        Args:
            cmd: The proposed strike.
            pose: Latest trainee keypoints with confidence.
            observer_latency_s: Live pipeline latency feeding ``R_keepout``.
            joint_state_force: Optional sensed SEA forces for the force-cap
                check. When absent the force check is skipped (command-time
                evaluation, before any contact).

        Returns:
            ``Result.ok(cmd)`` if every check passes, otherwise ``Result.err``
            with the specific ``ErrorCode`` of the first violated limit.
        """
        # 1. Command validity: the arm must exist on this frame.
        if not 0 <= cmd.arm_id < self.geometry.n_arms:
            return Result.err(ErrorCode.INVALID_COMMAND, f"unknown arm_id {cmd.arm_id}")

        # 2. Latency ceiling: past this the pose is too stale to trust at all,
        # so no command can be proven safe. Reject before using the pose.
        if observer_latency_s > self.config.max_latency_s:
            return Result.err(
                ErrorCode.LATENCY_EXCEEDED,
                f"latency {observer_latency_s:.3f}s > ceiling {self.config.max_latency_s:.3f}s",
            )

        base = self._arm_base_arrays[cmd.arm_id]
        target = cmd.target.as_array()

        # 3. Reach: the mechanical end-stop. A target beyond it cannot be
        # reached, and commanding it implies a bad plan; reject.
        reach = float(np.linalg.norm(target - base))
        if reach > self.geometry.reach_max_m:
            return Result.err(
                ErrorCode.REACH_EXCEEDED,
                f"reach {reach:.3f}m > max {self.geometry.reach_max_m:.3f}m",
            )

        # 4 + 5. Keep-out, fail-safe on dropout. Gather protected keypoints; a
        # missing/low-confidence/non-finite one is unknown, so we reject rather
        # than assume it is clear.
        radius = self.compute_keepout_radius(observer_latency_s)
        protected_idx = [int(kp) for kp in self.config.protected_keypoints]
        conf = pose.confidence[protected_idx]
        positions = pose.positions[protected_idx]  # (P, 3)
        if not np.all(np.isfinite(positions)) or np.any(conf < self.config.min_confidence):
            return Result.err(
                ErrorCode.KEYPOINT_DROPOUT,
                "a protected keypoint is missing or low-confidence; failing safe",
            )

        distances = point_to_segment_distance(base, target, positions)  # (P,)
        if np.any(distances < radius):
            closest = float(np.min(distances))
            return Result.err(
                ErrorCode.KEEPOUT_VIOLATION,
                f"trajectory passes {closest:.3f}m from a protected keypoint, "
                f"inside R_keepout {radius:.3f}m",
            )

        # 6. Force cap: sensed SEA force over the energy budget. Skipped when no
        # force is supplied (pre-contact command-time check).
        if joint_state_force is not None and joint_state_force.size > 0:
            peak = float(np.max(np.abs(joint_state_force)))
            if peak > self.config.force_cap_n:
                return Result.err(
                    ErrorCode.FORCE_CAP_EXCEEDED,
                    f"sensed force {peak:.1f}N > cap {self.config.force_cap_n:.1f}N",
                )

        return Result.ok(cmd)
