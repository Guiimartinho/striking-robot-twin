"""Shared test fixtures and plant-agnostic test doubles.

These doubles satisfy the HAL Protocols without any physics, so the safety suite
and the contract suite run in pure NumPy, with no mujoco required. That is the
Phase 0 design promise: the safety math is testable everywhere, every time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_twin.core.types import (
    NUM_KEYPOINTS,
    JointState,
    Keypoint,
    StrikeCommand,
    StrikeType,
    TraineePose,
    Vec3,
)
from robot_twin.safety.arbiter import RobotGeometry, SafetyArbiter, SafetyConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = REPO_ROOT / "models" / "scene.xml"

# Canonical ground-truth keypoints, matching models/trainee/humanoid.xml. Order
# follows the Keypoint enum so positions[int(kp)] is that keypoint.
_KEYPOINTS: dict[Keypoint, tuple[float, float, float]] = {
    Keypoint.HEAD: (0.0, 0.75, 1.62),
    Keypoint.NECK: (0.0, 0.75, 1.45),
    Keypoint.CHEST: (0.0, 0.72, 1.25),
    Keypoint.L_SHOULDER: (0.18, 0.73, 1.40),
    Keypoint.R_SHOULDER: (-0.18, 0.73, 1.40),
    Keypoint.L_ELBOW: (0.22, 0.78, 1.15),
    Keypoint.R_ELBOW: (-0.22, 0.78, 1.15),
    Keypoint.L_HAND: (0.12, 0.62, 1.50),
    Keypoint.R_HAND: (-0.12, 0.62, 1.50),
}


def make_pose(timestamp_s: float = 0.0) -> TraineePose:
    """A full-confidence ground-truth pose matching the MJCF mannequin."""
    positions = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
    for kp, xyz in _KEYPOINTS.items():
        positions[int(kp)] = xyz
    confidence = np.ones(NUM_KEYPOINTS, dtype=np.float64)
    return TraineePose(positions=positions, confidence=confidence, timestamp_s=timestamp_s)


class FixedObserver:
    """An ITraineeObserver that returns a fixed pose and latency.

    The base the fault wrappers degrade. No physics, no mujoco.
    """

    def __init__(self, pose: TraineePose, latency_s: float = 0.0) -> None:
        self._pose = pose
        self._latency = latency_s

    def get_pose(self) -> TraineePose:
        return self._pose

    def latency_s(self) -> float:
        return self._latency


class FakePlant:
    """An in-memory IRobotPlant for contract tests, no physics.

    Records the last command, advances a clock on step, and exposes a settable
    force so the force-cap path can be exercised without a simulator.
    """

    def __init__(self, n_joints: int = 2) -> None:
        self._n = n_joints
        self._position = np.zeros(n_joints, dtype=np.float64)
        self._velocity = np.zeros(n_joints, dtype=np.float64)
        self._force = np.zeros(n_joints, dtype=np.float64)
        self.last_command: StrikeCommand | None = None
        self.clock_s = 0.0
        self.stopped = False

    def command_strike(self, cmd: StrikeCommand) -> None:
        self.last_command = cmd

    def read_joint_state(self) -> JointState:
        return JointState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            force=self._force.copy(),
        )

    def step(self, dt: float) -> None:
        self.clock_s += dt

    def emergency_stop(self) -> None:
        self._velocity[:] = 0.0
        self.stopped = True

    def set_force(self, force: np.ndarray) -> None:
        self._force = np.asarray(force, dtype=np.float64)


@pytest.fixture
def safety_config() -> SafetyConfig:
    return SafetyConfig()


@pytest.fixture
def geometry() -> RobotGeometry:
    return RobotGeometry(arm_bases=(Vec3(0.0, 0.0, 1.0),), reach_max_m=0.9)


@pytest.fixture
def arbiter(safety_config: SafetyConfig, geometry: RobotGeometry) -> SafetyArbiter:
    return SafetyArbiter(config=safety_config, geometry=geometry)


@pytest.fixture
def pose() -> TraineePose:
    return make_pose()


@pytest.fixture
def observer(pose: TraineePose) -> FixedObserver:
    return FixedObserver(pose, latency_s=0.0)


@pytest.fixture
def safe_command() -> StrikeCommand:
    """A low body shot well clear of the head and neck keep-out spheres."""
    return StrikeCommand(
        strike_type=StrikeType.JAB,
        target=Vec3(0.0, 0.7, 1.0),
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )


@pytest.fixture
def head_command() -> StrikeCommand:
    """A high jab toward the jaw: within reach, but its line crosses the neck

    keep-out sphere, so it must be vetoed for KEEPOUT (not reach).
    """
    return StrikeCommand(
        strike_type=StrikeType.JAB,
        target=Vec3(0.0, 0.75, 1.45),
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )
