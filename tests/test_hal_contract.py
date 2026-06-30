"""HAL contract: every plant satisfies the same IRobotPlant behaviour.

The sim plant and the in-memory fake are exercised through one parametrized
suite, the way the real STM32 plant will be once it exists. The mujoco cases skip
cleanly when the package or the models are absent, so the contract still runs in
a minimal environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SCENE_PATH, FakePlant
from robot_twin.core.types import JointState, StrikeCommand, StrikeType, Vec3
from robot_twin.hal.interfaces import IRobotPlant, ITraineeObserver


def _make_fake() -> IRobotPlant:
    return FakePlant()


def _make_mujoco() -> IRobotPlant:
    pytest.importorskip("mujoco")
    if not SCENE_PATH.exists():
        pytest.skip(f"scene model missing: {SCENE_PATH}")
    from robot_twin.hal.mujoco_plant import MujocoPlant

    return MujocoPlant(SCENE_PATH)


@pytest.fixture(params=["fake", pytest.param("mujoco", marks=pytest.mark.mujoco)])
def plant(request) -> IRobotPlant:
    return _make_fake() if request.param == "fake" else _make_mujoco()


def _a_command() -> StrikeCommand:
    return StrikeCommand(
        strike_type=StrikeType.JAB,
        target=Vec3(0.0, 0.7, 1.0),
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )


class TestPlantContract:
    def test_is_robot_plant(self, plant) -> None:
        assert isinstance(plant, IRobotPlant)

    def test_read_joint_state_shapes(self, plant) -> None:
        js = plant.read_joint_state()
        assert isinstance(js, JointState)
        assert js.n_joints >= 1
        assert js.position.shape == js.velocity.shape == js.force.shape

    def test_command_strike_accepts(self, plant) -> None:
        plant.command_strike(_a_command())  # must not raise

    def test_step_advances(self, plant) -> None:
        plant.command_strike(_a_command())
        plant.step(0.01)  # must not raise
        js = plant.read_joint_state()
        assert np.all(np.isfinite(js.position))

    def test_emergency_stop_is_idempotent(self, plant) -> None:
        plant.emergency_stop()
        plant.emergency_stop()  # safe to call repeatedly
        js = plant.read_joint_state()
        assert np.allclose(js.velocity, 0.0)


@pytest.mark.mujoco
class TestSimObserverContract:
    def test_is_trainee_observer(self) -> None:
        pytest.importorskip("mujoco")
        if not SCENE_PATH.exists():
            pytest.skip("scene model missing")
        from robot_twin.hal.mujoco_plant import MujocoPlant
        from robot_twin.hal.sim_observer import SimGTObserver

        plant = MujocoPlant(SCENE_PATH)
        observer = SimGTObserver(plant.model, plant.data)
        assert isinstance(observer, ITraineeObserver)
        pose = observer.get_pose()
        assert np.all(np.isfinite(pose.positions))
        assert np.all(pose.confidence == 1.0)
        assert observer.latency_s() >= 0.0


class TestRealPlantStub:
    """The real plant is an honest stub: it must raise, never silently no-op."""

    def test_real_plant_raises(self) -> None:
        from robot_twin.hal.real_plant import RealPlant

        plant = RealPlant()
        with pytest.raises(NotImplementedError):
            plant.command_strike(_a_command())
        with pytest.raises(NotImplementedError):
            plant.read_joint_state()

    def test_camera_observer_raises(self) -> None:
        from robot_twin.hal.real_plant import CameraPoseObserver

        observer = CameraPoseObserver()
        with pytest.raises(NotImplementedError):
            observer.get_pose()
