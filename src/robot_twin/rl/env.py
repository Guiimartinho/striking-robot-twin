"""Minimal Gymnasium env wrapping the HAL plus the SafetyArbiter.

The env is HAL-clean: it depends only on the ``IRobotPlant`` / ``ITraineeObserver``
Protocols and the arbiter, never on mujoco. That is the whole point: a policy
trained here runs unchanged on the real robot, because it only ever touched the
interfaces. The concrete sim wiring lives in ``make_sim_env``, which is the only
function that imports a concrete plant.

Every action is routed through the SafetyArbiter before it can reach the plant.
A vetoed strike is simply not executed and is penalised via the reward, so the
policy is shaped to respect safety while the arbiter stays the hard guarantee.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from robot_twin.core.types import (
    NUM_KEYPOINTS,
    JointState,
    StrikeCommand,
    StrikeType,
    TraineePose,
    Vec3,
)
from robot_twin.hal.interfaces import IRobotPlant, ITraineeObserver
from robot_twin.rl.reward import compute_reward
from robot_twin.safety.arbiter import SafetyArbiter


class BoxingDrillEnv(gym.Env):
    """A one-strike-per-step drill env over a generic plant and observer."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant: IRobotPlant,
        observer: ITraineeObserver,
        arbiter: SafetyArbiter,
        *,
        target_center: Vec3,
        target_half_extent: Vec3,
        arm_id: int = 0,
        control_dt: float = 0.02,
        max_steps: int = 200,
        speed_range: tuple[float, float] = (0.5, 3.0),
    ) -> None:
        super().__init__()
        self._plant = plant
        self._observer = observer
        self._arbiter = arbiter
        self._arm_id = arm_id
        self._control_dt = control_dt
        self._max_steps = max_steps
        self._speed_lo, self._speed_hi = speed_range
        self._target_center = target_center
        self._target_half_extent = target_half_extent
        self._step_count = 0

        n_joints = self._plant.read_joint_state().n_joints
        obs_dim = NUM_KEYPOINTS * 3 + 3 * n_joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: 3 normalised target offsets + 1 normalised speed, all in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _build_obs(self, pose: TraineePose, joint_state: JointState) -> npt.NDArray[np.float32]:
        """Flatten pose keypoints and joint state into one observation vector."""
        return np.concatenate(
            [
                pose.positions.reshape(-1),
                joint_state.position,
                joint_state.velocity,
                joint_state.force,
            ]
        ).astype(np.float32)

    def _action_to_command(self, action: npt.NDArray[np.float32]) -> StrikeCommand:
        """Map a normalised action to a concrete StrikeCommand."""
        offset = Vec3(
            float(action[0]) * self._target_half_extent.x,
            float(action[1]) * self._target_half_extent.y,
            float(action[2]) * self._target_half_extent.z,
        )
        target = self._target_center + offset
        speed = self._speed_lo + (float(action[3]) + 1.0) * 0.5 * (self._speed_hi - self._speed_lo)
        return StrikeCommand(
            strike_type=StrikeType.JAB,
            target=target,
            speed_mps=speed,
            telegraph_s=0.2,
            arm_id=self._arm_id,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        """Begin a new episode.

        Phase 0 does not reset the underlying plant state (the Protocol has no
        reset; the sim plant carries its physics state). That is acceptable for
        the static-trainee Phase 0 drill and is a documented follow-up.
        """
        super().reset(seed=seed)
        self._step_count = 0
        obs = self._build_obs(self._observer.get_pose(), self._plant.read_joint_state())
        return obs, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        """Evaluate one strike: safety-gate, maybe execute, advance, score."""
        cmd = self._action_to_command(np.asarray(action, dtype=np.float32))
        pose = self._observer.get_pose()
        latency = self._observer.latency_s()
        joint_state = self._plant.read_joint_state()

        result = self._arbiter.evaluate(cmd, pose, latency, joint_state.force)
        if result.is_ok:
            self._plant.command_strike(cmd)
        self._plant.step(self._control_dt)

        self._step_count += 1
        new_pose = self._observer.get_pose()
        new_joint_state = self._plant.read_joint_state()
        obs = self._build_obs(new_pose, new_joint_state)

        # Phase 0 uses a coarse target-error proxy of 0.0 on allowed strikes; the
        # cartesian glove position is not exposed through the HAL, and the reward
        # is a placeholder until Phase 3 scores against video-mined timing.
        reward = compute_reward(result.code, target_error_m=0.0)
        truncated = self._step_count >= self._max_steps
        info = {"veto": result.code.value, "veto_message": result.message}
        return obs, reward, False, truncated, info


def make_sim_env(max_steps: int = 200) -> BoxingDrillEnv:
    """Build a fully wired sim env from the default app config.

    This is the only function here that imports a concrete plant, keeping the env
    class itself HAL-clean. Imports are local so that importing this module does
    not require mujoco unless a sim env is actually requested.
    """
    from robot_twin.config.app_config import AppConfig
    from robot_twin.hal.mujoco_plant import MujocoPlant
    from robot_twin.hal.sim_observer import SimGTObserver

    cfg = AppConfig.load()
    plant = MujocoPlant(cfg.model_path)
    observer = SimGTObserver(plant.model, plant.data, nominal_latency_s=cfg.nominal_latency_s)
    arbiter = SafetyArbiter(
        config=cfg.safety.to_safety_config(),
        geometry=cfg.geometry.to_robot_geometry(),
    )
    # Aim low at the trainee body (around the solar plexus); the head and neck
    # keep-out spheres make higher targets reject, which is the safe default.
    return BoxingDrillEnv(
        plant,
        observer,
        arbiter,
        target_center=Vec3(0.0, 0.7, 1.0),
        target_half_extent=Vec3(0.15, 0.04, 0.05),
        control_dt=cfg.control_dt,
        max_steps=max_steps,
    )
