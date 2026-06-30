"""MujocoPlant: the sim-side IRobotPlant. The only place mujoco is imported

besides the observer. Everything above the HAL stays mujoco-free.

This is a minimal but faithful Phase 0 plant. A strike command is mapped to the
motor-stage position setpoint; the series spring then transmits force to the
link, and that force is read back from the spring's deflection (the SEA way).
Trajectory shaping from speed/telegraph is future work; Phase 0 drives the
position actuator to the mapped setpoint.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robot_twin.core.types import FloatArray, JointState, StrikeCommand

# Glove tip offset from the motor stage origin when fully retracted (metres),
# summed from the MJCF link lengths: motor 0.10 + forearm 0.25 + glove 0.06.
_GLOVE_BASE_OFFSET_M: float = 0.41


class MujocoPlant:
    """A MuJoCo-backed striking plant satisfying IRobotPlant."""

    def __init__(self, model_path: Path, motor_actuator: str = "motor_y_act") -> None:
        """Load the MJCF and resolve the handles used on the hot path.

        Args:
            model_path: Path to the scene MJCF.
            motor_actuator: Name of the position actuator that drives the punch.

        Raises:
            FileNotFoundError: If the model path does not exist.
            RuntimeError: If the actuator or expected joints are missing.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self._act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_actuator)
        if self._act_id < 0:
            raise RuntimeError(f"actuator '{motor_actuator}' not found in model")
        ctrl_range = self.model.actuator_ctrlrange[self._act_id]
        self._ctrl_lo, self._ctrl_hi = float(ctrl_range[0]), float(ctrl_range[1])

        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
        if base_id < 0:
            raise RuntimeError("body 'arm_base' not found in model")
        self._arm_base_id = base_id

        # qpos/dof addresses for every (1-DOF) joint, resolved once.
        self._qpos_adr = self.model.jnt_qposadr.copy()
        self._dof_adr = self.model.jnt_dofadr.copy()
        self._stiffness = self.model.jnt_stiffness.copy()

        mujoco.mj_forward(self.model, self.data)

    def command_strike(self, cmd: StrikeCommand) -> None:
        """Map a high-level strike to the motor-stage position setpoint.

        Phase 0 mapping: extend the motor so the glove tip reaches the target's
        depth along the punch axis (+y), clamped to the mechanical range. The
        series spring decouples this setpoint from the contact force.
        """
        arm_base_y = float(self.data.xpos[self._arm_base_id][1])
        desired = cmd.target.y - arm_base_y - _GLOVE_BASE_OFFSET_M
        self.data.ctrl[self._act_id] = float(np.clip(desired, self._ctrl_lo, self._ctrl_hi))

    def read_joint_state(self) -> JointState:
        """Per-joint position, velocity and SEA force.

        Force is the series-spring reaction, ``stiffness * deflection``, which is
        zero for the (stiffness-free) motor joint and meaningful for the spring
        joint. This is the deflection-based force sensing the real SEA uses.
        """
        n = self.model.njnt
        position = np.empty(n, dtype=np.float64)
        velocity = np.empty(n, dtype=np.float64)
        force = np.empty(n, dtype=np.float64)
        for j in range(n):
            qadr = int(self._qpos_adr[j])
            vadr = int(self._dof_adr[j])
            position[j] = self.data.qpos[qadr]
            velocity[j] = self.data.qvel[vadr]
            force[j] = self._stiffness[j] * self.data.qpos[qadr]
        return JointState(position=position, velocity=velocity, force=force)

    def step(self, dt: float) -> None:
        """Advance physics by ``dt`` seconds using the model timestep.

        ``dt`` is covered by an integer number of solver substeps so callers can
        think in seconds while the integrator keeps its fixed timestep.
        """
        timestep = float(self.model.opt.timestep)
        n_sub = max(1, int(round(dt / timestep)))
        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)

    def emergency_stop(self) -> None:
        """Halt actuation: zero the command and freeze velocities.

        The software backup to the passive mechanical limit. Modelled here as an
        instantaneous velocity kill; on the real robot this is the drive brake.
        Safe to call repeatedly and at any time.
        """
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

    def force_array(self) -> FloatArray:
        """Convenience: just the per-joint SEA force, for the force-cap check."""
        return self.read_joint_state().force
