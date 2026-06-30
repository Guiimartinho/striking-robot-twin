"""MujocoPlant: the sim-side IRobotPlant. The only place mujoco is imported

besides the observer. Everything above the HAL stays mujoco-free.

A minimal but faithful multi-arm Phase 3 plant. A strike command is routed by
``arm_id`` to that arm's motor-stage position setpoint; the series spring then
transmits force to the link, and that force is read back from the spring's
deflection (the SEA way). Trajectory shaping from speed/telegraph is future work;
the plant drives the position actuator to the mapped setpoint.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robot_twin.core.types import FloatArray, JointState, StrikeCommand

# Glove tip offset from a motor stage origin when fully retracted (metres),
# summed from the MJCF link lengths: motor 0.10 + forearm 0.25 + glove 0.06.
_GLOVE_BASE_OFFSET_M: float = 0.41

_DEFAULT_ACTUATORS: tuple[str, ...] = ("arm0_motor_act", "arm1_motor_act")
_DEFAULT_BASES: tuple[str, ...] = ("arm0_base", "arm1_base")


class MujocoPlant:
    """A MuJoCo-backed multi-arm striking plant satisfying IRobotPlant."""

    def __init__(
        self,
        model_path: Path,
        motor_actuators: tuple[str, ...] = _DEFAULT_ACTUATORS,
        arm_base_bodies: tuple[str, ...] = _DEFAULT_BASES,
    ) -> None:
        """Load the MJCF and resolve the per-arm handles used on the hot path.

        Args:
            model_path: Path to the scene MJCF.
            motor_actuators: Position actuator name per arm, indexed by arm_id.
            arm_base_bodies: Base body name per arm, indexed by arm_id.

        Raises:
            FileNotFoundError: If the model path does not exist.
            ValueError: If the actuator and base lists differ in length.
            RuntimeError: If an actuator or base body is missing from the model.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if len(motor_actuators) != len(arm_base_bodies):
            raise ValueError("motor_actuators and arm_base_bodies must align by arm_id")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Resolve per-arm handles once: (actuator id, base body id, ctrl range).
        self._act_ids: list[int] = []
        self._base_ids: list[int] = []
        self._ctrl_lo: list[float] = []
        self._ctrl_hi: list[float] = []
        for act_name, base_name in zip(motor_actuators, arm_base_bodies, strict=True):
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id < 0:
                raise RuntimeError(f"actuator '{act_name}' not found in model")
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_id < 0:
                raise RuntimeError(f"body '{base_name}' not found in model")
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self._act_ids.append(act_id)
            self._base_ids.append(base_id)
            self._ctrl_lo.append(float(lo))
            self._ctrl_hi.append(float(hi))

        # qpos/dof addresses for every (1-DOF) joint, resolved once.
        self._qpos_adr = self.model.jnt_qposadr.copy()
        self._dof_adr = self.model.jnt_dofadr.copy()
        self._stiffness = self.model.jnt_stiffness.copy()

        mujoco.mj_forward(self.model, self.data)

    @property
    def n_arms(self) -> int:
        return len(self._act_ids)

    def command_strike(self, cmd: StrikeCommand) -> None:
        """Route a strike to its arm's motor-stage position setpoint.

        Phase mapping: extend that arm's motor so the glove tip reaches the
        target's depth along the punch axis (+y), clamped to the mechanical
        range. The series spring decouples this setpoint from the contact force.
        """
        if not 0 <= cmd.arm_id < self.n_arms:
            raise ValueError(f"unknown arm_id {cmd.arm_id} (have {self.n_arms} arms)")
        arm = cmd.arm_id
        arm_base_y = float(self.data.xpos[self._base_ids[arm]][1])
        desired = cmd.target.y - arm_base_y - _GLOVE_BASE_OFFSET_M
        self.data.ctrl[self._act_ids[arm]] = float(
            np.clip(desired, self._ctrl_lo[arm], self._ctrl_hi[arm])
        )

    def read_joint_state(self) -> JointState:
        """Per-joint position, velocity and SEA force across all arms.

        Force is the series-spring reaction, ``stiffness * deflection``, zero for
        the (stiffness-free) motor joints and meaningful for the spring joints.
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
        """Halt all actuation: zero every command and freeze velocities.

        The software backup to the passive mechanical limit. Modelled as an
        instantaneous velocity kill; on the real robot this is the drive brake.
        Safe to call repeatedly and at any time.
        """
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

    def force_array(self) -> FloatArray:
        """Convenience: just the per-joint SEA force, for the force-cap check."""
        return self.read_joint_state().force
