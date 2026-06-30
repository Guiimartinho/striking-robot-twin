"""The HAL contract: the spine of the project.

These Protocols are the only thing layers above the HAL are allowed to know
about a plant. The sim (``MujocoPlant`` + ``SimGTObserver``) and the real robot
(``RealPlant`` + ``CameraPoseObserver``) both satisfy them, so Domain, Safety,
Services and RL run unchanged against either.

Hard rule (enforced by review and by the contract tests): nothing that imports
these Protocols may import ``mujoco``, ``jax`` or ``cv2``. Both plants must pass
the same contract test suite.

Protocols are structural: a class satisfies the interface by having the methods,
without inheriting. ``@runtime_checkable`` lets tests assert conformance with
``isinstance`` while keeping that structural freedom.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robot_twin.core.types import JointState, StrikeCommand, TraineePose


@runtime_checkable
class IRobotPlant(Protocol):
    """A striking plant: arms that execute high-level strike commands."""

    def command_strike(self, cmd: StrikeCommand) -> None:
        """Command a high-level strike: target, speed, telegraph.

        The plant turns intent into actuator setpoints. Implementations must not
        block; the motion plays out across subsequent :meth:`step` calls.
        """
        ...

    def read_joint_state(self) -> JointState:
        """Return per-joint position, velocity and force.

        Force is sensed from SEA spring deflection, not a separate load cell.
        """
        ...

    def step(self, dt: float) -> None:
        """Advance the plant by ``dt`` seconds.

        In the sim this integrates physics. On the real robot it is a no-op or a
        synchronisation point: hardware advances in real time on its own.
        """
        ...

    def emergency_stop(self) -> None:
        """Halt all actuation immediately. The software backup to the passive,

        mechanical energy limit. Must be safe to call at any time, repeatedly.
        """
        ...


@runtime_checkable
class ITraineeObserver(Protocol):
    """A source of trainee pose: ground truth in sim, camera estimate on robot."""

    def get_pose(self) -> TraineePose:
        """Return the latest trainee keypoints with per-keypoint confidence."""
        ...

    def latency_s(self) -> float:
        """Observed end-to-end pipeline latency, seconds.

        This feeds ``R_keepout``: the keep-out volume is inflated by how far the
        head could have moved between estimating it and stopping the actuator.
        Implementations should report an honest, current estimate, not a
        nominal constant, so the SafetyArbiter inflates the margin under load.
        """
        ...
