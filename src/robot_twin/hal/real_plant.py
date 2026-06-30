"""Real-robot plant and observer stubs (STM32 / Jetson). Not yet implemented.

These exist so the architecture is visible and so the contract tests have a
second plant to point at once the firmware link is built. Every method raises
``NotImplementedError`` on purpose: there is no hardware yet, and a silent
no-op on a machine that punches people would be a safety hazard.

When the STM32 link lands, ``RealPlant`` speaks UART/CAN to the low-level
controller and ``CameraPoseObserver`` runs pose estimation on the Jetson. Both
must then pass the exact same contract tests as the sim plant.
"""

from __future__ import annotations

from robot_twin.core.types import JointState, StrikeCommand, TraineePose


class RealPlant:
    """STM32-backed striking plant. Phase: future (real hardware)."""

    def command_strike(self, cmd: StrikeCommand) -> None:  # noqa: D102
        raise NotImplementedError("RealPlant requires the STM32 firmware link")

    def read_joint_state(self) -> JointState:  # noqa: D102
        raise NotImplementedError("RealPlant requires the STM32 firmware link")

    def step(self, dt: float) -> None:  # noqa: D102
        raise NotImplementedError("RealPlant requires the STM32 firmware link")

    def emergency_stop(self) -> None:  # noqa: D102
        raise NotImplementedError("RealPlant requires the STM32 firmware link")


class CameraPoseObserver:
    """Jetson camera-based trainee observer. Phase: future (real hardware)."""

    def get_pose(self) -> TraineePose:  # noqa: D102
        raise NotImplementedError("CameraPoseObserver requires the Jetson pipeline")

    def latency_s(self) -> float:  # noqa: D102
        raise NotImplementedError("CameraPoseObserver requires the Jetson pipeline")
