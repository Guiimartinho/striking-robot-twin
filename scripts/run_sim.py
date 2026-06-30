"""Run the digital twin: a safety-gated strike demo (Phase 0).

Wires the sim plant, the ground-truth observer and the SafetyArbiter exactly as
any layer above the HAL would, then shows the two outcomes that matter:
  - a low body shot is approved, executed, and the SEA force is read back;
  - a shot aimed at the head is vetoed before it can move, and the plant is
    e-stopped.

No viewer window; this is the headless logic demo. For the 3D view use
scripts/viewer.py. Requires mujoco (pip install mujoco).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python scripts/run_sim.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_twin.config.app_config import AppConfig  # noqa: E402
from robot_twin.core.types import Keypoint, StrikeCommand, StrikeType, Vec3  # noqa: E402
from robot_twin.services.telemetry import configure_logging, get_logger  # noqa: E402

logger = get_logger("run_sim")


def main() -> int:
    configure_logging()
    cfg = AppConfig.load()

    try:
        from robot_twin.hal.mujoco_plant import MujocoPlant
        from robot_twin.hal.sim_observer import SimGTObserver
    except ImportError:
        logger.error("mujoco is not installed. Run: uv pip install mujoco")
        return 1

    from robot_twin.safety.arbiter import SafetyArbiter

    if not cfg.model_path.exists():
        logger.error("model not found: %s (run from the repo root)", cfg.model_path)
        return 1

    plant = MujocoPlant(cfg.model_path)
    observer = SimGTObserver(plant.model, plant.data, nominal_latency_s=cfg.nominal_latency_s)
    arbiter = SafetyArbiter(
        config=cfg.safety.to_safety_config(),
        geometry=cfg.geometry.to_robot_geometry(),
    )

    pose = observer.get_pose()
    latency = observer.latency_s()
    radius = arbiter.compute_keepout_radius(latency)
    logger.info("observer latency %.3fs -> R_keepout %.3fm", latency, radius)
    logger.info("head keypoint at %s", tuple(np.round(pose.keypoint(Keypoint.HEAD).as_array(), 3)))

    # 1. A safe low body shot: approved and executed.
    body = StrikeCommand(
        strike_type=StrikeType.CROSS,
        target=Vec3(0.0, 0.7, 1.0),
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )
    body_result = arbiter.evaluate(body, pose, latency)
    if body_result.is_ok:
        logger.info("body shot APPROVED, executing")
        plant.command_strike(body)
        force_cap = cfg.safety.force_cap_n
        peak_force = 0.0
        aborted = False
        for _ in range(250):  # up to ~0.5 s of control steps
            plant.step(cfg.control_dt)
            peak_force = max(peak_force, float(np.max(np.abs(plant.force_array()))))
            # Runtime force guard: the SEA force is monitored every cycle, not
            # only at command time, so contact over the energy budget aborts.
            if peak_force > force_cap:
                logger.warning(
                    "force cap ABORT: sensed %.1f N > cap %.1f N, e-stop",
                    peak_force,
                    force_cap,
                )
                plant.emergency_stop()
                aborted = True
                break
        if not aborted:
            logger.info("body shot done, peak sensed SEA force %.1f N", peak_force)
    else:
        logger.warning("body shot vetoed unexpectedly: %s", body_result.message)

    # 2. A shot aimed at the head: must be vetoed before any motion.
    head_target = pose.keypoint(Keypoint.HEAD)
    head = StrikeCommand(
        strike_type=StrikeType.JAB,
        target=head_target,
        speed_mps=2.0,
        telegraph_s=0.2,
        arm_id=0,
    )
    head_result = arbiter.evaluate(head, pose, latency)
    if head_result.is_err:
        logger.info("head shot VETOED (%s): %s", head_result.code.value, head_result.message)
        plant.emergency_stop()
    else:
        logger.error("SAFETY FAILURE: head shot was approved")
        return 2

    logger.info("Phase 0 demo complete: safe strike ran, head strike blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
