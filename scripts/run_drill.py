"""Run the full Phase 3 drill on the MuJoCo twin: the "run everything" demo.

Wires the whole stack onto the sim twin and runs complete sessions:
  - two SEA striker arms (multi-striker);
  - a ComboGrammar sampling jab / jab-cross / jab-cross-hook ... combos;
  - a Curriculum that ramps telegraph, speed and feints as the trainee improves;
  - a scripted trainee (dodging, then lunging);
  - the DodgeDetector / GuardDetector judging each strike;
  - the SafetyArbiter gating every command, aborting any lunge.

Prints a per-session report. The dodging session should climb the curriculum and
score; the lunging session should be aborted safely every time, with zero unsafe
steps in both. Requires mujoco (pip install mujoco).

Note: the Phase 2/3 trainee dodge is kinematic (the physical mannequin is
static); strikes are non-contact feints, so the dodge plays out in the observed
pose. Physical trainee articulation is a later phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_twin.config.app_config import AppConfig  # noqa: E402
from robot_twin.services.telemetry import configure_logging, get_logger  # noqa: E402

logger = get_logger("run_drill")


def main() -> int:
    configure_logging()
    cfg = AppConfig.load()

    try:
        from robot_twin.hal.mujoco_plant import MujocoPlant
        from robot_twin.hal.sim_observer import SimGTObserver
    except ImportError:
        logger.error("mujoco is not installed. Run: uv pip install mujoco")
        return 1

    from robot_twin.domain.combo import ComboGrammar
    from robot_twin.hal.sim_trainee import DodgePolicy, ScriptedTrainee
    from robot_twin.safety.arbiter import SafetyArbiter
    from robot_twin.services.curriculum import Curriculum
    from robot_twin.services.drill_engine import DrillEngine, DrillSession
    from robot_twin.services.scoring import Scorer

    if not cfg.model_path.exists():
        logger.error("model not found: %s (run from the repo root)", cfg.model_path)
        return 1

    arbiter = SafetyArbiter(
        config=cfg.safety.to_safety_config(),
        geometry=cfg.geometry.to_robot_geometry(),
    )

    for policy in (DodgePolicy.ALWAYS_DODGE, DodgePolicy.LUNGE):
        plant = MujocoPlant(cfg.model_path)
        observer = SimGTObserver(plant.model, plant.data, nominal_latency_s=cfg.nominal_latency_s)
        trainee = ScriptedTrainee(observer, policy=policy)
        engine = DrillEngine(plant, trainee, trainee, arbiter, Scorer())
        session = DrillSession(engine, ComboGrammar(n_arms=plant.n_arms), Curriculum())

        report = session.run(40, np.random.default_rng(7))
        logger.info("=== session: trainee policy = %s (%d arms) ===", policy.value, plant.n_arms)
        logger.info(
            "  combos=%d final_level=%s total_unsafe_steps=%d aborts=%d",
            report.combos,
            report.final_level,
            report.total_unsafe_steps,
            report.aborts,
        )
        s = report.summary
        logger.info(
            "  delivered=%d dodges=%d hits=%d feints=%d vetoes=%d dodge_rate=%.2f score=%.1f",
            s.resolved,
            s.dodges,
            s.hits,
            s.feints,
            s.vetoes,
            s.dodge_rate,
            s.score,
        )
        if report.total_unsafe_steps != 0:
            logger.error("  SAFETY FAILURE: %d unsafe steps", report.total_unsafe_steps)
            return 2

    logger.info("Phase 3 drill demo complete: full stack ran, zero unsafe steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
