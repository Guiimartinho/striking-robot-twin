"""Run the Phase 2 closed-loop drill on the MuJoCo twin.

Wires the sim plant, a scripted trainee (observer + opponent) and the
SafetyArbiter into the DrillEngine, then runs a short session and prints the
outcomes and the score summary. Shows the loop end to end: telegraphed strikes,
the trainee dodging, and the safety layer aborting a lunge.

Note: the Phase 2 trainee is kinematic (the physical mannequin is static), so a
dodge is modelled in the observed pose; physical glove contact with the static
torso can still raise force and trip the safety abort, which is itself the safety
layer working on the real plant. Physical trainee articulation is a later phase.

Requires mujoco (pip install mujoco).
"""

from __future__ import annotations

import sys
from pathlib import Path

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

    from robot_twin.hal.sim_trainee import DodgePolicy, ScriptedTrainee
    from robot_twin.safety.arbiter import SafetyArbiter
    from robot_twin.services.drill_engine import DrillEngine
    from robot_twin.services.scoring import Scorer

    if not cfg.model_path.exists():
        logger.error("model not found: %s (run from the repo root)", cfg.model_path)
        return 1

    arbiter = SafetyArbiter(
        config=cfg.safety.to_safety_config(),
        geometry=cfg.geometry.to_robot_geometry(),
    )

    # Two short sessions: a dodging trainee, then a lunging one to show the abort.
    for policy in (DodgePolicy.ALWAYS_DODGE, DodgePolicy.LUNGE):
        plant = MujocoPlant(cfg.model_path)
        observer = SimGTObserver(plant.model, plant.data, nominal_latency_s=cfg.nominal_latency_s)
        trainee = ScriptedTrainee(observer, policy=policy)
        scorer = Scorer()
        engine = DrillEngine(plant, trainee, trainee, arbiter, scorer)

        logger.info("=== session: trainee policy = %s ===", policy.value)
        for i, outcome in enumerate(engine.run_session(5)):
            logger.info(
                "  ep %d: %s%s%s",
                i,
                outcome.state.value,
                f", dodged={outcome.dodged}" if outcome.dodged is not None else "",
                f", safety={outcome.safety_code.value}" if outcome.safety_code else "",
            )
        s = scorer.summary()
        logger.info(
            "  summary: resolved=%d dodges=%d aborts=%d vetoes=%d dodge_rate=%.2f score=%.1f",
            s.resolved,
            s.dodges,
            s.aborts,
            s.vetoes,
            s.dodge_rate,
            s.score,
        )

    logger.info("Phase 2 drill demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
