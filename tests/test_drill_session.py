"""Phase 3.1 gate: multi-arm combos under a curriculum stay safe end to end.

The full stack (two arms, combo grammar, curriculum, feints, scripted trainee)
must keep the safety invariant: zero unsafe steps across every behaviour, lunges
safely aborted, and difficulty advancing as the trainee dodges.

All pure NumPy via the FakePlant: no mujoco needed for the safety gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakePlant, FixedObserver, make_pose
from robot_twin.core.types import StrikeType, Vec3
from robot_twin.domain.combo import ComboGrammar, ComboTemplate
from robot_twin.hal.sim_trainee import DodgePolicy, ScriptedTrainee
from robot_twin.safety.arbiter import RobotGeometry, SafetyArbiter, SafetyConfig
from robot_twin.services.curriculum import Curriculum, CurriculumLevel
from robot_twin.services.drill_engine import DrillEngine, DrillSession, DrillState
from robot_twin.services.scoring import Scorer


def _two_arm_arbiter() -> SafetyArbiter:
    return SafetyArbiter(
        config=SafetyConfig(),
        geometry=RobotGeometry(
            arm_bases=(Vec3(-0.18, 0.0, 1.0), Vec3(0.18, 0.0, 1.0)), reach_max_m=0.9
        ),
    )


def _session(policy: DodgePolicy) -> tuple[DrillSession, FakePlant, DrillEngine]:
    trainee = ScriptedTrainee(FixedObserver(make_pose(), latency_s=0.02), policy=policy)
    plant = FakePlant(n_joints=4)
    engine = DrillEngine(plant, trainee, trainee, _two_arm_arbiter(), Scorer())
    session = DrillSession(engine, ComboGrammar(n_arms=2), Curriculum())
    return session, plant, engine


class TestSessionSafetyGate:
    def test_dodging_session_is_safe_and_scores(self) -> None:
        session, plant, _engine = _session(DodgePolicy.ALWAYS_DODGE)
        report = session.run(40, np.random.default_rng(0))
        assert report.total_unsafe_steps == 0
        assert report.aborts == 0
        assert not plant.stopped
        assert report.summary.dodges > 0

    def test_lunging_session_is_safe_and_aborts(self) -> None:
        session, plant, _engine = _session(DodgePolicy.LUNGE)
        report = session.run(20, np.random.default_rng(1))
        assert report.total_unsafe_steps == 0
        assert report.aborts > 0  # the safety abort fired
        assert plant.stopped
        assert report.summary.dodges == 0

    def test_dodging_advances_difficulty(self) -> None:
        session, _plant, _engine = _session(DodgePolicy.ALWAYS_DODGE)
        report = session.run(60, np.random.default_rng(2))
        # A consistently dodging trainee should climb past the warmup level.
        assert report.final_level != "warmup"

    @pytest.mark.parametrize("policy", list(DodgePolicy))
    def test_every_policy_keeps_unsafe_steps_zero(self, policy) -> None:
        session, _plant, _engine = _session(policy)
        report = session.run(20, np.random.default_rng(3))
        assert report.total_unsafe_steps == 0


class TestComboAndFeints:
    def test_combo_uses_both_arms(self) -> None:
        trainee = ScriptedTrainee(FixedObserver(make_pose(), latency_s=0.02))
        engine = DrillEngine(FakePlant(n_joints=4), trainee, trainee, _two_arm_arbiter(), Scorer())
        grammar = ComboGrammar(n_arms=2)
        # jab-cross-hook spans both arms (lead, rear, lead); run it and confirm
        # it stays safe and actually drives both strikers.
        combo = grammar.materialise(
            ComboTemplate("jch", (StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK), 1.0)
        )
        outcome = engine.run_combo(combo, Curriculum().current(), np.random.default_rng(0))
        assert outcome.unsafe_steps == 0
        assert {s.arm_id for s in combo.strikes} == {0, 1}

    def test_all_feint_level_produces_feints(self) -> None:
        trainee = ScriptedTrainee(FixedObserver(make_pose(), latency_s=0.02))
        engine = DrillEngine(FakePlant(n_joints=4), trainee, trainee, _two_arm_arbiter(), Scorer())
        grammar = ComboGrammar(n_arms=2)
        all_feint = CurriculumLevel(
            0, "feint", telegraph_scale=1.0, speed_scale=1.0, feint_prob=1.0
        )
        combo = grammar.sample(np.random.default_rng(0))
        outcome = engine.run_combo(combo, all_feint, np.random.default_rng(0))
        assert all(o.state is DrillState.FEINTED for o in outcome.strikes)
        assert engine.scorer.summary().feints == len(combo.strikes)
        assert outcome.unsafe_steps == 0
