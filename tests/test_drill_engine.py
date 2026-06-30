"""Phase 2 gate: the closed loop trains end to end with zero safety violations.

The simulated trainee runs the drill under three behaviours, and the engine must:
  - never advance the plant on a tick the arbiter would veto (unsafe_steps == 0);
  - resolve a dodging trainee end to end;
  - safely abort (e-stop) when the trainee lunges into the strike;
  - veto at launch and throw nothing when the target is unsafe.

All pure NumPy via the FakePlant: no mujoco needed for the safety gate.
"""

from __future__ import annotations

import pytest

from conftest import FakePlant, FixedObserver, make_pose
from robot_twin.core.result import ErrorCode
from robot_twin.core.types import Vec3
from robot_twin.hal.sim_trainee import DodgePolicy, ScriptedTrainee
from robot_twin.safety.arbiter import RobotGeometry, SafetyArbiter, SafetyConfig
from robot_twin.services.drill_engine import DrillEngine, DrillState
from robot_twin.services.scoring import Scorer


def _make_engine(
    arbiter: SafetyArbiter, policy: DodgePolicy
) -> tuple[DrillEngine, FakePlant, Scorer]:
    trainee = ScriptedTrainee(FixedObserver(make_pose(), latency_s=0.02), policy=policy)
    plant = FakePlant()
    scorer = Scorer()
    engine = DrillEngine(plant, trainee, trainee, arbiter, scorer)
    return engine, plant, scorer


class TestDrillSafetyGate:
    def test_dodging_trainee_resolves_safely(self, arbiter) -> None:
        engine, plant, scorer = _make_engine(arbiter, DodgePolicy.ALWAYS_DODGE)
        outcomes = engine.run_session(20)
        assert all(o.state is DrillState.RESOLVED for o in outcomes)
        assert all(o.dodged for o in outcomes)
        assert sum(o.unsafe_steps for o in outcomes) == 0
        assert not plant.stopped
        assert scorer.summary().dodge_rate == 1.0

    def test_static_trainee_is_tagged_not_dodged(self, arbiter) -> None:
        engine, plant, scorer = _make_engine(arbiter, DodgePolicy.NEVER_DODGE)
        outcomes = engine.run_session(20)
        assert all(o.state is DrillState.RESOLVED for o in outcomes)
        assert not any(o.dodged for o in outcomes)
        assert sum(o.unsafe_steps for o in outcomes) == 0
        assert scorer.summary().dodge_rate == 0.0

    def test_lunging_trainee_is_safely_aborted(self, arbiter) -> None:
        engine, plant, scorer = _make_engine(arbiter, DodgePolicy.LUNGE)
        outcomes = engine.run_session(20)
        assert all(o.state is DrillState.ABORTED for o in outcomes)
        assert all(o.safety_code is ErrorCode.KEEPOUT_VIOLATION for o in outcomes)
        assert sum(o.unsafe_steps for o in outcomes) == 0
        assert plant.stopped  # the e-stop fired
        assert scorer.summary().aborts == 20

    def test_launch_veto_throws_nothing(self) -> None:
        # A reach so small the body target is out of reach: the strike must be
        # vetoed at launch and never executed.
        tight = SafetyArbiter(
            config=SafetyConfig(),
            geometry=RobotGeometry(arm_bases=(Vec3(0.0, 0.0, 1.0),), reach_max_m=0.2),
        )
        engine, plant, scorer = _make_engine(tight, DodgePolicy.NEVER_DODGE)
        outcomes = engine.run_session(5)
        assert all(o.state is DrillState.VETOED for o in outcomes)
        assert all(o.safety_code is ErrorCode.REACH_EXCEEDED for o in outcomes)
        assert sum(o.unsafe_steps for o in outcomes) == 0
        assert scorer.summary().vetoes == 5

    def test_zero_unsafe_steps_over_mixed_session(self, arbiter) -> None:
        # The headline gate: across every behaviour, no unsafe step ever happens,
        # and the abort mechanism demonstrably fires (non-vacuous).
        total_unsafe = 0
        aborts = 0
        for policy in (DodgePolicy.ALWAYS_DODGE, DodgePolicy.NEVER_DODGE, DodgePolicy.LUNGE):
            engine, _plant, scorer = _make_engine(arbiter, policy)
            for outcome in engine.run_session(30):
                total_unsafe += outcome.unsafe_steps
                aborts += 1 if outcome.state is DrillState.ABORTED else 0
        assert total_unsafe == 0
        assert aborts > 0


class TestDrillScoring:
    def test_dodges_outscore_hits(self, arbiter) -> None:
        dodging, _p1, dodge_scorer = _make_engine(arbiter, DodgePolicy.ALWAYS_DODGE)
        static, _p2, hit_scorer = _make_engine(arbiter, DodgePolicy.NEVER_DODGE)
        dodging.run_session(10)
        static.run_session(10)
        assert dodge_scorer.summary().score > hit_scorer.summary().score


@pytest.mark.parametrize("policy", list(DodgePolicy))
def test_every_policy_keeps_unsafe_steps_zero(arbiter, policy) -> None:
    engine, _plant, _scorer = _make_engine(arbiter, policy)
    assert sum(o.unsafe_steps for o in engine.run_session(10)) == 0
