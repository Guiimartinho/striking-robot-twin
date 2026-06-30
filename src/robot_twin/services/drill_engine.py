"""DrillEngine and DrillSession: the FSM that runs training reps, safely. Phase 3.

A single strike (``run_episode``) and a combo (``run_combo``) share one core,
``_throw_strike``: select a safe target, validate it at launch through the
SafetyArbiter, telegraph and throw, then re-validate every control tick while the
strike plays out. If the trainee lunges in (or contact force exceeds the cap) the
engine e-stops and the strike is a safe abort, never a violation. A feint is
telegraphed and withdrawn without follow-through.

``DrillSession`` ties the ComboGrammar (what to throw) and the Curriculum (how
hard to throw it) to the engine, sampling combos and advancing difficulty as the
trainee improves.

Two safety invariants the engine guarantees by construction (the gate checks):
  1. it never commands a strike the arbiter vetoes at launch;
  2. it never advances the plant on a tick where the arbiter would veto the
     current pose (``unsafe_steps`` stays 0).

Plant-agnostic: depends only on the HAL Protocols, the arbiter, the domain
detectors, the combo grammar, the curriculum and the scorer. No mujoco.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_twin.core.result import ErrorCode
from robot_twin.core.types import StrikeCommand, StrikeType, Vec3
from robot_twin.domain.combo import Combo, ComboGrammar
from robot_twin.domain.dodge_detector import DodgeDetector, DodgeResult, DodgeState
from robot_twin.domain.guard import GuardDetector, GuardState
from robot_twin.domain.target_selector import TargetSelector
from robot_twin.hal.interfaces import IOpponent, IRobotPlant, ITraineeObserver
from robot_twin.safety.arbiter import SafetyArbiter
from robot_twin.services.curriculum import Curriculum, CurriculumLevel
from robot_twin.services.scoring import Scorer, ScoreSummary


class DrillState(Enum):
    """Terminal state of a single strike."""

    VETOED = "vetoed"  # safety rejected the strike at launch; nothing was thrown
    ABORTED = "aborted"  # safety aborted the strike mid-flight; e-stopped
    RESOLVED = "resolved"  # strike delivered; dodge and guard were judged
    FEINTED = "feinted"  # telegraphed and withdrawn; not a delivered strike


@dataclass(frozen=True, slots=True)
class DrillConfig:
    """Timing and motion parameters for a strike, in SI units.

    Telegraph is deliberately slow so the trainee can read and react; the
    curriculum scales it down as difficulty rises.
    """

    telegraph_s: float = 0.4
    strike_s: float = 0.4
    recover_s: float = 0.2
    control_dt: float = 0.02
    strike_speed_mps: float = 1.5
    arm_id: int = 0  # default arm for the single-strike run_episode


@dataclass(frozen=True, slots=True)
class DrillOutcome:
    """Result of one strike."""

    state: DrillState
    dodged: bool | None  # None unless delivered and resolved
    guard_up: bool | None
    safety_code: ErrorCode | None  # the veto/abort reason, if any
    unsafe_steps: int  # plant steps taken while the arbiter would veto: must be 0
    dodge: DodgeResult | None = None


@dataclass(frozen=True, slots=True)
class ComboOutcome:
    """Result of a whole combo: the per-strike outcomes in order."""

    name: str
    strikes: tuple[DrillOutcome, ...]

    @property
    def unsafe_steps(self) -> int:
        return sum(o.unsafe_steps for o in self.strikes)

    @property
    def aborted(self) -> bool:
        return any(o.state is DrillState.ABORTED for o in self.strikes)


class DrillEngine:
    """Runs telegraphed strike reps against a trainee, safety-gated throughout."""

    def __init__(
        self,
        plant: IRobotPlant,
        observer: ITraineeObserver,
        opponent: IOpponent,
        arbiter: SafetyArbiter,
        scorer: Scorer,
        *,
        target_selector: TargetSelector | None = None,
        dodge_detector: DodgeDetector | None = None,
        guard_detector: GuardDetector | None = None,
        config: DrillConfig | None = None,
    ) -> None:
        self._plant = plant
        self._observer = observer
        self._opponent = opponent
        self._arbiter = arbiter
        self._scorer = scorer
        self._target_selector = target_selector or TargetSelector()
        self._dodge_detector = dodge_detector or DodgeDetector()
        self._guard_detector = guard_detector or GuardDetector()
        self._cfg = config or DrillConfig()

    @property
    def scorer(self) -> Scorer:
        return self._scorer

    def _force(self):
        """Current per-joint SEA force for the arbiter's force-cap check."""
        return self._plant.read_joint_state().force

    def _motion_loop(
        self, cmd: StrikeCommand, motion_s: float
    ) -> tuple[bool, ErrorCode | None, int]:
        """Step the strike while re-validating safety every tick.

        Returns ``(aborted, abort_code, unsafe_steps)``. ``unsafe_steps`` is the
        count of plant steps taken while the arbiter would veto: it is 0 by
        construction (the loop e-stops and returns before such a step).
        """
        dt = self._cfg.control_dt
        unsafe_steps = 0
        for _ in range(max(1, round(motion_s / dt))):
            self._opponent.advance(dt)
            cur_pose = self._observer.get_pose()
            eval_now = self._arbiter.evaluate(
                cmd, cur_pose, self._observer.latency_s(), self._force()
            )
            if eval_now.is_err:
                self._plant.emergency_stop()
                return True, eval_now.code, unsafe_steps
            self._plant.step(dt)
        return False, None, unsafe_steps

    def _throw_strike(
        self,
        arm_id: int,
        strike_type: StrikeType,
        telegraph_s: float,
        speed_mps: float,
        is_feint: bool,
    ) -> DrillOutcome:
        """Select, validate, throw and resolve one strike, then retract the arm."""
        arm_base = self._arbiter.geometry.arm_bases[arm_id]
        pose_launch = self._observer.get_pose()
        latency = self._observer.latency_s()
        target = self._target_selector.select(pose_launch, arm_base, strike_type)
        cmd = StrikeCommand(
            strike_type=strike_type,
            target=target,
            speed_mps=speed_mps,
            telegraph_s=telegraph_s,
            arm_id=arm_id,
        )

        launch_eval = self._arbiter.evaluate(cmd, pose_launch, latency, self._force())
        if launch_eval.is_err:
            self._scorer.record(dodged=None, guard_up=None, aborted=False, vetoed=True)
            self._retract_arm(arm_id)
            return DrillOutcome(DrillState.VETOED, None, None, launch_eval.code, 0)

        self._opponent.react_to_strike(target)
        self._plant.command_strike(cmd)
        motion_s = telegraph_s + (0.0 if is_feint else self._cfg.strike_s)
        aborted, abort_code, unsafe_steps = self._motion_loop(cmd, motion_s)

        if aborted:
            self._scorer.record(dodged=None, guard_up=None, aborted=True, vetoed=False)
            self._retract_arm(arm_id)
            return DrillOutcome(DrillState.ABORTED, None, None, abort_code, unsafe_steps)

        if is_feint:
            self._scorer.record(
                dodged=None, guard_up=None, aborted=False, vetoed=False, feinted=True
            )
            self._retract_arm(arm_id)
            return DrillOutcome(DrillState.FEINTED, None, None, None, unsafe_steps)

        pose_arrival = self._observer.get_pose()
        dodge = self._dodge_detector.evaluate(arm_base, target, pose_launch, pose_arrival)
        guard = self._guard_detector.assess(pose_arrival)
        dodged = dodge.state is DodgeState.DODGED
        guard_up = guard.state is GuardState.UP
        self._scorer.record(dodged=dodged, guard_up=guard_up, aborted=False, vetoed=False)
        self._retract_arm(arm_id)
        return DrillOutcome(DrillState.RESOLVED, dodged, guard_up, None, unsafe_steps, dodge)

    def run_episode(self) -> DrillOutcome:
        """Run one single-strike rep with the default arm and reset the trainee."""
        outcome = self._throw_strike(
            self._cfg.arm_id,
            StrikeType.JAB,
            self._cfg.telegraph_s,
            self._cfg.strike_speed_mps,
            is_feint=False,
        )
        self._opponent.reset()
        return outcome

    def run_session(self, n_episodes: int) -> list[DrillOutcome]:
        """Run a sequence of single-strike reps."""
        return [self.run_episode() for _ in range(n_episodes)]

    def run_combo(
        self, combo: Combo, level: CurriculumLevel, rng: np.random.Generator
    ) -> ComboOutcome:
        """Throw a combo at the given difficulty, then reset the trainee.

        Each punch is scaled by the level (telegraph, speed) and may be a feint
        with the level's probability. Punches are separated by their base gap.
        """
        outcomes: list[DrillOutcome] = []
        for spec in combo.strikes:
            if spec.gap_before_s > 0.0:
                self._idle(spec.gap_before_s)
            is_feint = bool(rng.random() < level.feint_prob)
            telegraph = spec.telegraph_s * level.telegraph_scale
            speed = self._cfg.strike_speed_mps * level.speed_scale
            outcomes.append(
                self._throw_strike(spec.arm_id, spec.strike_type, telegraph, speed, is_feint)
            )
        self._opponent.reset()
        return ComboOutcome(name=combo.name, strikes=tuple(outcomes))

    def _idle(self, duration_s: float) -> None:
        """Let time pass between punches: advance the trainee and the plant."""
        dt = self._cfg.control_dt
        for _ in range(max(1, round(duration_s / dt))):
            self._opponent.advance(dt)
            self._plant.step(dt)

    def _retract_arm(self, arm_id: int) -> None:
        """Retract one arm to its base. Moving away from the trainee is safe."""
        arm_base = self._arbiter.geometry.arm_bases[arm_id]
        retract = StrikeCommand(
            strike_type=StrikeType.JAB,
            target=Vec3(arm_base.x, arm_base.y, arm_base.z),
            speed_mps=self._cfg.strike_speed_mps,
            telegraph_s=0.0,
            arm_id=arm_id,
        )
        self._plant.command_strike(retract)
        for _ in range(max(1, round(self._cfg.recover_s / self._cfg.control_dt))):
            self._plant.step(self._cfg.control_dt)


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Summary of a full curriculum session."""

    combos: int
    final_level: str
    total_unsafe_steps: int
    aborts: int
    summary: ScoreSummary


class DrillSession:
    """Runs a full session: sample combos, deliver at curriculum difficulty."""

    def __init__(self, engine: DrillEngine, grammar: ComboGrammar, curriculum: Curriculum) -> None:
        self._engine = engine
        self._grammar = grammar
        self._curriculum = curriculum

    def run(self, n_combos: int, rng: np.random.Generator) -> SessionReport:
        """Run ``n_combos`` combos, advancing difficulty on the trainee's dodging."""
        total_unsafe = 0
        aborts = 0
        for _ in range(n_combos):
            combo = self._grammar.sample(rng)
            level = self._curriculum.current()
            outcome = self._engine.run_combo(combo, level, rng)
            total_unsafe += outcome.unsafe_steps
            for strike in outcome.strikes:
                if strike.state is DrillState.ABORTED:
                    aborts += 1
                # Only delivered (resolved) strikes inform progression.
                self._curriculum.record(
                    strike.dodged if strike.state is DrillState.RESOLVED else None
                )
        return SessionReport(
            combos=n_combos,
            final_level=self._curriculum.current().name,
            total_unsafe_steps=total_unsafe,
            aborts=aborts,
            summary=self._engine.scorer.summary(),
        )
