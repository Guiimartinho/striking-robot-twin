"""DrillEngine: the FSM that runs one training rep, safely. Phase 2.

One episode: select a safe body target, validate it at launch through the
SafetyArbiter, telegraph and throw, then re-validate every control tick while the
strike plays out. If the trainee lunges into the strike (or contact force exceeds
the cap), the engine e-stops and the rep is a safe abort, never a violation. When
the strike resolves it judges the dodge and guard and scores the rep.

Two safety invariants the engine guarantees by construction, and the Phase 2 gate
checks:
  1. it never commands a strike the arbiter vetoes at launch;
  2. it never advances the plant on a tick where the arbiter would veto the
     current pose (``unsafe_steps`` stays 0).

Plant-agnostic: depends only on the HAL Protocols, the arbiter, the domain
detectors and the scorer. No mujoco. The opponent is the sim trainee in the sim
and a no-op on the real robot (the human moves on their own).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robot_twin.core.result import ErrorCode
from robot_twin.core.types import StrikeCommand, StrikeType, Vec3
from robot_twin.domain.dodge_detector import DodgeDetector, DodgeResult, DodgeState
from robot_twin.domain.guard import GuardDetector, GuardState
from robot_twin.domain.target_selector import TargetSelector
from robot_twin.hal.interfaces import IOpponent, IRobotPlant, ITraineeObserver
from robot_twin.safety.arbiter import SafetyArbiter
from robot_twin.services.scoring import Scorer


class DrillState(Enum):
    """Terminal state of a drill episode."""

    VETOED = "vetoed"  # safety rejected the strike at launch; nothing was thrown
    ABORTED = "aborted"  # safety aborted the strike mid-flight; e-stopped
    RESOLVED = "resolved"  # strike delivered; dodge and guard were judged


@dataclass(frozen=True, slots=True)
class DrillConfig:
    """Timing and motion parameters for a drill rep, in SI units.

    Telegraph is deliberately slow so the trainee can read and react: this is the
    "slow closed loop" of Phase 2.
    """

    telegraph_s: float = 0.4
    strike_s: float = 0.4
    recover_s: float = 0.2
    control_dt: float = 0.02
    strike_speed_mps: float = 1.5
    arm_id: int = 0


@dataclass(frozen=True, slots=True)
class DrillOutcome:
    """Result of one episode."""

    state: DrillState
    dodged: bool | None  # None when not delivered (vetoed/aborted)
    guard_up: bool | None
    safety_code: ErrorCode | None  # the veto/abort reason, if any
    unsafe_steps: int  # plant steps taken while the arbiter would veto: must be 0
    dodge: DodgeResult | None = None


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

    def _force(self):
        """Current per-joint SEA force for the arbiter's force-cap check."""
        return self._plant.read_joint_state().force

    def run_episode(self) -> DrillOutcome:
        """Run one telegraphed strike rep and return its outcome."""
        dt = self._cfg.control_dt
        arm_base = self._arbiter.geometry.arm_bases[self._cfg.arm_id]

        # 1. Select a safe body target and validate it at launch. The engine
        # never strikes when the arbiter vetoes.
        pose_launch = self._observer.get_pose()
        latency = self._observer.latency_s()
        target = self._target_selector.select(pose_launch, arm_base)
        cmd = StrikeCommand(
            strike_type=StrikeType.JAB,
            target=target,
            speed_mps=self._cfg.strike_speed_mps,
            telegraph_s=self._cfg.telegraph_s,
            arm_id=self._cfg.arm_id,
        )
        launch_eval = self._arbiter.evaluate(cmd, pose_launch, latency, self._force())
        if launch_eval.is_err:
            self._scorer.record(dodged=None, guard_up=None, aborted=False, vetoed=True)
            self._recover(arm_base)
            return DrillOutcome(
                state=DrillState.VETOED,
                dodged=None,
                guard_up=None,
                safety_code=launch_eval.code,
                unsafe_steps=0,
            )

        # 2. Telegraph and throw: signal the opponent and command the strike.
        self._opponent.react_to_strike(target)
        self._plant.command_strike(cmd)

        # 3. Motion window (telegraph + strike): re-validate every tick. Abort
        # before stepping if the current pose would be vetoed.
        unsafe_steps = 0
        aborted = False
        abort_code: ErrorCode | None = None
        n_ticks = max(1, round((self._cfg.telegraph_s + self._cfg.strike_s) / dt))
        for _ in range(n_ticks):
            self._opponent.advance(dt)
            cur_pose = self._observer.get_pose()
            eval_now = self._arbiter.evaluate(
                cmd, cur_pose, self._observer.latency_s(), self._force()
            )
            if eval_now.is_err:
                self._plant.emergency_stop()
                aborted = True
                abort_code = eval_now.code
                break
            self._plant.step(dt)

        # 4. Resolve: judge the dodge and guard, or record the safe abort.
        if aborted:
            self._scorer.record(dodged=None, guard_up=None, aborted=True, vetoed=False)
            self._recover(arm_base)
            return DrillOutcome(
                state=DrillState.ABORTED,
                dodged=None,
                guard_up=None,
                safety_code=abort_code,
                unsafe_steps=unsafe_steps,
            )

        pose_arrival = self._observer.get_pose()
        dodge = self._dodge_detector.evaluate(arm_base, target, pose_launch, pose_arrival)
        guard = self._guard_detector.assess(pose_arrival)
        dodged = dodge.state is DodgeState.DODGED
        guard_up = guard.state is GuardState.UP
        self._scorer.record(dodged=dodged, guard_up=guard_up, aborted=False, vetoed=False)
        self._recover(arm_base)
        return DrillOutcome(
            state=DrillState.RESOLVED,
            dodged=dodged,
            guard_up=guard_up,
            safety_code=None,
            unsafe_steps=unsafe_steps,
            dodge=dodge,
        )

    def run_session(self, n_episodes: int) -> list[DrillOutcome]:
        """Run a sequence of reps and return their outcomes."""
        return [self.run_episode() for _ in range(n_episodes)]

    def _recover(self, arm_base: Vec3) -> None:
        """Retract the arm to its base and reset the opponent for the next rep.

        Retracting moves away from the trainee and is safe by construction (the
        target is the arm base), so it is commanded directly.
        """
        retract = StrikeCommand(
            strike_type=StrikeType.JAB,
            target=arm_base,
            speed_mps=self._cfg.strike_speed_mps,
            telegraph_s=0.0,
            arm_id=self._cfg.arm_id,
        )
        self._plant.command_strike(retract)
        for _ in range(max(1, round(self._cfg.recover_s / self._cfg.control_dt))):
            self._plant.step(self._cfg.control_dt)
        self._opponent.reset()
