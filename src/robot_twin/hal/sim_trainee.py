"""ScriptedTrainee: a sim-only model of the student's dodging. Phase 2.

The drill loop needs an opponent to validate the closed loop before any human is
involved. This object is both the observer the engine reads and the opponent it
drives: it shifts the observed keypoints kinematically to simulate a slip (a
lateral dodge) or a lunge into the strike. On the real robot there is no such
object: the human moves on their own and is seen through a camera, so the real
opponent is the no-op ``PassiveOpponent`` paired with a camera observer.

Kinematic, not physical: the pose is offset directly rather than driven through
contact dynamics. That is enough to validate the drill FSM, the dodge detection
and, crucially, that the SafetyArbiter aborts when the trainee lunges in. Full
physical articulation of the trainee is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robot_twin.core.types import TraineePose, Vec3
from robot_twin.hal.interfaces import ITraineeObserver


class DodgePolicy(Enum):
    """How the scripted trainee responds to an incoming strike."""

    ALWAYS_DODGE = "always_dodge"  # slip laterally clear of the strike line
    NEVER_DODGE = "never_dodge"  # stand still (a clean "hit" tag, no movement)
    LUNGE = "lunge"  # move into the strike, to exercise the safety abort


@dataclass(frozen=True, slots=True)
class TraineeConfig:
    """Motion parameters for the scripted trainee, in SI units."""

    dodge_distance_m: float = 0.30  # lateral slip distance
    reaction_delay_s: float = 0.10  # delay before the trainee starts moving
    move_speed_mps: float = 2.0  # how fast the body shifts once moving
    dodge_sign: float = 1.0  # +1 slips toward +x, -1 toward -x
    # A duck-in vector that brings the head and neck toward a body-height strike
    # line, used only by the LUNGE policy to provoke the arbiter.
    lunge_offset: Vec3 = Vec3(0.0, -0.05, -0.55)


class ScriptedTrainee:
    """Simulated student: an ITraineeObserver and an IOpponent in one.

    Wraps a base observer (ground truth in the sim) and adds a scripted whole-body
    offset that plays out over time after a strike is signalled.
    """

    def __init__(
        self,
        base: ITraineeObserver,
        policy: DodgePolicy = DodgePolicy.ALWAYS_DODGE,
        config: TraineeConfig | None = None,
    ) -> None:
        self._base = base
        self._policy = policy
        self._cfg = config or TraineeConfig()
        self._offset = Vec3(0.0, 0.0, 0.0)  # current applied offset
        self._target_offset = Vec3(0.0, 0.0, 0.0)  # where the offset is heading
        self._clock_s = 0.0  # time since the last react_to_strike
        self._reacting = False

    # -- IOpponent ------------------------------------------------------------
    def react_to_strike(self, target: Vec3) -> None:
        """Choose a response to the incoming strike and start the reaction clock."""
        self._clock_s = 0.0
        self._reacting = True
        if self._policy is DodgePolicy.ALWAYS_DODGE:
            # Slip laterally AWAY from the incoming arm (target.x is the arm's
            # plane), so each punch in a combo gets its own clean dodge.
            if target.x > 0.0:
                direction = -1.0
            elif target.x < 0.0:
                direction = 1.0
            else:
                direction = self._cfg.dodge_sign
            self._target_offset = Vec3(direction * self._cfg.dodge_distance_m, 0.0, 0.0)
        elif self._policy is DodgePolicy.LUNGE:
            self._target_offset = self._cfg.lunge_offset
        else:  # NEVER_DODGE
            self._target_offset = Vec3(0.0, 0.0, 0.0)

    def advance(self, dt: float) -> None:
        """Advance the body offset toward the target after the reaction delay."""
        self._clock_s += dt
        if not self._reacting or self._clock_s < self._cfg.reaction_delay_s:
            return
        step = self._cfg.move_speed_mps * dt
        self._offset = _move_toward(self._offset, self._target_offset, step)

    def reset(self) -> None:
        """Return to neutral stance, ready for the next rep."""
        self._offset = Vec3(0.0, 0.0, 0.0)
        self._target_offset = Vec3(0.0, 0.0, 0.0)
        self._clock_s = 0.0
        self._reacting = False

    # -- ITraineeObserver -----------------------------------------------------
    def get_pose(self) -> TraineePose:
        """Base pose with the current whole-body offset applied to every keypoint."""
        pose = self._base.get_pose()
        shifted = pose.positions + self._offset.as_array()
        return TraineePose(
            positions=shifted, confidence=pose.confidence, timestamp_s=pose.timestamp_s
        )

    def latency_s(self) -> float:
        return self._base.latency_s()

    # -- introspection (sim/eval only) ----------------------------------------
    @property
    def offset(self) -> Vec3:
        return self._offset


class PassiveOpponent:
    """The real-robot opponent: a no-op. The human moves on their own."""

    def react_to_strike(self, target: Vec3) -> None:  # noqa: D102
        return None

    def advance(self, dt: float) -> None:  # noqa: D102
        return None

    def reset(self) -> None:  # noqa: D102
        return None


def _move_toward(current: Vec3, goal: Vec3, max_step: float) -> Vec3:
    """Move ``current`` toward ``goal`` by at most ``max_step`` metres."""
    delta = goal - current
    dist = delta.norm()
    if dist <= max_step or dist == 0.0:
        return goal
    scale = max_step / dist
    return current + Vec3(delta.x * scale, delta.y * scale, delta.z * scale)
