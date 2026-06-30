"""Core value types shared across every layer of the twin.

These types are the lingua franca that crosses the HAL boundary. They must stay
free of any plant detail (no mujoco, no jax, no cv2) so that Domain, Safety and
Services can depend on them without ever learning whether they run in the sim or
on the real robot.

Design notes (the "why"):
- Positions are expressed in a single world frame, metres, right-handed.
- `Vec3` is a tiny frozen dataclass instead of a bare ``np.ndarray`` so that
  single points (a target, an arm base) read with intent at call sites while
  still converting cheaply to arrays for vectorised math.
- Bulk keypoint math uses ``FloatArray`` (shape ``(K, 3)``) because the hot path
  in the SafetyArbiter is vectorised over all protected keypoints at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Vec3:
    """An immutable 3D point or vector in the world frame, metres.

    Frozen so a target or arm base can be passed around without any layer
    mutating shared state behind another layer's back.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> FloatArray:
        """Return a fresh ``(3,)`` array for vectorised math."""
        return np.array((self.x, self.y, self.z), dtype=np.float64)

    @classmethod
    def from_array(cls, a: npt.ArrayLike) -> "Vec3":
        """Build a Vec3 from any 3-element array-like."""
        arr = np.asarray(a, dtype=np.float64).reshape(3)
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def norm(self) -> float:
        """Euclidean length."""
        return float(np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def distance_to(self, other: "Vec3") -> float:
        """Euclidean distance to another point."""
        return (self - other).norm()


class Keypoint(IntEnum):
    """Trainee body keypoints tracked for the punches-only MVP (upper body).

    The integer value doubles as the row index into ``TraineePose.positions``,
    so the order here is load-bearing: do not reorder without migrating poses.
    Lower body is intentionally absent: the MVP does not actuate or react to it.
    """

    HEAD = 0
    NECK = 1
    CHEST = 2
    L_SHOULDER = 3
    R_SHOULDER = 4
    L_ELBOW = 5
    R_ELBOW = 6
    L_HAND = 7
    R_HAND = 8


NUM_KEYPOINTS: int = len(Keypoint)


class StrikeType(Enum):
    """Punch families in scope for the MVP. No elbows, knees or kicks."""

    JAB = "jab"
    CROSS = "cross"
    HOOK = "hook"


@dataclass(frozen=True, slots=True)
class JointState:
    """Per-joint plant state read back through the HAL.

    ``force`` is sensed from Series Elastic Actuator spring deflection, not from
    a dedicated load cell: the spring is both the passive energy limiter and the
    force sensor. All three arrays are indexed by joint and must share length.
    """

    position: FloatArray  # rad or m, per joint
    velocity: FloatArray  # rad/s or m/s, per joint
    force: FloatArray  # N or N*m, derived from SEA spring deflection

    def __post_init__(self) -> None:
        n = self.position.shape[0]
        if self.velocity.shape[0] != n or self.force.shape[0] != n:
            raise ValueError(
                "JointState arrays must share length: "
                f"pos={self.position.shape}, vel={self.velocity.shape}, "
                f"force={self.force.shape}"
            )

    @property
    def n_joints(self) -> int:
        return int(self.position.shape[0])


@dataclass(frozen=True, slots=True)
class TraineePose:
    """A snapshot of the trainee's keypoints.

    Ground truth in the sim, a camera estimate on the real robot. ``confidence``
    lets Safety treat a dropped or low-confidence keypoint as unknown and fail
    safe instead of trusting a hallucinated position. ``timestamp_s`` is the
    plant clock at capture, used downstream to reason about staleness.
    """

    positions: FloatArray  # shape (NUM_KEYPOINTS, 3), world frame, metres
    confidence: FloatArray  # shape (NUM_KEYPOINTS,), in [0, 1]
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.positions.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(
                f"positions must be ({NUM_KEYPOINTS}, 3), got {self.positions.shape}"
            )
        if self.confidence.shape != (NUM_KEYPOINTS,):
            raise ValueError(
                f"confidence must be ({NUM_KEYPOINTS},), got {self.confidence.shape}"
            )

    def keypoint(self, kp: Keypoint) -> Vec3:
        """Position of a single keypoint as a Vec3."""
        return Vec3.from_array(self.positions[int(kp)])


@dataclass(frozen=True, slots=True)
class StrikeCommand:
    """A high-level strike request crossing the HAL toward the plant.

    The planner expresses intent (which punch, where to aim, how fast, how much
    to telegraph) and the plant turns it into actuator setpoints. For a strike
    at head height ``target`` is the offset point BEHIND the head, never the head
    itself; the SafetyArbiter is the independent check that this holds.
    """

    strike_type: StrikeType
    target: Vec3  # aim point in the world frame, metres
    speed_mps: float  # commanded end-effector speed
    telegraph_s: float  # wind-up time so the trainee can read and dodge
    arm_id: int  # which striker arm executes the command
    issued_at_s: float = 0.0  # plant clock when the command was issued

    def __post_init__(self) -> None:
        if self.speed_mps <= 0.0:
            raise ValueError(f"speed_mps must be > 0, got {self.speed_mps}")
        if self.telegraph_s < 0.0:
            raise ValueError(f"telegraph_s must be >= 0, got {self.telegraph_s}")
        if self.arm_id < 0:
            raise ValueError(f"arm_id must be >= 0, got {self.arm_id}")
