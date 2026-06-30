"""Fault injection over the observer pipeline. Identical in sim and real.

These wrappers take any ``ITraineeObserver`` and return one that conforms to the
same Protocol while degrading its output in a specific, controllable way. They
are the adversary the SafetyArbiter must survive: the Phase 0 gate is that the
arbiter rejects 100% of unsafe commands under these faults.

Faults compose by nesting: ``LatencyFault(DropoutFault(inner, ...), ...)``.

Determinism: ``NoiseFault`` takes an explicit ``np.random.Generator`` so runs
are reproducible; there is no hidden global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_twin.core.types import Keypoint, TraineePose, Vec3
from robot_twin.hal.interfaces import ITraineeObserver


class FaultMode(Enum):
    """Named fault families, for parametrising tests and telemetry tags."""

    LATENCY = "latency"
    DROPOUT = "dropout"
    NOISE = "noise"
    LUNGE = "lunge"


@dataclass
class LatencyFault:
    """Inflate reported latency. Simulates a loaded or degraded pipeline.

    The pose passes through unchanged; only ``latency_s`` grows. This is the
    cleanest probe of the latency-inflated keep-out: more latency must widen
    ``R_keepout`` and turn a once-safe target into a rejected one.
    """

    inner: ITraineeObserver
    extra_latency_s: float

    def get_pose(self) -> TraineePose:
        return self.inner.get_pose()

    def latency_s(self) -> float:
        return self.inner.latency_s() + self.extra_latency_s


@dataclass
class DropoutFault:
    """Drop keypoints: zero their confidence and blank their position to NaN.

    Models the camera losing a joint (occlusion, motion blur). A dropped
    protected keypoint is unknown, so the arbiter must fail safe and reject.
    """

    inner: ITraineeObserver
    dropped: tuple[Keypoint, ...]

    def get_pose(self) -> TraineePose:
        pose = self.inner.get_pose()
        positions = pose.positions.copy()
        confidence = pose.confidence.copy()
        for kp in self.dropped:
            positions[int(kp)] = np.nan
            confidence[int(kp)] = 0.0
        return TraineePose(
            positions=positions, confidence=confidence, timestamp_s=pose.timestamp_s
        )

    def latency_s(self) -> float:
        return self.inner.latency_s()


@dataclass
class NoiseFault:
    """Add zero-mean Gaussian position noise to every keypoint.

    Models estimator jitter. The tracking-error budget in the keep-out radius is
    meant to absorb exactly this; the test asserts the arbiter still never lets a
    truly unsafe command through.
    """

    inner: ITraineeObserver
    sigma_m: float
    rng: np.random.Generator

    def get_pose(self) -> TraineePose:
        pose = self.inner.get_pose()
        noise = self.rng.normal(0.0, self.sigma_m, size=pose.positions.shape)
        return TraineePose(
            positions=pose.positions + noise,
            confidence=pose.confidence,
            timestamp_s=pose.timestamp_s,
        )

    def latency_s(self) -> float:
        return self.inner.latency_s()


@dataclass
class LungeFault:
    """Shift chosen keypoints by a fixed delta: the trainee steps INTO the strike.

    The nastiest case for a striking robot: the human moves toward the incoming
    arm faster than expected, so a trajectory that cleared the head a moment ago
    now intersects it. The arbiter must catch this via the inflated radius.
    """

    inner: ITraineeObserver
    delta: Vec3
    keypoints: tuple[Keypoint, ...] = (Keypoint.HEAD, Keypoint.NECK)

    def get_pose(self) -> TraineePose:
        pose = self.inner.get_pose()
        positions = pose.positions.copy()
        d = self.delta.as_array()
        for kp in self.keypoints:
            positions[int(kp)] = positions[int(kp)] + d
        return TraineePose(
            positions=positions, confidence=pose.confidence, timestamp_s=pose.timestamp_s
        )

    def latency_s(self) -> float:
        return self.inner.latency_s()
