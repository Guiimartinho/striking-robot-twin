"""SimGTObserver: the sim-side ITraineeObserver, ground truth from MuJoCo.

Reads the trainee keypoint sites straight out of the simulation, so pose is
exact and confidence is 1.0. The real observer estimates the same keypoints from
a camera with real noise and latency; degradation is added on top by the fault
harness, which is how Phase 0 stresses the SafetyArbiter without any camera.

Site naming convention: each Keypoint maps to a site named ``kp_<name>`` in
lower case, e.g. ``Keypoint.L_SHOULDER`` -> ``kp_l_shoulder``. This keeps the
MJCF and the enum in lockstep with no hand-maintained table.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robot_twin.core.types import NUM_KEYPOINTS, Keypoint, TraineePose


class SimGTObserver:
    """Ground-truth trainee observer backed by a MuJoCo model/data pair."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        nominal_latency_s: float = 0.0,
    ) -> None:
        """Resolve the keypoint site ids once.

        Args:
            model: The compiled MuJoCo model (shared with the plant).
            data: The MuJoCo data being stepped by the plant.
            nominal_latency_s: Baseline pipeline latency to report. Ground truth
                is effectively instantaneous, but a nonzero baseline lets the
                keep-out margin be exercised without a fault wrapper.

        Raises:
            RuntimeError: If any expected keypoint site is missing from the model.
        """
        self._model = model
        self._data = data
        self._latency = nominal_latency_s

        self._site_ids = np.empty(NUM_KEYPOINTS, dtype=np.int32)
        for kp in Keypoint:
            site_name = f"kp_{kp.name.lower()}"
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise RuntimeError(f"keypoint site '{site_name}' not found in model")
            self._site_ids[int(kp)] = sid

    def get_pose(self) -> TraineePose:
        """Current keypoint positions, all at full confidence."""
        positions = np.array(
            [self._data.site_xpos[int(sid)] for sid in self._site_ids],
            dtype=np.float64,
        )
        confidence = np.ones(NUM_KEYPOINTS, dtype=np.float64)
        return TraineePose(
            positions=positions,
            confidence=confidence,
            timestamp_s=float(self._data.time),
        )

    def latency_s(self) -> float:
        """Reported pipeline latency, seconds (the configured baseline)."""
        return self._latency
