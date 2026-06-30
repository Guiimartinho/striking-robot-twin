"""GuardDetector: assess the trainee's guard from keypoints. Phase 1.

Reads hand and head keypoints to judge whether the guard is up. Must stay
reliable under injected pose noise (the Phase 1 gate).
"""

from __future__ import annotations

from robot_twin.core.types import TraineePose


def is_guard_up(pose: TraineePose) -> bool:  # noqa: D103
    raise NotImplementedError("GuardDetector is Phase 1 (perception and detection)")
