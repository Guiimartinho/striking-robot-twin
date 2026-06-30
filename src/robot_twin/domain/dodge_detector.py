"""DodgeDetector: decide whether the trainee dodged in time. Phase 1.

Operates over the observer pose stream and must stay reliable under injected pose
noise (the Phase 1 gate). A failed dodge resolves to a non-impacting tag (a
logical haptic/buzzer/LED event), never a strike projected onto the head.
"""

from __future__ import annotations

from robot_twin.core.types import TraineePose


def detect_dodge(pose_before: TraineePose, pose_after: TraineePose) -> bool:  # noqa: D103
    raise NotImplementedError("DodgeDetector is Phase 1 (perception and detection)")
