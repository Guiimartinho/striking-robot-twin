"""Pose extraction from fight clips: 2D pose + 2D->3D lifting. Phase 3.

Pipeline plan (CLAUDE.md section 5): 2D pose per frame, then a lifter
(MotionBERT or VideoPose3D) to 3D. Output feeds strike-event detection. This
trains the robot's OFFENSIVE policy only, never trainee perception.
"""

from __future__ import annotations

from pathlib import Path


def extract_poses(clip_path: Path) -> None:  # noqa: D103
    raise NotImplementedError("Pose extraction is Phase 3 (video mining)")
