"""Render the Unitree G1 humanoid throwing a combo you choose, as a GIF.

    python scripts/render_combo_g1.py jab,jab,cross,hook
    python scripts/render_combo_g1.py cross,hook,jab

Uses the official Unitree G1 model (MuJoCo Menagerie, BSD-3) standing in front of
a heavy bag. The base is held at the "stand" keyframe and only the arm joints are
animated (kinematic), so the stationary striker does not fall. Punches are
authored keyframes on the real 7-DoF arms: jab and lead hook on the left arm,
cross on the right. Requires mujoco and pillow.

This is the visual body. The safety/drill logic above the HAL is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "assets" / "unitree_g1" / "boxer_g1.xml"  # placed by scripts/fetch_g1.py
_ARM_JOINTS = [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
]


def _pose(**overrides: float) -> dict[str, float]:
    return {**_GUARD, **overrides}


# Guard plus the held end pose of each punch, on the G1 arm joints (radians).
# Tuned so jab/cross reach the bag and the hook arcs across.
_GUARD = {
    "left_shoulder_pitch_joint": -0.5,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.3,
    "left_elbow_joint": 2.0,
    "right_shoulder_pitch_joint": -0.5,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": -0.3,
    "right_elbow_joint": 2.0,
}

_PUNCHES: dict[str, list[tuple[dict[str, float], int]]] = {
    "jab": [
        (
            _pose(
                left_shoulder_pitch_joint=-0.95,
                left_shoulder_roll_joint=0.05,
                left_shoulder_yaw_joint=0.15,
                left_elbow_joint=0.0,
            ),
            5,
        ),
        (_GUARD, 6),
    ],
    "cross": [
        (
            _pose(
                right_shoulder_pitch_joint=-0.95,
                right_shoulder_roll_joint=-0.05,
                right_shoulder_yaw_joint=-0.15,
                right_elbow_joint=0.0,
            ),
            5,
        ),
        (_GUARD, 6),
    ],
    "hook": [
        (
            _pose(
                left_shoulder_pitch_joint=-1.3,
                left_shoulder_roll_joint=0.1,
                left_shoulder_yaw_joint=0.2,
                left_elbow_joint=1.4,
            ),
            4,
        ),
        (
            _pose(
                left_shoulder_pitch_joint=-1.15,
                left_shoulder_roll_joint=0.1,
                left_shoulder_yaw_joint=1.0,
                left_elbow_joint=1.1,
            ),
            4,
        ),
        (_GUARD, 7),
    ],
}


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def build_timeline(punches: list[str]) -> list[tuple[dict[str, float], int]]:
    timeline: list[tuple[dict[str, float], int]] = [(_GUARD, 6)]
    for name in punches:
        if name not in _PUNCHES:
            raise ValueError(f"unknown punch '{name}'; choose from {sorted(_PUNCHES)}")
        timeline.extend(_PUNCHES[name])
    timeline.append((_GUARD, 8))
    return timeline


def render(punches: list[str], out_path: Path, duration_ms: int = 55) -> int:
    try:
        import mujoco
        from PIL import Image
    except ImportError:
        print("needs mujoco and pillow: uv pip install mujoco pillow", file=sys.stderr)
        return 1
    if not _MODEL.exists():
        print(f"model not found: {_MODEL}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(_MODEL))
    data = mujoco.MjData(model)
    stand = model.key_qpos[0].copy()
    adr = {
        j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in _ARM_JOINTS
    }

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.18, 0.0, 1.05]
    cam.distance, cam.azimuth, cam.elevation = 2.6, -60, -8

    def to_vec(pose: dict[str, float]) -> np.ndarray:
        return np.array([pose[j] for j in _ARM_JOINTS], dtype=np.float64)

    timeline = build_timeline(punches)
    frames = []
    current = to_vec(_GUARD)
    for target_pose, n_frames in timeline:
        target = to_vec(target_pose)
        for k in range(1, n_frames + 1):
            q = current + (target - current) * _smoothstep(k / n_frames)
            data.qpos[:] = stand
            for i, j in enumerate(_ARM_JOINTS):
                data.qpos[adr[j]] = q[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            frames.append(Image.fromarray(renderer.render()))
        current = target

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"saved {len(frames)} frames -> {out_path}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Render the Unitree G1 throwing a combo.")
    parser.add_argument("combo", nargs="?", default="jab,jab,cross,hook")
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args(argv[1:])
    punches = [p.strip().lower() for p in args.combo.split(",") if p.strip()]
    out = args.out or _ROOT / "renders" / f"g1_combo_{'_'.join(punches)}.gif"
    return render(punches, out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
