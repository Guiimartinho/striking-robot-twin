"""Play a retargeted motion-capture clip on the Unitree G1 and render it to a GIF.

The clip is a LAFAN1 motion retargeted to the G1 (Unitree / lvhaidong dataset),
a CSV with one row per frame: root position (3) + root quaternion in (x,y,z,w)
order (4), then the 29 G1 joint angles. The G1 in that dataset is exactly the
MuJoCo Menagerie G1, so columns map to our model's qpos by joint name; only the
quaternion is reordered to MuJoCo's (w,x,y,z).

    python scripts/render_mocap.py data/mocap/fight1_subject3.csv
    python scripts/render_mocap.py data/mocap/fight1_subject3.csv --start 600 --frames 300

This is the visual body: real fight motion on the real robot. The safety/drill
logic above the HAL is unchanged. Requires mujoco, numpy and pillow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "assets" / "unitree_g1" / "mocap_g1.xml"  # placed by scripts/fetch_g1.py

# The 29 G1 joints in the dataset's column order (after the 7 root columns).
_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # fmt: skip


def render(
    csv_path: Path,
    out_path: Path,
    start: int = 0,
    frames: int = 300,
    stride: int = 2,
    width: int = 960,
    height: int = 540,
) -> int:
    try:
        import mujoco
        from PIL import Image
    except ImportError:
        print("needs mujoco and pillow: uv pip install mujoco pillow", file=sys.stderr)
        return 1
    if not _MODEL.exists() or not csv_path.exists():
        print(f"missing model or clip: {_MODEL} / {csv_path}", file=sys.stderr)
        return 1

    motion = np.loadtxt(csv_path, delimiter=",")
    n_total = motion.shape[0]
    model = mujoco.MjModel.from_xml_path(str(_MODEL))
    data = mujoco.MjData(model)
    adr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in _JOINTS
    ]

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10

    end = min(n_total, start + frames)
    images = []
    for f in range(start, end, stride):
        row = motion[f]
        data.qpos[0:3] = row[0:3]
        # CSV quaternion is (x, y, z, w); MuJoCo wants (w, x, y, z).
        data.qpos[3:7] = [row[6], row[3], row[4], row[5]]
        for k, a in enumerate(adr):
            data.qpos[a] = row[7 + k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.95]  # track the pelvis
        renderer.update_scene(data, cam)
        images.append(Image.fromarray(renderer.render()))

    if not images:
        print("no frames rendered (check --start/--frames)", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 * stride / 30)  # source is 30 fps
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"saved {len(images)} frames ({start}..{end} stride {stride}) -> {out_path}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Play a retargeted mocap clip on the G1.")
    parser.add_argument("csv", type=Path, help="path to a retargeted G1 motion CSV")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args(argv[1:])
    out = args.out or _ROOT / "renders" / f"mocap_{args.csv.stem}.gif"
    return render(args.csv, out, args.start, args.frames, args.stride)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
