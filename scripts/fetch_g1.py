"""Fetch the Unitree G1 model and a sample fight mocap clip (not vendored in git).

The G1 meshes are ~35 MB, so they are not committed. This script sparse-clones
the official model from MuJoCo Menagerie (BSD-3) into assets/unitree_g1, copies
our boxing/mocap scenes next to it (so the model's meshdir resolves), and
downloads one retargeted LAFAN1 fight clip into data/mocap.

    python scripts/fetch_g1.py

Re-runnable: it skips work that is already done. Requires git and curl.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEST = _ROOT / "assets" / "unitree_g1"
_MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"
_SCENES = ("boxer_g1.xml", "mocap_g1.xml")
_CLIP = "fight1_subject3.csv"
_CLIP_URL = (
    f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{_CLIP}"
)


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_model() -> None:
    """Sparse-clone just the unitree_g1 folder from Menagerie into assets/."""
    if (_DEST / "g1.xml").exists():
        print(f"G1 model already present at {_DEST}")
        return
    tmp = _ROOT / "assets" / "_menagerie_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print("cloning unitree_g1 from MuJoCo Menagerie ...")
    _run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", _MENAGERIE, str(tmp)],
        cwd=_ROOT,
    )
    _run(["git", "sparse-checkout", "set", "unitree_g1"], cwd=tmp)
    shutil.copytree(tmp / "unitree_g1", _DEST)
    shutil.rmtree(tmp)
    print(f"G1 model placed at {_DEST}")


def place_scenes() -> None:
    """Copy our scenes next to g1.xml so its relative meshdir resolves."""
    for name in _SCENES:
        src = _ROOT / "models" / name
        shutil.copyfile(src, _DEST / name)
    print(f"copied scenes {_SCENES} into {_DEST}")


def fetch_clip() -> None:
    """Download one sample fight mocap clip into data/mocap."""
    out = _ROOT / "data" / "mocap" / _CLIP
    if out.exists():
        print(f"clip already present at {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {_CLIP} ...")
    urllib.request.urlretrieve(_CLIP_URL, out)
    print(f"clip saved to {out}")


def main() -> int:
    try:
        fetch_model()
        place_scenes()
        fetch_clip()
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print("done. try: python scripts/render_mocap.py data/mocap/fight1_subject3.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
