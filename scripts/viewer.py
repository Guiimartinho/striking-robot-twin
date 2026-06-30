"""Open the MuJoCo interactive viewer on a model (default: the scene).

Usage:
    python scripts/viewer.py [models/scene.xml]

This is the Python passive viewer. For the standalone native viewer use the
simulate.exe shipped with the MuJoCo download (see README).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = Path(argv[1]) if len(argv) > 1 else repo_root / "models" / "scene.xml"
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("mujoco is not installed. Run: uv pip install mujoco", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    # Blocking call: opens the window and runs until the user closes it.
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
