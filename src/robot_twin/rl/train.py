"""Training entry point for the combo policy. Phase 3.

Phase 0 ships the wiring point only. The plan (CLAUDE.md section 5): start with
Stable-Baselines3 PPO/SAC on a single env for a fast result, then scale to PPO in
JAX over thousands of MJX envs on Linux/WSL2 with GPU. The env is already
HAL-clean, so whatever trains here transfers to the real robot unchanged.

Run with: python -m robot_twin.rl.train
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Placeholder trainer.

    Intentionally not implemented in Phase 0: training is gated behind the
    perception and closed-loop phases. Building it now would optimise a loop that
    has not yet been validated for safety.
    """
    raise NotImplementedError(
        "RL training is Phase 3. Phase 0 validates safety and the closed loop first."
    )


if __name__ == "__main__":
    main()
