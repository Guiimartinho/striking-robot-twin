"""Reward shaping for the combo policy.

Phase 0 ships a deliberately simple, transparent reward so the env runs and the
safety coupling is visible: a vetoed strike is penalised, a safe strike that
lands near the intended target is rewarded. The real reward (Phase 3) will score
combo timing and cadence against the video-mined distributions and is kept out
of Phase 0 on purpose, to avoid baking in numbers before the loop is validated.

The safety penalty being part of the reward is intentional: the policy learns to
respect the SafetyArbiter, but the arbiter remains the hard guarantee regardless
of what the policy learns.
"""

from __future__ import annotations

from robot_twin.core.result import ErrorCode

# Phase 0 placeholder weights. Tunable; do not read meaning into the exact values.
_VETO_PENALTY: float = -1.0
_SAFE_STRIKE_BONUS: float = 0.1
_ON_TARGET_SCALE: float = 1.0


def compute_reward(veto_code: ErrorCode, target_error_m: float) -> float:
    """Reward for one drill step.

    Args:
        veto_code: ``ErrorCode.OK`` if the strike was allowed, otherwise the veto
            reason. Any veto yields the penalty.
        target_error_m: Distance between where the glove ended and the intended
            target, metres. Smaller is better. Ignored when vetoed.

    Returns:
        The scalar step reward.
    """
    if veto_code is not ErrorCode.OK:
        return _VETO_PENALTY
    return _SAFE_STRIKE_BONUS + _ON_TARGET_SCALE * max(0.0, 1.0 - target_error_m)
