"""Curriculum: difficulty progression for the drill. Phase 3.

Scales the wind-up (telegraph), strike speed, and feint probability by level, and
advances to the next level once the trainee is dodging reliably over a recent
window. Easier levels telegraph slowly and never feint; harder levels telegraph
fast, hit faster, and mix in feints. Randomisation of the combo itself comes from
the ComboGrammar; the curriculum sets how hard each rep is delivered.

Difficulty never weakens safety: the curriculum tunes timing and feints, while
every strike still passes through the SafetyArbiter unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CurriculumLevel:
    """How hard a rep is delivered. Scales multiply the combo's base values."""

    index: int
    name: str
    telegraph_scale: float  # < 1 means a shorter, harder-to-read wind-up
    speed_scale: float  # > 1 means a faster strike
    feint_prob: float  # chance a given strike is a feint (telegraph, no follow-through)


DEFAULT_LEVELS: tuple[CurriculumLevel, ...] = (
    CurriculumLevel(0, "warmup", telegraph_scale=1.5, speed_scale=0.7, feint_prob=0.0),
    CurriculumLevel(1, "easy", telegraph_scale=1.0, speed_scale=1.0, feint_prob=0.0),
    CurriculumLevel(2, "medium", telegraph_scale=0.75, speed_scale=1.3, feint_prob=0.15),
    CurriculumLevel(3, "hard", telegraph_scale=0.5, speed_scale=1.6, feint_prob=0.30),
)


class Curriculum:
    """Tracks recent dodge performance and advances difficulty accordingly."""

    def __init__(
        self,
        levels: tuple[CurriculumLevel, ...] = DEFAULT_LEVELS,
        advance_dodge_rate: float = 0.7,
        window: int = 6,
    ) -> None:
        if not levels:
            raise ValueError("at least one curriculum level is required")
        if window < 1:
            raise ValueError("window must be >= 1")
        self._levels = levels
        self._advance_at = advance_dodge_rate
        self._window = window
        self._index = 0
        self._recent: deque[bool] = deque(maxlen=window)

    def current(self) -> CurriculumLevel:
        """The level reps are currently delivered at."""
        return self._levels[self._index]

    def record(self, dodged: bool | None) -> None:
        """Record a delivered rep's dodge outcome and maybe advance the level.

        Safety events (aborted/vetoed strikes) pass ``None`` and do not count
        toward progression: the trainee only advances by reading real strikes.
        """
        if dodged is None:
            return
        self._recent.append(dodged)
        if len(self._recent) < self._window:
            return
        dodge_rate = sum(self._recent) / len(self._recent)
        if dodge_rate >= self._advance_at and self._index < len(self._levels) - 1:
            self._index += 1
            self._recent.clear()
