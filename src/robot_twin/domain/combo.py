"""Combo grammar: what to throw and when. Phase 3.

An authored distribution of canonical punch combos (jab, jab-cross,
jab-cross-hook, ...) with weights and base cadence. This is the deterministic,
debuggable, safety-arguable seed the project calls for; video mining later
replaces these authored weights and timings with measured ones, behind the same
``ComboGrammar`` interface, so nothing downstream changes.

Only high-level structure lives here (which punch, which arm, rough timing). The
articular trajectory is learned in the sim, never copied from video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_twin.core.types import StrikeType

# Default arm for each punch family: jab and lead hook on the lead arm (0),
# cross on the rear arm (1). Clamped to the available arms at build time.
_ARM_FOR_STRIKE: dict[StrikeType, int] = {
    StrikeType.JAB: 0,
    StrikeType.CROSS: 1,
    StrikeType.HOOK: 0,
}


@dataclass(frozen=True, slots=True)
class StrikeSpec:
    """One punch within a combo: which arm, which punch, and its base timing."""

    arm_id: int
    strike_type: StrikeType
    telegraph_s: float  # base wind-up; the curriculum scales this at execution
    gap_before_s: float  # base pause before this punch within the combo


@dataclass(frozen=True, slots=True)
class Combo:
    """An ordered sequence of punches thrown as one rep."""

    name: str
    strikes: tuple[StrikeSpec, ...]


@dataclass(frozen=True, slots=True)
class ComboTemplate:
    """A named punch sequence with a sampling weight."""

    name: str
    sequence: tuple[StrikeType, ...]
    weight: float


# The authored grammar. Weights are rough boxing priors, not measured; video
# mining will replace them. jab-led combos dominate, as on the pads.
DEFAULT_TEMPLATES: tuple[ComboTemplate, ...] = (
    ComboTemplate("jab", (StrikeType.JAB,), 3.0),
    ComboTemplate("double-jab", (StrikeType.JAB, StrikeType.JAB), 1.0),
    ComboTemplate("jab-cross", (StrikeType.JAB, StrikeType.CROSS), 3.0),
    ComboTemplate("jab-cross-hook", (StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK), 2.0),
    ComboTemplate("cross-hook", (StrikeType.CROSS, StrikeType.HOOK), 1.0),
)


@dataclass(frozen=True, slots=True)
class GrammarConfig:
    """Base cadence applied when materialising a template into strikes."""

    base_telegraph_s: float = 0.4  # slow, readable wind-up (Phase 2/3 slow loop)
    base_gap_s: float = 0.15  # pause between punches within a combo


class ComboGrammar:
    """Samples combos from a weighted set of templates."""

    def __init__(
        self,
        n_arms: int,
        templates: tuple[ComboTemplate, ...] = DEFAULT_TEMPLATES,
        config: GrammarConfig | None = None,
    ) -> None:
        if not templates:
            raise ValueError("at least one combo template is required")
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self._n_arms = n_arms
        self._templates = templates
        self._cfg = config or GrammarConfig()
        total = sum(t.weight for t in templates)
        if total <= 0.0:
            raise ValueError("combo template weights must sum to a positive value")
        self._probs = np.array([t.weight / total for t in templates], dtype=np.float64)

    def _arm_for(self, strike_type: StrikeType) -> int:
        """Arm assigned to a punch, clamped to the arms this plant actually has."""
        return _ARM_FOR_STRIKE[strike_type] % self._n_arms

    def materialise(self, template: ComboTemplate) -> Combo:
        """Turn a template into a concrete Combo with arms and base timing."""
        strikes = tuple(
            StrikeSpec(
                arm_id=self._arm_for(st),
                strike_type=st,
                telegraph_s=self._cfg.base_telegraph_s,
                gap_before_s=0.0 if i == 0 else self._cfg.base_gap_s,
            )
            for i, st in enumerate(template.sequence)
        )
        return Combo(name=template.name, strikes=strikes)

    def sample(self, rng: np.random.Generator) -> Combo:
        """Sample a combo from the weighted templates."""
        idx = int(rng.choice(len(self._templates), p=self._probs))
        return self.materialise(self._templates[idx])
