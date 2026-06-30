"""Tests for the combo grammar: valid combos, arm clamping, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from robot_twin.core.types import StrikeType
from robot_twin.domain.combo import ComboGrammar, ComboTemplate, GrammarConfig


class TestComboGrammar:
    def test_sample_returns_nonempty_combo(self) -> None:
        grammar = ComboGrammar(n_arms=2)
        combo = grammar.sample(np.random.default_rng(0))
        assert len(combo.strikes) >= 1
        assert combo.name

    def test_arm_ids_clamped_to_available_arms(self) -> None:
        # With a single arm, every strike must map to arm 0.
        grammar = ComboGrammar(n_arms=1)
        rng = np.random.default_rng(1)
        for _ in range(50):
            combo = grammar.sample(rng)
            assert all(s.arm_id == 0 for s in combo.strikes)

    def test_two_arms_used_across_punch_types(self) -> None:
        grammar = ComboGrammar(n_arms=2)
        # jab-cross materialises to arm 0 then arm 1 (lead then rear).
        template = ComboTemplate("jc", (StrikeType.JAB, StrikeType.CROSS), 1.0)
        combo = grammar.materialise(template)
        assert combo.strikes[0].arm_id == 0
        assert combo.strikes[1].arm_id == 1

    def test_deterministic_with_seed(self) -> None:
        g1 = ComboGrammar(n_arms=2)
        g2 = ComboGrammar(n_arms=2)
        names1 = [g1.sample(np.random.default_rng(42)).name for _ in range(5)]
        names2 = [g2.sample(np.random.default_rng(42)).name for _ in range(5)]
        assert names1 == names2

    def test_first_strike_has_no_gap(self) -> None:
        grammar = ComboGrammar(n_arms=2, config=GrammarConfig(base_gap_s=0.2))
        combo = grammar.materialise(ComboTemplate("jc", (StrikeType.JAB, StrikeType.CROSS), 1.0))
        assert combo.strikes[0].gap_before_s == 0.0
        assert combo.strikes[1].gap_before_s == 0.2

    def test_rejects_empty_templates(self) -> None:
        with pytest.raises(ValueError):
            ComboGrammar(n_arms=2, templates=())

    def test_rejects_zero_arms(self) -> None:
        with pytest.raises(ValueError):
            ComboGrammar(n_arms=0)
