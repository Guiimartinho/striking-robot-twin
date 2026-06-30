"""Tests for the curriculum: advances on dodging, holds otherwise."""

from __future__ import annotations

from robot_twin.services.curriculum import Curriculum, CurriculumLevel


class TestCurriculum:
    def test_starts_at_first_level(self) -> None:
        assert Curriculum().current().index == 0

    def test_advances_after_a_strong_window(self) -> None:
        cur = Curriculum(advance_dodge_rate=0.7, window=5)
        for _ in range(5):
            cur.record(True)
        assert cur.current().index == 1

    def test_holds_when_dodging_poorly(self) -> None:
        cur = Curriculum(advance_dodge_rate=0.7, window=5)
        for dodged in (True, False, False, True, False):
            cur.record(dodged)
        assert cur.current().index == 0

    def test_safety_events_do_not_count(self) -> None:
        cur = Curriculum(advance_dodge_rate=0.7, window=4)
        # None (aborted/vetoed) reps never fill the window or advance difficulty.
        for _ in range(20):
            cur.record(None)
        assert cur.current().index == 0

    def test_does_not_advance_past_last_level(self) -> None:
        levels = (
            CurriculumLevel(0, "a", 1.0, 1.0, 0.0),
            CurriculumLevel(1, "b", 0.5, 1.5, 0.2),
        )
        cur = Curriculum(levels=levels, advance_dodge_rate=0.5, window=2)
        for _ in range(20):
            cur.record(True)
        assert cur.current().index == 1  # clamped at the last level

    def test_harder_levels_telegraph_faster(self) -> None:
        levels = Curriculum().current()
        # The default level 0 (warmup) telegraphs slower than 1x.
        assert levels.telegraph_scale >= 1.0
