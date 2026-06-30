"""Tests for the Scorer: counts, score and dodge rate exclude safety events."""

from __future__ import annotations

from robot_twin.services.scoring import Scorer


class TestScorer:
    def test_counts_and_score(self) -> None:
        scorer = Scorer(dodge_points=1.0, guard_points=0.5, hit_penalty=0.0)
        scorer.record(dodged=True, guard_up=True, aborted=False, vetoed=False)
        scorer.record(dodged=False, guard_up=True, aborted=False, vetoed=False)
        scorer.record(dodged=None, guard_up=None, aborted=True, vetoed=False)
        scorer.record(dodged=None, guard_up=None, aborted=False, vetoed=True)

        s = scorer.summary()
        assert s.episodes == 4
        assert s.resolved == 2
        assert s.dodges == 1
        assert s.hits == 1
        assert s.guards_up == 2
        assert s.aborts == 1
        assert s.vetoes == 1
        # 1 dodge (1.0) + 2 guards (0.5 each) = 2.0
        assert s.score == 2.0

    def test_dodge_rate_excludes_safety_events(self) -> None:
        scorer = Scorer()
        # 1 dodge out of 2 delivered; the abort and veto are not counted.
        scorer.record(dodged=True, guard_up=False, aborted=False, vetoed=False)
        scorer.record(dodged=False, guard_up=False, aborted=False, vetoed=False)
        scorer.record(dodged=None, guard_up=None, aborted=True, vetoed=False)
        assert scorer.summary().dodge_rate == 0.5

    def test_empty_session_has_zero_dodge_rate(self) -> None:
        assert Scorer().summary().dodge_rate == 0.0
