"""Combo statistics from strike events. Phase 3.

Aggregates strike events into combo grammar (jab->cross->hook), cadence, firing
distance and telegraph timing distributions. These seed the DrillEngine FSM and
serve as the reward reference for the RL policy. Only high-level structure
transfers from video; joint trajectories are learned in the sim.
"""

from __future__ import annotations


def build_combo_distributions() -> None:  # noqa: D103
    raise NotImplementedError("Combo statistics are Phase 3 (video mining)")
