"""robot_twin: digital-twin-first control stack for a striking training robot.

The package is layered so that everything above the HAL (Domain, Safety,
Services, RL) depends only on ``hal.interfaces`` and ``core.types`` and never on
a concrete plant. See CLAUDE.md for the architecture and the safety contract.
"""

__version__ = "0.0.0"
