"""Application configuration: pydantic models, YAML-backed, env-driven.

The config is the single place where concrete numbers (geometry, safety limits,
timing) live, so they are reviewable and versioned rather than scattered as magic
numbers. Defaults are conservative and match the Phase 0 MJCF; a YAML file or the
``ROBOT_TWIN_CONFIG`` env var overrides them.

These pydantic models validate and then build the frozen domain dataclasses
(``SafetyConfig``, ``RobotGeometry``) consumed by the SafetyArbiter, keeping the
arbiter free of any config/IO dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from robot_twin.core.types import Keypoint, Vec3
from robot_twin.safety.arbiter import RobotGeometry, SafetyConfig

_DEFAULT_MODEL = Path("models/scene.xml")
_CONFIG_ENV_VAR = "ROBOT_TWIN_CONFIG"


class SafetySettings(BaseModel):
    """Serialisable mirror of SafetyConfig. See SafetyConfig for the why."""

    tracking_error_m: float = 0.05
    head_v_max_mps: float = 3.0
    actuator_stop_s: float = 0.05
    margin_m: float = 0.10
    max_latency_s: float = 0.20
    min_confidence: float = 0.5
    force_cap_n: float = 80.0
    protected_keypoints: list[str] = Field(default_factory=lambda: ["HEAD", "NECK"])

    def to_safety_config(self) -> SafetyConfig:
        """Build the frozen SafetyConfig the arbiter consumes."""
        protected = tuple(Keypoint[name] for name in self.protected_keypoints)
        return SafetyConfig(
            tracking_error_m=self.tracking_error_m,
            head_v_max_mps=self.head_v_max_mps,
            actuator_stop_s=self.actuator_stop_s,
            margin_m=self.margin_m,
            max_latency_s=self.max_latency_s,
            min_confidence=self.min_confidence,
            force_cap_n=self.force_cap_n,
            protected_keypoints=protected,
        )


class GeometrySettings(BaseModel):
    """Mechanical layout: arm bases and the software reach limit.

    ``reach_max_m`` is the software end-stop and must sit at or inside the MJCF
    joint range (the mechanical end-stop), never beyond it.
    """

    arm_bases: list[tuple[float, float, float]] = Field(
        default_factory=lambda: [(0.0, 0.0, 1.3)]
    )
    reach_max_m: float = 0.9

    def to_robot_geometry(self) -> RobotGeometry:
        """Build the frozen RobotGeometry the arbiter consumes."""
        bases = tuple(Vec3(*b) for b in self.arm_bases)
        return RobotGeometry(arm_bases=bases, reach_max_m=self.reach_max_m)


class AppConfig(BaseModel):
    """Top-level config for a sim run."""

    model_path: Path = _DEFAULT_MODEL
    control_dt: float = 0.02  # outer control loop period, seconds
    nominal_latency_s: float = 0.03  # baseline observer latency in the sim
    safety: SafetySettings = Field(default_factory=SafetySettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load and validate config from a YAML file."""
        import yaml  # local import: only this path needs PyYAML

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load from ``ROBOT_TWIN_CONFIG`` if set, else conservative defaults."""
        env_path = os.environ.get(_CONFIG_ENV_VAR)
        if env_path:
            return cls.from_yaml(Path(env_path))
        return cls()
