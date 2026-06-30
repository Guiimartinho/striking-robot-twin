"""Structured logging and telemetry helpers.

Phase 0 ships the logging foundation the rest of the codebase uses instead of
``print``: a single configuration point and a module-logger accessor. Rich
telemetry (rerun.io streams of poses, the keep-out volume and trajectories) is a
later phase; the hook is here so it slots in without touching call sites.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logging once. Idempotent.

    Kept centralised so production code never calls ``print`` and every module
    gets a consistent, greppable format. Safe to call from any entry point.
    """
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
