"""Exception-free result type for the control and safety hot path.

The real robot's firmware (STM32) runs exception-free: a control or safety cycle
must never unwind a stack or allocate on an error. We mirror that discipline in
the Python twin so the logic that ships is shaped the same. Errors on the hot
path are returned as values (``Result``), not raised. Exceptions stay confined to
init and setup, where failing loud is correct.

``ErrorCode`` is a closed enum so the SafetyArbiter and the control loop can
switch on it exhaustively, and so logs and telemetry carry a stable, greppable
reason for every rejected or aborted command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class ErrorCode(Enum):
    """Closed set of outcomes for hot-path operations.

    OK is the only success value. Every other member is a specific, actionable
    failure reason. Keep this exhaustive: a new failure mode gets a new member,
    never a free-form string smuggled through ``message`` alone.
    """

    OK = "ok"

    # Safety rejections (the SafetyArbiter veto reasons).
    KEEPOUT_VIOLATION = "keepout_violation"
    REACH_EXCEEDED = "reach_exceeded"
    FORCE_CAP_EXCEEDED = "force_cap_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"

    # Input and observation problems.
    INVALID_COMMAND = "invalid_command"
    OBSERVER_STALE = "observer_stale"
    KEYPOINT_DROPOUT = "keypoint_dropout"

    # Plant problems.
    PLANT_FAULT = "plant_fault"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """A value-or-error outcome. Never raises on the hot path.

    Construct via :meth:`ok` or :meth:`err`, never directly, so the invariant
    "ok iff code is OK" holds. ``value`` is present only on success; ``message``
    carries human context for an error and is empty on success.
    """

    value: T | None
    code: ErrorCode
    message: str = ""

    @classmethod
    def ok(cls, value: T) -> Result[T]:
        """Wrap a successful value."""
        return cls(value=value, code=ErrorCode.OK, message="")

    @classmethod
    def err(cls, code: ErrorCode, message: str = "") -> Result[T]:
        """Wrap a failure. ``code`` must not be OK."""
        if code is ErrorCode.OK:
            raise ValueError("Result.err requires a non-OK code")
        return cls(value=None, code=code, message=message)

    @property
    def is_ok(self) -> bool:
        return self.code is ErrorCode.OK

    @property
    def is_err(self) -> bool:
        return self.code is not ErrorCode.OK

    def unwrap(self) -> T:
        """Return the value, raising if this is an error.

        Use only in tests and setup, never on the hot path: the whole point of
        Result is to not raise where it matters.
        """
        if self.is_err:
            raise ValueError(f"unwrap on error result: {self.code.value} ({self.message})")
        # value is guaranteed present when code is OK by construction.
        return self.value  # type: ignore[return-value]

    def unwrap_or(self, default: T) -> T:
        """Return the value, or ``default`` if this is an error."""
        return self.value if self.is_ok else default  # type: ignore[return-value]
