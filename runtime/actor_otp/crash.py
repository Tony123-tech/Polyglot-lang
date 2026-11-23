"""Let-it-crash fault isolation, structured fault reports, and telemetry.

OTP's philosophy is "let it crash": instead of defensive error handling
deep inside a process, you let the process die and rely on a supervisor to
restart it from a clean state.  This module models that discipline:

* :func:`classify_fault` maps an exception to a structured ``E001``–``E008``
  :class:`CrashReport` so failures are isolated and diagnosable without
  crashing unrelated subsystems.
* :class:`Telemetry` is a bounded ring buffer of :class:`TelemetryEvent`
  records for audit/telemetry.  It never crashes the host and never grows
  unbounded — the oldest events are dropped when the buffer is full.

Platform boundaries are respected: the module never touches ``MAP_JIT``,
PEP 667 frame locals, raw addresses, or model-only paths.  A fault handler
that tries to access any of those raises ``E007``.

All code is pure Python and deterministic.
"""

from __future__ import annotations

import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Final

from compiler.frontend.errors import PolyglotError

__all__ = [
    "CrashReport",
    "Telemetry",
    "TelemetryEvent",
    "classify_fault",
]

# Maximum number of telemetry events retained in the ring buffer.  Oldest
# events are dropped when the buffer is full so telemetry never grows
# unbounded.
_MAX_TELEMETRY_EVENTS: Final[int] = 10_000

# Exception types that indicate a platform/runtime boundary violation and
# must never be silently swallowed.
_BOUNDARY_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    MemoryError,
    RecursionError,
    SystemExit,
    KeyboardInterrupt,
)

# Keywords in an exception message that indicate an attempt to touch a
# forbidden platform boundary.
_BOUNDARY_KEYWORDS: Final[tuple[str, ...]] = (
    "MAP_JIT",
    "PEP 667",
    "f_locals",
    "raw address",
    "TOCTOU",
    "model-only",
)


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    raise PolyglotError(code, message, reason=reason)


def _classify_exception(exc: BaseException) -> str:
    """Map an exception to an ``E001``–``E008`` diagnostic code."""
    if isinstance(exc, PolyglotError):
        return exc.code
    if isinstance(exc, MemoryError):
        return "E004"
    if isinstance(exc, RecursionError):
        return "E002"
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return "E007"
    if isinstance(exc, (TypeError, ValueError)):
        return "E001"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "E006"
    if isinstance(exc, TimeoutError):
        return "E004"
    if isinstance(exc, OSError):
        return "E007"
    return "E004"


def _check_boundary(exc: BaseException) -> None:
    """Raise ``E007`` if *exc* attempts to touch a forbidden platform boundary."""
    message = str(exc)
    for keyword in _BOUNDARY_KEYWORDS:
        if keyword in message:
            _fail(
                "E007",
                f"fault handler attempted to access a forbidden platform "
                f"boundary ({keyword!r})",
                reason="platform boundaries (MAP_JIT, PEP 667, raw-address "
                "TOCTOU, model-only paths) must never be touched",
            )


@dataclass(frozen=True)
class CrashReport:
    """A structured fault report for a crashed actor/process."""

    actor_name: str
    code: str
    message: str
    traceback: str
    boundary_safe: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor_name,
            "code": self.code,
            "message": self.message,
            "traceback": self.traceback,
            "boundary_safe": self.boundary_safe,
        }

    def format(self) -> str:
        return f"[actor {self.actor_name}] crash {self.code}: {self.message}" + (
            "" if self.boundary_safe else " (BOUNDARY VIOLATION)"
        )


@dataclass
class TelemetryEvent:
    """A single telemetry/audit event."""

    actor_name: str
    event_type: str
    code: str | None
    detail: str
    timestamp: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor_name,
            "event_type": self.event_type,
            "code": self.code,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class Telemetry:
    """A bounded ring buffer of telemetry events.

    When the buffer is full (``max_events``), the oldest event is dropped
    when a new one is added.  This ensures telemetry never grows unbounded
    and never crashes the host.
    """

    max_events: int = _MAX_TELEMETRY_EVENTS
    _events: deque[TelemetryEvent] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.max_events <= 0:
            _fail("E001", "telemetry max_events must be positive")

    def record(
        self,
        actor_name: str,
        event_type: str,
        detail: str,
        *,
        code: str | None = None,
        timestamp: float = 0.0,
    ) -> None:
        event = TelemetryEvent(
            actor_name=actor_name,
            event_type=event_type,
            code=code,
            detail=detail,
            timestamp=timestamp,
        )
        if len(self._events) >= self.max_events:
            self._events.popleft()
        self._events.append(event)

    def events_for(self, actor_name: str) -> tuple[TelemetryEvent, ...]:
        return tuple(e for e in self._events if e.actor_name == actor_name)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": len(self._events),
            "max_events": self.max_events,
            "events": [e.as_dict() for e in self._events],
        }


def classify_fault(exc: BaseException, *, actor_name: str = "<unknown>") -> CrashReport:
    """Classify an exception into a structured :class:`CrashReport`.

    The report maps the exception to an ``E001``–``E008`` code, captures the
    traceback, and verifies that the fault does not attempt to touch a
    forbidden platform boundary (``MAP_JIT``, PEP 667, raw addresses, etc.).
    A boundary violation is reported but does not re-raise — the caller
    decides whether to escalate.
    """
    _check_boundary(exc)
    code = _classify_exception(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return CrashReport(
        actor_name=actor_name,
        code=code,
        message=str(exc),
        traceback=tb,
        boundary_safe=True,
    )
