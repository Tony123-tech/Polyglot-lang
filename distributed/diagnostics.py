"""Structured diagnostics for the distributed runtime.

The public toolchain surface is E001--E008.  The experimental pillars in
``runtime/`` already give each code a stable meaning; distributed failures map
onto that namespace instead of inventing a parallel one:

========  =====================================================
Code      Distributed meaning
========  =====================================================
E001      type / layout / ABI boundary in a shipped state
E003      API misuse (bad argument, illegal state transition)
E006      missing handler, module, actor, or checkpoint
E007      unsupported native type or unavailable platform feature
E008      distributed boundary fault: malformed wire data, stale
          fencing token, expired lease, aborted migration, node loss
========  =====================================================

Every error carries a machine-readable ``context`` dictionary so a cluster
controller can render a structured diagnostic without parsing text.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

from runtime.diagnostics import ERROR_CODES, PillarError

# Stable, human-readable names for the E008 sub-kinds.  ``note`` is what
# log aggregators key on; ``reason`` is the free-form detail.
WIRE_FORMAT: Final = "WireFormatError"
STATE_MISMATCH: Final = "StateMismatchError"
LEASE_CONFLICT: Final = "LeaseConflictError"
STALE_FENCE: Final = "StaleFenceError"
LEASE_EXPIRED: Final = "LeaseExpiredError"
NODE_UNAVAILABLE: Final = "NodeUnavailableError"
MIGRATION_ABORTED: Final = "MigrationAbortedError"
TASK_REPLAY: Final = "TaskReplayError"
PROTOCOL: Final = "ProtocolError"
NODE_LOST: Final = "NodeLostError"

DISTRIBUTED_CODES: Final[dict[str, str]] = {
    "E001": ERROR_CODES["E001"],
    "E003": ERROR_CODES["E003"],
    "E006": ERROR_CODES["E006"],
    "E007": ERROR_CODES["E007"],
    "E008": ERROR_CODES["E008"],
}


class DistributedError(PillarError):
    """A distributed-runtime failure with structured context."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "E008",
        reason: str | None = None,
        help_text: str | None = None,
        note: str | None = None,
        context: dict[str, Any] | None = None,
        line: int = 0,
        col: int = 0,
        span: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            code,
            message,
            reason=reason,
            help_text=help_text,
            note=note,
            line=line,
            col=col,
            span=span,
        )
        self.context: dict[str, Any] = dict(context or {})

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic payload."""
        return {
            "code": self.code,
            "message": self.diagnostic.message,
            "reason": self.diagnostic.reason,
            "help": self.diagnostic.help_text,
            "note": self.diagnostic.note,
            "line": self.line,
            "col": self.col,
            "span": list(self.span) if self.span is not None else None,
            "context": _jsonable(self.context),
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)


def wire_error(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", WIRE_FORMAT)
    kwargs.setdefault(
        "help_text",
        "check the frame magic, format version, CRC, and payload length",
    )
    return DistributedError(message, **kwargs)


def lease_conflict(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", LEASE_CONFLICT)
    kwargs.setdefault(
        "help_text",
        "release the conflicting lease or request the correct access mode",
    )
    return DistributedError(message, **kwargs)


def stale_fence(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", STALE_FENCE)
    kwargs.setdefault(
        "help_text",
        "reload the object from its new owner; this token has been superseded",
    )
    return DistributedError(message, **kwargs)


def lease_expired(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", LEASE_EXPIRED)
    kwargs.setdefault(
        "help_text",
        "renew the lease before its TTL and retry the operation",
    )
    return DistributedError(message, **kwargs)


def node_unavailable(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", NODE_UNAVAILABLE)
    kwargs.setdefault(
        "help_text",
        "wait for the node to rejoin or migrate the actor to a healthy node",
    )
    return DistributedError(message, **kwargs)


def migration_aborted(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", MIGRATION_ABORTED)
    kwargs.setdefault(
        "help_text",
        "the source actor was restored from its checkpoint; retry the migration",
    )
    return DistributedError(message, **kwargs)


def task_replay(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", TASK_REPLAY)
    kwargs.setdefault(
        "help_text",
        "handlers must be deterministic up to each await point; inspect the "
        "journal and the recorded operation",
    )
    return DistributedError(message, **kwargs)


def protocol_error(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", PROTOCOL)
    kwargs.setdefault("help_text", "check the control-plane message ordering")
    return DistributedError(message, **kwargs)


def node_lost(message: str, **kwargs: Any) -> DistributedError:
    kwargs.setdefault("note", NODE_LOST)
    kwargs.setdefault(
        "help_text",
        "the actor was rolled back to its last replicated checkpoint",
    )
    return DistributedError(message, **kwargs)


@dataclass(frozen=True)
class DiagnosticRecord:
    """A timestamped diagnostic retained by a node or coordinator."""

    at: float
    node: str
    code: str
    note: str
    message: str
    reason: str
    help_text: str
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "node": self.node,
            "code": self.code,
            "note": self.note,
            "message": self.message,
            "reason": self.reason,
            "help": self.help_text,
            "context": _jsonable(self.context),
        }

    def format(self) -> str:
        return (
            f"error[{self.code}]: {self.message}\n"
            f"  reason: {self.reason}\n"
            f"  help: {self.help_text}\n"
            f"  note: {self.note}\n"
            f"  node: {self.node}"
        )


class DiagnosticLog:
    """A bounded, thread-safe log of structured distributed diagnostics."""

    def __init__(self, *, node: str = "local", capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.node = node
        self.capacity = capacity
        self._records: list[DiagnosticRecord] = []
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def record(
        self,
        error: DistributedError,
        *,
        context: dict[str, Any] | None = None,
    ) -> DiagnosticRecord:
        merged = dict(error.context)
        if context:
            merged.update(context)
        record = DiagnosticRecord(
            at=time.time(),
            node=self.node,
            code=error.code,
            note=error.diagnostic.note,
            message=error.diagnostic.message,
            reason=error.diagnostic.reason,
            help_text=error.diagnostic.help_text,
            context=merged,
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.capacity:
                del self._records[: len(self._records) - self.capacity]
            self._counts[record.note] = self._counts.get(record.note, 0) + 1
        return record

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def records(self) -> list[DiagnosticRecord]:
        with self._lock:
            return list(self._records)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def latest(self, note: str | None = None) -> DiagnosticRecord | None:
        with self._lock:
            for record in reversed(self._records):
                if note is None or record.note == note:
                    return record
        return None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._counts.clear()

    def as_json(self) -> str:
        return json.dumps(
            [record.as_dict() for record in self.records()],
            indent=2,
            sort_keys=True,
        )


__all__ = [
    "DISTRIBUTED_CODES",
    "LEASE_CONFLICT",
    "LEASE_EXPIRED",
    "MIGRATION_ABORTED",
    "NODE_LOST",
    "NODE_UNAVAILABLE",
    "PROTOCOL",
    "STALE_FENCE",
    "STATE_MISMATCH",
    "TASK_REPLAY",
    "WIRE_FORMAT",
    "DiagnosticLog",
    "DiagnosticRecord",
    "DistributedError",
    "lease_conflict",
    "lease_expired",
    "migration_aborted",
    "node_lost",
    "node_unavailable",
    "protocol_error",
    "stale_fence",
    "task_replay",
    "wire_error",
]
