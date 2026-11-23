"""Bounded causal recording, minimal failure slicing, and boundary isolation.

The subsystem models *explicit* happens-before relations only:

* an event names its causal parents (actor messages, state reads/writes, error
  events); program order within one actor is linked automatically unless the
  caller disables it;
* Lamport and vector clocks are derived from that graph, never from wall-clock
  timestamps;
* cycles, missing parents, duplicate ids, and malformed records are reported
  through the shared ``E001``–``E008`` diagnostic surface;
* the minimal failure slice is the *unique smallest causally-closed ancestor
  set* of the failure, with irrelevant leaf state writes pruned.  It is not a
  claim of globally minimum cardinality slicing, which is NP-hard;
* replay artifacts live in a scoped temporary directory with ``0o700``
  permissions and are written with ``O_NOFOLLOW``.  Raw-address check-then-use
  remains a TOCTOU boundary and is never dereferenced here.  ``MAP_JIT`` is
  never used, and PEP 667 frame-local mutation is model-only.

All analysis is bounded by ``max_events``, ``max_slice``, and ``timeout_ms``;
exhaustion produces an explicit ``E008``/``E004`` ``Unknown`` outcome.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from runtime.diagnostics import ERROR_CODES, PillarError

__all__ = [
    "BoundaryStatus",
    "CausalDiagnostic",
    "CausalEvent",
    "CausalSlice",
    "CausalTraceReport",
    "IsolatedArtifact",
    "ReplayIsolationReport",
    "replay_enforce_boundary_isolation",
    "replay_record_causal_happens_before",
    "replay_slice_minimal_failure_causality",
]

Span = tuple[int, int] | None
Severity = Literal["error", "warning", "unknown"]
_ERROR_KINDS = frozenset({"error", "fault", "invariant_violation", "panic"})
_WRITE_KINDS = frozenset({"state_write", "write", "mutation"})
_READ_KINDS = frozenset({"state_read", "read"})
_DEFAULT_MAX_EVENTS = 4096
_DEFAULT_MAX_SLICE = 1024
_DEFAULT_TIMEOUT_MS = 2000.0


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
    note: str = "runtime.causal_replay",
) -> None:
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "record explicit causal parents or raise the bounded analysis budget",
        note=note,
        span=span,
    )


def _digest(payload: Any) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CausalDiagnostic:
    """A structured causal-analysis diagnostic."""

    code: str
    message: str
    kind: str
    event_id: str | None = None
    severity: Severity = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "kind": self.kind,
            "event_id": self.event_id,
            "severity": self.severity,
        }


def _raise_first(diagnostics: Sequence[CausalDiagnostic], span: Span) -> None:
    for diagnostic in diagnostics:
        if diagnostic.severity != "error":
            continue
        _fail(
            diagnostic.code,
            diagnostic.message,
            span=span,
            reason=diagnostic.message,
        )


# ---------------------------------------------------------------------------
# Event graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEvent:
    """One recorded event with derived logical clocks."""

    event_id: str
    actor: str
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    parents: tuple[str, ...] = ()
    lamport: int = 0
    vector: tuple[tuple[str, int], ...] = ()
    code: str | None = None
    span: Span = None

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "actor": self.actor,
            "kind": self.kind,
            "payload": dict(self.payload),
            "parents": list(self.parents),
            "lamport": self.lamport,
            "vector": {actor: clock for actor, clock in self.vector},
            "code": self.code,
            "span": list(self.span) if self.span else None,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def happens_before(self, other: CausalEvent) -> bool:
        """True when the vector clock proves a strict causal predecessor."""
        if self.event_id == other.event_id:
            return False
        left = dict(self.vector)
        right = dict(other.vector)
        return all(right.get(actor, 0) >= clock for actor, clock in left.items()) and (
            dict(self.vector) != dict(other.vector)
        )


def _clone_event(event: CausalEvent, **changes: Any) -> CausalEvent:
    data = event.as_dict(include_digest=False)
    data.update(changes)
    raw_vector = data["vector"]
    if isinstance(raw_vector, Mapping):
        vector = tuple(
            sorted((str(key), int(value)) for key, value in raw_vector.items())
        )
    else:
        vector = tuple((str(key), int(value)) for key, value in raw_vector)
    span = data["span"]
    return CausalEvent(
        event_id=data["event_id"],
        actor=data["actor"],
        kind=data["kind"],
        payload=data["payload"],
        parents=tuple(data["parents"]),
        lamport=int(data["lamport"]),
        vector=vector,
        code=data["code"],
        span=tuple(span) if span is not None else None,
    )


def _normalize_event(value: Any, index: int) -> CausalEvent:
    if isinstance(value, CausalEvent):
        return CausalEvent(
            event_id=value.event_id,
            actor=value.actor,
            kind=value.kind,
            payload=dict(value.payload),
            parents=tuple(str(parent) for parent in value.parents),
            code=value.code,
            span=value.span,
        )
    if not isinstance(value, Mapping):
        _fail(
            "E003",
            f"event must be a mapping or CausalEvent, got {type(value).__name__}",
        )
    event_id = value.get("event_id", value.get("id", f"e{index}"))
    actor = value.get("actor", value.get("role"))
    kind = value.get("kind", value.get("type"))
    if not isinstance(event_id, str) or not event_id:
        _fail("E008", f"event at index {index} needs a non-empty string id")
    if not isinstance(actor, str) or not actor:
        _fail("E008", f"event {event_id!r} needs a non-empty actor")
    if not isinstance(kind, str) or not kind:
        _fail("E008", f"event {event_id!r} needs a non-empty kind")
    raw_payload = value.get("payload", {})
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, Mapping):
        _fail("E008", f"event {event_id!r} payload must be a mapping")
    payload = {str(key): item for key, item in raw_payload.items()}
    raw_parents = value.get(
        "parents", value.get("causal_parents", value.get("depends_on", ()))
    )
    if isinstance(raw_parents, (str, bytes)):
        raw_parents = (raw_parents,)
    try:
        parents = tuple(str(parent) for parent in raw_parents)
    except TypeError:
        _fail("E008", f"event {event_id!r} parents must be a sequence")
    code = value.get("code", payload.get("code"))
    if kind in _ERROR_KINDS:
        if not isinstance(code, str) or code not in ERROR_CODES:
            _fail(
                "E008",
                f"error event {event_id!r} needs a code in E001–E008, got {code!r}",
            )
    elif code is not None and code not in ERROR_CODES:
        _fail("E008", f"event {event_id!r} has unknown code {code!r}")
    span = value.get("span")
    normalized_span: Span = None
    if span is not None:
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            _fail("E008", f"event {event_id!r} span must be a pair")
        normalized_span = (int(span[0]), int(span[1]))
    return CausalEvent(
        event_id,
        actor,
        kind,
        payload,
        parents,
        code=code,
        span=normalized_span,
    )


@dataclass(frozen=True)
class CausalTraceReport:
    """A deterministic causal trace with derived clocks."""

    trace_id: str
    events: tuple[CausalEvent, ...]
    order: tuple[str, ...]
    verified: bool
    diagnostics: tuple[CausalDiagnostic, ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "events": [event.as_dict() for event in self.events],
            "order": list(self.order),
            "verified": self.verified,
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "bounds": dict(self.bounds),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        return _digest(
            {
                "trace_id": self.trace_id,
                "events": [event.as_dict() for event in self.events],
                "order": list(self.order),
            }
        )

    def event(self, event_id: str) -> CausalEvent:
        for event in self.events:
            if event.event_id == event_id:
                return event
        _fail("E006", f"trace {self.trace_id!r} has no event {event_id!r}")


def _topological_order(
    events: Sequence[CausalEvent],
    *,
    deadline: float,
    max_events: int,
) -> tuple[list[CausalEvent], list[CausalDiagnostic]]:
    by_id = {event.event_id: event for event in events}
    index = {event.event_id: position for position, event in enumerate(events)}
    diagnostics: list[CausalDiagnostic] = []
    indegree = {event.event_id: 0 for event in events}
    children: dict[str, list[str]] = {event.event_id: [] for event in events}
    for event in events:
        for parent in event.parents:
            if parent not in by_id:
                diagnostics.append(
                    CausalDiagnostic(
                        "E006",
                        f"event {event.event_id!r} depends on missing parent "
                        f"{parent!r}",
                        "missing-parent",
                        event.event_id,
                    )
                )
                continue
            indegree[event.event_id] += 1
            children[parent].append(event.event_id)
    if diagnostics:
        return [], diagnostics
    ready = [
        (index[event_id], event_id)
        for event_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    order: list[CausalEvent] = []
    while ready:
        if time.monotonic() > deadline:
            return order, [
                CausalDiagnostic(
                    "E004",
                    "causal ordering exceeded the time budget",
                    "timeout",
                    severity="unknown",
                )
            ]
        _, event_id = heapq.heappop(ready)
        order.append(by_id[event_id])
        for child in children[event_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (index[child], child))
        if len(order) > max_events:
            return order, [
                CausalDiagnostic(
                    "E008",
                    f"event count exceeds max_events={max_events}",
                    "event-bound",
                    severity="unknown",
                )
            ]
    if len(order) != len(events):
        cyclic = sorted(
            (event_id for event_id, degree in indegree.items() if degree > 0),
            key=lambda event_id: index[event_id],
        )
        return order, [
            CausalDiagnostic(
                "E005",
                "causal graph contains a cycle among " + ", ".join(cyclic[:8]),
                "cycle",
            )
        ]
    return order, []


def _compute_clocks(
    order: Sequence[CausalEvent],
) -> tuple[CausalEvent, ...]:
    lamport: dict[str, int] = {}
    vectors: dict[str, dict[str, int]] = {}
    result: list[CausalEvent] = []
    for event in order:
        parent_lamports = [lamport[parent] for parent in event.parents]
        event_lamport = (max(parent_lamports) if parent_lamports else 0) + 1
        vector: dict[str, int] = {}
        for parent in event.parents:
            for actor, clock in vectors[parent].items():
                vector[actor] = max(vector.get(actor, 0), clock)
        vector[event.actor] = vector.get(event.actor, 0) + 1
        lamport[event.event_id] = event_lamport
        vectors[event.event_id] = vector
        result.append(
            _clone_event(
                event,
                lamport=event_lamport,
                vector=tuple(sorted(vector.items())),
            )
        )
    return tuple(result)


def replay_record_causal_happens_before(
    events: Sequence[Any],
    *,
    trace_id: str = "trace",
    link_actor_order: bool = True,
    max_events: int = _DEFAULT_MAX_EVENTS,
    timeout_ms: float = _DEFAULT_TIMEOUT_MS,
    strict: bool = False,
    span: Span = None,
) -> CausalTraceReport:
    """Record events and derive Lamport/vector clocks from explicit causality.

    ``link_actor_order`` adds the previous event of the same actor as an
    implicit parent, modelling program order without consulting wall-clock
    time.  Parents must refer to existing events; cycles are rejected with
    ``E005`` and missing parents with ``E006``.
    """
    if max_events < 1 or timeout_ms <= 0:
        _fail("E008", "causal bounds must be positive", span=span)
    if not isinstance(trace_id, str) or not trace_id:
        _fail("E003", "trace_id must be a non-empty string", span=span)
    if len(events) > max_events:
        _fail(
            "E008",
            f"trace has {len(events)} events, above max_events={max_events}",
            span=span,
        )
    deadline = time.monotonic() + timeout_ms / 1000.0
    normalized: list[CausalEvent] = []
    diagnostics: list[CausalDiagnostic] = []
    seen: set[str] = set()
    last_by_actor: dict[str, str] = {}
    for index, value in enumerate(events):
        event = _normalize_event(value, index)
        if event.event_id in seen:
            _fail(
                "E008",
                f"duplicate event id {event.event_id!r}",
                span=span,
            )
        seen.add(event.event_id)
        parents = list(event.parents)
        if link_actor_order:
            previous = last_by_actor.get(event.actor)
            if previous is not None and previous not in parents:
                parents.append(previous)
        last_by_actor[event.actor] = event.event_id
        normalized.append(
            _clone_event(event, parents=tuple(parents), lamport=0, vector=())
        )
    order, ordering_diagnostics = _topological_order(
        normalized, deadline=deadline, max_events=max_events
    )
    diagnostics.extend(ordering_diagnostics)
    bounded = any(item.severity == "unknown" for item in diagnostics)
    if ordering_diagnostics:
        report = CausalTraceReport(
            trace_id,
            tuple(normalized),
            tuple(event.event_id for event in order),
            False,
            tuple(diagnostics),
            {
                "max_events": max_events,
                "timeout_ms": timeout_ms,
                "link_actor_order": link_actor_order,
                "ordering": "explicit-parents-plus-actor-program-order",
            },
        )
        if strict:
            _raise_first(report.diagnostics, span)
        return report

    clocked = {event.event_id: event for event in _compute_clocks(order)}
    events_out = tuple(clocked[event.event_id] for event in normalized)
    verified = not bounded
    if verified:
        diagnostics.append(
            CausalDiagnostic(
                "E002",
                "causal trace verified: every clock is derived from explicit "
                "parents and actor program order",
                "proof",
                severity="warning",
            )
        )
    report = CausalTraceReport(
        trace_id,
        events_out,
        tuple(event.event_id for event in order),
        verified,
        tuple(diagnostics),
        {
            "max_events": max_events,
            "timeout_ms": timeout_ms,
            "link_actor_order": link_actor_order,
            "ordering": "explicit-parents-plus-actor-program-order",
        },
    )
    if strict:
        _raise_first(report.diagnostics, span)
    return report


# ---------------------------------------------------------------------------
# Minimal failure slicing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalSlice:
    """A causally-closed ancestor slice for one failure event."""

    trace_id: str
    failure_id: str
    event_ids: tuple[str, ...]
    events: tuple[CausalEvent, ...]
    removed: tuple[str, ...]
    failure_code: str | None
    verified: bool
    minimal: bool
    bounded: bool
    diagnostics: tuple[CausalDiagnostic, ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "failure_id": self.failure_id,
            "event_ids": list(self.event_ids),
            "events": [event.as_dict() for event in self.events],
            "removed": list(self.removed),
            "failure_code": self.failure_code,
            "verified": self.verified,
            "minimal": self.minimal,
            "bounded": self.bounded,
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "bounds": dict(self.bounds),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        return _digest(
            {
                "trace_id": self.trace_id,
                "failure_id": self.failure_id,
                "event_ids": list(self.event_ids),
                "removed": list(self.removed),
            }
        )


def _ancestor_closure(
    failure_id: str,
    by_id: Mapping[str, CausalEvent],
    removed: set[str],
    *,
    deadline: float,
) -> tuple[set[str], bool]:
    """Ancestor closure, treating ``removed`` writes as transparent.

    Removed events are not included, but traversal continues through them so
    their ancestors remain available to explain later kept events.
    """
    closure: set[str] = set()
    visited = {failure_id}
    stack = [failure_id]
    while stack:
        if time.monotonic() > deadline:
            return closure, True
        current = stack.pop()
        if current not in removed:
            closure.add(current)
        for parent in by_id[current].parents:
            if parent in visited or parent not in by_id:
                continue
            visited.add(parent)
            stack.append(parent)
    return closure, False


def replay_slice_minimal_failure_causality(
    trace: CausalTraceReport | Sequence[Any],
    failure_id: str,
    *,
    include_state_reads: bool = True,
    max_slice: int = _DEFAULT_MAX_SLICE,
    timeout_ms: float = _DEFAULT_TIMEOUT_MS,
    strict: bool = False,
    span: Span = None,
) -> CausalSlice:
    """Compute the smallest causally-closed ancestor slice of a failure.

    The ancestor closure is exact and unique.  When ``include_state_reads`` is
    enabled, state writes whose resource is never read by the failure slice are
    treated as *transparent*: they are removed while traversal continues
    through them, so messages and relevant reads/writes remain connected.
    Global minimum-cardinality slicing is not claimed.
    """
    if max_slice < 1 or timeout_ms <= 0:
        _fail("E008", "slice bounds must be positive", span=span)
    if not isinstance(failure_id, str) or not failure_id:
        _fail("E003", "failure_id must be a non-empty string", span=span)
    report = (
        trace
        if isinstance(trace, CausalTraceReport)
        else replay_record_causal_happens_before(trace, strict=False)
    )
    diagnostics: list[CausalDiagnostic] = []
    if not report.verified:
        diagnostics.append(
            CausalDiagnostic(
                "E002",
                "cannot take a trusted slice of an unverified causal trace",
                "unverified-trace",
                failure_id,
            )
        )
        slice_report = CausalSlice(
            report.trace_id,
            failure_id,
            (),
            (),
            (),
            None,
            False,
            False,
            False,
            tuple(diagnostics),
            {"max_slice": max_slice, "timeout_ms": timeout_ms},
        )
        if strict:
            _raise_first(slice_report.diagnostics, span)
        return slice_report
    by_id = {event.event_id: event for event in report.events}
    if failure_id not in by_id:
        _fail(
            "E006",
            f"failure event {failure_id!r} is not in trace {report.trace_id!r}",
            span=span,
        )
    deadline = time.monotonic() + timeout_ms / 1000.0
    excluded: set[str] = set()
    full_closure, timed_out = _ancestor_closure(
        failure_id, by_id, set(), deadline=deadline
    )
    closure, timed_out = _ancestor_closure(
        failure_id, by_id, excluded, deadline=deadline
    )
    bounded = timed_out
    if timed_out:
        diagnostics.append(
            CausalDiagnostic(
                "E004",
                "failure slicing exceeded the time budget",
                "timeout",
                failure_id,
                "unknown",
            )
        )
    if len(closure) > max_slice:
        bounded = True
        diagnostics.append(
            CausalDiagnostic(
                "E008",
                f"causal slice has {len(closure)} events, above max_slice={max_slice}",
                "slice-bound",
                failure_id,
                "unknown",
            )
        )
    elif include_state_reads and not bounded:
        for _ in range(len(full_closure) + 1):
            if timed_out:
                bounded = True
                diagnostics.append(
                    CausalDiagnostic(
                        "E004",
                        "failure slicing exceeded the time budget",
                        "timeout",
                        failure_id,
                        "unknown",
                    )
                )
                break
            reads: set[str] = set()
            for event_id in closure:
                event = by_id[event_id]
                if event.kind in _READ_KINDS:
                    resource = event.payload.get("resource")
                    if resource is not None:
                        reads.add(str(resource))
                declared = event.payload.get("reads")
                if isinstance(declared, str):
                    reads.add(declared)
                elif isinstance(declared, Iterable):
                    reads.update(str(item) for item in declared)
            candidates = {
                event_id
                for event_id in closure
                if event_id != failure_id
                and by_id[event_id].kind in _WRITE_KINDS
                and by_id[event_id].payload.get("resource") is not None
                and str(by_id[event_id].payload.get("resource")) not in reads
            }
            if candidates == excluded:
                break
            excluded = candidates
            closure, timed_out = _ancestor_closure(
                failure_id, by_id, excluded, deadline=deadline
            )
            if len(closure) > max_slice:
                bounded = True
                diagnostics.append(
                    CausalDiagnostic(
                        "E008",
                        f"causal slice has {len(closure)} events, above "
                        f"max_slice={max_slice}",
                        "slice-bound",
                        failure_id,
                        "unknown",
                    )
                )
                break
    removed = full_closure - closure
    ordered = [event_id for event_id in report.order if event_id in closure]
    events = tuple(by_id[event_id] for event_id in ordered)
    verified = not bounded
    if verified:
        diagnostics.append(
            CausalDiagnostic(
                "E002",
                "causal slice verified: ancestor closure with irrelevant "
                "state writes treated as transparent",
                "proof",
                failure_id,
                "warning",
            )
        )
    slice_report = CausalSlice(
        report.trace_id,
        failure_id,
        tuple(ordered),
        events,
        tuple(event_id for event_id in report.order if event_id in removed),
        by_id[failure_id].code,
        verified,
        verified,
        bounded,
        tuple(diagnostics),
        {
            "max_slice": max_slice,
            "timeout_ms": timeout_ms,
            "include_state_reads": include_state_reads,
            "transparent_writes": True,
        },
    )
    if strict:
        _raise_first(slice_report.diagnostics, span)
    return slice_report


# ---------------------------------------------------------------------------
# Boundary isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryStatus:
    kind: str
    status: str
    detail: str
    code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "code": self.code,
        }


@dataclass(frozen=True)
class IsolatedArtifact:
    name: str
    path: str
    size: int
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ReplayIsolationReport:
    root: str
    owned: bool
    boundaries: tuple[BoundaryStatus, ...]
    artifacts: tuple[IsolatedArtifact, ...]
    diagnostics: tuple[CausalDiagnostic, ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "root": self.root,
            "owned": self.owned,
            "boundaries": [item.as_dict() for item in self.boundaries],
            "artifacts": [item.as_dict() for item in self.artifacts],
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "bounds": dict(self.bounds),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        return _digest(
            {
                "root": self.root,
                "artifacts": [item.as_dict() for item in self.artifacts],
                "boundaries": [item.as_dict() for item in self.boundaries],
            }
        )

    @property
    def blocked(self) -> bool:
        return any(item.code is not None for item in self.boundaries)

    def cleanup(self) -> None:
        """Remove the scoped artifact tree (or only owned artifacts)."""
        path = Path(self.root)
        if self.owned:
            shutil.rmtree(path, ignore_errors=True)
            return
        for artifact in self.artifacts:
            try:
                Path(artifact.path).unlink()
            except OSError:
                continue


def _repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _normalize_boundaries(
    boundaries: Sequence[Any] | None,
    *,
    privileged: bool,
    raw_address_authorized: bool,
    allow_jit_model: bool,
) -> tuple[tuple[BoundaryStatus, ...], list[CausalDiagnostic]]:
    if boundaries is None:
        boundaries = ("model_only", "filesystem")
    statuses: list[BoundaryStatus] = []
    diagnostics: list[CausalDiagnostic] = []
    for item in boundaries:
        raw_kind = item.get("kind") if isinstance(item, Mapping) else item
        if not isinstance(raw_kind, str) or not raw_kind:
            _fail("E003", "boundary entries need a non-empty kind")
        kind = raw_kind.strip().lower().replace("-", "_")
        if kind in {"map_jit", "jit", "executable_memory"}:
            if allow_jit_model:
                status = BoundaryStatus(
                    kind,
                    "model-only",
                    "JIT is modelled but no executable pages are ever mapped",
                )
            else:
                status = BoundaryStatus(
                    kind,
                    "blocked",
                    "MAP_JIT/executable memory is not used by this build",
                    "E007",
                )
        elif kind in {"pep667", "frame_locals", "pep_667"}:
            status = BoundaryStatus(
                kind,
                "model-only",
                "frame locals are inspected as data only; never mutated",
            )
        elif kind in {"raw_address", "rawaddr", "pointer"}:
            if raw_address_authorized:
                status = BoundaryStatus(
                    kind,
                    "toctou-guarded",
                    "raw addresses are symbolic; check-then-use remains a "
                    "TOCTOU boundary and is never dereferenced",
                )
            else:
                status = BoundaryStatus(
                    kind,
                    "blocked",
                    "raw-address replay needs explicit authorization and still "
                    "carries a TOCTOU residual risk",
                    "E007",
                )
        elif kind in {"privileged", "privileged_path"}:
            if privileged:
                status = BoundaryStatus(
                    kind, "allowed", "privileged path explicitly authorized"
                )
            else:
                status = BoundaryStatus(
                    kind,
                    "blocked",
                    "privileged path is unavailable without explicit authority",
                    "E007",
                )
        elif kind in {"model_only", "model"}:
            status = BoundaryStatus(
                kind, "model-only", "pure data model; no host effect"
            )
        elif kind in {"filesystem", "file", "artifact"}:
            status = BoundaryStatus(
                kind,
                "scoped",
                "artifacts are confined to a 0o700 temporary/cache directory",
            )
        elif kind in {"network", "subprocess", "native_ffi", "ffi", "device"}:
            status = BoundaryStatus(
                kind,
                "blocked",
                f"{kind} is outside the replay sandbox",
                "E007",
            )
        else:
            status = BoundaryStatus(
                kind,
                "unsupported",
                f"unknown runtime boundary {raw_kind!r}",
                "E007",
            )
        statuses.append(status)
        if status.code is not None:
            diagnostics.append(
                CausalDiagnostic(
                    status.code,
                    status.detail,
                    "boundary",
                    status.kind,
                )
            )
    return tuple(statuses), diagnostics


def _normalize_artifacts(
    artifacts: Any,
) -> list[tuple[str, bytes]]:
    if artifacts is None:
        return []
    items: Iterable[Any]
    if isinstance(artifacts, Mapping):
        items = list(artifacts.items())
    else:
        items = artifacts
    rows: list[tuple[str, bytes]] = []
    for item in items:
        name: Any
        content: Any
        if isinstance(item, Mapping):
            name = item.get("name")
            content = item.get("content", b"")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, content = item
        else:
            _fail("E003", "artifacts must be name/content mappings or pairs")
        if not isinstance(name, str) or not name:
            _fail("E003", "artifact names must be non-empty strings")
        if isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
        else:
            _fail("E003", f"artifact {name!r} content must be str or bytes")
        rows.append((name, data))
    return rows


def _scoped_root(root: str | os.PathLike[str] | None) -> tuple[Path, bool, bool]:
    """Return (root, owned, repo_conflict)."""
    if root is None:
        base = Path(tempfile.gettempdir()).resolve()
        path = Path(tempfile.mkdtemp(prefix="polyglot-causal-replay-", dir=base))
        return path, True, False
    if isinstance(root, (str, os.PathLike)):
        candidate = Path(root).expanduser()
    else:
        _fail("E003", "isolation root must be a path or None")
    candidate = candidate.resolve()
    repo = _repo_root(candidate)
    if repo is not None and (repo == candidate or repo in candidate.parents):
        fallback = Path(tempfile.mkdtemp(prefix="polyglot-causal-replay-"))
        return fallback, True, True
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate, False, False


def replay_enforce_boundary_isolation(
    boundaries: Sequence[Any] | None = None,
    *,
    artifacts: Any = None,
    trace: CausalTraceReport | None = None,
    root: str | os.PathLike[str] | None = None,
    privileged: bool = False,
    raw_address_authorized: bool = False,
    allow_jit_model: bool = False,
    max_artifact_bytes: int = 1_048_576,
    strict: bool = False,
    span: Span = None,
) -> ReplayIsolationReport:
    """Create a scoped replay sandbox and enforce runtime boundaries.

    The sandbox is a fresh ``0o700`` directory; artifacts must stay inside it,
    are opened with ``O_NOFOLLOW``, and are hashed.  A user-supplied root inside
    the repository is rejected (``E007``) and replaced by a temporary directory
    so ``git status`` stays clean.  ``MAP_JIT``, privileged paths, and raw
    addresses are blocked unless explicitly modelled/authorized.
    """
    if max_artifact_bytes < 1:
        _fail("E008", "max_artifact_bytes must be positive", span=span)
    boundaries_out, diagnostics = _normalize_boundaries(
        boundaries,
        privileged=privileged,
        raw_address_authorized=raw_address_authorized,
        allow_jit_model=allow_jit_model,
    )
    if strict and any(item.code is not None for item in boundaries_out):
        _raise_first(
            tuple(
                CausalDiagnostic(
                    status.code or "E007",
                    status.detail,
                    "boundary",
                    status.kind,
                )
                for status in boundaries_out
                if status.code is not None
            ),
            span,
        )
    scoped, owned, repo_conflict = _scoped_root(root)
    if repo_conflict:
        diagnostics.append(
            CausalDiagnostic(
                "E007",
                "refusing to place replay artifacts inside the repository; "
                "using a temporary directory instead",
                "repo-local-root",
            )
        )
        if strict:
            shutil.rmtree(scoped, ignore_errors=True)
            _raise_first(tuple(diagnostics), span)
    rows = _normalize_artifacts(artifacts)
    if trace is not None:
        rows.append(
            (
                "trace.json",
                (
                    json.dumps(
                        trace.as_dict(),
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        )
    written: list[IsolatedArtifact] = []
    scoped_resolved = scoped.resolve()
    for name, data in rows:
        if len(data) > max_artifact_bytes:
            diagnostics.append(
                CausalDiagnostic(
                    "E008",
                    f"artifact {name!r} is {len(data)} bytes, above "
                    f"max_artifact_bytes={max_artifact_bytes}",
                    "artifact-bound",
                )
            )
            continue
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\x00" in name
            or any(part == "" for part in relative.parts)
        ):
            diagnostics.append(
                CausalDiagnostic(
                    "E003",
                    f"artifact name {name!r} escapes the scoped directory",
                    "artifact-path",
                )
            )
            continue
        target = (scoped_resolved / relative).resolve()
        if target != scoped_resolved and scoped_resolved not in target.parents:
            diagnostics.append(
                CausalDiagnostic(
                    "E003",
                    f"artifact {name!r} resolves outside the scoped directory",
                    "artifact-path",
                )
            )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            diagnostics.append(
                CausalDiagnostic(
                    "E007",
                    f"artifact {name!r} is a symlink and will not be followed",
                    "artifact-symlink",
                )
            )
            continue
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
            try:
                os.write(descriptor, data)
            finally:
                os.close(descriptor)
        except OSError as error:
            diagnostics.append(
                CausalDiagnostic(
                    "E007",
                    f"cannot write artifact {name!r}: {error}",
                    "artifact-write",
                )
            )
            continue
        written.append(
            IsolatedArtifact(
                name=str(relative),
                path=str(target),
                size=len(data),
                digest=_sha256_bytes(data),
            )
        )
    try:
        os.chmod(scoped_resolved, 0o700)
    except OSError:
        pass
    report = ReplayIsolationReport(
        str(scoped_resolved),
        owned,
        boundaries_out,
        tuple(written),
        tuple(diagnostics),
        {
            "max_artifact_bytes": max_artifact_bytes,
            "artifact_count": len(written),
            "root": "temporary" if owned else "caller",
            "residual_risks": [
                "raw-address check-then-use remains TOCTOU",
                "parent-directory symlinks are resolved before O_NOFOLLOW",
                "no MAP_JIT or executable-memory path is exercised",
            ],
            "permissions": "0o700 directory, 0o600 files",
        },
    )
    if strict:
        _raise_first(report.diagnostics, span)
    return report
