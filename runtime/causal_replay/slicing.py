"""Delta slicing of a recorded causal graph.

A causal past is usually far larger than the set of events actually
responsible for a failure.  This module computes a *structurally closed*
slice from explicit dependencies:

* ``reads`` / ``writes`` payload keys (last writer per ``(actor, key)``);
* ``depends_on`` payload entries;
* the matching send event for every receive event.

When the caller supplies a replay oracle, the slice is reduced further with a
deterministic, bounded one-minimal delta-debugging pass.  Without an oracle the
result is explicitly labelled ``structural-closure``: it is causally closed and
replayable, but it is not claimed to be semantically minimal.  Trial/time
bounds report ``Unknown``/``bounded`` rather than pretending to have finished.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from runtime.causal_replay.graph import CausalGraph, ReplayEvent, _jsonable
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
Minimality = Literal["structural-closure", "oracle-1-minimal", "bounded"]
SliceStatus = Literal["sliced", "bounded", "unknown"]


def _fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    span: Span = None,
) -> NoReturn:
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "declare reads/writes/depends_on or supply a replay oracle",
        note="runtime.causal_replay/slicing",
        span=span,
    )


@dataclass(frozen=True)
class CausalSlice:
    """A replayable causal slice for one failure event."""

    failure_id: str
    event_ids: tuple[str, ...]
    events: tuple[ReplayEvent, ...]
    mutations: tuple[str, ...]
    errors: tuple[str, ...]
    frontier: tuple[str, ...]
    pruned: int
    causal_past: int
    minimality: Minimality
    trials: int
    status: SliceStatus
    digest: str
    unknown_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "event_ids": list(self.event_ids),
            "events": [event.as_dict() for event in self.events],
            "mutations": list(self.mutations),
            "errors": list(self.errors),
            "frontier": list(self.frontier),
            "pruned": self.pruned,
            "causal_past": self.causal_past,
            "minimality": self.minimality,
            "trials": self.trials,
            "status": self.status,
            "digest": self.digest,
            "unknown_reason": self.unknown_reason,
            "caveats": (
                "a structural slice is causally closed but not claimed to be "
                "semantically minimal without a replay oracle",
                "the oracle is assumed to be deterministic and side-effect free",
                "oracle trials and wall-clock time are explicitly bounded",
            ),
        }


def _event_reads(event: ReplayEvent) -> tuple[str, ...]:
    return tuple(event.payload.get("reads", ()))


def _event_writes(event: ReplayEvent) -> tuple[str, ...]:
    return tuple(event.payload.get("writes", ()))


def _event_depends_on(event: ReplayEvent) -> tuple[str, ...]:
    return tuple(event.payload.get("depends_on", ()))


def _build_dependencies(
    graph: CausalGraph,
    *,
    span: Span,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return semantic dependency edges and the send lookup."""
    send_by_message: dict[str, str] = {}
    for event in graph.events:
        if event.kind == "send" and event.message_id is not None:
            send_by_message[event.message_id] = event.event_id

    last_writer: dict[tuple[str, str], str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for event in graph.events:
        deps: list[str] = []
        for key in _event_reads(event):
            writer = last_writer.get((event.actor, key))
            if writer is not None:
                deps.append(writer)
        deps.extend(_event_depends_on(event))
        if event.kind == "recv" and event.message_id is not None:
            send = send_by_message.get(event.message_id)
            if send is None:
                _fail(
                    "E006",
                    f"receive event {event.event_id!r} has no send event",
                    span=span,
                )
            deps.append(send)
        unique: list[str] = []
        for dep in deps:
            if dep == event.event_id:
                _fail(
                    "E005",
                    f"event {event.event_id!r} depends on itself",
                    span=event.span or span,
                )
            if dep not in graph.clocks:
                _fail(
                    "E006",
                    f"event {event.event_id!r} depends on unknown event {dep!r}",
                    span=event.span or span,
                )
            if not graph.happens_before(dep, event.event_id):
                _fail(
                    "E008",
                    f"dependency {dep!r} does not happen before {event.event_id!r}",
                    span=event.span or span,
                )
            if dep not in unique:
                unique.append(dep)
        dependencies[event.event_id] = tuple(unique)
        for key in _event_writes(event):
            last_writer[(event.actor, key)] = event.event_id
    return dependencies, send_by_message


def _closure(
    dependencies: Mapping[str, tuple[str, ...]],
    roots: Sequence[str],
) -> set[str]:
    selected: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in selected:
            continue
        selected.add(current)
        stack.extend(dependencies.get(current, ()))
    return selected


def _forward_closure(
    dependencies: Mapping[str, tuple[str, ...]],
    removed: str,
    selected: set[str],
    *,
    protected: str,
) -> set[str]:
    """Remove *removed* and every selected event that transitively needs it."""
    result = set(selected)
    result.discard(removed)
    changed = True
    while changed:
        changed = False
        for event_id in tuple(result):
            if event_id == protected:
                continue
            if any(
                dep == removed or dep not in result
                for dep in dependencies.get(event_id, ())
            ):
                result.discard(event_id)
                changed = True
    return result


def _slice_digest(
    event_ids: Sequence[str],
    events: Sequence[ReplayEvent],
) -> str:
    payload = {
        "event_ids": list(event_ids),
        "events": [event.as_dict() for event in events],
    }
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def replay_slice_minimal_failure_causality(
    graph: CausalGraph,
    failure_id: str,
    *,
    oracle: Callable[[Sequence[ReplayEvent]], bool] | None = None,
    max_trials: int = 256,
    timeout_s: float | None = 5.0,
    strict: bool = True,
    span: Span = None,
) -> CausalSlice:
    """Compute a replayable causal slice for *failure_id*.

    The failure must be an ``error`` or ``invariant`` event.  The initial slice
    is the structural dependency closure described in the module docstring.
    If ``oracle`` is supplied, it receives the slice events in topological
    order and returns true when the failure is reproduced.  A deterministic
    reverse-order removal pass then keeps only removals that preserve the
    oracle, up to ``max_trials`` and ``timeout_s``.

    A truncated graph, a trial/time bound, or an unverifiable slice is reported
    as ``bounded``/``unknown`` and raises ``E002`` when ``strict`` is true.
    """
    if max_trials <= 0:
        _fail("E003", "max_trials must be positive", span=span)
    if timeout_s is not None and timeout_s < 0:
        _fail("E003", "timeout_s must be non-negative", span=span)
    failure = graph.event(failure_id)
    if failure.kind not in {"error", "invariant"}:
        _fail(
            "E003",
            f"event {failure_id!r} is not a failure or invariant event",
            reason="slicing starts from an error or invariant violation",
            span=span,
        )
    if graph.truncated:
        if strict:
            _fail(
                "E002",
                "cannot slice a truncated causal graph",
                reason=graph.unknown_reason or "recording budget was exhausted",
                span=span,
            )
        return CausalSlice(
            failure_id=failure_id,
            event_ids=(failure_id,),
            events=(failure,),
            mutations=(),
            errors=(failure_id,) if failure.kind == "error" else (),
            frontier=(),
            pruned=0,
            causal_past=0,
            minimality="bounded",
            trials=0,
            status="unknown",
            digest=_slice_digest((failure_id,), (failure,)),
            unknown_reason=graph.unknown_reason,
        )

    dependencies, _ = _build_dependencies(graph, span=span)
    selected = _closure(dependencies, (failure_id,))
    positions = {event_id: index for index, event_id in enumerate(graph.order)}
    selected_order = tuple(sorted(selected, key=positions.__getitem__))
    events_by_id = {event.event_id: event for event in graph.events}
    causal_past = len(graph.ancestors(failure_id))
    frontier = tuple(
        dep for dep in dependencies.get(failure_id, ()) if dep != failure_id
    )

    trials = 0
    status: SliceStatus = "sliced"
    minimality: Minimality = "structural-closure"
    unknown_reason: str | None = None
    current = set(selected)

    if oracle is not None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        minimality = "oracle-1-minimal"
        for event_id in reversed(selected_order):
            if event_id == failure_id or event_id not in current:
                continue
            if trials >= max_trials:
                status = "bounded"
                minimality = "bounded"
                unknown_reason = f"oracle trial bound max_trials={max_trials} exhausted"
                break
            if deadline is not None and time.monotonic() > deadline:
                status = "bounded"
                minimality = "bounded"
                unknown_reason = f"oracle timeout after {timeout_s:.3f}s"
                break
            candidate = _forward_closure(
                dependencies, event_id, current, protected=failure_id
            )
            candidate_order = tuple(sorted(candidate, key=positions.__getitem__))
            trials += 1
            try:
                reproduced = bool(
                    oracle(tuple(events_by_id[item] for item in candidate_order))
                )
            except Exception as error:
                _fail(
                    "E004",
                    "replay oracle raised an exception",
                    reason=f"{type(error).__name__}: {error}",
                    span=span,
                )
            if reproduced:
                current = candidate
        if status == "sliced":
            unknown_reason = None

    final_order = tuple(sorted(current, key=positions.__getitem__))
    final_events = tuple(events_by_id[item] for item in final_order)
    mutations = tuple(
        event.event_id for event in final_events if event.kind == "state_mutation"
    )
    errors = tuple(event.event_id for event in final_events if event.kind == "error")
    report = CausalSlice(
        failure_id=failure_id,
        event_ids=final_order,
        events=final_events,
        mutations=mutations,
        errors=errors,
        frontier=frontier,
        pruned=max(0, causal_past + 1 - len(final_order)),
        causal_past=causal_past,
        minimality=minimality,
        trials=trials,
        status=status,
        digest=_slice_digest(final_order, final_events),
        unknown_reason=unknown_reason,
    )
    if strict and status in {"bounded", "unknown"}:
        _fail(
            "E002",
            "minimal causal slice is Unknown",
            reason=unknown_reason or "bounded slicing did not complete",
            help_text="raise max_trials/timeout_s or reduce the trace",
            span=span,
        )
    return report


__all__ = [
    "CausalSlice",
    "Minimality",
    "SliceStatus",
    "replay_slice_minimal_failure_causality",
]
