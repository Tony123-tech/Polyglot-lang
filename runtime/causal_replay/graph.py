"""Deterministic causal event recording with vector and Lamport clocks.

The recorder never orders events by wall-clock time.  It builds causality from
three explicit sources:

* program order inside an actor (the previous event of the same actor);
* message edges (a ``send`` event with a ``message_id`` happens before the
  matching ``recv``);
* caller-declared ``parents`` and payload-level ``depends_on`` edges.

Each event receives a vector clock (the authoritative partial order) and a
Lamport clock (a compact scalar summary).  Bounded event/payload budgets are
explicit: exhausting them reports ``Unknown``/``E002`` rather than silently
truncating a causal graph.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn

from runtime.diagnostics import ERROR_CODES, PillarError

Span = tuple[int, int] | None
EventKind = Literal[
    "send",
    "recv",
    "state_mutation",
    "error",
    "invariant",
    "checkpoint",
    "external",
    "custom",
]
_VALID_KINDS: frozenset[str] = frozenset(
    {
        "send",
        "recv",
        "state_mutation",
        "error",
        "invariant",
        "checkpoint",
        "external",
        "custom",
    }
)


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
        or "record explicit causal edges and retry within the bound",
        note="runtime.causal_replay/graph",
        span=span,
    )


def _text(value: object, label: str, *, span: Span = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("E003", f"{label} must be a non-empty string", span=span)
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=repr)
        return items
    return repr(value)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _string_tuple(value: object, label: str, *, span: Span = None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        _fail("E008", f"{label} must be a sequence of strings", span=span)
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            _fail("E008", f"{label} must contain non-empty strings", span=span)
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class ReplayEvent:
    """One recorded event in a causal trace.

    ``timestamp`` is deliberately retained only for display.  It never
    participates in ordering.  Use ``parents``, ``message_id``, and the
    ``reads``/``writes``/``depends_on`` payload keys to declare causality.
    """

    event_id: str
    actor: str
    kind: EventKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    parents: tuple[str, ...] = ()
    message_id: str | None = None
    target: str | None = None
    span: Span = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id", span=self.span)
        _text(self.actor, "actor", span=self.span)
        if self.kind not in _VALID_KINDS:
            _fail("E003", f"unknown event kind {self.kind!r}", span=self.span)
        if self.message_id is not None:
            _text(self.message_id, "message_id", span=self.span)
        if self.target is not None:
            _text(self.target, "target", span=self.span)
        for parent in self.parents:
            _text(parent, "parent event id", span=self.span)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "actor": self.actor,
            "kind": self.kind,
            "payload": _jsonable(self.payload),
            "parents": list(self.parents),
            "message_id": self.message_id,
            "target": self.target,
            "span": list(self.span) if self.span is not None else None,
            "timestamp": self.timestamp,
        }


VectorClock = Mapping[str, int]


@dataclass(frozen=True)
class CausalGraph:
    """A recorded causal graph with vector clocks and explicit bounds."""

    events: tuple[ReplayEvent, ...]
    clocks: Mapping[str, VectorClock]
    lamports: Mapping[str, int]
    parents: Mapping[str, tuple[str, ...]]
    children: Mapping[str, tuple[str, ...]]
    order: tuple[str, ...]
    digest: str
    truncated: bool = False
    unknown_reason: str | None = None

    def event(self, event_id: str) -> ReplayEvent:
        for event in self.events:
            if event.event_id == event_id:
                return event
        _fail("E006", f"unknown event {event_id!r}")

    def happens_before(self, left: str, right: str) -> bool:
        """Return true when *left* causally precedes *right*.

        The vector clock is authoritative.  Equal or incomparable clocks are
        concurrent, not ordered.
        """
        if left == right:
            return False
        if left not in self.clocks:
            _fail("E006", f"unknown event {left!r}")
        if right not in self.clocks:
            _fail("E006", f"unknown event {right!r}")
        left_clock = self.clocks[left]
        right_clock = self.clocks[right]
        dominated = all(
            value <= right_clock.get(actor, 0) for actor, value in left_clock.items()
        )
        return dominated and dict(left_clock) != dict(right_clock)

    def concurrent(self, left: str, right: str) -> bool:
        return (
            left != right
            and not self.happens_before(left, right)
            and not self.happens_before(right, left)
        )

    def ancestors(
        self,
        event_id: str,
        *,
        include_self: bool = False,
        max_nodes: int = 100_000,
    ) -> tuple[str, ...]:
        """Return the causal past of *event_id* in deterministic order."""
        if event_id not in self.clocks:
            _fail("E006", f"unknown event {event_id!r}")
        seen: set[str] = {event_id} if include_self else set()
        stack = [event_id]
        while stack:
            current = stack.pop()
            for parent in self.parents.get(current, ()):
                if parent in seen:
                    continue
                seen.add(parent)
                if len(seen) > max_nodes:
                    _fail(
                        "E002",
                        "causal ancestor closure exceeded its bound",
                        reason=f"max_nodes={max_nodes}",
                    )
                stack.append(parent)
        positions = {event_id: index for index, event_id in enumerate(self.order)}
        return tuple(sorted(seen, key=positions.__getitem__))

    def descendants(
        self,
        event_id: str,
        *,
        include_self: bool = False,
        max_nodes: int = 100_000,
    ) -> tuple[str, ...]:
        if event_id not in self.clocks:
            _fail("E006", f"unknown event {event_id!r}")
        seen: set[str] = {event_id} if include_self else set()
        stack = [event_id]
        while stack:
            current = stack.pop()
            for child in self.children.get(current, ()):
                if child in seen:
                    continue
                seen.add(child)
                if len(seen) > max_nodes:
                    _fail("E002", "causal descendant closure exceeded its bound")
                stack.append(child)
        positions = {event_id: index for index, event_id in enumerate(self.order)}
        return tuple(sorted(seen, key=positions.__getitem__))

    def topological_order(self) -> tuple[str, ...]:
        return self.order

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "clocks": {
                event_id: dict(clock) for event_id, clock in sorted(self.clocks.items())
            },
            "lamports": {
                event_id: value for event_id, value in sorted(self.lamports.items())
            },
            "parents": {
                event_id: list(parents)
                for event_id, parents in sorted(self.parents.items())
            },
            "children": {
                event_id: list(children)
                for event_id, children in sorted(self.children.items())
            },
            "order": list(self.order),
            "digest": self.digest,
            "truncated": self.truncated,
            "unknown_reason": self.unknown_reason,
            "caveats": (
                "vector clocks record declared, program-order, and message causality",
                "wall-clock timestamps are recorded but never used for ordering",
                "external I/O and OS scheduling are not captured unless modeled as events",
                "bounded by max_events and max_payload_bytes; exhaustion reports Unknown",
            ),
        }


def _payload_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    )


def _validate_payload(
    event: ReplayEvent,
    *,
    max_payload_bytes: int,
    span: Span,
) -> None:
    size = _payload_size(event.payload)
    if size > max_payload_bytes:
        _fail(
            "E002",
            f"event {event.event_id!r} payload exceeds max_payload_bytes",
            reason=f"{size} > {max_payload_bytes}",
            span=event.span or span,
        )
    for key in ("reads", "writes", "depends_on"):
        _string_tuple(event.payload.get(key), f"payload[{key!r}]", span=span)
    if event.kind == "error":
        code = event.payload.get("code")
        if not isinstance(code, str) or code not in ERROR_CODES:
            _fail(
                "E008",
                f"error event {event.event_id!r} has no valid E001-E008 code",
                reason="error payloads must carry one of the shared diagnostic codes",
                span=event.span or span,
            )


def _merge_clock(target: dict[str, int], source: Mapping[str, int]) -> None:
    for actor, value in source.items():
        if value > target.get(actor, 0):
            target[actor] = value


def replay_record_causal_happens_before(
    events: Sequence[ReplayEvent],
    *,
    max_events: int = 10_000,
    max_payload_bytes: int = 65_536,
    strict: bool = True,
    span: Span = None,
) -> CausalGraph:
    """Record a deterministic causal graph from a sequence of events.

    The input order is treated as the recording order and must be a
    topological order of the declared causality: a parent, a message send, or
    a prior local event must appear before its dependent event.  The recorder
    computes vector and Lamport clocks and returns a graph whose
    :meth:`CausalGraph.happens_before` relation is authoritative.

    ``max_events`` and ``max_payload_bytes`` are explicit bounds.  Exceeding
    them raises ``E002`` when ``strict`` is true; otherwise the graph is
    returned with ``truncated=True`` and an ``unknown_reason``.  Wall-clock
    timestamps are never consulted.
    """
    if max_events <= 0 or max_payload_bytes <= 0:
        _fail("E003", "recording bounds must be positive", span=span)
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        _fail("E003", "events must be a sequence of ReplayEvent values", span=span)

    truncated = False
    unknown_reason: str | None = None
    selected: Sequence[ReplayEvent] = events
    if len(events) > max_events:
        if strict:
            _fail(
                "E002",
                "causal trace exceeds max_events",
                reason=f"{len(events)} > {max_events}; the trace is bounded",
                span=span,
            )
        selected = events[:max_events]
        truncated = True
        unknown_reason = f"trace truncated at max_events={max_events}"

    index_by_id: dict[str, int] = {}
    event_by_id: dict[str, ReplayEvent] = {}
    clocks: dict[str, dict[str, int]] = {}
    lamports: dict[str, int] = {}
    parents_map: dict[str, tuple[str, ...]] = {}
    actor_clocks: dict[str, dict[str, int]] = {}
    actor_lamports: dict[str, int] = {}
    actor_last: dict[str, str] = {}
    sends: dict[str, tuple[str, str, dict[str, int], int]] = {}

    for index, event in enumerate(selected):
        if not isinstance(event, ReplayEvent):
            _fail("E003", "events must contain ReplayEvent values", span=span)
        if event.event_id in index_by_id:
            _fail(
                "E008",
                f"duplicate event id {event.event_id!r}",
                reason="event ids must be unique",
                span=event.span or span,
            )
        _validate_payload(event, max_payload_bytes=max_payload_bytes, span=span)

        base = dict(actor_clocks.get(event.actor, {}))
        lamport_base = actor_lamports.get(event.actor, 0)
        explicit: list[str] = list(event.parents)
        if event.kind == "recv":
            if event.message_id is None:
                _fail(
                    "E008",
                    f"receive event {event.event_id!r} has no message_id",
                    span=event.span or span,
                )
            record = sends.get(event.message_id)
            if record is None:
                _fail(
                    "E006",
                    f"receive event {event.event_id!r} has no recorded send",
                    reason=f"message_id={event.message_id!r}",
                    span=event.span or span,
                )
            send_id, send_actor, send_clock, send_lamport = record
            if event.target is not None and event.target != send_actor:
                _fail(
                    "E008",
                    f"receive event {event.event_id!r} names the wrong sender",
                    reason=f"target={event.target!r}, actual={send_actor!r}",
                    span=event.span or span,
                )
            _merge_clock(base, send_clock)
            lamport_base = max(lamport_base, send_lamport)
            explicit.append(send_id)
        elif event.kind == "send":
            if event.message_id is None:
                _fail(
                    "E008",
                    f"send event {event.event_id!r} has no message_id",
                    span=event.span or span,
                )
            if event.message_id in sends:
                _fail(
                    "E008",
                    f"message_id {event.message_id!r} is reused",
                    reason="a message has exactly one send event",
                    span=event.span or span,
                )
            if event.target is None:
                _fail(
                    "E008",
                    f"send event {event.event_id!r} has no target actor",
                    span=event.span or span,
                )

        if event.actor in actor_last:
            explicit.append(actor_last[event.actor])
        explicit.extend(event.payload.get("depends_on", ()))

        unique_parents: list[str] = []
        for parent in explicit:
            if parent == event.event_id:
                _fail(
                    "E005",
                    f"event {event.event_id!r} depends on itself",
                    span=event.span or span,
                )
            if parent not in index_by_id:
                _fail(
                    "E006",
                    f"event {event.event_id!r} depends on unknown event {parent!r}",
                    span=event.span or span,
                )
            if index_by_id[parent] >= index:
                _fail(
                    "E008",
                    f"event {event.event_id!r} depends on a non-prior event",
                    reason=f"parent {parent!r} is not recorded before its child",
                    span=event.span or span,
                )
            if parent not in unique_parents:
                unique_parents.append(parent)
            _merge_clock(base, clocks[parent])
            lamport_base = max(lamport_base, lamports[parent])

        clock = dict(base)
        clock[event.actor] = max(clock.get(event.actor, 0), 0) + 1
        lamport = lamport_base + 1
        for parent in unique_parents:
            parent_clock = clocks[parent]
            if not all(
                value <= clock.get(actor, 0) for actor, value in parent_clock.items()
            ):
                _fail(
                    "E008",
                    f"vector clock for {event.event_id!r} does not dominate its parent",
                    reason=f"parent={parent!r}",
                    span=event.span or span,
                )

        index_by_id[event.event_id] = index
        event_by_id[event.event_id] = event
        clocks[event.event_id] = clock
        lamports[event.event_id] = lamport
        parents_map[event.event_id] = tuple(unique_parents)
        actor_clocks[event.actor] = clock
        actor_lamports[event.actor] = lamport
        actor_last[event.actor] = event.event_id
        if event.kind == "send":
            assert event.message_id is not None
            assert event.target is not None
            sends[event.message_id] = (
                event.event_id,
                event.actor,
                clock,
                lamport,
            )

    children_lists: dict[str, list[str]] = {event_id: [] for event_id in event_by_id}
    for event_id, parents in parents_map.items():
        for parent in parents:
            children_lists[parent].append(event_id)
    children = {
        event_id: tuple(children) for event_id, children in children_lists.items()
    }
    order = tuple(event.event_id for event in selected)
    payload = {
        "events": [event.as_dict() for event in selected],
        "clocks": clocks,
        "lamports": lamports,
        "parents": parents_map,
        "children": children,
        "order": order,
    }
    return CausalGraph(
        events=tuple(selected),
        clocks=clocks,
        lamports=lamports,
        parents=parents_map,
        children=children,
        order=order,
        digest=_digest(payload),
        truncated=truncated,
        unknown_reason=unknown_reason,
    )


__all__ = [
    "CausalGraph",
    "EventKind",
    "ReplayEvent",
    "VectorClock",
    "replay_record_causal_happens_before",
]
