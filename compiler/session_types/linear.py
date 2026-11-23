"""Linear and affine channel/capability consumption checking.

Channels are capabilities: a linear channel must be acquired once, used
according to its protocol, and closed exactly once.  An affine channel may be
dropped instead of closed, but it can never be duplicated.  Shared channels
may be duplicated, but they still cannot be used after close.

The checker records a structured trace and raises the shared ``E002``
diagnostic for linearity violations, ``E006`` for an unknown channel, and
``E003`` for API misuse.  Optional expected local session types turn the
event sequence into a protocol-conformance check, including branch labels and
recursion unrolling with an explicit ``Unknown`` bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from compiler.session_types.protocols import (
    LocalEnd,
    LocalRec,
    LocalRecv,
    LocalSend,
    LocalType,
    LocalVar,
    _normalize,
    _validate_local,
)
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
Linearity = Literal["linear", "affine", "shared"]
ChannelKind = Literal["channel", "capability"]
EventOp = Literal[
    "open", "send", "recv", "close", "release", "duplicate", "drop", "use"
]
Status = Literal["verified", "violation", "unknown"]

_OPEN_OPS: frozenset[EventOp] = frozenset({"send", "recv", "use"})
_CLOSE_OPS: frozenset[EventOp] = frozenset({"close", "release"})
_VALID_OPS: frozenset[str] = frozenset(
    {"open", "send", "recv", "close", "release", "duplicate", "drop", "use"}
)
_VALID_LINEARITY: frozenset[str] = frozenset({"linear", "affine", "shared"})
_VALID_KIND: frozenset[str] = frozenset({"channel", "capability"})


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
        or "close the channel once, or use an affine/shared capability",
        note="compiler.session_types/linear",
        span=span,
    )


def _check_text(value: object, label: str, *, span: Span = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("E003", f"{label} must be a non-empty string", span=span)
    return value


@dataclass(frozen=True)
class ChannelDecl:
    """A declared channel or capability endpoint."""

    name: str
    role: str = "default"
    linearity: Linearity = "linear"
    kind: ChannelKind = "channel"
    span: Span = None

    def __post_init__(self) -> None:
        _check_text(self.name, "channel name", span=self.span)
        _check_text(self.role, "channel role", span=self.span)
        if self.linearity not in _VALID_LINEARITY:
            _fail(
                "E003",
                f"unknown linearity {self.linearity!r}",
                reason="linearity must be linear, affine, or shared",
                span=self.span,
            )
        if self.kind not in _VALID_KIND:
            _fail(
                "E003",
                f"unknown channel kind {self.kind!r}",
                reason="kind must be channel or capability",
                span=self.span,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "linearity": self.linearity,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ChannelEvent:
    """One operation on a channel/capability."""

    op: EventOp
    channel: str
    label: str | None = None
    peer: str | None = None
    role: str | None = None
    count: int = 1
    span: Span = None

    def __post_init__(self) -> None:
        _check_text(self.channel, "channel name", span=self.span)
        if self.op not in _VALID_OPS:
            _fail("E003", f"unknown channel operation {self.op!r}", span=self.span)
        if self.count < 1:
            _fail("E003", "event count must be at least one", span=self.span)
        if self.role is not None:
            _check_text(self.role, "event role", span=self.span)
        if self.label is not None:
            _check_text(self.label, "event label", span=self.span)
        if self.peer is not None:
            _check_text(self.peer, "event peer", span=self.span)

    def describe(self, index: int) -> str:
        parts = [str(index), self.op, self.channel]
        if self.role is not None:
            parts.append(f"role={self.role}")
        if self.label is not None:
            parts.append(f"label={self.label}")
        if self.peer is not None:
            parts.append(f"peer={self.peer}")
        return ":".join(parts)


@dataclass(frozen=True)
class ChannelViolation:
    code: str
    message: str
    event_index: int
    channel: str
    trace: tuple[str, ...]
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "event_index": self.event_index,
            "channel": self.channel,
            "trace": list(self.trace),
            "span": list(self.span) if self.span is not None else None,
        }


@dataclass(frozen=True)
class ChannelUsage:
    name: str
    role: str
    kind: ChannelKind
    linearity: Linearity
    opens: int
    sends: int
    recvs: int
    duplicates: int
    closed: bool
    dropped: bool
    leaked: bool

    @property
    def state(self) -> str:
        if self.leaked:
            return "leaked"
        if self.dropped:
            return "dropped"
        if self.closed:
            return "closed"
        if self.opens:
            return "open"
        return "unopened"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "kind": self.kind,
            "linearity": self.linearity,
            "opens": self.opens,
            "sends": self.sends,
            "recvs": self.recvs,
            "duplicates": self.duplicates,
            "closed": self.closed,
            "dropped": self.dropped,
            "leaked": self.leaked,
            "state": self.state,
        }


@dataclass(frozen=True)
class LinearChannelReport:
    channels: tuple[ChannelUsage, ...]
    violations: tuple[ChannelViolation, ...]
    trace: tuple[str, ...]
    expected_roles: tuple[str, ...]
    status: Status
    unknown_reason: str | None
    conforms: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": [item.as_dict() for item in self.channels],
            "violations": [item.as_dict() for item in self.violations],
            "trace": list(self.trace),
            "expected_roles": list(self.expected_roles),
            "status": self.status,
            "unknown_reason": self.unknown_reason,
            "conforms": self.conforms,
            "caveats": (
                "linearity is checked for the recorded event sequence only",
                "an omitted close for a linear channel is reported as a leak",
                "protocol conformance is bounded by max_unroll",
            ),
        }


@dataclass
class _Usage:
    decl: ChannelDecl
    opens: int = 0
    sends: int = 0
    recvs: int = 0
    duplicates: int = 0
    closed: bool = False
    dropped: bool = False


def _add_violation(
    violations: list[ChannelViolation],
    *,
    code: str,
    message: str,
    index: int,
    channel: str,
    trace: Sequence[str],
    span: Span,
) -> None:
    violations.append(
        ChannelViolation(
            code=code,
            message=message,
            event_index=index,
            channel=channel,
            trace=tuple(trace),
            span=span,
        )
    )


def _advance_expected(
    state: LocalType,
    env: Mapping[str, LocalType],
    event: ChannelEvent,
    *,
    max_unroll: int,
    index: int,
    trace: Sequence[str],
    violations: list[ChannelViolation],
    span: Span,
) -> tuple[LocalType, dict[str, LocalType]] | None:
    head, new_env = _normalize(state, env, max_unroll=max_unroll, span=span)
    if head is None:
        _add_violation(
            violations,
            code="E002",
            message="protocol conformance is Unknown: recursion unrolling bound exhausted",
            index=index,
            channel=event.channel,
            trace=trace,
            span=span,
        )
        return None
    if isinstance(head, LocalEnd):
        if event.op in _OPEN_OPS - {"use"}:
            _add_violation(
                violations,
                code="E002",
                message=f"channel {event.channel!r} used after its protocol completed",
                index=index,
                channel=event.channel,
                trace=trace,
                span=span,
            )
        return head, new_env
    if not isinstance(head, LocalSend | LocalRecv):
        _add_violation(
            violations,
            code="E002",
            message=f"unexpected local protocol head {type(head).__name__}",
            index=index,
            channel=event.channel,
            trace=trace,
            span=span,
        )
        return None
    if event.op == "use":
        return head, new_env
    expected_op = "send" if isinstance(head, LocalSend) else "recv"
    if event.op != expected_op:
        _add_violation(
            violations,
            code="E002",
            message=(
                f"protocol expects {expected_op} on channel {event.channel!r}, "
                f"got {event.op}"
            ),
            index=index,
            channel=event.channel,
            trace=trace,
            span=span,
        )
        return None
    if event.peer is not None and event.peer != head.peer:
        _add_violation(
            violations,
            code="E002",
            message=(
                f"protocol expects peer {head.peer!r} on channel "
                f"{event.channel!r}, got {event.peer!r}"
            ),
            index=index,
            channel=event.channel,
            trace=trace,
            span=span,
        )
        return None
    branches = {branch.label: branch.protocol for branch in head.branches}
    label = event.label
    if label is None:
        if len(branches) != 1:
            _add_violation(
                violations,
                code="E003",
                message=(
                    f"channel {event.channel!r} requires an explicit branch label "
                    f"from {sorted(branches)}"
                ),
                index=index,
                channel=event.channel,
                trace=trace,
                span=span,
            )
            return None
        label = next(iter(branches))
    if label not in branches:
        _add_violation(
            violations,
            code="E002",
            message=(
                f"label {label!r} is not offered on channel {event.channel!r}; "
                f"expected one of {sorted(branches)}"
            ),
            index=index,
            channel=event.channel,
            trace=trace,
            span=span,
        )
        return None
    return branches[label], new_env


def session_verify_linear_channel_consumption(
    declarations: Sequence[ChannelDecl],
    events: Sequence[ChannelEvent],
    *,
    expected: Mapping[str, LocalType] | None = None,
    max_unroll: int = 64,
    strict: bool = True,
    span: Span = None,
) -> LinearChannelReport:
    """Verify linear/affine channel and capability consumption.

    ``declarations`` names every channel and its discipline.  ``events`` is
    the recorded sequence of operations.  ``expected`` optionally maps a role
    to the local session type its channel must follow; the event sequence is
    then checked for send/receive direction, peer, branch labels, and protocol
    completion.

    Violations are accumulated in the report and the first one is raised as a
    :class:`~runtime.diagnostics.PillarError` when ``strict`` is true.  A
    recursion-bound exhaustion is reported as ``Unknown`` (``E002``), never as
    a successful verification.
    """
    if max_unroll <= 0:
        _fail("E003", "max_unroll must be positive", span=span)
    if not declarations:
        _fail("E003", "at least one channel declaration is required", span=span)

    declared: dict[str, ChannelDecl] = {}
    for decl in declarations:
        if not isinstance(decl, ChannelDecl):
            _fail("E003", "declarations must contain ChannelDecl values", span=span)
        if decl.name in declared:
            _fail(
                "E003",
                f"channel {decl.name!r} is declared more than once",
                span=span,
            )
        declared[decl.name] = decl

    roles = frozenset(decl.role for decl in declared.values())
    expected_map = dict(expected or {})
    for role, local in expected_map.items():
        _check_text(role, "expected role", span=span)
        if role not in roles:
            _fail(
                "E006",
                f"expected protocol role {role!r} has no channel declaration",
                span=span,
            )
        if not isinstance(
            local, LocalEnd | LocalVar | LocalRec | LocalSend | LocalRecv
        ):
            _fail(
                "E003",
                f"expected protocol for {role!r} is not a local type",
                span=span,
            )
        _validate_local(local, roles, span=span)

    usages = {name: _Usage(decl=decl) for name, decl in declared.items()}
    violations: list[ChannelViolation] = []
    trace: list[str] = []
    exp_states: dict[str, LocalType] = dict(expected_map)
    exp_envs: dict[str, dict[str, LocalType]] = {role: {} for role in expected_map}
    exp_closed: dict[str, bool] = {role: False for role in expected_map}
    unknown_reason: str | None = None

    for index, event in enumerate(events):
        if not isinstance(event, ChannelEvent):
            _fail("E003", "events must contain ChannelEvent values", span=span)
        trace.append(event.describe(index))
        usage = usages.get(event.channel)
        if usage is None:
            _add_violation(
                violations,
                code="E006",
                message=f"unknown channel {event.channel!r}",
                index=index,
                channel=event.channel,
                trace=trace,
                span=event.span or span,
            )
            continue
        decl = usage.decl
        role = event.role or decl.role
        if event.role is not None and event.role != decl.role:
            _add_violation(
                violations,
                code="E001",
                message=(
                    f"event role {event.role!r} does not match channel role "
                    f"{decl.role!r} for {event.channel!r}"
                ),
                index=index,
                channel=event.channel,
                trace=trace,
                span=event.span or span,
            )
            continue

        if event.op == "open":
            if usage.opens or usage.closed or usage.dropped:
                _add_violation(
                    violations,
                    code="E002",
                    message=(
                        f"channel {event.channel!r} is reopened; a capability "
                        "cannot be silently duplicated"
                    ),
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            usage.opens += 1
        elif event.op in _OPEN_OPS:
            if not usage.opens:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} used before open",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            elif usage.closed or usage.dropped:
                verb = "read after close" if event.op == "recv" else "use after close"
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r}: {verb}",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            else:
                if event.op == "send":
                    usage.sends += 1
                elif event.op == "recv":
                    usage.recvs += 1
        elif event.op in _CLOSE_OPS:
            if not usage.opens:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} closed before open",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            elif usage.closed or usage.dropped:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} is closed more than once",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            else:
                usage.closed = True
        elif event.op == "duplicate":
            if decl.linearity != "shared":
                _add_violation(
                    violations,
                    code="E002",
                    message=(
                        f"{decl.linearity} channel {event.channel!r} cannot be "
                        "duplicated"
                    ),
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            if not usage.opens or usage.closed or usage.dropped:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} duplicated while not open",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            usage.duplicates += event.count
        elif event.op == "drop":
            if not usage.opens:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} dropped before open",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            elif usage.closed or usage.dropped:
                _add_violation(
                    violations,
                    code="E002",
                    message=f"channel {event.channel!r} dropped after close",
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            elif decl.linearity == "linear":
                _add_violation(
                    violations,
                    code="E002",
                    message=(
                        f"linear channel {event.channel!r} cannot be dropped; "
                        "close it exactly once"
                    ),
                    index=index,
                    channel=event.channel,
                    trace=trace,
                    span=event.span or span,
                )
            else:
                usage.dropped = True

        # Optional local session-type conformance for this role.
        if role in exp_states:
            if event.op in _OPEN_OPS:
                advanced = _advance_expected(
                    exp_states[role],
                    exp_envs[role],
                    event,
                    max_unroll=max_unroll,
                    index=index,
                    trace=trace,
                    violations=violations,
                    span=event.span or span,
                )
                if advanced is None:
                    unknown_reason = unknown_reason or "protocol conformance Unknown"
                else:
                    exp_states[role], exp_envs[role] = advanced
            elif event.op in _CLOSE_OPS or event.op == "drop":
                head, new_env = _normalize(
                    exp_states[role],
                    exp_envs[role],
                    max_unroll=max_unroll,
                    span=event.span or span,
                )
                if head is None:
                    unknown_reason = unknown_reason or "protocol conformance Unknown"
                    _add_violation(
                        violations,
                        code="E002",
                        message="protocol conformance is Unknown: recursion unrolling bound exhausted",
                        index=index,
                        channel=event.channel,
                        trace=trace,
                        span=event.span or span,
                    )
                elif not isinstance(head, LocalEnd):
                    _add_violation(
                        violations,
                        code="E002",
                        message=(
                            f"channel {event.channel!r} is closed before its "
                            "protocol reaches end"
                        ),
                        index=index,
                        channel=event.channel,
                        trace=trace,
                        span=event.span or span,
                    )
                else:
                    exp_states[role] = head
                    exp_envs[role] = new_env
                    exp_closed[role] = True

    # End-of-trace checks.
    for name, usage in sorted(usages.items()):
        if (
            usage.decl.linearity == "linear"
            and usage.opens
            and not usage.closed
            and not usage.dropped
        ):
            _add_violation(
                violations,
                code="E002",
                message=(
                    f"linear {usage.decl.kind} {name!r} is leaked; it was opened "
                    "but never closed"
                ),
                index=len(events),
                channel=name,
                trace=trace,
                span=usage.decl.span or span,
            )

    for role, state in sorted(exp_states.items()):
        head, _ = _normalize(state, exp_envs[role], max_unroll=max_unroll, span=span)
        if head is None:
            unknown_reason = unknown_reason or "protocol conformance Unknown"
            continue
        if not isinstance(head, LocalEnd):
            _add_violation(
                violations,
                code="E002",
                message=(
                    f"local protocol for role {role!r} is incomplete; "
                    f"remaining head is {type(head).__name__}"
                ),
                index=len(events),
                channel=role,
                trace=trace,
                span=span,
            )
        elif role not in exp_closed and any(
            decl.role == role and decl.linearity == "linear"
            for decl in declared.values()
        ):
            # A linear endpoint that reached end without a close is already
            # reported by the leak check above; do not duplicate it here.
            continue

    status: Status
    if unknown_reason is not None:
        status = "unknown"
    elif violations:
        status = "violation"
    else:
        status = "verified"
    report = LinearChannelReport(
        channels=tuple(
            ChannelUsage(
                name=name,
                role=usage.decl.role,
                kind=usage.decl.kind,
                linearity=usage.decl.linearity,
                opens=usage.opens,
                sends=usage.sends,
                recvs=usage.recvs,
                duplicates=usage.duplicates,
                closed=usage.closed,
                dropped=usage.dropped,
                leaked=(
                    usage.decl.linearity == "linear"
                    and usage.opens
                    and not usage.closed
                    and not usage.dropped
                ),
            )
            for name, usage in sorted(usages.items())
        ),
        violations=tuple(violations),
        trace=tuple(trace),
        expected_roles=tuple(sorted(expected_map)),
        status=status,
        unknown_reason=unknown_reason,
        conforms=not violations,
    )
    if strict and violations:
        first = violations[0]
        trace_text = " -> ".join(first.trace) or "<empty trace>"
        _fail(
            first.code,
            first.message,
            reason=f"event {first.event_index}; trace: {trace_text}",
            help_text="use exactly one owner, close linear channels once, and "
            "never use a channel after close",
            span=first.span or span,
        )
    if strict and unknown_reason is not None and not violations:
        _fail(
            "E002",
            "linear channel verification is Unknown",
            reason=unknown_reason,
            help_text="increase max_unroll or simplify the local protocol",
            span=span,
        )
    return report


__all__ = [
    "ChannelDecl",
    "ChannelEvent",
    "ChannelUsage",
    "ChannelViolation",
    "LinearChannelReport",
    "session_verify_linear_channel_consumption",
]
