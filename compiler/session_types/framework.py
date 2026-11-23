"""Bounded multiparty session types, projection, and linear channel checking.

The implementation is intentionally *bounded* and *model based*:

* global protocols are finite trees over messages, choices, and bounded
  repetition; recursion is not accepted as an unbounded fixed point;
* choices are expanded into a bounded number of linear variants, so the
  deadlock/progress prover never claims a result after silently truncating the
  variant space;
* deadlock freedom is checked under a synchronous rendezvous model, with
  explicit ``max_states``, ``max_steps``, and timeout budgets;
* linear channel consumption is a concrete event trace, not a claim about
  arbitrary future programs.

Every failure is reported through the shared ``E001``–``E008`` surface.  A
``strict`` call raises :class:`runtime.diagnostics.PillarError`; the default
returns a report containing the same diagnostic and protocol trace.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from runtime.diagnostics import PillarError

__all__ = [
    "ChoiceBranch",
    "ChoiceStep",
    "EndStep",
    "GlobalProtocol",
    "LinearChannelReport",
    "LinearConsumptionReport",
    "LinearDiagnostic",
    "LinearOperation",
    "LocalAction",
    "LocalEndpoint",
    "MessageStep",
    "RepeatStep",
    "SessionDiagnostic",
    "SessionProjectionReport",
    "session_synthesize_and_project_protocols",
    "session_verify_linear_channel_consumption",
]

Span = tuple[int, int] | None
Severity = Literal["error", "warning", "unknown"]
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_MAX_REPEAT = 64
_MAX_ROLES = 32
_KNOWN_KINDS = frozenset(
    {
        "open",
        "acquire",
        "send",
        "write",
        "recv",
        "read",
        "close",
        "release",
        "drop",
        "split",
        "duplicate",
        "clone",
    }
)


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
    note: str = "compiler.session_types",
) -> None:
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "provide a finite protocol, a larger bound, or a valid linear trace",
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
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Protocol model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageStep:
    """One directed message in a global protocol."""

    sender: str
    receiver: str
    label: str
    payload: str = "unit"


@dataclass(frozen=True)
class ChoiceBranch:
    """One labelled branch of a global choice."""

    label: str
    steps: tuple[Any, ...]


@dataclass(frozen=True)
class ChoiceStep:
    """A role-directed finite choice."""

    chooser: str
    branches: tuple[ChoiceBranch, ...]


@dataclass(frozen=True)
class RepeatStep:
    """A bounded repetition of a finite protocol body."""

    times: int
    body: tuple[Any, ...]


@dataclass(frozen=True)
class EndStep:
    """Explicit successful termination."""


Step = MessageStep | ChoiceStep | RepeatStep | EndStep


@dataclass(frozen=True)
class GlobalProtocol:
    """A finite global protocol over named roles.

    ``locals`` optionally provides a per-role endpoint specification instead of
    (or in addition to) a global sequence.  Local endpoints let callers model
    independently-chosen role behavior, which is where genuine deadlocks can
    occur; the global sequence alone is always a total order and therefore
    deadlock-free under the buffered model.
    """

    name: str
    roles: tuple[str, ...]
    steps: tuple[Any, ...]
    locals: tuple[tuple[str, tuple[LocalAction, ...]], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "roles": list(self.roles),
            "steps": [_step_dict(step) for step in self.steps],
        }
        if self.locals:
            payload["locals"] = {
                role: [_action_dict(action) for action in actions]
                for role, actions in self.locals
            }
        return payload


def _step_dict(step: Any) -> dict[str, Any]:
    if isinstance(step, MessageStep):
        return {
            "kind": "message",
            "sender": step.sender,
            "receiver": step.receiver,
            "label": step.label,
            "payload": step.payload,
        }
    if isinstance(step, ChoiceStep):
        return {
            "kind": "choice",
            "chooser": step.chooser,
            "branches": [
                {
                    "label": branch.label,
                    "steps": [
                        _step_dict(item)
                        for item in (
                            branch.steps
                            if hasattr(branch, "steps")
                            else branch.protocol
                        )
                    ],
                }
                for branch in step.branches
            ],
        }
    if isinstance(step, RepeatStep):
        return {
            "kind": "repeat",
            "times": step.times,
            "body": [_step_dict(item) for item in step.body],
        }
    return {"kind": "end"}


def _normalize_step(value: Any) -> Step:
    if isinstance(value, (MessageStep, ChoiceStep, RepeatStep, EndStep)):
        return value
    if not isinstance(value, Mapping):
        _fail(
            "E003",
            f"protocol step must be a mapping or step object, got {type(value).__name__}",
        )
    kind = str(value.get("kind", "")).lower()
    if kind in {"message", "send", "recv", "communicate"}:
        sender = value.get("sender") or value.get("from")
        receiver = value.get("receiver") or value.get("to")
        label = value.get("label") or value.get("message")
        if not isinstance(sender, str) or not isinstance(receiver, str):
            _fail("E003", "message steps need string sender and receiver roles")
        if not isinstance(label, str) or not label:
            _fail("E003", "message steps need a non-empty label")
        payload = value.get("payload", "unit")
        return MessageStep(sender, receiver, label, str(payload))
    if kind == "choice":
        chooser = value.get("chooser") or value.get("role")
        branches = value.get("branches")
        if not isinstance(chooser, str) or not isinstance(branches, Iterable):
            _fail("E003", "choice steps need a chooser and a branch sequence")
        rows: list[ChoiceBranch] = []
        for branch in branches:
            if not isinstance(branch, Mapping):
                _fail("E003", "choice branches must be mappings")
            label = branch.get("label")
            steps = branch.get("steps", ())
            if not isinstance(label, str) or not label:
                _fail("E003", "choice branches need a non-empty label")
            rows.append(
                ChoiceBranch(
                    label,
                    tuple(_normalize_step(item) for item in _as_sequence(steps)),
                )
            )
        return ChoiceStep(chooser, tuple(rows))
    if kind in {"repeat", "loop", "star"}:
        times = value.get("times", 0)
        if not isinstance(times, int):
            _fail("E003", "repeat times must be an integer")
        body = tuple(
            _normalize_step(item) for item in _as_sequence(value.get("body", ()))
        )
        return RepeatStep(times, body)
    if kind in {"end", "stop", "terminate"}:
        return EndStep()
    _fail("E003", f"unknown protocol step kind {kind!r}")


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        _fail("E003", "expected a sequence of protocol steps")
    return tuple(value)


def _normalize_protocol(value: Any) -> GlobalProtocol:
    if isinstance(value, GlobalProtocol):
        return value
    if isinstance(value, Mapping):
        name = value.get("name", "protocol")
        roles = value.get("roles")
        steps = value.get("steps")
        locals_value = value.get("locals")
        if not isinstance(name, str) or not name:
            _fail("E003", "protocol name must be a non-empty string")
        local_rows: list[tuple[str, tuple[LocalAction, ...]]] = []
        if locals_value is not None:
            if not isinstance(locals_value, Mapping):
                _fail("E003", "protocol locals must map roles to action sequences")
            for raw_role, raw_actions in locals_value.items():
                role = str(raw_role)
                local_rows.append(
                    (
                        role,
                        tuple(
                            _normalize_local_action(item, role)
                            for item in _as_sequence(raw_actions)
                        ),
                    )
                )
            if roles is None:
                roles = [role for role, _ in local_rows]
        if steps is None:
            steps = ()
        if roles is None:
            roles = _infer_roles(_as_sequence(steps))
        if not isinstance(roles, (list, tuple)):
            _fail("E003", "protocol roles must be a sequence of role names")
        return GlobalProtocol(
            name,
            tuple(str(role) for role in roles),
            tuple(_normalize_step(item) for item in _as_sequence(steps)),
            tuple(local_rows),
        )
    return GlobalProtocol(
        "protocol",
        _infer_roles(_as_sequence(value)),
        tuple(_normalize_step(item) for item in _as_sequence(value)),
    )


def _normalize_local_action(value: Any, role: str) -> LocalAction:
    """Normalize one role-local action specification."""
    if isinstance(value, LocalAction):
        return value
    if not isinstance(value, Mapping):
        _fail(
            "E003",
            f"local action must be a mapping or LocalAction, got "
            f"{type(value).__name__}",
        )
    kind = str(value.get("kind", "")).lower()
    if kind in {"message", "send", "recv", "receive", "!", "?"}:
        sender = value.get("sender", value.get("from"))
        receiver = value.get("receiver", value.get("to"))
        if sender is not None or receiver is not None:
            if sender == role:
                kind = "send"
                peer = receiver
            elif receiver == role:
                kind = "receive"
                peer = sender
            else:
                _fail(
                    "E001",
                    f"local action for {role!r} does not involve that role",
                )
        else:
            peer = value.get("peer")
            kind = "send" if kind in {"send", "!"} else "receive"
        label = value.get("label", value.get("message"))
        if not isinstance(peer, str) or not isinstance(label, str) or not label:
            _fail("E003", "local send/receive actions need a peer and label")
        return LocalAction(kind, peer, label)
    if kind in {"choice", "offer", "select", "choose"}:
        raw_branches = value.get("branches")
        if isinstance(raw_branches, Mapping):
            branch_items = list(raw_branches.items())
        elif isinstance(raw_branches, Iterable) and not isinstance(
            raw_branches, (str, bytes)
        ):
            branch_items = []
            for branch in raw_branches:
                if not isinstance(branch, Mapping):
                    _fail("E003", "choice branches must be mappings")
                branch_items.append((branch.get("label"), branch.get("steps", ())))
        else:
            _fail("E003", "local choice needs a branch mapping or sequence")
        rows: list[tuple[str, tuple[LocalAction, ...]]] = []
        for raw_label, raw_steps in branch_items:
            if not isinstance(raw_label, str) or not raw_label:
                _fail("E003", "local choice branches need non-empty labels")
            rows.append(
                (
                    raw_label,
                    tuple(
                        _normalize_local_action(item, role)
                        for item in _as_sequence(raw_steps)
                    ),
                )
            )
        return LocalAction(
            "offer" if kind == "offer" else "choose", branches=tuple(rows)
        )
    if kind in {"repeat", "loop", "star"}:
        times = value.get("times", 0)
        if not isinstance(times, int):
            _fail("E003", "local repeat times must be an integer")
        body = tuple(
            _normalize_local_action(item, role)
            for item in _as_sequence(value.get("body", ()))
        )
        return LocalAction("loop", label=str(times), body=body)
    if kind in {"end", "stop", "terminate"}:
        return LocalAction("end")
    _fail("E003", f"unknown local action kind {kind!r}")


def _infer_roles(steps: Sequence[Any]) -> tuple[str, ...]:
    roles: list[str] = []
    for step in steps:
        if isinstance(step, MessageStep):
            for role in (step.sender, step.receiver):
                if role not in roles:
                    roles.append(role)
        elif isinstance(step, ChoiceStep):
            if step.chooser not in roles:
                roles.append(step.chooser)
            for branch in step.branches:
                for role in _infer_roles(
                    branch.steps if hasattr(branch, "steps") else branch.protocol
                ):
                    if role not in roles:
                        roles.append(role)
        elif isinstance(step, RepeatStep):
            for role in _infer_roles(step.body):
                if role not in roles:
                    roles.append(role)
    return tuple(roles)


def _validate_protocol(protocol: GlobalProtocol, *, span: Span = None) -> None:
    if not protocol.name or not isinstance(protocol.name, str):
        _fail("E003", "protocol name must be a non-empty string", span=span)
    if not protocol.roles:
        _fail("E006", "protocol must declare at least one role", span=span)
    if len(protocol.roles) > _MAX_ROLES:
        _fail(
            "E008",
            f"protocol has {len(protocol.roles)} roles, above the bounded limit {_MAX_ROLES}",
            span=span,
        )
    if len(set(protocol.roles)) != len(protocol.roles):
        _fail("E001", "protocol roles must be unique", span=span)
    known = set(protocol.roles)
    for role in protocol.roles:
        if not _LABEL_RE.fullmatch(role):
            _fail("E001", f"invalid role name {role!r}", span=span)
    if protocol.locals:
        local_roles = [role for role, _ in protocol.locals]
        if len(set(local_roles)) != len(local_roles):
            _fail("E001", "local endpoints must be declared once per role", span=span)
        for role, actions in protocol.locals:
            if role not in known:
                _fail(
                    "E001",
                    f"local endpoint role {role!r} is not a protocol role",
                    span=span,
                )
            _validate_local_actions(
                actions,
                known,
                role=role,
                span=span,
                context=protocol.name,
            )
    if protocol.steps:
        _validate_steps(protocol.steps, known, span=span, context=protocol.name)
    elif not protocol.locals:
        _fail(
            "E003",
            f"protocol {protocol.name!r} needs a global step sequence or locals",
            span=span,
        )


def _validate_steps(
    steps: Sequence[Any], roles: set[str], *, span: Span, context: str
) -> None:
    if not steps:
        _fail("E003", f"protocol {context!r} has an empty step sequence", span=span)
    ended = False
    for index, step in enumerate(steps):
        if ended:
            _fail(
                "E003",
                f"protocol {context!r} has steps after End at position {index}",
                span=span,
            )
        if isinstance(step, MessageStep):
            if step.sender not in roles or step.receiver not in roles:
                _fail(
                    "E001",
                    f"message {step.label!r} references an unknown role "
                    f"({step.sender!r} -> {step.receiver!r})",
                    span=span,
                )
            if step.sender == step.receiver:
                _fail(
                    "E007",
                    f"self-directed message {step.label!r} is outside the "
                    "synchronous rendezvous model",
                    span=span,
                )
            if not _LABEL_RE.fullmatch(step.label):
                _fail("E001", f"invalid message label {step.label!r}", span=span)
        elif isinstance(step, ChoiceStep):
            if step.chooser not in roles:
                _fail(
                    "E001",
                    f"choice chooser {step.chooser!r} is not a protocol role",
                    span=span,
                )
            if len(step.branches) < 2:
                _fail("E003", "a choice needs at least two branches", span=span)
            labels = [branch.label for branch in step.branches]
            if len(set(labels)) != len(labels):
                _fail("E001", "choice branch labels must be unique", span=span)
            for branch in step.branches:
                if not _LABEL_RE.fullmatch(branch.label):
                    _fail(
                        "E001",
                        f"invalid choice branch label {branch.label!r}",
                        span=span,
                    )
                _validate_steps(
                    (branch.steps if hasattr(branch, "steps") else branch.protocol),
                    roles,
                    span=span,
                    context=f"{context}.{branch.label}",
                )
        elif isinstance(step, RepeatStep):
            if step.times < 0:
                _fail("E003", "repeat count must be non-negative", span=span)
            if step.times > _MAX_REPEAT:
                _fail(
                    "E008",
                    f"repeat count {step.times} exceeds the bounded limit {_MAX_REPEAT}",
                    span=span,
                )
            _validate_steps(step.body, roles, span=span, context=f"{context}.repeat")
            if step.times == 0:
                continue
        elif isinstance(step, EndStep):
            ended = True
        else:
            _fail("E003", f"unknown step object {type(step).__name__}", span=span)


def _validate_local_actions(
    actions: Sequence[LocalAction],
    roles: set[str],
    *,
    role: str,
    span: Span,
    context: str,
) -> None:
    if not actions:
        _fail(
            "E003",
            f"local endpoint {role!r} in {context!r} has no actions",
            span=span,
        )
    ended = False
    for index, action in enumerate(actions):
        if ended:
            _fail(
                "E003",
                f"local endpoint {role!r} has actions after End at {index}",
                span=span,
            )
        if action.kind in {"send", "receive"}:
            if action.peer not in roles:
                _fail(
                    "E001",
                    f"local action references unknown peer {action.peer!r}",
                    span=span,
                )
            if action.peer == role:
                _fail(
                    "E007",
                    f"self-directed local action in role {role!r} is unsupported",
                    span=span,
                )
            if not action.label or not _LABEL_RE.fullmatch(action.label):
                _fail(
                    "E001",
                    f"invalid local message label {action.label!r}",
                    span=span,
                )
        elif action.kind in {"choose", "offer"}:
            if len(action.branches) < 2:
                _fail(
                    "E003",
                    f"local choice in role {role!r} needs at least two branches",
                    span=span,
                )
            labels = [label for label, _ in action.branches]
            if len(set(labels)) != len(labels):
                _fail("E001", "local choice branch labels must be unique", span=span)
            for label, branch in action.branches:
                if not _LABEL_RE.fullmatch(label):
                    _fail(
                        "E001",
                        f"invalid local branch label {label!r}",
                        span=span,
                    )
                _validate_local_actions(
                    branch,
                    roles,
                    role=role,
                    span=span,
                    context=f"{context}.{label}",
                )
        elif action.kind == "loop":
            try:
                times = int(action.label or "0")
            except TypeError, ValueError:
                _fail("E003", "local repeat count must be an integer", span=span)
            if times < 0:
                _fail("E003", "local repeat count must be non-negative", span=span)
            if times > _MAX_REPEAT:
                _fail(
                    "E008",
                    f"local repeat count {times} exceeds the bounded limit "
                    f"{_MAX_REPEAT}",
                    span=span,
                )
            if times:
                _validate_local_actions(
                    action.body,
                    roles,
                    role=role,
                    span=span,
                    context=f"{context}.repeat",
                )
        elif action.kind == "end":
            ended = True
        else:
            _fail(
                "E003",
                f"unknown local action kind {action.kind!r}",
                span=span,
            )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalAction:
    """One projected local action for display and protocol traces."""

    kind: str
    peer: str | None = None
    label: str | None = None
    branches: tuple[tuple[str, tuple[LocalAction, ...]], ...] = ()
    body: tuple[LocalAction, ...] = ()

    def render(self) -> str:
        if self.kind == "send":
            return f"!{self.label}({self.peer})"
        if self.kind == "receive":
            return f"?{self.label}({self.peer})"
        if self.kind == "choose":
            arms = " + ".join(
                f"{label}: {_render_actions(actions)}"
                for label, actions in self.branches
            )
            return f"choose {{ {arms} }}"
        if self.kind == "offer":
            arms = " & ".join(
                f"{label}: {_render_actions(actions)}"
                for label, actions in self.branches
            )
            return f"offer {{ {arms} }}"
        if self.kind == "loop":
            return f"loop({self.label or len(self.body)}) {{ {_render_actions(self.body)} }}"
        return "end"


def _render_actions(actions: Sequence[LocalAction]) -> str:
    return "; ".join(action.render() for action in actions) or "end"


@dataclass(frozen=True)
class LocalEndpoint:
    """Projected local type for one role."""

    role: str
    actions: tuple[LocalAction, ...]
    text: str
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "actions": [_action_dict(action) for action in self.actions],
            "text": self.text,
            "digest": self.digest,
        }


def _action_dict(action: LocalAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "peer": action.peer,
        "label": action.label,
        "branches": [
            (label, [_action_dict(item) for item in actions])
            for label, actions in action.branches
        ],
        "body": [_action_dict(item) for item in action.body],
    }


def _project_steps(steps: Sequence[Any], role: str) -> tuple[LocalAction, ...]:
    result: list[LocalAction] = []
    for step in steps:
        if isinstance(step, MessageStep):
            if step.sender == role:
                result.append(LocalAction("send", step.receiver, step.label))
            elif step.receiver == role:
                result.append(LocalAction("receive", step.sender, step.label))
        elif isinstance(step, ChoiceStep):
            projected = tuple(
                (
                    branch.label,
                    _project_steps(
                        (branch.steps if hasattr(branch, "steps") else branch.protocol),
                        role,
                    ),
                )
                for branch in step.branches
            )
            if role == step.chooser:
                result.append(LocalAction("choose", branches=projected))
            else:
                nonempty = tuple(
                    (label, actions) for label, actions in projected if actions
                )
                if len(nonempty) == 1:
                    result.extend(nonempty[0][1])
                elif nonempty:
                    result.append(LocalAction("offer", branches=nonempty))
        elif isinstance(step, RepeatStep):
            body = _project_steps(step.body, role)
            if body and step.times:
                result.append(LocalAction("loop", body=body, label=str(step.times)))
        elif isinstance(step, EndStep):
            break
    return tuple(result)


def _project_all(protocol: GlobalProtocol) -> tuple[LocalEndpoint, ...]:
    local_map = dict(protocol.locals)
    endpoints: list[LocalEndpoint] = []
    for role in protocol.roles:
        actions = local_map.get(role)
        if actions is None:
            actions = _project_steps(protocol.steps, role)
        text = f"{role}: {_render_actions(actions)}"
        endpoints.append(
            LocalEndpoint(
                role,
                actions,
                text,
                _digest({"role": role, "actions": [_action_dict(a) for a in actions]}),
            )
        )
    return tuple(endpoints)


# ---------------------------------------------------------------------------
# Bounded deadlock/progress exploration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LinearMessage:
    sender: str
    receiver: str
    label: str


def _expand_linear(
    steps: Sequence[Any],
    *,
    max_variants: int,
) -> tuple[tuple[_LinearMessage, ...], ...] | None:
    variants: list[tuple[_LinearMessage, ...]] = [()]
    for step in steps:
        if len(variants) > max_variants:
            return None
        if isinstance(step, MessageStep):
            variants = [
                variant + (_LinearMessage(step.sender, step.receiver, step.label),)
                for variant in variants
            ]
        elif isinstance(step, EndStep):
            break
        elif isinstance(step, RepeatStep):
            for _ in range(step.times):
                body = _expand_linear(step.body, max_variants=max_variants)
                if body is None:
                    return None
                variants = [variant + suffix for variant in variants for suffix in body]
                if len(variants) > max_variants:
                    return None
        elif isinstance(step, ChoiceStep):
            branch_variants: list[tuple[_LinearMessage, ...]] = []
            for branch in step.branches:
                expanded = _expand_linear(
                    (branch.steps if hasattr(branch, "steps") else branch.protocol),
                    max_variants=max_variants,
                )
                if expanded is None:
                    return None
                branch_variants.extend(expanded)
            variants = [
                variant + suffix for variant in variants for suffix in branch_variants
            ]
            if len(variants) > max_variants:
                return None
    return tuple(variants)


def _local_trace(
    messages: Sequence[_LinearMessage], role: str
) -> tuple[tuple[str, str, str], ...]:
    actions: list[tuple[str, str, str]] = []
    for message in messages:
        if message.sender == role:
            actions.append(("send", message.receiver, message.label))
        elif message.receiver == role:
            actions.append(("receive", message.sender, message.label))
    return tuple(actions)


def _expand_local_messages(
    actions: Sequence[LocalAction],
    *,
    max_variants: int,
) -> tuple[tuple[tuple[str, str, str], ...], ...] | None:
    """Expand role-local choices into bounded linear action sequences."""
    variants: list[tuple[tuple[str, str, str], ...]] = [()]
    for action in actions:
        if len(variants) > max_variants:
            return None
        if action.kind == "send":
            variants = [
                variant + (("send", str(action.peer), str(action.label)),)
                for variant in variants
            ]
        elif action.kind == "receive":
            variants = [
                variant + (("receive", str(action.peer), str(action.label)),)
                for variant in variants
            ]
        elif action.kind in {"choose", "offer"}:
            branch_variants: list[tuple[tuple[str, str, str], ...]] = []
            for _, branch in action.branches:
                expanded = _expand_local_messages(branch, max_variants=max_variants)
                if expanded is None:
                    return None
                branch_variants.extend(expanded)
            variants = [
                variant + suffix for variant in variants for suffix in branch_variants
            ]
            if len(variants) > max_variants:
                return None
        elif action.kind == "loop":
            times = int(action.label or "0")
            body = _expand_local_messages(action.body, max_variants=max_variants)
            if body is None:
                return None
            for _ in range(times):
                variants = [variant + suffix for variant in variants for suffix in body]
                if len(variants) > max_variants:
                    return None
        elif action.kind == "end":
            break
    return tuple(variants)


@dataclass(frozen=True)
class _DeadlockExploration:
    deadlock_free: bool
    progress: bool
    bounded: bool
    states: int
    transitions: int
    trace: tuple[str, ...]
    reason: str


_Buffers = tuple[tuple[str, ...], ...]
_State = tuple[tuple[int, ...], _Buffers]


def _explore_buffered(
    traces: Mapping[str, Sequence[tuple[str, str, str]]],
    roles: Sequence[str],
    *,
    max_states: int,
    max_steps: int,
    max_buffer: int,
    deadline: float,
) -> _DeadlockExploration:
    """Explore bounded asynchronous interleavings with per-channel FIFOs.

    Sends are non-blocking and enqueue onto a directed channel buffer; receives
    block until the expected label is at the head of their incoming buffer.
    A state with no enabled action is a deadlock.  Incoming buffers are bounded
    by ``max_buffer``; exceeding that is reported as ``E008`` rather than
    treated as a proof.
    """
    channels = tuple(
        (sender, receiver)
        for sender in roles
        for receiver in roles
        if sender != receiver
    )
    channel_index = {channel: index for index, channel in enumerate(channels)}

    def terminal(state: _State) -> bool:
        positions, buffers = state
        return all(
            positions[index] == len(traces[role]) for index, role in enumerate(roles)
        ) and all(not buffer for buffer in buffers)

    start: _State = (tuple(0 for _ in roles), tuple(() for _ in channels))
    queue: deque[_State] = deque([start])
    seen = {start}
    parents: dict[_State, tuple[_State, str] | None] = {start: None}
    transitions = 0
    deadlock: _State | None = None

    while queue:
        if time.monotonic() > deadline:
            return _DeadlockExploration(
                False,
                False,
                True,
                len(seen),
                transitions,
                (),
                "timeout",
            )
        state = queue.popleft()
        if terminal(state):
            continue
        positions, buffers = state
        enabled: list[tuple[_State, str]] = []
        for index, role in enumerate(roles):
            position = positions[index]
            if position >= len(traces[role]):
                continue
            kind, peer, label = traces[role][position]
            roles.index(peer)
            if kind == "send":
                channel = (role, peer)
                buffer_index = channel_index[channel]
                incoming = buffers[buffer_index]
                if len(incoming) >= max_buffer:
                    # Some schedule exceeds the modelled buffer capacity.  The
                    # result is Unknown, not a proof: an unbounded schedule may
                    # deadlock or progress differently.
                    return _DeadlockExploration(
                        False,
                        False,
                        True,
                        len(seen),
                        transitions,
                        (),
                        "buffer bound",
                    )
                next_positions = list(positions)
                next_positions[index] += 1
                next_buffers = list(buffers)
                next_buffers[buffer_index] = incoming + (label,)
                enabled.append(
                    (
                        (tuple(next_positions), tuple(next_buffers)),
                        f"{role} --{label}--> {peer}",
                    )
                )
            elif kind == "receive":
                channel = (peer, role)
                buffer_index = channel_index.get(channel)
                if buffer_index is None:
                    continue
                incoming = buffers[buffer_index]
                if not incoming or incoming[0] != label:
                    continue
                next_positions = list(positions)
                next_positions[index] += 1
                next_buffers = list(buffers)
                next_buffers[buffer_index] = incoming[1:]
                enabled.append(
                    (
                        (tuple(next_positions), tuple(next_buffers)),
                        f"{role} <--{label}-- {peer}",
                    )
                )
        if not enabled:
            deadlock = state
            break
        for next_state, action in enabled:
            transitions += 1
            if transitions > max_steps:
                return _DeadlockExploration(
                    False,
                    False,
                    True,
                    len(seen),
                    transitions,
                    (),
                    "step bound",
                )
            if next_state not in seen:
                seen.add(next_state)
                parents[next_state] = (state, action)
                queue.append(next_state)
                if len(seen) > max_states:
                    return _DeadlockExploration(
                        False,
                        False,
                        True,
                        len(seen),
                        transitions,
                        (),
                        "state bound",
                    )

    if deadlock is not None:
        trace: list[str] = []
        cursor: _State | None = deadlock
        while cursor is not None and parents.get(cursor) is not None:
            parent, action = parents[cursor]  # type: ignore[misc]
            trace.append(action)
            cursor = parent
        trace.reverse()
        remaining = ", ".join(
            f"{role}@{deadlock[0][index]}" for index, role in enumerate(roles)
        )
        trace.append(f"blocked state: {remaining}")
        return _DeadlockExploration(
            False,
            False,
            False,
            len(seen),
            transitions,
            tuple(trace),
            "deadlock",
        )

    # Progress: every reachable state must reach a terminal state.  Positions
    # only advance, so the explored graph is acyclic and a reverse BFS from
    # terminal states is exact.
    reverse: dict[_State, list[_State]] = {state: [] for state in seen}
    for child, parent in parents.items():
        if parent is not None:
            reverse[child].append(parent[0])
    terminal_states = {state for state in seen if terminal(state)}
    can_finish = set(terminal_states)
    queue = deque(terminal_states)
    while queue:
        state = queue.popleft()
        for parent in reverse.get(state, ()):
            if parent not in can_finish:
                can_finish.add(parent)
                queue.append(parent)
    progress = can_finish == seen
    return _DeadlockExploration(
        True,
        progress,
        False,
        len(seen),
        transitions,
        (),
        "ok" if progress else "progress violation",
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionDiagnostic:
    code: str
    message: str
    kind: str
    trace: tuple[str, ...] = ()
    severity: Severity = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "kind": self.kind,
            "trace": list(self.trace),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class SessionProjectionReport:
    protocol: str
    roles: tuple[str, ...]
    endpoints: tuple[LocalEndpoint, ...]
    deadlock_free: bool
    progress: bool
    bounded: bool
    variants: int
    explored_states: int
    explored_transitions: int
    diagnostics: tuple[SessionDiagnostic, ...] = ()
    traces: tuple[tuple[str, ...], ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "protocol": self.protocol,
            "roles": list(self.roles),
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
            "deadlock_free": self.deadlock_free,
            "progress": self.progress,
            "bounded": self.bounded,
            "variants": self.variants,
            "explored_states": self.explored_states,
            "explored_transitions": self.explored_transitions,
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "traces": [list(trace) for trace in self.traces],
            "bounds": dict(self.bounds),
        }
        if include_digest:
            payload["digest"] = self.digest()
        return payload


def _raise_first(diagnostics: Sequence[SessionDiagnostic], span: Span) -> None:
    for diagnostic in diagnostics:
        if diagnostic.severity != "error":
            continue
        _fail(
            diagnostic.code,
            diagnostic.message,
            span=span,
            reason=diagnostic.message,
            help_text=(
                "fix the protocol or trace; the report contains the bounded trace"
            ),
            note="compiler.session_types",
        )


def session_synthesize_and_project_protocols(
    protocol: Any,
    *,
    max_variants: int = 64,
    max_states: int = 2048,
    max_steps: int = 4096,
    max_buffer: int = 64,
    timeout_ms: float = 1000.0,
    strict: bool = False,
    span: Span = None,
) -> SessionProjectionReport:
    """Project a finite protocol and prove bounded deadlock freedom.

    The result distinguishes three outcomes:

    * ``deadlock_free`` / ``progress`` are true only after every expanded
      linear variant was explored without hitting a bound or timeout;
    * ``bounded`` is true when a budget was exhausted, in which case the proof
      flags are false and an ``E008``/``E004`` diagnostic is present;
    * malformed protocol shapes are rejected before exploration with ``E001``,
      ``E003``, ``E006``, or ``E007``.

    A global ``steps`` sequence expands to a bounded set of linear variants,
    each a total order and therefore deadlock-free under the buffered FIFO
    model.  Per-role ``locals`` are also accepted; their independent choices are
    combined (bounded by ``max_variants``) and explored asynchronously, which is
    where genuine cross-role deadlocks are detected.
    """
    if (
        max_variants < 1
        or max_states < 1
        or max_steps < 1
        or max_buffer < 1
        or timeout_ms <= 0
    ):
        _fail("E008", "protocol bounds must be positive", span=span)
    normalized = _normalize_protocol(protocol)
    _validate_protocol(normalized, span=span)
    endpoints = _project_all(normalized)
    bounds = {
        "max_variants": max_variants,
        "max_states": max_states,
        "max_steps": max_steps,
        "max_buffer": max_buffer,
        "timeout_ms": timeout_ms,
        "model": "bounded-async-fifo",
    }
    diagnostics: list[SessionDiagnostic] = []
    traces: list[tuple[str, ...]] = []

    def variant_bound_report() -> SessionProjectionReport:
        diagnostics.append(
            SessionDiagnostic(
                "E008",
                f"choice expansion exceeded max_variants={max_variants}",
                "variant-bound",
            )
        )
        report = SessionProjectionReport(
            normalized.name,
            normalized.roles,
            endpoints,
            False,
            False,
            True,
            0,
            0,
            0,
            tuple(diagnostics),
            (),
            bounds,
        )
        if strict:
            _raise_first(report.diagnostics, span)
        return report

    combos: list[Mapping[str, Sequence[tuple[str, str, str]]]]
    if normalized.locals:
        bounds["variants_from"] = "locals"
        per_role: list[tuple[str, tuple[tuple[tuple[str, str, str], ...], ...]]] = []
        for role, actions in normalized.locals:
            variants = _expand_local_messages(actions, max_variants=max_variants)
            if variants is None:
                return variant_bound_report()
            per_role.append((role, variants))
        product: list[dict[str, Sequence[tuple[str, str, str]]]] = [{}]
        for role, variants in per_role:
            product = [
                {**combo, role: variant} for combo in product for variant in variants
            ]
            if len(product) > max_variants:
                return variant_bound_report()
        combos = product
    else:
        bounds["variants_from"] = "global"
        expanded = _expand_linear(normalized.steps, max_variants=max_variants)
        if expanded is None:
            return variant_bound_report()
        combos = [
            {role: _local_trace(variant, role) for role in normalized.roles}
            for variant in expanded
        ]

    deadline = time.monotonic() + timeout_ms / 1000.0
    states = transitions = 0
    deadlock_free = True
    progress = True
    for combo in combos:
        traces_map = {role: combo.get(role, ()) for role in normalized.roles}
        result = _explore_buffered(
            traces_map,
            normalized.roles,
            max_states=max_states,
            max_steps=max_steps,
            max_buffer=max_buffer,
            deadline=deadline,
        )
        states += result.states
        transitions += result.transitions
        if result.bounded:
            code = "E004" if result.reason == "timeout" else "E008"
            diagnostics.append(
                SessionDiagnostic(
                    code,
                    f"bounded exploration stopped: {result.reason}",
                    "bound",
                    result.trace,
                    "unknown",
                )
            )
            deadlock_free = False
            progress = False
            if result.trace:
                traces.append(result.trace)
            break
        if not result.deadlock_free:
            deadlock_free = False
            progress = False
            diagnostics.append(
                SessionDiagnostic(
                    "E002",
                    "protocol has a reachable deadlock",
                    "deadlock",
                    result.trace,
                )
            )
            traces.append(result.trace)
        elif not result.progress:
            progress = False
            diagnostics.append(
                SessionDiagnostic(
                    "E002",
                    "reachable state cannot make progress to termination",
                    "progress",
                    result.trace,
                )
            )
            if result.trace:
                traces.append(result.trace)

    bounded = any(item.kind == "bound" for item in diagnostics)
    if not bounded and deadlock_free and progress:
        diagnostics.append(
            SessionDiagnostic(
                "E002",
                "bounded proof: every expanded variant terminates under the "
                "asynchronous FIFO model",
                "proof",
                (),
                "warning",
            )
        )
    report = SessionProjectionReport(
        normalized.name,
        normalized.roles,
        endpoints,
        deadlock_free and not bounded,
        progress and not bounded,
        bounded,
        len(combos),
        states,
        transitions,
        tuple(diagnostics),
        tuple(traces),
        bounds,
    )
    if strict:
        _raise_first(report.diagnostics, span)
    return report


# ---------------------------------------------------------------------------
# Linear channel consumption
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearOperation:
    kind: str
    channel: str
    role: str | None = None
    endpoints: tuple[str, ...] = ()
    payload: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "role": self.role,
            "endpoints": list(self.endpoints),
            "payload": self.payload,
        }


def _normalize_operation(value: Any) -> LinearOperation:
    if isinstance(value, LinearOperation):
        return value
    if not isinstance(value, Mapping):
        _fail(
            "E003",
            f"linear operation must be a mapping or LinearOperation, got "
            f"{type(value).__name__}",
        )
    kind = str(value.get("kind", "")).lower()
    if kind not in _KNOWN_KINDS:
        _fail("E003", f"unknown linear operation kind {kind!r}")
    channel = value.get("channel") or value.get("name")
    if not isinstance(channel, str) or not channel:
        _fail("E003", "linear operations need a non-empty channel name")
    role = value.get("role")
    endpoints = value.get("endpoints", ())
    if isinstance(endpoints, str):
        endpoints_tuple = (endpoints,)
    elif isinstance(endpoints, Iterable):
        endpoints_tuple = tuple(str(item) for item in endpoints)
    else:
        _fail("E003", "endpoints must be a string or sequence")
    payload = value.get("payload")
    return LinearOperation(
        kind,
        channel,
        None if role is None else str(role),
        endpoints_tuple,
        None if payload is None else str(payload),
    )


@dataclass(frozen=True)
class LinearDiagnostic:
    code: str
    message: str
    channel: str
    index: int
    trace: tuple[str, ...] = ()
    severity: Severity = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "channel": self.channel,
            "index": self.index,
            "trace": list(self.trace),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class LinearChannelReport:
    channel: str
    state: str
    tokens: int
    opened_at: int | None
    closed_at: int | None
    operations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "state": self.state,
            "tokens": self.tokens,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "operations": list(self.operations),
        }


@dataclass(frozen=True)
class LinearConsumptionReport:
    accepted: bool
    channels: tuple[LinearChannelReport, ...]
    diagnostics: tuple[LinearDiagnostic, ...] = ()
    events: tuple[str, ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "accepted": self.accepted,
            "channels": [channel.as_dict() for channel in self.channels],
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "events": list(self.events),
            "bounds": dict(self.bounds),
        }
        if include_digest:
            payload["digest"] = self.digest()
        return payload


def session_verify_linear_channel_consumption(
    operations: Sequence[Any],
    *,
    allow_affine: bool = True,
    max_operations: int = 4096,
    strict: bool = False,
    span: Span = None,
) -> LinearConsumptionReport:
    """Check a concrete channel trace for linear/affine consumption.

    Each ``open`` creates one linear token.  ``send``/``recv`` keep the token
    alive, ``close`` consumes the remaining token, and an explicit ``drop`` is
    the only permitted affine discard.  Duplicating a token, using a channel
    before opening or after closing, or leaving a token alive at the end is a
    structured error (``E001``/``E002``/``E006``).
    """
    if max_operations < 1:
        _fail("E008", "max_operations must be positive", span=span)
    if len(operations) > max_operations:
        _fail(
            "E008",
            f"trace has {len(operations)} operations, above max_operations={max_operations}",
            span=span,
        )
    normalized = [_normalize_operation(value) for value in operations]
    states: dict[str, dict[str, Any]] = {}
    diagnostics: list[LinearDiagnostic] = []
    events: list[str] = []

    def record(
        code: str,
        message: str,
        channel: str,
        index: int,
        trace: list[str],
    ) -> None:
        diagnostics.append(
            LinearDiagnostic(code, message, channel, index, tuple(trace))
        )

    for index, operation in enumerate(normalized):
        channel = operation.channel
        state = states.get(channel)
        trace = list(state["trace"]) if state else []
        trace.append(f"{index}:{operation.kind}({channel})")
        events.append(f"{index}:{operation.kind}:{channel}")
        if operation.kind in {"open", "acquire"}:
            if state is not None and state["state"] == "open":
                record("E001", "channel opened twice", channel, index, trace)
                continue
            states[channel] = {
                "state": "open",
                "tokens": 1,
                "opened_at": index,
                "closed_at": None,
                "endpoints": set(operation.endpoints),
                "trace": trace,
                "operations": [operation.kind],
            }
            continue
        if state is None:
            record(
                "E006",
                "operation uses a channel before it is opened",
                channel,
                index,
                trace,
            )
            continue
        state["trace"] = trace
        state["operations"].append(operation.kind)
        if operation.kind in {"duplicate", "clone"}:
            record(
                "E001",
                "linear channel capability cannot be duplicated",
                channel,
                index,
                trace,
            )
            continue
        if state["state"] != "open":
            record(
                "E001",
                f"operation after {state['state']}",
                channel,
                index,
                trace,
            )
            continue
        if operation.endpoints:
            if operation.role is None:
                record(
                    "E001",
                    "endpoint-qualified operation needs a role",
                    channel,
                    index,
                    trace,
                )
                continue
            state["endpoints"].update(operation.endpoints)
        if (
            operation.role is not None
            and state["endpoints"]
            and operation.role not in state["endpoints"]
        ):
            record(
                "E001",
                f"role {operation.role!r} is not an endpoint of {channel!r}",
                channel,
                index,
                trace,
            )
            continue
        if operation.kind in {"send", "write", "recv", "read"}:
            if state["tokens"] < 1:
                record(
                    "E001", "operation needs a live linear token", channel, index, trace
                )
            continue
        if operation.kind == "split":
            if state["tokens"] != 1:
                record(
                    "E001",
                    "split requires exactly one live token",
                    channel,
                    index,
                    trace,
                )
                continue
            state["tokens"] = 2
            state["operations"].append("split")
            continue
        if operation.kind in {"close", "release"}:
            state["tokens"] -= 1
            if state["tokens"] <= 0:
                state["tokens"] = 0
                state["state"] = "closed"
                state["closed_at"] = index
            continue
        if operation.kind == "drop":
            if not allow_affine:
                record(
                    "E001",
                    "affine drop is disabled for this session",
                    channel,
                    index,
                    trace,
                )
                continue
            state["tokens"] -= 1
            if state["tokens"] <= 0:
                state["tokens"] = 0
                state["state"] = "dropped"
                state["closed_at"] = index
            continue

    channel_reports: list[LinearChannelReport] = []
    for channel, state in sorted(states.items()):
        if state["state"] == "open" and state["tokens"] > 0:
            diagnostics.append(
                LinearDiagnostic(
                    "E002",
                    f"channel {channel!r} is still open at trace end "
                    f"({state['tokens']} live token(s))",
                    channel,
                    len(normalized),
                    tuple(state["trace"]),
                )
            )
        channel_reports.append(
            LinearChannelReport(
                channel,
                str(state["state"]),
                int(state["tokens"]),
                state["opened_at"],
                state["closed_at"],
                tuple(state["operations"]),
            )
        )
    accepted = not any(item.severity == "error" for item in diagnostics)
    report = LinearConsumptionReport(
        accepted,
        tuple(channel_reports),
        tuple(diagnostics),
        tuple(events),
        {
            "allow_affine": allow_affine,
            "max_operations": max_operations,
            "model": "single-token-linear-capability",
        },
    )
    if strict:
        for diagnostic in report.diagnostics:
            if diagnostic.severity != "error":
                continue
            _fail(
                diagnostic.code,
                diagnostic.message,
                span=span,
                reason=diagnostic.message,
                help_text="consume, close, or explicitly drop every linear token",
                note="compiler.session_types",
            )
    return report
