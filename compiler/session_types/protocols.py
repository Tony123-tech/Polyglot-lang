"""Multiparty session types: global protocols, projection, and bounded checks.

The module models a deliberately small but precise fragment of multiparty
session types (MPST):

* a global protocol is a tree of binary interactions
  (``sender -> receiver: {label: continuation}``), recursion, and ``end``;
* projection turns a global protocol into one local endpoint per role;
* a bounded asynchronous scheduler explores every interleaving of the local
  endpoints.  A reachable stuck state is a deadlock; a reachable cycle with no
  path to ``end`` is a progress violation.

The scheduler is *bounded*.  State-space explosion, unbounded recursion, and
optional wall-clock timeouts are reported as ``Unknown`` (``E002``) rather
than being presented as a proof.  No claim of universal deadlock freedom for
arbitrary programs is made.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
Status = Literal["proved", "deadlock", "unknown"]

# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


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
        help_text=help_text or "fix the protocol or increase the analysis bound",
        note="compiler.session_types/protocols",
        span=span,
    )


def _check_name(value: object, label: str, *, span: Span = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("E003", f"{label} must be a non-empty string", span=span)
    return value


# ---------------------------------------------------------------------------
# Global protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalEnd:
    """The protocol has finished."""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "end"}


@dataclass(frozen=True)
class GlobalVar:
    """A reference to an enclosing recursion variable."""

    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "var", "name": self.name}


@dataclass(frozen=True)
class GlobalRec:
    """A recursive global protocol ``rec X. body``."""

    name: str
    body: GlobalProtocol

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "rec", "name": self.name, "body": self.body.to_dict()}


@dataclass(frozen=True)
class GlobalBranch:
    """One labelled branch of a global interaction."""

    label: str
    protocol: GlobalProtocol

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "protocol": self.protocol.to_dict()}


@dataclass(frozen=True)
class GlobalSelect:
    """``sender -> receiver: {label: continuation, ...}``."""

    sender: str
    receiver: str
    branches: tuple[GlobalBranch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "select",
            "sender": self.sender,
            "receiver": self.receiver,
            "branches": [branch.to_dict() for branch in self.branches],
        }


type GlobalProtocol = GlobalEnd | GlobalVar | GlobalRec | GlobalSelect

# ---------------------------------------------------------------------------
# Local endpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalEnd:
    """The local endpoint is finished."""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "end"}


@dataclass(frozen=True)
class LocalVar:
    """A reference to an enclosing local recursion variable."""

    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "var", "name": self.name}


@dataclass(frozen=True)
class LocalRec:
    """A recursive local endpoint ``rec X. body``."""

    name: str
    body: LocalType

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "rec", "name": self.name, "body": self.body.to_dict()}


@dataclass(frozen=True)
class LocalBranch:
    """One labelled branch of a local send or receive."""

    label: str
    protocol: LocalType

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "protocol": self.protocol.to_dict()}


@dataclass(frozen=True)
class LocalSend:
    """An internal choice: this role sends one of the labels to *peer*."""

    peer: str
    branches: tuple[LocalBranch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "send",
            "peer": self.peer,
            "branches": [branch.to_dict() for branch in self.branches],
        }


@dataclass(frozen=True)
class LocalRecv:
    """An external choice: this role receives one of the labels from *peer*."""

    peer: str
    branches: tuple[LocalBranch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "recv",
            "peer": self.peer,
            "branches": [branch.to_dict() for branch in self.branches],
        }


type LocalType = LocalEnd | LocalVar | LocalRec | LocalSend | LocalRecv

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_global(
    protocol: GlobalProtocol,
    roles: frozenset[str],
    *,
    bound: frozenset[str] = frozenset(),
    span: Span = None,
) -> None:
    if isinstance(protocol, GlobalEnd):
        return
    if isinstance(protocol, GlobalVar):
        _check_name(protocol.name, "recursion variable", span=span)
        if protocol.name not in bound:
            _fail(
                "E005",
                f"unbound recursion variable {protocol.name!r}",
                reason="a GlobalVar must be inside a matching GlobalRec",
                span=span,
            )
        return
    if isinstance(protocol, GlobalRec):
        _check_name(protocol.name, "recursion variable", span=span)
        if protocol.name in bound:
            _fail(
                "E003",
                f"shadowed recursion variable {protocol.name!r}",
                reason="nested recursion must use distinct variable names",
                span=span,
            )
        _validate_global(protocol.body, roles, bound=bound | {protocol.name}, span=span)
        return
    if not isinstance(protocol, GlobalSelect):
        _fail(
            "E003", f"unknown global protocol node {type(protocol).__name__}", span=span
        )
    _check_name(protocol.sender, "sender role", span=span)
    _check_name(protocol.receiver, "receiver role", span=span)
    if protocol.sender == protocol.receiver:
        _fail(
            "E003",
            f"role {protocol.sender!r} cannot send to itself",
            reason="global interactions must be between two distinct roles",
            span=span,
        )
    for role in (protocol.sender, protocol.receiver):
        if role not in roles:
            _fail("E001", f"role {role!r} is not declared", span=span)
    if not protocol.branches:
        _fail("E003", "an interaction must have at least one branch", span=span)
    labels: set[str] = set()
    for branch in protocol.branches:
        _check_name(branch.label, "branch label", span=span)
        if branch.label in labels:
            _fail(
                "E003",
                f"duplicate branch label {branch.label!r}",
                reason="labels identify choices and must be unique",
                span=span,
            )
        labels.add(branch.label)
        _validate_global(branch.protocol, roles, bound=bound, span=span)


def _roles_of(protocol: GlobalProtocol, *, seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if isinstance(protocol, GlobalEnd | GlobalVar):
        return seen
    if isinstance(protocol, GlobalRec):
        _roles_of(protocol.body, seen=seen)
        return seen
    seen.add(protocol.sender)
    seen.add(protocol.receiver)
    for branch in protocol.branches:
        _roles_of(branch.protocol, seen=seen)
    return seen


def _occurs(protocol: GlobalProtocol, role: str) -> bool:
    if isinstance(protocol, GlobalEnd | GlobalVar):
        return False
    if isinstance(protocol, GlobalRec):
        return _occurs(protocol.body, role)
    if protocol.sender == role or protocol.receiver == role:
        return True
    return any(_occurs(branch.protocol, role) for branch in protocol.branches)


def _validate_local(
    local: LocalType,
    roles: frozenset[str],
    *,
    bound: frozenset[str] = frozenset(),
    span: Span = None,
) -> None:
    if isinstance(local, LocalEnd):
        return
    if isinstance(local, LocalVar):
        _check_name(local.name, "recursion variable", span=span)
        if local.name not in bound:
            _fail("E005", f"unbound local recursion variable {local.name!r}", span=span)
        return
    if isinstance(local, LocalRec):
        _check_name(local.name, "recursion variable", span=span)
        if local.name in bound:
            _fail(
                "E003", f"shadowed local recursion variable {local.name!r}", span=span
            )
        _validate_local(local.body, roles, bound=bound | {local.name}, span=span)
        return
    if not isinstance(local, LocalSend | LocalRecv):
        _fail("E003", f"unknown local protocol node {type(local).__name__}", span=span)
    _check_name(local.peer, "peer role", span=span)
    if local.peer not in roles:
        _fail("E001", f"peer role {local.peer!r} is not declared", span=span)
    if not local.branches:
        _fail("E003", "a local choice must have at least one branch", span=span)
    labels: set[str] = set()
    for branch in local.branches:
        _check_name(branch.label, "branch label", span=span)
        if branch.label in labels:
            _fail("E003", f"duplicate local branch label {branch.label!r}", span=span)
        labels.add(branch.label)
        _validate_local(branch.protocol, roles, bound=bound, span=span)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _merge_locals(types: Sequence[LocalType], role: str, span: Span) -> LocalType:
    """Merge the projections of every branch of a choice not made by *role*.

    Standard MPST projection requires the projections to be identical.  The
    merge below accepts the slightly more useful case where every branch has
    the same direction and peer and the label sets match, recursively merging
    continuations.  Anything else is a projection error.
    """
    if not types:
        _fail("E003", "cannot merge an empty projection", span=span)
    first = types[0]
    if all(item == first for item in types):
        return first
    if all(isinstance(item, LocalEnd) for item in types):
        return LocalEnd()
    if any(isinstance(item, LocalEnd) for item in types):
        _fail(
            "E003",
            f"role {role!r} participates in only some branches of a choice",
            reason="all branches must agree on whether the role continues",
            help_text="make the role participate uniformly in every branch",
            span=span,
        )
    if not all(isinstance(item, LocalSend) for item in types) and not all(
        isinstance(item, LocalRecv) for item in types
    ):
        _fail(
            "E003",
            f"role {role!r} has incompatible directions across branches",
            reason="a role cannot send in one branch and receive in another "
            "unless it is the choice owner",
            span=span,
        )
    first_send = types[0]
    assert isinstance(first_send, LocalSend | LocalRecv)
    peers = {item.peer for item in types if isinstance(item, LocalSend | LocalRecv)}
    if len(peers) != 1:
        _fail(
            "E003",
            f"role {role!r} communicates with different peers across branches",
            span=span,
        )
    label_maps: list[dict[str, LocalType]] = []
    for item in types:
        assert isinstance(item, LocalSend | LocalRecv)
        label_maps.append({branch.label: branch.protocol for branch in item.branches})
    labels = set(label_maps[0])
    for label_map in label_maps[1:]:
        if set(label_map) != labels:
            _fail(
                "E003",
                f"role {role!r} sees different branch labels across branches",
                span=span,
            )
    merged_branches = tuple(
        LocalBranch(label, _merge_locals([m[label] for m in label_maps], role, span))
        for label in sorted(labels)
    )
    if isinstance(first_send, LocalSend):
        return LocalSend(first_send.peer, merged_branches)
    return LocalRecv(first_send.peer, merged_branches)


def project_global_protocol(
    protocol: GlobalProtocol,
    role: str,
    *,
    span: Span = None,
) -> LocalType:
    """Project a validated global protocol onto one role endpoint."""
    return _project(protocol, role, frozenset(), span)


def _project(
    protocol: GlobalProtocol,
    role: str,
    bound: frozenset[str],
    span: Span,
) -> LocalType:
    if isinstance(protocol, GlobalEnd):
        return LocalEnd()
    if isinstance(protocol, GlobalVar):
        if protocol.name not in bound:
            _fail("E005", f"unbound recursion variable {protocol.name!r}", span=span)
        return LocalVar(protocol.name)
    if isinstance(protocol, GlobalRec):
        if not _occurs(protocol.body, role):
            return LocalEnd()
        return LocalRec(
            protocol.name,
            _project(protocol.body, role, bound | {protocol.name}, span),
        )
    if protocol.sender == role:
        return LocalSend(
            protocol.receiver,
            tuple(
                LocalBranch(
                    branch.label,
                    _project(branch.protocol, role, bound, span),
                )
                for branch in protocol.branches
            ),
        )
    if protocol.receiver == role:
        return LocalRecv(
            protocol.sender,
            tuple(
                LocalBranch(
                    branch.label,
                    _project(branch.protocol, role, bound, span),
                )
                for branch in protocol.branches
            ),
        )
    projected = [
        _project(branch.protocol, role, bound, span) for branch in protocol.branches
    ]
    return _merge_locals(projected, role, span)


# ---------------------------------------------------------------------------
# Bounded interleaving exploration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Exploration:
    status: Status
    trace: tuple[str, ...]
    steps: int
    states: int
    reason: str | None = None


def _normalize(
    local: LocalType,
    env: Mapping[str, LocalType],
    *,
    max_unroll: int,
    span: Span,
) -> tuple[LocalType | None, dict[str, LocalType]]:
    """Unfold head recursion until a communication or ``end`` is exposed.

    ``None`` means the unrolling bound was exhausted; the caller must report
    ``Unknown`` rather than guessing a fixpoint.
    """
    current = local
    current_env = dict(env)
    for _ in range(max_unroll + 1):
        if isinstance(current, LocalRec):
            current_env[current.name] = current.body
            current = current.body
            continue
        if isinstance(current, LocalVar):
            if current.name not in current_env:
                _fail(
                    "E005",
                    f"unbound local recursion variable {current.name!r}",
                    span=span,
                )
            current = current_env[current.name]
            continue
        return current, current_env
    return None, current_env


def _normalize_all(
    local_types: Mapping[str, LocalType],
    envs: Mapping[str, Mapping[str, LocalType]],
    *,
    max_unroll: int,
    span: Span,
) -> tuple[dict[str, LocalType] | None, dict[str, dict[str, LocalType]]]:
    heads: dict[str, LocalType] = {}
    new_envs: dict[str, dict[str, LocalType]] = {}
    for role in sorted(local_types):
        head, env = _normalize(
            local_types[role],
            envs.get(role, {}),
            max_unroll=max_unroll,
            span=span,
        )
        if head is None:
            return None, {}
        heads[role] = head
        new_envs[role] = env
    return heads, new_envs


def _key(
    heads: Mapping[str, LocalType],
    envs: Mapping[str, Mapping[str, LocalType]],
) -> str:
    return repr(
        (
            tuple((role, head) for role, head in sorted(heads.items())),
            tuple(
                (role, tuple(sorted(env.items()))) for role, env in sorted(envs.items())
            ),
        )
    )


def _all_end(heads: Mapping[str, LocalType]) -> bool:
    return all(isinstance(head, LocalEnd) for head in heads.values())


def _branch_map(local: LocalSend | LocalRecv) -> dict[str, LocalType]:
    return {branch.label: branch.protocol for branch in local.branches}


def _enabled_steps(
    heads: Mapping[str, LocalType],
    span: Span,
) -> list[tuple[str, str, str, LocalType, LocalType]]:
    """Return all (sender, receiver, label, sender-cont, receiver-cont)."""
    steps: list[tuple[str, str, str, LocalType, LocalType]] = []
    for sender, head in sorted(heads.items()):
        if not isinstance(head, LocalSend):
            continue
        receiver = head.peer
        if receiver not in heads:
            _fail("E001", f"send target {receiver!r} is not a declared role", span=span)
        peer_head = heads[receiver]
        if not isinstance(peer_head, LocalRecv) or peer_head.peer != sender:
            continue
        sender_branches = _branch_map(head)
        receiver_branches = _branch_map(peer_head)
        if set(sender_branches) != set(receiver_branches):
            _fail(
                "E003",
                f"branch mismatch between {sender!r} and {receiver!r}",
                reason=f"send labels {sorted(sender_branches)} do not match "
                f"receive labels {sorted(receiver_branches)}",
                span=span,
            )
        for label in sorted(sender_branches):
            steps.append(
                (
                    sender,
                    receiver,
                    label,
                    sender_branches[label],
                    receiver_branches[label],
                )
            )
    return steps


def _advance(
    heads: Mapping[str, LocalType],
    envs: Mapping[str, Mapping[str, LocalType]],
    sender: str,
    receiver: str,
    sender_cont: LocalType,
    receiver_cont: LocalType,
    *,
    max_unroll: int,
    span: Span,
) -> tuple[dict[str, LocalType], dict[str, dict[str, LocalType]]] | None:
    new_heads = dict(heads)
    new_envs = {role: dict(env) for role, env in envs.items()}
    for role, cont in ((sender, sender_cont), (receiver, receiver_cont)):
        head, env = _normalize(
            cont,
            new_envs.get(role, {}),
            max_unroll=max_unroll,
            span=span,
        )
        if head is None:
            return None
        new_heads[role] = head
        new_envs[role] = env
    return new_heads, new_envs


def _explore(
    local_types: Mapping[str, LocalType],
    *,
    max_steps: int,
    max_states: int,
    timeout_s: float | None,
    max_unroll: int,
    span: Span,
) -> _Exploration:
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    initial_heads, initial_envs = _normalize_all(
        local_types, {}, max_unroll=max_unroll, span=span
    )
    if initial_heads is None:
        return _Exploration("unknown", (), 0, 0, "recursion unrolling bound exhausted")
    start_key = _key(initial_heads, initial_envs)
    queue: deque[
        tuple[
            dict[str, LocalType],
            dict[str, dict[str, LocalType]],
            tuple[str, ...],
        ]
    ] = deque([(initial_heads, initial_envs, ())])
    seen: dict[str, tuple[str, ...]] = {start_key: ()}
    all_end: dict[str, bool] = {start_key: _all_end(initial_heads)}
    edges: dict[str, list[str]] = {}
    steps = 0

    while queue:
        if deadline is not None and time.monotonic() > deadline:
            return _Exploration(
                "unknown",
                (),
                steps,
                len(seen),
                f"wall-clock timeout after {timeout_s:.3f}s",
            )
        heads, envs, trace = queue.popleft()
        current_key = _key(heads, envs)
        if _all_end(heads):
            continue
        enabled = _enabled_steps(heads, span)
        if not enabled:
            return _Exploration(
                "deadlock",
                trace,
                steps,
                len(seen),
                "reachable state has no enabled communication and is not at end",
            )
        for sender, receiver, label, sender_cont, receiver_cont in enabled:
            steps += 1
            if steps > max_steps:
                return _Exploration(
                    "unknown",
                    trace,
                    steps,
                    len(seen),
                    f"step bound max_steps={max_steps} exhausted",
                )
            advanced = _advance(
                heads,
                envs,
                sender,
                receiver,
                sender_cont,
                receiver_cont,
                max_unroll=max_unroll,
                span=span,
            )
            if advanced is None:
                return _Exploration(
                    "unknown",
                    trace,
                    steps,
                    len(seen),
                    "recursion unrolling bound exhausted during a step",
                )
            next_heads, next_envs = advanced
            next_key = _key(next_heads, next_envs)
            edges.setdefault(current_key, []).append(next_key)
            if next_key in seen:
                continue
            if len(seen) >= max_states:
                return _Exploration(
                    "unknown",
                    trace,
                    steps,
                    len(seen),
                    f"state bound max_states={max_states} exhausted",
                )
            next_trace = (*trace, f"{sender}->{receiver}:{label}")
            seen[next_key] = next_trace
            all_end[next_key] = _all_end(next_heads)
            queue.append((next_heads, next_envs, next_trace))

    # No stuck state was found.  A finite cycle with no path to ``end`` would
    # still be a progress violation, so compute the set of states that can
    # reach a terminal state by walking the recorded edges backwards.
    can_finish = {key for key, done in all_end.items() if done}
    changed = True
    while changed:
        changed = False
        for source, targets in edges.items():
            if source in can_finish:
                continue
            if any(target in can_finish for target in targets):
                can_finish.add(source)
                changed = True
    bad = sorted(set(seen) - can_finish)
    if bad:
        return _Exploration(
            "deadlock",
            seen[bad[0]],
            steps,
            len(seen),
            "reachable cycle has no path to end (progress violation)",
        )
    return _Exploration("proved", (), steps, len(seen), None)


# ---------------------------------------------------------------------------
# Synthesis from local endpoints
# ---------------------------------------------------------------------------


class _SynthUnknown(Exception):
    """Internal signal that the bounded synthesizer ran out of budget."""


def _synth(
    local_types: Mapping[str, LocalType],
    envs: Mapping[str, Mapping[str, LocalType]],
    *,
    max_unroll: int,
    max_steps: int,
    counter: list[int],
    stack: frozenset[str],
    span: Span,
) -> GlobalProtocol | None:
    counter[0] += 1
    if counter[0] > max_steps:
        raise _SynthUnknown
    heads, new_envs = _normalize_all(
        local_types, envs, max_unroll=max_unroll, span=span
    )
    if heads is None:
        raise _SynthUnknown
    if _all_end(heads):
        return GlobalEnd()
    if any(isinstance(head, LocalEnd) for head in heads.values()):
        return None
    key = _key(heads, new_envs)
    if key in stack:
        # A repeated state without a communication step is a bounded
        # synthesizer limitation, not a proof of deadlock.
        return None
    for sender, head in sorted(heads.items()):
        if not isinstance(head, LocalSend):
            continue
        receiver = head.peer
        if receiver not in heads:
            _fail("E001", f"send target {receiver!r} is not a declared role", span=span)
        peer_head = heads[receiver]
        if not isinstance(peer_head, LocalRecv) or peer_head.peer != sender:
            continue
        sender_branches = _branch_map(head)
        receiver_branches = _branch_map(peer_head)
        if set(sender_branches) != set(receiver_branches):
            _fail(
                "E003",
                f"branch mismatch between {sender!r} and {receiver!r}",
                span=span,
            )
        branches: list[GlobalBranch] = []
        for label in sorted(sender_branches):
            next_types = dict(local_types)
            next_types[sender] = sender_branches[label]
            next_types[receiver] = receiver_branches[label]
            sub = _synth(
                next_types,
                new_envs,
                max_unroll=max_unroll,
                max_steps=max_steps,
                counter=counter,
                stack=stack | {key},
                span=span,
            )
            if sub is None:
                break
            branches.append(GlobalBranch(label, sub))
        if len(branches) == len(sender_branches):
            return GlobalSelect(sender, receiver, tuple(branches))
    return None


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionReport:
    """The result of projecting and checking a multiparty protocol."""

    roles: tuple[str, ...]
    local_types: Mapping[str, LocalType]
    global_protocol: GlobalProtocol | None
    synthesized_protocol: GlobalProtocol | None
    status: Status
    steps: int
    states: int
    max_steps: int
    max_states: int
    trace: tuple[str, ...]
    deadlock_trace: tuple[str, ...] | None
    unknown_reason: str | None
    diagnostics: tuple[dict[str, str], ...]

    @property
    def proved(self) -> bool:
        return self.status == "proved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "local_types": {
                role: local.to_dict()
                for role, local in sorted(self.local_types.items())
            },
            "global_protocol": (
                self.global_protocol.to_dict()
                if self.global_protocol is not None
                else None
            ),
            "synthesized_protocol": (
                self.synthesized_protocol.to_dict()
                if self.synthesized_protocol is not None
                else None
            ),
            "status": self.status,
            "steps": self.steps,
            "states": self.states,
            "max_steps": self.max_steps,
            "max_states": self.max_states,
            "trace": list(self.trace),
            "deadlock_trace": (
                list(self.deadlock_trace) if self.deadlock_trace is not None else None
            ),
            "unknown_reason": self.unknown_reason,
            "diagnostics": [dict(item) for item in self.diagnostics],
            "caveats": (
                "deadlock freedom is proved only for the declared finite local "
                "types under the bounded interleaving exploration",
                "recursion is unrolled up to max_unroll; exhaustion reports Unknown",
                "wall-clock timeouts and state-space bounds report Unknown, never proof",
            ),
        }


def _coerce_local_map(
    protocol: Mapping[str, LocalType],
    roles: Sequence[str] | None,
    span: Span,
) -> tuple[dict[str, LocalType], tuple[str, ...]]:
    local_types = dict(protocol)
    if not local_types:
        _fail("E003", "a local protocol map must contain at least one role", span=span)
    for role, local in local_types.items():
        _check_name(role, "role", span=span)
        if not isinstance(
            local, LocalEnd | LocalVar | LocalRec | LocalSend | LocalRecv
        ):
            _fail(
                "E003",
                f"role {role!r} does not map to a local protocol",
                span=span,
            )
    declared = tuple(sorted(local_types)) if roles is None else tuple(roles)
    if len(set(declared)) != len(declared):
        _fail("E003", "roles must be unique", span=span)
    if set(declared) != set(local_types):
        _fail(
            "E001",
            "the role list does not match the local protocol map",
            reason=f"declared {sorted(declared)} but map has {sorted(local_types)}",
            span=span,
        )
    role_set = frozenset(declared)
    for role, local in local_types.items():
        _validate_local(local, role_set, span=span)
    return local_types, declared


def session_synthesize_and_project_protocols(
    protocol: GlobalProtocol | Mapping[str, LocalType],
    *,
    roles: Sequence[str] | None = None,
    max_steps: int = 256,
    max_states: int = 4096,
    max_unroll: int = 64,
    timeout_s: float | None = None,
    strict: bool = True,
    span: Span = None,
) -> SessionReport:
    """Project a global protocol (or synthesize one from local endpoints).

    The function accepts either a :data:`GlobalProtocol` tree or a mapping
    from role names to :data:`LocalType` endpoints.

    * Global input is validated, projected onto every role, and then checked by
      a bounded asynchronous scheduler.
    * Local input is checked by the same scheduler; when it is deadlock-free a
      global protocol is reconstructed from the local endpoints.

    A reachable stuck state raises ``E002`` with a protocol trace.  Exhausting
    ``max_steps``, ``max_states``, ``max_unroll``, or ``timeout_s`` reports
    ``Unknown`` and raises ``E002`` when ``strict`` is true.  With
    ``strict=False`` the report is returned for inspection.
    """
    if max_steps <= 0 or max_states <= 0 or max_unroll <= 0:
        _fail("E003", "analysis bounds must be positive", span=span)
    if timeout_s is not None and timeout_s < 0:
        _fail("E003", "timeout_s must be non-negative", span=span)

    if isinstance(protocol, Mapping):
        local_types, declared_roles = _coerce_local_map(protocol, roles, span)
        global_protocol = None
        synthesized: GlobalProtocol | None = None
    else:
        declared_roles = (
            tuple(sorted(_roles_of(protocol))) if roles is None else tuple(roles)
        )
        if len(set(declared_roles)) != len(declared_roles):
            _fail("E003", "roles must be unique", span=span)
        for role in declared_roles:
            _check_name(role, "role", span=span)
        role_set = frozenset(declared_roles)
        _validate_global(protocol, role_set, span=span)
        local_types = {
            role: project_global_protocol(protocol, role, span=span)
            for role in declared_roles
        }
        global_protocol = protocol
        synthesized = protocol

    exploration = _explore(
        local_types,
        max_steps=max_steps,
        max_states=max_states,
        timeout_s=timeout_s,
        max_unroll=max_unroll,
        span=span,
    )
    diagnostics: list[dict[str, str]] = []
    if isinstance(protocol, Mapping) and exploration.status == "proved":
        try:
            synthesized = _synth(
                local_types,
                {},
                max_unroll=max_unroll,
                max_steps=max_steps,
                counter=[0],
                stack=frozenset(),
                span=span,
            )
        except _SynthUnknown:
            synthesized = None
            diagnostics.append(
                {
                    "code": "E002",
                    "message": "global synthesis exceeded its bound; the local "
                    "endpoints were still proved deadlock-free",
                }
            )
        if synthesized is None:
            diagnostics.append(
                {
                    "code": "E002",
                    "message": "no global protocol was reconstructed from the "
                    "local endpoints within the bound",
                }
            )

    report = SessionReport(
        roles=declared_roles,
        local_types=local_types,
        global_protocol=global_protocol,
        synthesized_protocol=synthesized,
        status=exploration.status,
        steps=exploration.steps,
        states=exploration.states,
        max_steps=max_steps,
        max_states=max_states,
        trace=exploration.trace,
        deadlock_trace=(
            exploration.trace if exploration.status == "deadlock" else None
        ),
        unknown_reason=exploration.reason,
        diagnostics=tuple(diagnostics),
    )
    if strict and exploration.status == "deadlock":
        trace = " -> ".join(exploration.trace) or "<initial state>"
        _fail(
            "E002",
            "multiparty protocol is not deadlock-free",
            reason=f"{exploration.reason}; trace: {trace}",
            help_text="reorder the interactions or make every choice branch "
            "uniform for every role",
            span=span,
        )
    if strict and exploration.status == "unknown":
        _fail(
            "E002",
            "multiparty protocol deadlock freedom is Unknown",
            reason=exploration.reason or "bounded exploration exhausted",
            help_text="increase max_steps/max_states/max_unroll or simplify recursion",
            span=span,
        )
    return report


__all__ = [
    "GlobalBranch",
    "GlobalEnd",
    "GlobalProtocol",
    "GlobalRec",
    "GlobalSelect",
    "GlobalVar",
    "LocalBranch",
    "LocalEnd",
    "LocalRec",
    "LocalRecv",
    "LocalSend",
    "LocalType",
    "LocalVar",
    "SessionReport",
    "project_global_protocol",
    "session_synthesize_and_project_protocols",
]
