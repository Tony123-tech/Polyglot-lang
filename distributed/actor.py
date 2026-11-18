"""Actors, portable snapshots, and the replicated checkpoint store.

An actor is the unit of migration.  Its state is deliberately *plain data*:
wire-safe Python values, registered native values, channel handles, lease
tokens, and checkpointable tasks.  Nothing actor-specific lives in a thread or
a C frame, so a snapshot can be rebuilt on another node.

The runtime does not pretend that a suspended Rust future can be copied.  Tasks
move as :mod:`distributed.tasks` journals and are replayed; channels move as
:class:`distributed.connections.ChannelHandle` values and stay anchored to the
node that owns the socket.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from distributed.connections import ChannelHandle
from distributed.diagnostics import DistributedError, node_lost
from distributed.leases import LeaseToken
from distributed.tasks import CheckpointableTask, TaskEnvironment
from distributed.transmute import (
    decode_state,
    digest_bytes,
    encode_state,
    register_type,
)
from distributed.wire import KIND_CHECKPOINT

ACTIVE: Final = "active"
MIGRATING: Final = "migrating"
MIGRATED: Final = "migrated"
FAILED: Final = "failed"

MAX_MAILBOX: Final = 100_000


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

ActorHandler = Callable[["Actor", bytes], Any]
_ACTOR_HANDLERS: dict[str, ActorHandler] = {}


def register_actor_handler(
    name: str, fn: ActorHandler, *, replace: bool = True
) -> ActorHandler:
    if not name:
        raise DistributedError(
            "an actor handler needs a non-empty name",
            code="E003",
            note="ActorProtocolError",
        )
    if name in _ACTOR_HANDLERS and not replace:
        raise DistributedError(
            f"actor handler {name!r} is already registered",
            code="E003",
            note="ActorProtocolError",
        )
    _ACTOR_HANDLERS[name] = fn
    return fn


def actor_handler(name: str | None = None) -> Callable[[ActorHandler], ActorHandler]:
    def decorate(fn: ActorHandler) -> ActorHandler:
        register_actor_handler(name or fn.__name__, fn)
        return fn

    return decorate


def get_actor_handler(name: str) -> ActorHandler:
    fn = _ACTOR_HANDLERS.get(name)
    if fn is None:
        raise DistributedError(
            f"actor handler {name!r} is not registered on this node",
            code="E006",
            note="MissingHandlerError",
            help_text="both nodes must import the same actor handlers",
            context={"handler": name},
        )
    return fn


def registered_actor_handlers() -> dict[str, ActorHandler]:
    return dict(_ACTOR_HANDLERS)


@actor_handler("counter")
def _counter_handler(actor: Actor, payload: bytes) -> int:
    delta = int.from_bytes(payload, "big", signed=True) if payload else 1
    count = int(actor.state.get("count", 0)) + delta
    actor.state["count"] = count
    actor.state.setdefault("log", []).append(delta)
    return count


@actor_handler("echo")
def _echo_handler(actor: Actor, payload: bytes) -> bytes:
    actor.state.setdefault("messages", []).append(payload)
    return payload


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class ActorSnapshot:
    """A complete, portable image of one actor."""

    actor_id: str
    handler: str = "counter"
    generation: int = 1
    state: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, bytes] = field(default_factory=dict)
    channels: dict[str, ChannelHandle] = field(default_factory=dict)
    mailbox: list[bytes] = field(default_factory=list)
    lease_key: str = ""
    lease: LeaseToken | None = None
    epoch: int = 0
    created_at: float = field(default_factory=time.time)
    digest: str = ""

    def compute_digest(self) -> str:
        return digest_bytes(
            encode_state(
                [
                    self.actor_id,
                    self.handler,
                    self.generation,
                    self.state,
                    self.tasks,
                    self.channels,
                    self.mailbox,
                    self.lease_key,
                    self.lease,
                    self.epoch,
                ]
            )
        )

    def seal(self) -> ActorSnapshot:
        self.digest = self.compute_digest()
        return self

    def encode(self) -> bytes:
        if not self.digest:
            self.seal()
        return encode_state(self, kind=KIND_CHECKPOINT)

    @classmethod
    def decode(cls, data: bytes) -> ActorSnapshot:
        value = decode_state(data, expected_kind=KIND_CHECKPOINT)
        if not isinstance(value, ActorSnapshot):
            raise DistributedError(
                "checkpoint does not contain an actor snapshot",
                note="StateMismatchError",
                context={"type": type(value).__name__},
            )
        expected = value.compute_digest()
        if value.digest and value.digest != expected:
            raise DistributedError(
                f"actor {value.actor_id!r} snapshot digest mismatch",
                note="StateMismatchError",
                help_text=(
                    "the snapshot was corrupted in transit or produced by a "
                    "different layout"
                ),
                context={
                    "actor_id": value.actor_id,
                    "expected": expected,
                    "actual": value.digest,
                },
            )
        return value


register_type(ActorSnapshot, name="ActorSnapshot")


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


@dataclass
class Actor:
    actor_id: str
    handler: str = "counter"
    state: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, CheckpointableTask] = field(default_factory=dict)
    channels: dict[str, ChannelHandle] = field(default_factory=dict)
    mailbox: list[bytes] = field(default_factory=list)
    lease_key: str = ""
    lease: LeaseToken | None = None
    node: str = ""
    status: str = ACTIVE
    generation: int = 0
    epoch: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    migrated_from: str | None = None
    last_result: Any = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise DistributedError(
                "an actor needs a non-empty id",
                code="E003",
                note="ActorProtocolError",
            )
        if not self.lease_key:
            self.lease_key = f"actor:{self.actor_id}"
        self.state = dict(self.state)
        self.tasks = dict(self.tasks)
        self.channels = dict(self.channels)
        self.mailbox = list(self.mailbox)

    # -- mailbox / execution ---------------------------------------------
    def deliver(self, payload: bytes) -> None:
        if self.status == MIGRATED:
            raise DistributedError(
                f"actor {self.actor_id!r} has migrated away",
                code="E006",
                note="ActorMovedError",
                help_text="route the message to the actor's new node",
                context={"actor_id": self.actor_id, "node": self.node},
            )
        if len(self.mailbox) >= MAX_MAILBOX:
            raise DistributedError(
                f"actor {self.actor_id!r} mailbox is full",
                note="StateTooLargeError",
            )
        self.mailbox.append(bytes(payload))
        self.updated_at = time.time()

    def handle_one(self, environment: TaskEnvironment) -> Any:
        if not self.mailbox:
            return None
        payload = self.mailbox.pop(0)
        fn = get_actor_handler(self.handler)
        try:
            result = fn(self, payload)
        except DistributedError as error:
            self.errors.append(f"{error.code}: {error.diagnostic.message}")
            self.updated_at = time.time()
            raise
        self.last_result = result
        self.updated_at = time.time()
        return result

    def drain(self, environment: TaskEnvironment, *, limit: int = 100) -> int:
        processed = 0
        while self.mailbox and processed < limit:
            self.handle_one(environment)
            processed += 1
        return processed

    def add_task(self, task: CheckpointableTask) -> CheckpointableTask:
        if task.task_id in self.tasks:
            raise DistributedError(
                f"actor {self.actor_id!r} already has task {task.task_id!r}",
                code="E003",
                note="ActorProtocolError",
            )
        self.tasks[task.task_id] = task
        self.updated_at = time.time()
        return task

    def advance_tasks(self, environment: TaskEnvironment) -> list[CheckpointableTask]:
        advanced = []
        for task in list(self.tasks.values()):
            if task.finished:
                continue
            task.advance(environment)
            advanced.append(task)
        self.updated_at = time.time()
        return advanced

    # -- lease-protected fields ------------------------------------------
    def write(
        self,
        key: str,
        value: Any,
        *,
        authority: Any = None,
    ) -> None:
        if authority is not None:
            if self.lease is None:
                raise DistributedError(
                    f"actor {self.actor_id!r} has no lease for a write",
                    note="LeaseExpiredError",
                    help_text="acquire an exclusive lease before writing",
                    context={"actor_id": self.actor_id, "key": key},
                )
            authority.validate(self.lease, write=True)
        self.state[key] = value
        self.updated_at = time.time()

    def read(self, key: str, default: Any = None, *, authority: Any = None) -> Any:
        if authority is not None and self.lease is not None:
            authority.validate(self.lease, write=False)
        return self.state.get(key, default)

    # -- snapshot ---------------------------------------------------------
    def export_snapshot(self, *, epoch: int | None = None) -> ActorSnapshot:
        self.generation += 1
        snapshot = ActorSnapshot(
            actor_id=self.actor_id,
            handler=self.handler,
            generation=self.generation,
            state=_wire_safe(self.state, self.actor_id),
            tasks={
                task_id: task.encode()
                for task_id, task in self.tasks.items()
                if not task.finished
            },
            channels=dict(self.channels),
            mailbox=list(self.mailbox),
            lease_key=self.lease_key,
            lease=self.lease,
            epoch=self.epoch if epoch is None else epoch,
            created_at=self.created_at,
        )
        return snapshot.seal()

    @classmethod
    def from_snapshot(cls, snapshot: ActorSnapshot, *, node: str = "") -> Actor:
        tasks: dict[str, CheckpointableTask] = {}
        for task_id, data in snapshot.tasks.items():
            task = CheckpointableTask.decode(data)
            if task.task_id != task_id:
                raise DistributedError(
                    f"task key {task_id!r} holds task {task.task_id!r}",
                    note="StateMismatchError",
                )
            tasks[task_id] = task
        actor = cls(
            actor_id=snapshot.actor_id,
            handler=snapshot.handler,
            state=dict(snapshot.state),
            tasks=tasks,
            channels=dict(snapshot.channels),
            mailbox=list(snapshot.mailbox),
            lease_key=snapshot.lease_key,
            lease=snapshot.lease,
            node=node,
            status=ACTIVE,
            generation=snapshot.generation,
            epoch=snapshot.epoch,
            created_at=snapshot.created_at,
        )
        return actor

    def checkpoint(self) -> bytes:
        return self.export_snapshot().encode()

    def restore(self, data: bytes) -> None:
        snapshot = ActorSnapshot.decode(data)
        restored = Actor.from_snapshot(snapshot, node=self.node)
        self.handler = restored.handler
        self.state = restored.state
        self.tasks = restored.tasks
        self.channels = restored.channels
        self.mailbox = restored.mailbox
        self.lease_key = restored.lease_key
        self.lease = restored.lease
        self.generation = restored.generation
        self.epoch = restored.epoch
        self.status = ACTIVE
        self.updated_at = time.time()

    def describe(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "handler": self.handler,
            "node": self.node,
            "status": self.status,
            "generation": self.generation,
            "epoch": self.epoch,
            "lease_key": self.lease_key,
            "fencing": None if self.lease is None else self.lease.fencing,
            "mailbox": len(self.mailbox),
            "tasks": {task_id: task.describe() for task_id, task in self.tasks.items()},
            "state": _describe_state(self.state),
            "errors": list(self.errors),
        }


def _wire_safe(state: dict[str, Any], actor_id: str) -> dict[str, Any]:
    from distributed.transmute import encode_state

    for key, value in state.items():
        try:
            encode_state(value)
        except DistributedError as error:
            raise DistributedError(
                f"actor {actor_id!r} field {key!r} is not wire-serializable",
                code=error.code,
                note="StateMismatchError",
                reason=error.diagnostic.message,
                help_text=(
                    "register the type with distributed.transmute.register_type() "
                    "or store plain data / NativeValue"
                ),
                context={"actor_id": actor_id, "field": key},
            ) from error
    return dict(state)


def _describe_state(state: dict[str, Any]) -> dict[str, Any]:
    described: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            described[key] = value
        elif isinstance(value, (list, tuple, dict, set, frozenset)):
            described[key] = f"<{type(value).__name__} len={len(value)}>"
        else:
            described[key] = f"<{type(value).__name__}>"
    return described


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRecord:
    actor_id: str
    node: str
    generation: int
    digest: str
    data: bytes
    created_at: float


class CheckpointStore:
    """A bounded, thread-safe, replicated in-memory checkpoint log."""

    def __init__(self, *, history: int = 8) -> None:
        if history < 1:
            raise ValueError("history must be positive")
        self.history = history
        self._records: dict[str, list[CheckpointRecord]] = {}
        self._lock = threading.Lock()

    def put(
        self,
        actor_id: str,
        data: bytes,
        *,
        node: str,
        generation: int,
        digest: str | None = None,
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            actor_id=actor_id,
            node=node,
            generation=int(generation),
            digest=digest or digest_bytes(data),
            data=bytes(data),
            created_at=time.time(),
        )
        with self._lock:
            entries = self._records.setdefault(actor_id, [])
            entries.append(record)
            if len(entries) > self.history:
                del entries[: len(entries) - self.history]
        return record

    def latest(self, actor_id: str) -> CheckpointRecord | None:
        with self._lock:
            entries = self._records.get(actor_id)
            return entries[-1] if entries else None

    def history_for(self, actor_id: str) -> list[CheckpointRecord]:
        with self._lock:
            return list(self._records.get(actor_id, ()))

    def actor_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._records)

    def drop(self, actor_id: str) -> None:
        with self._lock:
            self._records.pop(actor_id, None)

    def restore_latest(self, actor_id: str, *, node: str = "") -> Actor:
        record = self.latest(actor_id)
        if record is None:
            raise DistributedError(
                f"no checkpoint for actor {actor_id!r}",
                code="E006",
                note="MissingCheckpointError",
                help_text="the actor must checkpoint before a node can fail",
                context={"actor_id": actor_id},
            )
        snapshot = ActorSnapshot.decode(record.data)
        actor = Actor.from_snapshot(snapshot, node=node)
        actor.status = ACTIVE
        return actor

    def restore_on_node(
        self, actor_id: str, node: str, *, epoch: int | None = None
    ) -> Actor:
        actor = self.restore_latest(actor_id, node=node)
        if epoch is not None:
            actor.epoch = epoch
        actor.migrated_from = "checkpoint"
        return actor


def node_lost_error(node: str, *, actor_id: str = "") -> DistributedError:
    return node_lost(
        f"node {node!r} was lost; actor state was restored from checkpoint",
        context={"node": node, "actor_id": actor_id},
    )


__all__ = [
    "ACTIVE",
    "FAILED",
    "MAX_MAILBOX",
    "MIGRATED",
    "MIGRATING",
    "Actor",
    "ActorHandler",
    "ActorSnapshot",
    "CheckpointRecord",
    "CheckpointStore",
    "actor_handler",
    "get_actor_handler",
    "node_lost_error",
    "register_actor_handler",
    "registered_actor_handlers",
]
