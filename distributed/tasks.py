"""Checkpointable tasks that can be paused mid-await and rebuilt elsewhere.

CPython does not expose a portable way to copy a suspended frame (and a PyO3
future's Rust state machine is even less portable).  The runtime therefore uses
**deterministic replay** instead of pretending to snapshot the C stack:

1. An async handler is written as an ordinary function that performs every
   suspension through ``ctx.await_(operation)``.
2. Each completed operation and its result become a *journal entry*.
3. To checkpoint, the runtime stores the handler name, its arguments, the
   journal, and a read-only locals snapshot.
4. On the target node the handler runs again from the top.  Replayed awaits
   return the journalled results without touching the network; the first
   operation past the journal is executed for real.

This is the same model durable-execution engines use.  The contract is honest
and checkable: handlers must be deterministic up to each await point.  A handler
that observes a different operation than the journal recorded is a replay
violation and raises ``E008`` (``TaskReplayError``) rather than silently
diverging.

The operations are wire values, so a journal and a pending operation are part
of the same self-describing snapshot as the rest of the actor state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from distributed.connections import AnchorChannel, ChannelHandle
from distributed.diagnostics import DistributedError, task_replay
from distributed.transmute import (
    decode_state,
    encode_state,
    register_type,
)
from distributed.transmute import (
    encode_checkpoint as encode_checkpoint_frame,
)

MAX_STEPS: Final = 10_000

STATUS_RUNNABLE: Final = "runnable"
STATUS_PENDING: Final = "pending"
STATUS_DONE: Final = "done"
STATUS_FAILED: Final = "failed"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SleepOp:
    seconds: float
    kind: ClassVar[str] = "sleep"


@dataclass(frozen=True)
class RecvOp:
    channel: ChannelHandle
    limit: int = 64
    kind: ClassVar[str] = "recv"


@dataclass(frozen=True)
class SendOp:
    channel: ChannelHandle
    data: bytes
    kind: ClassVar[str] = "send"


@dataclass(frozen=True)
class AwaitOp:
    name: str
    payload: Any = None
    kind: ClassVar[str] = "await"


@dataclass(frozen=True)
class CheckpointOp:
    label: str = ""
    kind: ClassVar[str] = "checkpoint"


for _op in (SleepOp, RecvOp, SendOp, AwaitOp, CheckpointOp):
    register_type(_op, name=_op.__name__)


class OperationPending(Exception):
    """Raised by an environment when an operation cannot complete yet."""

    def __init__(self, op: Any) -> None:
        self.op = op
        super().__init__(f"operation is pending: {type(op).__name__}")


class TaskEnvironment:
    """The node services a checkpointable handler is allowed to await on."""

    def perform(self, op: Any, *, task_id: str = "", index: int = 0) -> Any:
        raise NotImplementedError


class LocalEnvironment(TaskEnvironment):
    """A deterministic environment for tests and single-process actors.

    * sleeps use ``clock`` (monotonic by default);
    * channel operations use an anchor RPC ``call`` if supplied;
    * external awaits are callables returning ``(ready, result)``;
    * :class:`CheckpointOp` always completes.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        awaitables: dict[str, Callable[[Any], tuple[bool, Any]]] | None = None,
    ) -> None:
        self.clock = clock or time.monotonic
        self.call = call
        self.awaitables = dict(awaitables or {})
        self._deadlines: dict[tuple[str, int], float] = {}
        self.performed = 0

    def register(self, name: str, fn: Callable[[Any], tuple[bool, Any]]) -> None:
        self.awaitables[name] = fn

    def perform(self, op: Any, *, task_id: str = "", index: int = 0) -> Any:
        if isinstance(op, SleepOp):
            key = (task_id, index)
            deadline = self._deadlines.get(key)
            if deadline is None:
                deadline = self.clock() + float(op.seconds)
                self._deadlines[key] = deadline
            if self.clock() < deadline:
                raise OperationPending(op)
            self._deadlines.pop(key, None)
            self.performed += 1
            return None

        if isinstance(op, RecvOp):
            channel = self._channel(op.channel)
            messages = channel.recv(limit=op.limit)
            if not messages:
                raise OperationPending(op)
            channel.ack()
            self.performed += 1
            return messages

        if isinstance(op, SendOp):
            channel = self._channel(op.channel)
            seq = channel.send(op.data)
            self.performed += 1
            return seq

        if isinstance(op, AwaitOp):
            fn = self.awaitables.get(op.name)
            if fn is None:
                raise DistributedError(
                    f"no awaitable named {op.name!r} is registered on this node",
                    code="E006",
                    note="MissingHandlerError",
                    help_text="register it with environment.register(name, fn)",
                    context={"awaitable": op.name},
                )
            ready, result = fn(op.payload)
            if not ready:
                raise OperationPending(op)
            self.performed += 1
            return result

        if isinstance(op, CheckpointOp):
            self.performed += 1
            return op.label

        raise DistributedError(
            f"unsupported task operation {type(op).__name__}",
            code="E003",
            note="TaskProtocolError",
            help_text="await a SleepOp, RecvOp, SendOp, AwaitOp, or CheckpointOp",
        )

    def _channel(self, handle: ChannelHandle) -> AnchorChannel:
        if self.call is None:
            raise DistributedError(
                "this environment has no anchor RPC client",
                code="E003",
                note="TaskProtocolError",
                help_text="pass call=registry.handle_request to LocalEnvironment",
            )
        return AnchorChannel(handle, self.call)


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JournalEntry:
    index: int
    op: Any
    result: Any


register_type(JournalEntry, name="JournalEntry")


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

Handler = Callable[..., Any]
_HANDLERS: dict[str, Handler] = {}


def register_handler(name: str, fn: Handler, *, replace: bool = True) -> Handler:
    if not name:
        raise DistributedError(
            "a handler needs a non-empty name",
            code="E003",
            note="TaskProtocolError",
        )
    if name in _HANDLERS and not replace:
        raise DistributedError(
            f"handler {name!r} is already registered",
            code="E003",
            note="TaskProtocolError",
        )
    _HANDLERS[name] = fn
    return fn


def handler(name: str | None = None) -> Callable[[Handler], Handler]:
    """Decorator form of :func:`register_handler`."""

    def decorate(fn: Handler) -> Handler:
        register_handler(name or fn.__name__, fn)
        return fn

    return decorate


def get_handler(name: str) -> Handler:
    fn = _HANDLERS.get(name)
    if fn is None:
        raise DistributedError(
            f"task handler {name!r} is not registered on this node",
            code="E006",
            note="MissingHandlerError",
            help_text=(
                "both nodes must import/register the same handler before a migration"
            ),
            context={"handler": name},
        )
    return fn


def registered_handlers() -> dict[str, Handler]:
    return dict(_HANDLERS)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TaskContext:
    """The object passed as ``ctx`` to every checkpointable handler."""

    def __init__(
        self,
        task: CheckpointableTask,
        environment: TaskEnvironment,
    ) -> None:
        self.task = task
        self.environment = environment
        self.index = 0
        self.recorded: list[JournalEntry] = []
        self.locals: dict[str, Any] = dict(task.locals)

    # -- await surface ----------------------------------------------------
    def await_(self, op: Any) -> Any:
        if self.index < len(self.task.journal):
            entry = self.task.journal[self.index]
            if entry.op != op:
                raise task_replay(
                    f"task {self.task.task_id!r} replayed a different "
                    f"operation at index {self.index}",
                    reason=(f"journalled {entry.op!r}, handler produced {op!r}"),
                    help_text=(
                        "handlers must be deterministic up to each await; "
                        "branch only on journalled results"
                    ),
                    context={
                        "task_id": self.task.task_id,
                        "handler": self.task.handler,
                        "index": self.index,
                    },
                )
            self.recorded.append(entry)
            self.index += 1
            return entry.result

        try:
            result = self.environment.perform(
                op, task_id=self.task.task_id, index=self.index
            )
        except OperationPending:
            self.task.pending = op
            raise
        self.recorded.append(JournalEntry(self.index, op, result))
        self.index += 1
        return result

    # -- conveniences -----------------------------------------------------
    def sleep(self, seconds: float) -> None:
        self.await_(SleepOp(float(seconds)))

    def sleep_ms(self, milliseconds: float) -> None:
        self.sleep(float(milliseconds) / 1000.0)

    def send(self, channel: ChannelHandle, data: bytes) -> int:
        return self.await_(SendOp(channel, bytes(data)))

    def recv(self, channel: ChannelHandle, *, limit: int = 64) -> list[bytes]:
        return self.await_(RecvOp(channel, limit))

    def await_external(self, name: str, payload: Any = None) -> Any:
        return self.await_(AwaitOp(name, payload))

    def checkpoint(self, label: str = "") -> str:
        return self.await_(CheckpointOp(label))

    def local(self, name: str, value: Any) -> None:
        """Record an observational local for diagnostics.

        The snapshot is reported to operators but never used to resume
        control flow; only the journal is authoritative.
        """
        self.locals[name] = _safe_local(value)

    def locals_snapshot(self) -> dict[str, Any]:
        return {
            key: (
                self.locals[key]
                if _is_wire_safe(self.locals[key])
                else repr(self.locals[key])
            )
            for key in self.locals
        }


def _is_wire_safe(value: Any) -> bool:
    try:
        encode_state(value, tolerant=True)
    except DistributedError:
        return False
    return True


def _safe_local(value: Any) -> Any:
    if _is_wire_safe(value):
        return value
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


@dataclass
class CheckpointableTask:
    """A durable handler invocation with an await journal."""

    task_id: str
    handler: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    journal: list[JournalEntry] = field(default_factory=list)
    locals: dict[str, Any] = field(default_factory=dict)
    pending: Any = None
    status: str = STATUS_RUNNABLE
    result: Any = None
    error: str | None = None
    checkpoints: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise DistributedError(
                "a task needs a non-empty id",
                code="E003",
                note="TaskProtocolError",
            )
        if self.status not in {
            STATUS_RUNNABLE,
            STATUS_PENDING,
            STATUS_DONE,
            STATUS_FAILED,
        }:
            raise DistributedError(
                f"unknown task status {self.status!r}",
                code="E003",
                note="TaskProtocolError",
            )
        self.args = tuple(self.args)
        self.kwargs = dict(self.kwargs)
        self.journal = list(self.journal)

    # -- execution --------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.status in {STATUS_DONE, STATUS_FAILED}

    def advance(self, environment: TaskEnvironment) -> CheckpointableTask:
        """Run until the next suspension or completion."""
        if self.finished:
            return self
        fn = get_handler(self.handler)
        ctx = TaskContext(self, environment)
        self.status = STATUS_RUNNABLE
        try:
            result = fn(ctx, *self.args, **self.kwargs)
        except OperationPending as pending:
            self.journal = ctx.recorded
            self.pending = pending.op
            self.locals = ctx.locals_snapshot()
            self.status = STATUS_PENDING
            self.checkpoints += 1
            return self
        except DistributedError as error:
            self.journal = ctx.recorded
            self.locals = ctx.locals_snapshot()
            self.status = STATUS_FAILED
            self.error = f"{error.code}: {error.diagnostic.message}"
            self.pending = None
            self.checkpoints += 1
            raise
        except Exception as error:
            self.journal = ctx.recorded
            self.locals = ctx.locals_snapshot()
            self.status = STATUS_FAILED
            self.error = f"{type(error).__name__}: {error}"
            self.pending = None
            self.checkpoints += 1
            return self

        if ctx.index < len(self.journal):
            raise task_replay(
                f"task {self.task_id!r} completed after {ctx.index} await(s) "
                f"but the journal holds {len(self.journal)} entry(ies)",
                reason="the handler changed shape relative to its journal",
                help_text=(
                    "do not remove or reorder awaits after a task has been checkpointed"
                ),
                context={
                    "task_id": self.task_id,
                    "handler": self.handler,
                    "consumed": ctx.index,
                    "journalled": len(self.journal),
                },
            )
        self.journal = ctx.recorded
        self.locals = ctx.locals_snapshot()
        self.pending = None
        self.status = STATUS_DONE
        self.result = result
        self.checkpoints += 1
        return self

    def run_to_completion(
        self,
        environment: TaskEnvironment,
        *,
        max_steps: int = MAX_STEPS,
        sleep: Callable[[float], None] | None = None,
    ) -> CheckpointableTask:
        """Advance repeatedly, optionally sleeping between pending rounds.

        This is a convenience for sequential actors.  A real reactor calls
        :meth:`advance` when its pending operation becomes ready.
        """
        sleeper = sleep or time.sleep
        for _ in range(max_steps):
            if self.finished:
                return self
            self.advance(environment)
            if self.finished:
                return self
            sleeper(0.001)
        raise DistributedError(
            f"task {self.task_id!r} did not finish in {max_steps} steps",
            note="TaskReplayError",
            help_text="check for a never-ready operation",
            context={"task_id": self.task_id, "handler": self.handler},
        )

    # -- snapshot ---------------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "handler": self.handler,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "journal": list(self.journal),
            "locals": dict(self.locals),
            "pending": self.pending,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "checkpoints": self.checkpoints,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CheckpointableTask:
        journal = [
            entry if isinstance(entry, JournalEntry) else JournalEntry(**entry)
            for entry in payload.get("journal", [])
        ]
        return cls(
            task_id=payload["task_id"],
            handler=payload["handler"],
            args=tuple(payload.get("args", ())),
            kwargs=dict(payload.get("kwargs", {})),
            journal=journal,
            locals=dict(payload.get("locals", {})),
            pending=payload.get("pending"),
            status=payload.get("status", STATUS_RUNNABLE),
            result=payload.get("result"),
            error=payload.get("error"),
            checkpoints=int(payload.get("checkpoints", 0)),
            created_at=float(payload.get("created_at", time.time())),
        )

    def encode(self) -> bytes:
        return encode_checkpoint_frame(self.to_payload())

    @classmethod
    def decode(cls, data: bytes) -> CheckpointableTask:
        payload = decode_state(data)
        if not isinstance(payload, dict):
            raise task_replay(
                "task checkpoint did not decode to a mapping",
                note="StateMismatchError",
            )
        return cls.from_payload(payload)

    def clone(self) -> CheckpointableTask:
        """Round-trip through the wire codec, as a migration would."""
        return CheckpointableTask.decode(self.encode())

    def describe(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "handler": self.handler,
            "status": self.status,
            "await_index": len(self.journal),
            "pending": (
                type(self.pending).__name__ if self.pending is not None else None
            ),
            "checkpoints": self.checkpoints,
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """A tiny cooperative scheduler over checkpointable tasks."""

    def __init__(self, environment: TaskEnvironment) -> None:
        self.environment = environment
        self.tasks: dict[str, CheckpointableTask] = {}

    def submit(self, task: CheckpointableTask) -> CheckpointableTask:
        if task.task_id in self.tasks:
            raise DistributedError(
                f"task {task.task_id!r} already exists",
                code="E003",
                note="TaskProtocolError",
            )
        self.tasks[task.task_id] = task
        return task

    def add(self, task: CheckpointableTask) -> CheckpointableTask:
        self.tasks[task.task_id] = task
        return task

    def remove(self, task_id: str) -> CheckpointableTask | None:
        return self.tasks.pop(task_id, None)

    def step(self) -> list[CheckpointableTask]:
        """Advance every unfinished task once."""
        advanced = []
        for task in list(self.tasks.values()):
            if task.finished:
                continue
            task.advance(self.environment)
            advanced.append(task)
        return advanced

    def run_until_idle(self, *, max_rounds: int = MAX_STEPS) -> None:
        for _ in range(max_rounds):
            if not self.step():
                return
        raise DistributedError(
            f"scheduler did not reach idle in {max_rounds} rounds",
            note="TaskReplayError",
        )

    def pending(self) -> list[CheckpointableTask]:
        return [task for task in self.tasks.values() if not task.finished]

    def checkpoint_all(self) -> dict[str, bytes]:
        return {
            task_id: task.encode()
            for task_id, task in self.tasks.items()
            if not task.finished
        }

    def restore(self, payloads: dict[str, bytes]) -> None:
        for task_id, data in payloads.items():
            task = CheckpointableTask.decode(data)
            if task.task_id != task_id:
                raise task_replay(
                    f"checkpoint key {task_id!r} holds task {task.task_id!r}",
                    note="StateMismatchError",
                )
            self.tasks[task.task_id] = task


__all__ = [
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNABLE",
    "AwaitOp",
    "CheckpointOp",
    "CheckpointableTask",
    "Handler",
    "JournalEntry",
    "LocalEnvironment",
    "OperationPending",
    "RecvOp",
    "SendOp",
    "SleepOp",
    "TaskContext",
    "TaskEnvironment",
    "TaskScheduler",
    "get_handler",
    "handler",
    "register_handler",
    "registered_handlers",
]
