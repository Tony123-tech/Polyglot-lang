"""Experimental distributed runtime for Polyglot.

The package provides:

* :mod:`distributed.wire` -- CRC-checked, versioned framing primitives.
* :mod:`distributed.transmute` -- the dual-target Python/native state codec.
* :mod:`distributed.native_bridge` -- concrete AST layout <-> ``NativeValue``.
* :mod:`distributed.leases` -- distributed borrow checking with fencing.
* :mod:`distributed.connections` -- anchored sockets that survive migration.
* :mod:`distributed.tasks` -- checkpointable tasks with an await journal.
* :mod:`distributed.reactor` -- a checkpointable timer/readiness reactor.
* :mod:`distributed.actor` -- actor state, snapshots, checkpoints.
* :mod:`distributed.transport` -- in-memory and TCP transports.
* :mod:`distributed.cluster` -- membership, migration, and rollback.

Nothing here is imported by the normal ``polyglot`` CLI path.  The migration
model is cooperative replay: tasks move as journals, sockets stay anchored, and
native values move as concrete layouts.  See ``docs/distributed.md``.
"""

from __future__ import annotations

from distributed.actor import (
    ACTIVE,
    FAILED,
    MIGRATED,
    MIGRATING,
    Actor,
    ActorSnapshot,
    CheckpointRecord,
    CheckpointStore,
    actor_handler,
    register_actor_handler,
)
from distributed.cluster import COORDINATOR, Cluster, NodeRuntime
from distributed.connections import (
    AnchorChannel,
    AnchorRegistry,
    ChannelHandle,
    ConnectionAnchor,
    SocketStream,
)
from distributed.diagnostics import (
    DiagnosticLog,
    DiagnosticRecord,
    DistributedError,
    migration_aborted,
    node_lost,
)
from distributed.leases import (
    EXCLUSIVE,
    SHARED,
    LeaseClient,
    LeaseGuard,
    LeaseManager,
    LeaseToken,
)
from distributed.native_bridge import (
    NativeBridge,
    from_native_value,
    layouts_from_program,
    register_program_layouts,
    to_native_value,
)
from distributed.reactor import Reactor, TaskPump, Timer
from distributed.tasks import (
    AwaitOp,
    CheckpointableTask,
    CheckpointOp,
    LocalEnvironment,
    RecvOp,
    SendOp,
    SleepOp,
    TaskEnvironment,
    TaskScheduler,
    handler,
    register_handler,
)
from distributed.transmute import (
    NativeLayout,
    NativeValue,
    PolyglotVariant,
    copy_state,
    decode_state,
    digest_bytes,
    encode_checkpoint,
    encode_state,
    native_layout,
    register_native_layout,
    register_type,
)
from distributed.transport import (
    Envelope,
    InMemoryNetwork,
    TCPTransport,
    serve,
)
from distributed.wire import (
    FORMAT_VERSION,
    Frame,
    Reader,
    WireError,
    Writer,
    decode_frame,
    encode_frame,
)

__version__ = "0.1.0"

__all__ = [
    "ACTIVE",
    "COORDINATOR",
    "EXCLUSIVE",
    "FAILED",
    "FORMAT_VERSION",
    "MIGRATED",
    "MIGRATING",
    "SHARED",
    "Actor",
    "ActorSnapshot",
    "AnchorChannel",
    "AnchorRegistry",
    "AwaitOp",
    "ChannelHandle",
    "CheckpointOp",
    "CheckpointRecord",
    "CheckpointStore",
    "CheckpointableTask",
    "Cluster",
    "ConnectionAnchor",
    "DiagnosticLog",
    "DiagnosticRecord",
    "DistributedError",
    "Envelope",
    "Frame",
    "InMemoryNetwork",
    "LeaseClient",
    "LeaseGuard",
    "LeaseManager",
    "LeaseToken",
    "LocalEnvironment",
    "NativeBridge",
    "NativeLayout",
    "NativeValue",
    "NodeRuntime",
    "PolyglotVariant",
    "Reactor",
    "Reader",
    "RecvOp",
    "SendOp",
    "SleepOp",
    "SocketStream",
    "TCPTransport",
    "TaskEnvironment",
    "TaskPump",
    "TaskScheduler",
    "Timer",
    "WireError",
    "Writer",
    "actor_handler",
    "copy_state",
    "decode_frame",
    "decode_state",
    "digest_bytes",
    "encode_checkpoint",
    "encode_frame",
    "encode_state",
    "from_native_value",
    "handler",
    "layouts_from_program",
    "migration_aborted",
    "native_layout",
    "node_lost",
    "register_actor_handler",
    "register_handler",
    "register_native_layout",
    "register_program_layouts",
    "register_type",
    "serve",
    "to_native_value",
]
