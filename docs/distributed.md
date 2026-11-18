# Distributed actors and live migration

`distributed/` is an **opt-in experimental runtime** for moving a running
Polyglot actor between hosts without dropping its state, its in-flight awaits,
or its anchored network connection. It is not imported by the normal
`polyglot` CLI path and adds no runtime dependencies.

```python
from distributed import Cluster, CheckpointableTask, handler

cluster = Cluster()
cluster.add_node("edge-a")
cluster.add_node("edge-b")

cluster.spawn("cart", handler="counter", state={"count": 0}, node="edge-a")
cluster.invoke("cart", (5).to_bytes(2, "big", signed=True))
cluster.migrate("cart", "edge-a", "edge-b")
assert cluster.state("cart")["state"]["count"] == 5
```

Run the checks with:

```sh
python -m pytest tests/test_distributed.py -q
```

## What actually moves

| Concern | Representation on the wire | Rebuilt on the target by |
| --- | --- | --- |
| Python dataclasses, dicts, lists, enums | Self-describing tagged values (`distributed/transmute.py`) | Type registry lookup |
| Native Rust/PyO3 struct or enum | `NativeLayout` + `NativeValue` (field values, variant/payload) | Public constructor in the target extension module |
| Active async work | Handler name, arguments, completed-await journal, pending operation | Deterministic replay from the top |
| Network connection | `ChannelHandle` (anchor id, anchor node, cursors) | Re-attaching to the anchored socket |
| Mutual exclusion | `LeaseToken` (owner, epoch, fencing token, expiry) | Lease authority validation |
| Pending timers | Remaining duration + label (`distributed/reactor.py`) | Re-armed on the target clock |

The codec is versioned and CRC-checked (`distributed/wire.py`), bounds every
length and nesting level, preserves shared references and cycles, and sorts
struct fields and set elements so logically equal states produce identical
bytes.

## Migration protocol

```
source                          coordinator                      target
  |                                  |                              |
  |-- quiesce + snapshot ----------->|                              |
  |                                  |-- prepare (decode/validate) ->|
  |                                  |<-- digest + handler check ----|
  |                                  |-- lease transfer, fence++ --->|
  |                                  |<-- commit installed actor ----|
  |-- tombstone + mailbox forward -->|                              |
```

1. **Quiesce.** The source stops advancing the actor and seals an
   `ActorSnapshot`: state, task journals, channel handles, mailbox, lease key.
2. **Prepare.** The target decodes the snapshot, verifies its digest, checks
   that the actor handler, task handlers, types, and native layouts exist, and
   decodes every task. Nothing is published yet.
3. **Lease transfer.** The authority moves the actor's exclusive lease to the
   target and increments its per-key fencing token. Any write still holding
   the old token is rejected with `E008` / `StaleFenceError`.
4. **Commit.** The target publishes the actor and acknowledges. The source
   keeps a tombstone and forwards later messages to the new owner.

If any step fails, the source restores the pre-migration snapshot, re-acquires
its lease, aborts the target, records an `E008` / `MigrationAbortedError`
diagnostic, and re-raises the structured error. Nothing is half-migrated.

## Tasks: replay, not a copied C stack

CPython does not expose a portable suspended-frame image, and a PyO3 future's
Rust state machine is even less portable. The runtime does not pretend
otherwise. A checkpointable handler performs every suspension through the
context:

```python
from distributed import CheckpointableTask, handler


@handler("drain")
def drain(ctx, need, channel):
    received = []
    while len(received) < need:
        received.extend(ctx.recv(channel))  # a journalled await
    return received


task = CheckpointableTask("drain-1", "drain", args=(2, channel_handle))
```

Each completed await and its result become a `JournalEntry`. On the target the
handler runs again from the top; replayed awaits return their journalled
results without touching the network, and the first operation past the journal
runs for real. The contract:

* Handlers must be deterministic up to each await point.
* Side effects before an await repeat during replay. Put them after the await,
  or make them idempotent.
* Do not remove or reorder awaits after a checkpoint; the runtime raises
  `E008` / `TaskReplayError` instead of diverging silently.
* Locals recorded with `ctx.local(...)` are observational only. They appear in
  diagnostics; control flow resumes from the journal.

This is the durable-execution model. It is transparent to the handler author,
not to physics.

## Connections: anchors, not movable sockets

A kernel socket is bound to an address on one host. Moving it would require
kernel cooperation that user space does not have. Instead:

* `ConnectionAnchor` stays on a stable node, owns the socket and sequenced
  buffers, and is reachable over the control plane.
* `ChannelHandle` moves with the actor and names the anchor, its node, and the
  inbound/outbound cursors.
* Inbound messages are numbered and read without consuming; `ack(cursor)`
  discards them. A reconnect is a pure replay.
* Outbound writes are numbered and idempotent. Replaying a sequence returns the
  original acknowledgement; replaying different bytes is an error.
* Ownership changes require a strictly higher fencing token.

If the **anchor node** dies, the socket itself is lost. The actor can be
recovered from a checkpoint, but the peer connection cannot be resurrected;
reconnect at the application level. This is a property of TCP, not a gap in the
serializer.

## Distributed borrow checking

Rust's borrow checker is static. Across a network, a partitioned node can still
try to write, so the runtime maps borrows to validated leases:

| Rust | Lease | Rule |
| --- | --- | --- |
| `&T` | `shared` | Many owners, no writes |
| `&mut T` | `exclusive` | One owner, writes allowed |

Every grant carries an epoch and a monotonically increasing per-key fencing
token. The lease **owner is the hosting node**, not the actor id, so a source
node that missed a commit cannot keep writing under the same actor identity:
after the transfer its `(owner, epoch, fencing)` triple no longer matches the
authority. `LeaseManager.validate(token, write=True)` rejects:

* a token from an older cluster epoch (`StaleFenceError`),
* a token superseded by a transfer (`StaleFenceError`),
* an expired lease (`LeaseExpiredError`),
* a write through a shared lease (`LeaseConflictError`).

The authority is the single decision point for its key space. This repository
ships an in-memory authority and its RPC surface, not a consensus protocol; a
production deployment must run the authority behind a replicated log or one
authority per key shard. Fencing only protects writes while the authority is
reachable and consistent.

## Node loss and rollback

Actors checkpoint after every processed message, write, and task step into a
bounded `CheckpointStore`. On failure detection:

1. The node is marked dead and removed from routing.
2. The lease authority bumps the cluster epoch, which revokes every lease
   granted in the previous term (fencing counters are preserved), so no
   surviving or partitioned node can validate an old token.
3. Each actor on the dead node is restored from its latest checkpoint onto a
   healthy node.
4. Surviving actors renew their leases under the new epoch.
5. An `E008` / `NodeLostError` record is written for each recovery.

The checkpoint store in this prototype is in-memory and replicated by the
control plane. A durable deployment should back it with an external store.

## Transports

* `InMemoryNetwork` is deterministic, supports partitions, node kills, message
  drains, and request/reply correlation. Cluster tests use it.
* `TCPTransport` and `TCPNodeServer` use length-prefixed, CRC-checked envelope
  frames over a real socket. They are a smoke-tested data path, not a hardened
  Internet-facing server: no TLS, authentication, or backpressure policy is
  included.

## Diagnostics

The subsystem reuses the existing `E001`–`E008` namespace through
`runtime.diagnostics.PillarError`:

| Code | Distributed use |
| --- | --- |
| `E001` | Layout/ABI mismatch in shipped state |
| `E003` | API misuse or illegal state transition |
| `E006` | Missing handler, actor, layout, type, checkpoint, or anchor |
| `E007` | Unsupported native value or non-portable descriptor |
| `E008` | Wire corruption, stale fence, expired lease, aborted migration, node loss, replay divergence |

`DiagnosticLog` records structured `DiagnosticRecord` values with code, note,
message, reason, help, node, context, and timestamp. `as_json()` and
`Cluster.diagnostics()` produce machine-readable output.

## Limitations

* Both nodes must run the same handlers, actor handlers, registered types, and
  concrete native layouts. There is no code shipping.
* Arbitrary Python generators/coroutines, `asyncio.Task` objects, and Rust
  futures are **not** copied. Use `CheckpointableTask`.
* Raw pointers, references, trait objects, and live file descriptors are not
  transported. Native state moves as `NativeValue`; sockets stay anchored.
* The in-memory transport and checkpoint store are for development and tests.
* Lease renewal is explicit; there is no background renewal loop yet.
* Migration is cooperative: the source node must be alive and responsive to
  quiesce and roll back.
