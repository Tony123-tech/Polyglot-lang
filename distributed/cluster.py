"""Cluster nodes, the migration protocol, and automatic rollback.

The migration sequence is deliberately explicit so each step can fail and be
undone:

1. **quiesce**   -- the source stops advancing the actor and seals a snapshot.
2. **prepare**   -- the target decodes and validates the snapshot, layouts, and
                    handlers without publishing the actor.
3. **lease**     -- the authority transfers the actor's exclusive lease and
                    bumps its fencing token.  Late source writes are rejected.
4. **commit**    -- the target publishes the actor and acknowledges.
5. **tombstone** -- the source forwards its remaining mailbox and remembers
                    the new owner so late messages are redirected.

Any failure between 1 and 4 restores the source from the pre-migration
snapshot, re-acquires its lease, aborts the target, and raises ``E008`` with a
``MigrationAbortedError`` note.  A lost node is recovered from the replicated
checkpoint store on a surviving node after a cluster epoch bump.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from distributed.actor import (
    ACTIVE,
    MIGRATED,
    MIGRATING,
    Actor,
    ActorSnapshot,
    CheckpointRecord,
    CheckpointStore,
    get_actor_handler,
    node_lost_error,
)
from distributed.connections import AnchorRegistry
from distributed.diagnostics import (
    DiagnosticLog,
    DistributedError,
    migration_aborted,
    node_unavailable,
    protocol_error,
)
from distributed.leases import EXCLUSIVE, LeaseManager, LeaseToken
from distributed.tasks import LocalEnvironment, TaskEnvironment
from distributed.transmute import decode_state, encode_state
from distributed.transport import (
    KIND_PING,
    KIND_REPLY,
    Envelope,
    InMemoryNetwork,
)

COORDINATOR = "__coordinator__"


def _sanitize(value: Any) -> Any:
    if isinstance(value, DistributedError):
        return value.as_dict()
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _error_from_payload(value: Any) -> DistributedError:
    if isinstance(value, DistributedError):
        return value
    if isinstance(value, dict):
        return DistributedError(
            str(value.get("message", "remote operation failed")),
            code=str(value.get("code", "E008")),
            reason=value.get("reason"),
            help_text=value.get("help"),
            note=value.get("note"),
            context=value.get("context") or {},
        )
    return DistributedError(
        f"remote operation failed: {value!r}",
        note="ProtocolError",
    )


class NodeRuntime:
    """One cluster member: actors, anchors, checkpoints, and RPC dispatch."""

    def __init__(
        self,
        node: str,
        network: InMemoryNetwork,
        *,
        lease_authority: LeaseManager | None = None,
        checkpoints: CheckpointStore | None = None,
        epoch: int = 0,
    ) -> None:
        self.node = node
        self.network = network
        self.transport = network.transport(node)
        self.anchors = AnchorRegistry(node)
        self.lease_authority = lease_authority
        self.checkpoints = checkpoints or CheckpointStore()
        self.actors: dict[str, Actor] = {}
        self.tombstones: dict[str, str] = {}
        self.pending_snapshots: dict[str, ActorSnapshot] = {}
        self.membership: dict[str, Any] = {}
        self.log = DiagnosticLog(node=node)
        self.epoch = int(epoch)
        self.last_seen = time.monotonic()
        self.handled = 0
        self.router: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
        self.environment: TaskEnvironment = LocalEnvironment(call=self.anchor_call)

    # -- actors -----------------------------------------------------------
    def add_actor(self, actor: Actor) -> Actor:
        if actor.actor_id in self.actors:
            raise protocol_error(
                f"actor {actor.actor_id!r} already exists on {self.node!r}",
                context={"actor_id": actor.actor_id, "node": self.node},
            )
        actor.node = self.node
        actor.status = ACTIVE
        self.actors[actor.actor_id] = actor
        self.tombstones.pop(actor.actor_id, None)
        return actor

    def get_actor(self, actor_id: str) -> Actor | None:
        return self.actors.get(actor_id)

    def require_actor(self, actor_id: str) -> Actor:
        actor = self.actors.get(actor_id)
        if actor is None:
            raise DistributedError(
                f"actor {actor_id!r} does not exist on node {self.node!r}",
                code="E006",
                note="MissingActorError",
                help_text="check the actor's current node",
                context={"actor_id": actor_id, "node": self.node},
            )
        return actor

    def drop_actor(self, actor_id: str) -> Actor | None:
        return self.actors.pop(actor_id, None)

    def tombstone(self, actor_id: str, target: str) -> None:
        self.tombstones[actor_id] = target

    def checkpoint_actor(self, actor_id: str) -> CheckpointRecord:
        actor = self.require_actor(actor_id)
        data = actor.checkpoint()
        return self.checkpoints.put(
            actor_id,
            data,
            node=self.node,
            generation=actor.generation,
        )

    def _checkpoint(self, actor: Actor) -> CheckpointRecord:
        data = actor.checkpoint()
        return self.checkpoints.put(
            actor.actor_id,
            data,
            node=self.node,
            generation=actor.generation,
        )

    def restore_actor(
        self,
        actor_id: str,
        data: bytes | None = None,
    ) -> Actor:
        actor = self.actors.get(actor_id)
        if data is None:
            record = self.checkpoints.latest(actor_id)
            if record is None:
                raise DistributedError(
                    f"no checkpoint for actor {actor_id!r}",
                    code="E006",
                    note="MissingCheckpointError",
                    context={"actor_id": actor_id, "node": self.node},
                )
            data = record.data
        if actor is None:
            snapshot = ActorSnapshot.decode(data)
            actor = Actor.from_snapshot(snapshot, node=self.node)
            self.actors[actor_id] = actor
        else:
            actor.restore(data)
        actor.node = self.node
        actor.status = ACTIVE
        if self.lease_authority is not None:
            # The lease owner is the hosting *node*, not the actor id: after a
            # migration or a partial commit the old host still has code that
            # believes it owns the actor, and only a node-scoped owner lets the
            # authority reject its writes as stale.
            token = self.lease_authority.acquire(actor.lease_key, self.node, EXCLUSIVE)
            actor.lease = token
            actor.epoch = token.epoch
        return actor

    # -- anchors ----------------------------------------------------------
    def anchor_call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.router is not None:
            return self.router(op, payload)
        return self.anchors.handle_request(op, payload)

    # -- message loop -----------------------------------------------------
    def pump(self, *, max_messages: int = 10_000) -> int:
        messages = self.transport.drain()
        processed = 0
        for envelope in messages:
            self.last_seen = time.monotonic()
            self._dispatch(envelope)
            processed += 1
            if processed >= max_messages:
                break
        self.handled += processed
        return processed

    def _decode(self, envelope: Envelope) -> dict[str, Any]:
        try:
            payload = decode_state(envelope.payload)
        except DistributedError as error:
            raise protocol_error(
                f"malformed {envelope.kind} payload from "
                f"{envelope.src!r}: {error.diagnostic.message}",
                reason=error.diagnostic.reason,
                context={
                    "src": envelope.src,
                    "dst": envelope.dst,
                    "kind": envelope.kind,
                },
            ) from error
        if not isinstance(payload, dict):
            raise protocol_error(
                f"{envelope.kind} payload must be a mapping, not "
                f"{type(payload).__name__}"
            )
        return payload

    def _reply(self, envelope: Envelope, payload: dict[str, Any]) -> None:
        data = encode_state(_sanitize(payload))
        self.transport.send(envelope.reply(data))

    def _dispatch(self, envelope: Envelope) -> None:
        try:
            if envelope.kind == KIND_REPLY:
                return
            if envelope.kind == KIND_PING:
                self._reply(
                    envelope,
                    {
                        "ok": True,
                        "node": self.node,
                        "epoch": self.epoch,
                        "actors": sorted(self.actors),
                    },
                )
                return
            if envelope.kind == "hello":
                self._reply(
                    envelope,
                    {
                        "ok": True,
                        "node": self.node,
                        "epoch": self.epoch,
                    },
                )
                return
            payload = self._decode(envelope)
            result = self._handle(envelope.kind, payload)
            if result is not None:
                self._reply(envelope, result)
        except DistributedError as error:
            self.log.record(error)
            try:
                self._reply(envelope, {"ok": False, "error": error.as_dict()})
            except DistributedError:
                pass
        except Exception as error:
            wrapped = protocol_error(
                f"unhandled error while processing {envelope.kind}",
                reason=f"{type(error).__name__}: {error}",
                context={"src": envelope.src, "node": self.node},
            )
            self.log.record(wrapped)
            try:
                self._reply(envelope, {"ok": False, "error": wrapped.as_dict()})
            except DistributedError:
                pass

    def _handle(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if kind == "lease":
            if self.lease_authority is None:
                raise protocol_error(f"node {self.node!r} has no lease authority")
            return _sanitize(
                self.lease_authority.handle_request(
                    payload["op"], payload.get("payload", {})
                )
            )

        if kind == "anchor.rpc":
            return _sanitize(
                self.anchors.handle_request(payload["op"], payload.get("payload", {}))
            )

        if kind in {"deliver", "forward"}:
            actor_id = payload["actor_id"]
            data = payload.get("data", b"")
            actor = self.actors.get(actor_id)
            if actor is None:
                target = self.tombstones.get(actor_id)
                if target is not None:
                    self.transport.send(
                        Envelope(
                            src=self.node,
                            dst=target,
                            kind="deliver",
                            payload=encode_state({"actor_id": actor_id, "data": data}),
                        )
                    )
                    return {"ok": True, "forwarded": target}
                raise DistributedError(
                    f"actor {actor_id!r} does not exist on node {self.node!r}",
                    code="E006",
                    note="MissingActorError",
                )
            actor.deliver(data)
            return {"ok": True, "queued": len(actor.mailbox)}

        if kind == "invoke":
            return self._invoke(payload)

        if kind == "migrate.prepare":
            return self._prepare(payload)

        if kind == "migrate.commit":
            return self._commit(payload)

        if kind == "migrate.abort":
            actor_id = payload["actor_id"]
            self.pending_snapshots.pop(actor_id, None)
            return {"ok": True, "aborted": actor_id}

        if kind == "migrate.rollback":
            actor_id = payload["actor_id"]
            data = payload.get("data")
            actor = self.restore_actor(actor_id, data)
            return {"ok": True, "actor": actor.describe()}

        if kind == "checkpoint":
            op = payload["op"]
            if op == "put":
                record = self.checkpoints.put(
                    payload["actor_id"],
                    payload["data"],
                    node=payload.get("node", self.node),
                    generation=int(payload.get("generation", 0)),
                )
                return {
                    "ok": True,
                    "digest": record.digest,
                    "generation": record.generation,
                }
            if op == "latest":
                record = self.checkpoints.latest(payload["actor_id"])
                if record is None:
                    return {"ok": True, "record": None}
                return {
                    "ok": True,
                    "record": {
                        "actor_id": record.actor_id,
                        "node": record.node,
                        "generation": record.generation,
                        "digest": record.digest,
                        "data": record.data,
                    },
                }
            raise protocol_error(f"unknown checkpoint op {op!r}")

        if kind == "membership":
            self.membership.update(payload)
            return {"ok": True, "membership": dict(self.membership)}

        raise protocol_error(
            f"node {self.node!r} does not understand envelope kind {kind!r}",
            context={"node": self.node, "kind": kind},
        )

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        actor_id = payload["actor"]
        actor = self.actors.get(actor_id)
        if actor is None:
            target = self.tombstones.get(actor_id)
            if target is not None:
                self.transport.send(
                    Envelope(
                        src=self.node,
                        dst=target,
                        kind="deliver",
                        payload=encode_state(
                            {
                                "actor_id": actor_id,
                                "data": payload.get("data", b""),
                            }
                        ),
                    )
                )
                return {"ok": True, "forwarded": target}
            raise DistributedError(
                f"actor {actor_id!r} does not exist on node {self.node!r}",
                code="E006",
                note="MissingActorError",
                help_text="check the actor's current node",
                context={"actor_id": actor_id, "node": self.node},
            )
        op = payload.get("op", "deliver")
        if op == "deliver":
            actor.deliver(payload.get("data", b""))
            return {"ok": True, "queued": len(actor.mailbox)}
        if op == "step":
            result = actor.drain(self.environment, limit=int(payload.get("limit", 100)))
            self._checkpoint(actor)
            return {
                "ok": True,
                "result": actor.last_result,
                "processed": result,
                "describe": actor.describe(),
            }
        if op == "advance_tasks":
            actor.advance_tasks(self.environment)
            self._checkpoint(actor)
            return {
                "ok": True,
                "describe": actor.describe(),
            }
        if op == "snapshot":
            data = actor.checkpoint()
            record = self.checkpoints.put(
                actor_id,
                data,
                node=self.node,
                generation=actor.generation,
            )
            return {
                "ok": True,
                "data": data,
                "digest": record.digest,
                "generation": record.generation,
            }
        if op == "write":
            if self.lease_authority is not None:
                actor.write(
                    payload["key"],
                    payload["value"],
                    authority=self.lease_authority,
                )
            else:
                actor.write(payload["key"], payload["value"])
            self._checkpoint(actor)
            return {"ok": True, "describe": actor.describe()}
        if op == "state":
            return {"ok": True, "describe": actor.describe()}
        raise protocol_error(f"unknown invoke op {op!r}")

    def _prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        actor_id = payload["actor_id"]
        if actor_id in self.actors:
            raise protocol_error(
                f"actor {actor_id!r} already exists on {self.node!r}",
                context={"actor_id": actor_id, "node": self.node},
            )
        if actor_id in self.pending_snapshots:
            raise protocol_error(
                f"actor {actor_id!r} already has a prepared migration",
                context={"actor_id": actor_id, "node": self.node},
            )
        snapshot = ActorSnapshot.decode(payload["data"])
        if snapshot.actor_id != actor_id:
            raise DistributedError(
                f"prepared snapshot holds {snapshot.actor_id!r}, not {actor_id!r}",
                note="StateMismatchError",
            )
        get_actor_handler(snapshot.handler)
        # Decode every task now so a missing handler or bad journal is caught
        # before the actor is published.
        Actor.from_snapshot(snapshot, node=self.node)
        self.pending_snapshots[actor_id] = snapshot
        return {
            "ok": True,
            "digest": snapshot.digest,
            "generation": snapshot.generation,
        }

    def _commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        actor_id = payload["actor_id"]
        snapshot = self.pending_snapshots.pop(actor_id, None)
        if snapshot is None:
            raise protocol_error(
                f"no prepared migration for actor {actor_id!r}",
                context={"actor_id": actor_id, "node": self.node},
            )
        token = payload.get("token")
        if isinstance(token, dict):
            token = LeaseToken(**token)
        if token is not None and self.lease_authority is not None:
            self.lease_authority.validate(token, write=True)
        actor = Actor.from_snapshot(snapshot, node=self.node)
        actor.lease = token
        actor.epoch = token.epoch if token is not None else self.epoch
        actor.migrated_from = payload.get("source")
        actor.status = ACTIVE
        self.actors[actor_id] = actor
        return {
            "ok": True,
            "generation": actor.generation,
            "describe": actor.describe(),
        }

    # -- diagnostics ------------------------------------------------------
    def records(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.log.records()]


@dataclass
class NodeInfo:
    name: str
    epoch: int
    alive: bool = True


class Cluster:
    """A deterministic cluster for tests, demos, and protocol development."""

    def __init__(
        self,
        *,
        epoch: int = 0,
        ttl: float = 30.0,
        history: int = 8,
    ) -> None:
        self.coordinator = COORDINATOR
        self.network = InMemoryNetwork()
        self.network.register(self.coordinator)
        self.leases = LeaseManager(authority="coordinator", epoch=epoch, ttl=ttl)
        self.checkpoints = CheckpointStore(history=history)
        self.nodes: dict[str, NodeRuntime] = {}
        self.membership: dict[str, NodeInfo] = {}
        self.dead: set[str] = set()
        self.log = DiagnosticLog(node="coordinator")
        self.migrations = 0
        self.failures = 0
        self._backlog: list[Envelope] = []

    # -- membership -------------------------------------------------------
    def add_node(self, name: str) -> NodeRuntime:
        if name in self.nodes:
            raise protocol_error(f"node {name!r} already exists")
        self.network.register(name)
        runtime = NodeRuntime(
            name,
            self.network,
            lease_authority=self.leases,
            checkpoints=self.checkpoints,
            epoch=self.leases.epoch,
        )
        runtime.router = lambda op, payload, _name=name: self._anchor_route(
            _name, op, payload
        )
        self.nodes[name] = runtime
        self.membership[name] = NodeInfo(name=name, epoch=self.leases.epoch)
        return runtime

    def node(self, name: str) -> NodeRuntime:
        runtime = self.nodes.get(name)
        if runtime is None:
            raise node_unavailable(
                f"node {name!r} is not part of this cluster",
                context={"node": name},
            )
        return runtime

    def alive_nodes(self) -> list[str]:
        return sorted(
            name
            for name in self.nodes
            if name not in self.dead and self.network.is_alive(name)
        )

    def _pick_node(self, *, exclude: str | None = None) -> str:
        for name in self.alive_nodes():
            if name != exclude:
                return name
        raise node_unavailable(
            "the cluster has no healthy node",
            help_text="add a node or revive the failed one",
        )

    def _anchor_route(
        self, node: str, op: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        target = payload.get("anchor_node") or node
        if target == node or target not in self.nodes:
            return self.nodes[node].anchors.handle_request(op, payload)
        reply = self.call(node, target, "anchor.rpc", {"op": op, "payload": payload})
        return reply

    # -- control-plane messaging -----------------------------------------
    def _drain(self, node: str) -> list[Envelope]:
        if node in self.nodes:
            return self.nodes[node].transport.drain()
        messages, self._backlog = self._backlog, []
        messages.extend(self.network.drain(node))
        return messages

    def _stash(self, node: str, messages: list[Envelope]) -> None:
        if not messages:
            return
        if node in self.nodes:
            self.nodes[node].transport.restash(messages)
        else:
            self._backlog[0:0] = messages

    def pump(self, node: str, *, rounds: int = 1) -> int:
        runtime = self.nodes.get(node)
        if runtime is None:
            return 0
        total = 0
        for _ in range(rounds):
            total += runtime.pump()
        return total

    def pump_all(self, *, rounds: int = 3) -> int:
        total = 0
        for _ in range(rounds):
            for name in list(self.nodes):
                if name not in self.dead:
                    total += self.nodes[name].pump()
        return total

    def call(
        self,
        src: str,
        dst: str,
        kind: str,
        payload: dict[str, Any],
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if src not in self.nodes and src != self.coordinator:
            raise node_unavailable(f"unknown call source {src!r}")
        if dst not in self.nodes:
            raise node_unavailable(f"unknown call destination {dst!r}")
        envelope = Envelope(
            src=src,
            dst=dst,
            kind=kind,
            payload=encode_state(payload),
        )
        self.network.transport(src).send(envelope)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(dst)
            messages = self._drain(src)
            keep: list[Envelope] = []
            for message in messages:
                if (
                    message.correlation_id == envelope.msg_id
                    and message.kind == KIND_REPLY
                ):
                    reply = decode_state(message.payload)
                    if not isinstance(reply, dict):
                        raise protocol_error(f"{kind} reply is not a mapping")
                    if not reply.get("ok", False):
                        raise _error_from_payload(reply.get("error"))
                    return reply
                keep.append(message)
            self._stash(src, keep)
            # The target pump is synchronous; a missing reply is a real
            # failure rather than a scheduling artifact.
            break
        raise node_unavailable(
            f"no reply from {dst!r} for {kind}",
            note="ProtocolError",
            context={"src": src, "dst": dst, "kind": kind},
        )

    def notify(
        self,
        src: str,
        dst: str,
        kind: str,
        payload: dict[str, Any],
        *,
        pump: bool = True,
    ) -> None:
        self.network.transport(src).send(
            Envelope(
                src=src,
                dst=dst,
                kind=kind,
                payload=encode_state(payload),
            )
        )
        if pump:
            self.pump(dst)

    # -- actors -----------------------------------------------------------
    def spawn(
        self,
        actor_id: str,
        *,
        handler: str = "counter",
        state: dict[str, Any] | None = None,
        node: str | None = None,
        tasks: list[Any] | None = None,
    ) -> Actor:
        if self.owner_of(actor_id) is not None:
            raise protocol_error(f"actor {actor_id!r} already exists")
        name = node or self._pick_node()
        runtime = self.node(name)
        actor = Actor(
            actor_id=actor_id,
            handler=handler,
            state=dict(state or {}),
            node=name,
        )
        for task in tasks or ():
            actor.add_task(task)
        token = self.leases.acquire(actor.lease_key, name, EXCLUSIVE)
        actor.lease = token
        actor.epoch = token.epoch
        runtime.add_actor(actor)
        data = actor.checkpoint()
        self.checkpoints.put(
            actor_id,
            data,
            node=name,
            generation=actor.generation,
        )
        return actor

    def owner_of(self, actor_id: str) -> str | None:
        for name in self.alive_nodes():
            if actor_id in self.nodes[name].actors:
                return name
        return None

    def locate(self, actor_id: str) -> tuple[str, Actor] | None:
        owner = self.owner_of(actor_id)
        if owner is None:
            return None
        return owner, self.nodes[owner].require_actor(actor_id)

    def require_actor(self, actor_id: str) -> Actor:
        found = self.locate(actor_id)
        if found is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
                help_text="spawn it first or recover it from a checkpoint",
                context={"actor_id": actor_id},
            )
        return found[1]

    def invoke(
        self,
        actor_id: str,
        payload: bytes,
        *,
        node: str | None = None,
    ) -> Any:
        owner = node or self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
                context={"actor_id": actor_id},
            )
        delivered = self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "deliver", "data": payload},
        )
        forwarded = delivered.get("forwarded")
        if forwarded:
            return self.invoke(actor_id, payload, node=forwarded)
        stepped = self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "step"},
        )
        return stepped.get("result")

    def deliver(
        self, actor_id: str, payload: bytes, *, node: str | None = None
    ) -> None:
        owner = node or self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
                context={"actor_id": actor_id},
            )
        self.call(
            self.coordinator,
            owner,
            "deliver",
            {"actor_id": actor_id, "data": bytes(payload)},
        )

    def state(self, actor_id: str) -> dict[str, Any]:
        owner = self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
            )
        reply = self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "state"},
        )
        return reply["describe"]

    def write(self, actor_id: str, key: str, value: Any) -> None:
        owner = self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
            )
        self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "write", "key": key, "value": value},
        )

    def checkpoint(self, actor_id: str) -> CheckpointRecord:
        owner = self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
            )
        reply = self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "snapshot"},
        )
        record = self.checkpoints.latest(actor_id)
        if record is None or record.data != reply["data"]:
            record = self.checkpoints.put(
                actor_id,
                reply["data"],
                node=owner,
                generation=int(reply["generation"]),
                digest=reply["digest"],
            )
        return record

    def advance_tasks(self, actor_id: str) -> dict[str, Any]:
        owner = self.owner_of(actor_id)
        if owner is None:
            raise DistributedError(
                f"actor {actor_id!r} is not hosted by any live node",
                code="E006",
                note="MissingActorError",
            )
        reply = self.call(
            self.coordinator,
            owner,
            "invoke",
            {"actor": actor_id, "op": "advance_tasks"},
        )
        return reply["describe"]

    # -- leases -----------------------------------------------------------
    def acquire_lease(self, actor_id: str, *, mode: str = EXCLUSIVE) -> LeaseToken:
        actor = self.require_actor(actor_id)
        token = self.leases.acquire(actor.lease_key, actor.node, mode)
        actor.lease = token
        actor.epoch = token.epoch
        return token

    def release_lease(self, actor_id: str) -> None:
        actor = self.require_actor(actor_id)
        if actor.lease is not None:
            self.leases.release(actor.lease)
            actor.lease = None

    def refresh_leases(self) -> None:
        for name in self.alive_nodes():
            for actor in self.nodes[name].actors.values():
                token = self.leases.acquire(actor.lease_key, name, EXCLUSIVE)
                actor.lease = token
                actor.epoch = token.epoch

    # -- migration --------------------------------------------------------
    def migrate(
        self,
        actor_id: str,
        source: str,
        target: str,
    ) -> Actor:
        if source == target:
            raise DistributedError(
                "source and target nodes are the same",
                code="E003",
                note="ActorProtocolError",
                context={"actor_id": actor_id, "node": source},
            )
        source_runtime = self.node(source)
        target_runtime = self.node(target)
        actor = source_runtime.require_actor(actor_id)
        if actor.status != ACTIVE:
            raise protocol_error(
                f"actor {actor_id!r} is {actor.status!r}, not active",
                context={"actor_id": actor_id, "status": actor.status},
            )

        before = actor.checkpoint()
        actor.status = MIGRATING
        token: LeaseToken | None = None
        try:
            snapshot = actor.export_snapshot()
            data = snapshot.encode()
            source_runtime.checkpoints.put(
                actor_id,
                data,
                node=source,
                generation=snapshot.generation,
                digest=snapshot.digest,
            )
            prepared = self.call(
                self.coordinator,
                target,
                "migrate.prepare",
                {
                    "actor_id": actor_id,
                    "data": data,
                    "source": source,
                    "generation": snapshot.generation,
                },
            )
            if prepared.get("digest") != snapshot.digest:
                raise DistributedError(
                    "target prepared a different snapshot digest",
                    note="StateMismatchError",
                    context={
                        "actor_id": actor_id,
                        "expected": snapshot.digest,
                        "actual": prepared.get("digest"),
                    },
                )
            token = self.leases.transfer(
                actor.lease_key,
                source,
                target,
                mode=EXCLUSIVE,
            )
            self.call(
                self.coordinator,
                target,
                "migrate.commit",
                {
                    "actor_id": actor_id,
                    "source": source,
                    "token": token,
                },
            )
            installed = target_runtime.get_actor(actor_id)
            if installed is None:
                raise protocol_error(
                    "target acknowledged commit but has no actor",
                    context={"actor_id": actor_id, "node": target},
                )
            already = len(snapshot.mailbox)
            for message in actor.mailbox[already:]:
                installed.deliver(message)
            source_runtime.tombstone(actor_id, target)
            actor.status = MIGRATED
            source_runtime.drop_actor(actor_id)
            self.checkpoints.put(
                actor_id,
                data,
                node=target,
                generation=snapshot.generation,
                digest=snapshot.digest,
            )
            self.migrations += 1
            return installed
        except DistributedError as error:
            actor.status = ACTIVE
            try:
                actor.restore(before)
            except DistributedError:
                pass
            try:
                # Hand the lease back to the source.  This works whether the
                # transfer already happened (target holds it) or the failure
                # came earlier (source still holds it); either way the grant
                # gets a newer fencing token.
                replacement = self.leases.transfer(
                    actor.lease_key, target, source, mode=EXCLUSIVE
                )
            except DistributedError:
                try:
                    replacement = self.leases.acquire(
                        actor.lease_key, source, EXCLUSIVE
                    )
                except DistributedError:
                    replacement = None
            if replacement is not None:
                actor.lease = replacement
                actor.epoch = replacement.epoch
            try:
                self.call(
                    self.coordinator,
                    target,
                    "migrate.abort",
                    {"actor_id": actor_id},
                )
            except DistributedError:
                pass
            diagnostic = migration_aborted(
                f"migration of actor {actor_id!r} from {source!r} to "
                f"{target!r} was aborted",
                reason=f"{error.code}: {error.diagnostic.message}",
                context={
                    "actor_id": actor_id,
                    "source": source,
                    "target": target,
                    "fencing": None if token is None else token.fencing,
                },
            )
            self.log.record(diagnostic)
            raise diagnostic from error

    # -- failure handling -------------------------------------------------
    def kill(self, node: str) -> None:
        if node not in self.nodes:
            raise node_unavailable(f"unknown node {node!r}")
        if node in self.dead:
            return
        self.dead.add(node)
        self.network.kill(node)
        if node in self.membership:
            self.membership[node].alive = False
        self.log.record(
            node_lost_error(node),
            context={"node": node, "actors": sorted(self.nodes[node].actors)},
        )

    def recover(
        self,
        node: str,
        *,
        onto: str | None = None,
    ) -> list[Actor]:
        runtime = self.node(node)
        target = onto or self._pick_node(exclude=node)
        target_runtime = self.node(target)
        if node not in self.dead:
            raise protocol_error(
                f"node {node!r} has not been declared dead",
                context={"node": node},
            )

        self.leases.bump_epoch()
        self.refresh_leases()
        restored: list[Actor] = []
        for actor_id in sorted(runtime.actors):
            record = self.checkpoints.latest(actor_id)
            if record is None:
                error = DistributedError(
                    f"actor {actor_id!r} has no checkpoint to recover",
                    code="E006",
                    note="MissingCheckpointError",
                    context={"actor_id": actor_id, "node": node},
                )
                self.log.record(error)
                continue
            if actor_id in target_runtime.actors:
                continue
            try:
                snapshot = ActorSnapshot.decode(record.data)
                actor = Actor.from_snapshot(snapshot, node=target)
                # The epoch bump revoked the dead node's leases; the new host
                # claims the key under the new term.
                token = self.leases.acquire(actor.lease_key, target, EXCLUSIVE)
                actor.lease = token
                actor.epoch = token.epoch
                actor.status = ACTIVE
                actor.migrated_from = node
                target_runtime.add_actor(actor)
                self.checkpoints.put(
                    actor_id,
                    actor.checkpoint(),
                    node=target,
                    generation=actor.generation,
                )
                restored.append(actor)
                self.log.record(
                    node_lost_error(node, actor_id=actor_id),
                    context={
                        "actor_id": actor_id,
                        "recovered_on": target,
                        "generation": actor.generation,
                    },
                )
            except DistributedError as error:
                self.log.record(
                    error,
                    context={"actor_id": actor_id, "node": node},
                )
        self.failures += 1
        runtime.actors.clear()
        return restored

    def detect_failures(self, *, timeout: float = 30.0) -> list[str]:
        now = time.monotonic()
        failed = [
            name
            for name in self.alive_nodes()
            if now - self.nodes[name].last_seen > timeout
        ]
        for name in failed:
            self.kill(name)
        for name in failed:
            self.recover(name)
        return failed

    def heartbeat(self, node: str) -> dict[str, Any]:
        reply = self.call(self.coordinator, node, KIND_PING, {})
        self.nodes[node].last_seen = time.monotonic()
        return reply

    # -- inspection -------------------------------------------------------
    def diagnostics(self) -> list[dict[str, Any]]:
        records = [record.as_dict() for record in self.log.records()]
        for name in self.alive_nodes():
            records.extend(self.nodes[name].records())
        return records

    def describe(self) -> dict[str, Any]:
        return {
            "epoch": self.leases.epoch,
            "nodes": {
                name: {
                    "alive": name not in self.dead,
                    "actors": sorted(self.nodes[name].actors),
                }
                for name in self.nodes
            },
            "migrations": self.migrations,
            "failures": self.failures,
            "leases": self.leases.snapshot()["stats"],
        }


__all__ = [
    "COORDINATOR",
    "Cluster",
    "NodeInfo",
    "NodeRuntime",
]
