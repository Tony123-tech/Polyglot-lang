"""Tests for the experimental distributed runtime."""

from __future__ import annotations

import json
import socket
import struct
import time
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest

from compiler.analysis.monomorph import monomorphize
from compiler.core.lexer import lex
from compiler.core.parser import parse
from distributed import (
    Actor,
    AnchorChannel,
    AnchorRegistry,
    CheckpointableTask,
    Cluster,
    DistributedError,
    Envelope,
    InMemoryNetwork,
    LeaseClient,
    LeaseManager,
    LocalEnvironment,
    NativeValue,
    Reactor,
    SocketStream,
    TaskScheduler,
    TCPTransport,
    decode_state,
    encode_state,
    handler,
    register_native_layout,
    serve,
)
from distributed.diagnostics import (
    LEASE_CONFLICT,
    LEASE_EXPIRED,
    MIGRATION_ABORTED,
    NODE_LOST,
    NODE_UNAVAILABLE,
    STALE_FENCE,
    TASK_REPLAY,
)
from distributed.native_bridge import (
    from_native_value,
    register_program_layouts,
    to_native_value,
)
from distributed.tasks import (
    AwaitOp,
    JournalEntry,
    OperationPending,
    SleepOp,
)
from distributed.transmute import NativeLayout
from distributed.wire import (
    KIND_STATE,
    Reader,
    WireError,
    Writer,
    decode_frame,
    encode_frame,
    peek_kind,
)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Point:
    x: int
    y: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class TreeNode:
    name: str
    children: list[TreeNode] = field(default_factory=list)


class Color(Enum):
    RED = 1
    BLUE = 2


# ---------------------------------------------------------------------------
# Wire
# ---------------------------------------------------------------------------


def test_package_is_discovered_by_setuptools() -> None:
    root = Path(__file__).resolve().parent.parent
    with (root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "distributed*" in include


def test_frame_round_trip_and_crc() -> None:
    frame = encode_frame(KIND_STATE, b"hello")
    decoded = decode_frame(frame)
    assert decoded.kind == KIND_STATE
    assert decoded.payload == b"hello"
    assert peek_kind(frame) == KIND_STATE

    corrupt = bytearray(frame)
    corrupt[-1] ^= 0xFF
    with pytest.raises(WireError):
        decode_frame(bytes(corrupt))
    with pytest.raises(WireError):
        decode_frame(frame[:-1])
    with pytest.raises(WireError):
        decode_frame(b"XXXX" + frame[4:])


def test_varints_round_trip() -> None:
    values = [0, 1, 127, 128, 300, 2**31, 2**63 - 1]
    signed = [-1, -2, -127, -128, -(2**31), -(2**63)]
    writer = Writer()
    for value in values:
        writer.varint(value)
    for value in signed:
        writer.svarint(value)
    reader = Reader(writer.build())
    assert [reader.varint() for _ in values] == values
    assert [reader.svarint() for _ in signed] == signed
    reader.expect_end()


# ---------------------------------------------------------------------------
# Transmute / native bridge
# ---------------------------------------------------------------------------


def test_transmute_round_trips_graphs_and_cycles() -> None:
    point = Point(3, 4, ["a", "b"])
    assert decode_state(encode_state(point)) == point
    assert encode_state(point) == encode_state(Point(3, 4, ["a", "b"]))

    left = TreeNode("left")
    right = TreeNode("right")
    left.children.append(right)
    right.children.append(left)
    restored = decode_state(encode_state(left))
    assert restored.name == "left"
    assert restored.children[0].name == "right"
    assert restored.children[0].children[0] is restored

    shared = [1, 2, 3]
    pair = decode_state(encode_state([shared, shared]))
    assert pair[0] is pair[1]

    payload = {"color": Color.BLUE, "set": {3, 1, 2}, "bytes": b"x"}
    assert decode_state(encode_state(payload)) == payload


def test_transmute_rejects_unknown_types() -> None:
    class Local:
        pass

    with pytest.raises(DistributedError) as caught:
        encode_state(Local())
    assert caught.value.code == "E007"


def test_native_layout_bridge_round_trips_struct_and_enum() -> None:
    source = """
    struct Pair_i64 { left: i64, right: i64 }
    enum Choice { A, B(i64) }
    """
    program = monomorphize(parse(lex(source)))
    layouts = register_program_layouts(program)
    assert "Pair_i64" in layouts
    assert "Choice" in layouts

    value = to_native_value("Pair_i64", {"left": 1, "right": 2})
    assert value == NativeValue("Pair_i64", {"left": 1, "right": 2})
    assert decode_state(encode_state(value)) == value

    class Pair:
        def __init__(self, left: int, right: int) -> None:
            self.left = left
            self.right = right

    rebuilt = from_native_value({"Pair_i64": Pair}, value)
    assert (rebuilt.left, rebuilt.right) == (1, 2)

    class Choice:
        @staticmethod
        def B(payload: int) -> tuple[str, int]:
            return ("B", payload)

    variant = to_native_value("Choice", ("B", 9))
    assert from_native_value({"Choice": Choice}, variant) == ("B", 9)


def test_native_layout_missing_is_a_layout_error() -> None:
    with pytest.raises(DistributedError) as caught:
        encode_state(NativeValue("Never_registered_layout", {"value": 1}))
    assert caught.value.code == "E006"


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------


def test_lease_modes_fencing_and_expiry() -> None:
    clock = FakeClock()
    manager = LeaseManager(clock=clock, ttl=10)
    exclusive = manager.acquire("key", "a")
    assert exclusive.fencing == 1

    with pytest.raises(DistributedError) as conflict:
        manager.acquire("key", "b")
    assert conflict.value.code == "E008"
    assert conflict.value.diagnostic.note == LEASE_CONFLICT

    manager.release(exclusive)
    shared_a = manager.acquire("key", "a", "shared")
    shared_b = manager.acquire("key", "b", "shared")
    with pytest.raises(DistributedError) as shared_write:
        manager.validate(shared_a, write=True)
    assert shared_write.value.diagnostic.note == LEASE_CONFLICT

    manager.release(shared_b)
    new = manager.transfer("key", "a", "c")
    assert new.fencing > shared_a.fencing
    with pytest.raises(DistributedError) as stale:
        manager.validate(shared_a)
    assert stale.value.diagnostic.note == STALE_FENCE

    manager.bump_epoch()
    with pytest.raises(DistributedError) as old_epoch:
        manager.validate(new)
    assert old_epoch.value.diagnostic.note == STALE_FENCE

    clock.advance(1)
    short = manager.acquire("short", "a", ttl=2)
    clock.advance(3)
    with pytest.raises(DistributedError) as expired:
        manager.validate(short)
    assert expired.value.diagnostic.note == LEASE_EXPIRED


def test_lease_client_and_snapshot_restore() -> None:
    manager = LeaseManager(ttl=30)
    client = LeaseClient(manager.handle_request)
    token = client.acquire("key", "actor")
    assert client.holders("key")["actor"] == token
    assert client.fencing("key") == token.fencing
    client.release(token)

    token = manager.acquire("key", "actor")
    snapshot = manager.snapshot()
    restored = LeaseManager(ttl=30)
    restored.restore(snapshot)
    assert restored.validate(token) == token


# ---------------------------------------------------------------------------
# Anchored connections
# ---------------------------------------------------------------------------


def test_anchor_sequences_replays_and_fences() -> None:
    registry = AnchorRegistry("n1")
    anchor = registry.open(owner="actor", epoch=0)
    handle = anchor.handle()
    assert handle.anchor_node == "n1"
    assert encode_state(handle) != b""
    assert decode_state(encode_state(handle)) == handle

    assert anchor.deliver(b"a") == 0
    assert anchor.deliver(b"b") == 1
    assert anchor.deliver(b"c") == 2
    messages, cursor = anchor.recv(0, limit=2)
    assert [item.data for item in messages] == [b"a", b"b"]
    assert cursor == 2
    anchor.ack(2)
    assert anchor.inbound_cursor == 2
    messages, cursor = anchor.recv(None)
    assert [item.data for item in messages] == [b"c"]
    assert cursor == 3
    anchor.ack(3)

    assert anchor.send(0, b"x") is True
    assert anchor.send(0, b"x") is False
    assert anchor.send(1, b"y") is True
    with pytest.raises(DistributedError):
        anchor.send(3, b"gap")
    with pytest.raises(DistributedError):
        anchor.send(0, b"different")
    pending = anchor.pending_outbound()
    assert [(item.seq, item.data) for item in pending] == [
        (0, b"x"),
        (1, b"y"),
    ]
    anchor.complete(2)
    assert anchor.pending_outbound() == []

    anchor.bind("new-owner", epoch=1, fencing=2)
    with pytest.raises(DistributedError) as stale:
        anchor.send(2, b"late", owner="actor", epoch=0, fencing=1)
    assert stale.value.diagnostic.note == STALE_FENCE


def test_anchor_rpc_and_channel_cursor() -> None:
    registry = AnchorRegistry("n1")
    anchor = registry.open(owner="actor")
    channel = AnchorChannel(anchor.handle(), registry.handle_request)

    assert channel.send(b"out-1") == 0
    assert channel.send(b"out-2") == 1
    assert channel.pending_outbound() == [(0, b"out-1"), (1, b"out-2")]
    channel.complete(2)

    anchor.deliver(b"in-1")
    assert channel.recv() == [b"in-1"]
    channel.ack()
    assert channel.handle.inbound == 1
    assert registry.handle_request(
        "recv", {"anchor_id": anchor.anchor_id, "cursor": 1}
    ) == {"ok": True, "messages": [], "cursor": 1}


def test_socket_stream_bridges_real_bytes_to_anchor() -> None:
    left, right = socket.socketpair()
    registry = AnchorRegistry("n1")
    anchor = registry.open(owner="actor")
    stream = SocketStream(sock=left, anchor=anchor, registry=registry).start()
    try:
        right.sendall(struct.pack(">I", 3) + b"hey")
        deadline = time.monotonic() + 3
        while anchor.inbound_next < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        channel = AnchorChannel(anchor.handle(), registry.handle_request)
        assert channel.recv() == [b"hey"]

        channel.send(b"out")
        stream.flush()
        right.settimeout(2)
        header = right.recv(4)
        (length,) = struct.unpack(">I", header)
        assert right.recv(length) == b"out"
    finally:
        stream.close()
        right.close()


# ---------------------------------------------------------------------------
# Tasks and replay
# ---------------------------------------------------------------------------


@handler("distributed_test_add")
def _add(ctx, left, right):
    ctx.local("left", left)
    ctx.local("right", right)
    return left + right


@handler("distributed_test_sleep")
def _sleep(ctx, seconds):
    ctx.local("started", True)
    ctx.sleep(seconds)
    return "done"


@handler("distributed_test_two_awaits")
def _two_awaits(ctx):
    first = ctx.await_external("ready")
    second = ctx.await_external("ready")
    return first + second


@handler("distributed_test_bad_local")
def _bad_local(ctx):
    ctx.local("unserializable", object())
    return "ok"


@handler("distributed_test_mismatch")
def _mismatch(ctx):
    return ctx.await_external("actual")


def test_task_advances_checkpoints_and_replays() -> None:
    clock = FakeClock()
    environment = LocalEnvironment(clock=clock)
    task = CheckpointableTask("add", "distributed_test_add", args=(2, 3))
    assert task.advance(environment).result == 5
    assert task.status == "done"
    assert task.locals == {"left": 2, "right": 3}

    sleeping = CheckpointableTask("sleep", "distributed_test_sleep", args=(2,))
    assert sleeping.advance(environment).status == "pending"
    assert isinstance(sleeping.pending, SleepOp)
    clock.advance(2.5)
    assert sleeping.advance(environment).result == "done"
    assert len(sleeping.journal) == 1

    restored = CheckpointableTask.decode(sleeping.encode())
    assert restored.result == "done"
    assert restored.journal == sleeping.journal

    ready = [(True, 1), (False, None)]
    environment.register("ready", lambda payload: ready.pop(0))
    replayed = CheckpointableTask("replay", "distributed_test_two_awaits")
    assert replayed.advance(environment).status == "pending"
    assert len(replayed.journal) == 1
    encoded = replayed.encode()
    clone = CheckpointableTask.decode(encoded)
    assert clone.pending is not None
    environment.register("ready", lambda payload: (True, 4))
    assert clone.advance(environment).result == 5
    assert len(clone.journal) == 2


def test_task_records_unsafe_locals_as_repr() -> None:
    task = CheckpointableTask("locals", "distributed_test_bad_local")
    task.advance(LocalEnvironment())
    assert task.status == "done"
    assert isinstance(task.locals["unserializable"], str)


def test_task_replay_violation_is_structured() -> None:
    task = CheckpointableTask(
        "bad",
        "distributed_test_mismatch",
        journal=[JournalEntry(0, AwaitOp("expected"), 1)],
    )
    with pytest.raises(DistributedError) as caught:
        task.advance(LocalEnvironment())
    assert caught.value.code == "E008"
    assert caught.value.diagnostic.note == TASK_REPLAY


def test_task_scheduler_and_pending_operation() -> None:
    task = CheckpointableTask("scheduled", "distributed_test_add", args=(1, 2))
    scheduler = TaskScheduler(LocalEnvironment())
    scheduler.submit(task)
    scheduler.run_until_idle()
    assert task.result == 3
    assert not scheduler.pending()

    with pytest.raises(OperationPending):
        LocalEnvironment().perform(SleepOp(1), task_id="x", index=0)


# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------


def test_reactor_timers_snapshot_and_restore() -> None:
    clock = FakeClock()
    reactor = Reactor(clock=clock)
    fired: list[str] = []
    reactor.add_timer(1.0, lambda: fired.append("first"), label="first")
    reactor.poll(0)
    assert fired == []
    clock.advance(1.1)
    reactor.poll(0)
    assert fired == ["first"]

    reactor.add_timer(5.0, lambda: fired.append("second"), label="second")
    snapshot = reactor.snapshot()
    restored = Reactor(clock=clock)
    restored.restore(snapshot, callbacks={"second": lambda: fired.append("second")})
    clock.advance(5.1)
    restored.poll(0)
    assert fired == ["first", "second"]
    reactor.close()
    restored.close()


def test_reactor_rejects_raw_descriptor_restore() -> None:
    reactor = Reactor()
    fired: list[str] = []
    with pytest.raises(DistributedError) as caught:
        reactor.restore(
            {
                "timers": [{"label": "later", "remaining": 0.0}],
                "readers": [{"fd": 1, "label": "x"}],
            },
            callbacks={"later": lambda: fired.append("later")},
        )
    assert caught.value.code == "E007"
    # The rejected checkpoint must not leave the timer armed.
    reactor.poll(0)
    assert fired == []
    assert reactor.snapshot()["timers"] == []
    reactor.close()


# ---------------------------------------------------------------------------
# Cluster, migration, rollback, failure recovery
# ---------------------------------------------------------------------------


@handler("distributed_test_collect")
def _collect(ctx, need, channel):
    got: list[bytes] = []
    ctx.local("need", need)
    while len(got) < need:
        got.extend(ctx.recv(channel, limit=8))
    return [bytes(item) for item in got]


def _cluster() -> Cluster:
    cluster = Cluster()
    cluster.add_node("n1")
    cluster.add_node("n2")
    return cluster


def test_cluster_spawn_invoke_and_migrate() -> None:
    cluster = _cluster()
    actor = cluster.spawn("counter", state={"count": 0}, node="n1")
    assert cluster.invoke("counter", (5).to_bytes(2, "big", signed=True)) == 5
    assert cluster.state("counter")["state"]["count"] == 5

    old_token = actor.lease
    migrated = cluster.migrate("counter", "n1", "n2")
    assert cluster.owner_of("counter") == "n2"
    assert migrated.state["count"] == 5
    assert migrated.lease is not None
    assert old_token is not None
    # Lease ownership is node-scoped so a source node that missed the commit
    # cannot keep writing under the same actor identity.
    assert old_token.owner == "n1"
    assert migrated.lease.owner == "n2"
    assert migrated.lease.fencing > old_token.fencing
    with pytest.raises(DistributedError) as stale:
        cluster.leases.validate(old_token)
    assert stale.value.diagnostic.note == STALE_FENCE

    # A source-side object that survived an ambiguous commit cannot write
    # under its old identity: the lease is owned by the destination node.
    with pytest.raises(DistributedError) as split_brain:
        actor.write("count", 99, authority=cluster.leases)
    assert split_brain.value.diagnostic.note == STALE_FENCE

    assert cluster.invoke("counter", (2).to_bytes(2, "big", signed=True)) == 7
    assert cluster.state("counter")["state"]["count"] == 7
    assert "counter" in cluster.nodes["n1"].tombstones
    assert cluster.nodes["n1"].get_actor("counter") is None


def test_cluster_migration_forwards_late_mailbox_messages() -> None:
    cluster = _cluster()
    cluster.spawn("counter", state={"count": 0}, node="n1")
    cluster.migrate("counter", "n1", "n2")

    cluster.notify(
        cluster.coordinator,
        "n1",
        "deliver",
        {"actor_id": "counter", "data": (3).to_bytes(2, "big", signed=True)},
    )
    cluster.pump("n2")
    assert cluster.invoke("counter", b"") == 4


def test_cluster_migrates_native_state() -> None:
    register_native_layout(
        NativeLayout("Pair_test", fields=(("left", "i64"), ("right", "i64")))
    )
    cluster = _cluster()
    pair = NativeValue("Pair_test", {"left": 11, "right": 22})
    cluster.spawn("native", state={"pair": pair}, node="n1")
    installed = cluster.migrate("native", "n1", "n2")
    assert installed.state["pair"] == pair
    assert isinstance(installed.state["pair"], NativeValue)


def test_cluster_migrates_task_mid_await_with_anchored_channel() -> None:
    cluster = _cluster()
    anchor = cluster.node("n1").anchors.open(owner="client")
    handle = anchor.handle()
    task = CheckpointableTask("collect", "distributed_test_collect", args=(2, handle))
    cluster.spawn("collector", node="n1", state={}, tasks=[task])

    assert cluster.advance_tasks("collector")["tasks"]["collect"]["status"] == "pending"
    anchor.deliver(b"one")
    assert cluster.advance_tasks("collector")["tasks"]["collect"]["status"] == "pending"

    cluster.migrate("collector", "n1", "n2")
    assert cluster.owner_of("collector") == "n2"
    anchor.deliver(b"two")
    description = cluster.advance_tasks("collector")
    assert description["tasks"]["collect"]["status"] == "done"

    installed = cluster.require_actor("collector")
    assert installed.tasks["collect"].result == [b"one", b"two"]
    assert installed.tasks["collect"].locals["need"] == 2


def test_cluster_migration_rolls_back_on_target_rejection() -> None:
    cluster = _cluster()
    cluster.spawn("counter", state={"count": 0}, node="n1")
    cluster.invoke("counter", (4).to_bytes(2, "big", signed=True))
    # Force the target's prepare phase to fail because the id is already taken.
    cluster.node("n2").add_actor(Actor("counter", node="n2"))

    with pytest.raises(DistributedError) as caught:
        cluster.migrate("counter", "n1", "n2")
    assert caught.value.code == "E008"
    assert caught.value.diagnostic.note == MIGRATION_ABORTED
    assert cluster.owner_of("counter") == "n1"
    actor = cluster.require_actor("counter")
    assert actor.status == "active"
    assert actor.state["count"] == 4
    assert actor.lease is not None
    assert actor.lease.owner == "n1"
    assert cluster.leases.validate(actor.lease) == actor.lease
    assert cluster.state("counter")["status"] == "active"
    assert any(record["note"] == MIGRATION_ABORTED for record in cluster.diagnostics())


def test_cluster_node_loss_recovery_uses_checkpoint_and_new_epoch() -> None:
    cluster = _cluster()
    cluster.spawn("counter", state={"count": 0}, node="n1")
    cluster.invoke("counter", (6).to_bytes(2, "big", signed=True))
    old_token = cluster.require_actor("counter").lease
    old_epoch = cluster.leases.epoch

    cluster.kill("n1")
    restored = cluster.recover("n1", onto="n2")
    assert len(restored) == 1
    assert cluster.owner_of("counter") == "n2"
    assert cluster.state("counter")["state"]["count"] == 6
    assert cluster.leases.epoch == old_epoch + 1

    new_token = cluster.require_actor("counter").lease
    assert new_token is not None
    assert new_token.owner == "n2"
    assert new_token.epoch == cluster.leases.epoch
    assert old_token is not None
    with pytest.raises(DistributedError) as stale:
        cluster.leases.validate(old_token)
    assert stale.value.diagnostic.note == STALE_FENCE
    assert any(record["note"] == NODE_LOST for record in cluster.diagnostics())
    json.dumps(cluster.diagnostics())


def test_cluster_writes_are_lease_protected() -> None:
    cluster = _cluster()
    cluster.spawn("counter", state={"count": 0}, node="n1")
    cluster.write("counter", "label", "hello")
    assert cluster.require_actor("counter").state["label"] == "hello"

    actor = cluster.require_actor("counter")
    actor.lease = None
    with pytest.raises(DistributedError) as caught:
        cluster.write("counter", "label", "blocked")
    assert caught.value.code == "E008"


def test_in_memory_network_partition_and_revival() -> None:
    network = InMemoryNetwork()
    network.register("a")
    network.register("b")
    transport = network.transport("a")
    envelope = Envelope(src="a", dst="b", kind="ping", payload=b"x")
    assert transport.send(envelope) is True
    assert network.recv("b").payload == b"x"

    network.partition("a", "b")
    assert transport.send(envelope) is False
    assert network.dropped == 1
    network.heal("a", "b")
    assert transport.send(envelope) is True

    network.kill("b")
    with pytest.raises(DistributedError) as caught:
        transport.send(envelope)
    assert caught.value.diagnostic.note == NODE_UNAVAILABLE
    network.revive("b")
    assert transport.send(envelope) is True


# ---------------------------------------------------------------------------
# TCP smoke
# ---------------------------------------------------------------------------


def test_tcp_transport_round_trip() -> None:
    def echo(envelope: Envelope) -> Envelope:
        payload = decode_state(envelope.payload)
        return envelope.reply(
            encode_state({"echo": payload["value"], "dst": envelope.dst})
        )

    server = serve(echo, node="server")
    client = TCPTransport.connect(server.address, "client")
    try:
        envelope = Envelope(
            src="client",
            dst="server",
            kind="test",
            payload=encode_state({"value": 7}),
        )
        reply = client.call(envelope)
        payload = decode_state(reply.payload)
        assert payload == {"echo": 7, "dst": "server"}
    finally:
        client.close()
        server.stop()
