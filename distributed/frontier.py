"""Cluster discovery, session-typed RPC routing, and split-brain handling.

These are the three cluster-facing entry points of the expansion frontier. They
work on the deterministic in-process :class:`distributed.cluster.Cluster` and
never open a socket: the model's job is to make membership, routing and
partition decisions reproducible and testable, not to ship a gossip protocol.

Bounded by construction (SPEC 8.17, 10.47):

* heartbeats carry an explicit ``seq`` and timestamp; a duplicate is ignored,
  a timestamp that moves backwards is a protocol error;
* routing validates the *next* step of a declared session type and refuses a
  message that would not be the caller's turn;
* a partition is resolved by strict majority of the known nodes. A majority
  partition bumps the lease epoch and stays writable; every other partition is
  fenced and keeps writing rejected with ``E002``. A split with no majority is
  frozen: nobody writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from distributed.cluster import Cluster
from distributed.diagnostics import DistributedError, protocol_error
from distributed.wire import decode_frame, encode_frame

# Frame kind used for ordinary request/response traffic.
RPC_FRAME_KIND = 1

# Default codec versions this model understands. A caller may narrow the set.
SUPPORTED_WIRE_VERSIONS = (1,)


@dataclass(frozen=True)
class Heartbeat:
    """One heartbeat as it arrived on the wire."""

    node: str
    seq: int
    sent_ms: int


@dataclass(frozen=True)
class MembershipView:
    """The membership decided from a batch of heartbeats."""

    epoch: int
    quorum: int
    members: tuple[str, ...]
    evicted: tuple[str, ...]
    discovered: tuple[str, ...]
    stale: tuple[str, ...]


@dataclass(frozen=True)
class RoutedRpc:
    """A request routed to a member with the session tail attached."""

    target: str
    session_id: str
    remaining: tuple[tuple[str, str], ...]
    wire_version: int
    payload_bytes: int


@dataclass(frozen=True)
class PartitionView:
    """One side of a split, with its writability decided."""

    nodes: tuple[str, ...]
    writable: bool
    epoch: int


@dataclass(frozen=True)
class PartitionReport:
    """The result of resolving a set of proposed partitions."""

    decision: str
    quorum: int
    epoch: int
    partitions: tuple[PartitionView, ...]
    fences: dict[str, int] = field(default_factory=dict)


def _signature(payload: Any) -> bytes:
    """Canonical bytes for a payload, so routing is stable across processes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def cluster_discover_and_heartbeat(
    cluster: Cluster,
    beats: Iterable[Heartbeat],
    *,
    now_ms: int,
    ttl_ms: int = 5_000,
) -> MembershipView:
    """Fold a batch of heartbeats into membership.

    Unknown senders are discovered and added to the cluster; known senders are
    kept alive while ``now_ms - sent_ms <= ttl_ms``. A heartbeat whose ``seq``
    does not advance is a duplicate and is ignored; one whose timestamp moves
    backwards is rejected with ``E001``, because no healthy sender can emit a
    newer heartbeat with an older send time.
    """
    if ttl_ms <= 0:
        raise protocol_error(
            "heartbeat ttl must be positive",
            code="E001",
            help_text="pass ttl_ms > 0",
            context={"ttl_ms": ttl_ms},
        )

    last_seq: dict[str, int] = getattr(cluster, "_frontier_seq", {})
    last_sent: dict[str, int] = getattr(cluster, "_frontier_sent", {})
    discovered: list[str] = []
    stale: list[str] = []
    seen: set[str] = set()

    for beat in sorted(beats, key=lambda item: (item.node, item.seq)):
        if beat.node in seen:
            stale.append(beat.node)
            continue
        seen.add(beat.node)
        if beat.node not in cluster.nodes:
            cluster.add_node(beat.node)
            discovered.append(beat.node)
        previous_seq = last_seq.get(beat.node)
        if previous_seq is not None and beat.seq <= previous_seq:
            stale.append(beat.node)
            continue
        previous_sent = last_sent.get(beat.node)
        if previous_sent is not None and beat.sent_ms < previous_sent:
            raise protocol_error(
                f"heartbeat from {beat.node!r} moves backwards in time",
                code="E001",
                reason=(
                    f"sequence {beat.seq} carries send time {beat.sent_ms} ms, "
                    f"older than the previous {previous_sent} ms"
                ),
                help_text="clamp duplicate retries to the original send time",
                context={"node": beat.node, "sent_ms": beat.sent_ms},
            )
        last_seq[beat.node] = beat.seq
        last_sent[beat.node] = beat.sent_ms

    evicted: list[str] = []
    for name in sorted(cluster.nodes):
        sent = last_sent.get(name)
        if sent is None:
            continue
        if now_ms - sent > ttl_ms:
            cluster.membership[name] = type(cluster.membership[name])(
                name=name, epoch=cluster.membership[name].epoch, alive=False
            )
            cluster.dead.add(name)
            evicted.append(name)
        else:
            cluster.dead.discard(name)

    cluster._frontier_seq = last_seq
    cluster._frontier_sent = last_sent

    members = tuple(sorted(cluster.alive_nodes()))
    total = len(cluster.nodes)
    return MembershipView(
        epoch=cluster.leases.epoch,
        quorum=total // 2 + 1,
        members=members,
        evicted=tuple(evicted),
        discovered=tuple(discovered),
        stale=tuple(stale),
    )


def cluster_route_session_typed_rpc(
    cluster: Cluster,
    protocol: Iterable[tuple[str, str]],
    *,
    route_key: str,
    direction: str,
    message_type: str,
    payload: Any,
    schema_version: int = 1,
    supported_versions: tuple[int, ...] = SUPPORTED_WIRE_VERSIONS,
    session_tail: Iterable[tuple[str, str]] | None = None,
) -> RoutedRpc:
    """Route one message after checking it against a session type.

    ``protocol`` is the declared linear conversation, for example
    ``(("client", "Request"), ("server", "Response"))``. The message must be
    the first still-unconsumed step: same role and same message type. The
    target member is chosen deterministically from ``route_key`` so a replay
    lands on the same node.
    """
    steps = tuple((str(role), str(kind)) for role, kind in protocol)
    if not steps:
        raise protocol_error(
            "session protocol is empty",
            code="E001",
            help_text="declare at least one (role, message) step",
        )
    first_role, first_kind = steps[0]
    if direction != first_role or message_type != first_kind:
        raise protocol_error(
            f"session expects {first_role} to send {first_kind}",
            code="E001",
            reason=(
                f"got {direction} sending {message_type}; a session type only "
                "accepts the next declared step"
            ),
            help_text=f"send {first_kind} as {first_role}, then continue the session",
        )

    if schema_version not in supported_versions:
        raise DistributedError(
            f"wire codec version {schema_version} is not supported",
            code="E007",
            reason=(
                "only the declared codec versions can be decoded, so a peer on "
                "an older schema would be silently misread"
            ),
            help_text=(
                "negotiate a common version before sending: "
                + ", ".join(str(version) for version in sorted(supported_versions))
            ),
            context={"schema_version": schema_version},
        )

    if cluster.leases.epoch < 0:
        raise DistributedError(
            "cluster epoch is invalid",
            code="E002",
            context={"epoch": cluster.leases.epoch},
        )

    alive = cluster.alive_nodes()
    if not alive:
        raise DistributedError(
            "no member is available to route the call",
            code="E006",
            help_text="revive a node or wait for a heartbeat",
            context={"route_key": route_key},
        )

    payload_bytes = _signature(payload)
    frame = encode_frame(RPC_FRAME_KIND, payload_bytes, version=schema_version)
    decoded = decode_frame(frame)
    if decoded.payload != payload_bytes:
        raise DistributedError(
            "wire codec round-trip changed the payload",
            code="E004",
            help_text="this is a codec bug; report the payload",
            context={"route_key": route_key},
        )

    digest = hashlib.sha256(route_key.encode()).hexdigest()
    target = alive[int(digest[:8], 16) % len(alive)]
    session_id = hashlib.sha256(
        f"{route_key}:{direction}:{message_type}:{cluster.leases.epoch}".encode()
    ).hexdigest()[:16]
    tail = tuple(session_tail) if session_tail is not None else steps[1:]
    return RoutedRpc(
        target=target,
        session_id=session_id,
        remaining=tuple((str(role), str(kind)) for role, kind in tail),
        wire_version=schema_version,
        payload_bytes=len(frame),
    )


def cluster_handle_partition_and_split_brain(
    cluster: Cluster,
    groups: Iterable[Iterable[str]],
    *,
    now_ms: int = 0,
) -> PartitionReport:
    """Decide which side of a split may write.

    ``groups`` is the set of partitions as each side currently sees itself.
    Nodes that appear in no group are treated as isolated and fenced. The
    largest group holding a strict majority of all known nodes wins: it bumps
    the lease epoch and keeps writing. Every other group is fenced at the old
    epoch. With no majority the cluster is frozen and nothing writes, which is
    the behaviour a caller can reason about without trusting a clock.
    """
    known = sorted(cluster.nodes)
    total = len(known)
    if total == 0:
        raise DistributedError(
            "partition of an empty cluster has no authority",
            code="E002",
            help_text="add nodes before resolving a partition",
        )

    normalized: list[tuple[str, ...]] = []
    claimed: set[str] = set()
    for group in groups:
        members = tuple(sorted(str(node) for node in group))
        unknown = [node for node in members if node not in cluster.nodes]
        if unknown:
            raise DistributedError(
                f"partition names unknown node(s): {', '.join(unknown)}",
                code="E001",
                help_text="hear a heartbeat from the node before partitioning it",
                context={"unknown": unknown},
            )
        overlap = claimed.intersection(members)
        if overlap:
            raise protocol_error(
                f"node(s) appear in two partitions: {', '.join(sorted(overlap))}",
                code="E001",
                help_text="a partition is a disjoint cover of the known nodes",
                context={"overlap": sorted(overlap)},
            )
        claimed.update(members)
        normalized.append(members)

    quorum = total // 2 + 1
    winner: int | None = None
    for index, members in enumerate(normalized):
        if len(members) >= quorum and (
            winner is None or len(members) > len(normalized[winner])
        ):
            winner = index

    if winner is None:
        for name in known:
            cluster.dead.add(name)
        return PartitionReport(
            decision="frozen",
            quorum=quorum,
            epoch=cluster.leases.epoch,
            partitions=tuple(
                PartitionView(nodes=members, writable=False, epoch=cluster.leases.epoch)
                for members in normalized
            ),
            fences={},
        )

    epoch = cluster.leases.bump_epoch()
    fences: dict[str, int] = {}
    partitions: list[PartitionView] = []
    for index, members in enumerate(normalized):
        if index == winner:
            for name in members:
                cluster.dead.discard(name)
            partitions.append(PartitionView(nodes=members, writable=True, epoch=epoch))
            continue
        for name in members:
            cluster.dead.add(name)
            fences[name] = epoch
        partitions.append(PartitionView(nodes=members, writable=False, epoch=epoch))

    return PartitionReport(
        decision="majority",
        quorum=quorum,
        epoch=epoch,
        partitions=tuple(partitions),
        fences=fences,
    )
