"""Tests for cluster discovery, session-typed routing, and split-brain decisions."""

from __future__ import annotations

import pytest

from distributed.cluster import Cluster
from distributed.diagnostics import DistributedError
from distributed.frontier import (
    Heartbeat,
    cluster_discover_and_heartbeat,
    cluster_handle_partition_and_split_brain,
    cluster_route_session_typed_rpc,
)

PROTOCOL = (("client", "Request"), ("server", "Response"))


def _cluster(*nodes: str) -> Cluster:
    cluster = Cluster()
    for node in nodes:
        cluster.add_node(node)
    return cluster


def test_discovery_adds_unknown_sender() -> None:
    cluster = _cluster("a")
    view = cluster_discover_and_heartbeat(
        cluster, [Heartbeat("a", 1, 100), Heartbeat("b", 1, 120)], now_ms=150
    )
    assert view.discovered == ("b",)
    assert set(view.members) == {"a", "b"}
    assert view.quorum == 2


def test_duplicate_heartbeat_is_ignored() -> None:
    cluster = _cluster("a")
    beats = [Heartbeat("a", 2, 100), Heartbeat("a", 1, 90)]
    view = cluster_discover_and_heartbeat(cluster, beats, now_ms=150)
    assert view.stale == ("a",)
    assert "a" in view.members


def test_backwards_timestamp_is_a_protocol_error() -> None:
    cluster = _cluster("a")
    cluster_discover_and_heartbeat(cluster, [Heartbeat("a", 1, 200)], now_ms=250)
    with pytest.raises(DistributedError) as error:
        cluster_discover_and_heartbeat(cluster, [Heartbeat("a", 2, 100)], now_ms=250)
    assert error.value.code == "E001"
    assert "backwards" in str(error.value)


def test_silent_node_is_evicted_after_ttl() -> None:
    cluster = _cluster("a", "b")
    cluster_discover_and_heartbeat(
        cluster,
        [Heartbeat("a", 1, 100), Heartbeat("b", 1, 400)],
        now_ms=500,
        ttl_ms=300,
    )
    view = cluster_discover_and_heartbeat(
        cluster, [Heartbeat("b", 2, 480)], now_ms=600, ttl_ms=300
    )
    assert view.evicted == ("a",)
    assert view.members == ("b",)


def test_session_route_is_deterministic() -> None:
    cluster = _cluster("a", "b", "c")
    first = cluster_route_session_typed_rpc(
        cluster,
        PROTOCOL,
        route_key="acct-42",
        direction="client",
        message_type="Request",
        payload={"amount": 7},
    )
    second = cluster_route_session_typed_rpc(
        cluster,
        PROTOCOL,
        route_key="acct-42",
        direction="client",
        message_type="Request",
        payload={"amount": 7},
    )
    assert first.target == second.target
    assert first.session_id == second.session_id
    assert first.remaining == (("server", "Response"),)
    assert first.wire_version == 1


def test_route_rejects_out_of_turn_message() -> None:
    cluster = _cluster("a")
    with pytest.raises(DistributedError) as error:
        cluster_route_session_typed_rpc(
            cluster,
            PROTOCOL,
            route_key="acct-1",
            direction="server",
            message_type="Response",
            payload={},
        )
    assert error.value.code == "E001"
    assert "session expects" in str(error.value)


def test_route_rejects_unsupported_codec_version() -> None:
    cluster = _cluster("a")
    with pytest.raises(DistributedError) as error:
        cluster_route_session_typed_rpc(
            cluster,
            PROTOCOL,
            route_key="k",
            direction="client",
            message_type="Request",
            payload={"x": 1},
            schema_version=9,
        )
    assert error.value.code == "E007"


def test_route_without_members_reports_missing_node() -> None:
    cluster = Cluster()
    with pytest.raises(DistributedError) as error:
        cluster_route_session_typed_rpc(
            cluster,
            PROTOCOL,
            route_key="k",
            direction="client",
            message_type="Request",
            payload={},
        )
    assert error.value.code == "E006"


def test_majority_partition_wins_and_minority_is_fenced() -> None:
    cluster = _cluster("a", "b", "c")
    before = cluster.leases.epoch
    report = cluster_handle_partition_and_split_brain(cluster, [["a", "b"], ["c"]])
    assert report.decision == "majority"
    assert report.epoch == before + 1
    writable = [view.nodes for view in report.partitions if view.writable]
    assert writable == [("a", "b")]
    assert report.fences == {"c": report.epoch}
    assert "c" in cluster.dead


def test_split_without_majority_freezes_the_cluster() -> None:
    cluster = _cluster("a", "b", "c", "d")
    report = cluster_handle_partition_and_split_brain(
        cluster, [["a"], ["b"], ["c"], ["d"]]
    )
    assert report.decision == "frozen"
    assert report.quorum == 3
    assert all(not view.writable for view in report.partitions)
    assert set(cluster.dead) == {"a", "b", "c", "d"}


def test_overlapping_partitions_are_rejected() -> None:
    cluster = _cluster("a", "b")
    with pytest.raises(DistributedError) as error:
        cluster_handle_partition_and_split_brain(cluster, [["a", "b"], ["b"]])
    assert error.value.code == "E001"
    assert "two partitions" in str(error.value)


def test_unknown_node_in_partition_is_rejected() -> None:
    cluster = _cluster("a")
    with pytest.raises(DistributedError) as error:
        cluster_handle_partition_and_split_brain(cluster, [["a", "ghost"]])
    assert error.value.code == "E001"
    assert "unknown node" in str(error.value)
