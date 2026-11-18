"""Deterministic tests for the DSM/consensus reference foundations.

Everything here is local: no network, no threads, no cargo, and no wall-clock
dependence.  Each rejection path also pins the exact caller-supplied span.
"""

from __future__ import annotations

import pytest

from distributed import dsm
from distributed.dsm import (
    Acquire,
    CASNode,
    CausalEvent,
    FenceClaim,
    HLCStamp,
    HybridLogicalClock,
    MerkleTree,
    MVCCBatch,
    NewEpoch,
    PageAccess,
    Release,
    SoftwarePageTable,
    Tick,
    drive_future,
    dsm_atomic_cross_node_cas,
    dsm_compile_struct_to_crdt,
    dsm_distributed_gc_lease,
    dsm_false_sharing_page_split,
    dsm_merkle_tree_state_diff,
    dsm_mvcc_snapshot_isolation,
    dsm_raft_quorum_future_yield,
    dsm_split_brain_partition_fence,
    dsm_vector_clock_causality,
    dsm_virtual_page_fault_fetch,
)
from runtime.diagnostics import PillarError

SPAN = (17, 42)

REQUIRED_API = (
    "dsm_virtual_page_fault_fetch",
    "dsm_compile_struct_to_crdt",
    "dsm_raft_quorum_future_yield",
    "dsm_merkle_tree_state_diff",
    "dsm_vector_clock_causality",
    "dsm_split_brain_partition_fence",
    "dsm_mvcc_snapshot_isolation",
    "dsm_false_sharing_page_split",
    "dsm_distributed_gc_lease",
    "dsm_atomic_cross_node_cas",
)


def assert_diagnostic(
    error: PillarError,
    code: str,
    *,
    note: str | None = None,
    text: str | None = None,
) -> None:
    assert isinstance(error, PillarError)
    assert error.code == code
    assert error.span == SPAN
    assert error.line == 7 and error.col == 9
    assert f"span {SPAN[0]}..{SPAN[1]}" in str(error)
    if note is not None:
        assert error.diagnostic.note == note
    if text is not None:
        assert text in str(error)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_all_ten_public_entry_points_are_exported() -> None:
    for name in REQUIRED_API:
        assert callable(getattr(dsm, name))
        assert name in dsm.__all__
    assert "REJECTED_BACKENDS" in dsm.__all__


# ---------------------------------------------------------------------------
# 61. Virtual page fault fetch
# ---------------------------------------------------------------------------


def test_page_fetch_simulates_resident_and_remote_pages() -> None:
    table = SoftwarePageTable()
    table.install("p0", b"local")
    table.publish("p1", "remote-data", "node-b")

    first = dsm_virtual_page_fault_fetch(table, "p1", span=SPAN, line=7, col=9)
    assert first.page_id == "p1"
    assert first.data == b"remote-data"
    assert first.source == "remote"
    assert (first.faults, first.hits) == (1, 0)
    assert first.trace == ("fault:p1<-node-b",)

    second = table.fetch("p1")
    assert second.source == "resident"
    assert (second.faults, second.hits) == (1, 1)
    assert second.trace == ("fault:p1<-node-b", "hit:p1")

    direct = dsm_virtual_page_fault_fetch({"p0": "local"}, "p0")
    assert direct.source == "resident"
    assert direct.data == b"local"


def test_page_fetch_missing_page_is_e006() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_virtual_page_fault_fetch({}, "missing", span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E006", note="PageNotFound", text="not installed")


@pytest.mark.parametrize("backend", ["userfaultfd", "mach", "mprotect", "sigsegv"])
def test_privileged_page_fault_backends_are_rejected(backend: str) -> None:
    with pytest.raises(PillarError) as caught:
        dsm_virtual_page_fault_fetch({}, "p", backend=backend, span=SPAN, line=7, col=9)
    assert_diagnostic(
        caught.value,
        "E007",
        note="PrivilegedFeatureRejected",
        text="is not implemented",
    )


def test_page_table_validates_payload_backend_and_span() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_virtual_page_fault_fetch({"p": 123}, "p", span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E001")

    with pytest.raises(PillarError) as caught:
        dsm_virtual_page_fault_fetch({}, "p", backend="magic", span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_virtual_page_fault_fetch({}, "p", span=(9, 4))
    assert caught.value.code == "E003"
    assert caught.value.span is None


# ---------------------------------------------------------------------------
# 62. Struct -> CRDT lowering
# ---------------------------------------------------------------------------


def test_compile_struct_to_crdt_kinds_and_metadata() -> None:
    schema = dsm_compile_struct_to_crdt(
        {
            "title": "lww",
            "tags": "or-set",
            "votes": {"crdt": "pn-counter", "origin": "field-metadata"},
        },
        name="Doc",
        span=SPAN,
        line=7,
        col=9,
    )
    assert schema.name == "Doc"
    assert [(field.name, field.kind) for field in schema.fields] == [
        ("title", "lww-register"),
        ("tags", "or-set"),
        ("votes", "pn-counter"),
    ]
    assert schema.fields[2].metadata == {"origin": "field-metadata"}
    assert schema.describe()["fields"][0]["kind"] == "lww-register"


def test_crdt_replicas_converge_after_reordered_merges() -> None:
    schema = dsm_compile_struct_to_crdt(
        {"title": "lww-register", "tags": "or-set", "votes": "pn-counter"}
    )
    left, right = schema.document(), schema.document()
    left.apply("title", "set", value="left", timestamp=1, writer="a")
    right.apply("title", "set", value="right", timestamp=2, writer="b")
    left.apply("tags", "add", value="red", tag="a:1")
    left.apply("tags", "add", value="blue", tag="b:1")
    right.apply("tags", "add", value="green", tag="c:1")
    right.apply("tags", "remove", value="blue")
    left.apply("votes", "increment", replica="a", amount=3)
    right.apply("votes", "decrement", replica="b", amount=1)

    left.merge(right, span=SPAN, line=7, col=9)
    right.merge(left, span=SPAN, line=7, col=9)

    assert left.snapshot() == right.snapshot()
    assert left.fields["title"].value == "right"
    assert left.fields["tags"].value == frozenset({"red", "blue", "green"})
    assert left.fields["votes"].value == 2


def test_or_set_observed_remove_only_deletes_observed_tags() -> None:
    schema = dsm_compile_struct_to_crdt({"tags": "or-set"})
    left, right = schema.document(), schema.document()
    left.apply("tags", "add", value="x", tag="t1")
    right.merge(left)
    right.apply("tags", "remove", value="x")
    right.apply("tags", "add", value="x", tag="t2")
    left.merge(right)
    right.merge(left)
    assert left.fields["tags"].value == frozenset({"x"})
    assert right.snapshot() == left.snapshot()


def test_crdt_rejects_unknown_kind_and_schema_drift() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_compile_struct_to_crdt({"x": "set"}, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003", note="UnknownCRDTKind")

    left = dsm_compile_struct_to_crdt({"x": "lww-register"}).document()
    right = dsm_compile_struct_to_crdt({"x": "or-set"}).document()
    with pytest.raises(PillarError) as caught:
        left.merge(right, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E008", text="schema drift")


# ---------------------------------------------------------------------------
# 63. Raft-style log and cooperative future
# ---------------------------------------------------------------------------


def test_raft_quorum_math_and_commit_states() -> None:
    log = dsm_raft_quorum_future_yield(
        3, ["a", "b"], term=4, acks=2, span=SPAN, line=7, col=9
    )
    assert log.quorum == 2
    assert (log.cluster_size, log.term, log.role) == (3, 4, "leader")
    assert log.commit_index == 2
    assert [entry.index for entry in log.entries] == [1, 2]
    assert all(entry.committed for entry in log.entries)

    even = dsm_raft_quorum_future_yield(4, ["x"], span=SPAN, line=7, col=9)
    assert even.quorum == 3
    assert even.entries[0].acks == 3 and even.entries[0].committed

    weak = dsm_raft_quorum_future_yield(4, ["y"], acks=2)
    assert weak.quorum == 3
    assert not weak.entries[0].committed and weak.commit_index == 0


def test_raft_future_yields_a_pyfuture_like_stub() -> None:
    log = dsm_raft_quorum_future_yield(3, ["a", "b"], acks=2)
    observed: list[dict[str, object]] = []

    def on_yield(stub) -> None:
        assert not stub.done()
        stub.set_result(("override",))
        assert stub.done()
        observed.append(stub.describe())

    assert drive_future(log.future(), on_yield=on_yield) == ("override",)
    assert len(observed) == 1
    assert observed[0]["label"] == "raft-commit"
    assert observed[0]["done"] is True
    assert drive_future(log.future()) == ("a", "b")


def test_raft_future_cancellation_is_e008() -> None:
    log = dsm_raft_quorum_future_yield(3, ["a"], span=SPAN, line=7, col=9)
    with pytest.raises(PillarError) as caught:
        drive_future(log.future(), on_yield=lambda stub: stub.cancel())
    assert_diagnostic(caught.value, "E008", note="ConsensusDrift", text="cancelled")


@pytest.mark.parametrize("backend", ["kernel", "kernel-raft"])
def test_raft_rejects_kernel_consensus(backend: str) -> None:
    with pytest.raises(PillarError) as caught:
        dsm_raft_quorum_future_yield(3, [], backend=backend, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E007", note="PrivilegedFeatureRejected")


def test_raft_validates_quorum_arguments() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_raft_quorum_future_yield(0, [], span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_raft_quorum_future_yield(3, ["a"], acks=4, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003", text="cluster size")


# ---------------------------------------------------------------------------
# 64. Merkle tree state diff
# ---------------------------------------------------------------------------


def test_merkle_identical_states_have_equal_roots() -> None:
    diff = dsm_merkle_tree_state_diff(
        {"p1": b"a", "p2": "bb"},
        {"p2": b"bb", "p1": "a"},
        span=SPAN,
        line=7,
        col=9,
    )
    assert diff.equal
    assert diff.old_root == diff.new_root
    assert (diff.changed, diff.added, diff.removed) == ((), (), ())


def test_merkle_minimal_diff_and_membership_proof() -> None:
    diff = dsm_merkle_tree_state_diff(
        {"a": b"one", "b": b"two"},
        {"a": b"ONE", "c": "three"},
        span=SPAN,
        line=7,
        col=9,
    )
    assert not diff.equal
    assert diff.changed == ("a",)
    assert (diff.added, diff.removed) == (("c",), ("b",))

    proof = diff.new_tree.proof("a", span=SPAN, line=7, col=9)
    assert proof
    assert MerkleTree.verify("a", b"ONE", proof, diff.new_root)
    assert not MerkleTree.verify("a", b"one", proof, diff.new_root)
    assert MerkleTree.verify("c", "three", diff.new_tree.proof("c"), diff.new_root)

    with pytest.raises(PillarError) as caught:
        diff.new_tree.proof("b", span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E006", note="PageNotFound")


def test_merkle_rolling_update_matches_full_rebuild() -> None:
    tree = MerkleTree.from_pages({"p0": b"a", "p1": b"b", "p2": b"c"})
    updated = tree.update("p1", b"BB", span=SPAN, line=7, col=9)
    rebuilt = MerkleTree.from_pages({"p0": b"a", "p1": b"BB", "p2": b"c"})
    assert updated.root == rebuilt.root
    assert updated.root != tree.root
    assert MerkleTree.verify("p1", b"BB", updated.proof("p1"), updated.root)


def test_merkle_rejects_rdma_and_malformed_data() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_merkle_tree_state_diff({}, {}, backend="rdma", span=SPAN, line=7, col=9)
    assert_diagnostic(
        caught.value, "E007", note="PrivilegedFeatureRejected", text="RDMA"
    )

    with pytest.raises(PillarError) as caught:
        MerkleTree.from_pages({"p": 7}, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E001")

    with pytest.raises(PillarError) as caught:
        MerkleTree.from_pages({1: b"x"}, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E001")

    with pytest.raises(PillarError) as caught:
        MerkleTree.verify(
            "p", b"x", [("not-a-digest", True)], "root", span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E008")


# ---------------------------------------------------------------------------
# 65. Vector clocks, HLC, and causality anomalies
# ---------------------------------------------------------------------------


def test_vector_clock_happens_before_and_concurrent() -> None:
    events = [
        CausalEvent("e1", "a", {"a": 1}, hlc=HLCStamp(1000, 0, "a")),
        CausalEvent(
            "e2",
            "b",
            {"a": 1, "b": 1},
            dependencies=("e1",),
            hlc=HLCStamp(1001, 0, "b"),
        ),
        CausalEvent("e3", "a", {"a": 2, "b": 1}, dependencies=("e2",)),
        CausalEvent("e4", "c", {"c": 1}),
    ]
    report = dsm_vector_clock_causality(
        events, physical_now=1000, max_skew_ms=5, span=SPAN, line=7, col=9
    )
    assert ("e1", "e2") in report.happened_before
    assert ("e2", "e3") in report.happened_before
    assert ("e1", "e4") in report.concurrent
    assert report.final_vector == {"a": 2, "b": 1, "c": 1}
    assert report.final_hlc.logical > 0
    assert [event.event_id for event in report.events] == ["e1", "e2", "e3", "e4"]


def test_hlc_observe_then_tick_advances_monotonically() -> None:
    clock = HybridLogicalClock(100)
    first = clock.tick()
    assert (first.physical, first.logical) == (100, 1)

    observed = clock.observe(HLCStamp(105, 7, "peer"))
    assert (observed.physical, observed.logical, observed.node) == (105, 8, "peer")
    later = clock.tick()
    assert later.key > observed.key
    assert clock.describe()["logical"] == 9


@pytest.mark.parametrize(
    ("events", "text"),
    [
        (
            [
                CausalEvent("dup", "a", {"a": 1}),
                CausalEvent("dup", "b", {"b": 1}),
            ],
            "duplicate",
        ),
        ([CausalEvent("x", "a", {"a": 1}, dependencies=("ghost",))], "unknown"),
        (
            [
                CausalEvent("x", "a", {"a": 1}),
                CausalEvent("y", "a", {"a": 1}),
            ],
            "non-monotonic",
        ),
        (
            [
                CausalEvent("x", "a", {"a": 1}),
                CausalEvent("y", "b", {"a": 0, "b": 1}, dependencies=("x",)),
            ],
            "omits dependency",
        ),
    ],
)
def test_causality_anomalies_are_e008(events, text: str) -> None:
    with pytest.raises(PillarError) as caught:
        dsm_vector_clock_causality(events, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E008", note="CausalityViolation", text=text)


def test_causality_rejects_future_hlc_and_malformed_vector() -> None:
    future = [CausalEvent("x", "a", {"a": 1}, hlc=HLCStamp(9999, 0, "a"))]
    with pytest.raises(PillarError) as caught:
        dsm_vector_clock_causality(
            future, physical_now=1000, max_skew_ms=5, span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E008", text="future")

    with pytest.raises(PillarError) as caught:
        dsm_vector_clock_causality(
            [CausalEvent("x", "a", {"a": "bad"})], span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E001")


# ---------------------------------------------------------------------------
# 66. Split-brain fencing
# ---------------------------------------------------------------------------


def test_fence_picks_unique_winner_and_authorizes_it() -> None:
    result = dsm_split_brain_partition_fence(
        "page-0",
        [FenceClaim("old", 3), FenceClaim("new", 5)],
        strict=False,
        span=SPAN,
        line=7,
        col=9,
    )
    assert (result.winner.node, result.winner.token) == ("new", 5)
    assert [claim.node for claim in result.stale] == ["old"]
    assert result.epoch == 0
    assert (
        result.authorize(FenceClaim("new", 5), span=SPAN, line=7, col=9).node == "new"
    )


def test_fence_detects_split_brain_duplicate_token() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_split_brain_partition_fence(
            "page-0",
            [FenceClaim("a", 5), FenceClaim("b", 5)],
            span=SPAN,
            line=7,
            col=9,
        )
    assert_diagnostic(caught.value, "E008", note="SplitBrainError", text="split brain")


def test_fence_rejects_stale_primary_and_unauthorized_claim() -> None:
    current = FenceClaim("leader", 7, epoch=1)
    with pytest.raises(PillarError) as caught:
        dsm_split_brain_partition_fence(
            "page-0",
            [FenceClaim("stale", 6, 1)],
            current=current,
            span=SPAN,
            line=7,
            col=9,
        )
    assert_diagnostic(
        caught.value, "E008", note="StaleFenceError", text="stale primary"
    )

    result = dsm_split_brain_partition_fence(
        "page-0", [current], current=current, span=SPAN, line=7, col=9
    )
    with pytest.raises(PillarError) as caught:
        result.authorize(FenceClaim("stale", 6, 1), span=SPAN, line=7, col=9)
    assert_diagnostic(
        caught.value, "E008", note="StaleFenceError", text="stale fencing token"
    )


def test_fence_validates_claims_and_resource() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_split_brain_partition_fence(
            "r", [FenceClaim("a", -1)], span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_split_brain_partition_fence("r", [], span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")


# ---------------------------------------------------------------------------
# 67. MVCC snapshot isolation
# ---------------------------------------------------------------------------


def test_mvcc_snapshot_reads_ignore_later_commits() -> None:
    batches = [
        MVCCBatch("w1", writes={"x": 1}),
        MVCCBatch("reader-1", reads=("x",), commit=False, snapshot_ts=1),
        MVCCBatch("w2", writes={"x": 2}),
        MVCCBatch("reader-2", reads=("x",), commit=False, snapshot_ts=1),
        MVCCBatch("reader-3", reads=("x",), commit=False),
    ]
    report = dsm_mvcc_snapshot_isolation(
        batches, initial={"x": 0}, span=SPAN, line=7, col=9
    )
    assert [read.value for read in report.reads] == [1, 1, 2]
    assert [read.snapshot_ts for read in report.reads] == [1, 1, 2]
    assert report.committed == ("w1", "w2")
    assert report.rolled_back == ("reader-1", "reader-2", "reader-3")
    assert [
        (version.commit_ts, version.value, version.txn)
        for version in report.versions["x"]
    ] == [
        (0, 0, "init"),
        (1, 1, "w1"),
        (2, 2, "w2"),
    ]


def test_mvcc_write_conflict_is_e008() -> None:
    batches = [
        MVCCBatch("w1", writes={"x": 1}),
        MVCCBatch("w2", writes={"x": 3}, snapshot_ts=0),
    ]
    with pytest.raises(PillarError) as caught:
        dsm_mvcc_snapshot_isolation(batches, initial={"x": 0}, span=SPAN, line=7, col=9)
    assert_diagnostic(
        caught.value, "E008", note="StateMismatchError", text="write conflict"
    )


def test_mvcc_rejects_future_snapshot_and_duplicate_transaction() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_mvcc_snapshot_isolation(
            [MVCCBatch("r", reads=("x",), snapshot_ts=5)], span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E008", text="future")

    with pytest.raises(PillarError) as caught:
        dsm_mvcc_snapshot_isolation(
            [MVCCBatch("dup"), MVCCBatch("dup")], span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E003")


# ---------------------------------------------------------------------------
# 68. False-sharing page splitting
# ---------------------------------------------------------------------------


def test_false_sharing_splits_ping_ponged_page_into_hot_regions() -> None:
    accesses = [
        PageAccess("page", "a", 0),
        PageAccess("page", "b", 4),
        PageAccess("page", "a", 128),
        PageAccess("page", "b", 132),
    ]
    report = dsm_false_sharing_page_split(
        accesses,
        page_size=256,
        partitions=2,
        invalidation_threshold=2,
        span=SPAN,
        line=7,
        col=9,
    )
    assert report.accesses == 4
    assert report.invalidations == {"page": 3}
    assert len(report.plans) == 1
    plan = report.plans[0]
    assert plan.invalidations == 3
    assert [(sub.index, sub.start, sub.end, sub.hot) for sub in plan.sub_pages] == [
        (0, 0, 128, True),
        (1, 128, 256, True),
    ]
    assert len(plan.hot_regions) == 2


def test_false_sharing_ignores_single_writer_pages() -> None:
    accesses = [PageAccess("p", "a", 0), PageAccess("p", "a", 8)]
    report = dsm_false_sharing_page_split(accesses, page_size=64)
    assert report.plans == ()
    assert report.invalidations == {"p": 0}


def test_false_sharing_validates_bounds_and_parameters() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_false_sharing_page_split(
            [PageAccess("p", "a", 60, 8)], page_size=64, span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_false_sharing_page_split([], partitions=0, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_false_sharing_page_split(
            [("p", "a", 0)],
            span=SPAN,
            line=7,
            col=9,  # type: ignore[list-item]
        )
    assert_diagnostic(caught.value, "E001")


# ---------------------------------------------------------------------------
# 69. Lease-based distributed GC
# ---------------------------------------------------------------------------


def test_gc_lease_refcount_reclaims_after_last_release() -> None:
    report = dsm_distributed_gc_lease(
        [Acquire("blob", "a", 5), Release("blob", "a"), Tick(1)],
        span=SPAN,
        line=7,
        col=9,
    )
    assert report.reclaimed == ("blob",)
    assert report.live == {}
    assert (report.grants_issued, report.epoch, report.now) == (1, 0, 1)


def test_gc_lease_expiry_and_epoch_reclamation() -> None:
    operations = [Acquire("x", "a", 2), Acquire("x", "b", 5), Tick(2)]
    after_expiry = dsm_distributed_gc_lease(operations, span=SPAN, line=7, col=9)
    assert "x" not in after_expiry.reclaimed
    assert [grant.holder for grant in after_expiry.live["x"]] == ["b"]
    assert after_expiry.expirations == 1

    final = dsm_distributed_gc_lease(
        [*operations, NewEpoch(1)], span=SPAN, line=7, col=9
    )
    assert final.reclaimed == ("x",)
    assert final.live == {}
    assert final.epoch == 1


def test_gc_lease_rejects_stale_release_and_epoch_regression() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_distributed_gc_lease(
            [Acquire("x", "a", 5), NewEpoch(1), Release("x", "a")],
            span=SPAN,
            line=7,
            col=9,
        )
    assert_diagnostic(
        caught.value, "E008", note="LeaseExpiredError", text="stale or unknown"
    )

    with pytest.raises(PillarError) as caught:
        dsm_distributed_gc_lease([NewEpoch(2), NewEpoch(1)], span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E008", text="epoch must increase")

    with pytest.raises(PillarError) as caught:
        dsm_distributed_gc_lease([Acquire("x", "a", 0)], span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")


# ---------------------------------------------------------------------------
# 70. Atomic cross-node CAS
# ---------------------------------------------------------------------------


def test_cas_success_commits_all_nodes_without_mutating_inputs() -> None:
    nodes = {"n1": 1, "n2": 1, "n3": 1}
    result = dsm_atomic_cross_node_cas(nodes, 1, 2, span=SPAN, line=7, col=9)
    assert result.success
    assert result.committed == ("n1", "n2", "n3")
    assert result.rounds == ("prepare", "commit")
    assert result.ballots == 1
    assert all(node.value == 2 and node.phase == "committed" for node in result.nodes)
    assert nodes == {"n1": 1, "n2": 1, "n3": 1}


def test_cas_failure_aborts_prepare_and_leaves_values_unchanged() -> None:
    nodes = [CASNode("n1", 1), CASNode("n2", 7), CASNode("n3", 1)]
    result = dsm_atomic_cross_node_cas(nodes, 1, 2, span=SPAN, line=7, col=9)
    assert not result.success
    assert result.failed == ("n2",)
    assert result.committed == () and result.rounds == ("prepare",)
    assert [node.value for node in result.nodes] == [1, 7, 1]
    assert [node.phase for node in result.nodes] == ["aborted"] * 3


def test_cas_drift_after_prepare_is_e008() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_atomic_cross_node_cas(
            {"n1": 1, "n2": 1},
            1,
            2,
            drift_after_prepare={"n2": 99},
            span=SPAN,
            line=7,
            col=9,
        )
    assert_diagnostic(
        caught.value, "E008", note="ConsensusDrift", text="consensus drift"
    )


@pytest.mark.parametrize("backend", ["kernel", "kernel-raft", "rdma"])
def test_cas_rejects_privileged_backends(backend: str) -> None:
    with pytest.raises(PillarError) as caught:
        dsm_atomic_cross_node_cas(
            {"n1": 1}, 1, 2, backend=backend, span=SPAN, line=7, col=9
        )
    assert_diagnostic(caught.value, "E007", note="PrivilegedFeatureRejected")


def test_cas_validates_nodes_and_drift_targets() -> None:
    with pytest.raises(PillarError) as caught:
        dsm_atomic_cross_node_cas({}, 1, 2, span=SPAN, line=7, col=9)
    assert_diagnostic(caught.value, "E003")

    with pytest.raises(PillarError) as caught:
        dsm_atomic_cross_node_cas(
            {"n1": 1},
            1,
            2,
            drift_after_prepare={"ghost": 5},
            span=SPAN,
            line=7,
            col=9,
        )
    assert_diagnostic(caught.value, "E003")
