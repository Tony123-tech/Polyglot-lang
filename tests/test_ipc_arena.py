"""Shared-memory ring arena, borrow leases, and reclaim."""

from __future__ import annotations

import mmap
from pathlib import Path

import pytest

from runtime.diagnostics import PillarError
from runtime.ipc import (
    ipc_enforce_epoch_borrow_lease,
    ipc_map_shared_ring_arena,
    ipc_reclaim_abandoned_slots,
    ipc_release_borrow_lease,
    ipc_unlink_arena,
)


@pytest.fixture()
def arena_path(tmp_path: Path) -> Path:
    return tmp_path / "ring.bin"


def test_header_is_initialised_on_create(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=8, slot_size=64)
    try:
        header = arena.header()
        assert (header.version, header.slots, header.slot_size) == (1, 8, 64)
        assert (header.head, header.tail, header.epoch, header.inflight) == (0, 0, 0, 0)
        assert arena_path.exists()
    finally:
        arena.close()


def test_push_then_pop_reads_the_same_bytes(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=32)
    try:
        assert arena.push(b"hello arena") == 0
        lease = arena.pop()
        assert lease is not None
        assert bytes(lease.view[: len(b"hello arena")]) == b"hello arena"
        arena.release(lease)
        assert arena.header().tail == 1
    finally:
        arena.close()


def test_pop_is_zero_copy(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=32)
    try:
        arena.push(b"abc")
        lease = arena.pop()
        assert lease is not None
        # The view points into the mapping, not into a copied bytes object.
        assert isinstance(lease.view.obj, mmap.mmap)
        arena.release(lease)
    finally:
        arena.close()


def test_empty_ring_returns_none(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        assert arena.pop() is None
    finally:
        arena.close()


def test_full_ring_refuses_the_push(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=3, slot_size=16)
    try:
        # A ring of N slots holds N-1 payloads; the spare slot separates head from tail.
        assert arena.push(b"one") == 0
        assert arena.push(b"two") == 1
        assert arena.push(b"three") == -1
    finally:
        arena.close()


def test_outstanding_lease_blocks_overwrite(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"x")
        lease = arena.pop()
        assert lease is not None
        assert arena.push(b"y") == -1
        arena.release(lease)
        assert arena.push(b"y") >= 0
    finally:
        arena.close()


def test_release_must_be_fifo(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"a")
        arena.push(b"b")
        first = arena.pop()
        assert first is not None
        with pytest.raises(PillarError) as error:
            arena.release(type(first)(index=1, view=first.view, epoch=first.epoch))
        assert error.value.code == "E003"
    finally:
        arena.close()


def test_oversized_payload_is_rejected(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=8)
    try:
        with pytest.raises(PillarError) as error:
            arena.push(b"this is much too long")
        assert error.value.code == "E003"
    finally:
        arena.close()


def test_two_mappings_share_the_same_ring(arena_path: Path) -> None:
    producer = ipc_map_shared_ring_arena(arena_path, slots=8, slot_size=32)
    consumer = ipc_map_shared_ring_arena(arena_path, slots=8, slot_size=32)
    try:
        assert producer.push(b"across maps") == 0
        lease = consumer.pop()
        assert lease is not None
        assert bytes(lease.view[: len(b"across maps")]) == b"across maps"
        consumer.release(lease)
        assert producer.header().tail == 1
    finally:
        consumer.close()
        producer.close()


def test_foreign_file_is_refused(arena_path: Path) -> None:
    arena_path.write_bytes(b"\x00" * 512)
    with pytest.raises(PillarError) as error:
        ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    assert error.value.code == "E008"


def test_missing_arena_without_create_is_refused(arena_path: Path) -> None:
    with pytest.raises(PillarError) as error:
        ipc_map_shared_ring_arena(arena_path, create=False)
    assert error.value.code == "E006"


def test_slot_count_mismatch_is_rejected(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=8, slot_size=16)
    arena.close()
    with pytest.raises(PillarError) as error:
        ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    assert error.value.code == "E003"


def test_borrow_lease_is_granted_and_released(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"data")
        lease = arena.pop()
        assert lease is not None
        borrow = ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        assert borrow.epoch == 0
        assert borrow.index == 0
        assert borrow.token().startswith("reader-1:0:0:")
        released = ipc_release_borrow_lease(arena, holder="reader-1")
        assert released == borrow
    finally:
        arena.close()


def test_duplicate_holder_is_rejected(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"data")
        lease = arena.pop()
        assert lease is not None
        ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        with pytest.raises(PillarError) as error:
            ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        assert error.value.code == "E003"
    finally:
        arena.close()


def test_stale_epoch_lease_is_refused(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"data")
        lease = arena.pop()
        assert lease is not None
        ipc_reclaim_abandoned_slots(arena, now_epoch=5)
        with pytest.raises(PillarError) as error:
            ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        assert error.value.code == "E002"
        assert "stale" in str(error.value)
    finally:
        arena.close()


def test_reclaim_resets_the_ring_and_bumps_the_epoch(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        arena.push(b"leaked")
        lease = arena.pop()
        assert lease is not None
        ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        arena.push(b"queued")
        report = ipc_reclaim_abandoned_slots(arena, now_epoch=9)
        assert report.reclaimed == ("reader-1",)
        assert report.epoch == 9
        assert report.slots_zeroed == 1
        assert report.dropped_queued == 1
        assert arena.epoch == 9
        assert arena.pop() is None
        assert arena.leases == {}
    finally:
        arena.close()


def test_reclaim_epoch_must_advance(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=4, slot_size=16)
    try:
        with pytest.raises(PillarError) as error:
            ipc_reclaim_abandoned_slots(arena, now_epoch=0)
        assert error.value.code == "E002"
    finally:
        arena.close()


def test_engine_block_is_zeroed_on_reclaim(arena_path: Path) -> None:
    arena = ipc_map_shared_ring_arena(arena_path, slots=2, slot_size=16)
    try:
        arena.push(b"secret")
        lease = arena.pop()
        assert lease is not None
        ipc_enforce_epoch_borrow_lease(arena, lease, holder="reader-1")
        ipc_reclaim_abandoned_slots(arena, now_epoch=1)
        raw = arena_path.read_bytes()
        assert b"secret" not in raw
    finally:
        arena.close()
        ipc_unlink_arena(arena_path)
