"""Reclaim slots abandoned by consumers that never released them.

Reclaiming is deliberately blunt: the ring is reset (every queued slot is
dropped), the leased slot bytes are zeroed, leases are cleared, and the epoch
advances so that any lease still held elsewhere becomes stale. Callers must
supply the new epoch explicitly, which makes the destructive step impossible to
perform by accident.

Bounded model (SPEC 8.20, 10.50): reclaim is whole-arena, not per-slot. A
per-slot variant would need a shared free list in the header; the header format
reserves that work for a later version.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .arena import HEADER_SIZE, SharedRingArena


@dataclass(frozen=True)
class ReclaimReport:
    """What a reclaim pass destroyed and which holders it evicted."""

    reclaimed: tuple[str, ...]
    epoch: int
    slots_zeroed: int
    bytes_zeroed: int
    dropped_queued: int


def ipc_reclaim_abandoned_slots(
    arena: SharedRingArena,
    *,
    now_epoch: int,
) -> ReclaimReport:
    """Reset the ring, zero abandoned slots, and fence old leases by epoch."""
    header = arena.header()
    if now_epoch <= header.epoch:
        raise PillarError(
            "E002",
            f"reclaim epoch must advance past {header.epoch}, got {now_epoch}",
            help_text="pass the next fencing epoch, not the current one",
        )
    if arena._map is None:
        raise PillarError("E003", "arena is closed")

    reclaimed = tuple(sorted(arena.leases))
    slots_zeroed = 0
    bytes_zeroed = 0
    for lease in arena.leases.values():
        start, end = arena._slot_bounds(lease.index)  # type: ignore[attr-defined]
        arena._map[start:end] = b"\x00" * (end - start)
        slots_zeroed += 1
        bytes_zeroed += end - start
    arena.leases.clear()

    dropped = header.used
    assert arena._map is not None
    if dropped:
        span = header.slots * arena.slot_size
        arena._map[HEADER_SIZE : HEADER_SIZE + span] = b"\x00" * span
    arena._write_header(epoch=now_epoch, head=0, tail=0, inflight=0, next_slot=0)
    return ReclaimReport(
        reclaimed=reclaimed,
        epoch=now_epoch,
        slots_zeroed=slots_zeroed,
        bytes_zeroed=bytes_zeroed,
        dropped_queued=dropped,
    )
