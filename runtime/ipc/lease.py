"""Epoch-fenced borrow leases over an arena slot.

A consumer that popped a slot holds a :class:`~runtime.ipc.arena.RingLease`.
Before it hands that buffer to another subsystem it must obtain a borrow lease
stamped with the arena's current epoch. If the arena is reclaimed in the
meantime the epoch moves on and the old lease is refused with ``E002``, which is
the cheap defence against use-after-reclaim (the TOCTOU window is closed by the
epoch comparison, not by a per-byte check).

Bounded model (SPEC 8.20, 10.50): the holder table is in-process bookkeeping.
Only the epoch and the ring cursor are shared through the mapped header.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .arena import RingLease, SharedRingArena


@dataclass(frozen=True)
class BorrowLease:
    """A consumer's claim on one popped slot, valid only in its epoch."""

    holder: str
    epoch: int
    index: int
    fence: int

    def token(self) -> str:
        return f"{self.holder}:{self.epoch}:{self.index}:{self.fence:08x}"


def _fence(holder: str, epoch: int, index: int) -> int:
    digest = hashlib.sha256(f"{holder}|{epoch}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def ipc_enforce_epoch_borrow_lease(
    arena: SharedRingArena,
    lease: RingLease,
    *,
    holder: str,
) -> BorrowLease:
    """Validate a popped slot against the arena epoch and register the holder."""
    if not holder:
        raise PillarError("E003", "borrow lease needs a non-empty holder name")
    if holder in arena.leases:
        raise PillarError(
            "E003",
            f"{holder} already holds a borrow lease; release it first",
            help_text="one live lease per holder keeps the reclaim table unambiguous",
        )
    epoch = arena.epoch
    if lease.epoch != epoch:
        raise PillarError(
            "E002",
            f"lease from epoch {lease.epoch} is stale; the arena is at epoch {epoch}",
            reason="the arena was reclaimed after the slot was popped",
            help_text="pop the slot again before borrowing it",
        )
    granted = BorrowLease(
        holder=holder,
        epoch=epoch,
        index=lease.index,
        fence=_fence(holder, epoch, lease.index),
    )
    arena.leases[holder] = granted
    return granted


def ipc_release_borrow_lease(arena: SharedRingArena, *, holder: str) -> BorrowLease:
    """Drop a borrow lease and return what was freed."""
    lease = arena.leases.pop(holder, None)
    if lease is None:
        raise PillarError(
            "E003", f"{holder} holds no borrow lease", help_text="grant before release"
        )
    return lease  # type: ignore[return-value]
