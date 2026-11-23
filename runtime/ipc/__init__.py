"""Zero-copy shared-memory ring arena with epoch-fenced borrow leases."""

from __future__ import annotations

from .arena import (
    FORMAT_VERSION,
    MAGIC,
    ArenaHeader,
    RingLease,
    SharedRingArena,
    ipc_map_shared_ring_arena,
    ipc_unlink_arena,
)
from .lease import BorrowLease, ipc_enforce_epoch_borrow_lease, ipc_release_borrow_lease
from .reclaim import ReclaimReport, ipc_reclaim_abandoned_slots

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "ArenaHeader",
    "BorrowLease",
    "ReclaimReport",
    "RingLease",
    "SharedRingArena",
    "ipc_enforce_epoch_borrow_lease",
    "ipc_map_shared_ring_arena",
    "ipc_reclaim_abandoned_slots",
    "ipc_release_borrow_lease",
    "ipc_unlink_arena",
]
