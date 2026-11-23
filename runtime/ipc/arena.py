"""A memory-mapped ring arena for zero-copy hand-off between processes.

Layout (little endian)::

    offset 0                    header (56 bytes)
    offset 56                   slot 0 | slot 1 | ... | slot N-1

``head`` is the next slot a producer will fill, ``tail`` the next slot a
consumer will read. ``inflight`` counts slots handed to a consumer but not yet
released, so a producer never overwrites a buffer someone is still reading.

Reading is genuinely zero-copy: :meth:`SharedRingArena.pop` returns a
``memoryview`` that points into the mapping. Writing copies the caller's bytes
into the slot once, which is unavoidable across an address space.

Bounded model (SPEC 8.20, 10.50): the arena is single-host (a file in
``/dev/shm`` or any tmpfs), the header is shared through ``mmap``, and the
per-holder lease table is local bookkeeping in this process. Slot payloads are
fixed-size byte buffers, not Arrow tensors.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from runtime.diagnostics import PillarError

MAGIC = b"PLYGRING"
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQQ")  # magic, version, slots, slot_size, next_slot, head, tail, epoch, inflight
HEADER_SIZE = HEADER.size
MAX_SLOTS = 1 << 20
MAX_SLOT_SIZE = 1 << 20


@dataclass(frozen=True)
class ArenaHeader:
    version: int
    slots: int
    slot_size: int
    next_slot: int
    head: int
    tail: int
    epoch: int
    inflight: int

    @property
    def used(self) -> int:
        return (self.head - self.tail) % self.slots


@dataclass(frozen=True)
class RingLease:
    """A popped slot that the consumer still owns."""

    index: int
    view: memoryview
    epoch: int

    @property
    def size(self) -> int:
        return len(self.view)


class SharedRingArena:
    """A file-backed ring buffer shared by every process that maps it."""

    def __init__(self, path: str | Path, *, slots: int, slot_size: int) -> None:
        self.path = Path(path)
        self.slots = slots
        self.slot_size = slot_size
        self._map: mmap.mmap | None = None
        self._views: list[memoryview] = []
        self.leases: dict[str, object] = {}

    # -- header ---------------------------------------------------------
    def _header_bytes(self) -> bytes:
        assert self._map is not None
        return self._map[:HEADER_SIZE]

    def header(self) -> ArenaHeader:
        if self._map is None:
            raise PillarError("E003", "arena is closed")
        magic, version, slots, slot_size, next_slot, head, tail, epoch, inflight = HEADER.unpack(
            self._header_bytes()
        )
        if magic != MAGIC:
            raise PillarError(
                "E008",
                f"arena header magic is {magic!r}, expected {MAGIC!r}",
                help_text="the file is not a polyglot ring arena",
            )
        if version != FORMAT_VERSION:
            raise PillarError(
                "E007",
                f"arena format version {version} is not supported",
                help_text=f"supported version: {FORMAT_VERSION}",
            )
        return ArenaHeader(version, slots, slot_size, next_slot, head, tail, epoch, inflight)

    def _write_header(self, **changes: int) -> None:
        assert self._map is not None
        header = self.header()
        values = {
            "version": header.version,
            "slots": header.slots,
            "slot_size": header.slot_size,
            "next_slot": header.next_slot,
            "head": header.head,
            "tail": header.tail,
            "epoch": header.epoch,
            "inflight": header.inflight,
        }
        values.update(changes)
        self._map[:HEADER_SIZE] = HEADER.pack(
            MAGIC,
            values["version"],
            values["slots"],
            values["slot_size"],
            values["next_slot"],
            values["head"],
            values["tail"],
            values["epoch"],
            values["inflight"],
        )

    @property
    def epoch(self) -> int:
        return self.header().epoch

    def _slot_bounds(self, index: int) -> tuple[int, int]:
        start = HEADER_SIZE + index * self.slot_size
        return start, start + self.slot_size

    # -- ring -----------------------------------------------------------
    def push(self, payload: bytes | bytearray | memoryview) -> int:
        """Copy ``payload`` into the next free slot; return its index."""
        header = self.header()
        data = bytes(payload)
        if len(data) > self.slot_size:
            raise PillarError(
                "E003",
                f"payload of {len(data)} bytes does not fit a {self.slot_size}-byte slot",
                help_text="split the message or map an arena with larger slots",
            )
        if header.inflight or (header.head + 1) % header.slots == header.tail:
            return -1
        assert self._map is not None
        start, _end = self._slot_bounds(header.head)
        self._map[start : start + len(data)] = data
        self._write_header(head=(header.head + 1) % header.slots)
        return header.head

    def pop(self) -> RingLease | None:
        """Hand the consumer a zero-copy view of the oldest slot."""
        header = self.header()
        if header.head == header.tail:
            return None
        assert self._map is not None
        start, end = self._slot_bounds(header.tail)
        view = memoryview(self._map)[start:end]
        self._views.append(view)
        self._write_header(inflight=header.inflight + 1)
        return RingLease(index=header.tail, view=view, epoch=header.epoch)

    def release(self, lease: RingLease) -> None:
        """Give the slot back; the tail only advances in FIFO order."""
        header = self.header()
        if header.inflight == 0:
            raise PillarError("E003", "no lease is outstanding", help_text="pop before release")
        if lease.index != header.tail:
            raise PillarError(
                "E003",
                f"lease {lease.index} released out of order; tail is {header.tail}",
                help_text="release leases in the order they were popped",
            )
        self._drop_view(lease.view)
        self._write_header(tail=(header.tail + 1) % header.slots, inflight=header.inflight - 1)

    def _drop_view(self, view: memoryview) -> None:
        for index, live in enumerate(self._views):
            if live is view:
                del self._views[index]
                break
        try:
            view.release()
        except ValueError:
            pass

    def release_all_views(self) -> None:
        """Drop every outstanding consumer view (used by reclaim)."""
        views, self._views = self._views, []
        for view in views:
            try:
                view.release()
            except ValueError:
                pass

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self.release_all_views()
        if self._map is not None:
            self._map.close()
            self._map = None


def ipc_map_shared_ring_arena(
    path: str | Path,
    *,
    slots: int = 64,
    slot_size: int = 256,
    create: bool = True,
) -> SharedRingArena:
    """Map (or create) a ring arena at ``path``.

    Creating initialises the header; opening an existing file validates the
    magic, so mapping a random file fails with ``E008`` instead of corrupting
    memory.
    """
    if not 2 <= slots <= MAX_SLOTS:
        raise PillarError(
            "E003",
            f"slot count must be between 2 and {MAX_SLOTS}, got {slots}",
            help_text="a ring needs at least one slot plus one empty sentinel",
        )
    if not 1 <= slot_size <= MAX_SLOT_SIZE:
        raise PillarError("E003", f"slot size must be between 1 and {MAX_SLOT_SIZE}, got {slot_size}")

    target = Path(path)
    exists = target.exists()
    if not exists and not create:
        raise PillarError(
            "E006",
            f"arena file does not exist: {target}",
            help_text="create the arena first, or pass create=True",
        )

    arena = SharedRingArena(target, slots=slots, slot_size=slot_size)
    size = HEADER_SIZE + slots * slot_size
    if not exists:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(HEADER.pack(MAGIC, FORMAT_VERSION, slots, slot_size, 0, 0, 0, 0, 0))
            handle.write(b"\x00" * (slots * slot_size))
    elif target.stat().st_size < size:
        raise PillarError(
            "E008",
            f"arena file is {target.stat().st_size} bytes, needs at least {size}",
            help_text="delete the stale arena and map it again",
        )

    with open(target, "r+b") as handle:
        arena._map = mmap.mmap(handle.fileno(), max(size, target.stat().st_size))
    header = arena.header()  # validates magic and version
    if header.slots != slots and exists:
        raise PillarError(
            "E003",
            f"arena file was created with {header.slots} slots, not {slots}",
            help_text="map with the slot count recorded in the header",
        )
    return arena


def ipc_unlink_arena(path: str | Path) -> None:
    """Remove the backing file (best effort; open mappings keep it alive)."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
