"""Copy-on-write rollback journaling for hot patches.

The journal is a before-image log with two snapshot flavours:

* object snapshots copy each registered ``__dict__`` (or slot map) before the
  patch touches it, so restoring is a handful of ``setattr`` calls;
* buffer snapshots copy an explicitly registered memory range and restore it
  with ``memmove``.

The default object snapshot is copy-on-write at attribute granularity: nested
mutable values are shared, so patches must replace attributes instead of
mutating nested containers in place.  ``deep=True`` opts into a full deep copy.
OS-level page CoW without ``fork`` is not portable; the Rust counterpart
offers ``mmap(MAP_PRIVATE)`` snapshots for explicitly registered ranges.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import hashlib
import mmap
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import ast_nodes as ast
from runtime.hotpatch.errors import HotpatchError
from runtime.hotpatch.events import emit


def _buffer_address(target: Any) -> tuple[int, int, Any]:
    """Return ``(address, size, keepalive)`` for a writable buffer-like value."""
    if isinstance(target, tuple) and len(target) == 2:
        address, size = target
        return int(address), int(size), None
    if isinstance(target, bytearray):
        if not target:
            return 0, 0, target
        array = (ctypes.c_char * len(target)).from_buffer(target)
        return ctypes.addressof(array), len(target), array
    if isinstance(target, mmap.mmap):
        if target.closed or len(target) == 0:
            return 0, 0, target
        array = (ctypes.c_char * len(target)).from_buffer(target)
        return ctypes.addressof(array), len(target), array
    if isinstance(target, memoryview):
        if target.readonly:
            raise HotpatchError(
                "a read-only memoryview cannot be journaled",
                reason="the journal restores buffers in place",
                help_text="register a writable bytearray, mmap, or (address, size)",
            )
        array = (ctypes.c_char * target.nbytes).from_buffer(target)
        return ctypes.addressof(array), target.nbytes, array
    if isinstance(target, ctypes.Array):
        return ctypes.addressof(target), ctypes.sizeof(target), target
    raise HotpatchError(
        f"cannot journal buffer of type {type(target).__name__}",
        reason="no writable address is available",
        help_text="pass (address, size), bytearray, mmap, or a writable memoryview",
    )


@dataclass
class BufferSnapshot:
    address: int
    size: int
    before: bytes
    label: str = ""
    keepalive: Any = field(default=None, repr=False)

    def restore(self) -> None:
        if self.size:
            ctypes.memmove(self.address, self.before, self.size)

    def digest(self) -> bytes:
        return hashlib.blake2b(self.before, digest_size=16).digest()


@dataclass
class ObjectSnapshot:
    obj: Any
    values: dict[str, Any]
    slots: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def restore(self) -> None:
        for name, value in self.values.items():
            setattr(self.obj, name, value)
        for name, value in self.slots.items():
            object.__setattr__(self.obj, name, value)

    def digest(self) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        for name in sorted(self.values):
            digest.update(name.encode())
            digest.update(repr(self.values[name]).encode())
        for name in sorted(self.slots):
            digest.update(name.encode())
            digest.update(repr(self.slots[name]).encode())
        return digest.digest()


@dataclass
class ProgramSnapshot:
    program: ast.Program
    items: list[Any]
    label: str = ""

    def restore(self) -> None:
        self.program.items[:] = self.items


@dataclass
class Snapshot:
    generation: int
    label: str
    created_ns: int
    objects: list[ObjectSnapshot]
    buffers: list[BufferSnapshot]
    programs: list[ProgramSnapshot]
    undo: list[Callable[[], None]]
    span: tuple[int, int] | None = None
    source: str | None = None

    def digest(self) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        for snapshot in self.objects:
            digest.update(snapshot.digest())
        for snapshot in self.buffers:
            digest.update(snapshot.digest())
        for program in self.programs:
            digest.update(repr(program.items).encode())
        return digest.digest()

    def restore(self) -> tuple[int, int, int, int, list[str]]:
        restored_objects = 0
        restored_buffers = 0
        restored_programs = 0
        undo_errors: list[str] = []
        for program in reversed(self.programs):
            program.restore()
            restored_programs += 1
        for snapshot in reversed(self.buffers):
            snapshot.restore()
            restored_buffers += 1
        for snapshot in reversed(self.objects):
            snapshot.restore()
            restored_objects += 1
        for undo in reversed(self.undo):
            try:
                undo()
            except Exception as error:  # pragma: no cover - defensive
                undo_errors.append(f"{type(error).__name__}: {error}")
        return (
            restored_objects,
            restored_buffers,
            restored_programs,
            len(self.undo),
            undo_errors,
        )


class RollbackJournal:
    """A thread-safe before-image journal with instant rollback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshots: list[Snapshot] = []
        self._generation = 0
        self.last_rollback_ns = 0
        self.last_commit_ns = 0
        self.rollbacks = 0
        self.commits = 0

    # -- snapshots ---------------------------------------------------------

    def checkpoint(
        self,
        *,
        objects: Iterable[Any] = (),
        buffers: Iterable[Any] = (),
        program: ast.Program | None = None,
        deep: bool = False,
        label: str = "",
        span: tuple[int, int] | None = None,
        source: str | None = None,
    ) -> Snapshot:
        started = time.perf_counter_ns()
        object_snapshots = [self._capture_object(obj, deep, label) for obj in objects]
        buffer_snapshots = [self._capture_buffer(buf, label) for buf in buffers]
        programs: list[ProgramSnapshot] = []
        if program is not None:
            clone = copy.deepcopy(program.items)
            programs.append(ProgramSnapshot(program, clone, label))
        with self._lock:
            self._generation += 1
            snapshot = Snapshot(
                generation=self._generation,
                label=label,
                created_ns=started,
                objects=object_snapshots,
                buffers=buffer_snapshots,
                programs=programs,
                undo=[],
                span=span,
                source=source,
            )
            self._snapshots.append(snapshot)
        return snapshot

    def _capture_object(self, obj: Any, deep: bool, label: str) -> ObjectSnapshot:
        values: dict[str, Any] = {}
        slots: dict[str, Any] = {}
        namespace = getattr(obj, "__dict__", None)
        if isinstance(namespace, dict):
            values = copy.deepcopy(namespace) if deep else dict(namespace)
        for name in getattr(type(obj), "__slots__", ()) or ():
            if isinstance(name, str) and hasattr(obj, name):
                value = getattr(obj, name)
                slots[name] = copy.deepcopy(value) if deep else value
        if not values and not slots:
            raise HotpatchError(
                f"cannot journal {type(obj).__name__}: no __dict__ or slots",
                reason="the object has no mutable state to restore",
                help_text="journal dataclasses or instances with writable attributes",
            )
        return ObjectSnapshot(obj, values, slots, label)

    def _capture_buffer(self, target: Any, label: str) -> BufferSnapshot:
        address, size, keepalive = _buffer_address(target)
        before = ctypes.string_at(address, size) if size else b""
        return BufferSnapshot(address, size, before, label, keepalive)

    # -- manual undo entries ----------------------------------------------

    def record(
        self, undo: Callable[[], None], *, label: str = ""
    ) -> Callable[[], None]:
        snapshot = self._current(label)
        snapshot.undo.append(undo)
        return undo

    def _current(self, label: str = "") -> Snapshot:
        with self._lock:
            if not self._snapshots:
                self._generation += 1
                self._snapshots.append(
                    Snapshot(
                        generation=self._generation,
                        label=label,
                        created_ns=time.perf_counter_ns(),
                        objects=[],
                        buffers=[],
                        programs=[],
                        undo=[],
                    )
                )
            return self._snapshots[-1]

    # -- lifecycle ---------------------------------------------------------

    def commit(self, snapshot: Snapshot | None = None) -> int:
        started = time.perf_counter_ns()
        with self._lock:
            target = snapshot or (self._snapshots[-1] if self._snapshots else None)
            if target is not None and target in self._snapshots:
                self._snapshots.remove(target)
            self.commits += 1
            generation = target.generation if target is not None else 0
            commits = self.commits
            pending = len(self._snapshots)
        self.last_commit_ns = time.perf_counter_ns() - started
        # Sequence ids and counts only: never objects, labels, spans or paths.
        try:
            emit(
                "rollback",
                operation="commit",
                generation=generation,
                commits=commits,
                pending=pending,
            )
        except Exception:  # noqa: S110 - telemetry must never break commit
            pass
        return generation

    def rollback(self, snapshot: Snapshot | None = None) -> int:
        started = time.perf_counter_ns()
        with self._lock:
            target = snapshot or (self._snapshots.pop() if self._snapshots else None)
            if target is not None and target in self._snapshots:
                self._snapshots.remove(target)
            self.rollbacks += 1
            generation = target.generation if target is not None else 0
            pending = len(self._snapshots)
            rollbacks = self.rollbacks
        if target is None:
            self.last_rollback_ns = time.perf_counter_ns() - started
            try:
                emit(
                    "rollback",
                    operation="rollback",
                    generation=generation,
                    rollbacks=rollbacks,
                    pending=pending,
                )
            except Exception:  # noqa: S110 - telemetry is diagnostic-only
                pass
            return 0
        (
            restored_objects,
            restored_buffers,
            restored_programs,
            undo_calls,
            undo_errors,
        ) = target.restore()
        self.last_rollback_ns = time.perf_counter_ns() - started
        try:
            emit(
                "rollback",
                operation="rollback",
                generation=generation,
                rollbacks=rollbacks,
                pending=pending,
                restored_objects=restored_objects,
                restored_buffers=restored_buffers,
                restored_programs=restored_programs,
                undo_calls=undo_calls,
            )
        except Exception:  # noqa: S110 - telemetry must never break rollback
            pass
        if undo_errors:
            raise HotpatchError(
                "rollback completed with failing undo callbacks",
                span=target.span,
                source=target.source,
                reason="; ".join(undo_errors),
            )
        return target.generation

    def verify(self, snapshot: Snapshot | None = None) -> bool:
        """Return true when the current state still matches ``snapshot``."""
        target = snapshot or (self._snapshots[-1] if self._snapshots else None)
        if target is None:
            return True
        digest = hashlib.blake2b(digest_size=16)
        for object_snapshot in target.objects:
            values = getattr(object_snapshot.obj, "__dict__", {})
            for name in sorted(object_snapshot.values):
                digest.update(name.encode())
                digest.update(repr(values.get(name)).encode())
        for buffer_snapshot in target.buffers:
            current = ctypes.string_at(buffer_snapshot.address, buffer_snapshot.size)
            digest.update(current)
        return digest.digest() == target.digest()

    @contextlib.contextmanager
    def transaction(
        self,
        *,
        objects: Iterable[Any] = (),
        buffers: Iterable[Any] = (),
        program: ast.Program | None = None,
        deep: bool = False,
        label: str = "hot-patch",
        span: tuple[int, int] | None = None,
        source: str | None = None,
        verify: Callable[[], bool] | None = None,
    ) -> Iterator[Snapshot]:
        snapshot = self.checkpoint(
            objects=objects,
            buffers=buffers,
            program=program,
            deep=deep,
            label=label,
            span=span,
            source=source,
        )
        try:
            yield snapshot
        except Exception:
            self.rollback(snapshot)
            raise
        if verify is not None and not verify():
            self.rollback(snapshot)
            raise HotpatchError(
                "hot-patch verification failed; state was rolled back",
                span=span,
                source=source,
                reason="the verifier rejected the post-patch state",
                help_text="inspect the patch plan and retry with a compatible body",
            )
        self.commit(snapshot)

    def pending(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(snapshot.generation for snapshot in self._snapshots)


def hotpatch_rollback_journal_commit(
    journal: RollbackJournal | None = None,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> RollbackJournal:
    """Return a journal, creating one when the caller does not supply it.

    The returned object exposes ``checkpoint``, ``commit``, ``rollback``, and
    ``transaction``.  ``span``/``source`` are retained for diagnostics raised
    by a failed verification or a corrupt undo log.
    """
    if journal is not None and not isinstance(journal, RollbackJournal):
        raise HotpatchError(
            "the journal must be a RollbackJournal",
            span=span,
            source=source,
            reason=f"got {type(journal).__name__}",
        )
    return journal or RollbackJournal()


Journal = RollbackJournal

__all__ = [
    "BufferSnapshot",
    "Journal",
    "ObjectSnapshot",
    "ProgramSnapshot",
    "RollbackJournal",
    "Snapshot",
    "hotpatch_rollback_journal_commit",
]
