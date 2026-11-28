"""Deterministic IO, reactor and resource-wrapper models for batch eight.

Channels, buffers, reactors and handle registries in this module are pure
Python models.  A reactor wakeup thread is owned by a context manager that
signals it and joins it on exit, including when the body raises.  Audits only
see resources wrapped by these helpers; no operating-system handle or process
descriptor is ever enumerated.  All timing values are supplied by the caller.

Failures use the shared ``E001``–``E008`` diagnostic surface through
:class:`compiler.frontend.errors.PolyglotError`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from compiler.frontend.errors import PolyglotError

__all__ = [
    "cio_buffer_reader",
    "cio_buffer_writer",
    "cio_channel_close",
    "cio_channel_context",
    "cio_channel_create",
    "cio_channel_get",
    "cio_channel_put",
    "cio_event_loop_tick",
    "cio_fd_budget",
    "cio_handle_audit",
    "cio_handle_close",
    "cio_handle_context",
    "cio_handle_track",
    "cio_io_result",
    "cio_io_retry",
    "cio_io_trace",
    "cio_line_reader",
    "cio_path_platform",
    "cio_reactor_poll",
    "cio_reactor_register",
    "cio_reactor_wakeup",
    "cio_read_exact",
    "cio_stream_chunk",
    "cio_stream_reassemble",
    "cio_write_batch",
]

_CODES = frozenset(f"E{value:03d}" for value in range(1, 10))
_PRIMITIVES = frozenset({"event", "queue", "timer"})
_PLATFORMS: dict[str, tuple[str, ...]] = {
    "posix": ("file", "pipe", "socket", "timer"),
    "linux": ("file", "pipe", "socket", "timer"),
    "darwin": ("file", "pipe", "socket", "timer"),
    "windows": ("file", "pipe", "socket", "timer"),
    "wasm": ("timer",),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    """Raise a structured :class:`PolyglotError` for this module."""
    if code not in _CODES:
        raise ValueError(f"unknown diagnostic code {code!r}")
    raise PolyglotError(code, message, reason=reason, note="runtime.concurrency.io")


def _require_name(value: object, label: str) -> str:
    """Return a non-empty str or raise E003."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty str")
    return value


def _require_int(value: object, label: str, minimum: int) -> int:
    """Return an integer at least *minimum* or raise E003."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("E003", f"{label} must be an int >= {minimum}")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    """Return a string-keyed mapping or raise E003."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail("E003", f"{label} must be a mapping with str keys")
    return value


def _require_sequence(value: object, label: str) -> Sequence[Any]:
    """Return a list or tuple or raise E003."""
    if not isinstance(value, (list, tuple)):
        _fail("E003", f"{label} must be a list or tuple")
    return value


def _require_callable(value: object, label: str) -> Callable[..., Any]:
    """Return a callable or raise E003."""
    if not callable(value):
        _fail("E003", f"{label} must be callable")
    return value


def _name_tuple(values: object, label: str) -> tuple[str, ...]:
    """Validate and sort a sequence of unique names (E003 on duplicates)."""
    items = _require_sequence(values, label)
    names = {_require_name(item, f"{label} item") for item in items}
    if len(names) != len(items):
        _fail("E003", f"{label} contains duplicates")
    return tuple(sorted(names))


@dataclass
class _Channel:
    """Bounded FIFO state owned by the channel helpers."""

    name: str
    capacity: int
    items: deque[Any] = field(default_factory=deque)
    closed: bool = False
    accepted: int = 0
    delivered: int = 0
    rejected: int = 0
    high_water: int = 0


@dataclass
class _Handle:
    """One wrapped resource with an idempotent closer."""

    handle_id: int
    name: str
    kind: str
    resource: Any
    closer: Callable[[Any], None]
    closed: bool = False
    error: str | None = None


@dataclass
class _ThreadOwner:
    """Joinable thread group for the reactor wakeup helper."""

    prefix: str
    join_timeout: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _threads: list[threading.Thread] = field(default_factory=list)
    _closed: bool = False

    def start(self, name: str, fn: Callable[[], Any]) -> threading.Thread:
        """Start a named worker and track it for a later join."""
        if self._closed:
            _fail("E003", "thread owner is already closed")

        def runner() -> None:
            try:
                fn()
            except BaseException as error:
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"

        thread = threading.Thread(
            target=runner, name=f"{self.prefix}-{name}", daemon=False
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def close(self) -> bool:
        """Signal cooperative stop and join every started thread."""
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            threads = tuple(self._threads)
        self.stop_event.set()
        for thread in threads:
            thread.join(timeout=self.join_timeout)
        alive = tuple(thread.name for thread in threads if thread.is_alive())
        if alive:
            _fail("E004", "reactor thread did not stop", reason=f"alive: {alive}")
        return True

    @property
    def thread_names(self) -> tuple[str, ...]:
        """Names of all started threads in start order."""
        with self._lock:
            return tuple(thread.name for thread in self._threads)


@contextmanager
def _owned_threads(prefix: str, join_timeout: float) -> Any:
    """Yield an owner that is always closed, even on body failure."""
    owner = _ThreadOwner(prefix, join_timeout)
    try:
        yield owner
    finally:
        owner.close()


def _new_registry() -> dict[str, Any]:
    """Create an empty reactor registry with a deterministic id counter."""
    return {"entries": {}, "next_id": 1}


def _channel_stats(channel: _Channel) -> dict[str, Any]:
    """Snapshot a channel's bounded counters."""
    return {
        "capacity": channel.capacity,
        "size": len(channel.items),
        "accepted": channel.accepted,
        "delivered": channel.delivered,
        "rejected": channel.rejected,
        "high_water": channel.high_water,
        "closed": channel.closed,
    }


def _require_channel(value: object) -> _Channel:
    """Return a channel created by this module or raise E003/E006."""
    if not isinstance(value, _Channel):
        _fail("E003", "channel must come from cio_channel_create")
    return value


def _require_registry(value: object) -> dict[str, Any]:
    """Validate a reactor registry structure or raise E003."""
    if not isinstance(value, dict) or "entries" not in value or "next_id" not in value:
        _fail("E003", "registry must come from cio_reactor_register")
    return value


# ---------------------------------------------------------------------------
# Public functions (ids 0761-0785)
# ---------------------------------------------------------------------------


def cio_channel_create(capacity: int, *, name: str = "channel") -> dict[str, Any]:
    """Create a bounded channel state record owned by this module.

    The channel is a pure in-memory queue; no file descriptors or OS resources
    are allocated.  Its counters support deterministic backpressure audits.
    """
    limit = _require_int(capacity, "capacity", 1)
    channel = _Channel(_require_name(name, "name"), limit)
    return {
        "channel": channel,
        "capacity": limit,
        "name": channel.name,
        "stats": _channel_stats(channel),
    }


def cio_channel_put(
    channel: _Channel,
    item: Any,
    *,
    overflow: str = "reject",
) -> dict[str, Any]:
    """Enqueue an item, applying bounded backpressure without blocking.

    ``overflow='reject'`` refuses a full channel and ``overflow='drop_oldest'``
    discards the oldest item so the new one fits.  Closed channels raise E003.
    """
    state = _require_channel(channel)
    if overflow not in {"reject", "drop_oldest"}:
        _fail("E003", f"unknown overflow policy {overflow!r}")
    if state.closed:
        _fail("E003", "cannot put on a closed channel")
    discarded: Any = None
    if len(state.items) >= state.capacity:
        if overflow == "reject":
            state.rejected += 1
            return {
                "accepted": False,
                "discarded": None,
                "size": len(state.items),
                "stats": _channel_stats(state),
            }
        discarded = state.items.popleft()
    state.items.append(item)
    state.accepted += 1
    state.high_water = max(state.high_water, len(state.items))
    return {
        "accepted": True,
        "discarded": discarded,
        "size": len(state.items),
        "stats": _channel_stats(state),
    }


def cio_channel_get(channel: _Channel) -> dict[str, Any]:
    """Pop the oldest buffered item, reporting emptiness instead of waiting."""
    state = _require_channel(channel)
    if state.closed:
        _fail("E003", "cannot get from a closed channel")
    if not state.items:
        return {"item": None, "empty": True, "size": 0, "stats": _channel_stats(state)}
    item = state.items.popleft()
    state.delivered += 1
    return {
        "item": item,
        "empty": False,
        "size": len(state.items),
        "stats": _channel_stats(state),
    }


def cio_channel_close(channel: _Channel, *, drain: bool = False) -> dict[str, Any]:
    """Close a channel and optionally discard its buffered items."""
    state = _require_channel(channel)
    discarded: tuple[Any, ...] = ()
    if state.closed:
        return {
            "closed": True,
            "already_closed": True,
            "discarded": discarded,
            "stats": _channel_stats(state),
        }
    if drain:
        discarded = tuple(state.items)
        state.items.clear()
    state.closed = True
    return {
        "closed": True,
        "already_closed": False,
        "discarded": discarded,
        "stats": _channel_stats(state),
    }


@contextmanager
def cio_channel_context(capacity: int, *, name: str = "channel") -> Any:
    """Yield a channel that is always closed when the ``with`` block exits.

    Closing on exception is guaranteed by the ``finally`` clause, so a failing
    body cannot leave a wrapped channel open.
    """
    created = cio_channel_create(capacity, name=name)
    channel: _Channel = created["channel"]
    envelope = {"channel": channel, "name": channel.name, "closed": False}
    try:
        yield envelope
    finally:
        cio_channel_close(channel)
        envelope["closed"] = True


def cio_buffer_writer(
    data: bytes,
    *,
    capacity: int,
    offset: int = 0,
) -> dict[str, Any]:
    """Write bytes into a bounded buffer, returning accepted and overflow bytes."""
    limit = _require_int(capacity, "capacity", 0)
    start = _require_int(offset, "offset", 0)
    if not isinstance(data, bytes):
        _fail("E003", "data must be bytes")
    room = max(0, limit - start)
    written = data[:room]
    overflow = data[room:]
    return {
        "written": written,
        "overflow": overflow,
        "next_offset": start + len(written),
        "capacity": limit,
        "truncated": bool(overflow),
    }


def cio_buffer_reader(
    buffer: bytes,
    *,
    offset: int = 0,
    size: int | None = None,
) -> dict[str, Any]:
    """Read a bounded slice from a bytes buffer with explicit offsets."""
    if not isinstance(buffer, bytes):
        _fail("E003", "buffer must be bytes")
    start = _require_int(offset, "offset", 0)
    length = None if size is None else _require_int(size, "size", 0)
    stop = len(buffer) if length is None else min(len(buffer), start + length)
    data = buffer[start:stop]
    return {
        "data": data,
        "offset": start,
        "next_offset": stop,
        "exhausted": stop >= len(buffer),
        "length": len(data),
    }


def cio_reactor_register(
    registry: dict[str, Any] | None = None,
    name: str = "source",
    *,
    kind: str = "event",
    primitive: str = "event",
    ready_at: int = 0,
) -> dict[str, Any]:
    """Register a wrapped event source with a deterministic ready tick.

    Only ``event``, ``queue`` and ``timer`` primitives are supported; anything
    else raises E007 because it would require a platform-specific facility.
    """
    target = _new_registry() if registry is None else _require_registry(registry)
    source = _require_name(name, "name")
    if kind not in {"event", "queue", "timer"}:
        _fail("E003", f"unsupported source kind {kind!r}")
    if primitive not in _PRIMITIVES:
        _fail("E007", f"unsupported reactor primitive {primitive!r}")
    tick = _require_int(ready_at, "ready_at", 0)
    if source in target["entries"]:
        _fail("E006", f"source {source!r} is already registered")
    entry = {
        "id": target["next_id"],
        "name": source,
        "kind": kind,
        "primitive": primitive,
        "ready_at": tick,
        "fired": False,
    }
    target["entries"][source] = entry
    target["next_id"] += 1
    return {"registry": target, "entry": entry, "id": entry["id"], "name": source}


def cio_reactor_poll(
    registry: dict[str, Any],
    *,
    tick: int = 0,
) -> dict[str, Any]:
    """Fire every registered source due at or before *tick*, in stable order."""
    target = _require_registry(registry)
    current = _require_int(tick, "tick", 0)
    due = sorted(
        (
            entry
            for entry in target["entries"].values()
            if not entry["fired"] and entry["ready_at"] <= current
        ),
        key=lambda entry: (entry["ready_at"], entry["name"]),
    )
    events = []
    for entry in due:
        entry["fired"] = True
        events.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "kind": entry["kind"],
                "ready_at": entry["ready_at"],
            }
        )
    pending = tuple(
        sorted(
            entry["name"] for entry in target["entries"].values() if not entry["fired"]
        )
    )
    return {
        "events": tuple(events),
        "pending": pending,
        "tick": current,
        "fired": len(events),
    }


@contextmanager
def cio_reactor_wakeup(*, join_timeout: float = 2.0) -> Any:
    """Yield a reactor whose owned wakeup thread is joined on context exit.

    The worker waits on a cooperative :class:`threading.Event`; the context
    manager always sets that event and joins before exiting, including when
    the body raises.  No OS-level selector or descriptor is used.
    """
    if not isinstance(join_timeout, (int, float)) or isinstance(join_timeout, bool):
        _fail("E003", "join_timeout must be a number")
    if join_timeout <= 0:
        _fail("E003", "join_timeout must be > 0")
    registry = _new_registry()
    event = threading.Event()
    lock = threading.Lock()
    records: list[dict[str, Any]] = []
    envelope: dict[str, Any] = {
        "registry": registry,
        "wake_event": event,
        "records": records,
        "closed": False,
        "threads_joined": False,
        "thread_names": (),
    }

    def waiter() -> None:
        event.wait(join_timeout)
        with lock:
            records.append({"kind": "wakeup", "reason": "shutdown"})

    with _owned_threads("cio-reactor", join_timeout) as owner:
        owner.start("wakeup", waiter)
        try:
            yield envelope
        finally:
            event.set()
            envelope["closed"] = True
    envelope["threads_joined"] = True
    envelope["thread_names"] = owner.thread_names


def cio_stream_chunk(data: bytes, *, size: int) -> dict[str, Any]:
    """Split bytes into fixed-size chunks carrying absolute offsets."""
    width = _require_int(size, "size", 1)
    if not isinstance(data, bytes):
        _fail("E003", "data must be bytes")
    chunks = tuple(
        {
            "offset": offset,
            "data": data[offset : offset + width],
            "length": len(data[offset : offset + width]),
        }
        for offset in range(0, len(data), width)
    )
    return {"chunks": chunks, "count": len(chunks), "total": len(data)}


def cio_stream_reassemble(
    chunks: Sequence[Mapping[str, Any]],
    *,
    expected: int | None = None,
) -> dict[str, Any]:
    """Reassemble offset-tagged chunks, reporting gaps and overlaps."""
    records = _require_sequence(chunks, "chunks")
    limit = None if expected is None else _require_int(expected, "expected", 0)
    pieces: list[tuple[int, bytes]] = []
    for index, item in enumerate(records):
        record = _require_mapping(item, f"chunk {index}")
        offset = _require_int(record.get("offset"), f"offset of chunk {index}", 0)
        payload = record.get("data", b"")
        if not isinstance(payload, bytes):
            _fail("E003", f"chunk {index} data must be bytes")
        pieces.append((offset, payload))
    pieces.sort(key=lambda piece: piece[0])
    assembled = bytearray()
    missing: list[int] = []
    overlaps: list[int] = []
    cursor = 0
    for offset, payload in pieces:
        if offset > cursor:
            missing.extend(range(cursor, offset))
            cursor = offset
        elif offset < cursor:
            overlaps.append(offset)
        assembled[offset : offset + len(payload)] = payload
        cursor = max(cursor, offset + len(payload))
    data = bytes(assembled)
    complete = not missing and (limit is None or len(data) == limit)
    return {
        "data": data,
        "missing": tuple(missing),
        "overlaps": tuple(overlaps),
        "complete": complete,
        "length": len(data),
    }


def cio_line_reader(text: str, *, keepends: bool = False) -> dict[str, Any]:
    """Split text into lines with deterministic handling of trailing newlines."""
    if not isinstance(text, str):
        _fail("E003", "text must be str")
    lines = tuple(text.splitlines(keepends=keepends))
    return {
        "lines": lines,
        "count": len(lines),
        "terminated": text.endswith(("\n", "\r")) or text == "",
        "keepends": keepends,
    }


def cio_io_result(
    value: Any = None,
    *,
    error: Any = None,
    name: str = "io",
) -> dict[str, Any]:
    """Wrap an IO outcome into a tagged result record (E003 for bad names)."""
    label = _require_name(name, "name")
    ok = error is None
    return {
        "name": label,
        "ok": ok,
        "status": "ok" if ok else "error",
        "value": value if ok else None,
        "error": None if ok else str(error),
    }


def cio_io_retry(
    attempts: Sequence[Callable[[], Any]],
    *,
    on_error: Callable[[BaseException], bool] | None = None,
) -> dict[str, Any]:
    """Run IO attempts in order, recording cooperative retry decisions."""
    steps = _require_sequence(attempts, "attempts")
    for index, step in enumerate(steps):
        _require_callable(step, f"attempt {index}")
    if on_error is not None:
        _require_callable(on_error, "on_error")
    errors: list[dict[str, Any]] = []
    value: Any = None
    succeeded = False
    for index, step in enumerate(steps):
        try:
            value = step()
        except BaseException as error:
            errors.append(
                {
                    "attempt": index,
                    "code": "E004",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            if on_error is None or not on_error(error):
                break
            continue
        succeeded = True
        break
    return {
        "succeeded": succeeded,
        "value": value,
        "attempts_run": len(errors) + int(succeeded),
        "errors": tuple(errors),
        "retries": max(0, len(errors) - 1),
    }


def cio_handle_track(
    registry: dict[str, Any],
    resource: Any,
    *,
    name: str,
    kind: str = "generic",
    closer: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Wrap a resource with an idempotent closer in a module-owned registry."""
    target = _require_registry(registry)
    label = _require_name(name, "name")
    if kind not in {"generic", "file", "socket", "tempdir"}:
        _fail("E003", f"unsupported handle kind {kind!r}")
    if closer is None:
        method = getattr(resource, "close", None)
        if not callable(method):
            _fail("E003", f"no closer for {label!r}")
        close = lambda obj: obj.close()
    else:
        close = _require_callable(closer, "closer")
    handle_id = target["next_id"]
    target["next_id"] += 1
    handle = _Handle(handle_id, label, kind, resource, close)
    target.setdefault("handles", {})[handle_id] = handle
    return {"registry": target, "handle": handle, "id": handle_id, "name": label}


def cio_handle_close(handle: _Handle) -> dict[str, Any]:
    """Close a tracked handle exactly once, capturing closer failures."""
    if not isinstance(handle, _Handle):
        _fail("E006", "handle was not created by cio_handle_track")
    already = handle.closed
    if not already:
        handle.closed = True
        try:
            handle.closer(handle.resource)
        except BaseException as error:
            handle.error = f"{type(error).__name__}: {error}"
    return {
        "id": handle.handle_id,
        "name": handle.name,
        "closed": handle.closed,
        "already_closed": already,
        "error": handle.error,
    }


def cio_handle_audit(
    registry: dict[str, Any],
    *,
    close_all: bool = False,
) -> dict[str, Any]:
    """Audit wrapped handles, optionally closing open ones.

    Only handles registered through :func:`cio_handle_track` are visible; the
    audit never inspects OS descriptors or the process handle table.
    """
    target = _require_registry(registry)
    handles = target.get("handles", {})
    if close_all:
        for handle in handles.values():
            if not handle.closed:
                cio_handle_close(handle)
    open_records = tuple(
        {"id": handle.handle_id, "name": handle.name, "kind": handle.kind}
        for handle in handles.values()
        if not handle.closed
    )
    errors = tuple(handle.error for handle in handles.values() if handle.error)
    return {
        "total": len(handles),
        "open": open_records,
        "open_count": len(open_records),
        "closed_count": len(handles) - len(open_records),
        "errors": errors,
        "clean": not open_records and not errors,
    }


@contextmanager
def cio_handle_context(
    registry: dict[str, Any],
    resource: Any,
    *,
    name: str,
    kind: str = "generic",
    closer: Callable[[Any], None] | None = None,
) -> Any:
    """Yield a tracked handle that is closed on exit, even if the body raises."""
    tracked = cio_handle_track(registry, resource, name=name, kind=kind, closer=closer)
    handle: _Handle = tracked["handle"]
    envelope = {
        "handle": handle,
        "id": handle.handle_id,
        "name": handle.name,
        "closed": False,
    }
    try:
        yield envelope
    finally:
        result = cio_handle_close(handle)
        envelope["closed"] = result["closed"]


def cio_path_platform(path: str, *, platform: str = "posix") -> dict[str, Any]:
    """Classify an IO path for a named platform without touching the filesystem."""
    if not isinstance(path, str) or not path:
        _fail("E003", "path must be a non-empty str")
    supported = _PLATFORMS.get(platform)
    if supported is None:
        _fail("E007", f"unsupported platform primitive {platform!r}")
    if path.startswith("\\\\") or (len(path) >= 2 and path[1] == ":"):
        kind = "file"
    elif path.startswith("pipe:"):
        kind = "pipe"
    elif path.startswith("socket:"):
        kind = "socket"
    elif path.startswith("timer:"):
        kind = "timer"
    else:
        kind = "file"
    ok = kind in supported
    return {
        "path": path,
        "platform": platform,
        "kind": kind,
        "supported": ok,
        "fallback": "in-memory reactor" if not ok else None,
        "available": supported,
    }


def cio_fd_budget(
    current: int,
    *,
    capacity: int,
    reserve: int = 0,
    release: int = 0,
) -> dict[str, Any]:
    """Model descriptor budget reservation and release with injected values."""
    used = _require_int(current, "current", 0)
    limit = _require_int(capacity, "capacity", 1)
    extra = _require_int(reserve, "reserve", 0)
    freed = _require_int(release, "release", 0)
    proposed = used - freed + extra
    granted = proposed <= limit
    after = proposed if granted else used
    return {
        "granted": granted,
        "after": after,
        "capacity": limit,
        "reserve": extra,
        "release": freed,
        "reason": "within budget" if granted else "budget exceeded",
    }


def cio_write_batch(
    writes: Sequence[Mapping[str, Any]],
    *,
    threshold: int,
) -> dict[str, Any]:
    """Group named writes into batches flushed at an injected size threshold."""
    records = _require_sequence(writes, "writes")
    limit = _require_int(threshold, "threshold", 1)
    batches: list[tuple[str, ...]] = []
    pending: list[str] = []
    total = 0
    for index, item in enumerate(records):
        record = _require_mapping(item, f"write {index}")
        name = _require_name(record.get("name"), f"write {index} name")
        size = _require_int(record.get("size"), f"size of {name!r}", 0)
        pending.append(name)
        total += size
        if total >= limit:
            batches.append(tuple(pending))
            pending = []
            total = 0
    return {
        "batches": tuple(batches),
        "flushed": len(batches),
        "pending": tuple(pending),
        "pending_size": total,
    }


def cio_read_exact(chunks: Sequence[bytes], *, size: int) -> dict[str, Any]:
    """Accumulate chunks until exactly *size* bytes are available."""
    records = _require_sequence(chunks, "chunks")
    wanted = _require_int(size, "size", 0)
    parts: list[bytes] = []
    total = 0
    consumed = 0
    for index, chunk in enumerate(records):
        if not isinstance(chunk, bytes):
            _fail("E003", f"chunk {index} must be bytes")
        parts.append(chunk)
        total += len(chunk)
        consumed += 1
        if total >= wanted:
            break
    data = b"".join(parts)[:wanted]
    return {
        "data": data,
        "complete": len(data) == wanted,
        "short": wanted - len(data),
        "consumed": consumed,
        "size": wanted,
    }


def cio_event_loop_tick(
    events: Sequence[Mapping[str, Any]],
    *,
    tick: int = 0,
) -> dict[str, Any]:
    """Fire due event-loop callbacks for one injected tick.

    Callbacks are invoked in ``(at, name)`` order and their results or errors
    are captured.  Deferred events remain untouched for the next tick.
    """
    entries = _require_sequence(events, "events")
    current = _require_int(tick, "tick", 0)
    prepared: list[tuple[int, str, Mapping[str, Any]]] = []
    for index, item in enumerate(entries):
        record = _require_mapping(item, f"event {index}")
        name = _require_name(record.get("name"), f"event {index} name")
        at = _require_int(record.get("at"), f"at of {name!r}", 0)
        callback = record.get("callback")
        if callback is not None:
            _require_callable(callback, f"callback of {name!r}")
        prepared.append((at, name, record))
    prepared.sort(key=lambda entry: (entry[0], entry[1]))
    fired: list[dict[str, Any]] = []
    deferred: list[str] = []
    results: list[dict[str, Any]] = []
    for at, name, record in prepared:
        if at > current:
            deferred.append(name)
            continue
        callback = record.get("callback")
        status = "ok"
        value: Any = None
        if callback is not None:
            try:
                value = callback()
            except BaseException as error:
                status = "error"
                value = f"{type(error).__name__}: {error}"
        fired.append({"name": name, "at": at, "status": status})
        results.append({"name": name, "value": value})
    return {
        "fired": tuple(fired),
        "deferred": tuple(deferred),
        "results": tuple(results),
        "tick": current,
    }


def cio_io_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize IO records into a sequence-ordered trace with kind counts."""
    entries = _require_sequence(records, "records")
    stamped: list[tuple[int, str, int, Mapping[str, Any]]] = []
    for index, item in enumerate(entries):
        record = _require_mapping(item, f"record {index}")
        seq = _require_int(record.get("seq"), f"seq of record {index}", 0)
        _require_name(record.get("name"), f"name of record {index}")
        kind = _require_name(record.get("kind", "io"), f"kind of record {index}")
        stamped.append((seq, kind, index, record))
    stamped.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    events = tuple(
        {
            "seq": seq,
            "step": index,
            "name": record.get("name"),
            "kind": kind,
            "tick": record.get("tick", seq),
        }
        for index, (seq, kind, _, record) in enumerate(stamped)
    )
    counts: dict[str, int] = {}
    for event in events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    ticks = tuple(event["tick"] for event in events)
    return {
        "events": events,
        "counts": {key: counts[key] for key in sorted(counts)},
        "first_tick": ticks[0] if ticks else None,
        "last_tick": ticks[-1] if ticks else None,
        "count": len(events),
    }
