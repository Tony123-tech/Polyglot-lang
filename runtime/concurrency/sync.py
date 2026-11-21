"""Deterministic synchronization primitives and models for batch eight.

This module pairs real, joinable ``threading`` primitives with small pure
models of wait queues, lock ownership and fairness policies.  Every helper
that starts a thread uses an owned close path that signals a stop event and
joins the thread before returning.  Context managers release locks and guards
even when their body raises.  Timing is injected (a clock callable or an
integer tick) instead of measured with real sleeping.

Failures are reported through the shared ``E001``–``E008`` diagnostic surface
via :class:`compiler.frontend.errors.PolyglotError`.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from compiler.frontend.errors import PolyglotError

__all__ = [
    "csync_acquire_ordered",
    "csync_atomic_compare_expect",
    "csync_barrier_arrive",
    "csync_barrier_thread_run",
    "csync_check_poison",
    "csync_condition_signal_run",
    "csync_condition_waiters",
    "csync_counter_snapshot",
    "csync_deadlock_probe",
    "csync_detect_cycle",
    "csync_fair_queue",
    "csync_guard",
    "csync_guard_context_report",
    "csync_guard_group",
    "csync_lock_audit",
    "csync_lock_order_plan",
    "csync_notify_policy",
    "csync_owned_lock",
    "csync_poison_state",
    "csync_reader_writer_plan",
    "csync_reentrant_depth",
    "csync_rw_lock_run",
    "csync_semaphore_tickets",
    "csync_spin_tick",
    "csync_timeout_remaining",
]

_CODES = frozenset(f"E{value:03d}" for value in range(1, 9))
_RELEASERS = ("release", "close", "shutdown", "stop")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    """Raise a structured :class:`PolyglotError` for this module."""
    if code not in _CODES:
        raise ValueError(f"unknown diagnostic code {code!r}")
    raise PolyglotError(code, message, reason=reason, note="runtime.concurrency.sync")


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


def _require_number(value: object, label: str) -> int | float:
    """Return a finite real number or raise E003."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E003", f"{label} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        _fail("E003", f"{label} must be finite")
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
    if isinstance(values, Mapping):
        items = tuple(values)
    else:
        items = _require_sequence(values, label)
    names = {_require_name(item, f"{label} item") for item in items}
    if len(names) != len(items):
        _fail("E003", f"{label} contains duplicates")
    return tuple(sorted(names))


def _infer_releaser(resource: Any, name: str) -> Callable[[Any], None]:
    """Pick a deterministic release method or raise E003 for bare resources."""
    method = next(
        (
            candidate
            for candidate in _RELEASERS
            if callable(getattr(resource, candidate, None))
        ),
        None,
    )
    if method is None:
        _fail("E003", f"no releaser for {name!r}")
    return lambda obj: getattr(obj, method)()


@dataclass
class _ThreadOwner:
    """Small joinable thread group used by the thread-starting helpers."""

    prefix: str
    join_timeout: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _threads: list[threading.Thread] = field(default_factory=list)
    _outcomes: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def start(
        self,
        name: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        *,
        before: Callable[[], None] | None = None,
        after: Callable[[], None] | None = None,
    ) -> threading.Thread:
        """Start a named worker whose result or exception is recorded."""
        if self._closed:
            _fail("E003", "thread owner is already closed")

        def runner() -> None:
            try:
                if before is not None:
                    before()
                value = fn(*args)
            except BaseException as error:
                outcome = {"name": name, "ok": False, "value": None, "error": error}
            else:
                outcome = {"name": name, "ok": True, "value": value, "error": None}
            finally:
                if after is not None:
                    after()
            with self._lock:
                self._outcomes.append(outcome)

        thread = threading.Thread(
            target=runner, name=f"{self.prefix}-{name}", daemon=False
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def close(self) -> bool:
        """Set the stop event and join every started thread."""
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
            _fail("E004", "threads did not stop", reason=f"alive: {alive}")
        return True

    @property
    def outcomes(self) -> tuple[dict[str, Any], ...]:
        """Immutable snapshot of worker outcomes."""
        with self._lock:
            return tuple(self._outcomes)


@contextmanager
def _owned_threads(prefix: str, join_timeout: float) -> Any:
    """Yield an owner whose close always runs, including on body exceptions."""
    owner = _ThreadOwner(prefix, join_timeout)
    try:
        yield owner
    finally:
        owner.close()


def _outcome_report(owner: _ThreadOwner, names: Sequence[str]) -> dict[str, Any]:
    """Convert an owner's outcomes into a deterministic report."""
    by_name = {item["name"]: item for item in owner.outcomes}
    values: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for name in sorted(names):
        item = by_name.get(name)
        if item is None:
            failures.append({"name": name, "code": "E004", "error": "no outcome"})
        elif item["ok"]:
            values[name] = item["value"]
        else:
            error = item["error"]
            failures.append(
                {
                    "name": name,
                    "code": "E004",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "values": {name: values[name] for name in sorted(values)},
        "failures": tuple(failures),
        "threads_joined": True,
    }


def _cycle_path(adjacency: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Return one directed cycle as a path of names, or ``None``."""
    colors: dict[str, int] = {name: 0 for name in adjacency}
    stack: list[str] = []

    def visit(name: str) -> tuple[str, ...] | None:
        colors[name] = 1
        stack.append(name)
        for child in adjacency[name]:
            if colors[child] == 1:
                start = stack.index(child)
                return tuple(stack[start:] + [child])
            if colors[child] == 0:
                found = visit(child)
                if found is not None:
                    return found
        stack.pop()
        colors[name] = 2
        return None

    for name in sorted(adjacency):
        if colors[name] == 0:
            found = visit(name)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Public functions (ids 0736-0760)
# ---------------------------------------------------------------------------


def csync_lock_order_plan(
    locks: Mapping[str, Any],
    *,
    release_reverse: bool = True,
) -> dict[str, Any]:
    """Plan lock acquisition in sorted key order and release order.

    A lock-like object must expose both ``acquire`` and ``release``; otherwise
    E003 is raised.  Results are plain tuples so callers can compare plans.
    """
    mapping = _require_mapping(locks, "locks")
    names = _name_tuple(mapping, "lock names")
    for name in names:
        if not callable(getattr(mapping[name], "acquire", None)) or not callable(
            getattr(mapping[name], "release", None)
        ):
            _fail("E003", f"lock {name!r} lacks acquire()/release()")
    release = tuple(reversed(names)) if release_reverse else names
    return {
        "acquire": names,
        "release": release,
        "count": len(names),
        "deterministic": True,
    }


def csync_detect_cycle(
    edges: Sequence[tuple[str, str]],
    *,
    raise_on_cycle: bool = True,
) -> dict[str, Any]:
    """Detect a cycle in a wait-for edge list with depth-first search.

    When *raise_on_cycle* is true a detected cycle raises E005; otherwise the
    cycle is returned as data.  Duplicate edges are collapsed deterministically.
    """
    pairs = _require_sequence(edges, "edges")
    adjacency: dict[str, tuple[str, ...]] = {}
    for index, pair in enumerate(pairs):
        items = _require_sequence(pair, f"edge {index}")
        if len(items) != 2:
            _fail("E003", f"edge {index} must be a (from, to) pair")
        source = _require_name(items[0], "edge source")
        target = _require_name(items[1], "edge target")
        adjacency.setdefault(source, ())
        adjacency.setdefault(target, ())
        adjacency[source] = tuple(sorted({*adjacency[source], target}))
    cycle = _cycle_path(adjacency)
    if cycle is not None and raise_on_cycle:
        _fail("E005", "lock-order cycle detected", reason=" -> ".join(cycle))
    return {
        "cycle": cycle,
        "acyclic": cycle is None,
        "nodes": tuple(sorted(adjacency)),
        "edges": tuple(
            (name, child) for name in sorted(adjacency) for child in adjacency[name]
        ),
    }


@contextmanager
def csync_acquire_ordered(
    locks: Mapping[str, Any],
    *,
    timeout: float | None = None,
    clock: Callable[[], float] | None = None,
) -> Any:
    """Acquire locks in sorted order with rollback, yielding that order.

    The deadline is computed from the injected *clock*, so tests need no real
    waiting.  Locks already acquired are released in reverse order when the
    body raises or when acquisition times out (E004).
    """
    plan = csync_lock_order_plan(locks)
    if clock is not None:
        _require_callable(clock, "clock")
    if timeout is not None:
        limit = _require_number(timeout, "timeout")
        if limit < 0:
            _fail("E003", "timeout must be >= 0 or None")
    deadline = (
        None if timeout is None else (0.0 if clock is None else clock()) + timeout
    )
    acquired: list[Any] = []
    try:
        for name in plan["acquire"]:
            lock = locks[name]
            if deadline is None:
                ok = lock.acquire()
            else:
                remaining = deadline - (0.0 if clock is None else clock())
                if remaining <= 0:
                    _fail("E004", f"lock order timeout at {name!r}")
                ok = lock.acquire(timeout=remaining)
            if not ok:
                _fail("E004", f"could not acquire lock {name!r}")
            acquired.append(lock)
        yield plan["acquire"]
    finally:
        for lock in reversed(acquired):
            lock.release()


@contextmanager
def csync_guard(
    resource: Any,
    *,
    releaser: Callable[[Any], None] | None = None,
    name: str = "resource",
) -> Any:
    """Yield an RAII envelope that releases *resource* on context exit.

    The envelope records the release state so callers can assert deterministic
    cleanup.  Release failures are converted into E004 diagnostics.
    """
    _require_name(name, "name")
    release = releaser if releaser is not None else _infer_releaser(resource, name)
    _require_callable(release, "releaser")
    envelope = {"name": name, "resource": resource, "released": False}
    try:
        yield envelope
    finally:
        if not envelope["released"]:
            try:
                release(resource)
            except BaseException as error:
                _fail(
                    "E004",
                    f"releaser failed for {name!r}",
                    reason=f"{type(error).__name__}: {error}",
                )
            envelope["released"] = True


@contextmanager
def csync_guard_group(
    resources: Mapping[str, Any],
    *,
    releaser: Callable[[Any], None] | None = None,
) -> Any:
    """Yield named guards that are released in reverse order on exit.

    Even if the body raises, every guard is released exactly once.  The
    yielded mapping preserves the sorted lock names for deterministic order.
    """
    mapping = _require_mapping(resources, "resources")
    names = _name_tuple(mapping, "resource names")
    envelopes = {
        name: {"name": name, "resource": mapping[name], "released": False}
        for name in names
    }
    try:
        yield envelopes
    finally:
        for name in reversed(names):
            envelope = envelopes[name]
            if envelope["released"]:
                continue
            release = releaser or _infer_releaser(envelope["resource"], name)
            try:
                release(envelope["resource"])
            except BaseException as error:
                _fail(
                    "E004",
                    f"releaser failed for {name!r}",
                    reason=f"{type(error).__name__}: {error}",
                )
            envelope["released"] = True


def csync_reentrant_depth(
    events: Sequence[str],
    *,
    owner: str = "main",
) -> dict[str, Any]:
    """Model reentrant acquire/release events and report final depth.

    Releasing at depth zero is E003.  The history captures each state change
    and whether the reentrant lock was fully balanced.
    """
    _require_name(owner, "owner")
    steps = _require_sequence(events, "events")
    depth = 0
    history: list[dict[str, Any]] = []
    for index, event in enumerate(steps):
        if event not in {"acquire", "release"}:
            _fail("E003", f"invalid event {event!r} at index {index}")
        before = depth
        if event == "acquire":
            depth += 1
        else:
            if depth == 0:
                _fail("E003", "release without matching acquire")
            depth -= 1
        history.append(
            {
                "index": index,
                "event": event,
                "before": before,
                "after": depth,
                "owner": owner,
            }
        )
    return {
        "depth": depth,
        "balanced": depth == 0,
        "history": tuple(history),
        "owner": owner,
        "operations": len(steps),
    }


def csync_semaphore_tickets(
    initial: int,
    events: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Simulate FIFO semaphore tickets over acquire/release events.

    Failed acquires become waiters and are granted in event order when tokens
    are released.  The returned counters are deterministic and side-effect free.
    """
    available = _require_int(initial, "initial", 0)
    steps = _require_sequence(events, "events")
    waiters: deque[str] = deque()
    granted: list[str] = []
    denied: list[str] = []
    for index, event in enumerate(steps):
        items = _require_sequence(event, f"event {index}")
        if len(items) != 2:
            _fail("E003", f"event {index} must be (operation, name)")
        operation = items[0]
        name = _require_name(items[1], "waiter name")
        if operation == "acquire":
            if available > 0:
                available -= 1
                granted.append(name)
            else:
                waiters.append(name)
                denied.append(name)
        elif operation == "release":
            available += 1
            if waiters:
                next_name = waiters.popleft()
                granted.append(next_name)
                available -= 1
        else:
            _fail("E003", f"invalid semaphore operation {operation!r}")
    return {
        "tickets": available,
        "granted": tuple(granted),
        "denied": tuple(denied),
        "waiters": tuple(waiters),
    }


def csync_barrier_arrive(
    participants: Sequence[str],
    arrived: Sequence[str],
) -> dict[str, Any]:
    """Model barrier arrival without blocking the caller.

    Unknown or duplicate arrivals are E003.  The barrier opens only when every
    participant has arrived, at which point all arrivals are reported released.
    """
    parties = _name_tuple(participants, "participants")
    arrivals = _require_sequence(arrived, "arrived")
    seen = {_require_name(name, "arrived name") for name in arrivals}
    if len(seen) != len(arrivals):
        _fail("E003", "duplicate barrier arrival")
    unknown = tuple(sorted(seen - set(parties)))
    if unknown:
        _fail("E003", f"unknown barrier participants: {unknown}")
    open_now = seen == set(parties)
    return {
        "open": open_now,
        "waiting": tuple(sorted(set(parties) - seen)),
        "released": tuple(sorted(seen)) if open_now else (),
        "parties": parties,
        "arrived": tuple(sorted(seen)),
    }


def csync_condition_waiters(
    events: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Simulate a condition wait queue driven by wait/notify events.

    ``notify`` wakes the longest-waiting name; ``notify_all`` wakes everyone.
    Invalid operations or unknown names raise E003.
    """
    steps = _require_sequence(events, "events")
    waiting: deque[str] = deque()
    notified: list[str] = []
    history: list[dict[str, Any]] = []
    for index, event in enumerate(steps):
        items = _require_sequence(event, f"event {index}")
        if len(items) != 2:
            _fail("E003", f"event {index} must be (operation, name)")
        operation, name = items[0], _require_name(items[1], "waiter name")
        if operation == "wait":
            if name in waiting:
                _fail("E003", f"waiter {name!r} is already waiting")
            waiting.append(name)
        elif operation == "notify":
            if not waiting:
                _fail("E006", "no waiter to notify")
            notified.append(waiting.popleft())
        elif operation == "notify_all":
            notified.extend(waiting)
            waiting.clear()
        else:
            _fail("E003", f"invalid condition operation {operation!r}")
        history.append(
            {
                "index": index,
                "operation": operation,
                "name": name,
                "waiting": tuple(waiting),
                "notified": tuple(notified),
            }
        )
    return {
        "waiting": tuple(waiting),
        "notified": tuple(notified),
        "history": tuple(history),
        "size": len(waiting),
    }


def csync_notify_policy(
    waiters: Sequence[str],
    *,
    policy: str,
    count: int = 1,
) -> dict[str, Any]:
    """Select waiters using a named deterministic notification policy.

    ``one`` selects the first *count* waiters and ``all`` selects everyone.
    Randomized policies are outside the portable model and raise E007.
    """
    names = _name_tuple(waiters, "waiters")
    if policy not in {"one", "all"}:
        if policy == "random":
            _fail("E007", "randomized notification is unsupported")
        _fail("E003", f"unknown notification policy {policy!r}")
    number = _require_int(count, "count", 1)
    selected = names if policy == "all" else names[:number]
    return {
        "selected": selected,
        "remaining": tuple(name for name in names if name not in selected),
        "policy": policy,
        "count": len(selected),
    }


def csync_reader_writer_plan(
    queue: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Simulate reader/writer admission with writer priority.

    Readers share access while no writer is active or waiting; a writer is
    admitted only when no reader is active.  Deferred requests keep their
    order for a later tick.
    """
    steps = _require_sequence(queue, "queue")
    active_readers = 0
    writer_active = False
    waiting_writers = 0
    admitted: list[str] = []
    deferred: list[str] = []
    for index, event in enumerate(steps):
        items = _require_sequence(event, f"request {index}")
        if len(items) != 2:
            _fail("E003", f"request {index} must be (kind, name)")
        kind = items[0]
        name = _require_name(items[1], "request name")
        if kind == "read":
            if not writer_active and waiting_writers == 0:
                active_readers += 1
                admitted.append(name)
            else:
                deferred.append(name)
        elif kind == "write":
            if not writer_active and active_readers == 0:
                writer_active = True
                admitted.append(name)
            else:
                waiting_writers += 1
                deferred.append(name)
        else:
            _fail("E003", f"invalid request kind {kind!r}")
    return {
        "admitted": tuple(admitted),
        "deferred": tuple(deferred),
        "active_readers": active_readers,
        "writer_active": writer_active,
        "waiting_writers": waiting_writers,
    }


def csync_spin_tick(
    probe: Callable[[int], bool],
    *,
    max_ticks: int,
    clock: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Poll *probe* up to *max_ticks* times using injected tick values.

    The loop always terminates because the bound is explicit; no sleeping or
    busy waiting on wall-clock time occurs.  A failing probe is reported, not
    raised, so callers can decide how to escalate.
    """
    check = _require_callable(probe, "probe")
    limit = _require_int(max_ticks, "max_ticks", 0)
    if clock is not None:
        _require_callable(clock, "clock")
    ticks: list[int] = []
    ok = False
    for index in range(limit):
        tick = index if clock is None else clock()
        ticks.append(tick)
        if check(tick):
            ok = True
            break
    return {
        "ok": ok,
        "ticks": len(ticks),
        "tick_values": tuple(ticks),
        "last": ticks[-1] if ticks else None,
    }


def csync_timeout_remaining(deadline: float, *, now: float) -> dict[str, Any]:
    """Compute remaining time to a deadline from an injected clock value."""
    end = _require_number(deadline, "deadline")
    current = _require_number(now, "now")
    remaining = max(0.0, end - current)
    return {
        "remaining": remaining,
        "expired": remaining == 0.0,
        "deadline": end,
        "now": current,
    }


def csync_poison_state(
    state: Mapping[str, Any],
    *,
    reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a copy of *state* marked poisoned with a structured reason."""
    original = _require_mapping(state, "state")
    message = _require_name(reason, "reason")
    poison = {"reason": message, "error": error, "owner": original.get("owner")}
    updated = dict(original)
    updated["poisoned"] = True
    updated["poison"] = poison
    return {"state": updated, "poison": poison, "poisoned": True}


def csync_check_poison(
    state: Mapping[str, Any],
    *,
    raise_on_poison: bool = False,
) -> dict[str, Any]:
    """Inspect a state record for poison and optionally raise E004."""
    mapping = _require_mapping(state, "state")
    poison = mapping.get("poison")
    reports: tuple[dict[str, Any], ...] = ()
    if mapping.get("poisoned"):
        report = _require_mapping(poison or {}, "poison")
        reports = (
            {
                "reason": report.get("reason", "poisoned"),
                "error": report.get("error"),
                "owner": report.get("owner"),
            },
        )
    if reports and raise_on_poison:
        _fail("E004", "poisoned state detected", reason=str(reports[0]["reason"]))
    return {"poisoned": bool(reports), "reports": reports, "clean": not reports}


def csync_atomic_compare_expect(
    cell: dict[str, Any],
    *,
    expect: Any,
    update: Any,
) -> dict[str, Any]:
    """Perform one compare-and-set on a cell dict (E003 for malformed cells)."""
    if not isinstance(cell, dict) or "value" not in cell:
        _fail("E003", "cell must be a dict with a 'value' key")
    before = cell["value"]
    swapped = before == expect
    if swapped:
        cell["value"] = update
    return {
        "swapped": swapped,
        "before": before,
        "after": cell["value"],
        "expected": expect,
        "update": update,
    }


def csync_counter_snapshot(
    deltas: Iterable[int],
    *,
    initial: int = 0,
) -> dict[str, Any]:
    """Apply integer deltas and record a checkpoint after every change."""
    values = [item for item in deltas]
    start = _require_int(initial, "initial", 0)
    total = start
    checkpoints: list[dict[str, Any]] = []
    applied: list[int] = []
    for index, value in enumerate(values):
        delta = _require_int(value, f"delta {index}", -(2**63))
        total += delta
        applied.append(delta)
        checkpoints.append({"index": index, "delta": delta, "value": total})
    return {
        "value": total,
        "deltas": tuple(applied),
        "checkpoints": tuple(checkpoints),
        "changes": len(applied),
    }


@contextmanager
def csync_owned_lock(
    name: str,
    *,
    lock: threading.Lock | None = None,
    held: set[str] | None = None,
) -> Any:
    """Yield an owned lock envelope that releases on exit, even on a raise.

    Re-entering the same owner name while it is held raises E003, which keeps
    nested ownership bugs visible instead of silently succeeding.
    """
    _require_name(name, "name")
    if lock is None:
        lock = threading.Lock()
    if held is None:
        held = set()
    if name in held:
        _fail("E003", f"lock {name!r} is already held by this owner")
    lock.acquire()
    held.add(name)
    envelope = {"name": name, "owner": threading.get_ident(), "held": True}
    try:
        yield envelope
    finally:
        envelope["held"] = False
        held.discard(name)
        lock.release()


def csync_fair_queue(
    events: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Simulate a FIFO ticket queue with explicit enqueue/dequeue events."""
    steps = _require_sequence(events, "events")
    queue: deque[tuple[int, str]] = deque()
    next_ticket = 0
    served: list[str] = []
    missed: list[str] = []
    for index, event in enumerate(steps):
        items = _require_sequence(event, f"event {index}")
        if len(items) != 2:
            _fail("E003", f"event {index} must be (operation, name)")
        operation, name = items[0], _require_name(items[1], "name")
        if operation == "enqueue":
            queue.append((next_ticket, name))
            next_ticket += 1
        elif operation == "dequeue":
            if queue:
                served.append(queue.popleft()[1])
            else:
                missed.append(name)
        else:
            _fail("E003", f"invalid queue operation {operation!r}")
    return {
        "served": tuple(served),
        "waiting": tuple(name for _, name in queue),
        "missed": tuple(missed),
        "next_ticket": next_ticket,
    }


def csync_deadlock_probe(
    waits: Mapping[str, str | None],
    *,
    raise_on_cycle: bool = True,
) -> dict[str, Any]:
    """Detect wait-for cycles among named threads or locks (E005 on cycle)."""
    mapping = _require_mapping(waits, "waits")
    adjacency: dict[str, tuple[str, ...]] = {}
    for name, dependency in mapping.items():
        _require_name(name, "waiter name")
        if dependency is not None:
            _require_name(dependency, "dependency name")
        adjacency[name] = () if dependency is None else (dependency,)
    cycle = _cycle_path(adjacency)
    if cycle is not None and raise_on_cycle:
        _fail("E005", "wait-for cycle detected", reason=" -> ".join(cycle))
    return {
        "cycle": cycle,
        "acyclic": cycle is None,
        "blocked": tuple(
            sorted(name for name, value in mapping.items() if value is not None)
        ),
        "nodes": tuple(sorted(adjacency)),
    }


def csync_rw_lock_run(
    readers: Mapping[str, Callable[[], Any]],
    writers: Mapping[str, Callable[[], Any]],
    *,
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Run named readers and writers with serialized writer execution.

    All worker threads belong to one owned group that is joined before the
    function returns.  Writer serialization uses a real ``threading.Lock``;
    reader outcomes are captured without imposing an artificial global order.
    """
    reading = _require_mapping(readers, "readers")
    writing = _require_mapping(writers, "writers")
    names = (
        *_name_tuple(reading, "reader names"),
        *_name_tuple(writing, "writer names"),
    )
    if len(set(names)) != len(names):
        _fail("E003", "reader and writer names must be distinct")
    jobs: dict[str, Callable[[], Any]] = {**reading, **writing}
    for name in names:
        _require_callable(jobs[name], f"job {name!r}")
    gate = threading.Lock()
    with _owned_threads("csync-rw", join_timeout) as owner:
        for name in sorted(reading):
            owner.start(name, reading[name])
        for name in sorted(writing):
            owner.start(name, writing[name], before=gate.acquire, after=gate.release)
    report = _outcome_report(owner, names)
    return {
        **report,
        "readers": tuple(sorted(reading)),
        "writers": tuple(sorted(writing)),
    }


def csync_condition_signal_run(
    waiters: Mapping[str, Callable[[], Any]],
    *,
    event: threading.Event | None = None,
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Start waiter threads, signal them cooperatively and join them all.

    The event is set immediately after all threads start, so no real sleeping
    is needed.  A caller-supplied event may be pre-set to model a missed wakeup.
    """
    mapping = _require_mapping(waiters, "waiters")
    names = _name_tuple(mapping, "waiter names")
    for name in names:
        _require_callable(mapping[name], f"waiter {name!r}")
    signal = threading.Event() if event is None else event
    if not isinstance(signal, threading.Event):
        _fail("E003", "event must be a threading.Event or None")
    with _owned_threads("csync-signal", join_timeout) as owner:
        for name in names:
            owner.start(
                name,
                mapping[name],
                before=lambda signal=signal, t=join_timeout: signal.wait(t),
            )
        signal.set()
    report = _outcome_report(owner, names)
    return {**report, "released": names, "signalled": signal.is_set()}


def csync_lock_audit(guards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit only guard records produced by this module's model.

    No OS handles are inspected.  Records missing a name or release state raise
    E003 so foreign objects cannot silently pollute the report.
    """
    held: list[str] = []
    released: list[str] = []
    count = 0
    for index, item in enumerate(guards):
        record = _require_mapping(item, f"guard {index}")
        name = _require_name(record.get("name"), f"guard {index} name")
        state = record.get("held", record.get("released"))
        if state is None:
            _fail("E003", f"guard {name!r} has no release state")
        (held if state else released).append(name)
        count += 1
    return {
        "total": count,
        "held": tuple(sorted(held)),
        "released": tuple(sorted(released)),
        "clean": not held,
    }


def csync_barrier_thread_run(
    participants: Mapping[str, Callable[[], Any]],
    *,
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Release a real barrier from every worker, even if its body raises.

    Each thread calls ``barrier.wait`` in a ``finally`` block so one failing
    participant cannot strand the others.  All threads are joined on exit.
    """
    mapping = _require_mapping(participants, "participants")
    names = _name_tuple(mapping, "participant names")
    for name in names:
        _require_callable(mapping[name], f"participant {name!r}")
    if not names:
        return {"parties": 0, "values": {}, "failures": (), "threads_joined": True}
    barrier = threading.Barrier(len(names), timeout=join_timeout)
    with _owned_threads("csync-barrier", join_timeout) as owner:
        for name in names:
            owner.start(
                name,
                mapping[name],
                after=lambda barrier=barrier, t=join_timeout: barrier.wait(t),
            )
    return {**_outcome_report(owner, names), "parties": len(names)}


def csync_guard_context_report(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize guard open/close events into a deterministic report."""
    steps = _require_sequence(events, "events")
    opened: set[str] = set()
    closed: set[str] = set()
    sequence: list[dict[str, Any]] = []
    for index, item in enumerate(steps):
        record = _require_mapping(item, f"event {index}")
        name = _require_name(record.get("name"), f"event {index} name")
        action = record.get("action")
        if action not in {"open", "close"}:
            _fail("E003", f"invalid guard action {action!r}")
        if action == "open":
            if name in opened:
                _fail("E003", f"guard {name!r} opened twice")
            opened.add(name)
        else:
            if name not in opened:
                _fail("E006", f"guard {name!r} closed without opening")
            closed.add(name)
        sequence.append({"index": index, "name": name, "action": action})
    return {
        "opened": tuple(sorted(opened)),
        "closed": tuple(sorted(closed)),
        "leaks": tuple(sorted(opened - closed)),
        "balanced": opened == closed,
        "sequence": tuple(sequence),
    }
