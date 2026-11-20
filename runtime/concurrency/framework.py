"""FRAME 7: Concurrency, Async & Resource Lifecycles.

This module provides ten deterministic, stdlib-only concurrency primitives.
The implementations are *portable models*: threads, events, conditions and
wrapper registries stand in for OS-specific reactors, file descriptors and
cancellation primitives.  Consequently:

* cancellation is cooperative -- a running thread cannot be forcibly killed;
  callers supply a hook (an ``Event.set``, a socket shutdown, ...) that lets
  the job finish quickly;
* every helper that starts a thread owns a ``close``/``shutdown`` path that
  joins it, and every context manager closes even when its body raises;
* ``conc_audit_open_handles`` tracks only resources wrapped by this module.
  Real sockets are accepted as duck-typed ``close()`` resources but are never
  enumerated from the OS, so no ``psutil``/``resource`` probing is performed.

No task, worker or waiter is intentionally left running after a context exits.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, Literal, Protocol, Self


class ConcurrencyError(RuntimeError):
    """Base error for lifecycle and concurrency invariant violations."""


class LockLike(Protocol):
    """Structural type accepted by the ordered-lock acquisition helper."""

    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool:
        """Acquire the lock, optionally with a timeout."""
        ...

    def release(self) -> None:
        """Release a previously acquired lock."""
        ...


# ---------------------------------------------------------------------------
# 61. Scoped task spawning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskFailure:
    """A named failure collected while closing a :class:`TaskScope`."""

    name: str
    error: BaseException


class ScopedTaskError(ConcurrencyError):
    """Aggregate error raised by a scope whose tasks failed."""

    def __init__(self, failures: Sequence[TaskFailure]) -> None:
        self.failures: tuple[TaskFailure, ...] = tuple(failures)
        summary = ", ".join(
            f"{item.name}: {type(item.error).__name__}({item.error})"
            for item in self.failures
        )
        super().__init__(f"scoped tasks failed: {summary}")


class TaskScope:
    """Explicit lifetime boundary: no submitted task outlives ``close()``.

    The scope owns a joinable thread pool.  Closing cancels work that has not
    started, waits for running work, records failures and shuts the pool down.
    """

    def __init__(self, max_workers: int = 4, *, name: str = "scope") -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.name = name
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"f7-scope-{name}"
        )
        self._lock = threading.Lock()
        self._tasks: list[tuple[str, Future[Any]]] = []
        self._failures: list[TaskFailure] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the scope has been closed for new submissions."""
        return self._closed

    @property
    def failures(self) -> tuple[TaskFailure, ...]:
        """Failures observed when the scope was closed."""
        return tuple(self._failures)

    @property
    def pending(self) -> int:
        """Number of submitted tasks that have not finished yet."""
        with self._lock:
            return sum(1 for _, future in self._tasks if not future.done())

    def __enter__(self) -> Self:
        """Enter the scope; the same object is returned."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the scope, then surface task failures if the body succeeded."""
        self.close()
        if exc_type is None and self._failures:
            raise ScopedTaskError(self._failures)

    def close(self) -> None:
        """Cancel queued tasks, join running tasks and stop the executor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks)
        for _, future in tasks:
            future.cancel()
        wait([future for _, future in tasks])
        for name, future in tasks:
            if future.cancelled():
                continue
            error = future.exception()
            if error is not None:
                self._failures.append(TaskFailure(name, error))
        self._executor.shutdown(wait=True, cancel_futures=True)


def conc_spawn_scoped_task[T](
    scope: TaskScope,
    name: str,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> Future[T]:
    """Submit ``fn`` to ``scope``; the scope joins it when the scope exits.

    The returned future exposes the task result, but lifetime management stays
    with the scope: late submissions are rejected and queued work is cancelled
    on close rather than being orphaned.
    """
    if not isinstance(scope, TaskScope):
        raise TypeError("scope must be a TaskScope")
    if not name:
        raise ValueError("task name must be non-empty")
    with scope._lock:
        if scope._closed:
            raise ConcurrencyError(f"scope {scope.name!r} is closed")
        future: Future[T] = scope._executor.submit(fn, *args, **kwargs)
        scope._tasks.append((name, future))
    return future


# ---------------------------------------------------------------------------
# 62. Deterministic lock ordering
# ---------------------------------------------------------------------------


@contextmanager
def conc_acquire_lock_ordered(
    locks: Mapping[str, LockLike],
    *,
    timeout: float | None = None,
) -> Iterator[tuple[str, ...]]:
    """Acquire all ``locks`` in sorted-name order and release in reverse.

    A global order (lexicographic by key) removes the circular-wait condition
    as long as every participant uses this helper.  On any acquisition failure
    the locks already held are released before the error propagates.
    """
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be >= 0 or None")
    ordered = tuple(sorted(locks))
    acquired: list[LockLike] = []
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        for key in ordered:
            lock = locks[key]
            if deadline is None:
                ok = lock.acquire()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConcurrencyError(f"lock order timeout at {key!r}")
                ok = lock.acquire(True, remaining)
            if not ok:
                if deadline is None:
                    raise ConcurrencyError(f"could not acquire lock {key!r}")
                raise ConcurrencyError(f"lock order timeout at {key!r}")
            acquired.append(lock)
        yield ordered
    finally:
        for lock in reversed(acquired):
            lock.release()


# ---------------------------------------------------------------------------
# 63. RAII release guard
# ---------------------------------------------------------------------------


class ResourceGuard[T]:
    """Idempotent context-manager guard that releases a resource on exit.

    ``__del__`` is a best-effort fallback for forgotten guards; deterministic
    cleanup must go through ``with`` or :meth:`release`.
    """

    def __init__(
        self,
        resource: T,
        releaser: Callable[[T], None] | None = None,
        *,
        name: str = "resource",
    ) -> None:
        self.name = name
        self._resource = resource
        if releaser is not None:
            self._releaser = releaser
        else:
            method = next(
                (
                    candidate
                    for candidate in ("release", "close", "shutdown", "stop")
                    if callable(getattr(resource, candidate, None))
                ),
                None,
            )
            if method is None:
                raise ValueError(
                    f"no releaser for {name!r} and resource has no release/close method"
                )
            self._releaser = lambda obj: getattr(obj, method)()
        self._lock = threading.Lock()
        self._released = False

    @property
    def resource(self) -> T:
        """The wrapped resource."""
        return self._resource

    @property
    def released(self) -> bool:
        """Whether the guard has already invoked the releaser."""
        return self._released

    def release(self) -> bool:
        """Release exactly once; return ``True`` iff this call did the release."""
        with self._lock:
            if self._released:
                return False
            self._released = True
        self._releaser(self._resource)
        return True

    def __enter__(self) -> T:
        """Return the wrapped resource for use inside a ``with`` block."""
        return self._resource

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Release the resource regardless of the body's outcome."""
        self.release()

    def __del__(self) -> None:
        """Best-effort fallback release (GC timing is implementation-specific)."""
        try:
            self.release()
        except Exception:  # noqa: S110 - destructors must never raise
            pass


def conc_release_guard_on_drop[T](
    resource: T,
    releaser: Callable[[T], None] | None = None,
    *,
    name: str = "resource",
) -> ResourceGuard[T]:
    """Wrap ``resource`` in an RAII guard with deterministic release.

    A releaser can be supplied explicitly for channels and reservations that
    expose no ``release()``/``close()`` method.
    """
    return ResourceGuard(resource, releaser, name=name)


# ---------------------------------------------------------------------------
# 64. Bounded channel with backpressure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    """Immutable counters describing a channel's bounded behaviour."""

    capacity: int
    accepted: int
    delivered: int
    rejected_puts: int
    high_water: int


class ChannelClosedError(ConcurrencyError):
    """Raised when operating on a closed channel."""


class ChannelEmptyError(ConcurrencyError):
    """Raised when no item arrives before a ``get`` timeout."""


class BoundedChannel[T]:
    """Fixed-capacity FIFO with blocking, close-aware put/get operations."""

    def __init__(self, capacity: int, *, name: str = "channel") -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.name = name
        self._cv = threading.Condition()
        self._items: deque[T] = deque()
        self._closed = False
        self._accepted = 0
        self._delivered = 0
        self._rejected_puts = 0
        self._high_water = 0

    @property
    def size(self) -> int:
        """Number of buffered items."""
        with self._cv:
            return len(self._items)

    @property
    def closed(self) -> bool:
        """Whether the channel has been closed."""
        with self._cv:
            return self._closed

    @property
    def stats(self) -> ChannelStats:
        """Snapshot of counters; ``high_water`` never exceeds ``capacity``."""
        with self._cv:
            return ChannelStats(
                self.capacity,
                self._accepted,
                self._delivered,
                self._rejected_puts,
                self._high_water,
            )

    def put(self, item: T, *, timeout: float | None = None) -> bool:
        """Block until capacity is available; return ``False`` on timeout."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0 or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while len(self._items) >= self.capacity and not self._closed:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._rejected_puts += 1
                        return False
                self._cv.wait(remaining)
            if self._closed:
                self._rejected_puts += 1
                raise ChannelClosedError(f"channel {self.name!r} is closed")
            self._items.append(item)
            self._accepted += 1
            self._high_water = max(self._high_water, len(self._items))
            self._cv.notify_all()
            return True

    def get(self, *, timeout: float | None = None) -> T:
        """Pop the oldest item, waiting up to ``timeout`` seconds."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0 or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while not self._items and not self._closed:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ChannelEmptyError(f"channel {self.name!r} is empty")
                self._cv.wait(remaining)
            if self._items:
                item = self._items.popleft()
                self._delivered += 1
                self._cv.notify_all()
                return item
            raise ChannelClosedError(f"channel {self.name!r} is closed and drained")

    def close(self) -> None:
        """Close the channel and wake every blocked producer and consumer."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def __enter__(self) -> Self:
        """Return this channel."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the channel on exit."""
        self.close()


@contextmanager
def conc_channel_bounded_backpressure(
    capacity: int,
    *,
    name: str = "channel",
) -> Iterator[BoundedChannel[Any]]:
    """Create a bounded channel whose lifetime is bound to the ``with`` block."""
    channel: BoundedChannel[Any] = BoundedChannel(capacity, name=name)
    try:
        yield channel
    finally:
        channel.close()


# ---------------------------------------------------------------------------
# 65. Cancellation token propagation
# ---------------------------------------------------------------------------


class CancellationToken:
    """Hierarchical, thread-safe cancellation signal with future linkage."""

    def __init__(
        self,
        *,
        name: str = "root",
        parent: CancellationToken | None = None,
    ) -> None:
        self.name = name
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._children: weakref.WeakSet[CancellationToken] = weakref.WeakSet()
        self._callback_errors: list[BaseException] = []
        self._parent = parent
        if parent is not None:
            parent._register_child(self)

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def callback_errors(self) -> tuple[BaseException, ...]:
        """Errors raised by cancellation callbacks (propagation continues)."""
        with self._lock:
            return tuple(self._callback_errors)

    def _register_child(self, child: CancellationToken) -> None:
        with self._lock:
            already_cancelled = self._event.is_set()
            if not already_cancelled:
                self._children.add(child)
        if already_cancelled:
            child.cancel()

    def cancel(self) -> bool:
        """Cancel this token, its children and linked futures exactly once."""
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            children = tuple(self._children)
            callbacks = tuple(self._callbacks)
        for child in children:
            child.cancel()
        for callback in callbacks:
            try:
                callback()
            except Exception as error:  # one bad callback must not stop the rest
                with self._lock:
                    self._callback_errors.append(error)
        return True

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` now if cancelled, otherwise on cancellation."""
        with self._lock:
            fire_now = self._event.is_set()
            if not fire_now:
                self._callbacks.append(callback)
        if fire_now:
            callback()

    def link_future(self, future: Future[Any]) -> None:
        """Cancel ``future`` when this token is cancelled (if still pending)."""
        self.add_callback(lambda: future.cancel() if not future.done() else None)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation; return whether it happened before timeout."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        """Raise :class:`concurrent.futures.CancelledError` if cancelled."""
        if self._event.is_set():
            raise CancelledError(f"token {self.name!r} was cancelled")


def conc_cancel_token_propagate(
    token: CancellationToken,
    *,
    futures: Iterable[Future[Any]] = (),
    name: str = "child",
) -> CancellationToken:
    """Create a child token of ``token`` that also cancels ``futures``.

    Cancelling the parent cancels the child, and cancelling the child cancels
    every linked pending future.  A child created after cancellation starts
    already cancelled, so late work cannot miss the signal.
    """
    if not isinstance(token, CancellationToken):
        raise TypeError("token must be a CancellationToken")
    child = CancellationToken(name=name, parent=token)
    for future in futures:
        child.link_future(future)
    return child


# ---------------------------------------------------------------------------
# 66. Clean timeouts
# ---------------------------------------------------------------------------


class FutureTimeoutError(ConcurrencyError):
    """Raised when a future did not finish in time; ``stopped`` reports cleanup."""

    def __init__(self, message: str, *, stopped: bool) -> None:
        super().__init__(message)
        self.stopped = stopped
        self.timed_out = True


def conc_timeout_future_cleanly[T](
    future: Future[T],
    *,
    timeout: float,
    on_timeout: Callable[[Future[T]], None] | None = None,
    cancel_grace: float = 0.0,
) -> T:
    """Return ``future``'s result or cancel it cleanly on timeout.

    ``on_timeout`` is the cooperative stop hook (for example setting an Event
    or shutting down a socket).  The pending future is cancelled and the helper
    waits at most ``cancel_grace`` seconds for a running job to acknowledge.
    No descriptors are allocated here, so nothing can be leaked by the helper.
    """
    if timeout < 0:
        raise ValueError("timeout must be >= 0")
    if cancel_grace < 0:
        raise ValueError("cancel_grace must be >= 0")
    done, _ = wait({future}, timeout=timeout)
    if future in done:
        return future.result()
    try:
        if on_timeout is not None:
            on_timeout(future)
    finally:
        cancelled = future.cancel()
    stopped = True
    if not cancelled:
        settled, _ = wait({future}, timeout=cancel_grace)
        stopped = future in settled
    raise FutureTimeoutError(
        f"future timed out after {timeout}s (stopped={stopped})", stopped=stopped
    )


# ---------------------------------------------------------------------------
# 67. Reactor poll/wakeup model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollResult:
    """Outcome of one reactor poll."""

    woke: bool
    reason: Literal["work", "shutdown", "timeout"]
    pending_work: int
    shutdown: bool


class ReactorWakeup:
    """Portable event-loop wakeup: an ``Event`` for shutdown, a condition for work."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._pending = 0
        self._polls = 0

    @property
    def pending_work(self) -> int:
        """Number of pending work notifications."""
        with self._cv:
            return self._pending

    @property
    def is_shutdown(self) -> bool:
        """Whether shutdown has been signalled."""
        return self._stop.is_set()

    @property
    def polls(self) -> int:
        """Number of polls performed (including timeouts)."""
        with self._cv:
            return self._polls

    def notify_work(self, count: int = 1) -> None:
        """Announce new work and wake all sleeping pollers."""
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._cv:
            self._pending += count
            self._cv.notify_all()

    def shutdown(self) -> None:
        """Signal shutdown and wake every poller; idempotent and permanent."""
        with self._cv:
            self._stop.set()
            self._cv.notify_all()

    def drain(self) -> int:
        """Consume and return all pending work notifications."""
        with self._cv:
            count = self._pending
            self._pending = 0
            return count

    def poll(self, *, timeout: float | None = None) -> PollResult:
        """Block until work or shutdown arrives, or ``timeout`` elapses."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0 or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            self._polls += 1
            while True:
                if self._stop.is_set():
                    return PollResult(True, "shutdown", self._pending, True)
                if self._pending > 0:
                    return PollResult(True, "work", self._pending, False)
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return PollResult(False, "timeout", 0, False)
                self._cv.wait(remaining)


@contextmanager
def conc_reactor_poll_wakeup() -> Iterator[ReactorWakeup]:
    """Provide a reactor wakeup model that is shut down on context exit."""
    reactor = ReactorWakeup()
    try:
        yield reactor
    finally:
        reactor.shutdown()


# ---------------------------------------------------------------------------
# 68. Graceful worker-pool drain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobFailure:
    """A failed job's name and formatted exception."""

    name: str
    error: str


@dataclass(frozen=True)
class DrainReport:
    """Structured result of draining a :class:`WorkerPool`."""

    submitted: int
    completed: int
    failed: tuple[JobFailure, ...]
    cancelled: int
    rejected: int
    all_done: bool
    timed_out: bool


class PoolDrainingError(ConcurrencyError):
    """Raised when submitting to a pool that is draining or closed."""


class WorkerPool:
    """Small joinable worker pool with explicit drain state."""

    def __init__(self, max_workers: int = 4, *, name: str = "pool") -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.name = name
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"f7-pool-{name}"
        )
        self._lock = threading.Lock()
        self._jobs: list[tuple[str, Future[Any]]] = []
        self._draining = False
        self._closed = False
        self._rejected = 0
        self._last_report: DrainReport | None = None

    @property
    def draining(self) -> bool:
        """Whether the pool has stopped accepting submissions."""
        return self._draining

    @property
    def closed(self) -> bool:
        """Whether the pool's executor has been shut down."""
        return self._closed

    def submit[T](
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Future[T]:
        """Submit a named job, rejecting new work once draining starts."""
        with self._lock:
            if self._draining:
                self._rejected += 1
                raise PoolDrainingError(f"pool {self.name!r} is draining")
            future: Future[T] = self._executor.submit(fn, *args, **kwargs)
            self._jobs.append((name, future))
            return future

    def __enter__(self) -> Self:
        """Return this pool."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Drain the pool on exit."""
        if not self._closed:
            conc_worker_pool_graceful_drain(self)


def conc_worker_pool_graceful_drain(
    pool: WorkerPool,
    *,
    timeout: float | None = None,
    cancel_pending: bool = False,
) -> DrainReport:
    """Stop intake, drain jobs and join every worker thread.

    By default queued jobs are allowed to finish (``cancel_pending=False``).
    With ``timeout`` set, unfinished jobs are cancelled when the deadline
    passes; ``cancel_pending=True`` cancels queued jobs before waiting.
    """
    if not isinstance(pool, WorkerPool):
        raise TypeError("pool must be a WorkerPool")
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be >= 0 or None")
    with pool._lock:
        if pool._closed and pool._last_report is not None:
            return pool._last_report
        pool._draining = True
        jobs = tuple(pool._jobs)
        rejected = pool._rejected
    if cancel_pending:
        for _, future in jobs:
            future.cancel()
    _, not_done = wait([future for _, future in jobs], timeout=timeout)
    timed_out = bool(not_done)
    if timed_out:
        for future in not_done:
            future.cancel()
    pool._executor.shutdown(wait=True, cancel_futures=True)
    with pool._lock:
        pool._closed = True

    completed = 0
    cancelled = 0
    failures: list[JobFailure] = []
    for name, future in jobs:
        if future.cancelled() or not future.done():
            cancelled += 1
        else:
            error = future.exception()
            if error is None:
                completed += 1
            else:
                failures.append(JobFailure(name, f"{type(error).__name__}: {error}"))
    report = DrainReport(
        submitted=len(jobs),
        completed=completed,
        failed=tuple(failures),
        cancelled=cancelled,
        rejected=rejected,
        all_done=not timed_out,
        timed_out=timed_out,
    )
    pool._last_report = report
    return report


# ---------------------------------------------------------------------------
# 69. Poisoned state detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoisonReport:
    """Structured description of a poisoned lock or panicked worker."""

    kind: Literal["lock", "worker"]
    identity: str
    reason: str
    details: tuple[str, ...] = ()


class PoisonedStateError(ConcurrencyError):
    """Error carrying one or more :class:`PoisonReport` records."""

    def __init__(self, reports: Sequence[PoisonReport]) -> None:
        self.reports: tuple[PoisonReport, ...] = tuple(reports)
        summary = "; ".join(
            f"{item.kind}[{item.identity}]: {item.reason}" for item in self.reports
        )
        super().__init__(f"poisoned state detected: {summary}")


class GuardedLock:
    """RLock plus ownership/failure bookkeeping for poison detection."""

    def __init__(self, *, name: str = "lock") -> None:
        self.name = name
        self._lock = threading.RLock()
        self._owner: int | None = None
        self._poison: str | None = None
        self._failures: list[str] = []

    @property
    def owner(self) -> int | None:
        """Thread ident currently holding the lock, if any."""
        return self._owner

    @property
    def poison_message(self) -> str | None:
        """Explicit poison reason, when set."""
        return self._poison

    def poison(self, reason: str) -> None:
        """Mark the lock poisoned so later detection reports it."""
        self._poison = reason

    def acquire(self, *, timeout: float | None = None) -> bool:
        """Acquire the underlying RLock and record the owning thread ident."""
        if timeout is None:
            ok = self._lock.acquire()
        else:
            ok = self._lock.acquire(True, timeout)
        if ok:
            self._owner = threading.get_ident()
        return ok

    def release(self) -> None:
        """Release the underlying RLock and clear ownership."""
        self._owner = None
        self._lock.release()

    def __enter__(self) -> Self:
        """Acquire the lock for a ``with`` block."""
        if not self.acquire():
            raise ConcurrencyError(f"could not acquire {self.name!r}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Record body exceptions, then release the lock."""
        if exc_type is not None:
            self._failures.append(f"{exc_type.__name__}: {exc}")
        self.release()


def conc_detect_poisoned_state(
    source: GuardedLock | Future[Any] | WorkerPool,
    *,
    name: str | None = None,
    raise_on_poison: bool = True,
) -> tuple[PoisonReport, ...]:
    """Inspect a lock, future or pool and return structured poison reports.

    With ``raise_on_poison=True`` (the default) reports are also raised as a
    :class:`PoisonedStateError`, preventing silent corruption.  Passing
    ``False`` allows callers to inspect reports without raising.
    """
    reports: list[PoisonReport] = []
    if isinstance(source, GuardedLock):
        identity = name or source.name
        if source._poison is not None:
            reports.append(
                PoisonReport("lock", identity, source._poison, tuple(source._failures))
            )
        owner = source._owner
        if owner is not None:
            live = {thread.ident for thread in threading.enumerate()}
            if owner not in live:
                reports.append(
                    PoisonReport(
                        "lock",
                        identity,
                        f"owner thread {owner} is no longer alive",
                    )
                )
        if source._failures:
            reports.append(
                PoisonReport(
                    "lock",
                    identity,
                    "critical section raised an exception",
                    tuple(source._failures),
                )
            )
    elif isinstance(source, Future):
        if source.done() and not source.cancelled():
            error = source.exception()
            if error is not None:
                reports.append(
                    PoisonReport(
                        "worker",
                        name or "future",
                        f"{type(error).__name__}: {error}",
                    )
                )
    elif isinstance(source, WorkerPool):
        with source._lock:
            jobs = tuple(source._jobs)
        for job_name, future in jobs:
            if future.done() and not future.cancelled():
                error = future.exception()
                if error is not None:
                    reports.append(
                        PoisonReport(
                            "worker",
                            name or job_name,
                            f"{type(error).__name__}: {error}",
                        )
                    )
    else:
        raise TypeError("source must be a GuardedLock, Future or WorkerPool")
    result = tuple(reports)
    if result and raise_on_poison:
        raise PoisonedStateError(result)
    return result


# ---------------------------------------------------------------------------
# 70. Open-handle auditing via wrapper registry
# ---------------------------------------------------------------------------

HandleKind = Literal["file", "socket", "tempdir", "generic"]


@dataclass(frozen=True)
class HandleRecord:
    """Snapshot of one tracked handle."""

    handle_id: int
    kind: HandleKind
    name: str
    closed: bool


@dataclass(frozen=True)
class AuditReport:
    """Result of an open-handle audit."""

    total: int
    open: tuple[HandleRecord, ...]
    closed_count: int
    cleanup_errors: tuple[str, ...]

    @property
    def open_count(self) -> int:
        """Number of handles still open at audit time."""
        return len(self.open)

    @property
    def clean(self) -> bool:
        """Whether no handle leaked and no cleanup error occurred."""
        return not self.open and not self.cleanup_errors


class TrackedHandle[T]:
    """Wrapper that closes a resource once and reports to its registry."""

    def __init__(
        self,
        registry: HandleRegistry,
        handle_id: int,
        kind: HandleKind,
        name: str,
        resource: T,
        closer: Callable[[T], None],
    ) -> None:
        self._registry = registry
        self._id = handle_id
        self.kind = kind
        self.name = name
        self._resource = resource
        self._closer = closer
        self._lock = threading.Lock()
        self._closed = False

    @property
    def handle_id(self) -> int:
        """Stable registry id of this handle."""
        return self._id

    @property
    def resource(self) -> T:
        """The wrapped resource."""
        return self._resource

    @property
    def closed(self) -> bool:
        """Whether the wrapper (or the resource itself) is closed."""
        if self._closed:
            return True
        externally_closed = getattr(self._resource, "closed", None)
        return externally_closed is True

    def close(self) -> None:
        """Close the resource exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            closer = self._closer
            resource = self._resource
        closer(resource)

    def __enter__(self) -> T:
        """Return the wrapped resource for use inside a ``with`` block."""
        return self._resource

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the resource on exit, even if the body raised."""
        self.close()


class HandleRegistry:
    """Registry of wrapped files, socket-like objects and temp directories."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handles: dict[int, TrackedHandle[Any]] = {}
        self._errors: dict[int, str] = {}
        self._next_id = 1
        self._temp_root: str | None = None

    def snapshot(self) -> tuple[TrackedHandle[Any], ...]:
        """Return a stable snapshot of all tracked handles."""
        with self._lock:
            return tuple(self._handles.values())

    def track[T](
        self,
        kind: HandleKind,
        name: str,
        resource: T,
        *,
        closer: Callable[[T], None] | None = None,
    ) -> TrackedHandle[T]:
        """Register ``resource`` with an explicit or inferred ``close()``."""
        if closer is None:
            close_method = getattr(resource, "close", None)
            if not callable(close_method):
                raise ValueError(f"no closer for {name!r} and resource has no close()")
            closer = lambda obj: obj.close()
        with self._lock:
            handle_id = self._next_id
            self._next_id += 1
        handle: TrackedHandle[T] = TrackedHandle(
            self, handle_id, kind, name, resource, closer
        )
        with self._lock:
            self._handles[handle_id] = handle
        return handle

    def track_file(
        self,
        path: str | os.PathLike[str],
        *,
        mode: str = "wb",
    ) -> TrackedHandle[IO[Any]]:
        """Open ``path`` and track the resulting file object."""
        resource = open(path, mode)  # noqa: SIM115 - ownership moves to the guard
        return self.track("file", os.fspath(path), resource)

    def track_socket_like[T](self, resource: T, *, name: str) -> TrackedHandle[T]:
        """Track any socket-like object exposing ``close()`` (no socket import)."""
        return self.track("socket", name, resource)

    def _temp_base(self) -> str:
        with self._lock:
            if self._temp_root is None:
                parent = os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"
                self._temp_root = os.path.join(
                    parent, f"polyglot-f7-{os.getpid()}-{id(self)}"
                )
            root = self._temp_root
        os.makedirs(root, exist_ok=True)
        return root

    @staticmethod
    def _remove_tree(path: str) -> None:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    HandleRegistry._remove_tree(entry.path)
                else:
                    os.unlink(entry.path)
        os.rmdir(path)

    def track_tempdir(self, *, prefix: str = "f7-") -> TrackedHandle[str]:
        """Create and track a real temporary directory tree."""
        root = self._temp_base()
        path = os.path.join(root, f"{prefix}{os.getpid()}-{id(self)}")
        os.mkdir(path)
        return self.track("tempdir", path, path, closer=self._remove_tree)

    def _is_closed(self, handle: TrackedHandle[Any]) -> bool:
        return handle.closed or (
            handle.kind == "tempdir" and not os.path.exists(handle.resource)
        )

    def _record_error(self, handle_id: int, message: str) -> None:
        with self._lock:
            self._errors[handle_id] = message

    def cleanup_all(self) -> tuple[str, ...]:
        """Close every still-open tracked handle; return cleanup errors."""
        errors: list[str] = []
        for handle in self.snapshot():
            if self._is_closed(handle):
                continue
            try:
                handle.close()
            except Exception as error:
                message = f"{handle.name}: {type(error).__name__}: {error}"
                errors.append(message)
                self._record_error(handle.handle_id, message)
        with self._lock:
            root = self._temp_root
        if root is not None:
            try:
                os.rmdir(root)
            except OSError:
                pass  # still contains live tempdirs or was already removed
        return tuple(errors)

    def audit(self, *, close_all: bool = False) -> AuditReport:
        """Build an audit report, optionally closing leftovers first."""
        if close_all:
            self.cleanup_all()
        open_records: list[HandleRecord] = []
        closed_count = 0
        with self._lock:
            handles = tuple(self._handles.values())
            errors = tuple(self._errors.values())
        for handle in handles:
            is_closed = self._is_closed(handle)
            if is_closed:
                closed_count += 1
            else:
                open_records.append(
                    HandleRecord(handle.handle_id, handle.kind, handle.name, False)
                )
        return AuditReport(
            total=len(handles),
            open=tuple(open_records),
            closed_count=closed_count,
            cleanup_errors=errors,
        )


DEFAULT_HANDLE_REGISTRY = HandleRegistry()


def conc_audit_open_handles(
    registry: HandleRegistry | None = None,
    *,
    close_all: bool = False,
) -> AuditReport:
    """Audit tracked handles for leaks, optionally force-closing leftovers.

    ``registry`` defaults to the module-level registry.  Detection covers only
    handles created through ``HandleRegistry`` wrappers: no process-wide file
    descriptor or socket enumeration is attempted (platform boundary).
    """
    target = DEFAULT_HANDLE_REGISTRY if registry is None else registry
    if not isinstance(target, HandleRegistry):
        raise TypeError("registry must be a HandleRegistry or None")
    return target.audit(close_all=close_all)


__all__ = [
    "DEFAULT_HANDLE_REGISTRY",
    "AuditReport",
    "BoundedChannel",
    "CancellationToken",
    "ChannelClosedError",
    "ChannelEmptyError",
    "ChannelStats",
    "ConcurrencyError",
    "DrainReport",
    "FutureTimeoutError",
    "GuardedLock",
    "HandleRecord",
    "HandleRegistry",
    "JobFailure",
    "LockLike",
    "PoisonReport",
    "PoisonedStateError",
    "PollResult",
    "PoolDrainingError",
    "ReactorWakeup",
    "ResourceGuard",
    "ScopedTaskError",
    "TaskFailure",
    "TaskScope",
    "TrackedHandle",
    "WorkerPool",
    "conc_acquire_lock_ordered",
    "conc_audit_open_handles",
    "conc_cancel_token_propagate",
    "conc_channel_bounded_backpressure",
    "conc_detect_poisoned_state",
    "conc_reactor_poll_wakeup",
    "conc_release_guard_on_drop",
    "conc_spawn_scoped_task",
    "conc_timeout_future_cleanly",
    "conc_worker_pool_graceful_drain",
]
