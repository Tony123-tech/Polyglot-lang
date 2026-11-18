"""A checkpointable readiness/timer reactor.

The native backend owns a single-threaded ``mio`` reactor: futures register
wakers for timer deadlines and file-descriptor readiness.  This module mirrors
that model for the distributed runtime.

Two migration rules matter:

* **Timers** are logical: a pending timer is snapshotted as a *remaining
  duration* plus a label.  The target re-arms it, so clock skew between hosts
  cannot make a timer fire in the past or sleep twice as long.
* **File descriptors are not portable.**  A live socket is owned by a
  :class:`distributed.connections.ConnectionAnchor`; what moves is a channel
  handle.  A reactor snapshot that contains raw descriptors therefore refuses
  to restore unless the caller supplies an explicit resolver (for example,
  re-attaching an anchored socket on the same node).  The alternative would be
  a silent descriptor reuse bug.

This is the Python-side counterpart of the generated Rust reactor; it is used
by the cooperative task pump and by tests that need deterministic timers.
"""

from __future__ import annotations

import heapq
import selectors
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from distributed.diagnostics import DistributedError, wire_error
from distributed.transmute import decode_state, encode_state

Callback = Callable[..., Any]


@dataclass
class Timer:
    timer_id: int
    deadline: float
    callback: Callback
    label: str = ""
    cancelled: bool = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


@dataclass
class ReactorStats:
    polls: int = 0
    timer_fires: int = 0
    readiness_events: int = 0
    errors: list[str] = field(default_factory=list)


class Reactor:
    """A single-threaded selector with a min-heap of timer deadlines."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        selector: selectors.BaseSelector | None = None,
        max_errors: int = 64,
    ) -> None:
        self.clock = clock or time.monotonic
        self.selector = selector or selectors.DefaultSelector()
        self.max_errors = max_errors
        self.stats = ReactorStats()
        self._timers: list[tuple[float, int, Timer]] = []
        self._sequence = 0
        self._closed = False
        self._labels: dict[str, Timer] = {}

    # -- timers -----------------------------------------------------------
    def add_timer(
        self,
        delay: float,
        callback: Callback,
        *,
        label: str = "",
    ) -> Timer:
        if self._closed:
            raise DistributedError(
                "reactor is closed",
                code="E003",
                note="ReactorClosedError",
            )
        if delay < 0:
            raise DistributedError(
                f"timer delay must be non-negative, not {delay}",
                code="E003",
                note="ReactorProtocolError",
            )
        self._sequence += 1
        timer = Timer(
            timer_id=self._sequence,
            deadline=self.clock() + float(delay),
            callback=callback,
            label=label or f"timer:{self._sequence}",
        )
        heapq.heappush(self._timers, (timer.deadline, timer.timer_id, timer))
        self._labels[timer.label] = timer
        return timer

    def cancel(self, timer: Timer) -> None:
        timer.cancelled = True
        self._labels.pop(timer.label, None)

    def _due(self) -> int:
        fired = 0
        now = self.clock()
        while self._timers and self._timers[0][0] <= now:
            _deadline, _timer_id, timer = heapq.heappop(self._timers)
            if timer.cancelled:
                continue
            self._labels.pop(timer.label, None)
            fired += 1
            self.stats.timer_fires += 1
            self._invoke(timer.callback, ())
        return fired

    # -- readiness --------------------------------------------------------
    def add_reader(
        self,
        fd: int,
        callback: Callback,
        *,
        label: str = "",
    ) -> None:
        self.selector.register(fd, selectors.EVENT_READ, (callback, label))

    def add_writer(
        self,
        fd: int,
        callback: Callback,
        *,
        label: str = "",
    ) -> None:
        self.selector.register(fd, selectors.EVENT_WRITE, (callback, label))

    def modify(
        self,
        fd: int,
        events: int,
        callback: Callback,
        *,
        label: str = "",
    ) -> None:
        self.selector.modify(fd, events, (callback, label))

    def remove(self, fd: int) -> None:
        try:
            self.selector.unregister(fd)
        except KeyError, ValueError:
            return

    def _invoke(self, callback: Callback, args: tuple[Any, ...]) -> None:
        try:
            callback(*args)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if len(self.stats.errors) >= self.max_errors:
                del self.stats.errors[0]
            self.stats.errors.append(message)

    # -- poll loop --------------------------------------------------------
    def poll(self, timeout: float | None = None) -> int:
        if self._closed:
            return 0
        self.stats.polls += 1
        events = self._due()
        wait = timeout
        if self._timers:
            until = max(0.0, self._timers[0][0] - self.clock())
            wait = until if wait is None else min(wait, until)
        if wait is None:
            wait = 0.1
        if wait > 0 or not self.selector.get_map():
            try:
                ready = self.selector.select(wait)
            except OSError as error:
                self.stats.errors.append(f"select: {error}")
                return events
        else:
            ready = self.selector.select(0)
        for key, mask in ready:
            callback, _label = key.data
            self.stats.readiness_events += 1
            self._invoke(callback, (mask,))
        events += len(ready)
        events += self._due()
        return events

    def run_once(self, timeout: float = 0.1) -> int:
        return self.poll(timeout)

    def run_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout: float | None = None,
        tick: float = 0.05,
    ) -> bool:
        deadline = None if timeout is None else self.clock() + timeout
        while not predicate():
            if deadline is not None and self.clock() >= deadline:
                return False
            self.poll(tick)
        return True

    # -- checkpoint -------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        timers = []
        for _deadline, _timer_id, timer in sorted(self._timers):
            if timer.cancelled:
                continue
            timers.append(
                {
                    "label": timer.label,
                    "remaining": max(0.0, timer.deadline - now),
                }
            )
        readers = []
        writers = []
        for key in self.selector.get_map().values():
            callback, label = key.data
            del callback
            entry = {"fd": key.fd, "label": label}
            if key.events & selectors.EVENT_READ:
                readers.append(entry)
            if key.events & selectors.EVENT_WRITE:
                writers.append(entry)
        return {"timers": timers, "readers": readers, "writers": writers}

    def checkpoint(self) -> bytes:
        return encode_state(self.snapshot())

    def restore(
        self,
        payload: Mapping[str, Any],
        *,
        callbacks: Mapping[str, Callback] | None = None,
        resolver: Callable[[str], int | None] | None = None,
    ) -> None:
        """Re-arm timers, and descriptors only through an explicit resolver.

        Validation happens before any state changes, so a rejected checkpoint
        (missing timer callback or non-portable descriptor) leaves the reactor
        untouched.
        """
        timer_callbacks = dict(callbacks or {})
        timers: list[tuple[float, str, Callback]] = []
        for entry in payload.get("timers", []):
            label = str(entry["label"])
            callback = timer_callbacks.get(label)
            if callback is None:
                raise DistributedError(
                    f"no callback registered for timer {label!r}",
                    code="E006",
                    note="MissingTimerError",
                    help_text=(
                        "pass callbacks={'label': callable} when restoring a "
                        "reactor checkpoint"
                    ),
                    context={"label": label},
                )
            timers.append((float(entry["remaining"]), label, callback))

        readers = list(payload.get("readers", []))
        writers = list(payload.get("writers", []))
        raw = readers + writers
        if raw and resolver is None:
            raise DistributedError(
                "reactor checkpoint contains raw file descriptors; they "
                "cannot be transported across nodes",
                code="E007",
                note="UnsupportedNativeTypeError",
                help_text=(
                    "own sockets on a ConnectionAnchor and migrate a "
                    "ChannelHandle, or pass resolver= to re-attach locally"
                ),
                context={
                    "descriptors": [entry["fd"] for entry in raw],
                },
            )

        for remaining, label, callback in timers:
            self.add_timer(remaining, callback, label=label)

        if resolver is None:
            return
        for entry in readers:
            fd = resolver(entry["label"])
            if fd is not None:
                self.add_reader(fd, _noop, label=entry["label"])
        for entry in writers:
            fd = resolver(entry["label"])
            if fd is not None:
                self.add_writer(fd, _noop, label=entry["label"])

    def restore_checkpoint(
        self,
        data: bytes,
        *,
        callbacks: Mapping[str, Callback] | None = None,
        resolver: Callable[[str], int | None] | None = None,
    ) -> None:
        try:
            payload = decode_state(data)
        except DistributedError as error:
            raise wire_error(
                f"reactor checkpoint is malformed: {error.diagnostic.message}",
                reason=error.diagnostic.reason,
            ) from error
        if not isinstance(payload, dict):
            raise wire_error("reactor checkpoint must decode to a mapping")
        self.restore(payload, callbacks=callbacks, resolver=resolver)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self._closed = True
        self._timers.clear()
        self._labels.clear()
        self.selector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _noop(*_args: Any) -> None:
    return None


class TaskPump:
    """Drive a checkpointable task scheduler and a reactor together."""

    def __init__(
        self,
        reactor: Reactor,
        scheduler: Any,
        *,
        tick: float = 0.01,
    ) -> None:
        self.reactor = reactor
        self.scheduler = scheduler
        self.tick = tick

    def step(self) -> list[Any]:
        self.reactor.poll(0)
        return self.scheduler.step()

    def run(
        self,
        *,
        predicate: Callable[[], bool] | None = None,
        timeout: float | None = None,
    ) -> bool:
        deadline = None if timeout is None else self.reactor.clock() + float(timeout)
        while True:
            self.step()
            if predicate is not None:
                if predicate():
                    return True
            elif not self.scheduler.pending():
                return True
            if deadline is not None and self.reactor.clock() >= deadline:
                return False
            self.reactor.poll(self.tick)


__all__ = [
    "Callback",
    "Reactor",
    "ReactorStats",
    "TaskPump",
    "Timer",
]
