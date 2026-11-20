"""Behavioural tests for FRAME 7 (concurrency, async and resource lifecycles).

All workloads are small and bounded; every thread, executor, channel and
wrapper is explicitly joined or closed in a ``finally`` block so the suite
leaves no background activity behind.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor

import pytest

from runtime.concurrency.framework import (
    BoundedChannel,
    CancellationToken,
    ChannelClosedError,
    ConcurrencyError,
    FutureTimeoutError,
    GuardedLock,
    HandleRegistry,
    PoisonedStateError,
    PoisonReport,
    PollResult,
    PoolDrainingError,
    ScopedTaskError,
    TaskScope,
    WorkerPool,
    conc_acquire_lock_ordered,
    conc_audit_open_handles,
    conc_cancel_token_propagate,
    conc_channel_bounded_backpressure,
    conc_detect_poisoned_state,
    conc_reactor_poll_wakeup,
    conc_release_guard_on_drop,
    conc_spawn_scoped_task,
    conc_timeout_future_cleanly,
    conc_worker_pool_graceful_drain,
)


def test_spawn_scoped_task_waits_for_all_and_reports_failures() -> None:
    counter = {"value": 0}
    guard = threading.Lock()

    def work(value: int) -> int:
        with guard:
            counter["value"] += 1
        return value * 2

    scope = TaskScope(max_workers=3, name="jobs")
    try:
        futures = [
            conc_spawn_scoped_task(scope, f"job-{index}", work, index)
            for index in range(12)
        ]
        assert [future.result(timeout=5) for future in futures] == [
            index * 2 for index in range(12)
        ]
        assert scope.pending == 0
    finally:
        scope.close()
    assert counter["value"] == 12
    with pytest.raises(ConcurrencyError, match="closed"):
        conc_spawn_scoped_task(scope, "late", work, 0)

    def boom() -> None:
        raise ValueError("task failed")

    with (
        pytest.raises(ScopedTaskError) as info,
        TaskScope(max_workers=2, name="failing") as failing,
    ):
        conc_spawn_scoped_task(failing, "boom", boom)
    assert info.value.failures[0].name == "boom"
    assert isinstance(info.value.failures[0].error, ValueError)


def test_acquire_lock_ordered_avoids_deadlock_and_rolls_back() -> None:
    first = threading.Lock()
    second = threading.Lock()
    errors: list[BaseException] = []
    orders: list[tuple[str, ...]] = []
    stop = threading.Event()

    def worker(seed: int) -> None:
        try:
            for index in range(25):
                if stop.is_set():
                    return
                mapping = (
                    {"alpha": first, "beta": second}
                    if (seed + index) % 2 == 0
                    else {"beta": second, "alpha": first}
                )
                with conc_acquire_lock_ordered(mapping, timeout=2) as order:
                    orders.append(order)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=worker, args=(seed,), name=f"f7-lock-{seed}")
        for seed in range(6)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        stop.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert orders and set(orders) == {("alpha", "beta")}

    held = threading.Lock()
    assert held.acquire(blocking=False)
    spare = threading.Lock()
    try:
        with (
            pytest.raises(ConcurrencyError, match="lock order timeout"),
            conc_acquire_lock_ordered({"alpha": held, "beta": spare}, timeout=0.02),
        ):
            pytest.fail("acquisition should have timed out")
        assert spare.acquire(blocking=False)
        spare.release()
    finally:
        held.release()


def test_release_guard_on_drop_releases_once_even_on_exception() -> None:
    lock = threading.Lock()
    assert lock.acquire(blocking=False)
    guard = conc_release_guard_on_drop(lock, name="lock-guard")
    try:
        with pytest.raises(ValueError, match="body"), guard as acquired:
            assert acquired is lock
            assert not lock.acquire(blocking=False)
            raise ValueError("body")
        assert lock.acquire(blocking=False)
        lock.release()
        assert guard.released
        assert guard.release() is False
    finally:
        if not guard.released:
            guard.release()

    calls: list[object] = []
    reservation = object()
    custom = conc_release_guard_on_drop(
        reservation, lambda obj: calls.append(obj), name="reservation"
    )
    try:
        assert custom.release() is True
        assert calls == [reservation]
        assert custom.release() is False
    finally:
        custom.release()


def test_channel_bounded_backpressure_and_high_water() -> None:
    with conc_channel_bounded_backpressure(2, name="tiny") as channel:
        assert channel.put("a")
        assert channel.put("b")
        assert channel.size == 2
        assert channel.put("c", timeout=0.02) is False
        assert channel.get(timeout=1) == "a"
        assert channel.put("c", timeout=1)
        assert channel.get(timeout=1) == "b"
        assert channel.get(timeout=1) == "c"
        assert channel.stats.high_water == 2

    total = 40
    errors: list[BaseException] = []
    received: list[int] = []
    with conc_channel_bounded_backpressure(2, name="stress") as channel:

        def consumer() -> None:
            try:
                for _ in range(total):
                    received.append(channel.get(timeout=5))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=consumer, name="f7-channel-consumer")
        thread.start()
        try:
            for value in range(total):
                assert channel.put(value, timeout=5)
        finally:
            thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
        assert sorted(received) == list(range(total))
        stats = channel.stats
        assert stats.high_water <= 2
        assert stats.accepted == total
        assert stats.delivered == total


def test_channel_close_wakes_blocked_producer() -> None:
    channel = BoundedChannel[int](1, name="close-wake")
    errors: list[BaseException] = []
    closed_seen = threading.Event()

    def producer() -> None:
        try:
            channel.put(1, timeout=2)
            try:
                channel.put(2, timeout=5)
            except ChannelClosedError:
                closed_seen.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=producer, name="f7-channel-producer")
    thread.start()
    try:
        assert channel.get(timeout=2) == 1
        channel.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert closed_seen.is_set()
        assert errors == []
        with pytest.raises(ChannelClosedError):
            channel.get(timeout=1)
    finally:
        channel.close()
        if thread.is_alive():
            thread.join(timeout=5)


def test_cancel_token_propagates_to_child_and_queued_futures() -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="f7-token")
    release = threading.Event()
    started = threading.Event()

    def blocker() -> None:
        started.set()
        release.wait(5)

    try:
        active = executor.submit(blocker)
        assert started.wait(2)
        root = CancellationToken(name="root")
        queued = [executor.submit(lambda: 42) for _ in range(3)]
        child = conc_cancel_token_propagate(root, futures=queued, name="child")
        assert not child.is_cancelled
        assert root.cancel() is True
        assert root.cancel() is False
        assert child.is_cancelled
        assert all(future.cancelled() for future in queued)
        with pytest.raises(CancelledError):
            child.raise_if_cancelled()
        late = conc_cancel_token_propagate(root, name="late")
        assert late.is_cancelled
        release.set()
        active.result(timeout=5)
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_timeout_future_cleanly_stops_cooperative_job() -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="f7-timeout")
    release = threading.Event()
    try:
        settled = executor.submit(lambda: "done")
        assert conc_timeout_future_cleanly(settled, timeout=1) == "done"

        blocking = executor.submit(lambda: release.wait(5))
        with pytest.raises(FutureTimeoutError) as info:
            conc_timeout_future_cleanly(
                blocking,
                timeout=0.02,
                on_timeout=lambda _future: release.set(),
                cancel_grace=2.0,
            )
        assert info.value.timed_out is True
        assert info.value.stopped is True
        assert blocking.done()
        assert blocking.result(timeout=2) is True
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_reactor_poll_wakeup_on_work_and_shutdown() -> None:
    results: list[PollResult] = []
    work_seen = threading.Event()
    loop_started = threading.Event()

    with conc_reactor_poll_wakeup() as reactor:
        assert reactor.poll(timeout=0.01).reason == "timeout"

        def poller() -> None:
            loop_started.set()
            while True:
                result = reactor.poll(timeout=2)
                results.append(result)
                if result.reason == "work":
                    work_seen.set()
                elif result.reason == "shutdown":
                    return

        thread = threading.Thread(target=poller, name="f7-reactor-poller")
        thread.start()
        try:
            assert loop_started.wait(2)
            reactor.notify_work()
            assert work_seen.wait(2)
            reactor.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
        finally:
            reactor.shutdown()
            if thread.is_alive():
                thread.join(timeout=5)
    assert any(item.reason == "work" and item.pending_work >= 1 for item in results)
    assert results[-1].reason == "shutdown"
    assert results[-1].shutdown is True


def test_worker_pool_graceful_drain_completes_jobs_and_joins_workers() -> None:
    pool = WorkerPool(max_workers=3, name="drain")

    def job(value: int) -> int:
        return value * value

    def failing() -> None:
        raise ValueError("job panicked")

    try:
        futures = [pool.submit(f"job-{index}", job, index) for index in range(12)]
        pool.submit("broken", failing)
        report = conc_worker_pool_graceful_drain(pool)
        assert report.submitted == 13
        assert report.completed == 12
        assert tuple(item.name for item in report.failed) == ("broken",)
        assert report.failed[0].error.startswith("ValueError:")
        assert report.all_done and not report.timed_out
        assert report.cancelled == 0
        assert all(future.done() for future in futures)
        with pytest.raises(PoolDrainingError):
            pool.submit("late", job, 99)
        assert not any(
            thread.name.startswith("f7-pool-drain") for thread in threading.enumerate()
        )
    finally:
        if not pool.closed:
            conc_worker_pool_graceful_drain(pool, cancel_pending=True)


def test_detect_poisoned_state_reports_lock_and_worker_panics() -> None:
    lock = GuardedLock(name="shared")
    with pytest.raises(ZeroDivisionError), lock:
        _ = 1 / 0
    reports = conc_detect_poisoned_state(lock, raise_on_poison=False)
    assert reports and reports[0].kind == "lock"
    assert any(
        "ZeroDivisionError" in detail for report in reports for detail in report.details
    )
    with pytest.raises(PoisonedStateError) as info:
        conc_detect_poisoned_state(lock)
    assert info.value.reports[0].identity == "shared"

    poisoned = GuardedLock(name="explicit")
    poisoned.poison("checksum mismatch")
    assert conc_detect_poisoned_state(poisoned, raise_on_poison=False)[0].reason == (
        "checksum mismatch"
    )

    ident_holder: list[int] = []

    def identify() -> None:
        ident_holder.append(threading.get_ident())

    thread = threading.Thread(target=identify, name="f7-ident")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    stale = GuardedLock(name="stale")
    stale._owner = ident_holder[0]
    assert any(
        "no longer alive" in report.reason
        for report in conc_detect_poisoned_state(stale, raise_on_poison=False)
    )

    def panic() -> None:
        raise RuntimeError("worker panic")

    pool = WorkerPool(max_workers=2, name="poison-pool")
    try:
        pool.submit("panicking-job", panic)
        report = conc_worker_pool_graceful_drain(pool)
        assert report.failed
        with pytest.raises(PoisonedStateError) as worker_info:
            conc_detect_poisoned_state(pool)
        assert worker_info.value.reports[0].kind == "worker"
        assert "worker panic" in worker_info.value.reports[0].reason
        assert isinstance(worker_info.value.reports[0], PoisonReport)
    finally:
        if not pool.closed:
            conc_worker_pool_graceful_drain(pool, cancel_pending=True)


def test_audit_open_handles_detects_and_closes_every_wrapper() -> None:
    registry = HandleRegistry()
    try:
        with registry.track_tempdir(prefix="f7-audit-") as temp_path:
            file_path = os.path.join(temp_path, "sample.txt")
            with registry.track_file(file_path, mode="w") as handle:
                handle.write("ok")
            report = conc_audit_open_handles(registry)
            assert report.total == 2
            assert report.closed_count == 1
            assert report.open_count == 1
            assert not report.clean
            assert os.path.isfile(file_path)
        assert not os.path.exists(temp_path)
        report = conc_audit_open_handles(registry)
        assert report.total == 2
        assert report.closed_count == 2
        assert report.clean

        class FakeSocket:
            def __init__(self) -> None:
                self.closed = False
                self.close_calls = 0

            def close(self) -> None:
                self.closed = True
                self.close_calls += 1

        fake = FakeSocket()
        tracked = registry.track_socket_like(fake, name="fake-socket")
        report = conc_audit_open_handles(registry)
        assert report.open_count == 1
        assert report.open[0].kind == "socket"
        tracked.close()
        assert fake.close_calls == 1
        assert conc_audit_open_handles(registry).clean

        leftover = registry.track_tempdir(prefix="f7-leak-")
        assert conc_audit_open_handles(registry).open_count == 1
        assert conc_audit_open_handles(registry, close_all=True).clean
        assert not os.path.exists(leftover.resource)
    finally:
        assert registry.cleanup_all() == ()
