"""Contract and behavioral tests for batch 8 (ids 0701-0800).

The ten legacy rows get smoke coverage here (their full suite lives in
``tests.test_framework_f7``).  Every new helper gets at least one behavioral
assertion, and error paths are checked against the shared E001-E008 codes.
Threads started by context managers must be joined by the time the ``with``
block exits; the final test verifies that invariant explicitly.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from concurrent.futures import Future
from importlib import import_module
from typing import Any

import pytest

from compiler.frontend.errors import PolyglotError
from runtime.concurrency import framework as legacy
from runtime.concurrency import io as cio
from runtime.concurrency import lifecycle as clife
from runtime.concurrency import sync as csync
from runtime.concurrency import tasks as ctask
from runtime.concurrency.rollout import FRAMEWORK_ROWS

_OWNED_PREFIXES = (
    "cio-reactor-",
    "clife-pool-",
    "csync-barrier-",
    "csync-rw-",
    "csync-signal-",
    "ctask-scope-",
    "ctask-wave-",
    "ctask-drain-",
)


class _Closable:
    """Tiny closable resource used by handle and lifecycle tests."""

    def __init__(self, events: list[str] | None = None, name: str = "res") -> None:
        self.events = events if events is not None else []
        self.name = name
        self.closed = False

    def close(self) -> None:
        """Record the close operation idempotently."""
        if not self.closed:
            self.closed = True
            self.events.append(self.name)


def _expect_code(
    code: str, function: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    """Assert that a call raises a PolyglotError with the expected code."""
    with pytest.raises(PolyglotError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def _new_channel(capacity: int = 2) -> Any:
    """Create a channel state via the public constructor."""
    return cio.cio_channel_create(capacity)["channel"]


def _new_registry() -> dict[str, Any]:
    """Create an empty reactor registry via the public constructor."""
    registry = cio.cio_reactor_register(name="seed")["registry"]
    registry["entries"].clear()
    registry["next_id"] = 1
    return registry


def _new_resource_registry() -> dict[str, Any]:
    """Create an empty lifecycle registry via the public constructor."""
    envelope = clife.clife_resource_register(_Closable(name="seed"), name="seed")[
        "registry"
    ]
    envelope["registry"].resources.clear()
    envelope["registry"].order.clear()
    return envelope


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------


def test_rollout_metadata_has_exactly_one_hundred_callable_rows() -> None:
    assert len(FRAMEWORK_ROWS) == 100
    assert all(len(row) == 4 for row in FRAMEWORK_ROWS)
    assert len({row[0] for row in FRAMEWORK_ROWS}) == 100
    for name, module, category, test_module in FRAMEWORK_ROWS:
        assert test_module == "tests.test_concurrency_rollout"
        assert category in {"tasks", "synchronization", "io", "lifecycle"}
        function = getattr(import_module(module), name)
        assert callable(function), name
        assert function.__doc__, name
    modules = [row[1] for row in FRAMEWORK_ROWS]
    assert modules[:10] == ["runtime.concurrency.framework"] * 10
    assert modules[10:35] == ["runtime.concurrency.tasks"] * 25
    assert modules[35:60] == ["runtime.concurrency.sync"] * 25
    assert modules[60:85] == ["runtime.concurrency.io"] * 25
    assert modules[85:] == ["runtime.concurrency.lifecycle"] * 15
    assert FRAMEWORK_ROWS[10][0] == "ctask_spawn_scoped"
    assert FRAMEWORK_ROWS[-1][0] == "clife_finalize_report"


def test_each_new_module_exports_the_required_count_and_prefix() -> None:
    for module, prefix, count in (
        (ctask, "ctask_", 25),
        (csync, "csync_", 25),
        (cio, "cio_", 25),
        (clife, "clife_", 15),
    ):
        names = module.__all__
        assert len(names) == count
        assert names == sorted(names)
        assert all(name.startswith(prefix) for name in names)
        assert all(callable(getattr(module, name)) for name in names)


# ---------------------------------------------------------------------------
# Legacy smoke coverage (full suite: tests/test_framework_f7.py)
# ---------------------------------------------------------------------------


def test_legacy_conc_spawn_scoped_task_smoke() -> None:
    scope = legacy.TaskScope(max_workers=2, name="smoke")
    try:
        future = legacy.conc_spawn_scoped_task(scope, "double", lambda n: n * 2, 4)
        assert future.result(timeout=5) == 8
    finally:
        scope.close()


def test_legacy_conc_acquire_lock_ordered_smoke() -> None:
    locks = {"beta": threading.Lock(), "alpha": threading.Lock()}
    with legacy.conc_acquire_lock_ordered(locks) as order:
        assert order == ("alpha", "beta")


def test_legacy_conc_release_guard_on_drop_smoke() -> None:
    released: list[str] = []
    guard = legacy.conc_release_guard_on_drop("resource", released.append)
    assert guard.release() is True
    assert guard.release() is False
    assert released == ["resource"]


def test_legacy_conc_channel_bounded_backpressure_smoke() -> None:
    with legacy.conc_channel_bounded_backpressure(2, name="smoke") as channel:
        assert channel.put("item") is True
        assert channel.get() == "item"


def test_legacy_conc_cancel_token_propagate_smoke() -> None:
    root = legacy.CancellationToken()
    child = legacy.conc_cancel_token_propagate(root, name="child")
    assert root.cancel() is True
    assert child.is_cancelled


def test_legacy_conc_timeout_future_cleanly_smoke() -> None:
    future: Future[int] = Future()
    future.set_result(7)
    assert legacy.conc_timeout_future_cleanly(future, timeout=0) == 7


def test_legacy_conc_reactor_poll_wakeup_smoke() -> None:
    with legacy.conc_reactor_poll_wakeup() as reactor:
        reactor.notify_work(1)
        result = reactor.poll(timeout=0)
        assert result.woke and result.reason == "work"


def test_legacy_conc_worker_pool_graceful_drain_smoke() -> None:
    pool = legacy.WorkerPool(max_workers=2, name="smoke")
    future = pool.submit("job", lambda: 3)
    report = legacy.conc_worker_pool_graceful_drain(pool)
    assert report.completed == 1 and future.result() == 3


def test_legacy_conc_detect_poisoned_state_smoke() -> None:
    future: Future[int] = Future()
    future.set_exception(ValueError("broken"))
    reports = legacy.conc_detect_poisoned_state(future, raise_on_poison=False)
    assert len(reports) == 1 and reports[0].kind == "worker"


def test_legacy_conc_audit_open_handles_smoke() -> None:
    registry = legacy.HandleRegistry()
    handle = registry.track_socket_like(io.StringIO(), name="socket-smoke")
    assert legacy.conc_audit_open_handles(registry).open_count == 1
    handle.close()
    assert legacy.conc_audit_open_handles(registry).clean


# ---------------------------------------------------------------------------
# tasks: ids 0711-0735
# ---------------------------------------------------------------------------


def test_ctask_spawn_scoped_runs_jobs_and_joins_threads() -> None:
    report = ctask.ctask_spawn_scoped(
        {f"j{index}": (lambda index=index: index * index) for index in range(4)},
        max_workers=2,
    )
    assert report["values"] == {"j0": 0, "j1": 1, "j2": 4, "j3": 9}
    assert report["submitted"] == 4 and report["threads_joined"]
    _expect_code("E003", ctask.ctask_spawn_scoped, {"bad": 42})


def test_ctask_join_all_partitions_settled_and_pending() -> None:
    report = ctask.ctask_join_all(
        [
            {"name": "a", "status": "done", "value": 1},
            {"name": "b", "status": "failed", "error": "boom"},
            {"name": "c", "status": "pending"},
        ]
    )
    assert report["values"] == (1,)
    assert report["failures"][0]["name"] == "b"
    assert report["pending"] == (2,) and not report["all_done"]
    _expect_code("E003", ctask.ctask_join_all, [object()])


def test_ctask_join_first_applies_predicate_in_input_order() -> None:
    futures = [
        {"name": "a", "status": "done", "value": 2},
        {"name": "b", "status": "done", "value": 4},
    ]
    assert (
        ctask.ctask_join_first(futures, predicate=lambda value: value > 3)["name"]
        == "b"
    )
    assert (
        ctask.ctask_join_first(futures, predicate=lambda value: value > 9)["index"]
        is None
    )
    assert ctask.ctask_join_first(futures)["settled"] == (0, 1)


def test_ctask_cancel_propagate_uses_breadth_first_order() -> None:
    report = ctask.ctask_cancel_propagate("root", {"root": ["b", "a"], "a": ["c"]})
    assert report["order"] == ("root", "a", "b", "c")
    assert report["cancelled"] == ("a", "b", "c", "root")
    _expect_code(
        "E005", ctask.ctask_cancel_propagate, "root", {"root": ["b"], "b": ["root"]}
    )


def test_ctask_cancel_scope_only_requests_running_cancellation() -> None:
    report = ctask.ctask_cancel_scope(
        {
            "a": {"status": "pending"},
            "b": {"status": "running"},
            "c": {"status": "done"},
        }
    )
    assert report["cancelled"] == ("a",)
    assert report["cancel_requested"] == ("b",)
    assert report["unchanged"] == ("c",) and report["cooperative"]
    _expect_code("E003", ctask.ctask_cancel_scope, {"a": {"status": "bogus"}})


def test_ctask_timeout_budget_allocates_and_consumes_ticks() -> None:
    report = ctask.ctask_timeout_budget(10, {"a": 1, "b": 3}, clock=2)
    assert report["base_budgets"] == {"a": 3, "b": 7}
    assert report["budgets"] == {"a": 1, "b": 7}
    assert report["allocated"] == 8 and report["consumed"] == 2
    _expect_code("E003", ctask.ctask_timeout_budget, -1, {})


def test_ctask_retry_policy_computes_deterministic_schedule() -> None:
    policy = {
        "max_attempts": 3,
        "base_delay": 1,
        "factor": 2,
        "max_delay": 3,
        "jitter": [1],
    }
    schedule = ctask.ctask_retry_policy(policy)
    assert schedule["schedule"] == (2, 2, 3)
    assert schedule["delay"] == 2 and schedule["should_retry"]
    assert not ctask.ctask_retry_policy(policy, failures=3)["should_retry"]


def test_ctask_retry_run_stops_on_success() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    report = ctask.ctask_retry_run(
        [flaky, flaky, flaky],
        should_retry=lambda error: True,
        clock=lambda: len(calls),
    )
    assert report["succeeded"] and report["value"] == "ok"
    assert report["attempts_run"] == 3 and len(report["errors"]) == 2


def test_ctask_dependency_graph_detects_cycles_and_missing_tasks() -> None:
    report = ctask.ctask_dependency_graph({"a": [], "b": ["a"], "c": ["a"]})
    assert report["layers"] == (("a",), ("b", "c"))
    assert report["edges"] == (("a", "b"), ("a", "c"))
    _expect_code("E005", ctask.ctask_dependency_graph, {"a": ["b"], "b": ["a"]})
    _expect_code("E006", ctask.ctask_dependency_graph, {"a": ["missing"]})


def test_ctask_topological_schedule_uses_priority_tie_break() -> None:
    report = ctask.ctask_topological_schedule(
        {"root": [], "z": ["root"], "a": ["root"]},
        priorities={"z": 0, "a": 5},
    )
    assert report["order"] == ("root", "z", "a")
    assert report["edges"] == 2


def test_ctask_graph_ready_set_splits_ready_and_blocked() -> None:
    report = ctask.ctask_graph_ready_set({"a": [], "b": ["a"], "c": ["b"]}, ["a"])
    assert report["ready"] == ("b",) and report["blocked"] == ("c",)
    _expect_code("E006", ctask.ctask_graph_ready_set, {"a": []}, ["missing"])


def test_ctask_graph_wave_run_joins_every_wave() -> None:
    report = ctask.ctask_graph_wave_run(
        {"a": [], "b": ["a"]}, {"a": lambda: 1, "b": lambda: 2}
    )
    assert report["values"] == {"a": 1, "b": 2}
    assert report["waves"] == (("a",), ("b",)) and report["threads_joined"]
    _expect_code("E006", ctask.ctask_graph_wave_run, {"a": []}, {})


def test_ctask_future_settle_prevents_double_settlement() -> None:
    settled = ctask.ctask_future_settle({"name": "f", "status": "pending"}, value=5)
    assert settled["status"] == "done" and settled["record"]["value"] == 5
    assert settled["changed"]
    _expect_code("E006", ctask.ctask_future_settle, settled["record"], value=6)


def test_ctask_future_chain_applies_transforms_in_order() -> None:
    report = ctask.ctask_future_chain(
        [
            {"name": "a", "status": "done", "value": 2},
            {"name": "b", "status": "done", "value": 3},
        ],
        [lambda value: value + 1, lambda value: value * 2],
    )
    assert report["value"] == 6 and report["completed"]
    assert len(report["steps"]) == 2


def test_ctask_result_collect_partitions_outcomes() -> None:
    report = ctask.ctask_result_collect(
        [
            {"name": "a", "ok": True, "value": 1},
            {"name": "b", "ok": False, "error": "x"},
            {"name": "c", "status": "pending"},
        ]
    )
    assert report["values"] == (1,)
    assert report["failures"][0]["name"] == "b"
    assert report["pending"] == ("c",)
    assert report["counts"] == {"values": 1, "failures": 1, "pending": 1}


def test_ctask_error_aggregate_groups_by_code() -> None:
    report = ctask.ctask_error_aggregate(
        [
            {"code": "E004", "name": "a", "message": "boom"},
            {"name": "b", "message": "x"},
        ]
    )
    assert report["count"] == 2
    assert report["by_code"]["E004"] == ("a: boom", "b: x")
    _expect_code("E003", ctask.ctask_error_aggregate, [{"code": "E009", "name": "bad"}])


def test_ctask_pause_resume_cancellation_is_terminal() -> None:
    report = ctask.ctask_pause_resume(["pause", "resume", "cancel", "resume"])
    assert report["state"] == "cancelled" and report["pauses"] == 1
    assert report["resumes"] == 1 and report["history"][3]["changed"] is False
    _expect_code("E003", ctask.ctask_pause_resume, ["sleep"])


def test_ctask_yield_round_robin_skips_empty_queues() -> None:
    report = ctask.ctask_yield_round_robin({"a": [1, 2], "b": [3]}, turns=3)
    assert report["order"] == (("a", 1), ("b", 3), ("a", 2))
    assert report["turns_taken"] == 3 and report["exhausted"]


def test_ctask_priority_queue_drops_tail_over_limit() -> None:
    report = ctask.ctask_priority_queue(
        [
            {"name": "a", "priority": 2},
            {"name": "b", "priority": 1},
            {"name": "c", "priority": 1, "seq": 0},
        ],
        limit=2,
    )
    assert report["ordered"] == ("c", "b") and report["dropped"] == ("a",)


def test_ctask_rate_limit_token_bucket_over_injected_ticks() -> None:
    report = ctask.ctask_rate_limit([0, 0, 1, 1], capacity=2, refill=1)
    assert report["allowed"] == (True, True, True, False)
    assert report["rejected"] == 1 and report["tokens"] == 0
    _expect_code("E003", ctask.ctask_rate_limit, [2, 1], capacity=2, refill=1)


def test_ctask_backoff_plan_caps_delays_deterministically() -> None:
    report = ctask.ctask_backoff_plan(4, base=2, factor=2, cap=5, jitter=[1])
    assert report["delays"] == (3, 4, 5, 5)
    assert report["capped_at"] == 2 and report["total"] == 17


def test_ctask_deadline_check_reports_expiry_and_overrun() -> None:
    report = ctask.ctask_deadline_check(12, deadline=10, started=2)
    assert report["remaining"] == 0 and report["expired"]
    assert report["overrun"] == 2 and report["elapsed"] == 10
    _expect_code("E003", ctask.ctask_deadline_check, 5, deadline=1, started=2)


def test_ctask_shutdown_drain_skips_cancelled_jobs() -> None:
    report = ctask.ctask_shutdown_drain(
        {"a": lambda: 1, "b": lambda: 2}, cancel_queued=["b"]
    )
    assert report["completed"] == ("a",) and report["cancelled"] == ("b",)
    assert report["threads_joined"]
    _expect_code(
        "E006", ctask.ctask_shutdown_drain, {"a": lambda: 1}, cancel_queued=["missing"]
    )


def test_ctask_task_tree_computes_depths_and_subtree_sizes() -> None:
    report = ctask.ctask_task_tree(
        {
            "root": None,
            "a": "root",
            "b": "root",
            "c": "a",
        }
    )
    assert report["roots"] == ("root",)
    assert report["depths"] == {"a": 1, "b": 1, "c": 2, "root": 0}
    assert report["sizes"]["root"] == 4 and report["sizes"]["c"] == 1
    _expect_code("E005", ctask.ctask_task_tree, {"a": "b", "b": "a"})


def test_ctask_trace_events_sorts_by_sequence() -> None:
    report = ctask.ctask_trace_events(
        [
            {"seq": 2, "name": "b", "kind": "close"},
            {"seq": 1, "name": "a", "kind": "open"},
        ]
    )
    assert report["sequence"] == (1, 2)
    assert [event["name"] for event in report["events"]] == ["a", "b"]
    assert report["counts"] == {"close": 1, "open": 1}


# ---------------------------------------------------------------------------
# sync: ids 0736-0760
# ---------------------------------------------------------------------------


def test_csync_lock_order_plan_sorts_names_and_release_order() -> None:
    report = csync.csync_lock_order_plan({"b": threading.Lock(), "a": threading.Lock()})
    assert report["acquire"] == ("a", "b")
    assert report["release"] == ("b", "a") and report["deterministic"]
    _expect_code("E003", csync.csync_lock_order_plan, {"a": object()})


def test_csync_detect_cycle_returns_path_or_raises_e005() -> None:
    report = csync.csync_detect_cycle([("a", "b"), ("b", "c")], raise_on_cycle=False)
    assert report["acyclic"] and report["nodes"] == ("a", "b", "c")
    _expect_code("E005", csync.csync_detect_cycle, [("a", "b"), ("b", "a")])


def test_csync_acquire_ordered_releases_on_body_failure() -> None:
    locks = {"a": threading.Lock(), "b": threading.Lock()}
    with csync.csync_acquire_ordered(locks, clock=lambda: 100.0, timeout=5.0) as order:
        assert order == ("a", "b")
        assert not locks["a"].acquire(blocking=False)
    assert locks["a"].acquire(blocking=False)
    locks["a"].release()
    with pytest.raises(RuntimeError), csync.csync_acquire_ordered(locks):
        raise RuntimeError("body failed")
    assert locks["a"].acquire(blocking=False)
    locks["a"].release()


def test_csync_guard_releases_even_when_body_raises() -> None:
    released: list[str] = []
    with csync.csync_guard("res", releaser=released.append, name="guard") as guard:
        assert guard["resource"] == "res" and not guard["released"]
    assert guard["released"] and released == ["res"]
    with (
        pytest.raises(ValueError),
        csync.csync_guard("other", releaser=released.append),
    ):
        raise ValueError("body failed")
    assert released == ["res", "other"]


def test_csync_guard_group_releases_in_reverse_order() -> None:
    released: list[Any] = []
    resources = {"a": object(), "b": object()}
    with csync.csync_guard_group(resources, releaser=released.append) as guards:
        assert list(guards) == ["a", "b"]
    assert released == [resources["b"], resources["a"]]
    assert all(guard["released"] for guard in guards.values())


def test_csync_reentrant_depth_tracks_balance() -> None:
    report = csync.csync_reentrant_depth(["acquire", "acquire", "release", "release"])
    assert report["depth"] == 0 and report["balanced"]
    assert len(report["history"]) == 4
    _expect_code("E003", csync.csync_reentrant_depth, ["release"])


def test_csync_semaphore_tickets_grants_waiters_fifo() -> None:
    report = csync.csync_semaphore_tickets(
        1, [("acquire", "a"), ("acquire", "b"), ("release", "c")]
    )
    assert report["granted"] == ("a", "b")
    assert report["denied"] == ("b",) and report["waiters"] == ()


def test_csync_barrier_arrive_opens_only_when_all_arrive() -> None:
    assert not csync.csync_barrier_arrive(["a", "b"], ["a"])["open"]
    report = csync.csync_barrier_arrive(["a", "b"], ["b", "a"])
    assert report["open"] and report["released"] == ("a", "b")
    _expect_code("E003", csync.csync_barrier_arrive, ["a"], ["unknown"])


def test_csync_condition_waiters_notifies_in_arrival_order() -> None:
    report = csync.csync_condition_waiters(
        [
            ("wait", "a"),
            ("wait", "b"),
            ("notify", "x"),
            ("notify_all", "y"),
        ]
    )
    assert report["waiting"] == () and report["notified"] == ("a", "b")
    assert len(report["history"]) == 4
    _expect_code("E006", csync.csync_condition_waiters, [("notify", "x")])


def test_csync_notify_policy_supports_deterministic_modes_only() -> None:
    report = csync.csync_notify_policy(["b", "a"], policy="one", count=1)
    assert report["selected"] == ("a",) and report["remaining"] == ("b",)
    assert csync.csync_notify_policy(["a", "b"], policy="all")["selected"] == ("a", "b")
    _expect_code("E007", csync.csync_notify_policy, ["a"], policy="random")


def test_csync_reader_writer_plan_prioritizes_waiting_writers() -> None:
    report = csync.csync_reader_writer_plan(
        [
            ("read", "r1"),
            ("write", "w1"),
            ("read", "r2"),
        ]
    )
    assert report["admitted"] == ("r1",)
    assert report["deferred"] == ("w1", "r2") and report["waiting_writers"] == 1


def test_csync_spin_tick_is_bounded_by_max_ticks() -> None:
    report = csync.csync_spin_tick(lambda tick: tick >= 2, max_ticks=5)
    assert report["ok"] and report["ticks"] == 3
    assert report["tick_values"] == (0, 1, 2)
    assert not csync.csync_spin_tick(lambda tick: False, max_ticks=2)["ok"]


def test_csync_timeout_remaining_uses_injected_clock() -> None:
    report = csync.csync_timeout_remaining(10.0, now=3.0)
    assert report["remaining"] == 7.0 and not report["expired"]
    assert csync.csync_timeout_remaining(3.0, now=9.0)["expired"]


def test_csync_poison_state_marks_a_copy() -> None:
    original = {"owner": "worker"}
    report = csync.csync_poison_state(original, reason="corrupt")
    assert report["poisoned"] and report["state"]["poison"]["reason"] == "corrupt"
    assert "poisoned" not in original


def test_csync_check_poison_can_raise_e004() -> None:
    state = csync.csync_poison_state({"owner": "worker"}, reason="corrupt")["state"]
    report = csync.csync_check_poison(state)
    assert report["poisoned"] and report["reports"][0]["reason"] == "corrupt"
    _expect_code("E004", csync.csync_check_poison, state, raise_on_poison=True)
    assert csync.csync_check_poison({})["clean"]


def test_csync_atomic_compare_expect_only_swaps_on_match() -> None:
    cell = {"value": 1}
    report = csync.csync_atomic_compare_expect(cell, expect=1, update=2)
    assert report["swapped"] and cell["value"] == 2
    assert not csync.csync_atomic_compare_expect(cell, expect=1, update=3)["swapped"]


def test_csync_counter_snapshot_records_each_checkpoint() -> None:
    report = csync.csync_counter_snapshot([1, -2, 3], initial=10)
    assert report["value"] == 12
    assert [item["value"] for item in report["checkpoints"]] == [11, 9, 12]


def test_csync_owned_lock_joins_state_on_exit_even_on_error() -> None:
    lock = threading.Lock()
    held: set[str] = set()
    with csync.csync_owned_lock("worker", lock=lock, held=held) as guard:
        assert guard["held"] and "worker" in held
        assert not lock.acquire(blocking=False)
    assert not held and lock.acquire(blocking=False)
    lock.release()
    with (
        pytest.raises(ValueError),
        csync.csync_owned_lock("worker", lock=lock, held=held),
    ):
        raise ValueError("body failed")
    assert not held


def test_csync_fair_queue_serves_tickets_in_order() -> None:
    report = csync.csync_fair_queue(
        [
            ("enqueue", "a"),
            ("enqueue", "b"),
            ("dequeue", "x"),
        ]
    )
    assert report["served"] == ("a",) and report["waiting"] == ("b",)
    assert report["next_ticket"] == 2


def test_csync_deadlock_probe_detects_wait_cycles() -> None:
    report = csync.csync_deadlock_probe({"a": "b", "b": None}, raise_on_cycle=False)
    assert report["acyclic"] and report["blocked"] == ("a",)
    _expect_code("E005", csync.csync_deadlock_probe, {"a": "b", "b": "a"})


def test_csync_rw_lock_run_joins_all_workers() -> None:
    report = csync.csync_rw_lock_run(
        {"r1": lambda: 1, "r2": lambda: 2}, {"w1": lambda: 3}
    )
    assert report["values"] == {"r1": 1, "r2": 2, "w1": 3}
    assert report["threads_joined"] and report["writers"] == ("w1",)


def test_csync_condition_signal_run_releases_waiters() -> None:
    seen: list[int] = []
    report = csync.csync_condition_signal_run({"w1": lambda: seen.append(1)})
    assert report["released"] == ("w1",) and report["signalled"]
    assert report["values"] == {"w1": None} and seen == [1]


def test_csync_lock_audit_only_inspects_wrapped_guards() -> None:
    report = csync.csync_lock_audit(
        [
            {"name": "a", "held": False},
            {"name": "b", "held": True},
        ]
    )
    assert report["released"] == ("a",) and report["held"] == ("b",)
    assert not report["clean"]
    _expect_code("E003", csync.csync_lock_audit, [{"name": "x"}])


def test_csync_barrier_thread_run_never_strands_participants() -> None:
    report = csync.csync_barrier_thread_run({"a": lambda: 1, "b": lambda: 2})
    assert report["values"] == {"a": 1, "b": 2} and report["parties"] == 2

    def boom() -> None:
        raise ValueError("failed")

    failed = csync.csync_barrier_thread_run({"bad": boom, "good": lambda: 7})
    assert failed["failures"][0]["name"] == "bad"
    assert failed["values"] == {"good": 7} and failed["threads_joined"]


def test_csync_guard_context_report_finds_leaks() -> None:
    report = csync.csync_guard_context_report(
        [
            {"name": "a", "action": "open"},
            {"name": "a", "action": "close"},
        ]
    )
    assert report["balanced"] and report["leaks"] == ()
    _expect_code(
        "E006", csync.csync_guard_context_report, [{"name": "x", "action": "close"}]
    )


# ---------------------------------------------------------------------------
# io: ids 0761-0785
# ---------------------------------------------------------------------------


def test_cio_channel_create_reports_empty_bounded_state() -> None:
    report = cio.cio_channel_create(2, name="pipe")
    assert report["stats"]["capacity"] == 2 and report["stats"]["size"] == 0
    assert report["name"] == "pipe"
    _expect_code("E003", cio.cio_channel_create, 0)


def test_cio_channel_put_applies_backpressure_and_drop_oldest() -> None:
    channel = _new_channel(1)
    assert cio.cio_channel_put(channel, "a")["accepted"]
    assert not cio.cio_channel_put(channel, "b")["accepted"]
    dropped = cio.cio_channel_put(channel, "c", overflow="drop_oldest")
    assert dropped["accepted"] and dropped["discarded"] == "a"
    _expect_code("E003", cio.cio_channel_put, channel, "x", overflow="unknown")


def test_cio_channel_get_pops_fifo_and_reports_empty() -> None:
    channel = _new_channel(2)
    cio.cio_channel_put(channel, "first")
    cio.cio_channel_put(channel, "second")
    assert cio.cio_channel_get(channel)["item"] == "first"
    assert cio.cio_channel_get(channel)["item"] == "second"
    assert cio.cio_channel_get(channel)["empty"]


def test_cio_channel_close_can_drain_buffered_items() -> None:
    channel = _new_channel(2)
    cio.cio_channel_put(channel, "x")
    report = cio.cio_channel_close(channel, drain=True)
    assert report["closed"] and report["discarded"] == ("x",)
    assert report["stats"]["closed"]
    _expect_code("E003", cio.cio_channel_get, channel)


def test_cio_channel_context_closes_on_body_failure() -> None:
    with cio.cio_channel_context(1, name="ctx") as env:
        assert not env["closed"]
    assert env["closed"] and env["channel"].closed
    with pytest.raises(ValueError), cio.cio_channel_context(1) as failed:
        raise ValueError("body failed")
    assert failed["closed"] and failed["channel"].closed


def test_cio_buffer_writer_respects_capacity_and_offset() -> None:
    report = cio.cio_buffer_writer(b"abcdef", capacity=4, offset=1)
    assert report["written"] == b"abc" and report["overflow"] == b"def"
    assert report["next_offset"] == 4 and report["truncated"]
    _expect_code("E003", cio.cio_buffer_writer, "text", capacity=2)


def test_cio_buffer_reader_returns_bounded_slice() -> None:
    report = cio.cio_buffer_reader(b"abcdef", offset=2, size=2)
    assert report["data"] == b"cd" and report["next_offset"] == 4
    assert not report["exhausted"]
    assert cio.cio_buffer_reader(b"ab", offset=1)["exhausted"]


def test_cio_reactor_register_validates_primitive_support() -> None:
    registry = _new_registry()
    report = cio.cio_reactor_register(
        registry, "timer-1", primitive="timer", ready_at=3
    )
    assert report["id"] == 1 and report["entry"]["ready_at"] == 3
    _expect_code("E006", cio.cio_reactor_register, registry, "timer-1")
    _expect_code("E007", cio.cio_reactor_register, registry, "epoll", primitive="epoll")


def test_cio_reactor_poll_fires_due_sources_by_tick() -> None:
    registry = _new_registry()
    cio.cio_reactor_register(registry, "early", ready_at=1)
    cio.cio_reactor_register(registry, "late", ready_at=3)
    first = cio.cio_reactor_poll(registry, tick=2)
    assert [event["name"] for event in first["events"]] == ["early"]
    assert first["pending"] == ("late",)
    assert [
        event["name"] for event in cio.cio_reactor_poll(registry, tick=5)["events"]
    ] == ["late"]


def test_cio_reactor_wakeup_joins_thread_on_exit() -> None:
    with cio.cio_reactor_wakeup() as env:
        owned = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("cio-reactor-")
        ]
        assert owned and all(thread.is_alive() for thread in owned)
    assert env["closed"] and env["threads_joined"]
    assert env["records"] == [{"kind": "wakeup", "reason": "shutdown"}]
    assert not any(
        thread.is_alive()
        for thread in threading.enumerate()
        if thread.name.startswith("cio-reactor-")
    )


def test_cio_stream_chunk_and_reassemble_detect_gaps() -> None:
    chunks = cio.cio_stream_chunk(b"abcdefg", size=3)["chunks"]
    assert [chunk["offset"] for chunk in chunks] == [0, 3, 6]
    report = cio.cio_stream_reassemble([chunks[0], chunks[2]])
    assert not report["complete"] and 3 in report["missing"]
    whole = cio.cio_stream_reassemble(chunks, expected=7)
    assert whole["complete"] and whole["data"] == b"abcdefg"


def test_cio_line_reader_preserves_line_endings_on_request() -> None:
    report = cio.cio_line_reader("a\nb\n", keepends=True)
    assert report["lines"] == ("a\n", "b\n") and report["count"] == 2
    assert cio.cio_line_reader("a\nb")["terminated"] is False


def test_cio_io_result_tags_success_and_error() -> None:
    assert cio.cio_io_result(4)["status"] == "ok"
    failed = cio.cio_io_result(error="boom", name="read")
    assert failed["status"] == "error" and failed["value"] is None
    _expect_code("E003", cio.cio_io_result, name="")


def test_cio_io_retry_records_failures_before_success() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient")
        return "data"

    report = cio.cio_io_retry([flaky, flaky], on_error=lambda error: True)
    assert report["succeeded"] and report["value"] == "data"
    assert len(report["errors"]) == 1 and report["retries"] == 0


def test_cio_handle_track_and_close_are_idempotent() -> None:
    registry = _new_registry()
    resource = _Closable()
    tracked = cio.cio_handle_track(registry, resource, name="h", kind="file")
    assert tracked["id"] == 1
    first = cio.cio_handle_close(tracked["handle"])
    second = cio.cio_handle_close(tracked["handle"])
    assert first["closed"] and second["already_closed"]
    assert resource.closed and len(resource.events) == 1
    _expect_code("E006", cio.cio_handle_close, object())


def test_cio_handle_audit_sees_only_wrapped_handles() -> None:
    registry = _new_registry()
    tracked = cio.cio_handle_track(registry, _Closable(), name="leak")
    report = cio.cio_handle_audit(registry)
    assert report["open_count"] == 1 and not report["clean"]
    cleaned = cio.cio_handle_audit(registry, close_all=True)
    assert cleaned["clean"] and cleaned["closed_count"] == 1
    assert tracked["handle"].closed


def test_cio_handle_context_closes_on_body_failure() -> None:
    registry = _new_registry()
    resource = _Closable()
    with cio.cio_handle_context(registry, resource, name="wrapped") as env:
        assert not env["closed"]
    assert env["closed"] and resource.closed
    other = _Closable(name="other")
    with (
        pytest.raises(RuntimeError),
        cio.cio_handle_context(registry, other, name="wrapped-2"),
    ):
        raise RuntimeError("body failed")
    assert other.closed


def test_cio_path_platform_flags_unsupported_primitives() -> None:
    report = cio.cio_path_platform("pipe://task", platform="wasm")
    assert report["kind"] == "pipe" and not report["supported"]
    assert report["fallback"] == "in-memory reactor"
    _expect_code("E007", cio.cio_path_platform, "file", platform="unknown")


def test_cio_fd_budget_rejects_over_reservation() -> None:
    report = cio.cio_fd_budget(3, capacity=4, reserve=2)
    assert not report["granted"] and report["after"] == 3
    assert cio.cio_fd_budget(3, capacity=4, release=1, reserve=1)["granted"]


def test_cio_write_batch_flushes_at_threshold() -> None:
    report = cio.cio_write_batch(
        [{"name": "a", "size": 2}, {"name": "b", "size": 2}, {"name": "c", "size": 1}],
        threshold=3,
    )
    assert report["batches"] == (("a", "b"),)
    assert report["flushed"] == 1 and report["pending"] == ("c",)


def test_cio_read_exact_reports_short_reads() -> None:
    report = cio.cio_read_exact([b"ab", b"cd"], size=5)
    assert report["data"] == b"abcd" and report["short"] == 1
    assert not report["complete"] and report["consumed"] == 2
    assert cio.cio_read_exact([b"abc"], size=3)["complete"]


def test_cio_event_loop_tick_fires_due_callbacks() -> None:
    fired: list[str] = []
    report = cio.cio_event_loop_tick(
        [
            {"name": "now", "at": 1, "callback": lambda: fired.append("now")},
            {"name": "later", "at": 4},
        ],
        tick=2,
    )
    assert [event["name"] for event in report["fired"]] == ["now"]
    assert report["deferred"] == ("later",) and fired == ["now"]


def test_cio_io_trace_sorts_records_by_sequence() -> None:
    report = cio.cio_io_trace(
        [
            {"seq": 2, "name": "b", "kind": "close", "tick": 9},
            {"seq": 1, "name": "a", "kind": "open", "tick": 3},
        ]
    )
    assert report["first_tick"] == 3 and report["last_tick"] == 9
    assert [event["name"] for event in report["events"]] == ["a", "b"]
    assert report["counts"] == {"close": 1, "open": 1}


# ---------------------------------------------------------------------------
# lifecycle: ids 0786-0800
# ---------------------------------------------------------------------------


def test_clife_resource_register_creates_wrapped_state() -> None:
    events: list[str] = []
    registry = _new_resource_registry()
    report = clife.clife_resource_register(
        _Closable(events, "data"), name="data", registry=registry
    )
    assert report["state"] == "registered" and report["name"] == "data"
    _expect_code(
        "E003",
        clife.clife_resource_register,
        _Closable(),
        name="bad",
        kind="unknown",
        registry=registry,
    )


def test_clife_resource_acquire_rejects_unknown_resources() -> None:
    registry = _new_resource_registry()
    clife.clife_resource_register(_Closable(), name="data", registry=registry)
    report = clife.clife_resource_acquire(registry, "data")
    assert report["state"] == "acquired" and report["previous"] == "registered"
    _expect_code("E006", clife.clife_resource_acquire, registry, "missing")


def test_clife_resource_release_closes_once() -> None:
    events: list[str] = []
    registry = _new_resource_registry()
    clife.clife_resource_register(
        _Closable(events, "data"), name="data", registry=registry
    )
    clife.clife_resource_acquire(registry, "data")
    report = clife.clife_resource_release(registry, "data")
    assert report["closed"] and events == ["data"]
    _expect_code("E003", clife.clife_resource_release, registry, "data")


def test_clife_resource_context_releases_on_body_failure() -> None:
    events: list[str] = []
    registry = _new_resource_registry()
    clife.clife_resource_register(
        _Closable(events, "data"), name="data", registry=registry
    )
    with clife.clife_resource_context(registry, "data") as env:
        assert not env["released"]
    assert env["released"] and events == ["data"]
    other_events: list[str] = []
    clife.clife_resource_register(
        _Closable(other_events, "other"), name="other", registry=registry
    )
    with pytest.raises(ValueError), clife.clife_resource_context(registry, "other"):
        raise ValueError("body failed")
    assert other_events == ["other"]


def test_clife_shutdown_plan_reverses_startup_order() -> None:
    report = clife.clife_shutdown_plan(
        {
            "app": ["db", "cache"],
            "db": [],
            "cache": [],
        },
        roots=["app"],
    )
    assert report["startup"] == ("cache", "db", "app")
    assert report["order"] == ("app", "db", "cache")
    _expect_code("E005", clife.clife_shutdown_plan, {"a": ["b"], "b": ["a"]})
    _expect_code("E006", clife.clife_shutdown_plan, {"a": []}, roots=["missing"])


def test_clife_shutdown_run_collects_callback_errors() -> None:
    calls: list[str] = []

    def fail() -> None:
        raise RuntimeError("stop failed")

    report = clife.clife_shutdown_run(
        [
            ("first", lambda: calls.append("first")),
            ("second", fail),
            ("third", lambda: calls.append("third")),
        ]
    )
    assert report["order"] == ("first", "second", "third")
    assert report["completed"] == ("first", "third")
    assert report["failed"][0]["name"] == "second" and not report["all_done"]
    assert calls == ["first", "third"]


def test_clife_pool_create_reports_capacity() -> None:
    report = clife.clife_pool_create("jobs", capacity=3)
    assert report["name"] == "jobs" and report["capacity"] == 3
    assert not report["closed"]


def test_clife_pool_submit_rejects_closed_pool() -> None:
    pool = clife.clife_pool_create("jobs")
    assert clife.clife_pool_submit(pool, "job", lambda: 1)["queued"] == 1
    clife.clife_pool_drain(pool)
    _expect_code("E006", clife.clife_pool_submit, pool, "late", lambda: 2)


def test_clife_pool_drain_joins_workers_and_reports_failures() -> None:
    def boom() -> None:
        raise ValueError("job failed")

    pool = clife.clife_pool_create("jobs")
    clife.clife_pool_submit(pool, "ok", lambda: 7)
    clife.clife_pool_submit(pool, "bad", boom)
    report = clife.clife_pool_drain(pool)
    assert report["values"] == {"ok": 7} and report["threads_joined"]
    assert report["failures"][0]["name"] == "bad"
    assert report["completed"] == ("ok",)
    assert clife.clife_pool_drain(pool)["already_closed"]


def test_clife_pool_context_joins_threads_even_on_body_failure() -> None:
    with clife.clife_pool_context(2, name="ctx") as pool:
        assert not pool["closed"]
        clife.clife_pool_submit(pool, "job", lambda: 11)
    assert pool["closed"] and pool["threads_joined"]
    assert not any(
        thread.is_alive()
        for thread in threading.enumerate()
        if thread.name.startswith("clife-pool-")
    )
    assert pool["last_report"]["values"] == {"job": 11}
    with (
        pytest.raises(ValueError),
        clife.clife_pool_context(1, name="failing") as failed,
    ):
        clife.clife_pool_submit(failed, "job", lambda: 1)
        raise ValueError("body failed")
    assert failed["closed"] and failed["threads_joined"]


def test_clife_poison_mark_creates_structured_record() -> None:
    report = clife.clife_poison_mark({"kind": "pool"}, reason="leaked", owner="worker")
    assert report["marked"] and report["poison"]["owner"] == "worker"
    assert report["state"]["poisoned"]


def test_clife_poison_check_can_raise_e004() -> None:
    poison = clife.clife_poison_mark({}, reason="corrupt")["poison"]
    report = clife.clife_poison_check(poison)
    assert report["poisoned"] and report["reports"][0]["reason"] == "corrupt"
    _expect_code("E004", clife.clife_poison_check, poison, raise_on_poison=True)
    assert clife.clife_poison_check(None)["clean"]


def test_clife_leak_audit_closes_only_wrapped_resources() -> None:
    events: list[str] = []
    registry = _new_resource_registry()
    clife.clife_resource_register(
        _Closable(events, "data"), name="data", registry=registry
    )
    audit = clife.clife_leak_audit(registry)
    assert audit["open"] == ("data",) and not audit["clean"]
    cleaned = clife.clife_leak_audit(registry, close_all=True)
    assert cleaned["clean"] and cleaned["closed"] == ("data",)
    assert events == ["data"]


def test_clife_guard_stack_closes_in_reverse_order() -> None:
    events: list[str] = []
    report = clife.clife_guard_stack(
        [
            {"name": "outer", "resource": _Closable(events, "outer")},
            {"name": "inner", "resource": _Closable(events, "inner")},
        ]
    )
    assert report["open_order"] == ("outer", "inner")
    assert report["close_order"] == ("inner", "outer")
    assert events == ["inner", "outer"] and report["balanced"]
    _expect_code("E003", clife.clife_guard_stack, [{"name": "dup"}, {"name": "dup"}])


def test_clife_finalize_report_lists_leaks_and_errors() -> None:
    report = clife.clife_finalize_report(
        [
            {"seq": 3, "name": "b", "action": "error"},
            {"seq": 1, "name": "a", "action": "open"},
            {"seq": 2, "name": "c", "action": "close"},
        ]
    )
    assert report["opened"] == ("a",) and report["closed"] == ("c",)
    assert report["leaks"] == ("a",) and report["errors"] == ("b",)
    assert not report["clean"]
    _expect_code(
        "E003", clife.clife_finalize_report, [{"name": "x", "action": "unknown"}]
    )


# ---------------------------------------------------------------------------
# Cross-module lifecycle invariant
# ---------------------------------------------------------------------------


def test_no_owned_thread_survives_context_manager_exit() -> None:
    before = {thread.ident for thread in threading.enumerate()}
    with cio.cio_reactor_wakeup() as reactor:
        assert reactor["registry"]["entries"] == {}
    with clife.clife_pool_context(2, name="liveness") as pool:
        clife.clife_pool_submit(pool, "job", lambda: 1)
    barrier_report = csync.csync_barrier_thread_run({"a": lambda: 1})
    assert barrier_report["threads_joined"]
    report = ctask.ctask_shutdown_drain({"job": lambda: 2})
    assert report["threads_joined"]
    survivors = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive()
        and thread.ident not in before
        and thread.name.startswith(_OWNED_PREFIXES)
    ]
    assert survivors == []
