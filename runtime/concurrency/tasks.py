"""Deterministic task, cancellation and scheduling primitives for batch eight.

The helpers in this module model task graphs, cooperative cancellation and
completion combinators without relying on wall-clock sleeping.  Time is always
supplied by the caller as integer ticks or as an injected callable, and every
helper that starts a thread owns a close path that joins that thread before the
helper returns.  Cancellation never forcibly kills a worker: it records a
cooperative request or sets an :class:`threading.Event` that the worker checks.

All failures use the shared ``E001``–``E008`` diagnostic surface through
:class:`compiler.frontend.errors.PolyglotError`.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from compiler.frontend.errors import PolyglotError

__all__ = [
    "ctask_backoff_plan",
    "ctask_cancel_propagate",
    "ctask_cancel_scope",
    "ctask_deadline_check",
    "ctask_dependency_graph",
    "ctask_error_aggregate",
    "ctask_future_chain",
    "ctask_future_settle",
    "ctask_graph_ready_set",
    "ctask_graph_wave_run",
    "ctask_join_all",
    "ctask_join_first",
    "ctask_pause_resume",
    "ctask_priority_queue",
    "ctask_rate_limit",
    "ctask_result_collect",
    "ctask_retry_policy",
    "ctask_retry_run",
    "ctask_shutdown_drain",
    "ctask_spawn_scoped",
    "ctask_task_tree",
    "ctask_timeout_budget",
    "ctask_topological_schedule",
    "ctask_trace_events",
    "ctask_yield_round_robin",
]

_CODES = frozenset(f"E{value:03d}" for value in range(1, 9))
_STATUSES = frozenset({"pending", "running", "done", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    """Raise a structured :class:`PolyglotError` for this module."""
    if code not in _CODES:
        raise ValueError(f"unknown diagnostic code {code!r}")
    raise PolyglotError(code, message, reason=reason, note="runtime.concurrency.tasks")


def _require_name(value: object, label: str) -> str:
    """Return a non-empty string name or raise E003."""
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
    """Validate a sequence of unique names and return it sorted."""
    if isinstance(values, Mapping):
        items = tuple(values)
    else:
        items = _require_sequence(values, label)
    names = {_require_name(item, f"{label} item") for item in items}
    if len(names) != len(items):
        _fail("E003", f"{label} contains duplicate names")
    return tuple(sorted(names))


def _normal_graph(
    specs: object,
) -> tuple[dict[str, tuple[str, ...]], tuple[tuple[str, ...], ...]]:
    """Normalize a dependency graph and compute its Kahn layers (E005/E006)."""
    mapping = _require_mapping(specs, "graph")
    graph = {
        _require_name(name, "task name"): tuple(
            sorted(set(_name_tuple(deps, f"dependencies of {name!r}")))
        )
        for name, deps in mapping.items()
    }
    for name, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                _fail("E006", f"task {name!r} depends on unknown task {dep!r}")
    indegree = {name: len(deps) for name, deps in graph.items()}
    children: dict[str, list[str]] = {name: [] for name in graph}
    for name, deps in graph.items():
        for dep in deps:
            children[dep].append(name)
    layers: list[tuple[str, ...]] = []
    ready = sorted(name for name, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        layers.append(tuple(ready))
        following: list[str] = []
        for name in ready:
            visited += 1
            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    following.append(child)
        ready = sorted(following)
    if visited != len(graph):
        stuck = tuple(sorted(name for name, count in indegree.items() if count > 0))
        _fail("E005", "task dependency cycle", reason=f"tasks involved: {stuck}")
    return graph, tuple(layers)


@dataclass
class _ThreadOwner:
    """Joinable thread group with a single cooperative close path."""

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
        kwargs: Mapping[str, Any] | None = None,
        *,
        gate: threading.BoundedSemaphore | None = None,
    ) -> threading.Thread:
        """Start one named worker and record its outcome, success or failure."""
        if self._closed:
            _fail("E003", "thread owner is already closed")
        arguments = dict(kwargs or {})

        def runner() -> None:
            if gate is not None:
                gate.acquire()
            try:
                value = fn(*args, **arguments)
            except BaseException as error:
                outcome = {"name": name, "ok": False, "value": None, "error": error}
            else:
                outcome = {"name": name, "ok": True, "value": value, "error": None}
            finally:
                if gate is not None:
                    gate.release()
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
        """Signal cooperative stop, join every started thread and verify it."""
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
            _fail("E004", "worker threads did not stop", reason=f"alive: {alive}")
        return True

    @property
    def outcomes(self) -> tuple[dict[str, Any], ...]:
        """Immutable snapshot of worker outcomes in completion order."""
        with self._lock:
            return tuple(self._outcomes)


@contextmanager
def _owned_threads(prefix: str, join_timeout: float) -> Any:
    """Yield a thread owner that is always closed on context exit."""
    owner = _ThreadOwner(prefix, join_timeout)
    try:
        yield owner
    finally:
        owner.close()


def _owner_report(owner: _ThreadOwner, names: Sequence[str]) -> dict[str, Any]:
    """Build a deterministic report from a closed owner's outcomes."""
    by_name = {item["name"]: item for item in owner.outcomes}
    values: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for name in sorted(names):
        item = by_name.get(name)
        if item is None:
            failures.append(
                {"code": "E004", "name": name, "error": "worker produced no outcome"}
            )
        elif item["ok"]:
            values[name] = item["value"]
        else:
            error = item["error"]
            failures.append(
                {
                    "code": "E004",
                    "name": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "values": values,
        "failures": tuple(failures),
        "thread_names": tuple(sorted(by_name)),
        "threads_joined": True,
    }


def _as_future_record(value: Any, index: int) -> dict[str, Any]:
    """Classify a mapping or future-like object as a settled record."""
    if isinstance(value, Mapping):
        name = _require_name(value.get("name", f"future-{index}"), "future name")
        error = value.get("error")
        status = value.get("status", "done" if "value" in value else "pending")
        if status not in _STATUSES:
            _fail("E003", f"future {name!r} has invalid status {status!r}")
        if error is not None:
            return {
                "index": index,
                "name": name,
                "settled": True,
                "ok": False,
                "value": None,
                "error": error,
            }
        if status in {"done", "failed", "cancelled"}:
            return {
                "index": index,
                "name": name,
                "settled": True,
                "ok": status == "done",
                "value": value.get("value"),
                "error": None if status == "done" else status,
            }
        return {
            "index": index,
            "name": name,
            "settled": False,
            "ok": False,
            "value": None,
            "error": None,
        }
    done = getattr(value, "done", None)
    result = getattr(value, "result", None)
    if not callable(done) or not callable(result):
        _fail("E003", f"future at index {index} has no done()/result() interface")
    if not done():
        return {
            "index": index,
            "name": f"future-{index}",
            "settled": False,
            "ok": False,
            "value": None,
            "error": None,
        }
    try:
        data = result()
    except BaseException as error:
        return {
            "index": index,
            "name": f"future-{index}",
            "settled": True,
            "ok": False,
            "value": None,
            "error": error,
        }
    return {
        "index": index,
        "name": f"future-{index}",
        "settled": True,
        "ok": True,
        "value": data,
        "error": None,
    }


def _cycle_free(adjacency: Mapping[str, tuple[str, ...]]) -> bool:
    """Return whether a directed adjacency map has no cycle (DFS colours)."""
    colors: dict[str, int] = {name: 0 for name in adjacency}

    def visit(name: str) -> bool:
        colors[name] = 1
        for child in adjacency[name]:
            if colors[child] == 1:
                return False
            if colors[child] == 0 and not visit(child):
                return False
        colors[name] = 2
        return True

    return all(colors[name] != 0 or visit(name) for name in sorted(adjacency))


# ---------------------------------------------------------------------------
# Public functions (ids 0711-0735)
# ---------------------------------------------------------------------------


def ctask_spawn_scoped(
    jobs: Mapping[str, Callable[[], Any]],
    *,
    max_workers: int = 4,
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Run named jobs in a bounded scope and join every worker before returning.

    ``max_workers`` caps concurrently executing jobs with a semaphore; all
    threads are still started up front and joined by the owned close path.  A
    raised job exception becomes an E004 failure record rather than escaping.
    """
    mapping = _require_mapping(jobs, "jobs")
    _require_int(max_workers, "max_workers", 1)
    _require_number(join_timeout, "join_timeout")
    if join_timeout <= 0:
        _fail("E003", "join_timeout must be > 0")
    names = _name_tuple(mapping, "job names")
    for name in names:
        _require_callable(mapping[name], f"job {name!r}")
    gate = threading.BoundedSemaphore(max_workers)
    with _owned_threads("ctask-scope", join_timeout) as owner:
        for name in names:
            owner.start(name, mapping[name], gate=gate)
    report = _owner_report(owner, names)
    report["max_workers"] = max_workers
    report["submitted"] = len(names)
    return report


def ctask_join_all(futures: Sequence[Any]) -> dict[str, Any]:
    """Collect already-settled futures in input order without blocking.

    Mapping records may carry ``value``, ``error`` and ``status`` keys; other
    objects must expose ``done()``/``result()``.  Pending entries are reported
    but never waited on, keeping the helper deterministic.
    """
    items = _require_sequence(futures, "futures")
    records = [_as_future_record(item, index) for index, item in enumerate(items)]
    values = tuple(
        record["value"] for record in records if record["settled"] and record["ok"]
    )
    failures = tuple(
        {
            "index": record["index"],
            "name": record["name"],
            "error": str(record["error"]),
        }
        for record in records
        if record["settled"] and not record["ok"]
    )
    pending = tuple(record["index"] for record in records if not record["settled"])
    return {
        "values": values,
        "failures": failures,
        "pending": pending,
        "all_done": not pending and not failures,
        "count": len(records),
    }


def ctask_join_first(
    futures: Sequence[Any],
    *,
    predicate: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    """Select the first settled future whose value satisfies *predicate*.

    The scan is ordered by input index, so the winner is reproducible.  A
    predicate is called only for successfully settled records.
    """
    items = _require_sequence(futures, "futures")
    if predicate is not None:
        _require_callable(predicate, "predicate")
    records = [_as_future_record(item, index) for index, item in enumerate(items)]
    settled = tuple(record["index"] for record in records if record["settled"])
    winner: dict[str, Any] | None = None
    for record in records:
        if not record["settled"] or not record["ok"]:
            continue
        if predicate is not None and not predicate(record["value"]):
            continue
        winner = dict(record)
        break
    return {
        "index": None if winner is None else winner["index"],
        "name": None if winner is None else winner["name"],
        "value": None if winner is None else winner["value"],
        "settled": settled,
        "remaining": tuple(
            index for index in range(len(records)) if index not in settled
        ),
        "all_done": len(settled) == len(records),
    }


def ctask_cancel_propagate(
    root: str,
    children: Mapping[str, Iterable[str]],
    *,
    cancelled: Iterable[str] = (),
) -> dict[str, Any]:
    """Propagate cancellation from *root* through a deterministic child tree.

    Traversal is breadth-first with lexicographic sibling order.  Cycles are
    rejected with E005, and a node cancelled before propagation is treated as
    an already-cancelled source with its own subtree.
    """
    _require_name(root, "root")
    mapping = _require_mapping(children, "children")
    adjacency: dict[str, tuple[str, ...]] = {}
    for name, kids in mapping.items():
        _require_name(name, "parent name")
        adjacency[name] = tuple(
            sorted({_require_name(kid, "child name") for kid in kids})
        )
    adjacency.setdefault(root, ())
    for name, kids in tuple(adjacency.items()):
        for kid in kids:
            adjacency.setdefault(kid, ())
    if not _cycle_free(adjacency):
        _fail("E005", "cancellation graph has a cycle")
    sources = [_require_name(name, "cancelled name") for name in cancelled]
    for source in sources:
        if source not in adjacency:
            _fail("E006", f"unknown cancelled task {source!r}")
    queue = deque([root, *sorted(set(sources))])
    order: list[str] = []
    seen: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        order.append(current)
        queue.extend(adjacency[current])
    return {
        "order": tuple(order),
        "cancelled": tuple(sorted(seen)),
        "skipped": tuple(sorted(set(adjacency) - seen)),
        "cooperative": True,
        "sources": tuple(sorted({root, *sources})),
    }


def ctask_cancel_scope(
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    reason: str = "cancelled",
) -> dict[str, Any]:
    """Request cooperative cancellation for pending and running scope tasks.

    Running tasks only receive a recorded request; the helper never interrupts
    their thread.  Finished tasks keep their recorded status.
    """
    mapping = _require_mapping(tasks, "tasks")
    _require_name(reason, "reason")
    cancelled: list[str] = []
    requests: list[str] = []
    unchanged: list[str] = []
    for name in sorted(mapping):
        _require_name(name, "task name")
        record = _require_mapping(mapping[name], f"task {name!r}")
        status = record.get("status")
        if status not in _STATUSES:
            _fail("E003", f"task {name!r} has invalid status {status!r}")
        if status == "pending":
            cancelled.append(name)
        elif status == "running":
            requests.append(name)
        else:
            unchanged.append(name)
    return {
        "cancelled": tuple(cancelled),
        "cancel_requested": tuple(requests),
        "unchanged": tuple(unchanged),
        "reason": reason,
        "cooperative": True,
        "counts": {
            "cancelled": len(cancelled),
            "requested": len(requests),
            "unchanged": len(unchanged),
        },
    }


def ctask_timeout_budget(
    total: int,
    weights: Mapping[str, int],
    *,
    clock: int = 0,
) -> dict[str, Any]:
    """Allocate an integer timeout budget by largest remainder.

    *clock* represents already-elapsed ticks and consumes the stages in name
    order, so callers can recompute the remaining budget without measuring real
    time.  Invalid budgets raise E003.
    """
    limit = _require_int(total, "total", 0)
    elapsed = _require_int(clock, "clock", 0)
    mapping = _require_mapping(weights, "weights")
    values = {
        _require_name(name, "weight name"): _require_int(weight, f"weight {name!r}", 0)
        for name, weight in mapping.items()
    }
    names = tuple(sorted(values))
    weight_sum = sum(values.values())
    base: dict[str, int] = {}
    if weight_sum:
        exact = {name: limit * values[name] / weight_sum for name in names}
        base = {name: int(exact[name]) for name in names}
        leftover = limit - sum(base.values())
        ranked = sorted(names, key=lambda name: (-(exact[name] - base[name]), name))
        for name in ranked[:leftover]:
            base[name] += 1
    else:
        base = {name: 0 for name in names}
    remaining_clock = elapsed
    adjusted = dict(base)
    for name in names:
        spent = min(adjusted[name], remaining_clock)
        adjusted[name] -= spent
        remaining_clock -= spent
    return {
        "budgets": adjusted,
        "base_budgets": base,
        "allocated": sum(adjusted.values()),
        "consumed": elapsed - remaining_clock,
        "expired": remaining_clock > 0,
        "stages": names,
    }


def ctask_retry_policy(
    policy: Mapping[str, Any],
    *,
    failures: int = 0,
) -> dict[str, Any]:
    """Compute a deterministic retry decision and delay schedule.

    The policy accepts ``max_attempts``, ``base_delay``, ``factor``,
    ``max_delay`` and an optional ``jitter`` sequence.  All arithmetic is on
    integers, which keeps schedules reproducible across platforms.
    """
    settings = _require_mapping(policy, "policy")
    count = _require_int(failures, "failures", 0)
    attempts = _require_int(settings.get("max_attempts", 3), "max_attempts", 1)
    base = _require_int(settings.get("base_delay", 1), "base_delay", 0)
    factor = _require_int(settings.get("factor", 2), "factor", 1)
    cap = _require_int(settings.get("max_delay", 16), "max_delay", 0)
    jitter = _require_sequence(settings.get("jitter", ()), "jitter")
    jitter_values = tuple(_require_int(value, "jitter value", 0) for value in jitter)
    schedule: list[int] = []
    for index in range(attempts):
        raw = base * (factor**index)
        extra = jitter_values[index] if index < len(jitter_values) else 0
        schedule.append(min(cap, raw) + extra)
    next_attempt = count + 1
    return {
        "attempt": next_attempt,
        "max_attempts": attempts,
        "delay": schedule[count] if count < attempts else 0,
        "should_retry": next_attempt <= attempts,
        "schedule": tuple(schedule),
        "capped": cap,
    }


def ctask_retry_run(
    attempts: Sequence[Callable[[], Any]],
    *,
    should_retry: Callable[[BaseException], bool] | None = None,
    clock: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Execute retry attempts synchronously with an injectable tick source.

    No sleeping occurs: delays are reported as ticks.  An attempt failure is
    retried while *should_retry* returns true and attempts remain; the final
    failure is returned as structured E004 data rather than raised.
    """
    steps = _require_sequence(attempts, "attempts")
    for index, step in enumerate(steps):
        _require_callable(step, f"attempt {index}")
    if should_retry is not None:
        _require_callable(should_retry, "should_retry")
    if clock is not None:
        _require_callable(clock, "clock")
    errors: list[dict[str, Any]] = []
    value: Any = None
    succeeded = False
    ticks_before = 0 if clock is None else clock()
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
            if should_retry is None or not should_retry(error):
                break
            continue
        succeeded = True
        break
    ticks_after = 0 if clock is None else clock()
    return {
        "succeeded": succeeded,
        "value": value,
        "attempts_run": len(errors) + int(succeeded),
        "errors": tuple(errors),
        "elapsed_ticks": ticks_after - ticks_before,
    }


def ctask_dependency_graph(specs: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """Validate task dependencies and return normalized layers and edges.

    Unknown dependencies raise E006 and dependency cycles raise E005.  Layers
    are computed with Kahn's algorithm over lexicographically sorted nodes.
    """
    graph, layers = _normal_graph(specs)
    edges = tuple(sorted((dep, name) for name, deps in graph.items() for dep in deps))
    return {
        "graph": {name: tuple(deps) for name, deps in sorted(graph.items())},
        "layers": layers,
        "roots": layers[0] if layers else (),
        "edges": edges,
        "task_count": len(graph),
    }


def ctask_topological_schedule(
    graph: Mapping[str, Iterable[str]],
    *,
    priorities: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Order tasks topologically, breaking ties by priority then name.

    Priorities are optional; missing entries default to zero.  Cycles are
    reported as E005 with the offending tasks in the error reason.
    """
    normalized, layers = _normal_graph(graph)
    ranking: dict[str, int] = {}
    if priorities is not None:
        for name, value in _require_mapping(priorities, "priorities").items():
            ranking[name] = _require_int(value, f"priority {name!r}", 0)
    ordered_layers = tuple(
        tuple(sorted(layer, key=lambda name: (ranking.get(name, 0), name)))
        for layer in layers
    )
    order = tuple(name for layer in ordered_layers for name in layer)
    return {
        "order": order,
        "layers": ordered_layers,
        "count": len(normalized),
        "edges": sum(len(deps) for deps in normalized.values()),
    }


def ctask_graph_ready_set(
    graph: Mapping[str, Iterable[str]],
    completed: Iterable[str],
) -> dict[str, Any]:
    """Report tasks whose dependencies are satisfied by *completed*.

    Completed names must belong to the graph (E006).  The result separates
    ready, blocked and already-completed task names deterministically.
    """
    normalized, _ = _normal_graph(graph)
    done = _name_tuple(completed, "completed")
    for name in done:
        if name not in normalized:
            _fail("E006", f"unknown completed task {name!r}")
    done_set = set(done)
    ready: list[str] = []
    blocked: list[str] = []
    for name, deps in normalized.items():
        if name in done_set:
            continue
        (ready if set(deps) <= done_set else blocked).append(name)
    return {
        "ready": tuple(sorted(ready)),
        "blocked": tuple(sorted(blocked)),
        "completed": done,
        "remaining": len(ready) + len(blocked),
    }


def ctask_graph_wave_run(
    graph: Mapping[str, Iterable[str]],
    actions: Mapping[str, Callable[[], Any]],
    *,
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Execute graph layers wave by wave, joining every wave's threads.

    Each dependency layer gets its own owned thread group, which is closed
    before the next wave starts.  Missing actions raise E006 and worker
    exceptions are returned as E004 failures.
    """
    _, layers = _normal_graph(graph)
    callable_map = _require_mapping(actions, "actions")
    for name in callable_map:
        _require_callable(callable_map[name], f"action {name!r}")
    values: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for layer in layers:
        for name in layer:
            if name not in callable_map:
                _fail("E006", f"no action registered for task {name!r}")
        with _owned_threads("ctask-wave", join_timeout) as owner:
            for name in layer:
                owner.start(name, callable_map[name])
        report = _owner_report(owner, layer)
        values.update(report["values"])
        failures.extend(report["failures"])
    return {
        "values": {name: values[name] for name in sorted(values)},
        "failures": tuple(failures),
        "waves": layers,
        "threads_joined": True,
    }


def ctask_future_settle(
    record: Mapping[str, Any],
    *,
    value: Any = None,
    error: Any = None,
) -> dict[str, Any]:
    """Settle one future record with a value or an error.

    Only pending and running records can be settled; attempting to settle a
    finished record raises E006 so double-settlement cannot be silent.
    """
    mapping = _require_mapping(record, "record")
    name = _require_name(mapping.get("name"), "future name")
    status = mapping.get("status", "pending")
    if status not in {"pending", "running"}:
        _fail("E006", f"future {name!r} is already settled")
    settled = "failed" if error is not None else "done"
    updated = dict(mapping)
    updated["status"] = settled
    updated["value"] = value if error is None else None
    updated["error"] = error
    return {
        "record": updated,
        "name": name,
        "status": settled,
        "changed": updated != mapping,
    }


def ctask_future_chain(
    records: Sequence[Mapping[str, Any]],
    transforms: Sequence[Callable[[Any], Any]],
) -> dict[str, Any]:
    """Thread values through transformations until a record is not done.

    Each step records the input name, produced value and status.  A transform
    that raises stops the chain and is captured as an E004 step.
    """
    items = _require_sequence(records, "records")
    hooks = _require_sequence(transforms, "transforms")
    for index, hook in enumerate(hooks):
        _require_callable(hook, f"transform {index}")
    steps: list[dict[str, Any]] = []
    value: Any = None
    failed_at: int | None = None
    for index, item in enumerate(items):
        mapping = _require_mapping(item, f"record {index}")
        status = mapping.get("status", "pending")
        if status not in _STATUSES:
            _fail("E003", f"record {index} has invalid status {status!r}")
        if status != "done":
            steps.append(
                {"index": index, "name": mapping.get("name"), "status": status}
            )
            failed_at = index
            break
        value = mapping.get("value")
        if index < len(hooks):
            try:
                value = hooks[index](value)
            except BaseException as error:
                steps.append(
                    {
                        "index": index,
                        "name": mapping.get("name"),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                failed_at = index
                break
        steps.append(
            {
                "index": index,
                "name": mapping.get("name"),
                "status": "done",
                "value": value,
            }
        )
    return {
        "value": value,
        "steps": tuple(steps),
        "failed_at": failed_at,
        "completed": failed_at is None and len(steps) == len(items),
    }


def ctask_result_collect(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Partition outcome records into values, failures and pending entries.

    A record is successful when ``ok`` is true or its status is ``done``;
    anything carrying ``error`` is a failure and everything else is pending.
    """
    collected: list[dict[str, Any]] = []
    for index, item in enumerate(outcomes):
        record = _require_mapping(item, f"outcome {index}")
        name = _require_name(record.get("name", f"outcome-{index}"), "outcome name")
        error = record.get("error")
        status = record.get("status")
        explicit = record.get("ok")
        if error is not None or explicit is False:
            kind = "failures"
        elif explicit is True or status == "done":
            kind = "values"
        else:
            kind = "pending"
        collected.append({"kind": kind, "name": name, "record": record})
    groups = {
        "values": tuple(
            record["record"].get("value")
            for record in collected
            if record["kind"] == "values"
        ),
        "failures": tuple(
            {"name": record["name"], "error": str(record["record"].get("error"))}
            for record in collected
            if record["kind"] == "failures"
        ),
        "pending": tuple(
            record["name"] for record in collected if record["kind"] == "pending"
        ),
    }
    counts = {key: len(value) for key, value in groups.items()}
    return {**groups, "counts": counts, "total": len(collected)}


def ctask_error_aggregate(
    errors: Iterable[Mapping[str, Any]],
    *,
    code: str = "E004",
) -> dict[str, Any]:
    """Aggregate failure records into a sorted, structured summary.

    The fallback *code* is applied to records that omit one; unknown E-codes
    raise E003.  Groups are keyed by code and ordered for stable output.
    """
    if code not in _CODES:
        _fail("E003", f"unsupported diagnostic code {code!r}")
    grouped: dict[str, list[str]] = {}
    count = 0
    for index, item in enumerate(errors):
        record = _require_mapping(item, f"error {index}")
        entry_code = record.get("code", code)
        if entry_code not in _CODES:
            _fail("E003", f"unsupported diagnostic code {entry_code!r}")
        name = _require_name(record.get("name", f"error-{index}"), "error name")
        grouped.setdefault(entry_code, []).append(
            f"{name}: {record.get('message', '')}"
        )
        count += 1
    by_code = {key: tuple(sorted(value)) for key, value in sorted(grouped.items())}
    primary = None
    if by_code:
        first_code = next(iter(by_code))
        primary = {"code": first_code, "message": by_code[first_code][0]}
    summary = "; ".join(f"{key}({len(value)})" for key, value in by_code.items())
    return {
        "count": count,
        "by_code": by_code,
        "primary": primary,
        "summary": summary,
        "clean": count == 0,
    }


def ctask_pause_resume(
    events: Sequence[str],
    *,
    initial: str = "running",
) -> dict[str, Any]:
    """Apply a deterministic pause/resume/cancel state machine.

    Cancellation is terminal; pause and resume are no-ops in incompatible
    states.  Every input event produces a history entry with before/after
    states, so the trace is easy to assert on.
    """
    if initial not in {"running", "paused", "cancelled"}:
        _fail("E003", f"invalid initial state {initial!r}")
    steps = _require_sequence(events, "events")
    state = initial
    history: list[dict[str, Any]] = []
    pauses = 0
    resumes = 0
    for index, event in enumerate(steps):
        if event not in {"pause", "resume", "cancel"}:
            _fail("E003", f"invalid event {event!r} at index {index}")
        before = state
        if state != "cancelled":
            if event == "pause" and state == "running":
                state = "paused"
                pauses += 1
            elif event == "resume" and state == "paused":
                state = "running"
                resumes += 1
            elif event == "cancel":
                state = "cancelled"
        history.append(
            {
                "index": index,
                "event": event,
                "before": before,
                "after": state,
                "changed": before != state,
            }
        )
    return {
        "state": state,
        "history": tuple(history),
        "pauses": pauses,
        "resumes": resumes,
        "cancelled": state == "cancelled",
    }


def ctask_yield_round_robin(
    queues: Mapping[str, Sequence[Any]],
    *,
    turns: int,
) -> dict[str, Any]:
    """Consume at most *turns* items from named queues in round-robin order.

    Empty queues are skipped, and iteration always begins with the smallest
    queue name.  The remaining queues are returned as tuples for comparison.
    """
    mapping = _require_mapping(queues, "queues")
    count = _require_int(turns, "turns", 0)
    pending = {
        _require_name(name, "queue name"): deque(
            _require_sequence(items, f"queue {name!r}")
        )
        for name, items in mapping.items()
    }
    names = tuple(sorted(pending))
    order: list[tuple[str, Any]] = []
    cursor = 0
    while len(order) < count and any(pending[name] for name in names):
        name = names[cursor % len(names)]
        cursor += 1
        if pending[name]:
            order.append((name, pending[name].popleft()))
    return {
        "order": tuple(order),
        "remaining": {name: tuple(pending[name]) for name in names},
        "turns_taken": len(order),
        "exhausted": not any(pending[name] for name in names),
    }


def ctask_priority_queue(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Order named items by priority, sequence then name with a size limit.

    Lower priority values run first.  When *limit* is set, the tail is dropped
    and reported explicitly instead of being silently discarded.
    """
    entries = _require_sequence(items, "items")
    maximum = None if limit is None else _require_int(limit, "limit", 0)
    ordered: list[tuple[int, int, str, Mapping[str, Any]]] = []
    for index, item in enumerate(entries):
        mapping = _require_mapping(item, f"item {index}")
        name = _require_name(mapping.get("name"), "item name")
        priority = _require_int(mapping.get("priority", 0), f"priority of {name!r}", 0)
        sequence = _require_int(mapping.get("seq", index), f"seq of {name!r}", 0)
        ordered.append((priority, sequence, name, mapping))
    ordered.sort(key=lambda entry: entry[:3])
    chosen = ordered if maximum is None else ordered[:maximum]
    dropped = () if maximum is None else tuple(entry[2] for entry in ordered[maximum:])
    return {
        "ordered": tuple(entry[2] for entry in chosen),
        "items": tuple(entry[3] for entry in chosen),
        "dropped": dropped,
        "limit": maximum,
    }


def ctask_rate_limit(
    ticks: Sequence[int],
    *,
    capacity: int,
    refill: int,
) -> dict[str, Any]:
    """Model a token bucket over injected integer ticks.

    Tokens refill by ``refill * elapsed_ticks`` and never exceed *capacity*.
    Ticks must be non-negative and non-decreasing, otherwise E003 is raised.
    """
    moments = _require_sequence(ticks, "ticks")
    cap = _require_int(capacity, "capacity", 1)
    amount = _require_int(refill, "refill", 0)
    values = [_require_int(tick, "tick", 0) for tick in moments]
    if any(later < earlier for earlier, later in pairwise(values)):
        _fail("E003", "ticks must be non-decreasing")
    tokens = cap
    previous = values[0] if values else 0
    timeline: list[dict[str, Any]] = []
    allowed: list[bool] = []
    for tick in values:
        tokens = min(cap, tokens + amount * (tick - previous))
        previous = tick
        granted = tokens >= 1
        if granted:
            tokens -= 1
        allowed.append(granted)
        timeline.append({"tick": tick, "allowed": granted, "tokens": tokens})
    return {
        "allowed": tuple(allowed),
        "tokens": tokens,
        "rejected": allowed.count(False),
        "timeline": tuple(timeline),
        "capacity": cap,
    }


def ctask_backoff_plan(
    failures: int,
    *,
    base: int,
    factor: int,
    cap: int,
    jitter: Sequence[int] = (),
) -> dict[str, Any]:
    """Compute exponential backoff delays with deterministic jitter.

    Each delay is ``min(cap, base * factor**i)`` plus an optional supplied
    jitter value.  No randomness is used, so plans are identical per inputs.
    """
    count = _require_int(failures, "failures", 0)
    start = _require_int(base, "base", 0)
    growth = _require_int(factor, "factor", 1)
    ceiling = _require_int(cap, "cap", 0)
    steps = _require_sequence(jitter, "jitter")
    noise = tuple(_require_int(value, "jitter value", 0) for value in steps)
    delays: list[int] = []
    capped_at: int | None = None
    for index in range(count):
        raw = start * (growth**index)
        if raw > ceiling and capped_at is None:
            capped_at = index
        delays.append(min(raw, ceiling) + (noise[index] if index < len(noise) else 0))
    return {
        "delays": tuple(delays),
        "total": sum(delays),
        "capped_at": capped_at,
        "max_delay": ceiling,
    }


def ctask_deadline_check(
    now: int,
    *,
    deadline: int,
    started: int = 0,
) -> dict[str, Any]:
    """Compute remaining budget and expiry from integer ticks.

    The deadline must not precede the start tick (E003).  Overruns are
    reported separately from the elapsed time for clearer diagnostics.
    """
    current = _require_int(now, "now", 0)
    end = _require_int(deadline, "deadline", 0)
    beginning = _require_int(started, "started", 0)
    if end < beginning:
        _fail("E003", "deadline must not precede started")
    return {
        "remaining": max(0, end - current),
        "elapsed": max(0, current - beginning),
        "budget": end - beginning,
        "expired": current >= end,
        "overrun": max(0, current - end),
    }


def ctask_shutdown_drain(
    tasks: Mapping[str, Callable[[], Any]],
    *,
    cancel_queued: Iterable[str] = (),
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Run uncancelled shutdown tasks and join all started threads.

    Queued names listed in *cancel_queued* are never started.  Running work
    is not killed; worker exceptions are reported under E004 in the result.
    """
    mapping = _require_mapping(tasks, "tasks")
    queued = _name_tuple(cancel_queued, "cancel_queued")
    for name in queued:
        if name not in mapping:
            _fail("E006", f"unknown queued task {name!r}")
    names = _name_tuple(mapping, "task names")
    for name in names:
        _require_callable(mapping[name], f"task {name!r}")
    started = tuple(name for name in names if name not in queued)
    with _owned_threads("ctask-drain", join_timeout) as owner:
        for name in started:
            owner.start(name, mapping[name])
    report = _owner_report(owner, started)
    return {
        "completed": tuple(report["values"]),
        "values": report["values"],
        "failures": report["failures"],
        "cancelled": queued,
        "threads_joined": True,
    }


def ctask_task_tree(nodes: Mapping[str, str | None]) -> dict[str, Any]:
    """Build a task parent/child tree with depths and subtree sizes.

    Unknown parents raise E006 and parent cycles raise E005.  Children and
    roots are sorted so the resulting tree is stable across runs.
    """
    mapping = _require_mapping(nodes, "nodes")
    parents: dict[str, str | None] = {}
    for name, parent in mapping.items():
        _require_name(name, "task name")
        if parent is not None:
            _require_name(parent, "parent name")
            if parent not in mapping:
                _fail("E006", f"task {name!r} has unknown parent {parent!r}")
        parents[name] = parent
    children: dict[str, tuple[str, ...]] = {
        name: tuple(
            sorted(child for child, parent in parents.items() if parent == name)
        )
        for name in parents
    }
    roots = tuple(sorted(name for name, parent in parents.items() if parent is None))
    depths: dict[str, int] = {}
    for root in roots:
        stack = [(root, 0)]
        while stack:
            name, depth = stack.pop()
            if name in depths:
                _fail("E005", "task tree has a cycle")
            depths[name] = depth
            stack.extend((child, depth + 1) for child in reversed(children[name]))
    if len(depths) != len(parents):
        _fail("E005", "task tree has a cycle or unreachable nodes")
    sizes: dict[str, int] = {}

    def subtree(name: str) -> int:
        total = 1 + sum(subtree(child) for child in children[name])
        sizes[name] = total
        return total

    for root in roots:
        subtree(root)
    return {
        "children": {name: children[name] for name in sorted(children)},
        "roots": roots,
        "depths": {name: depths[name] for name in sorted(depths)},
        "sizes": {name: sizes[name] for name in sorted(sizes)},
        "count": len(parents),
    }


def ctask_trace_events(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize task lifecycle records into a sequence-ordered trace.

    Records are sorted by their ``seq`` key and then by name, with the input
    order used as a final tie-breaker.  Event kinds are counted in sorted key
    order for deterministic reporting.
    """
    entries = _require_sequence(records, "records")
    stamped: list[tuple[int, str, int, Mapping[str, Any]]] = []
    for index, item in enumerate(entries):
        mapping = _require_mapping(item, f"record {index}")
        sequence = _require_int(mapping.get("seq"), f"seq of record {index}", 0)
        _require_name(mapping.get("name"), f"name of record {index}")
        kind = _require_name(mapping.get("kind", "event"), f"kind of record {index}")
        stamped.append((sequence, kind, index, mapping))
    stamped.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    events = tuple(
        {
            "seq": sequence,
            "step": index,
            "kind": kind,
            "name": mapping.get("name"),
            "detail": mapping.get("detail"),
        }
        for index, (sequence, kind, _, mapping) in enumerate(stamped)
    )
    counts: dict[str, int] = {}
    for event in events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    return {
        "events": events,
        "sequence": tuple(event["seq"] for event in events),
        "counts": {key: counts[key] for key in sorted(counts)},
        "count": len(events),
    }
