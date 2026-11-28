"""Resource lifecycle, shutdown and pool helpers for batch eight.

The registries in this module only ever see resources explicitly wrapped by
these helpers; no process-wide descriptor or handle enumeration is performed.
Threads are started exclusively through an owned group whose close path joins
every worker, and every context manager closes or drains its resource when the
body raises.  Audit and shutdown inputs are validated with the shared
``E001``–``E008`` diagnostic surface.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from compiler.frontend.errors import PolyglotError

__all__ = [
    "clife_finalize_report",
    "clife_guard_stack",
    "clife_leak_audit",
    "clife_poison_check",
    "clife_poison_mark",
    "clife_pool_context",
    "clife_pool_create",
    "clife_pool_drain",
    "clife_pool_submit",
    "clife_resource_acquire",
    "clife_resource_context",
    "clife_resource_register",
    "clife_resource_release",
    "clife_shutdown_plan",
    "clife_shutdown_run",
]

_CODES = frozenset(f"E{value:03d}" for value in range(1, 10))
_RELEASERS = ("close", "release", "shutdown", "stop")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    """Raise a structured :class:`PolyglotError` for this module."""
    if code not in _CODES:
        raise ValueError(f"unknown diagnostic code {code!r}")
    raise PolyglotError(
        code, message, reason=reason, note="runtime.concurrency.lifecycle"
    )


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
    """Validate and sort unique names (E003 on duplicates)."""
    if isinstance(values, Mapping):
        items = tuple(values)
    else:
        items = _require_sequence(values, label)
    names = {_require_name(item, f"{label} item") for item in items}
    if len(names) != len(items):
        _fail("E003", f"{label} contains duplicates")
    return tuple(sorted(names))


@dataclass
class _Resource:
    """A wrapped resource with a deterministic lifecycle state."""

    name: str
    kind: str
    resource: Any
    closer: Callable[[Any], None]
    state: str = "registered"
    error: str | None = None


@dataclass
class _Registry:
    """Registry of wrapped resources in registration order."""

    resources: dict[str, _Resource] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


@dataclass
class _Pool:
    """Pool state with a queued job list and explicit closed flag."""

    name: str
    capacity: int
    jobs: list[tuple[str, Callable[[], Any]]] = field(default_factory=list)
    closed: bool = False


@dataclass
class _ThreadOwner:
    """Joinable worker group owned by the pool helpers."""

    prefix: str
    join_timeout: float
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _threads: list[threading.Thread] = field(default_factory=list)
    _outcomes: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def start(self, name: str, fn: Callable[[], Any]) -> None:
        """Start a named job and record its outcome for the final report."""
        if self._closed:
            _fail("E003", "thread owner is already closed")

        def runner() -> None:
            try:
                value = fn()
            except BaseException as error:
                outcome = {"name": name, "ok": False, "error": error}
            else:
                outcome = {"name": name, "ok": True, "value": value}
            with self._lock:
                self._outcomes.append(outcome)

        thread = threading.Thread(
            target=runner, name=f"{self.prefix}-{name}", daemon=False
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def close(self) -> bool:
        """Join every started worker and reject any thread that is still alive."""
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            threads = tuple(self._threads)
        for thread in threads:
            thread.join(timeout=self.join_timeout)
        alive = tuple(thread.name for thread in threads if thread.is_alive())
        if alive:
            _fail("E004", "pool worker did not stop", reason=f"alive: {alive}")
        return True

    @property
    def outcomes(self) -> tuple[dict[str, Any], ...]:
        """Immutable snapshot of job outcomes."""
        with self._lock:
            return tuple(self._outcomes)

    @property
    def thread_names(self) -> tuple[str, ...]:
        """Names of all started workers in start order."""
        with self._lock:
            return tuple(thread.name for thread in self._threads)


@contextmanager
def _owned_threads(prefix: str, join_timeout: float) -> Any:
    """Yield an owner that always closes on context exit."""
    owner = _ThreadOwner(prefix, join_timeout)
    try:
        yield owner
    finally:
        owner.close()


def _new_registry() -> dict[str, Any]:
    """Create a fresh wrapped-resource registry."""
    return {"registry": _Registry()}


def _require_registry(value: object) -> dict[str, Any]:
    """Validate an envelope created by :func:`clife_resource_register`."""
    if not isinstance(value, dict) or not isinstance(value.get("registry"), _Registry):
        _fail("E003", "registry must come from clife_resource_register")
    return value


def _require_pool(value: object) -> dict[str, Any]:
    """Validate an envelope created by :func:`clife_pool_create`."""
    if not isinstance(value, dict) or not isinstance(value.get("pool"), _Pool):
        _fail("E003", "pool must come from clife_pool_create")
    return value


def _infer_closer(resource: Any, name: str) -> Callable[[Any], None]:
    """Find a deterministic close method or report E003."""
    method = next(
        (
            candidate
            for candidate in _RELEASERS
            if callable(getattr(resource, candidate, None))
        ),
        None,
    )
    if method is None:
        _fail("E003", f"no closer available for {name!r}")
    return lambda obj: getattr(obj, method)()


def _graph_layers(components: Mapping[str, Iterable[str]]) -> tuple[str, ...]:
    """Return a topological startup order, raising E005/E006 on bad graphs."""
    mapping = _require_mapping(components, "components")
    graph: dict[str, tuple[str, ...]] = {}
    for name, deps in mapping.items():
        key = _require_name(name, "component name")
        values = _require_sequence(deps, f"dependencies of {key!r}")
        graph[key] = tuple(
            sorted({_require_name(dep, "dependency name") for dep in values})
        )
    for name, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                _fail("E006", f"component {name!r} depends on unknown {dep!r}")
    remaining = {name: set(deps) for name, deps in graph.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            _fail("E005", "shutdown graph has a cycle")
        for name in ready:
            order.append(name)
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)
    return tuple(order)


# ---------------------------------------------------------------------------
# Public functions (ids 0786-0800)
# ---------------------------------------------------------------------------


def clife_resource_register(
    resource: Any,
    *,
    name: str,
    kind: str = "generic",
    closer: Callable[[Any], None] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a resource with a lifecycle state and register it for audits."""
    envelope = _new_registry() if registry is None else _require_registry(registry)
    label = _require_name(name, "name")
    if kind not in {"generic", "file", "socket", "tempdir"}:
        _fail("E003", f"unsupported resource kind {kind!r}")
    target: _Registry = envelope["registry"]
    if label in target.resources:
        _fail("E006", f"resource {label!r} is already registered")
    close = closer if closer is not None else _infer_closer(resource, label)
    _require_callable(close, "closer")
    wrapped = _Resource(label, kind, resource, close)
    target.resources[label] = wrapped
    target.order.append(label)
    return {
        "registry": envelope,
        "resource": wrapped,
        "name": label,
        "state": wrapped.state,
    }


def clife_resource_acquire(
    registry: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Move a registered or released resource into the acquired state."""
    envelope = _require_registry(registry)
    label = _require_name(name, "name")
    target: _Registry = envelope["registry"]
    wrapped = target.resources.get(label)
    if wrapped is None:
        _fail("E006", f"unknown resource {label!r}")
    if wrapped.state == "acquired":
        _fail("E003", f"resource {label!r} is already acquired")
    previous = wrapped.state
    wrapped.state = "acquired"
    return {
        "name": label,
        "previous": previous,
        "state": wrapped.state,
        "resource": wrapped,
    }


def clife_resource_release(
    registry: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Close an acquired resource once and record any closer failure."""
    envelope = _require_registry(registry)
    label = _require_name(name, "name")
    target: _Registry = envelope["registry"]
    wrapped = target.resources.get(label)
    if wrapped is None:
        _fail("E006", f"unknown resource {label!r}")
    if wrapped.state != "acquired":
        _fail("E003", f"resource {label!r} is not acquired")
    try:
        wrapped.closer(wrapped.resource)
    except BaseException as error:
        wrapped.error = f"{type(error).__name__}: {error}"
    wrapped.state = "released"
    return {
        "name": label,
        "state": wrapped.state,
        "closed": wrapped.error is None,
        "error": wrapped.error,
        "resource": wrapped,
    }


@contextmanager
def clife_resource_context(
    registry: dict[str, Any],
    name: str,
) -> Any:
    """Yield an acquired resource envelope, releasing it even on a body raise."""
    acquired = clife_resource_acquire(registry, name)
    envelope = {
        "name": acquired["name"],
        "resource": acquired["resource"],
        "released": False,
    }
    try:
        yield envelope
    finally:
        clife_resource_release(registry, name)
        envelope["released"] = True


def clife_shutdown_plan(
    components: Mapping[str, Iterable[str]],
    *,
    roots: Iterable[str] = (),
) -> dict[str, Any]:
    """Compute a reverse-topological shutdown order for wrapped components.

    ``components[name]`` lists the components that ``name`` depends on, so
    dependents are stopped before their dependencies.  Unknown roots raise
    E006 and dependency cycles raise E005.
    """
    startup = _graph_layers(components)
    chosen = _name_tuple(roots, "roots")
    for root in chosen:
        if root not in startup:
            _fail("E006", f"unknown root component {root!r}")
    shutdown = tuple(reversed(startup))
    return {
        "order": shutdown,
        "startup": startup,
        "roots": chosen,
        "count": len(shutdown),
    }


def clife_shutdown_run(
    steps: Sequence[tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    """Run shutdown callbacks in the given order, collecting E004 failures."""
    entries = _require_sequence(steps, "steps")
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    order: list[str] = []
    for index, item in enumerate(entries):
        pair = _require_sequence(item, f"step {index}")
        if len(pair) != 2:
            _fail("E003", f"step {index} must be a (name, callback) pair")
        name = _require_name(pair[0], f"step {index} name")
        callback = _require_callable(pair[1], f"callback of {name!r}")
        order.append(name)
        try:
            callback()
        except BaseException as error:
            failed.append(
                {
                    "code": "E004",
                    "name": name,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
        else:
            completed.append(name)
    return {
        "order": tuple(order),
        "completed": tuple(completed),
        "failed": tuple(failed),
        "all_done": not failed,
        "count": len(order),
    }


def clife_pool_create(
    name: str = "pool",
    *,
    capacity: int = 4,
) -> dict[str, Any]:
    """Create an in-memory worker-pool state with an explicit capacity."""
    label = _require_name(name, "name")
    limit = _require_int(capacity, "capacity", 1)
    pool = _Pool(label, limit)
    return {"pool": pool, "name": label, "capacity": limit, "closed": pool.closed}


def clife_pool_submit(
    pool: dict[str, Any],
    name: str,
    job: Callable[[], Any],
) -> dict[str, Any]:
    """Queue a named job on a live pool (E006 once the pool is closed)."""
    envelope = _require_pool(pool)
    state: _Pool = envelope["pool"]
    label = _require_name(name, "name")
    _require_callable(job, "job")
    if state.closed:
        _fail("E006", f"pool {state.name!r} is closed")
    if any(existing == label for existing, _ in state.jobs):
        _fail("E003", f"job {label!r} is already queued")
    state.jobs.append((label, job))
    return {
        "pool": envelope,
        "name": label,
        "job_id": len(state.jobs),
        "queued": len(state.jobs),
    }


def clife_pool_drain(
    pool: dict[str, Any],
    *,
    cancel_queued: Iterable[str] = (),
    join_timeout: float = 2.0,
) -> dict[str, Any]:
    """Start uncancelled jobs, join every worker and close the pool."""
    envelope = _require_pool(pool)
    state: _Pool = envelope["pool"]
    cancelled = _name_tuple(cancel_queued, "cancel_queued")
    for name in cancelled:
        if not any(existing == name for existing, _ in state.jobs):
            _fail("E006", f"unknown queued job {name!r}")
    if not isinstance(join_timeout, (int, float)) or isinstance(join_timeout, bool):
        _fail("E003", "join_timeout must be a number")
    if join_timeout <= 0:
        _fail("E003", "join_timeout must be > 0")
    if state.closed:
        return {
            "completed": (),
            "values": {},
            "failures": (),
            "cancelled": cancelled,
            "threads_joined": True,
            "already_closed": True,
        }
    running = [(name, job) for name, job in state.jobs if name not in cancelled]
    values: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    with _owned_threads("clife-pool", join_timeout) as owner:
        for name, job in running:
            owner.start(name, job)
    for outcome in owner.outcomes:
        if outcome["ok"]:
            values[outcome["name"]] = outcome["value"]
        else:
            error = outcome["error"]
            failures.append(
                {
                    "code": "E004",
                    "name": outcome["name"],
                    "message": f"{type(error).__name__}: {error}",
                }
            )
    state.jobs.clear()
    state.closed = True
    envelope["closed"] = True
    return {
        "completed": tuple(sorted(values)),
        "values": {name: values[name] for name in sorted(values)},
        "failures": tuple(sorted(failures, key=lambda item: item["name"])),
        "cancelled": cancelled,
        "threads_joined": True,
        "already_closed": False,
    }


@contextmanager
def clife_pool_context(
    capacity: int = 4,
    *,
    name: str = "pool",
    join_timeout: float = 2.0,
) -> Any:
    """Yield a pool that is drained and joined when the ``with`` block exits."""
    envelope = clife_pool_create(name, capacity=capacity)
    envelope["closed"] = False
    envelope["threads_joined"] = False
    try:
        yield envelope
    finally:
        report = clife_pool_drain(envelope, join_timeout=join_timeout)
        envelope["closed"] = True
        envelope["threads_joined"] = report["threads_joined"]
        envelope["last_report"] = report


def clife_poison_mark(
    state: Mapping[str, Any],
    *,
    reason: str,
    owner: str | None = None,
) -> dict[str, Any]:
    """Mark a state record as poisoned with a structured, deterministic reason."""
    original = _require_mapping(state, "state")
    message = _require_name(reason, "reason")
    poison = {"reason": message, "owner": owner, "kind": original.get("kind", "state")}
    updated = dict(original)
    updated["poisoned"] = True
    updated["poison"] = poison
    return {"state": updated, "poison": poison, "marked": True}


def clife_poison_check(
    poison: Mapping[str, Any] | None,
    *,
    raise_on_poison: bool = False,
) -> dict[str, Any]:
    """Check a poison record and optionally raise E004 with it."""
    if poison is None:
        return {"poisoned": False, "reports": (), "clean": True}
    record = _require_mapping(poison, "poison")
    report = {
        "reason": record.get("reason", "poisoned"),
        "owner": record.get("owner"),
        "kind": record.get("kind", "state"),
    }
    if raise_on_poison:
        _fail("E004", "lifecycle poison detected", reason=str(report["reason"]))
    return {"poisoned": True, "reports": (report,), "clean": False}


def clife_leak_audit(
    registry: dict[str, Any],
    *,
    close_all: bool = False,
) -> dict[str, Any]:
    """Audit only wrapped resources, optionally closing every open one."""
    envelope = _require_registry(registry)
    target: _Registry = envelope["registry"]
    open_names: list[str] = []
    closed_names: list[str] = []
    errors: list[str] = []
    for name in target.order:
        wrapped = target.resources[name]
        if wrapped.state != "released":
            if close_all:
                try:
                    wrapped.closer(wrapped.resource)
                except BaseException as error:
                    wrapped.error = f"{type(error).__name__}: {error}"
                wrapped.state = "released"
                if wrapped.error is None:
                    closed_names.append(name)
            else:
                open_names.append(name)
        else:
            closed_names.append(name)
        if wrapped.error:
            errors.append(f"{name}: {wrapped.error}")
    return {
        "total": len(target.order),
        "open": tuple(open_names),
        "closed": tuple(closed_names),
        "errors": tuple(errors),
        "clean": not open_names and not errors,
    }


def clife_guard_stack(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Open guards and close them in reverse order, capturing closer errors."""
    entries = _require_sequence(guards, "guards")
    opened: list[str] = []
    released: list[str] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        record = _require_mapping(item, f"guard {index}")
        name = _require_name(record.get("name"), f"guard {index} name")
        if name in opened:
            _fail("E003", f"guard {name!r} appears twice")
        opened.append(name)
    for name in reversed(opened):
        index = next(i for i, item in enumerate(entries) if item["name"] == name)
        record = entries[index]
        closer = record.get("closer")
        resource = record.get("resource")
        if closer is None and resource is not None:
            closer = _infer_closer(resource, name)
        if closer is not None:
            _require_callable(closer, f"closer of {name!r}")
            try:
                closer(resource)
            except BaseException as error:
                errors.append(
                    {
                        "name": name,
                        "code": "E004",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
        released.append(name)
    return {
        "open_order": tuple(opened),
        "close_order": tuple(released),
        "depth": len(opened),
        "released": tuple(released),
        "errors": tuple(errors),
        "balanced": len(opened) == len(released),
    }


def clife_finalize_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize lifecycle events into a sorted finalization report."""
    entries = _require_sequence(events, "events")
    prepared: list[tuple[int, str, Mapping[str, Any]]] = []
    for index, item in enumerate(entries):
        record = _require_mapping(item, f"event {index}")
        name = _require_name(record.get("name"), f"event {index} name")
        seq = _require_int(record.get("seq", index), f"seq of {name!r}", 0)
        action = record.get("action")
        if action not in {"open", "close", "error"}:
            _fail("E003", f"invalid lifecycle action {action!r}")
        prepared.append((seq, name, record))
    prepared.sort(key=lambda item: (item[0], item[1]))
    opened = {record["name"] for _, _, record in prepared if record["action"] == "open"}
    closed = {
        record["name"] for _, _, record in prepared if record["action"] == "close"
    }
    errors = tuple(
        sorted(
            record["name"] for _, _, record in prepared if record["action"] == "error"
        )
    )
    leaks = tuple(sorted(opened - closed))
    sequence = tuple(
        {"seq": seq, "name": name, "action": record["action"]}
        for seq, name, record in prepared
    )
    clean = not leaks and not errors
    return {
        "opened": tuple(sorted(opened)),
        "closed": tuple(sorted(closed)),
        "leaks": leaks,
        "errors": errors,
        "clean": clean,
        "sequence": sequence,
    }
