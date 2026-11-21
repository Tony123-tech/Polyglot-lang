"""Deterministic, model-only patch transaction helpers.

A transaction in this module is an ordinary Python dictionary that records an
in-process model state, a list of staged changes, dependency ids, preconditions
and an append-only undo journal.  Nothing here rewrites foreign executable
text, dereferences a raw address, changes a page protection flag, or installs a
runtime callback.  Raw addresses that appear in model state are opaque string
or integer keys and are never interpreted as pointers.

Diagnostics follow the frontend ``E001``-``E008`` surface via
:class:`compiler.frontend.errors.PolyglotError`:

* ``E001`` – ABI/type mismatch between a staged value and its target;
* ``E002`` – a strict precondition or merge obligation was not satisfied;
* ``E003`` – API misuse (wrong type, unknown kind, closed transaction);
* ``E005`` – self-dependency or dependency cycle;
* ``E008`` – malformed transaction or journal state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "hptx_add_dependency",
    "hptx_begin",
    "hptx_check_preconditions",
    "hptx_commit",
    "hptx_detect_cycle",
    "hptx_diff_states",
    "hptx_fingerprint",
    "hptx_journal_append",
    "hptx_journal_slice",
    "hptx_list_changes",
    "hptx_merge_changes",
    "hptx_order_by_dependency",
    "hptx_prepare",
    "hptx_preview_state",
    "hptx_replay_journal",
    "hptx_reserve_id",
    "hptx_rollback",
    "hptx_seal",
    "hptx_stage_change",
    "hptx_stage_many",
    "hptx_summarize",
    "hptx_unseal",
    "hptx_unstage_change",
    "hptx_validate_change_kind",
    "hptx_verify_journal",
]

_MODEL_BOUNDARY = (
    "Model-only patch transaction: no executable text is rewritten, no address "
    "is dereferenced, and no page protection is changed."
)
_STATUSES = frozenset({"draft", "sealed", "prepared", "committed", "rolled_back"})
_KINDS = frozenset({"set", "insert", "delete"})
_JOURNAL_ACTIONS = frozenset({"commit", "rollback"})
_PRECONDITION_OPS = frozenset({"eq", "neq", "exists", "absent"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured :class:`PolyglotError` for a hot-patch transaction."""
    raise PolyglotError(
        code,
        message,
        reason=reason or message,
        note="runtime.hotpatch.tx",
    )


def _jsonable(value: object, label: str) -> Any:
    """Return a JSON-compatible copy of *value* or reject it with E003."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("E003", f"{label} must be a finite number")
        return value
    if isinstance(value, list):
        return [
            _jsonable(item, f"{label}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, f"{label}[{key!r}]")
            for key, item in value.items()
        }
    _fail("E003", f"{label} must contain only JSON-compatible values")


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Validate and copy a string-keyed mapping."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E003", f"{label} keys must be strings")
        result[key] = _jsonable(item, f"{label}[{key!r}]")
    return result


def _identifier(value: object, label: str) -> str:
    """Return a non-empty, whitespace-free identifier."""
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        _fail("E003", f"{label} must be a non-empty identifier without whitespace")
    return value


def _strings(value: object, label: str) -> list[str]:
    """Validate a sequence of identifiers and return a deduplicated list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("E003", f"{label} must be a sequence of identifiers")
    result: list[str] = []
    for item in value:
        name = _identifier(item, label)
        if name not in result:
            result.append(name)
    return result


def _canonical(payload: object) -> str:
    """Canonical JSON text used for all deterministic digests."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: object) -> str:
    """SHA-256 hex digest of a canonical JSON payload."""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _require_tx(transaction: object, label: str = "transaction") -> dict[str, Any]:
    """Validate and deep-copy a transaction record, or raise E008."""
    record = _mapping(transaction, label)
    required = ("id", "status", "state", "changes", "dependencies", "journal")
    missing = [key for key in required if key not in record]
    if missing:
        _fail("E008", f"malformed transaction: missing {missing[0]!r}")
    if not isinstance(record["id"], str) or not record["id"]:
        _fail("E008", "malformed transaction: invalid id")
    if record["status"] not in _STATUSES:
        _fail("E008", f"malformed transaction: unknown status {record['status']!r}")
    if not isinstance(record["changes"], list) or not isinstance(
        record["dependencies"], list
    ):
        _fail("E008", "malformed transaction: changes and dependencies must be lists")
    for change in record["changes"]:
        _require_change(change)
    for dependency in record["dependencies"]:
        if not isinstance(dependency, str) or not dependency:
            _fail("E008", "malformed transaction: dependency ids must be strings")
    record.setdefault("preconditions", [])
    record.setdefault("metadata", {})
    if not isinstance(record["preconditions"], list):
        _fail("E008", "malformed transaction: preconditions must be a list")
    _journal_list(record["journal"])
    record.setdefault("model_only", True)
    record.setdefault("boundary", _MODEL_BOUNDARY)
    return record


def _require_change(change: object, index: int = 0) -> dict[str, Any]:
    """Validate one staged change, or raise E008."""
    record = _mapping(change, f"change[{index}]")
    required = ("key", "kind", "old", "new", "seq", "had_old")
    missing = [key for key in required if key not in record]
    if missing:
        _fail("E008", f"malformed change at index {index}: missing {missing[0]!r}")
    if not isinstance(record["key"], str) or not record["key"]:
        _fail("E008", f"malformed change at index {index}: invalid key")
    if record["kind"] not in _KINDS:
        _fail("E008", f"malformed change at index {index}: unknown kind")
    if not isinstance(record["seq"], int) or isinstance(record["seq"], bool):
        _fail("E008", f"malformed change at index {index}: seq must be an integer")
    if not isinstance(record["had_old"], bool):
        _fail("E008", f"malformed change at index {index}: had_old must be a bool")
    return record


def _change_digest(change: Mapping[str, Any]) -> str:
    """Digest of the immutable identity fields of a staged change."""
    return _digest(
        {
            "key": change["key"],
            "kind": change["kind"],
            "old": change.get("old"),
            "new": change.get("new"),
            "seq": change.get("seq"),
            "had_old": change.get("had_old"),
        }
    )


def _stage(
    transaction: object, key: object, value: object, kind: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one change to a draft transaction."""
    record = _require_tx(transaction)
    if record["status"] != "draft":
        _fail("E003", f"cannot stage a change while status is {record['status']!r}")
    name = _identifier(key, "change key")
    if kind not in _KINDS:
        _fail("E003", f"unknown change kind {kind!r}")
    if any(change["key"] == name for change in record["changes"]):
        _fail("E003", f"a change for {name!r} is already staged")
    new_value = _jsonable(value, "change value")
    had_old = name in record["state"]
    change = {
        "key": name,
        "kind": kind,
        "old": record["state"].get(name),
        "new": new_value,
        "seq": len(record["changes"]) + 1,
        "had_old": had_old,
    }
    change["digest"] = _change_digest(change)
    record["changes"].append(change)
    return record, dict(change)


def _apply_changes(
    state: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply staged changes in sequence order to a state copy."""
    result = dict(state)
    for change in sorted(changes, key=lambda item: item["seq"]):
        if change["kind"] == "delete":
            result.pop(change["key"], None)
        else:
            result[change["key"]] = change["new"]
    return result


def _reverse_changes(
    state: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Undo staged changes in reverse sequence order."""
    result = dict(state)
    for change in sorted(changes, key=lambda item: item["seq"], reverse=True):
        key = change["key"]
        if change["kind"] == "insert":
            result.pop(key, None)
        elif change["had_old"]:
            result[key] = change["old"]
        else:
            result.pop(key, None)
    return result


def _journal_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Digestible portion of a journal entry."""
    return {
        "seq": entry.get("seq"),
        "action": entry.get("action"),
        "tx_id": entry.get("tx_id"),
        "before": entry.get("before"),
        "after": entry.get("after"),
        "prev": entry.get("prev"),
    }


def _journal_list(journal: object, label: str = "journal") -> list[dict[str, Any]]:
    """Validate a journal's digest chain and return copied entries."""
    if journal is None:
        return []
    if not isinstance(journal, Sequence) or isinstance(journal, (str, bytes)):
        _fail("E008", f"{label} must be a sequence of journal entries")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for index, raw in enumerate(journal):
        entry = _mapping(raw, f"{label}[{index}]")
        required = ("seq", "action", "tx_id", "before", "after", "prev", "digest")
        missing = [key for key in required if key not in entry]
        if missing:
            _fail("E008", f"{label} entry {index} is missing {missing[0]!r}")
        if entry["action"] not in _JOURNAL_ACTIONS:
            _fail("E008", f"{label} entry {index} has an unknown action")
        if entry["seq"] != index + 1:
            _fail("E008", f"{label} entry {index} has a non-sequential seq")
        if entry["prev"] != previous:
            _fail("E008", f"{label} entry {index} breaks the digest chain")
        if _digest(_journal_payload(entry)) != entry["digest"]:
            _fail("E008", f"{label} entry {index} has a digest mismatch")
        entries.append(entry)
        previous = entry["digest"]
    return entries


def _append_journal(
    journal: object,
    action: str,
    tx_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append a chained entry and return the new journal with that entry."""
    if action not in _JOURNAL_ACTIONS:
        _fail("E003", f"unknown journal action {action!r}")
    entries = _journal_list(journal)
    payload = {
        "seq": len(entries) + 1,
        "action": action,
        "tx_id": tx_id,
        "before": _mapping(before, "before state"),
        "after": _mapping(after, "after state"),
        "prev": entries[-1]["digest"] if entries else None,
    }
    entry = dict(payload)
    entry["digest"] = _digest(payload)
    return entries + [entry], entry


def _cycle_of(graph: Mapping[str, Sequence[str]]) -> list[str]:
    """Return one deterministic cycle in *graph*, or an empty list."""
    nodes = {key: list(graph[key]) for key in graph}
    markers: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        markers[node] = 1
        stack.append(node)
        for neighbour in sorted(nodes.get(node, [])):
            if neighbour not in nodes:
                continue
            if markers.get(neighbour) == 1:
                start = stack.index(neighbour)
                return stack[start:] + [neighbour]
            if markers.get(neighbour, 0) == 0:
                found = visit(neighbour)
                if found:
                    return found
        stack.pop()
        markers[node] = 2
        return []

    for node in sorted(nodes):
        if markers.get(node, 0) == 0:
            found = visit(node)
            if found:
                return found
    return []


def _graph_copy(graph: object, label: str) -> dict[str, list[str]]:
    """Validate a dependency graph mapping."""
    raw = _mapping(graph, label)
    return {key: _strings(value, f"{label}[{key!r}]") for key, value in raw.items()}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def hptx_add_dependency(
    transaction: object, dependency_id: object, *, graph: object = None
) -> dict[str, Any]:
    """Add a dependency to a transaction, rejecting cycles with E005."""
    record = _require_tx(transaction)
    dependency = _identifier(dependency_id, "dependency id")
    if dependency == record["id"]:
        _fail("E005", "a transaction cannot depend on itself")
    if dependency in record["dependencies"]:
        return {
            "transaction": record,
            "dependencies": list(record["dependencies"]),
            "cycle_free": True,
            "model_only": True,
        }
    probe = _graph_copy(graph or {}, "graph")
    probe[record["id"]] = list(record["dependencies"]) + [dependency]
    cycle = _cycle_of(probe)
    if cycle:
        _fail("E005", f"dependency cycle: {' -> '.join(cycle)}")
    record["dependencies"].append(dependency)
    return {
        "transaction": record,
        "dependencies": list(record["dependencies"]),
        "cycle_free": True,
        "model_only": True,
    }


def hptx_begin(
    tx_id: object,
    *,
    state: object = None,
    metadata: object = None,
    dependencies: object = (),
) -> dict[str, Any]:
    """Create a fresh draft transaction over a copied model state."""
    name = _identifier(tx_id, "transaction id")
    return {
        "id": name,
        "status": "draft",
        "state": _mapping(state or {}, "state"),
        "changes": [],
        "dependencies": _strings(dependencies, "dependencies"),
        "preconditions": [],
        "metadata": _mapping(metadata or {}, "metadata"),
        "journal": [],
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hptx_check_preconditions(
    state: object, conditions: object, *, strict: bool = False
) -> dict[str, Any]:
    """Evaluate ``eq``/``neq``/``exists``/``absent`` preconditions."""
    model = _mapping(state, "state")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        _fail("E003", "conditions must be a sequence")
    results: list[dict[str, Any]] = []
    for index, raw in enumerate(conditions):
        condition = _mapping(raw, f"conditions[{index}]")
        key = _identifier(condition.get("key"), f"conditions[{index}].key")
        op = condition.get("op")
        if op not in _PRECONDITION_OPS:
            _fail("E003", f"unknown precondition op {op!r}")
        present = key in model
        actual = model.get(key)
        expected = condition.get("value")
        ok = {
            "eq": present and actual == expected,
            "neq": present and actual != expected,
            "exists": present,
            "absent": not present,
        }[op]
        results.append(
            {
                "key": key,
                "op": op,
                "expected": expected,
                "actual": actual,
                "present": present,
                "ok": bool(ok),
            }
        )
    failed = sum(1 for item in results if not item["ok"])
    if strict and failed:
        _fail("E002", f"{failed} precondition(s) were not satisfied")
    return {
        "results": results,
        "passed": len(results) - failed,
        "failed": failed,
        "all_ok": failed == 0,
        "model_only": True,
    }


def hptx_commit(transaction: object) -> dict[str, Any]:
    """Apply prepared changes to the model state and append an undo entry."""
    record = _require_tx(transaction)
    if record["status"] != "prepared":
        _fail("E003", f"cannot commit a transaction in status {record['status']!r}")
    if not record["changes"]:
        _fail("E003", "cannot commit a transaction without staged changes")
    before = dict(record["state"])
    after = _apply_changes(before, record["changes"])
    journal, entry = _append_journal(
        record["journal"], "commit", record["id"], before, after
    )
    record["state"] = after
    record["journal"] = journal
    record["status"] = "committed"
    return {
        "transaction": record,
        "state": dict(after),
        "journal_seq": entry["seq"],
        "digest": entry["digest"],
        "model_only": True,
    }


def hptx_detect_cycle(graph: object) -> dict[str, Any]:
    """Detect one dependency cycle without mutating the supplied graph."""
    nodes = _graph_copy(graph, "graph")
    cycle = _cycle_of(nodes)
    return {
        "cycle": cycle,
        "cyclic_nodes": sorted(set(cycle)),
        "has_cycle": bool(cycle),
        "nodes": sorted(nodes),
        "model_only": True,
    }


def hptx_diff_states(before: object, after: object) -> dict[str, Any]:
    """Compare two model states by key, reporting adds, removes and changes."""
    old = _mapping(before, "before")
    new = _mapping(after, "after")
    added: list[str] = []
    removed: list[str] = []
    changed: list[dict[str, Any]] = []
    for key in sorted(set(old) | set(new)):
        if key not in old:
            added.append(key)
        elif key not in new:
            removed.append(key)
        elif old[key] != new[key]:
            changed.append(
                {
                    "key": key,
                    "old": old[key],
                    "new": new[key],
                    "digest": _digest(new[key]),
                }
            )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "count": len(added) + len(removed) + len(changed),
        "model_only": True,
    }


def hptx_fingerprint(transaction: object) -> dict[str, Any]:
    """Compute a canonical SHA-256 fingerprint of a transaction record."""
    record = _require_tx(transaction)
    text = _canonical(record)
    return {
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "algorithm": "sha256",
        "canonical_length": len(text),
        "model_only": True,
    }


def hptx_journal_append(
    journal: object,
    *,
    action: str,
    tx_id: object,
    before: object,
    after: object,
) -> dict[str, Any]:
    """Append a chained commit/rollback entry to a model journal."""
    name = _identifier(tx_id, "transaction id")
    entries, entry = _append_journal(journal, action, name, before, after)
    return {
        "journal": entries,
        "entry": entry,
        "seq": entry["seq"],
        "head": entry["digest"],
        "model_only": True,
    }


def hptx_journal_slice(
    journal: object, *, start: int = 1, stop: int | None = None
) -> dict[str, Any]:
    """Return the half-open ``[start, stop)`` range of journal seq numbers."""
    entries = _journal_list(journal)
    if not isinstance(start, int) or isinstance(start, bool) or start < 1:
        _fail("E003", "start must be a positive integer")
    if stop is not None and (
        not isinstance(stop, int) or isinstance(stop, bool) or stop < start
    ):
        _fail("E003", "stop must be an integer not smaller than start")
    selected = [
        entry
        for entry in entries
        if entry["seq"] >= start and (stop is None or entry["seq"] < stop)
    ]
    return {
        "entries": selected,
        "count": len(selected),
        "first_seq": selected[0]["seq"] if selected else None,
        "last_seq": selected[-1]["seq"] if selected else None,
        "model_only": True,
    }


def hptx_list_changes(transaction: object) -> dict[str, Any]:
    """List staged changes in deterministic sequence order."""
    record = _require_tx(transaction)
    changes = sorted(record["changes"], key=lambda item: item["seq"])
    return {
        "changes": changes,
        "count": len(changes),
        "keys": [change["key"] for change in changes],
        "model_only": True,
    }


def hptx_merge_changes(
    left: object, right: object, *, strict: bool = False
) -> dict[str, Any]:
    """Merge two staged-change lists, reporting conflicting keys."""
    merged: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for label, group in (("left", left), ("right", right)):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            _fail("E003", f"{label} changes must be a sequence")
        for index, raw in enumerate(group):
            change = _require_change(raw, index)
            key = change["key"]
            if key not in merged:
                merged[key] = dict(change)
                continue
            incumbent = merged[key]
            if _change_digest(incumbent) != _change_digest(change):
                conflicts.append(
                    {
                        "key": key,
                        "left": incumbent.get("new"),
                        "right": change.get("new"),
                        "left_kind": incumbent["kind"],
                        "right_kind": change["kind"],
                    }
                )
    if strict and conflicts:
        _fail("E003", f"merge has {len(conflicts)} conflicting key(s)")
    ordered = sorted(merged.values(), key=lambda item: (item["key"], item["seq"]))
    return {
        "changes": ordered,
        "conflicts": conflicts,
        "count": len(ordered),
        "model_only": True,
    }


def hptx_order_by_dependency(ids: object, graph: object) -> dict[str, Any]:
    """Topologically order transaction ids from fewest to most dependencies."""
    names = _strings(ids, "ids")
    edges = _graph_copy(graph, "graph")
    known = set(names)
    prerequisites = {
        name: sorted({dep for dep in edges.get(name, []) if dep in known})
        for name in names
    }
    levels: list[list[str]] = []
    done: set[str] = set()
    while len(done) < len(names):
        ready = sorted(
            name
            for name in names
            if name not in done and all(dep in done for dep in prerequisites[name])
        )
        if not ready:
            _fail("E005", "dependency graph contains a cycle")
        levels.append(ready)
        done.update(ready)
    return {
        "order": [name for level in levels for name in level],
        "levels": levels,
        "model_only": True,
    }


def hptx_prepare(transaction: object, *, graph: object = None) -> dict[str, Any]:
    """Seal and prepare a draft transaction after checking its dependencies."""
    record = _require_tx(transaction)
    if record["status"] not in ("draft", "sealed"):
        _fail("E003", f"cannot prepare a transaction in status {record['status']!r}")
    if not record["changes"]:
        _fail("E003", "cannot prepare a transaction without staged changes")
    edges = _graph_copy(graph or {}, "graph")
    edges[record["id"]] = list(record["dependencies"])
    cycle = _cycle_of(edges)
    if cycle:
        _fail("E005", f"dependency cycle: {' -> '.join(cycle)}")
    seal = record.get("seal") or _digest(
        [change["digest"] for change in record["changes"]]
    )
    record["seal"] = seal
    record["status"] = "prepared"
    return {
        "transaction": record,
        "seal": seal,
        "dependencies": list(record["dependencies"]),
        "model_only": True,
    }


def hptx_preview_state(transaction: object) -> dict[str, Any]:
    """Return the model state that committing this transaction would produce."""
    record = _require_tx(transaction)
    applied = [
        {"key": change["key"], "kind": change["kind"], "new": change["new"]}
        for change in sorted(record["changes"], key=lambda item: item["seq"])
    ]
    return {
        "state": _apply_changes(record["state"], record["changes"]),
        "applied": applied,
        "count": len(applied),
        "model_only": True,
    }


def hptx_replay_journal(
    journal: object, *, initial_state: object = None, stop_after: int | None = None
) -> dict[str, Any]:
    """Replay commit/rollback snapshots from a validated model journal."""
    entries = _journal_list(journal)
    state = _mapping(initial_state or {}, "initial_state")
    if stop_after is not None and (
        not isinstance(stop_after, int)
        or isinstance(stop_after, bool)
        or stop_after < 1
    ):
        _fail("E003", "stop_after must be a positive integer")
    actions: list[str] = []
    applied = 0
    for entry in entries:
        if stop_after is not None and entry["seq"] > stop_after:
            break
        state = _mapping(entry["after"], "journal after state")
        actions.append(entry["action"])
        applied += 1
    return {
        "state": state,
        "actions": actions,
        "applied": applied,
        "head": entries[-1]["digest"] if entries else None,
        "model_only": True,
    }


def hptx_reserve_id(
    namespace: object, *, used: object = (), width: int = 4
) -> dict[str, Any]:
    """Reserve the lowest unused deterministic id in a namespace."""
    name = _identifier(namespace, "namespace")
    taken = set(_strings(used, "used"))
    if not isinstance(width, int) or isinstance(width, bool) or not 1 <= width <= 8:
        _fail("E003", "width must be an integer between 1 and 8")
    index = 1
    while f"{name}-{index:0{width}d}" in taken:
        index += 1
    return {
        "id": f"{name}-{index:0{width}d}",
        "namespace": name,
        "index": index,
        "used_count": len(taken),
        "model_only": True,
    }


def hptx_rollback(transaction: object) -> dict[str, Any]:
    """Undo prepared or committed changes and append a rollback entry."""
    record = _require_tx(transaction)
    if record["status"] not in ("prepared", "committed"):
        _fail("E003", f"cannot roll back a transaction in status {record['status']!r}")
    before = dict(record["state"])
    after = _reverse_changes(before, record["changes"])
    journal, entry = _append_journal(
        record["journal"], "rollback", record["id"], before, after
    )
    record["state"] = after
    record["journal"] = journal
    record["status"] = "rolled_back"
    return {
        "transaction": record,
        "state": dict(after),
        "restored_keys": sorted(set(before) | set(after)),
        "journal_seq": entry["seq"],
        "model_only": True,
    }


def hptx_seal(transaction: object) -> dict[str, Any]:
    """Freeze the staged change list of a draft transaction."""
    record = _require_tx(transaction)
    if record["status"] != "draft" or not record["changes"]:
        _fail("E003", "only a draft with staged changes can be sealed")
    seal = _digest([change["digest"] for change in record["changes"]])
    record["seal"] = seal
    record["status"] = "sealed"
    return {
        "transaction": record,
        "seal": seal,
        "change_count": len(record["changes"]),
        "model_only": True,
    }


def hptx_stage_change(
    transaction: object, key: object, value: object, *, kind: str = "set"
) -> dict[str, Any]:
    """Stage one model-state change in a draft transaction."""
    record, change = _stage(transaction, key, value, kind)
    return {
        "transaction": record,
        "change": change,
        "count": len(record["changes"]),
        "model_only": True,
    }


def hptx_stage_many(transaction: object, changes: object) -> dict[str, Any]:
    """Stage several ``key``/``value`` changes atomically in one call."""
    if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)):
        _fail("E003", "changes must be a sequence")
    record = _require_tx(transaction)
    staged: list[dict[str, Any]] = []
    for index, raw in enumerate(changes):
        change = _mapping(raw, f"changes[{index}]")
        if "key" not in change or "value" not in change:
            _fail("E003", f"changes[{index}] needs 'key' and 'value'")
        record, added = _stage(
            record, change["key"], change["value"], change.get("kind", "set")
        )
        staged.append(added)
    return {
        "transaction": record,
        "staged": staged,
        "count": len(staged),
        "model_only": True,
    }


def hptx_summarize(transaction: object) -> dict[str, Any]:
    """Summarize a transaction without exposing its full model state."""
    record = _require_tx(transaction)
    return {
        "id": record["id"],
        "status": record["status"],
        "change_count": len(record["changes"]),
        "keys": sorted({change["key"] for change in record["changes"]}),
        "dependency_count": len(record["dependencies"]),
        "journal_count": len(record["journal"]),
        "seal": record.get("seal"),
        "digest": hptx_fingerprint(record)["digest"],
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hptx_unseal(transaction: object) -> dict[str, Any]:
    """Reopen a sealed transaction for further staging."""
    record = _require_tx(transaction)
    if record["status"] != "sealed":
        _fail("E003", f"cannot unseal a transaction in status {record['status']!r}")
    previous = record.pop("seal", None)
    record["status"] = "draft"
    return {
        "transaction": record,
        "previous_seal": previous,
        "status": record["status"],
        "model_only": True,
    }


def hptx_unstage_change(transaction: object, key: object) -> dict[str, Any]:
    """Remove one staged change and renumber the remaining sequence."""
    record = _require_tx(transaction)
    name = _identifier(key, "change key")
    if record["status"] != "draft":
        _fail("E003", "only a draft transaction can be unstaged")
    matches = [change for change in record["changes"] if change["key"] == name]
    if not matches:
        _fail("E003", f"no staged change for {name!r}")
    removed = dict(matches[0])
    remaining = [change for change in record["changes"] if change["key"] != name]
    for index, change in enumerate(remaining, start=1):
        change["seq"] = index
        change["digest"] = _change_digest(change)
    record["changes"] = remaining
    return {
        "transaction": record,
        "removed": removed,
        "count": len(remaining),
        "model_only": True,
    }


def hptx_validate_change_kind(
    change: object, *, current: object = None, allow_type_change: bool = False
) -> dict[str, Any]:
    """Validate a staged change's kind against a model-state snapshot."""
    record = _require_change(change)
    state = _mapping(current or {}, "current")
    key = record["key"]
    kind = record["kind"]
    checks: list[str] = [
        f"change kind {kind!r} is supported",
        f"key {key!r} is an identifier",
    ]
    present = key in state
    if kind == "insert" and present:
        _fail("E001", f"cannot insert an existing key {key!r}")
    if kind == "delete" and not present:
        _fail("E001", f"cannot delete a missing key {key!r}")
    if kind == "set" and present and not allow_type_change:
        old_value = state[key]
        new_value = record["new"]
        if type(old_value) is not type(new_value):
            _fail(
                "E001",
                f"ABI/type mismatch for {key!r}: "
                f"{type(old_value).__name__} -> {type(new_value).__name__}",
            )
        checks.append("replacement value preserves the model type")
    return {
        "key": key,
        "kind": kind,
        "present": present,
        "checks": checks,
        "valid": True,
        "model_only": True,
    }


def hptx_verify_journal(journal: object, *, strict: bool = True) -> dict[str, Any]:
    """Verify journal structure, digest chain and sequence numbering."""
    entries = _journal_list(journal)
    issues: list[str] = []
    seen: set[int] = set()
    for index, entry in enumerate(entries):
        if entry["seq"] in seen:
            issues.append(f"duplicate seq {entry['seq']} at index {index}")
        seen.add(entry["seq"])
        if entry["seq"] != index + 1:
            issues.append(f"seq gap at index {index}")
    if strict and issues:
        _fail("E008", "; ".join(issues))
    return {
        "ok": not issues,
        "count": len(entries),
        "issues": issues,
        "head": entries[-1]["digest"] if entries else None,
        "model_only": True,
    }
