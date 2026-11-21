"""Cooperative, model-only RCU epoch helpers for hot-patch rollout.

An epoch domain is an ordinary Python dictionary holding reader registrations,
active read-section snapshots, retired AST-root ids, a history of advances and
a callback log.  The domain models the shape of user-space epoch reclamation:
readers cooperate by entering and leaving sections, and reclamation happens
only when every active reader has observed a later epoch.  There is no kernel
RCU syscall, no thread-local access, no page reclamation and no address
dereference; retired ids are opaque model keys.

Malformed domain state is reported with ``E008`` through
:class:`compiler.frontend.errors.PolyglotError`; invalid API input is ``E003``
and semantic reader conflicts are ``E002``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "hpe_advance",
    "hpe_callback_log",
    "hpe_domain",
    "hpe_domain_merge",
    "hpe_enter_read_section",
    "hpe_epoch_delta",
    "hpe_epoch_fingerprint",
    "hpe_epoch_guard_token",
    "hpe_epoch_report",
    "hpe_exit_read_section",
    "hpe_grace_period",
    "hpe_min_reader_epoch",
    "hpe_reader_snapshot",
    "hpe_reclaim",
    "hpe_reclaim_ready",
    "hpe_register_reader",
    "hpe_retire",
    "hpe_retired_queue",
    "hpe_unregister_reader",
    "hpe_validate_epoch_state",
]

_MODEL_BOUNDARY = (
    "Model-only cooperative epochs: readers are dictionary keys, retired ids "
    "are model state, and no kernel RCU, memory mapping or pointer is used."
)
_REQUIRED_KEYS = (
    "domain",
    "current",
    "registered",
    "readers",
    "retired",
    "history",
    "callbacks",
    "events",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured :class:`PolyglotError` for an epoch-domain failure."""
    raise PolyglotError(
        code,
        message,
        reason=reason or message,
        note="runtime.hotpatch.epochs",
    )


def _identifier(value: object, label: str) -> str:
    """Return a non-empty, whitespace-free identifier."""
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        _fail("E003", f"{label} must be a non-empty identifier without whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    """Return an integer at least *minimum*, or raise E003."""
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("E003", f"{label} must be an integer")
    if value < minimum:
        _fail("E003", f"{label} must be at least {minimum}")
    return value


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


def _digest(payload: object) -> str:
    """SHA-256 hex digest of a canonical JSON payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_list(
    value: object, label: str, fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Validate a list of simple dictionaries carrying *fields*."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("E008", f"{label} must be a list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            _fail("E008", f"{label}[{index}] must be a mapping")
        item: dict[str, Any] = {}
        for field in fields:
            if field not in raw:
                _fail("E008", f"{label}[{index}] is missing {field!r}")
            item[field] = _jsonable(raw[field], f"{label}[{index}].{field}")
        records.append(item)
    return records


def _state(value: object, label: str = "epoch state") -> dict[str, Any]:
    """Validate and deep-copy an epoch domain state, or raise E008."""
    if not isinstance(value, Mapping):
        _fail("E008", f"{label} must be a mapping")
    record: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E008", f"{label} keys must be strings")
        record[key] = _jsonable(item, f"{label}[{key!r}]")
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        _fail("E008", f"malformed epoch state: missing {missing[0]!r}")
    name = _identifier(record["domain"], "domain")
    current = _integer(record["current"], "current")
    if not isinstance(record["registered"], list) or not isinstance(
        record["readers"], Mapping
    ):
        _fail("E008", "malformed epoch state: invalid reader registries")
    for token in record["registered"]:
        _identifier(token, "reader token")
    readers: dict[str, int] = {}
    for token, epoch in record["readers"].items():
        readers[_identifier(token, "reader token")] = _integer(epoch, "reader epoch")
    retired = _record_list(
        record["retired"], "retired", ("id", "epoch", "callback", "seq")
    )
    for entry in retired:
        _identifier(entry["id"], "retired id")
        _identifier(entry["callback"], "callback")
        entry["epoch"] = _integer(entry["epoch"], "retired epoch")
        entry["seq"] = _integer(entry["seq"], "retired seq")
        if entry["epoch"] > current:
            _fail("E008", f"retired node {entry['id']!r} is from a future epoch")
    record["history"] = _record_list(
        record["history"], "history", ("from", "to", "seq")
    )
    record["callbacks"] = _record_list(
        record["callbacks"], "callbacks", ("id", "name", "seq")
    )
    record["events"] = _record_list(
        record["events"], "events", ("action", "token", "epoch", "seq")
    )
    record["domain"] = name
    record["current"] = current
    record["registered"] = list(record["registered"])
    record["readers"] = readers
    record["retired"] = retired
    record.setdefault("model_only", True)
    record.setdefault("boundary", _MODEL_BOUNDARY)
    return record


def _event(state: dict[str, Any], action: str, token: str) -> dict[str, Any]:
    """Append a reader event and return its record."""
    entry = {
        "action": action,
        "token": token,
        "epoch": state["current"],
        "seq": len(state["events"]) + 1,
    }
    state["events"].append(entry)
    return entry


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def hpe_advance(state: object, *, steps: int = 1) -> dict[str, Any]:
    """Advance a domain's epoch and append one history record."""
    record = _state(state)
    count = _integer(steps, "steps", minimum=1)
    previous = record["current"]
    entry = {
        "from": previous,
        "to": previous + count,
        "seq": len(record["history"]) + 1,
    }
    record["history"].append(entry)
    record["current"] = previous + count
    return {
        "state": record,
        "previous": previous,
        "current": record["current"],
        "history_seq": entry["seq"],
        "model_only": True,
    }


def hpe_callback_log(state: object, *, name: object = None) -> dict[str, Any]:
    """List reclamation callbacks in deterministic sequence order."""
    record = _state(state)
    wanted = None if name is None else _identifier(name, "callback name")
    entries = sorted(record["callbacks"], key=lambda item: item["seq"])
    selected = [item for item in entries if wanted is None or item["name"] == wanted]
    return {
        "callbacks": selected,
        "count": len(selected),
        "name_filter": wanted,
        "model_only": True,
    }


def hpe_domain(name: object, *, epoch: int = 0) -> dict[str, Any]:
    """Create an empty cooperative epoch domain with deterministic shape."""
    label = _identifier(name, "domain name")
    value = _integer(epoch, "epoch")
    return {
        "domain": label,
        "current": value,
        "registered": [],
        "readers": {},
        "retired": [],
        "history": [],
        "callbacks": [],
        "events": [],
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpe_domain_merge(
    left: object, right: object, *, strategy: object = "rebase"
) -> dict[str, Any]:
    """Merge two epoch domains derived from the same model domain."""
    old = _state(left, "left state")
    new = _state(right, "right state")
    mode = _identifier(strategy, "strategy")
    if mode not in ("rebase", "strict"):
        _fail("E003", f"unknown merge strategy {mode!r}")
    if old["domain"] != new["domain"]:
        _fail("E003", "cannot merge domains with different names")
    conflicts: list[str] = []
    if old["current"] != new["current"]:
        conflicts.append(f"current epoch {old['current']} vs {new['current']}")
    if mode == "strict" and conflicts:
        _fail("E002", "; ".join(conflicts))
    registered = sorted(set(old["registered"]) | set(new["registered"]))
    readers = dict(old["readers"])
    for token, epoch in new["readers"].items():
        if token in readers and readers[token] != epoch:
            conflicts.append(f"reader {token} epoch differs")
        readers[token] = max(readers[token], epoch) if token in readers else epoch
    retired: dict[str, dict[str, Any]] = {}
    for entry in [*old["retired"], *new["retired"]]:
        existing = retired.get(entry["id"])
        if existing is None:
            retired[entry["id"]] = dict(entry)
        elif existing != entry:
            conflicts.append(f"retired node {entry['id']} differs")
            existing["epoch"] = max(existing["epoch"], entry["epoch"])
    history = [*old["history"], *new["history"]]
    for index, entry in enumerate(history, start=1):
        entry["seq"] = index
    callbacks = [*old["callbacks"], *new["callbacks"]]
    for index, entry in enumerate(callbacks, start=1):
        entry["seq"] = index
    events = [*old["events"], *new["events"]]
    for index, entry in enumerate(events, start=1):
        entry["seq"] = index
    merged = {
        "domain": old["domain"],
        "current": max(old["current"], new["current"]),
        "registered": registered,
        "readers": readers,
        "retired": sorted(retired.values(), key=lambda item: item["id"]),
        "history": history,
        "callbacks": callbacks,
        "events": events,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }
    hpe_validate_epoch_state(merged)
    return {
        "state": merged,
        "conflicts": conflicts,
        "strategy": mode,
        "model_only": True,
    }


def hpe_enter_read_section(state: object, token: object) -> dict[str, Any]:
    """Mark a registered reader active at the domain's current epoch."""
    record = _state(state)
    name = _identifier(token, "reader token")
    if name not in record["registered"]:
        _fail("E003", f"reader {name!r} is not registered")
    if name in record["readers"]:
        _fail("E002", f"reader {name!r} is already in a read section")
    record["readers"][name] = record["current"]
    event = _event(record, "enter", name)
    return {
        "state": record,
        "token": name,
        "epoch": record["current"],
        "event_seq": event["seq"],
        "model_only": True,
    }


def hpe_epoch_delta(before: object, after: object) -> dict[str, Any]:
    """Compute a deterministic delta between two epoch domains."""
    old = _state(before, "before state")
    new = _state(after, "after state")
    if old["domain"] != new["domain"]:
        _fail("E003", "cannot compare different epoch domains")
    old_retired = {entry["id"] for entry in old["retired"]}
    new_retired = {entry["id"] for entry in new["retired"]}
    old_callbacks = {entry["seq"] for entry in old["callbacks"]}
    return {
        "domain": old["domain"],
        "epoch_before": old["current"],
        "epoch_after": new["current"],
        "epoch_advance": new["current"] - old["current"],
        "readers_added": sorted(set(new["readers"]) - set(old["readers"])),
        "readers_removed": sorted(set(old["readers"]) - set(new["readers"])),
        "retired_added": sorted(new_retired - old_retired),
        "retired_removed": sorted(old_retired - new_retired),
        "new_callbacks": sorted(
            entry["id"]
            for entry in new["callbacks"]
            if entry["seq"] not in old_callbacks
        ),
        "history_records": len(new["history"]) - len(old["history"]),
        "model_only": True,
    }


def hpe_epoch_fingerprint(state: object) -> dict[str, Any]:
    """Compute a canonical SHA-256 fingerprint of an epoch domain."""
    record = _state(state)
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "algorithm": "sha256",
        "canonical_length": len(text),
        "model_only": True,
    }


def hpe_epoch_guard_token(state: object) -> dict[str, Any]:
    """Create a digest-bound token for the domain's current epoch state."""
    record = _state(state)
    payload = {
        "domain": record["domain"],
        "epoch": record["current"],
        "readers": record["readers"],
        "head": hpe_epoch_fingerprint(record)["digest"],
    }
    return {
        **payload,
        "digest": _digest(payload),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpe_epoch_report(state: object) -> dict[str, Any]:
    """Summarize a domain's cooperative reclamation position."""
    record = _state(state)
    ready = hpe_reclaim_ready(record)["ids"]
    return {
        "domain": record["domain"],
        "epoch": record["current"],
        "registered": len(record["registered"]),
        "active_readers": len(record["readers"]),
        "retired": len(record["retired"]),
        "reclaimable": len(ready),
        "callbacks": len(record["callbacks"]),
        "history": len(record["history"]),
        "digest": hpe_epoch_fingerprint(record)["digest"],
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpe_exit_read_section(state: object, token: object) -> dict[str, Any]:
    """Mark an active reader quiescent and report the epoch it observed."""
    record = _state(state)
    name = _identifier(token, "reader token")
    if name not in record["readers"]:
        _fail("E003", f"reader {name!r} is not in a read section")
    observed = record["readers"].pop(name)
    event = _event(record, "exit", name)
    return {
        "state": record,
        "token": name,
        "observed_epoch": observed,
        "event_seq": event["seq"],
        "model_only": True,
    }


def hpe_grace_period(state: object) -> dict[str, Any]:
    """Determine which retired ids have passed a cooperative grace period."""
    record = _state(state)
    active = dict(record["readers"])
    nodes: list[dict[str, Any]] = []
    ready: list[str] = []
    for entry in sorted(record["retired"], key=lambda item: (item["seq"], item["id"])):
        blocking = sorted(
            token for token, epoch in active.items() if epoch <= entry["epoch"]
        )
        elapsed = not blocking
        nodes.append(
            {
                "id": entry["id"],
                "retired_epoch": entry["epoch"],
                "elapsed": elapsed,
                "blocking": blocking,
            }
        )
        if elapsed:
            ready.append(entry["id"])
    return {
        "nodes": nodes,
        "ready": ready,
        "active_readers": len(active),
        "model_only": True,
    }


def hpe_min_reader_epoch(state: object) -> dict[str, Any]:
    """Return the oldest epoch observed by any active reader."""
    record = _state(state)
    epochs = sorted(record["readers"].values())
    return {
        "epoch": epochs[0] if epochs else None,
        "active_readers": len(epochs),
        "current": record["current"],
        "model_only": True,
    }


def hpe_reader_snapshot(state: object, *, active_only: bool = False) -> dict[str, Any]:
    """List reader registrations with their active epoch state."""
    record = _state(state)
    if not isinstance(active_only, bool):
        _fail("E003", "active_only must be a bool")
    readers: list[dict[str, Any]] = []
    for token in sorted(record["registered"]):
        active = token in record["readers"]
        if active_only and not active:
            continue
        readers.append(
            {"token": token, "active": active, "epoch": record["readers"].get(token)}
        )
    return {"readers": readers, "count": len(readers), "model_only": True}


def hpe_reclaim(state: object) -> dict[str, Any]:
    """Reclaim retired ids whose grace period has elapsed in the model."""
    record = _state(state)
    ready = set(hpe_reclaim_ready(record)["ids"])
    reclaimed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for entry in record["retired"]:
        if entry["id"] not in ready:
            pending.append(dict(entry))
            continue
        callback = {
            "id": entry["id"],
            "name": entry["callback"],
            "seq": len(record["callbacks"]) + 1 + len(reclaimed),
        }
        record["callbacks"].append(callback)
        reclaimed.append(
            {"id": entry["id"], "epoch": entry["epoch"], "callback": callback}
        )
    record["retired"] = pending
    return {
        "state": record,
        "reclaimed": reclaimed,
        "callbacks": list(record["callbacks"][-len(reclaimed) :]) if reclaimed else [],
        "pending": pending,
        "model_only": True,
    }


def hpe_reclaim_ready(state: object) -> dict[str, Any]:
    """List retired ids that are eligible for cooperative reclamation."""
    return {
        "ids": hpe_grace_period(state)["ready"],
        "model_only": True,
    }


def hpe_register_reader(state: object, token: object) -> dict[str, Any]:
    """Register a reader token without starting a read section."""
    record = _state(state)
    name = _identifier(token, "reader token")
    if name in record["registered"]:
        _fail("E003", f"reader {name!r} is already registered")
    record["registered"].append(name)
    event = _event(record, "register", name)
    return {
        "state": record,
        "token": name,
        "registered": sorted(record["registered"]),
        "event_seq": event["seq"],
        "model_only": True,
    }


def hpe_retire(
    state: object, node_id: object, *, callback: object = "reclaim"
) -> dict[str, Any]:
    """Retire an opaque model node id at the current epoch."""
    record = _state(state)
    name = _identifier(node_id, "retired node id")
    action = _identifier(callback, "callback")
    if any(entry["id"] == name for entry in record["retired"]):
        _fail("E003", f"model node {name!r} is already retired")
    entry = {
        "id": name,
        "epoch": record["current"],
        "callback": action,
        "seq": len(record["retired"]) + 1,
    }
    record["retired"].append(entry)
    return {
        "state": record,
        "record": entry,
        "model_only": True,
    }


def hpe_retired_queue(state: object, *, min_epoch: object = None) -> dict[str, Any]:
    """List retired model nodes ordered by retire epoch and id."""
    record = _state(state)
    floor = None if min_epoch is None else _integer(min_epoch, "min_epoch")
    entries = sorted(record["retired"], key=lambda item: (item["epoch"], item["id"]))
    selected = [
        dict(item) for item in entries if floor is None or item["epoch"] >= floor
    ]
    return {"retired": selected, "count": len(selected), "model_only": True}


def hpe_unregister_reader(
    state: object, token: object, *, force: bool = False
) -> dict[str, Any]:
    """Unregister a reader, refusing to drop an active section unless forced."""
    record = _state(state)
    name = _identifier(token, "reader token")
    if name not in record["registered"]:
        _fail("E003", f"reader {name!r} is not registered")
    if not isinstance(force, bool):
        _fail("E003", "force must be a bool")
    active = name in record["readers"]
    if active and not force:
        _fail("E002", f"reader {name!r} is still inside a read section")
    record["registered"].remove(name)
    record["readers"].pop(name, None)
    event = _event(record, "unregister", name)
    return {
        "state": record,
        "token": name,
        "was_active": active,
        "event_seq": event["seq"],
        "model_only": True,
    }


def hpe_validate_epoch_state(state: object) -> dict[str, Any]:
    """Validate a domain state's structure and report its key counters."""
    record = _state(state)
    return {
        "ok": True,
        "domain": record["domain"],
        "epoch": record["current"],
        "readers": len(record["readers"]),
        "retired": len(record["retired"]),
        "history": len(record["history"]),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }
