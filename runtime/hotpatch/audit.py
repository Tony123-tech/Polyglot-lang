"""Deterministic, model-only audit trails for hot-patch operations.

A trail is a plain Python dictionary with a name, a list of digest-chained
entries and a list of model sink names.  Entries describe patch transactions,
guard decisions, epoch advances and journal commits; they never contain live
objects, pointers or executable bytes.  Serialization is canonical JSON so the
same events always produce the same digest without depending on wall-clock
time, process ids or dictionary iteration order.

Malformed trail state is rejected with ``E008`` through
:class:`compiler.frontend.errors.PolyglotError`; API misuse uses ``E003`` and a
missing required audit event uses ``E006``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "hpa_audit_guard",
    "hpa_counter_summary",
    "hpa_deserialize",
    "hpa_diff_trails",
    "hpa_entries_by_actor",
    "hpa_entries_by_target",
    "hpa_entry_digest",
    "hpa_export_summary",
    "hpa_filter_entries",
    "hpa_fingerprint",
    "hpa_merge_trails",
    "hpa_record",
    "hpa_record_batch",
    "hpa_redact",
    "hpa_retention_prune",
    "hpa_serialize",
    "hpa_slice_trail",
    "hpa_timeline",
    "hpa_trail",
    "hpa_verify_chain",
]

_MODEL_BOUNDARY = (
    "Model-only audit trail: entries are canonical JSON events; no live object, "
    "raw address, executable text or page flag is recorded or modified."
)
_ENTRY_FIELDS = ("event", "actor", "target", "detail")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured :class:`PolyglotError` for an audit failure."""
    raise PolyglotError(
        code,
        message,
        reason=reason or message,
        note="runtime.hotpatch.audit",
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


def _digest(payload: object) -> str:
    """SHA-256 hex digest of a canonical JSON payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Digestible portion of an audit entry."""
    return {
        "seq": entry.get("seq"),
        "event": entry.get("event"),
        "actor": entry.get("actor"),
        "target": entry.get("target"),
        "detail": entry.get("detail"),
        "prev": entry.get("prev"),
    }


def _trail(value: object, label: str = "trail") -> dict[str, Any]:
    """Validate a trail's structure and digest chain, or raise E008."""
    raw = _mapping(value, label)
    required = ("trail", "entries", "head")
    missing = [key for key in required if key not in raw]
    if missing:
        _fail("E008", f"malformed {label}: missing {missing[0]!r}")
    name = raw["trail"]
    if not isinstance(name, str) or not name:
        _fail("E008", f"malformed {label}: invalid name")
    if not isinstance(raw["entries"], list):
        _fail("E008", f"malformed {label}: entries must be a list")
    previous: str | None = None
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw["entries"]):
        entry = _mapping(item, f"{label}.entries[{index}]")
        fields = (*_ENTRY_FIELDS, "seq", "digest", "prev")
        for field in fields:
            if field not in entry:
                _fail("E008", f"{label}.entries[{index}] is missing {field!r}")
        if entry["seq"] != index + 1:
            _fail("E008", f"{label}.entries[{index}] has a non-sequential seq")
        for field in _ENTRY_FIELDS:
            if not isinstance(entry[field], str):
                _fail("E008", f"{label}.entries[{index}].{field} must be a string")
        if entry["prev"] != previous:
            _fail("E008", f"{label}.entries[{index}] breaks the digest chain")
        if _digest(_entry_payload(entry)) != entry["digest"]:
            _fail("E008", f"{label}.entries[{index}] has a digest mismatch")
        entries.append(entry)
        previous = entry["digest"]
    if raw["head"] != previous:
        _fail("E008", f"malformed {label}: head does not match the last entry")
    sinks = raw.get("sinks", [])
    if not isinstance(sinks, list) or any(not isinstance(item, str) for item in sinks):
        _fail("E008", f"malformed {label}: sinks must be a list of strings")
    return {
        "trail": name,
        "entries": entries,
        "head": previous,
        "sinks": list(sinks),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def _rebuild(
    entries: Sequence[Mapping[str, Any]], name: str, sinks: Sequence[str]
) -> dict[str, Any]:
    """Rebuild a fresh digest chain over copied event fields."""
    rebuilt: list[dict[str, Any]] = []
    previous: str | None = None
    for index, raw in enumerate(entries, start=1):
        entry = {field: raw[field] for field in _ENTRY_FIELDS}
        entry["seq"] = index
        entry["prev"] = previous
        entry["digest"] = _digest(_entry_payload(entry))
        rebuilt.append(entry)
        previous = entry["digest"]
    return {
        "trail": name,
        "entries": rebuilt,
        "head": previous,
        "sinks": list(sinks),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def _event_fields(
    event: object, actor: object, target: object, detail: object
) -> dict[str, str]:
    """Validate caller-supplied audit event fields."""
    if detail is None:
        text = ""
    elif isinstance(detail, str):
        text = detail
    else:
        text = json.dumps(
            _jsonable(detail, "detail"), sort_keys=True, separators=(",", ":")
        )
    return {
        "event": _identifier(event, "event"),
        "actor": _identifier(actor, "actor"),
        "target": _identifier(target, "target"),
        "detail": text,
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def hpa_audit_guard(
    trail: object, *, required: object = ("commit", "rollback")
) -> dict[str, Any]:
    """Check that mandatory event types are present in the audit trail."""
    record = _trail(trail)
    needed = _strings(required, "required")
    present = {entry["event"] for entry in record["entries"]}
    coverage = {
        name: sum(1 for entry in record["entries"] if entry["event"] == name)
        for name in needed
    }
    missing = [name for name in needed if name not in present]
    if missing:
        _fail("E006", f"audit trail is missing required events: {', '.join(missing)}")
    return {
        "coverage": coverage,
        "missing": missing,
        "ok": True,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpa_counter_summary(trail: object) -> dict[str, Any]:
    """Count audit entries by event name and by actor."""
    record = _trail(trail)
    by_event: dict[str, int] = {}
    by_actor: dict[str, int] = {}
    for entry in record["entries"]:
        by_event[entry["event"]] = by_event.get(entry["event"], 0) + 1
        by_actor[entry["actor"]] = by_actor.get(entry["actor"], 0) + 1
    return {
        "events": dict(sorted(by_event.items())),
        "actors": dict(sorted(by_actor.items())),
        "total": len(record["entries"]),
        "model_only": True,
    }


def hpa_deserialize(text: object, *, name: object = None) -> dict[str, Any]:
    """Parse canonical JSON text back into a validated audit trail."""
    if not isinstance(text, str) or not text:
        _fail("E003", "text must be a non-empty JSON string")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError) as error:
        _fail("E008", f"malformed audit JSON: {error}")
    record = _trail(payload, "serialized trail")
    if name is not None:
        record["trail"] = _identifier(name, "trail name")
    return {"trail": record, "count": len(record["entries"]), "model_only": True}


def hpa_diff_trails(left: object, right: object) -> dict[str, Any]:
    """Compare two trails by entry digest without mutating either trail."""
    old = _trail(left, "left trail")
    new = _trail(right, "right trail")
    old_digests = {entry["digest"] for entry in old["entries"]}
    new_digests = {entry["digest"] for entry in new["entries"]}
    added = [entry for entry in new["entries"] if entry["digest"] not in old_digests]
    removed = [entry for entry in old["entries"] if entry["digest"] not in new_digests]
    return {
        "added": added,
        "removed": removed,
        "added_count": len(added),
        "removed_count": len(removed),
        "model_only": True,
    }


def hpa_entry_digest(
    entry: object, *, seq: int = 1, prev: object = None
) -> dict[str, Any]:
    """Compute the canonical digest of one audit event's fields."""
    raw = _mapping(entry, "entry")
    fields = _event_fields(
        raw.get("event"), raw.get("actor"), raw.get("target"), raw.get("detail", "")
    )
    number = _integer(seq, "seq", minimum=1)
    if prev is not None and not isinstance(prev, str):
        _fail("E003", "prev must be a digest string or None")
    fields["seq"] = number
    fields["prev"] = prev
    digest = _digest(_entry_payload(fields))
    return {"digest": digest, "algorithm": "sha256", "seq": number, "model_only": True}


def hpa_entries_by_actor(trail: object) -> dict[str, Any]:
    """Group audit entries by actor with deterministic ordering."""
    record = _trail(trail)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in record["entries"]:
        grouped.setdefault(entry["actor"], []).append(dict(entry))
    ordered = {actor: grouped[actor] for actor in sorted(grouped)}
    return {
        "actors": ordered,
        "counts": {actor: len(entries) for actor, entries in ordered.items()},
        "model_only": True,
    }


def hpa_entries_by_target(trail: object) -> dict[str, Any]:
    """Group audit entries by target with deterministic ordering."""
    record = _trail(trail)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in record["entries"]:
        grouped.setdefault(entry["target"], []).append(dict(entry))
    ordered = {target: grouped[target] for target in sorted(grouped)}
    return {
        "targets": ordered,
        "counts": {target: len(entries) for target, entries in ordered.items()},
        "model_only": True,
    }


def hpa_export_summary(trail: object) -> dict[str, Any]:
    """Export a compact, model-only summary suitable for diagnostics."""
    record = _trail(trail)
    events = sorted({entry["event"] for entry in record["entries"]})
    actors = sorted({entry["actor"] for entry in record["entries"]})
    return {
        "trail": record["trail"],
        "event_count": len(record["entries"]),
        "events": events,
        "actors": actors,
        "sinks": list(record["sinks"]),
        "head": record["head"],
        "fingerprint": hpa_fingerprint(record)["digest"],
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpa_filter_entries(
    trail: object,
    *,
    event: object = None,
    actor: object = None,
    target: object = None,
    prefix: bool = False,
) -> dict[str, Any]:
    """Filter audit entries by exact or prefix event/actor/target values."""
    record = _trail(trail)
    if not isinstance(prefix, bool):
        _fail("E003", "prefix must be a bool")
    filters = {
        "event": event,
        "actor": actor,
        "target": target,
    }
    for label, value in filters.items():
        if value is not None and not isinstance(value, str):
            _fail("E003", f"{label} filter must be a string or None")
    selected: list[dict[str, Any]] = []
    for entry in record["entries"]:
        keep = True
        for label, value in filters.items():
            if value is None:
                continue
            candidate = entry[label]
            keep = keep and (
                candidate.startswith(value) if prefix else candidate == value
            )
        if keep:
            selected.append(dict(entry))
    return {"entries": selected, "count": len(selected), "model_only": True}


def hpa_fingerprint(trail: object) -> dict[str, Any]:
    """Compute a canonical SHA-256 fingerprint of a whole trail."""
    record = _trail(trail)
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "algorithm": "sha256",
        "canonical_length": len(text),
        "model_only": True,
    }


def hpa_merge_trails(left: object, right: object) -> dict[str, Any]:
    """Append one trail's events to another with rebased digests."""
    old = _trail(left, "left trail")
    new = _trail(right, "right trail")
    if old["trail"] != new["trail"]:
        _fail("E003", "cannot merge audit trails with different names")
    offset = len(old["entries"])
    combined = [*old["entries"], *new["entries"]]
    merged = _rebuild(combined, old["trail"], [*old["sinks"], *new["sinks"]])
    return {
        "trail": merged,
        "added": merged["entries"][offset:],
        "count": len(merged["entries"]),
        "model_only": True,
    }


def hpa_record(
    trail: object, *, event: object, actor: object, target: object, detail: object = ""
) -> dict[str, Any]:
    """Append one deterministic audit event and return the chained entry."""
    record = _trail(trail)
    fields = _event_fields(event, actor, target, detail)
    entry = {**fields, "seq": len(record["entries"]) + 1, "prev": record["head"]}
    entry["digest"] = _digest(_entry_payload(entry))
    record["entries"].append(entry)
    record["head"] = entry["digest"]
    return {
        "trail": record,
        "entry": dict(entry),
        "seq": entry["seq"],
        "head": entry["digest"],
        "model_only": True,
    }


def hpa_record_batch(trail: object, events: object) -> dict[str, Any]:
    """Append several audit events in the order supplied by the caller."""
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        _fail("E003", "events must be a sequence")
    record = _trail(trail)
    appended: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        item = _mapping(raw, f"events[{index}]")
        required = ("event", "actor", "target")
        missing = [key for key in required if key not in item]
        if missing:
            _fail("E003", f"events[{index}] is missing {missing[0]!r}")
        result = hpa_record(
            record,
            event=item["event"],
            actor=item["actor"],
            target=item["target"],
            detail=item.get("detail", ""),
        )
        record = result["trail"]
        appended.append(result["entry"])
    return {
        "trail": record,
        "entries": appended,
        "count": len(appended),
        "model_only": True,
    }


def hpa_redact(
    trail: object, *, fields: object = ("detail",), replacement: object = "[redacted]"
) -> dict[str, Any]:
    """Redact selected entry fields and rebuild the digest chain."""
    record = _trail(trail)
    names = _strings(fields, "fields")
    unknown = [name for name in names if name not in _ENTRY_FIELDS and name != "seq"]
    if unknown:
        _fail("E003", f"unknown audit fields: {', '.join(unknown)}")
    if not isinstance(replacement, str):
        _fail("E003", "replacement must be a string")
    redacted = 0
    copies: list[dict[str, Any]] = []
    for entry in record["entries"]:
        copy = dict(entry)
        for name in names:
            if name in copy:
                copy[name] = replacement
                redacted += 1
        copies.append(copy)
    rebuilt = _rebuild(copies, record["trail"], record["sinks"])
    return {
        "trail": rebuilt,
        "redacted": redacted,
        "fields": names,
        "model_only": True,
    }


def hpa_retention_prune(
    trail: object, *, keep: object = None, before: object = None
) -> dict[str, Any]:
    """Prune a trail by trailing count or exclusive seq cutoff."""
    record = _trail(trail)
    if keep is None and before is None:
        _fail("E003", "supply keep or before")
    count = None if keep is None else _integer(keep, "keep", minimum=1)
    cutoff = None if before is None else _integer(before, "before", minimum=1)
    selected = [
        entry
        for entry in record["entries"]
        if (cutoff is None or entry["seq"] < cutoff)
    ]
    if count is not None:
        selected = selected[-count:]
    removed = [dict(entry) for entry in record["entries"] if entry not in selected]
    pruned = _rebuild(selected, record["trail"], record["sinks"])
    return {
        "trail": pruned,
        "removed": removed,
        "kept": len(selected),
        "model_only": True,
    }


def hpa_serialize(trail: object) -> dict[str, Any]:
    """Serialize a trail as canonical JSON text without any file IO."""
    record = _trail(trail)
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "text": text,
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "length": len(text),
        "model_only": True,
    }


def hpa_slice_trail(
    trail: object, *, start: int = 1, stop: int | None = None
) -> dict[str, Any]:
    """Select a half-open seq range and rebuild a fresh digest chain."""
    record = _trail(trail)
    first = _integer(start, "start", minimum=1)
    if stop is not None:
        last = _integer(stop, "stop", minimum=1)
        if last < first:
            _fail("E003", "stop must not be smaller than start")
    selected = [
        dict(entry)
        for entry in record["entries"]
        if entry["seq"] >= first and (stop is None or entry["seq"] < last)
    ]
    rebuilt = _rebuild(selected, record["trail"], record["sinks"])
    return {
        "trail": rebuilt,
        "entries": rebuilt["entries"],
        "count": len(selected),
        "model_only": True,
    }


def hpa_timeline(trail: object) -> dict[str, Any]:
    """List audit events in seq order as a compact timeline."""
    record = _trail(trail)
    timeline = [
        {
            "seq": entry["seq"],
            "event": entry["event"],
            "actor": entry["actor"],
            "target": entry["target"],
        }
        for entry in sorted(record["entries"], key=lambda item: item["seq"])
    ]
    return {"events": timeline, "count": len(timeline), "model_only": True}


def hpa_trail(name: object, *, sinks: object = ()) -> dict[str, Any]:
    """Create an empty audit trail with a name and model sink list."""
    label = _identifier(name, "trail name")
    return {
        "trail": label,
        "entries": [],
        "head": None,
        "sinks": _strings(sinks, "sinks"),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpa_verify_chain(trail: object, *, strict: bool = True) -> dict[str, Any]:
    """Verify structural continuity and digest chaining of a trail."""
    record = _trail(trail)
    issues: list[str] = []
    seen: set[int] = set()
    for index, entry in enumerate(record["entries"]):
        if entry["seq"] in seen:
            issues.append(f"duplicate seq {entry['seq']} at index {index}")
        seen.add(entry["seq"])
        if entry["seq"] != index + 1:
            issues.append(f"seq gap at index {index}")
    if strict and issues:
        _fail("E008", "; ".join(issues))
    return {
        "ok": not issues,
        "count": len(record["entries"]),
        "issues": issues,
        "head": record["head"],
        "model_only": True,
    }
