"""Deterministic report construction, parsing, merging, and rendering.

Reports are JSON-compatible mappings serialized with sorted keys and compact
separators.  Malformed serialized input raises E008 through
:class:`~compiler.frontend.errors.PolyglotError`; missing files raise E006 and
API misuse raises E003.  No command is executed and no native binary is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "br_attach_metadata",
    "br_build_cache_manifest",
    "br_build_check_record",
    "br_build_gate_record",
    "br_build_parity_report",
    "br_build_release_report",
    "br_canonicalize_report",
    "br_check_report_schema",
    "br_collect_blockers",
    "br_diff_reports",
    "br_flatten_report",
    "br_merge_reports",
    "br_normalize_check_names",
    "br_parse_cache_manifest",
    "br_parse_report_json",
    "br_read_report",
    "br_render_report_markdown",
    "br_render_report_text",
    "br_report_exit_status",
    "br_report_fingerprint",
    "br_select_report_fields",
    "br_stable_sort_records",
    "br_summarize_results",
    "br_validate_gate_report",
    "br_write_report",
]

SCHEMA_VERSION = 1


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured reporting diagnostic."""
    raise PolyglotError(code, message, phase="build-ci")


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    """Require a string, optionally allowing the empty string."""
    if not isinstance(value, str) or (not value and not allow_empty):
        _fail("E003", f"{name} must be a{' non-empty' if not allow_empty else ''} str")
    return value


def _path(value: object, name: str) -> Path:
    """Require a path-like value and expand ``~`` only."""
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    _fail("E003", f"{name} must be a str or pathlib.Path")


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require a mapping with string keys and return a shallow copy."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{name} must be a mapping")
    return {_text(key, f"{name} key"): item for key, item in value.items()}


def _sequence(value: object, name: str) -> list[Any]:
    """Require a non-string sequence and return a mutable copy."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail("E003", f"{name} must be a sequence")
    return list(value)


def _jsonable(value: Any, where: str) -> Any:
    """Convert *value* into JSON-safe data or raise E003."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("E003", f"{where} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            _text(key, f"{where} key"): _jsonable(item, f"{where}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, f"{where}[{index}]") for index, item in enumerate(value)
        ]
    _fail("E003", f"{where} has unsupported type {type(value).__name__}")


def _canonical_text(payload: Any) -> str:
    """Serialize a JSON-safe payload with stable key and item ordering."""
    safe = _jsonable(payload, "payload")
    try:
        return json.dumps(
            safe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("E003", f"payload is not canonicalizable: {error}")


def _digest(payload: Any) -> str:
    """Canonical SHA-256 hex digest of a JSON-compatible payload."""
    return hashlib.sha256(_canonical_text(payload).encode("utf-8")).hexdigest()


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    """Flatten nested mappings into dotted paths with canonical leaf values."""
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(value[key], child, out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}[{index}]", out)
    else:
        out[prefix] = _canonical_text(value)


def _deep_merge(left: Any, right: Any, path: str, conflicts: list[str]) -> Any:
    """Recursively merge two JSON trees, recording conflicting leaf paths."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = dict(left)
        for key in right:
            child = f"{path}.{key}" if path else str(key)
            if key in result:
                result[key] = _deep_merge(result[key], right[key], child, conflicts)
            else:
                result[key] = right[key]
        return result
    if isinstance(left, list) and isinstance(right, list):
        conflicts.append(path or "root")
        return list(right)
    if _canonical_text(left) != _canonical_text(right):
        conflicts.append(path or "root")
    return right


def br_attach_metadata(report: object, *, metadata: object) -> dict[str, Any]:
    """Attach caller metadata to a report without mutating the input mapping."""
    base = _mapping(report, "report")
    extra = _mapping(metadata, "metadata")
    merged = dict(base)
    merged["metadata"] = _jsonable(extra, "metadata")
    return {"report": merged, "digest": _digest(merged), "metadata_keys": sorted(extra)}


def br_build_cache_manifest(entries: object, *, root: object) -> dict[str, Any]:
    """Build a deterministic manifest from measured cache entries."""
    directory = _path(root, "root")
    records: list[dict[str, Any]] = []
    for entry in _sequence(entries, "entries"):
        item = _mapping(entry, "entry")
        size = item.get("size_bytes", 0)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("E003", "entry size_bytes must be a non-negative int")
        records.append(
            {
                "name": _text(item.get("name"), "entry name"),
                "digest": _text(item.get("digest"), "entry digest"),
                "size_bytes": size,
            }
        )
    records.sort(key=lambda item: item["name"])
    manifest = {
        "schema": SCHEMA_VERSION,
        "root": str(directory),
        "entries": records,
    }
    return {
        "manifest": manifest,
        "digest": _digest(manifest),
        "count": len(records),
    }


def br_build_check_record(
    name: object,
    *,
    ok: object,
    detail: object = "",
    command: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one check record with a stable fingerprint."""
    label = _text(name, "name")
    if not isinstance(ok, bool):
        _fail("E003", "ok must be a bool")
    argv = tuple(_text(item, "command item") for item in _sequence(command, "command"))
    record = {
        "name": label,
        "ok": ok,
        "detail": _text(detail, "detail", allow_empty=True),
        "command": list(argv),
    }
    return {
        "name": label,
        "ok": ok,
        "detail": record["detail"],
        "command": list(argv),
        "fingerprint": _digest(record),
    }


def br_build_gate_record(
    name: object, *, checks: object, blockers: Sequence[str] = ()
) -> dict[str, Any]:
    """Build a gate record from normalized check mappings."""
    label = _text(name, "name")
    records = [
        br_build_check_record(
            item.get("name", label),
            ok=item.get("ok", False),
            detail=item.get("detail", ""),
            command=item.get("command", ()),
        )
        for item in (_mapping(entry, "check") for entry in _sequence(checks, "checks"))
    ]
    extra = tuple(_text(item, "blocker") for item in _sequence(blockers, "blockers"))
    failed = tuple(record["name"] for record in records if not record["ok"])
    all_blockers = tuple(dict.fromkeys((*failed, *extra)))
    record = {
        "name": label,
        "passed": not all_blockers,
        "checks": [
            {
                "name": item["name"],
                "ok": item["ok"],
                "detail": item["detail"],
            }
            for item in records
        ],
        "blockers": list(all_blockers),
    }
    return {
        "name": label,
        "passed": record["passed"],
        "check_count": len(records),
        "blockers": list(all_blockers),
        "digest": _digest(record),
    }


def br_build_parity_report(
    rows: object, *, required_backends: Sequence[str] = ()
) -> dict[str, Any]:
    """Build a parity report from structured cross-backend rows."""
    required = tuple(
        _text(item, "backend")
        for item in _sequence(required_backends, "required_backends")
    )
    normalized: list[dict[str, Any]] = []
    mismatches: list[str] = []
    missing: list[str] = []
    for row in _sequence(rows, "rows"):
        item = _mapping(row, "row")
        case = _text(item.get("case"), "case")
        backends = tuple(
            _text(backend, "backend")
            for backend in _sequence(item.get("backends", ()), "backends")
        )
        agreement = item.get("agreement", False)
        if not isinstance(agreement, bool):
            _fail("E003", "row agreement must be a bool")
        normalized.append(
            {"case": case, "backends": list(backends), "agreement": agreement}
        )
        if not agreement:
            mismatches.append(case)
        for backend in required:
            if backend not in backends:
                missing.append(f"{case}:{backend}")
    mismatches.sort()
    missing.sort()
    report = {
        "schema": SCHEMA_VERSION,
        "rows": sorted(normalized, key=lambda item: item["case"]),
        "required_backends": list(required),
        "mismatches": mismatches,
        "missing": missing,
    }
    return {
        "report": report,
        "passed": not mismatches and not missing,
        "mismatches": mismatches,
        "missing": missing,
        "digest": _digest(report),
    }


def br_build_release_report(
    gates: object,
    *,
    blockers: Sequence[str] = (),
    metadata: object | None = None,
) -> dict[str, Any]:
    """Build an aggregate release report from gate statuses."""
    source = _mapping(gates, "gates")
    normalized: dict[str, Any] = {}
    for name in sorted(source):
        value = source[name]
        if isinstance(value, bool):
            records = [
                {"name": name, "ok": value, "detail": "explicit status", "command": []}
            ]
        else:
            records = [
                _mapping(item, f"gate {name} check")
                for item in _sequence(value, f"gate {name}")
            ]
        gate = br_build_gate_record(name, checks=records)
        normalized[name] = {
            "passed": gate["passed"],
            "checks": gate["check_count"],
            "blockers": gate["blockers"],
        }
    extra = tuple(_text(item, "blocker") for item in _sequence(blockers, "blockers"))
    all_blockers = tuple(
        dict.fromkeys(
            (
                *(item for gate in normalized.values() for item in gate["blockers"]),
                *extra,
            )
        )
    )
    report = {
        "schema": SCHEMA_VERSION,
        "gates": normalized,
        "blockers": list(all_blockers),
        "metadata": {} if metadata is None else _jsonable(metadata, "metadata"),
    }
    return {
        "report": report,
        "digest": _digest(report),
        "passed": not all_blockers,
        "gate_count": len(normalized),
    }


def br_canonicalize_report(report: object) -> dict[str, Any]:
    """Return canonical JSON text and digest for a report mapping."""
    text = _canonical_text(_mapping(report, "report"))
    encoded = text.encode("utf-8")
    return {
        "text": text,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def br_check_report_schema(
    report: object, *, required_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """Check required keys and known field types without raising on absence."""
    value = _mapping(report, "report")
    required = tuple(
        _text(item, "key") for item in _sequence(required_keys, "required_keys")
    )
    missing = sorted(key for key in required if key not in value)
    type_errors: list[str] = []
    if "gates" in value and not isinstance(value["gates"], Mapping):
        type_errors.append("gates must be a mapping")
    if "checks" in value and not isinstance(value["checks"], Sequence):
        type_errors.append("checks must be a sequence")
    if "blockers" in value and not isinstance(value["blockers"], Sequence):
        type_errors.append("blockers must be a sequence")
    return {
        "valid": not missing and not type_errors,
        "missing": missing,
        "type_errors": sorted(type_errors),
        "key_count": len(value),
    }


def br_collect_blockers(report: object) -> dict[str, Any]:
    """Collect explicit blockers and failures from gates and checks."""
    value = _mapping(report, "report")
    blockers = [
        _text(item, "blocker")
        for item in _sequence(value.get("blockers", ()), "blockers")
    ]
    for name in sorted(_mapping(value.get("gates", {}), "gates")):
        gate = value["gates"][name]
        if isinstance(gate, Mapping) and not gate.get("passed", True):
            blockers.append(f"{name}: gate failed")
    for check in _sequence(value.get("checks", ()), "checks"):
        record = _mapping(check, "check")
        if not record.get("ok", True):
            blockers.append(f"{record.get('name', 'check')}: check failed")
    unique = sorted(set(blockers))
    return {"blockers": unique, "count": len(unique)}


def br_diff_reports(left: object, right: object) -> dict[str, Any]:
    """Flatten and compare two reports into added, removed, and changed paths."""
    left_flat: dict[str, Any] = {}
    right_flat: dict[str, Any] = {}
    _flatten(_mapping(left, "left"), "", left_flat)
    _flatten(_mapping(right, "right"), "", right_flat)
    added = sorted(set(right_flat) - set(left_flat))
    removed = sorted(set(left_flat) - set(right_flat))
    changed = [
        {"path": path, "left": left_flat[path], "right": right_flat[path]}
        for path in sorted(set(left_flat) & set(right_flat))
        if left_flat[path] != right_flat[path]
    ]
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "equal": not added and not removed and not changed,
    }


def br_flatten_report(report: object) -> dict[str, Any]:
    """Flatten a report into deterministic dotted leaf paths."""
    flat: dict[str, Any] = {}
    _flatten(_mapping(report, "report"), "", flat)
    entries = [
        {"path": path, "value": flat[path], "digest": _digest(flat[path])}
        for path in sorted(flat)
    ]
    return {"entries": entries, "count": len(entries)}


def br_merge_reports(reports: object) -> dict[str, Any]:
    """Merge report mappings in order, recording conflicting leaf paths."""
    source = _sequence(reports, "reports")
    if not source:
        _fail("E003", "reports must not be empty")
    merged: Any = {}
    conflicts: list[str] = []
    for index, report in enumerate(source):
        merged = _deep_merge(
            merged, _mapping(report, f"report[{index}]"), "", conflicts
        )
    unique = sorted(set(conflicts))
    canonical = _jsonable(merged, "merged")
    return {
        "report": canonical,
        "sources": list(range(len(source))),
        "conflicts": unique,
        "digest": _digest(canonical),
    }


def br_normalize_check_names(names: object) -> dict[str, Any]:
    """Sort and deduplicate check names while reporting duplicates."""
    values = tuple(_text(item, "name") for item in _sequence(names, "names"))
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return {
        "names": sorted(seen),
        "duplicates": sorted(duplicates),
        "count": len(seen),
    }


def br_parse_cache_manifest(text: object) -> dict[str, Any]:
    """Parse and validate a serialized cache manifest (E008 on bad data)."""
    raw = _text(text, "text")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        _fail("E008", f"malformed cache manifest JSON: {error}")
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA_VERSION:
        _fail("E008", "cache manifest has an unsupported schema")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        _fail("E008", "cache manifest entries must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            _fail("E008", f"cache manifest entry {index} is malformed")
    return {
        "manifest": dict(payload),
        "entry_count": len(entries),
        "digest": _digest(payload),
    }


def br_parse_report_json(text: object) -> dict[str, Any]:
    """Parse canonical report JSON into a mapping, raising E008 if invalid."""
    raw = _text(text, "text")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        _fail("E008", f"malformed report JSON: {error}")
    if not isinstance(payload, Mapping):
        _fail("E008", "report JSON must decode to an object")
    return {
        "report": dict(payload),
        "digest": _digest(payload),
        "bytes": len(raw.encode("utf-8")),
    }


def br_read_report(path: object) -> dict[str, Any]:
    """Read and parse a report file, raising E006 when it is absent."""
    target = _path(path, "path")
    if not target.is_file():
        _fail("E006", f"report file is missing: {target}")
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as error:
        _fail("E004", f"cannot read report {target}: {error}")
    parsed = br_parse_report_json(raw)
    return {
        "path": str(target),
        "report": parsed["report"],
        "digest": parsed["digest"],
        "bytes": parsed["bytes"],
    }


def br_render_report_markdown(report: object) -> dict[str, Any]:
    """Render a gate report as deterministic Markdown."""
    value = _mapping(report, "report")
    lines = ["# Release Gate", ""]
    status = "PASS" if br_report_exit_status(value)["passed"] else "FAIL"
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for check in sorted(
        (
            _mapping(item, "check")
            for item in _sequence(value.get("checks", ()), "checks")
        ),
        key=lambda item: str(item.get("name", "")),
    ):
        name = str(check.get("name", "")).replace("|", "\\|")
        detail = str(check.get("detail", "")).replace("|", "\\|").replace("\n", " ")
        state = "PASS" if check.get("ok", False) else "FAIL"
        lines.append(f"| {name} | {state} | {detail} |")
    for gate_name in sorted(_mapping(value.get("gates", {}), "gates")):
        gate = value["gates"][gate_name]
        state = (
            "PASS"
            if isinstance(gate, Mapping) and gate.get("passed", False)
            else "FAIL"
        )
        lines.append(f"| {gate_name} | {state} | gate |")
    return {"text": "\n".join(lines), "lines": lines, "status": status}


def br_render_report_text(report: object) -> dict[str, Any]:
    """Render a gate report as a deterministic plain-text summary."""
    value = _mapping(report, "report")
    status_info = br_report_exit_status(value)
    lines = [
        f"REPORT: {'PASS' if status_info['passed'] else 'FAIL'}",
        f"blockers: {status_info['blocker_count']}",
    ]
    for check in sorted(
        (
            _mapping(item, "check")
            for item in _sequence(value.get("checks", ()), "checks")
        ),
        key=lambda item: str(item.get("name", "")),
    ):
        state = "PASS" if check.get("ok", False) else "FAIL"
        detail = str(check.get("detail", "")).replace("\n", " ")
        suffix = f" - {detail}" if detail else ""
        lines.append(f"  {state} {check.get('name', 'check')}{suffix}")
    return {
        "text": "\n".join(lines),
        "lines": lines,
        "status": "PASS" if status_info["passed"] else "FAIL",
    }


def br_report_exit_status(report: object, *, strict: bool = True) -> dict[str, Any]:
    """Derive a process exit code and reason from a gate report."""
    value = _mapping(report, "report")
    if not isinstance(strict, bool):
        _fail("E003", "strict must be a bool")
    br_validate_gate_report(value)
    blockers = br_collect_blockers(value)["blockers"]
    failed = blockers if strict else []
    return {
        "exit_code": 1 if failed else 0,
        "passed": not failed,
        "reason": "; ".join(failed) if failed else "all checks passed",
        "blocker_count": len(blockers),
    }


def br_report_fingerprint(report: object) -> dict[str, Any]:
    """Compute a canonical content fingerprint for a report mapping."""
    value = _mapping(report, "report")
    canonical = br_canonicalize_report(value)
    return {
        "algorithm": "sha256",
        "digest": canonical["digest"],
        "bytes": canonical["bytes"],
    }


def br_select_report_fields(report: object, *, fields: Sequence[str]) -> dict[str, Any]:
    """Select named top-level fields from a report without mutating it."""
    value = _mapping(report, "report")
    names = tuple(_text(item, "field") for item in _sequence(fields, "fields"))
    selected = {name: value[name] for name in names if name in value}
    missing = sorted(name for name in names if name not in value)
    return {
        "selected": selected,
        "missing": missing,
        "count": len(selected),
        "digest": _digest(selected),
    }


def br_stable_sort_records(records: object, *, key: str) -> dict[str, Any]:
    """Sort record mappings by one field with canonical JSON tie-breaking."""
    field = _text(key, "key")
    items = [_mapping(item, "record") for item in _sequence(records, "records")]
    for item in items:
        if field not in item:
            _fail("E003", f"record is missing sort key {field!r}")
    sorted_items = sorted(
        items,
        key=lambda item: (_canonical_text(item.get(field)), _canonical_text(item)),
    )
    return {
        "records": sorted_items,
        "key": field,
        "count": len(sorted_items),
        "digest": _digest(sorted_items),
    }


def br_summarize_results(results: object) -> dict[str, Any]:
    """Summarize check results with deterministic counts and pass rate."""
    names: set[str] = set()
    passed = 0
    failed = 0
    for item in _sequence(results, "results"):
        record = _mapping(item, "result")
        name = _text(record.get("name"), "result name")
        names.add(name)
        if not isinstance(record.get("ok"), bool):
            _fail("E003", "result ok must be a bool")
        if record["ok"]:
            passed += 1
        else:
            failed += 1
    total = passed + failed
    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "names": sorted(names),
        "rate": round(passed / total, 6) if total else 0.0,
    }


def br_validate_gate_report(report: object) -> dict[str, Any]:
    """Validate the gate-report shape, raising E008 for malformed reports."""
    value = _mapping(report, "report")
    if not isinstance(value.get("checks", ()), Sequence) or isinstance(
        value.get("checks", ()), (str, bytes, bytearray)
    ):
        _fail("E008", "report checks must be a list")
    if not isinstance(value.get("blockers", ()), Sequence) or isinstance(
        value.get("blockers", ()), (str, bytes, bytearray)
    ):
        _fail("E008", "report blockers must be a list")
    checks = _sequence(value.get("checks", ()), "checks")
    for index, item in enumerate(checks):
        if not isinstance(item, Mapping):
            _fail("E008", f"check {index} must be an object")
        if not isinstance(item.get("name"), str) or not isinstance(
            item.get("ok"), bool
        ):
            _fail("E008", f"check {index} must have string name and bool ok")
    blockers = _sequence(value.get("blockers", ()), "blockers")
    for index, item in enumerate(blockers):
        if not isinstance(item, str):
            _fail("E008", f"blocker {index} must be a string")
    return {
        "valid": True,
        "check_count": len(checks),
        "blocker_count": len(blockers),
        "digest": _digest(value),
    }


def br_write_report(path: object, *, report: object) -> dict[str, Any]:
    """Atomically write canonical report JSON to an explicit path."""
    target = _path(path, "path")
    payload = _canonical_text(_mapping(report, "report"))
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{digest[:12]}.tmp")
        temporary.write_bytes(encoded)
        replaced = target.exists()
        os.replace(temporary, target)
    except OSError as error:
        _fail("E004", f"cannot write report {target}: {error}")
    return {
        "path": str(target),
        "digest": digest,
        "bytes": len(encoded),
        "replaced": replaced,
    }
