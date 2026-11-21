"""Pure modeling of cross-backend parity comparisons.

These helpers consume structured observations and snapshots supplied by the
caller; they never execute a native binary or spawn a command.  Values are
canonicalized before comparison so results are independent of dict insertion
order and Python's per-process ``hash()``.  Invalid inputs raise E003 through
:class:`~compiler.frontend.errors.PolyglotError`, unsupported backends or
platforms raise E007, and malformed serialized snapshots raise E008.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "bp_build_parity_matrix",
    "bp_build_snapshot",
    "bp_canonicalize_observation",
    "bp_compare_diagnostics",
    "bp_compare_numeric_outputs",
    "bp_compare_observations",
    "bp_compare_text_outputs",
    "bp_compare_values",
    "bp_extract_diagnostics",
    "bp_find_mismatches",
    "bp_group_by_case",
    "bp_index_by_backend",
    "bp_load_snapshot",
    "bp_measure_agreement",
    "bp_merge_snapshots",
    "bp_normalize_output",
    "bp_normalize_value",
    "bp_parity_fingerprint",
    "bp_plan_parity_cases",
    "bp_report_parity",
    "bp_require_backend",
    "bp_require_platform",
    "bp_select_backends",
    "bp_snapshot_diff",
    "bp_validate_snapshot",
]

SNAPSHOT_SCHEMA = 1
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DIAGNOSTIC_RE = re.compile(
    r"\b(?P<code>E\d{3})[ \t]*:[ \t]*(?P<message>.*?)"
    r"(?:[ \t]+at[ \t]+(?P<line>\d+):(?P<col>\d+))?[^\S\r\n]*$",
    re.MULTILINE,
)


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured parity diagnostic."""
    raise PolyglotError(code, message, phase="build-ci")


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    """Require a string, optionally allowing the empty string."""
    if not isinstance(value, str) or (not value and not allow_empty):
        _fail("E003", f"{name} must be a{' non-empty' if not allow_empty else ''} str")
    return value


def _component(value: object, name: str) -> str:
    """Validate a backend, platform, or case identifier."""
    text = _text(value, name)
    if text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        _fail("E003", f"{name} must be a safe identifier: {text!r}")
    return text


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require a string-keyed mapping and return a shallow copy."""
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
    """Serialize a JSON-safe payload with stable ordering."""
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


def _snapshot_key(backend: str, case: str) -> str:
    """Stable composite key for one backend/case observation."""
    return f"{backend}::{case}"


def bp_build_parity_matrix(observations: object) -> dict[str, Any]:
    """Build a case-by-backend matrix from canonical observations."""
    index = bp_index_by_backend(observations)["index"]
    backends = sorted(index)
    cases = sorted({case for per_case in index.values() for case in per_case})
    rows: list[dict[str, Any]] = []
    for case in cases:
        values = []
        for backend in backends:
            if case not in index[backend]:
                continue
            record = index[backend][case]
            comparable = {
                "value": record["value"],
                "status": record["status"],
                "output": record["output"],
            }
            values.append({"backend": backend, "digest": _digest(comparable)})
        agreement = len({item["digest"] for item in values}) <= 1
        rows.append(
            {
                "case": case,
                "backends": values,
                "agreement": agreement,
                "backend_count": len(values),
            }
        )
    return {
        "rows": rows,
        "backends": backends,
        "cases": cases,
        "digest": _digest(rows),
    }


def bp_build_snapshot(
    observations: object, *, platform_name: object, toolchain: object
) -> dict[str, Any]:
    """Build a versioned snapshot of canonical observations."""
    platform = _text(platform_name, "platform_name").strip().lower()
    versions = _mapping(toolchain, "toolchain")
    normalized_toolchain: dict[str, str] = {}
    for name, value in versions.items():
        normalized_toolchain[_text(name, "toolchain key")] = _text(
            value, f"toolchain[{name!r}]"
        )
    records = [
        bp_canonicalize_observation(item)["record"]
        for item in _sequence(observations, "observations")
    ]
    records.sort(key=lambda item: (item["backend"], item["case"]))
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "platform": platform,
        "toolchain": dict(sorted(normalized_toolchain.items())),
        "observations": records,
    }
    return {
        "snapshot": snapshot,
        "digest": _digest(snapshot),
        "observation_count": len(records),
    }


def bp_canonicalize_observation(observation: object) -> dict[str, Any]:
    """Normalize one backend observation into a comparable record."""
    item = _mapping(observation, "observation")
    record = {
        "backend": _text(item.get("backend"), "backend"),
        "case": _text(item.get("case"), "case"),
        "value": _jsonable(item.get("value"), "value"),
        "status": _text(item.get("status", "ok"), "status"),
        "platform": _text(item.get("platform", "unknown"), "platform", allow_empty=True)
        or "unknown",
        "output": _text(item.get("output", ""), "output", allow_empty=True),
    }
    return {"record": record, "digest": _digest(record)}


def bp_compare_diagnostics(left: object, right: object) -> dict[str, Any]:
    """Compare diagnostic lists by sorted structured identity, not by order."""
    normalized_left = [
        _mapping(item, "left diagnostic") for item in _sequence(left, "left")
    ]
    normalized_right = [
        _mapping(item, "right diagnostic") for item in _sequence(right, "right")
    ]

    def key(item: Mapping[str, Any]) -> str:
        return _canonical_text(
            {
                "code": item.get("code", ""),
                "line": item.get("line", 0),
                "col": item.get("col", 0),
                "message": item.get("message", ""),
            }
        )

    left_keys = sorted(key(item) for item in normalized_left)
    right_keys = sorted(key(item) for item in normalized_right)
    differences = [
        {"side": "left", "value": value}
        for value in sorted(set(left_keys) - set(right_keys))
    ]
    differences.extend(
        {"side": "right", "value": value}
        for value in sorted(set(right_keys) - set(left_keys))
    )
    return {
        "equal": left_keys == right_keys,
        "left": left_keys,
        "right": right_keys,
        "differences": differences,
    }


def bp_compare_numeric_outputs(
    left: object,
    right: object,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare numeric outputs with explicit relative and absolute tolerances."""
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or relative_tolerance < 0
    ):
        _fail("E003", "relative_tolerance must be non-negative")
    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or absolute_tolerance < 0
    ):
        _fail("E003", "absolute_tolerance must be non-negative")
    values: list[float] = []
    for value, name in ((left, "left"), (right, "right")):
        if isinstance(value, bool):
            _fail("E003", f"{name} must be numeric")
        try:
            number = float(value)
        except TypeError, ValueError:
            _fail("E003", f"{name} is not numeric: {value!r}")
        if not math.isfinite(number):
            _fail("E003", f"{name} must be finite")
        values.append(number)
    equal = math.isclose(
        values[0],
        values[1],
        rel_tol=float(relative_tolerance),
        abs_tol=float(absolute_tolerance),
    )
    return {
        "equal": equal,
        "left": values[0],
        "right": values[1],
        "delta": abs(values[0] - values[1]),
        "tolerance": {
            "relative": float(relative_tolerance),
            "absolute": float(absolute_tolerance),
        },
        "reason": "within tolerance" if equal else "outside tolerance",
    }


def bp_compare_observations(left: object, right: object) -> dict[str, Any]:
    """Compare two observations field by field after canonicalization."""
    first = bp_canonicalize_observation(left)["record"]
    second = bp_canonicalize_observation(right)["record"]
    differences: list[dict[str, Any]] = []
    for field in sorted(set(first) | set(second)):
        left_text = _canonical_text(first.get(field))
        right_text = _canonical_text(second.get(field))
        if left_text != right_text:
            differences.append({"field": field, "left": left_text, "right": right_text})
    return {
        "equal": not differences,
        "differences": differences,
        "left_digest": _digest(first),
        "right_digest": _digest(second),
    }


def bp_compare_text_outputs(
    left: object,
    right: object,
    *,
    ignore_whitespace: bool = True,
    case_sensitive: bool = True,
) -> dict[str, Any]:
    """Compare normalized text outputs line by line."""
    if not isinstance(ignore_whitespace, bool) or not isinstance(case_sensitive, bool):
        _fail("E003", "comparison flags must be bools")
    first = bp_normalize_output(left)["lines"]
    second = bp_normalize_output(right)["lines"]
    if not case_sensitive:
        first = [line.lower() for line in first]
        second = [line.lower() for line in second]
    if ignore_whitespace:
        first = [line.strip() for line in first if line.strip()]
        second = [line.strip() for line in second if line.strip()]
    differences = [
        {"index": index, "left": a, "right": b}
        for index, (a, b) in enumerate(zip(first, second))
        if a != b
    ]
    if len(first) != len(second):
        for index in range(min(len(first), len(second)), max(len(first), len(second))):
            source = first if len(first) > len(second) else second
            side = "left" if len(first) > len(second) else "right"
            differences.append({"index": index, "side": side, "value": source[index]})
    return {
        "equal": not differences,
        "left_lines": first,
        "right_lines": second,
        "differences": differences,
        "digest": _digest([first, second]),
    }


def bp_compare_values(left: object, right: object) -> dict[str, Any]:
    """Compare two values through canonical serialization and SHA-256."""
    first = bp_normalize_value(left)
    second = bp_normalize_value(right)
    equal = first["text"] == second["text"] and first["kind"] == second["kind"]
    return {
        "equal": equal,
        "left": first["text"],
        "right": second["text"],
        "kinds": [first["kind"], second["kind"]],
        "reason": "identical" if equal else "canonical forms differ",
    }


def bp_extract_diagnostics(
    text: object, *, pattern: str | None = None
) -> dict[str, Any]:
    """Extract structured diagnostics from text with a caller-supplied pattern."""
    raw = _text(text, "text", allow_empty=True)
    expression = _text(pattern, "pattern") if pattern is not None else None
    try:
        compiled = (
            re.compile(expression, re.MULTILINE) if expression else _DIAGNOSTIC_RE
        )
    except re.error as error:
        _fail("E003", f"invalid diagnostic pattern: {error}")
    diagnostics: list[dict[str, Any]] = []
    for match in compiled.finditer(raw):
        groups = match.groupdict()
        diagnostics.append(
            {
                "code": groups.get("code", ""),
                "message": (groups.get("message") or "").strip(),
                "line": int(groups["line"]) if groups.get("line") else 0,
                "col": int(groups["col"]) if groups.get("col") else 0,
            }
        )
    return {"diagnostics": diagnostics, "count": len(diagnostics)}


def bp_find_mismatches(matrix: object) -> dict[str, Any]:
    """Extract disagreeing cases from a parity matrix."""
    value = _mapping(matrix, "matrix")
    rows = _sequence(value.get("rows", ()), "rows")
    mismatches: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = _mapping(row, f"row {index}")
        if not isinstance(item.get("agreement"), bool):
            _fail("E003", "row agreement must be a bool")
        if item["agreement"]:
            continue
        records = [
            _mapping(entry, "backend")
            for entry in _sequence(item.get("backends", ()), "backends")
        ]
        mismatches.append(
            {
                "case": _text(item.get("case"), "case"),
                "backends": sorted(
                    {_text(record.get("backend"), "backend") for record in records}
                ),
                "digests": sorted(
                    {_text(record.get("digest"), "digest") for record in records}
                ),
            }
        )
    mismatches.sort(key=lambda item: item["case"])
    return {"mismatches": mismatches, "count": len(mismatches)}


def bp_group_by_case(observations: object) -> dict[str, Any]:
    """Group canonical observations by case name in stable order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in _sequence(observations, "observations"):
        record = bp_canonicalize_observation(item)["record"]
        groups.setdefault(record["case"], []).append(record)
    for records in groups.values():
        records.sort(key=lambda item: item["backend"])
    ordered = {case: groups[case] for case in sorted(groups)}
    return {
        "groups": ordered,
        "case_count": len(ordered),
        "observation_count": sum(len(items) for items in ordered.values()),
    }


def bp_index_by_backend(observations: object) -> dict[str, Any]:
    """Index canonical observations as backend -> case -> record."""
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for item in _sequence(observations, "observations"):
        record = bp_canonicalize_observation(item)["record"]
        per_case = index.setdefault(record["backend"], {})
        if record["case"] in per_case:
            _fail(
                "E003",
                f"duplicate observation for {record['backend']}/{record['case']}",
            )
        per_case[record["case"]] = {**record, "digest": _digest(record)}
    ordered = {
        backend: dict(sorted(index[backend].items())) for backend in sorted(index)
    }
    return {
        "index": ordered,
        "backends": sorted(ordered),
        "cases": sorted({case for per_case in ordered.values() for case in per_case}),
    }


def bp_load_snapshot(text: object) -> dict[str, Any]:
    """Parse a serialized snapshot, raising E008 for malformed payloads."""
    raw = _text(text, "text")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        _fail("E008", f"malformed snapshot JSON: {error}")
    validated = bp_validate_snapshot(payload)
    return {
        "snapshot": payload,
        "digest": validated["digest"],
        "observation_count": validated["observation_count"],
    }


def bp_measure_agreement(matrix: object) -> dict[str, Any]:
    """Measure per-case agreement and backend coverage in a matrix."""
    value = _mapping(matrix, "matrix")
    rows = _sequence(value.get("rows", ()), "rows")
    matching = 0
    by_backend: dict[str, int] = {}
    for row in rows:
        item = _mapping(row, "row")
        if not isinstance(item.get("agreement"), bool):
            _fail("E003", "row agreement must be a bool")
        matching += 1 if item["agreement"] else 0
        for entry in _sequence(item.get("backends", ()), "backends"):
            backend = _text(_mapping(entry, "backend").get("backend"), "backend")
            by_backend[backend] = by_backend.get(backend, 0) + 1
    total = len(rows)
    return {
        "total_cases": total,
        "matching_cases": matching,
        "mismatching_cases": total - matching,
        "rate": round(matching / total, 6) if total else 0.0,
        "by_backend": dict(sorted(by_backend.items())),
    }


def bp_merge_snapshots(snapshots: object) -> dict[str, Any]:
    """Merge compatible snapshots, rejecting conflicting observation digests."""
    items = _sequence(snapshots, "snapshots")
    if not items:
        _fail("E003", "snapshots must not be empty")
    validated = [bp_validate_snapshot(item)["snapshot"] for item in items]
    observations: dict[str, dict[str, Any]] = {}
    for snapshot in validated:
        for observation in snapshot["observations"]:
            key = _snapshot_key(observation["backend"], observation["case"])
            digest = _digest(observation)
            if key in observations and _digest(observations[key]) != digest:
                _fail("E008", f"conflicting observation for {key}")
            observations[key] = observation
    merged = {
        "schema": SNAPSHOT_SCHEMA,
        "platform": validated[0]["platform"],
        "toolchain": validated[0]["toolchain"],
        "observations": [observations[key] for key in sorted(observations)],
    }
    return {
        "snapshot": merged,
        "sources": list(range(len(items))),
        "observation_count": len(observations),
        "digest": _digest(merged),
    }


def bp_normalize_output(
    stdout: object,
    *,
    strip_ansi: bool = True,
    normalize_newlines: bool = True,
) -> dict[str, Any]:
    """Normalize captured output text for stable textual comparison."""
    text = _text(stdout, "stdout", allow_empty=True)
    if not isinstance(strip_ansi, bool) or not isinstance(normalize_newlines, bool):
        _fail("E003", "normalization flags must be bools")
    if strip_ansi:
        text = _ANSI_RE.sub("", text)
    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    return {
        "text": text,
        "lines": lines,
        "line_count": len(lines),
        "digest": _digest(lines),
    }


def bp_normalize_value(value: object) -> dict[str, Any]:
    """Normalize one value into a kind, canonical text form, and digest."""
    if value is None:
        kind, text = "null", "null"
    elif isinstance(value, bool):
        kind, text = "boolean", "true" if value else "false"
    elif isinstance(value, int):
        kind, text = "number", str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            _fail("E003", "float value must be finite")
        kind, text = "number", repr(value)
    elif isinstance(value, str):
        kind, text = "string", json.dumps(value, ensure_ascii=True)
    elif isinstance(value, (list, tuple)):
        kind, text = "array", _canonical_text(value)
    elif isinstance(value, Mapping):
        kind, text = "object", _canonical_text(value)
    else:
        _fail("E003", f"unsupported value type: {type(value).__name__}")
    return {
        "kind": kind,
        "text": text,
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def bp_parity_fingerprint(matrix: object) -> dict[str, Any]:
    """Fingerprint a parity matrix with canonical SHA-256."""
    value = _mapping(matrix, "matrix")
    text = _canonical_text(value)
    encoded = text.encode("utf-8")
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def bp_plan_parity_cases(cases: object, *, backends: Sequence[str]) -> dict[str, Any]:
    """Plan a deterministic backend-by-case comparison grid."""
    names = tuple(_component(item, "case") for item in _sequence(cases, "cases"))
    targets = tuple(
        _component(item, "backend") for item in _sequence(backends, "backends")
    )
    if not names or not targets:
        _fail("E003", "cases and backends must not be empty")
    rows = [{"case": case, "backends": list(targets)} for case in names]
    return {"cases": rows, "count": len(rows), "backend_count": len(targets)}


def bp_report_parity(
    matrix: object, *, required_backends: Sequence[str] = ()
) -> dict[str, Any]:
    """Build a pass/fail parity report with blockers and a stable digest."""
    required = tuple(
        _component(item, "backend")
        for item in _sequence(required_backends, "required_backends")
    )
    mismatches = bp_find_mismatches(matrix)["mismatches"]
    coverage = bp_measure_agreement(matrix)
    blockers = [f"{item['case']}: backend disagreement" for item in mismatches]
    value = _mapping(matrix, "matrix")
    for row in _sequence(value.get("rows", ()), "rows"):
        item = _mapping(row, "row")
        case = _text(item.get("case"), "case")
        present = {
            _text(_mapping(entry, "backend").get("backend"), "backend")
            for entry in _sequence(item.get("backends", ()), "backends")
        }
        blockers.extend(
            f"{case}: missing {backend}"
            for backend in required
            if backend not in present
        )
    blockers.sort()
    report = {
        "schema": SNAPSHOT_SCHEMA,
        "coverage": coverage,
        "mismatches": mismatches,
        "blockers": blockers,
        "required_backends": list(required),
    }
    return {
        "report": report,
        "passed": not blockers,
        "blockers": blockers,
        "digest": _digest(report),
    }


def bp_require_backend(
    backend: object, *, supported: Sequence[str], strict: bool = True
) -> dict[str, Any]:
    """Validate a backend name against an explicit support list."""
    name = _component(backend, "backend").strip().lower()
    allowed = tuple(
        _component(item, "backend").strip().lower()
        for item in _sequence(supported, "supported")
    )
    supported_flag = name in allowed
    if not supported_flag and strict:
        _fail("E007", f"unsupported backend: {name}")
    return {
        "backend": name,
        "supported": supported_flag,
        "reason": "supported" if supported_flag else "unsupported backend",
        "strict": strict,
    }


def bp_require_platform(
    platform_name: object, *, supported: Sequence[str], strict: bool = True
) -> dict[str, Any]:
    """Validate a platform name against an explicit support list."""
    name = _component(platform_name, "platform_name").strip().lower()
    allowed = tuple(
        _component(item, "platform").strip().lower()
        for item in _sequence(supported, "supported")
    )
    supported_flag = name in allowed
    if not supported_flag and strict:
        _fail("E007", f"unsupported platform: {name}")
    return {
        "platform": name,
        "supported": supported_flag,
        "reason": "supported" if supported_flag else "unsupported platform",
        "strict": strict,
    }


def bp_select_backends(
    observations: object, *, backends: Sequence[str]
) -> dict[str, Any]:
    """Select observations for named backends without mutating the input."""
    targets = tuple(_text(item, "backend") for item in _sequence(backends, "backends"))
    records = [
        bp_canonicalize_observation(item)["record"]
        for item in _sequence(observations, "observations")
    ]
    selected = [record for record in records if record["backend"] in targets]
    present = {record["backend"] for record in selected}
    missing = sorted(name for name in targets if name not in present)
    selected.sort(key=lambda item: (item["backend"], item["case"]))
    return {
        "observations": selected,
        "selected": sorted(present),
        "missing": missing,
        "count": len(selected),
    }


def bp_snapshot_diff(left: object, right: object) -> dict[str, Any]:
    """Compare two snapshots by backend/case observation digests."""
    first = bp_validate_snapshot(left)["snapshot"]
    second = bp_validate_snapshot(right)["snapshot"]
    left_map = {
        _snapshot_key(item["backend"], item["case"]): _digest(item)
        for item in first["observations"]
    }
    right_map = {
        _snapshot_key(item["backend"], item["case"]): _digest(item)
        for item in second["observations"]
    }
    added = sorted(set(right_map) - set(left_map))
    removed = sorted(set(left_map) - set(right_map))
    changed = sorted(
        key for key in set(left_map) & set(right_map) if left_map[key] != right_map[key]
    )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "equal": not added and not removed and not changed,
    }


def bp_validate_snapshot(snapshot: object) -> dict[str, Any]:
    """Validate a snapshot mapping, raising E008 for malformed payloads."""
    if not isinstance(snapshot, Mapping):
        _fail("E008", "snapshot must be a JSON object")
    value = _mapping(snapshot, "snapshot")
    if value.get("schema") != SNAPSHOT_SCHEMA:
        _fail("E008", f"unsupported snapshot schema: {value.get('schema')!r}")
    if not isinstance(value.get("platform"), str):
        _fail("E008", "snapshot platform must be a string")
    if not isinstance(value.get("toolchain"), Mapping):
        _fail("E008", "snapshot toolchain must be an object")
    observations = value.get("observations")
    if not isinstance(observations, list):
        _fail("E008", "snapshot observations must be a list")
    for index, item in enumerate(observations):
        if not isinstance(item, Mapping):
            _fail("E008", f"snapshot observation {index} must be an object")
        if not isinstance(item.get("backend"), str) or not isinstance(
            item.get("case"), str
        ):
            _fail("E008", f"snapshot observation {index} has invalid identity")
        if "value" not in item:
            _fail("E008", f"snapshot observation {index} has no value")
    return {
        "valid": True,
        "snapshot": value,
        "observation_count": len(observations),
        "digest": _digest(value),
    }
