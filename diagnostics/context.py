"""Deterministic diagnostic-context helpers for the framework rollout.

A context is a plain, JSON-safe ``dict`` carrying source text, compiler phase,
backend, path, code aliases, one optional span, and a revision counter.  Every
public helper is pure: it validates its inputs, returns a new structured value,
and raises :class:`~compiler.frontend.errors.PolyglotError` for API misuse
(``E003``) or malformed serialized state (``E008``).  Span and position helpers
delegate to :mod:`api.visualizer` so the editor-facing range shape stays
identical to the rest of the toolchain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NoReturn

from api.visualizer import offset_to_lsp_position, span_to_lsp_range
from compiler.frontend.errors import PolyglotError
from runtime.diagnostics import ERROR_CODES

__all__ = [
    "dctx_attach_span",
    "dctx_code_summary",
    "dctx_context_digest",
    "dctx_context_to_json",
    "dctx_filter_severity",
    "dctx_from_json",
    "dctx_group_by_phase",
    "dctx_line_col_to_offset",
    "dctx_line_count",
    "dctx_line_text",
    "dctx_make_record",
    "dctx_merge_records",
    "dctx_new",
    "dctx_normalize_severity",
    "dctx_offset_to_line_col",
    "dctx_record_from_exception",
    "dctx_record_from_mapping",
    "dctx_record_position",
    "dctx_resolve_code",
    "dctx_span_clamp",
    "dctx_span_overlap",
    "dctx_span_to_lsp",
    "dctx_span_union",
    "dctx_with_code_alias",
    "dctx_with_source",
]

_CONTEXT_SCHEMA: Final = "polyglot-diagnostic-context/v2"
_PHASE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lex": "lexing",
        "parse": "parsing",
        "parser": "parsing",
        "type": "typecheck",
        "types": "typecheck",
        "mono": "monomorphization",
        "emit": "codegen",
    }
)
_BACKEND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"py": "python", "cpython": "python", "rs": "rust", "native": "rust"}
)
_SEVERITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"warn": "warning", "fatal": "error", "critical": "error", "note": "info"}
)
_SEVERITY_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {"error": 3, "warning": 2, "info": 1}
)
_CODE_HINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LexError": "E003",
        "ParseError": "E003",
        "TranslateError": "E003",
        "TypeCheckError": "E001",
        "BorrowError": "E002",
        "TerminationError": "E002",
        "ToolchainError": "E004",
        "DependencyCycleError": "E005",
        "MissingArtifactError": "E006",
        "UnsupportedNativePatternError": "E007",
        "InvalidIntegerParseLiteralError": "E008",
    }
)
_CONTEXT_KEYS: Final[tuple[str, ...]] = (
    "source",
    "phase",
    "backend",
    "path",
    "code_map",
    "span",
    "line",
    "col",
    "revision",
    "records",
)


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured diagnostic error for invalid helper input."""
    raise PolyglotError(code, message, phase="context")


def _text(value: object, label: str) -> str:
    """Validate a required string argument."""
    if not isinstance(value, str):
        _fail("E003", f"{label} must be a str")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    """Validate a required mapping argument."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    return value


def _integer(value: object, label: str) -> int:
    """Validate an integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{label} must be an int")
    return value


def _canonical(
    value: object, aliases: Mapping[str, str], label: str, *, default: str
) -> str:
    """Normalize an enum-like value through a caller-supplied alias table."""
    text = _text(value, label).strip()
    if not text:
        return default
    lowered = text.lower()
    return aliases.get(lowered, lowered)


def _code(value: object, *, code_map: Mapping[str, str] | None = None) -> str:
    """Resolve a diagnostic code or alias to the public E001-E008 surface."""
    if isinstance(value, BaseException):
        direct = getattr(value, "code", None)
        if isinstance(direct, str) and direct.strip().upper() in ERROR_CODES:
            return direct.strip().upper()
        value = type(value).__name__
    text = _text(value, "code").strip()
    for candidate in (text, text.lower(), text.upper(), type(value).__name__):
        mapped = (code_map or {}).get(candidate)
        if mapped is not None:
            text = str(mapped).strip().upper()
            break
    if text.upper() in ERROR_CODES:
        return text.upper()
    hinted = _CODE_HINTS.get(text) or _CODE_HINTS.get(text.title())
    if hinted:
        return hinted
    _fail("E003", f"unmappable diagnostic code {value!r}")


def _severity(value: object) -> str:
    """Normalize a severity label and return its canonical spelling."""
    text = _text(value, "severity").strip().lower()
    normalized = _SEVERITY_ALIASES.get(text, text)
    if normalized not in _SEVERITY_RANK:
        _fail("E003", f"unknown severity {value!r}")
    return normalized


def _span(value: object, *, length: int | None = None) -> tuple[int, int] | None:
    """Validate and clamp a half-open span; ``None`` is passed through."""
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
    ):
        _fail("E003", "span must be a two-item pair of ints")
    start, end = int(value[0]), int(value[1])
    start, end = max(0, start), max(0, end)
    if end < start:
        start, end = end, start
    if length is not None:
        start, end = min(start, length), min(end, length)
    return start, end


def _new_context(
    source: str,
    phase: str,
    backend: str,
    path: str | None,
    code_map: Mapping[str, Any],
    span: tuple[int, int] | None,
    *,
    line: int = 1,
    col: int = 1,
    revision: int = 0,
    records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble a validated context dictionary with a stable key order."""
    aliases = {
        _text(str(key), "code_map key").strip().lower(): _code(item, code_map={})
        for key, item in sorted(code_map.items(), key=lambda pair: str(pair[0]))
    }
    if path is not None:
        path = _text(path, "path")
    return {
        "source": source,
        "phase": phase,
        "backend": backend,
        "path": path,
        "code_map": dict(sorted(aliases.items())),
        "span": list(span) if span is not None else None,
        "line": line,
        "col": col,
        "revision": revision,
        "records": [dict(record) for record in records],
    }


def _context(value: object) -> dict[str, Any]:
    """Validate a context mapping and return a defensive copy."""
    raw = _mapping(value, "context")
    missing = [key for key in _CONTEXT_KEYS if key not in raw]
    if missing:
        _fail("E003", f"context is missing keys: {', '.join(missing)}")
    source = _text(raw["source"], "context.source")
    normalized = _span(raw["span"], length=len(source))
    return {
        "source": source,
        "phase": _canonical(
            raw["phase"], _PHASE_ALIASES, "context.phase", default="compile"
        ),
        "backend": _canonical(
            raw["backend"], _BACKEND_ALIASES, "context.backend", default="python"
        ),
        "path": raw["path"],
        "code_map": dict(_mapping(raw["code_map"], "context.code_map")),
        "span": list(normalized) if normalized is not None else None,
        "line": _integer(raw["line"], "context.line"),
        "col": _integer(raw["col"], "context.col"),
        "revision": _integer(raw["revision"], "context.revision"),
        "records": [dict(item) for item in raw["records"]],
    }


def _source_line_col(source: str, offset: int) -> tuple[int, int]:
    """One-based line and column for a clamped character offset."""
    position = offset_to_lsp_position(source, offset)
    return position["line"] + 1, position["character"] + 1


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-safe value with stable key and separator ordering."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _record(value: object) -> dict[str, Any]:
    """Validate a diagnostic record mapping and fill in safe defaults."""
    raw = _mapping(value, "record")
    if "code" not in raw or "message" not in raw:
        _fail("E003", "record requires code and message")
    code = _code(raw["code"])
    span = _span(raw.get("span"))
    return {
        "code": code,
        "message": _text(raw["message"], "record.message"),
        "severity": _severity(raw.get("severity", "error")),
        "span": list(span) if span is not None else None,
        "line": _integer(raw.get("line", 1), "record.line"),
        "col": _integer(raw.get("col", 1), "record.col"),
        "phase": _text(raw.get("phase", "compile"), "record.phase"),
        "backend": _text(raw.get("backend", "python"), "record.backend"),
        "path": raw.get("path"),
        "note": raw.get("note"),
        "help": raw.get("help"),
        "cause": raw.get("cause"),
    }


def dctx_new(
    source: str,
    phase: str = "compile",
    backend: str = "python",
    path: str | None = None,
    code_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a reusable, JSON-safe context for one source compilation unit."""
    text = _text(source, "source")
    selected_phase = _canonical(phase, _PHASE_ALIASES, "phase", default="compile")
    selected_backend = _canonical(
        backend, _BACKEND_ALIASES, "backend", default="python"
    )
    return _new_context(
        text,
        selected_phase,
        selected_backend,
        path,
        dict(code_map or {}),
        None,
    )


def dctx_with_source(context: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Return a copy of *context* bound to another source string."""
    existing = _context(context)
    text = _text(source, "source")
    previous_span = _span(existing["span"], length=len(existing["source"]))
    if previous_span is None:
        span: tuple[int, int] | None = None
        line, col = 1, 1
    else:
        span = _span(previous_span, length=len(text))
        line, col = _source_line_col(text, span[0])
    return _new_context(
        text,
        existing["phase"],
        existing["backend"],
        existing["path"],
        existing["code_map"],
        span,
        line=line,
        col=col,
        revision=existing["revision"] + 1,
        records=existing["records"],
    )


def dctx_attach_span(
    context: Mapping[str, Any], span: Sequence[int] | tuple[int, int] | None
) -> dict[str, Any]:
    """Attach a validated half-open span and refresh its one-based position."""
    existing = _context(context)
    normalized = _span(span, length=len(existing["source"]))
    line, col = (
        _source_line_col(existing["source"], normalized[0])
        if normalized is not None
        else (1, 1)
    )
    return _new_context(
        existing["source"],
        existing["phase"],
        existing["backend"],
        existing["path"],
        existing["code_map"],
        normalized,
        line=line,
        col=col,
        revision=existing["revision"] + 1,
        records=existing["records"],
    )


def dctx_with_code_alias(
    context: Mapping[str, Any], key: str, code: str
) -> dict[str, Any]:
    """Add a diagnostic alias to a context without mutating the original."""
    existing = _context(context)
    name = _text(key, "alias key").strip()
    if not name:
        _fail("E003", "alias key must not be empty")
    aliases = dict(existing["code_map"])
    aliases[name.lower()] = _code(code)
    return _new_context(
        existing["source"],
        existing["phase"],
        existing["backend"],
        existing["path"],
        aliases,
        _span(existing["span"], length=len(existing["source"])),
        line=existing["line"],
        col=existing["col"],
        revision=existing["revision"] + 1,
        records=existing["records"],
    )


def dctx_resolve_code(
    context: Mapping[str, Any], key: str, *, default: str | None = None
) -> dict[str, Any]:
    """Resolve a code or alias through a context into E001-E008."""
    existing = _context(context)
    name = _text(key, "code key").strip()
    aliases = existing["code_map"]
    mapped = aliases.get(name) or aliases.get(name.lower()) or aliases.get(name.upper())
    if mapped is None and name.upper() in ERROR_CODES:
        return {"key": name, "code": name.upper(), "origin": "direct", "valid": True}
    if mapped is None and default is None:
        _fail("E003", f"unresolved diagnostic alias {key!r}")
    resolved = _code(mapped if mapped is not None else default)
    return {
        "key": name,
        "code": resolved,
        "origin": "alias" if mapped is not None else "default",
        "valid": True,
    }


def dctx_line_count(source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Count source lines without creating an extra line for a trailing LF."""
    text = (
        _text(source["source"], "source")
        if isinstance(source, Mapping)
        else _text(source, "source")
    )
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "empty": not lines,
        "ends_with_newline": text.endswith(("\n", "\r")),
    }


def dctx_line_text(context: Mapping[str, Any], line: int) -> dict[str, Any]:
    """Return the text and width of a one-based source line."""
    existing = _context(context)
    number = _integer(line, "line")
    lines = existing["source"].splitlines()
    in_range = 1 <= number <= len(lines)
    text = lines[number - 1] if in_range else ""
    return {
        "line": number,
        "text": text,
        "length": len(text),
        "in_range": in_range,
    }


def dctx_offset_to_line_col(context: Mapping[str, Any], offset: int) -> dict[str, Any]:
    """Convert a character offset to one-based line/column and LSP position."""
    existing = _context(context)
    value = _integer(offset, "offset")
    clamped = max(0, min(value, len(existing["source"])))
    position = offset_to_lsp_position(existing["source"], clamped)
    return {
        "offset": clamped,
        "line": position["line"] + 1,
        "col": position["character"] + 1,
        "position": position,
        "clamped": clamped != value,
    }


def dctx_line_col_to_offset(
    context: Mapping[str, Any], line: int, col: int
) -> dict[str, Any]:
    """Convert a one-based line/column pair to a clamped character offset."""
    existing = _context(context)
    number = _integer(line, "line")
    column = _integer(col, "col")
    lines = existing["source"].splitlines(keepends=True)
    if number < 1 or not lines:
        offset = 0
    elif number > len(lines):
        offset = len(existing["source"])
    else:
        offset = sum(len(text) for text in lines[: number - 1]) + max(0, column - 1)
    clamped = max(0, min(offset, len(existing["source"])))
    return {
        "offset": clamped,
        "line": max(1, number),
        "col": max(1, column),
        "clamped": clamped != offset,
    }


def dctx_span_to_lsp(
    context: Mapping[str, Any],
    span: Sequence[int] | tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Convert a character span to the shared zero-based LSP range shape."""
    existing = _context(context)
    normalized = _span(
        existing["span"] if span is None else span, length=len(existing["source"])
    )
    selected = (
        normalized
        if normalized is not None
        else (existing["line"] - 1, existing["col"] - 1)
    )
    lsp_range = span_to_lsp_range(
        existing["source"],
        list(selected),
        line=existing["line"],
        col=existing["col"],
    )
    return {
        "span": list(selected),
        "range": lsp_range,
        "start_line": lsp_range["start"]["line"],
        "end_line": lsp_range["end"]["line"],
    }


def dctx_span_clamp(
    context: Mapping[str, Any], span: Sequence[int] | tuple[int, int]
) -> dict[str, Any]:
    """Clamp a span to the current source and report whether it changed."""
    existing = _context(context)
    original = _span(span)
    assert original is not None
    clamped = _span(original, length=len(existing["source"]))
    assert clamped is not None
    return {
        "span": list(clamped),
        "original": list(original),
        "clamped": clamped != original,
        "source_length": len(existing["source"]),
    }


def dctx_span_overlap(
    left: Sequence[int] | tuple[int, int], right: Sequence[int] | tuple[int, int]
) -> dict[str, Any]:
    """Classify two spans and report their intersection when present."""
    first = _span(left)
    second = _span(right)
    assert first is not None and second is not None
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    if first == second:
        relation = "equal"
    elif start >= end and (first[1] == second[0] or second[1] == first[0]):
        relation = "adjacent"
    elif start >= end:
        relation = "before" if first[1] <= second[0] else "after"
    elif first[0] <= second[0] and first[1] >= second[1]:
        relation = "contains"
    elif second[0] <= first[0] and second[1] >= first[1]:
        relation = "contained_by"
    else:
        relation = "overlap"
    return {
        "left": list(first),
        "right": list(second),
        "overlap": start < end,
        "intersection": [start, end] if start < end else None,
        "relation": relation,
    }


def dctx_span_union(spans: Sequence[Sequence[int] | tuple[int, int]]) -> dict[str, Any]:
    """Combine spans into their enclosing range and list internal gaps."""
    if not isinstance(spans, (list, tuple)) or not spans:
        _fail("E003", "spans must be a non-empty sequence")
    normalized = [_span(span) for span in spans]
    ordered = sorted(span for span in normalized if span is not None)
    assert ordered
    start = min(span[0] for span in ordered)
    end = max(span[1] for span in ordered)
    gaps: list[list[int]] = []
    cursor = ordered[0][1]
    for low, high in ordered[1:]:
        if low > cursor:
            gaps.append([cursor, low])
        cursor = max(cursor, high)
    return {
        "span": [start, end],
        "count": len(ordered),
        "gaps": gaps,
        "contiguous": not gaps,
    }


def dctx_normalize_severity(value: str) -> dict[str, Any]:
    """Normalize a severity alias and expose its ordering rank."""
    severity = _severity(value)
    return {
        "severity": severity,
        "rank": _SEVERITY_RANK[severity],
        "canonical": value.strip().lower() == severity,
    }


def dctx_make_record(
    context: Mapping[str, Any],
    code: str,
    message: str,
    *,
    severity: str = "error",
    span: Sequence[int] | tuple[int, int] | None = None,
    note: str | None = None,
    help_text: str | None = None,
    cause: int | str | None = None,
) -> dict[str, Any]:
    """Build a diagnostic record with context defaults and a stable fingerprint."""
    existing = _context(context)
    normalized = _span(span, length=len(existing["source"]))
    source_line, source_col = (
        _source_line_col(existing["source"], normalized[0])
        if normalized is not None
        else (existing["line"], existing["col"])
    )
    record = _record(
        {
            "code": _code(code, code_map=existing["code_map"]),
            "message": message,
            "severity": severity,
            "span": normalized,
            "line": source_line,
            "col": source_col,
            "phase": existing["phase"],
            "backend": existing["backend"],
            "path": existing["path"],
            "note": note,
            "help": help_text,
            "cause": cause,
        }
    )
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "code": record["code"],
                "message": record["message"],
                "span": record["span"],
            }
        ).encode("utf-8")
    ).hexdigest()
    record["fingerprint"] = fingerprint
    return record


def dctx_record_from_exception(
    context: Mapping[str, Any], error: BaseException, *, code: str | None = None
) -> dict[str, Any]:
    """Convert an exception's structured attributes into a diagnostic record."""
    existing = _context(context)
    if not isinstance(error, BaseException):
        _fail("E003", "error must be a BaseException")
    resolved = _code(code, code_map=existing["code_map"]) if code else _code(error)
    reason = getattr(error, "reason", None)
    message = (
        reason.strip()
        if isinstance(reason, str) and reason.strip()
        else (str(error).splitlines()[0] if str(error) else type(error).__name__)
    )
    line = getattr(error, "line", existing["line"])
    col = getattr(error, "col", existing["col"])
    return dctx_make_record(
        existing,
        resolved,
        message,
        severity=str(getattr(error, "severity", "error")),
        span=getattr(error, "span", None),
        note=getattr(error, "note", None),
        help_text=getattr(error, "help_text", None),
        cause=getattr(error, "cause", None),
    ) | {"line": _integer(line, "error.line"), "col": _integer(col, "error.col")}


def dctx_record_from_mapping(
    context: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a record from an editor/tooling mapping using context defaults."""
    existing = _context(context)
    raw = _mapping(payload, "payload")
    message = raw.get("message") or raw.get("reason") or raw.get("detail")
    if not isinstance(message, str) or not message.strip():
        _fail("E003", "payload requires a non-empty message")
    return dctx_make_record(
        existing,
        raw.get("code", "E003"),
        message,
        severity=str(raw.get("severity", "error")),
        span=raw.get("span"),
        note=raw.get("note"),
        help_text=raw.get("help") or raw.get("help_text"),
        cause=raw.get("cause") or raw.get("caused_by"),
    )


def dctx_record_position(
    context: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve a record's one-based position, preferring an explicit span."""
    existing = _context(context)
    normalized = _record(record)
    span = _span(normalized["span"], length=len(existing["source"]))
    if span is not None:
        line, col = _source_line_col(existing["source"], span[0])
        origin = "span"
    else:
        line, col = normalized["line"], normalized["col"]
        origin = "fields"
    return {
        "line": max(1, line),
        "col": max(1, col),
        "span": list(span) if span is not None else None,
        "origin": origin,
    }


def dctx_merge_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deduplicate records by identity and return them in stable code order."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    normalized = [_record(record) for record in records]
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in normalized:
        key = (
            record["code"],
            record["message"],
            tuple(record["span"]) if record["span"] else None,
            record["line"],
            record["col"],
        )
        seen.setdefault(key, record)
    merged = sorted(
        seen.values(),
        key=lambda item: (
            item["code"],
            item["line"],
            item["col"],
            item["message"],
        ),
    )
    return {
        "records": merged,
        "total": len(normalized),
        "unique": len(merged),
        "duplicates": len(normalized) - len(merged),
    }


def dctx_group_by_phase(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group normalized records by compiler phase with sorted phase names."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        normalized = _record(record)
        groups.setdefault(normalized["phase"], []).append(normalized)
    ordered = dict(sorted(groups.items()))
    return {
        "groups": ordered,
        "phases": list(ordered),
        "counts": {name: len(items) for name, items in ordered.items()},
        "total": sum(len(items) for items in ordered.values()),
    }


def dctx_filter_severity(
    records: Sequence[Mapping[str, Any]], minimum: str
) -> dict[str, Any]:
    """Keep records at or above a severity threshold, preserving input order."""
    threshold = _severity(minimum)
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    normalized = [_record(record) for record in records]
    kept = [
        record
        for record in normalized
        if _SEVERITY_RANK[record["severity"]] >= _SEVERITY_RANK[threshold]
    ]
    return {
        "records": kept,
        "minimum": threshold,
        "kept": len(kept),
        "removed": len(normalized) - len(kept),
    }


def dctx_code_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count every public diagnostic code plus the supplied record total."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    counts = {code: 0 for code in sorted(ERROR_CODES)}
    for record in records:
        counts[_code(_record(record)["code"])] += 1
    return {
        "counts": counts,
        "by_code": {code: value for code, value in counts.items() if value},
        "total": sum(counts.values()),
    }


def dctx_context_digest(context: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint a context with a deterministic SHA-256 of its JSON form."""
    existing = _context(context)
    payload = _canonical_json(existing).encode("utf-8")
    return {
        "digest": hashlib.sha256(payload).hexdigest(),
        "algorithm": "sha256",
        "bytes": len(payload),
        "schema": _CONTEXT_SCHEMA,
    }


def dctx_context_to_json(
    context: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    pretty: bool = False,
) -> dict[str, Any]:
    """Serialize a context and optional records to a stable JSON payload."""
    existing = _context(context)
    selected = (
        [_record(record) for record in records]
        if records is not None
        else [_record(record) for record in existing["records"]]
    )
    existing["records"] = selected
    payload = json.dumps(
        existing,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return {
        "payload": payload,
        "schema": _CONTEXT_SCHEMA,
        "record_count": len(selected),
        "bytes": len(payload.encode("utf-8")),
    }


def dctx_from_json(payload: str) -> dict[str, Any]:
    """Parse a serialized context payload, rejecting malformed state as E008."""
    text = _text(payload, "payload")
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as error:
        _fail("E008", f"malformed context payload: {error}")
    if not isinstance(parsed, dict):
        _fail("E008", "context payload must decode to an object")
    try:
        return _context(parsed)
    except PolyglotError as error:
        _fail("E008", f"malformed context payload: {error.diagnostic.message}")
