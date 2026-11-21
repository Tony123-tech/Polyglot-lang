"""Deterministic terminal, editor, and serialized diagnostic formatting.

Every formatter consumes plain diagnostic record mappings and returns a JSON-
safe ``dict`` with explicit keys.  Source text, styles, widths, and paths are
always supplied by the caller: the module never reads the environment, probes
the terminal, or consults the clock.  Invalid arguments raise
:class:`~compiler.frontend.errors.PolyglotError`; malformed serialized payloads
use ``E008`` and a missing source artifact uses ``E006``.
"""

from __future__ import annotations

import csv
import io
import json
import re
import textwrap
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NoReturn

from api.visualizer import span_to_lsp_range
from compiler.frontend.errors import PolyglotError

__all__ = [
    "dfmt_ansi",
    "dfmt_caret",
    "dfmt_code_frame",
    "dfmt_csv",
    "dfmt_escape_control",
    "dfmt_from_json",
    "dfmt_github_annotation",
    "dfmt_gutter",
    "dfmt_header",
    "dfmt_highlight",
    "dfmt_inline",
    "dfmt_json_diagnostic",
    "dfmt_json_payload",
    "dfmt_lsp_diagnostics",
    "dfmt_markdown",
    "dfmt_path_label",
    "dfmt_sarif_like",
    "dfmt_snippet",
    "dfmt_stack",
    "dfmt_strip_ansi",
    "dfmt_summary_line",
    "dfmt_table",
    "dfmt_terminal_width",
    "dfmt_truncate",
    "dfmt_wrap_message",
]

_FORMAT_SCHEMA: Final = "polyglot-diagnostics/v2"
_ANSI: Final[Mapping[str, str]] = MappingProxyType(
    {"error": "\x1b[31m", "warning": "\x1b[33m", "info": "\x1b[36m"}
)
_ANSI_RESET: Final = "\x1b[0m"
_ANSI_RE: Final[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")
_SEVERITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"warn": "warning", "fatal": "error", "critical": "error", "note": "info"}
)
_LSP_SEVERITY: Final[Mapping[str, int]] = MappingProxyType(
    {"error": 1, "warning": 2, "info": 3}
)
_SARIF_LEVEL: Final[Mapping[str, str]] = MappingProxyType(
    {"error": "error", "warning": "warning", "info": "note"}
)
_TMP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])(?:/private/var/folders|/var/folders|/tmp|"
    r"[A-Za-z]:[/\\]Temp)(?=[/\\]|$)",
    re.IGNORECASE,
)


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured formatting error for invalid input."""
    raise PolyglotError(code, message, phase="formatting")


def _text(value: object, label: str) -> str:
    """Validate a required string argument."""
    if not isinstance(value, str):
        _fail("E003", f"{label} must be a str")
    return value


def _integer(value: object, label: str) -> int:
    """Validate an integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{label} must be an int")
    return value


def _severity(value: object) -> str:
    """Normalize a severity alias for formatting."""
    text = _text(value, "severity").strip().lower()
    normalized = _SEVERITY_ALIASES.get(text, text)
    if normalized not in _LSP_SEVERITY:
        _fail("E003", f"unknown severity {value!r}")
    return normalized


def _span(value: object, *, length: int | None = None) -> tuple[int, int] | None:
    """Validate and optionally clamp a half-open character span."""
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        _fail("E003", "span must be a two-item pair of ints")
    start, end = max(0, int(value[0])), max(0, int(value[1]))
    if end < start:
        start, end = end, start
    if length is not None:
        start, end = min(start, length), min(end, length)
    return start, end


def _record(value: object) -> dict[str, Any]:
    """Normalize a record mapping, filling in deterministic defaults."""
    if not isinstance(value, Mapping):
        _fail("E003", "record must be a mapping")
    raw: Mapping[str, Any] = value
    if not isinstance(raw.get("code"), str) or not isinstance(raw.get("message"), str):
        _fail("E003", "record requires string code and message")
    span = _span(raw.get("span"))
    return {
        "code": raw["code"].strip().upper(),
        "message": raw["message"],
        "severity": _severity(raw.get("severity", "error")),
        "span": list(span) if span is not None else None,
        "line": max(1, _integer(raw.get("line", 1), "record.line")),
        "col": max(1, _integer(raw.get("col", 1), "record.col")),
        "phase": _text(raw.get("phase", "compile"), "record.phase"),
        "backend": _text(raw.get("backend", "python"), "record.backend"),
        "path": raw.get("path"),
        "note": raw.get("note"),
        "help": raw.get("help"),
        "cause": raw.get("cause"),
    }


def _source(value: object) -> str:
    """Require an explicit source artifact for source-based formatting."""
    if value is None:
        _fail("E006", "source artifact is required for this formatter")
    return _text(value, "source")


def _range_for(record: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Return the visualizer LSP range for a record's span or line/column."""
    return span_to_lsp_range(
        source,
        record.get("span"),
        line=record.get("line", 1),
        col=record.get("col", 1),
    )


def _prefix_pattern(prefix: str) -> re.Pattern[str]:
    """Build a portable textual matcher for a POSIX or Windows path prefix."""
    parts = [part for part in re.split(r"[/\\]", prefix.strip().rstrip("/\\")) if part]
    if not parts:
        raise ValueError("empty prefix")
    body = r"[/\\]".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![A-Za-z0-9_])(?:[/\\]?{body})(?=[/\\]|$)", re.IGNORECASE)


def dfmt_ansi(text: str, severity: str, *, enabled: bool = True) -> dict[str, Any]:
    """Wrap text in the severity's ANSI color code when color is enabled."""
    content = _text(text, "text")
    selected = _severity(severity)
    if not isinstance(enabled, bool):
        _fail("E003", "enabled must be a bool")
    if not enabled:
        return {"text": content, "severity": selected, "enabled": False}
    return {
        "text": f"{_ANSI[selected]}{content}{_ANSI_RESET}",
        "severity": selected,
        "enabled": True,
    }


def dfmt_strip_ansi(text: str) -> dict[str, Any]:
    """Remove SGR color escapes and report how many sequences were removed."""
    content = _text(text, "text")
    cleaned, count = _ANSI_RE.subn("", content)
    return {"text": cleaned, "removed": count}


def dfmt_caret(prefix: str, highlight: str, *, tab_width: int = 4) -> dict[str, Any]:
    """Build an aligned caret run for a highlighted source fragment."""
    before = _text(prefix, "prefix")
    marked = _text(highlight, "highlight")
    width = _integer(tab_width, "tab_width")
    if width < 1:
        _fail("E003", "tab_width must be at least 1")
    prefix_width = len(before.expandtabs(width))
    caret_width = max(1, len((before + marked).expandtabs(width)) - prefix_width)
    return {
        "caret": "^" * caret_width,
        "prefix_width": prefix_width,
        "highlight_width": caret_width,
        "tab_width": width,
    }


def dfmt_gutter(number: int, width: int | None = None) -> dict[str, Any]:
    """Format a right-aligned line-number gutter cell."""
    value = _integer(number, "number")
    size = _integer(width, "width") if width is not None else len(str(max(1, value)))
    if size < 1:
        _fail("E003", "width must be at least 1")
    return {"text": f"{value:>{size}}", "width": size, "number": value}


def dfmt_header(record: Mapping[str, Any], *, color: bool = False) -> dict[str, Any]:
    """Render a diagnostic's severity, code, and message as a header line."""
    normalized = _record(record)
    header = f"{normalized['severity']}[{normalized['code']}]: {normalized['message']}"
    rendered = dfmt_ansi(header, normalized["severity"], enabled=color)["text"]
    return {
        "text": rendered,
        "code": normalized["code"],
        "severity": normalized["severity"],
    }


def dfmt_inline(record: Mapping[str, Any]) -> dict[str, Any]:
    """Render a diagnostic as one compact ``path:line:col`` line."""
    normalized = _record(record)
    path = normalized["path"] or "<source>"
    text = (
        f"{path}:{normalized['line']}:{normalized['col']}: "
        f"{normalized['severity']}[{normalized['code']}]: {normalized['message']}"
    )
    return {
        "text": text,
        "path": path,
        "line": normalized["line"],
        "col": normalized["col"],
    }


def dfmt_snippet(
    record: Mapping[str, Any],
    source: str,
    *,
    color: bool = False,
    max_lines: int = 3,
    tab_width: int = 4,
) -> dict[str, Any]:
    """Render a source snippet with line numbers and a caret-annotated span."""
    text = _source(source)
    normalized = _record(record)
    limit = _integer(max_lines, "max_lines")
    if limit < 1:
        _fail("E003", "max_lines must be at least 1")
    tabs = _integer(tab_width, "tab_width")
    if tabs < 1:
        _fail("E003", "tab_width must be at least 1")
    lsp_range = _range_for(normalized, text)
    start_line = max(1, lsp_range["start"]["line"] + 1)
    start_col = max(1, lsp_range["start"]["character"] + 1)
    end_line = max(start_line, lsp_range["end"]["line"] + 1)
    end_col = max(1, lsp_range["end"]["character"] + 1)
    lines = text.splitlines()
    header = dfmt_header(normalized, color=color)["text"]
    path = normalized["path"] or "<source>"
    output = [header, f"  --> {path}:{start_line}:{start_col}"]
    if not lines:
        output.append("   |  <no source available>")
        return {
            "lines": output,
            "text": "\n".join(output),
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "suppressed": 0,
            "header": header,
        }
    first = start_line - 1
    last = min(end_line - 1, len(lines) - 1)
    shown = list(range(first, min(last, first + limit - 1) + 1))
    gutter = len(str(shown[-1] + 1))
    for index in shown:
        number = index + 1
        raw = lines[index]
        low = start_col - 1 if number == start_line else 0
        high = end_col - 1 if number == end_line else len(raw)
        if end_line > start_line and number == end_line and high <= 0:
            continue
        low = max(0, min(low, len(raw)))
        high = max(0, min(high, len(raw)))
        if high <= low:
            high = min(len(raw), low + 1)
        caret_info = dfmt_caret(raw[:low], raw[low:high], tab_width=tabs)
        caret = caret_info["caret"]
        if color:
            caret = dfmt_ansi(caret, normalized["severity"], enabled=True)["text"]
        output.append(f"{number:>{gutter}} | {raw.expandtabs(tabs)}")
        output.append(f"{' ' * gutter} | {' ' * caret_info['prefix_width']}{caret}")
    suppressed = last - shown[-1]
    if suppressed > 0:
        suffix = "line" if suppressed == 1 else "lines"
        output.append(f"{' ' * gutter} |  ... {suppressed} more {suffix} suppressed")
    output.append(f"{' ' * gutter} |")
    if normalized["help"]:
        output.append(f"  help: {normalized['help']}")
    if normalized["note"]:
        output.append(f"  note: {normalized['note']}")
    output.append(
        f"  context: phase={normalized['phase']} backend={normalized['backend']}"
    )
    return {
        "lines": output,
        "text": "\n".join(output),
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "suppressed": suppressed,
        "header": header,
    }


def dfmt_code_frame(
    record: Mapping[str, Any], source: str, *, context_lines: int = 1
) -> dict[str, Any]:
    """Render a numbered source window around a record's span."""
    text = _source(source)
    normalized = _record(record)
    padding = _integer(context_lines, "context_lines")
    if padding < 0:
        _fail("E003", "context_lines must not be negative")
    lsp_range = _range_for(normalized, text)
    first = max(0, lsp_range["start"]["line"] - padding)
    last = min(len(text.splitlines()) - 1, lsp_range["end"]["line"] + padding)
    focus = set(
        range(
            lsp_range["start"]["line"] + 1,
            lsp_range["end"]["line"] + 2,
        )
    )
    lines = text.splitlines()
    rendered: list[str] = []
    focus_lines: list[int] = []
    for index in range(first, last + 1):
        number = index + 1
        if number in focus:
            focus_lines.append(number)
        marker = ">>" if number in focus else "  "
        rendered.append(f"{marker} {number:>{len(str(last + 1))}} | {lines[index]}")
    return {
        "lines": rendered,
        "text": "\n".join(rendered),
        "start": first + 1,
        "end": last + 1,
        "focus_lines": focus_lines,
    }


def dfmt_highlight(
    source: str,
    span: Sequence[int] | tuple[int, int],
    *,
    marker: str = "*",
) -> dict[str, Any]:
    """Split source into labelled segments around one highlighted span."""
    text = _source(source)
    normalized = _span(span, length=len(text))
    assert normalized is not None
    mark = _text(marker, "marker")
    if not mark:
        _fail("E003", "marker must not be empty")
    start, end = normalized
    segments = [{"text": text[:start], "start": 0, "end": start, "kind": "before"}]
    segments.append({"text": text[start:end], "start": start, "end": end, "kind": mark})
    segments.append(
        {"text": text[end:], "start": end, "end": len(text), "kind": "after"}
    )
    return {
        "segments": segments,
        "marked": mark,
        "span": [start, end],
        "text": "".join(segment["text"] for segment in segments),
    }


def dfmt_wrap_message(
    message: str, *, width: int = 60, indent: str = ""
) -> dict[str, Any]:
    """Wrap a diagnostic message to a fixed width without environment probes."""
    content = _text(message, "message")
    size = _integer(width, "width")
    prefix = _text(indent, "indent")
    if size < 1:
        _fail("E003", "width must be at least 1")
    wrapped = textwrap.wrap(
        content,
        width=size,
        initial_indent=prefix,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix]
    return {
        "lines": wrapped,
        "text": "\n".join(wrapped),
        "width": size,
        "original_length": len(content),
    }


def dfmt_truncate(text: str, width: int, *, suffix: str = "…") -> dict[str, Any]:
    """Truncate text to a display width with an explicit suffix."""
    content = _text(text, "text")
    size = _integer(width, "width")
    ending = _text(suffix, "suffix")
    if size < 1:
        _fail("E003", "width must be at least 1")
    if len(content) <= size:
        return {
            "text": content,
            "truncated": False,
            "original_length": len(content),
            "width": size,
        }
    room = max(0, size - len(ending))
    shortened = content[:room] + ending
    return {
        "text": shortened,
        "truncated": True,
        "original_length": len(content),
        "width": size,
    }


def dfmt_terminal_width(
    lines: Sequence[str], *, cap: int = 120, tab_width: int = 4
) -> dict[str, Any]:
    """Compute the widest supplied line under a caller-provided cap."""
    if not isinstance(lines, (list, tuple)):
        _fail("E003", "lines must be a sequence")
    limit = _integer(cap, "cap")
    tabs = _integer(tab_width, "tab_width")
    if limit < 1 or tabs < 1:
        _fail("E003", "cap and tab_width must be at least 1")
    widths = [len(_text(line, "line").expandtabs(tabs)) for line in lines]
    widest = max(widths, default=0)
    return {
        "width": min(widest, limit),
        "raw_width": widest,
        "line_count": len(lines),
        "capped": widest > limit,
    }


def dfmt_escape_control(text: str) -> dict[str, Any]:
    """Replace control characters with visible escape spellings."""
    content = _text(text, "text")
    replacements = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\x1b": "\\e"}
    escaped = "".join(replacements.get(char, char) for char in content)
    return {
        "text": escaped,
        "escaped": sum(content.count(char) for char in replacements),
    }


def dfmt_json_diagnostic(record: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize one normalized record to a stable JSON object string."""
    normalized = _record(record)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return {
        "payload": payload,
        "keys": sorted(normalized),
        "bytes": len(payload.encode("utf-8")),
    }


def dfmt_json_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    pretty: bool = False,
    schema: str = _FORMAT_SCHEMA,
) -> dict[str, Any]:
    """Serialize many records to a versioned JSON payload."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    if not isinstance(pretty, bool):
        _fail("E003", "pretty must be a bool")
    normalized = [_record(record) for record in records]
    root = {
        "schema": _text(schema, "schema"),
        "count": len(normalized),
        "diagnostics": normalized,
    }
    payload = json.dumps(
        root,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return {
        "payload": payload,
        "schema": root["schema"],
        "count": len(normalized),
        "bytes": len(payload.encode("utf-8")),
    }


def dfmt_from_json(payload: str) -> dict[str, Any]:
    """Parse a versioned diagnostics payload, rejecting malformed state."""
    text = _text(payload, "payload")
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as error:
        _fail("E008", f"malformed diagnostics payload: {error}")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("diagnostics"), list):
        _fail("E008", "payload must be an object with a diagnostics array")
    try:
        records = [_record(item) for item in parsed["diagnostics"]]
    except PolyglotError as error:
        _fail("E008", f"malformed diagnostic record: {error.diagnostic.message}")
    return {
        "schema": parsed.get("schema", _FORMAT_SCHEMA),
        "records": records,
        "count": len(records),
    }


def dfmt_csv(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render records as RFC-4180 CSV with a fixed column order."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    header = ["code", "severity", "line", "col", "path", "message"]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    normalized = [_record(record) for record in records]
    for record in normalized:
        writer.writerow(
            [
                record["code"],
                record["severity"],
                record["line"],
                record["col"],
                record["path"] or "",
                record["message"],
            ]
        )
    return {
        "text": stream.getvalue(),
        "header": header,
        "rows": len(normalized),
    }


def dfmt_markdown(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render records as a markdown table with escaped pipe characters."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    header = ["Code", "Severity", "Location", "Message"]
    rows: list[list[str]] = []
    for record in records:
        normalized = _record(record)
        location = f"{normalized['path'] or '<source>'}:{normalized['line']}:{normalized['col']}"
        rows.append(
            [
                normalized["code"],
                normalized["severity"],
                location.replace("|", "\\|"),
                normalized["message"].replace("|", "\\|"),
            ]
        )
    table = dfmt_table(header, rows)["text"]
    return {"text": table, "header": header, "rows": len(rows)}


def dfmt_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    align: str = "left",
) -> dict[str, Any]:
    """Build a plain-text table with deterministic column padding."""
    if not isinstance(headers, (list, tuple)) or not isinstance(rows, (list, tuple)):
        _fail("E003", "headers and rows must be sequences")
    alignment = _text(align, "align").lower()
    if alignment not in {"left", "right"}:
        _fail("E003", "align must be left or right")
    names = [_text(item, "header") for item in headers]
    cells = [[_text(str(item), "cell") for item in row] for row in rows]
    widths = [len(name) for name in names]
    for row in cells:
        if len(row) != len(names):
            _fail("E003", "row width does not match the header")
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines: list[str] = []
    for row in [names, *cells]:
        parts = [
            value.rjust(widths[index])
            if alignment == "right"
            else value.ljust(widths[index])
            for index, value in enumerate(row)
        ]
        lines.append("| " + " | ".join(parts) + " |")
    lines.insert(1, "| " + " | ".join("-" * size for size in widths) + " |")
    return {
        "text": "\n".join(lines),
        "widths": widths,
        "rows": len(cells),
        "align": alignment,
    }


def dfmt_lsp_diagnostics(
    records: Sequence[Mapping[str, Any]], source: str
) -> dict[str, Any]:
    """Convert records to the Language Server Protocol diagnostic shape."""
    text = _source(source)
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    converted: list[dict[str, Any]] = []
    for record in records:
        normalized = _record(record)
        lsp_range = _range_for(normalized, text)
        converted.append(
            {
                "range": lsp_range,
                "severity": _LSP_SEVERITY[normalized["severity"]],
                "code": normalized["code"],
                "message": normalized["message"],
                "source": normalized["backend"],
            }
        )
    return {"diagnostics": converted, "count": len(converted)}


def dfmt_sarif_like(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a compact SARIF-like result payload from diagnostic records."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    results: list[dict[str, Any]] = []
    for record in records:
        normalized = _record(record)
        results.append(
            {
                "ruleId": normalized["code"],
                "level": _SARIF_LEVEL[normalized["severity"]],
                "message": {"text": normalized["message"]},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": normalized["path"] or "<source>"
                            },
                            "region": {
                                "startLine": normalized["line"],
                                "startColumn": normalized["col"],
                            },
                        }
                    }
                ],
            }
        )
    return {
        "runs": [{"tool": "polyglot", "results": results}],
        "count": len(results),
    }


def dfmt_github_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Render a GitHub Actions workflow annotation line for one record."""
    normalized = _record(record)
    level = normalized["severity"]
    message = (
        normalized["message"]
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )
    path = (normalized["path"] or "<source>").replace("%", "%25")
    text = (
        f"::{level} file={path},line={normalized['line']},"
        f"col={normalized['col']},title={normalized['code']}::{message}"
    )
    return {"text": text, "level": level, "path": path}


def dfmt_stack(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render a causal diagnostic chain as an indented arrow tree."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    normalized = [_record(record) for record in records]
    lines = [
        f"{'  ' * index}-> [{item['code']}] {item['message']}"
        for index, item in enumerate(normalized)
    ]
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "depth": len(lines),
        "roots": sum(1 for item in normalized if item["cause"] is None),
    }


def dfmt_summary_line(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize records by severity and code on one deterministic line."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    normalized = [_record(record) for record in records]
    severities = {name: 0 for name in sorted(_LSP_SEVERITY)}
    codes: dict[str, int] = {}
    for record in normalized:
        severities[record["severity"]] += 1
        codes[record["code"]] = codes.get(record["code"], 0) + 1
    counts = ", ".join(f"{name}={value}" for name, value in severities.items())
    text = f"diagnostics: total={len(normalized)} {counts}"
    return {
        "text": text,
        "total": len(normalized),
        "severities": severities,
        "codes": dict(sorted(codes.items())),
    }


def dfmt_path_label(
    path: str,
    *,
    home: str | None = None,
    workspace: str | None = None,
    home_label: str = "~",
    workspace_label: str = "<workspace>",
    tmp_label: str = "<tmp>",
) -> dict[str, Any]:
    """Redact explicitly supplied path prefixes with portable text matching."""
    content = _text(path, "path")
    patterns: list[tuple[re.Pattern[str], str, str]] = []
    for prefix, label, name in (
        (workspace, workspace_label, "workspace"),
        (home, home_label, "home"),
    ):
        if prefix is None:
            continue
        try:
            pattern = _prefix_pattern(_text(prefix, name))
        except ValueError:
            _fail("E003", f"{name} prefix must contain a path segment")
        patterns.append((pattern, _text(label, f"{name}_label"), name))
    patterns.append((_TMP_PATTERN, _text(tmp_label, "tmp_label"), "tmp"))
    matched: list[str] = []
    for pattern, label, name in patterns:
        content, count = pattern.subn(label, content)
        if count:
            matched.extend([name] * count)
    return {
        "text": content,
        "redacted": bool(matched),
        "matched": matched,
    }
