"""Deterministic visualizer helpers for diagnostic sources and trees.

A view is a plain, JSON-safe ``dict`` holding source text, explicit annotated
spans, and a revision counter.  Span positions use :mod:`api.visualizer`
helpers so editor ranges match the existing LSP surface.  Tree, diff, minimap,
and timeline helpers operate only on values supplied by the caller: no HTTP,
server call, environment probe, or clock read occurs here.  Invalid inputs
raise :class:`~compiler.frontend.errors.PolyglotError`, with ``E008`` reserved
for malformed serialized views or trees.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NoReturn

from api.visualizer import offset_to_lsp_position, span_to_lsp_range
from compiler.frontend.errors import PolyglotError

__all__ = [
    "dviz_add_span",
    "dviz_annotate",
    "dviz_color_for_category",
    "dviz_diff_lines",
    "dviz_diff_spans",
    "dviz_highlight_line",
    "dviz_legend",
    "dviz_line_map",
    "dviz_minimap",
    "dviz_new_view",
    "dviz_node",
    "dviz_offset_position",
    "dviz_source_gutter",
    "dviz_span_range",
    "dviz_timeline",
    "dviz_tree_count",
    "dviz_tree_find",
    "dviz_tree_flatten",
    "dviz_tree_from_json",
    "dviz_tree_from_records",
    "dviz_tree_prune",
    "dviz_tree_to_json",
    "dviz_view_digest",
    "dviz_view_report",
    "dviz_with_source",
]

_VIEW_SCHEMA: Final = "polyglot-visualizer-view/v2"
_TREE_SCHEMA: Final = "polyglot-visualizer-tree/v2"
_VIEW_KEYS: Final[tuple[str, ...]] = (
    "schema",
    "source",
    "title",
    "phase",
    "spans",
    "revision",
)
_COLORS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "error": ("red", "\x1b[31m"),
        "warning": ("amber", "\x1b[33m"),
        "info": ("cyan", "\x1b[36m"),
        "item": ("blue", "\x1b[34m"),
        "expression": ("green", "\x1b[32m"),
        "truncated": ("gray", "\x1b[90m"),
    }
)


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured visualization error for invalid input."""
    raise PolyglotError(code, message, phase="visualizer")


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


def _finite(value: object, label: str) -> float:
    """Validate a finite numeric coordinate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E003", f"{label} must be a finite number")
    result = float(value)
    if result < 0 or not math.isfinite(result):
        _fail("E003", f"{label} must be a non-negative finite number")
    return result


def _span(value: object, *, length: int | None = None) -> tuple[int, int] | None:
    """Validate and clamp a half-open character span."""
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


def _view(value: object) -> dict[str, Any]:
    """Validate a view mapping and return a defensive copy."""
    if not isinstance(value, Mapping):
        _fail("E003", "view must be a mapping")
    raw: Mapping[str, Any] = value
    missing = [key for key in _VIEW_KEYS if key not in raw]
    if missing:
        _fail("E003", f"view is missing keys: {', '.join(missing)}")
    source = _text(raw["source"], "view.source")
    if not isinstance(raw["spans"], (list, tuple)):
        _fail("E003", "view.spans must be a sequence")
    spans = []
    for item in raw["spans"]:
        if not isinstance(item, Mapping):
            _fail("E003", "each span must be a mapping")
        normalized = _span(item.get("span"), length=len(source))
        if normalized is None:
            _fail("E003", "each span requires a valid span")
        spans.append({**dict(item), "span": list(normalized)})
    return {
        "schema": _text(raw["schema"], "view.schema"),
        "source": source,
        "title": _text(raw["title"], "view.title"),
        "phase": _text(raw["phase"], "view.phase"),
        "spans": spans,
        "revision": _integer(raw["revision"], "view.revision"),
    }


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-safe value with stable key and separator ordering."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _node(
    label: object,
    *,
    category: str = "item",
    children: Sequence[Mapping[str, Any]] = (),
    props: Mapping[str, Any] | None = None,
    node_id: str = "n1",
) -> dict[str, Any]:
    """Build a validated visualizer tree node."""
    if not isinstance(children, (list, tuple)):
        _fail("E003", "children must be a sequence")
    return {
        "id": _text(node_id, "node_id"),
        "label": _text(label, "label"),
        "category": _text(category, "category"),
        "props": dict(props or {}),
        "children": [dict(child) for child in children],
    }


def _valid_node(value: object) -> dict[str, Any]:
    """Validate a tree node mapping recursively, preserving its shape."""
    if not isinstance(value, Mapping):
        _fail("E003", "tree node must be a mapping")
    raw: Mapping[str, Any] = value
    for key in ("id", "label", "category", "props", "children"):
        if key not in raw:
            _fail("E003", f"tree node is missing {key!r}")
    if not isinstance(raw["props"], Mapping) or not isinstance(
        raw["children"], (list, tuple)
    ):
        _fail("E003", "tree node props and children have invalid types")
    return _node(
        raw["label"],
        category=str(raw["category"]),
        children=[_valid_node(child) for child in raw["children"]],
        props=raw["props"],
        node_id=str(raw["id"]),
    )


def dviz_new_view(
    source: str, *, title: str = "source", phase: str = "compile"
) -> dict[str, Any]:
    """Create an empty visualizer view bound to explicit source text."""
    return {
        "schema": _VIEW_SCHEMA,
        "source": _text(source, "source"),
        "title": _text(title, "title"),
        "phase": _text(phase, "phase"),
        "spans": [],
        "revision": 0,
    }


def dviz_with_source(view: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Return a copy of a view bound to new source, clamping existing spans."""
    existing = _view(view)
    text = _text(source, "source")
    existing["source"] = text
    existing["spans"] = [
        {**span, "span": list(_span(span["span"], length=len(text)) or (0, 0))}
        for span in existing["spans"]
    ]
    existing["revision"] += 1
    return existing


def dviz_add_span(
    view: Mapping[str, Any],
    span: Sequence[int] | tuple[int, int],
    label: str,
    *,
    category: str = "info",
    span_id: str | None = None,
) -> dict[str, Any]:
    """Annotate a view with a labeled span and its resolved LSP range."""
    existing = _view(view)
    normalized = _span(span, length=len(existing["source"]))
    assert normalized is not None
    identifier = (
        _text(span_id, "span_id")
        if span_id is not None
        else f"s{len(existing['spans']) + 1}"
    )
    position = offset_to_lsp_position(existing["source"], normalized[0])
    lsp_range = span_to_lsp_range(existing["source"], list(normalized))
    record = {
        "id": identifier,
        "span": list(normalized),
        "label": _text(label, "label"),
        "category": _text(category, "category"),
        "line": position["line"] + 1,
        "col": position["character"] + 1,
        "range": lsp_range,
    }
    existing["spans"].append(record)
    existing["revision"] += 1
    return {
        "view": existing,
        "span": record,
        "count": len(existing["spans"]),
        "revision": existing["revision"],
    }


def dviz_span_range(
    view: Mapping[str, Any],
    span: Sequence[int] | tuple[int, int] | None = None,
    *,
    line: int | None = None,
    col: int | None = None,
) -> dict[str, Any]:
    """Convert a view span to the shared zero-based LSP range shape."""
    existing = _view(view)
    normalized = _span(span, length=len(existing["source"]))
    result = span_to_lsp_range(
        existing["source"],
        list(normalized) if normalized is not None else None,
        line=line,
        col=col,
    )
    return {
        "range": result,
        "span": list(normalized) if normalized is not None else None,
        "source_length": len(existing["source"]),
    }


def dviz_offset_position(view: Mapping[str, Any], offset: int) -> dict[str, Any]:
    """Convert a character offset to a zero-based LSP position in a view."""
    existing = _view(view)
    value = _integer(offset, "offset")
    clamped = max(0, min(value, len(existing["source"])))
    return {
        "offset": clamped,
        "position": offset_to_lsp_position(existing["source"], clamped),
        "clamped": clamped != value,
    }


def dviz_line_map(view: Mapping[str, Any]) -> dict[str, Any]:
    """Map each source line to its one-based number and character offsets."""
    existing = _view(view)
    lines: list[dict[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(existing["source"].splitlines(keepends=True), 1):
        text = raw.rstrip("\r\n")
        lines.append(
            {
                "line": index,
                "text": text,
                "start": cursor,
                "end": cursor + len(text),
                "length": len(text),
            }
        )
        cursor += len(raw)
    return {
        "lines": lines,
        "count": len(lines),
        "source_length": len(existing["source"]),
    }


def dviz_source_gutter(
    view: Mapping[str, Any],
    *,
    first: int = 1,
    last: int | None = None,
    tab_width: int = 4,
) -> dict[str, Any]:
    """Render numbered source lines for a caller-selected line window."""
    existing = _view(view)
    lines = dviz_line_map(existing)["lines"]
    start = max(1, _integer(first, "first"))
    end = _integer(last, "last") if last is not None else len(lines)
    tabs = _integer(tab_width, "tab_width")
    if tabs < 1:
        _fail("E003", "tab_width must be at least 1")
    selected = [line for line in lines if start <= line["line"] <= end]
    width = len(str(max([end, start], default=1)))
    rendered = [
        f"{line['line']:>{width}} | {line['text'].expandtabs(tabs)}"
        for line in selected
    ]
    return {
        "lines": rendered,
        "text": "\n".join(rendered),
        "first": start,
        "last": end,
        "line_count": len(rendered),
    }


def dviz_highlight_line(
    view: Mapping[str, Any], line: int, *, marker: str = "^", tab_width: int = 4
) -> dict[str, Any]:
    """Build a caret line for all annotated spans touching one source line."""
    existing = _view(view)
    number = _integer(line, "line")
    mark = _text(marker, "marker")
    tabs = _integer(tab_width, "tab_width")
    if tabs < 1 or not mark:
        _fail("E003", "tab_width must be positive and marker non-empty")
    lines = dviz_line_map(existing)["lines"]
    if not 1 <= number <= len(lines):
        _fail("E003", f"line {number} is outside the view source")
    target = lines[number - 1]
    columns: set[int] = set()
    span_ids: list[str] = []
    for span in existing["spans"]:
        start, end = span["span"]
        if start > target["end"] or end < target["start"]:
            continue
        low = max(start, target["start"]) - target["start"]
        high = min(end, target["end"]) - target["start"]
        columns.update(range(low, max(low + 1, high)))
        span_ids.append(span["id"])
    marker_chars = [" "] * target["length"]
    for column in columns:
        if column < len(marker_chars):
            marker_chars[column] = mark
    marker_line = "".join(marker_chars).expandtabs(tabs)
    return {
        "line": number,
        "text": target["text"],
        "marker_line": marker_line,
        "columns": sorted(columns),
        "span_ids": span_ids,
    }


def dviz_annotate(view: Mapping[str, Any]) -> dict[str, Any]:
    """Append inline labels from overlapping spans to each source line."""
    existing = _view(view)
    annotated: list[str] = []
    record: list[dict[str, Any]] = []
    for line in dviz_line_map(existing)["lines"]:
        labels = [
            span["label"]
            for span in existing["spans"]
            if span["span"][0] <= line["end"] and span["span"][1] >= line["start"]
        ]
        text = line["text"]
        if labels:
            text += "  ; " + ", ".join(labels)
        annotated.append(text)
        record.append({"line": line["line"], "labels": labels, "text": text})
    return {
        "lines": annotated,
        "text": "\n".join(annotated),
        "annotations": record,
        "span_count": len(existing["spans"]),
    }


def dviz_node(
    label: str,
    *,
    category: str = "item",
    children: Sequence[Mapping[str, Any]] = (),
    props: Mapping[str, Any] | None = None,
    node_id: str = "n1",
) -> dict[str, Any]:
    """Create a JSON-safe tree node with stable field ordering."""
    return _node(
        label, category=category, children=children, props=props, node_id=node_id
    )


def dviz_tree_from_records(
    records: Sequence[Mapping[str, Any]], *, root_label: str = "diagnostics"
) -> dict[str, Any]:
    """Group diagnostic records into a phase/severity/record tree."""
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    counter = [0]

    def make(
        label: str, category: str, children: list[dict[str, Any]]
    ) -> dict[str, Any]:
        counter[0] += 1
        return _node(
            label, category=category, children=children, node_id=f"n{counter[0]}"
        )

    leaves: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            _fail("E003", "each record must be a mapping")
        leaves.append(
            {
                "code": str(record.get("code", "E003")),
                "message": str(record.get("message", "diagnostic")),
                "severity": str(record.get("severity", "error")),
                "phase": str(record.get("phase", "compile")),
            }
        )
    severity_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for leaf in sorted(
        leaves,
        key=lambda item: (
            item["phase"],
            item["severity"],
            item["code"],
            item["message"],
        ),
    ):
        key = (leaf["phase"], leaf["severity"])
        node = severity_nodes.setdefault(
            key, make(leaf["severity"], leaf["severity"], [])
        )
        node["children"].append(
            make(f"[{leaf['code']}] {leaf['message']}", leaf["severity"], [])
        )
    phase_nodes: dict[str, dict[str, Any]] = {}
    for (phase, _severity), node in sorted(severity_nodes.items()):
        parent = phase_nodes.setdefault(phase, make(phase, "phase", []))
        parent["children"].append(node)
    root = make(root_label, "root", [node for _, node in sorted(phase_nodes.items())])
    count = dviz_tree_count(root)
    return {"tree": root, "count": count["nodes"], "depth": count["depth"]}


def dviz_tree_count(node: Mapping[str, Any]) -> dict[str, Any]:
    """Count nodes, leaves, and maximum depth of a visualizer tree."""
    valid = _valid_node(node)

    def walk(current: Mapping[str, Any], depth: int) -> tuple[int, int, int]:
        children = current["children"]
        if not children:
            return 1, 1, depth
        totals = [walk(child, depth + 1) for child in children]
        return (
            1 + sum(item[0] for item in totals),
            sum(item[1] for item in totals),
            max(item[2] for item in totals),
        )

    nodes, leaves, depth = walk(valid, 0)
    return {"nodes": nodes, "leaves": leaves, "depth": depth}


def dviz_tree_find(node: Mapping[str, Any], key: str, value: object) -> dict[str, Any]:
    """Find tree nodes whose named field equals an explicit value."""
    valid = _valid_node(node)
    field = _text(key, "key")
    matches: list[dict[str, Any]] = []

    def walk(current: Mapping[str, Any], path: str) -> None:
        if current.get(field) == value:
            matches.append({"path": path, "node": dict(current)})
        for index, child in enumerate(current["children"]):
            walk(child, f"{path}/{index}")

    walk(valid, "/")
    return {"matches": matches, "count": len(matches), "key": field}


def dviz_tree_flatten(node: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a tree into deterministic path-ordered node summaries."""
    valid = _valid_node(node)
    flattened: list[dict[str, Any]] = []

    def walk(current: Mapping[str, Any], path: str, depth: int) -> None:
        flattened.append(
            {
                "id": current["id"],
                "path": path,
                "depth": depth,
                "label": current["label"],
                "category": current["category"],
            }
        )
        for index, child in enumerate(current["children"]):
            walk(child, f"{path}/{index}", depth + 1)

    walk(valid, "/", 0)
    return {"nodes": flattened, "count": len(flattened)}


def dviz_tree_prune(
    node: Mapping[str, Any], *, max_depth: int = 3, max_nodes: int = 20
) -> dict[str, Any]:
    """Prune a tree by depth and node budget while marking hidden children."""
    valid = _valid_node(node)
    depth_limit = _nonnegative_int(max_depth, "max_depth")
    budget = _nonnegative_int(max_nodes, "max_nodes")
    counter = [0]

    def hide(count: int) -> dict[str, Any]:
        counter[0] += 1
        return _node(
            f"... {count} children hidden",
            category="truncated",
            node_id=f"n{counter[0]}",
        )

    def walk(current: Mapping[str, Any], depth: int) -> dict[str, Any]:
        counter[0] += 1
        node_id = counter[0]
        if node_id > budget:
            return hide(1)
        if depth >= depth_limit and current["children"]:
            return hide(len(current["children"]))
        children = [walk(child, depth + 1) for child in current["children"]]
        return _node(
            current["label"],
            category=current["category"],
            children=children,
            props=current["props"],
            node_id=f"n{node_id}",
        )

    pruned = walk(valid, 0)
    total = dviz_tree_count(valid)["nodes"]
    kept = dviz_tree_count(pruned)["nodes"]
    return {"tree": pruned, "kept": kept, "dropped": total - kept}


def _nonnegative_int(value: object, label: str) -> int:
    """Validate a non-negative integer for pruning limits."""
    number = _integer(value, label)
    if number < 0:
        _fail("E003", f"{label} must not be negative")
    return number


def dviz_tree_to_json(
    tree: Mapping[str, Any], *, pretty: bool = False
) -> dict[str, Any]:
    """Serialize a visualizer tree with stable key ordering."""
    valid = _valid_node(tree)
    if not isinstance(pretty, bool):
        _fail("E003", "pretty must be a bool")
    payload = json.dumps(
        valid,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return {
        "payload": payload,
        "bytes": len(payload.encode("utf-8")),
        "schema": _TREE_SCHEMA,
    }


def dviz_tree_from_json(payload: str) -> dict[str, Any]:
    """Parse a serialized visualizer tree, rejecting malformed state as E008."""
    text = _text(payload, "payload")
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as error:
        _fail("E008", f"malformed tree payload: {error}")
    try:
        return _valid_node(parsed)
    except PolyglotError as error:
        _fail("E008", f"malformed tree payload: {error.diagnostic.message}")


def dviz_color_for_category(category: str) -> dict[str, Any]:
    """Map a tree or span category to a deterministic color token."""
    name = _text(category, "category")
    color, ansi = _COLORS.get(name, ("neutral", "\x1b[37m"))
    return {"category": name, "color": color, "ansi": ansi, "known": name in _COLORS}


def dviz_legend(categories: Sequence[str]) -> dict[str, Any]:
    """Build a sorted color legend for the supplied categories."""
    if not isinstance(categories, (list, tuple)):
        _fail("E003", "categories must be a sequence")
    entries = [
        {**dviz_color_for_category(name), "label": name}
        for name in sorted({_text(item, "category") for item in categories})
    ]
    return {"entries": entries, "count": len(entries)}


def dviz_timeline(
    spans: Sequence[Mapping[str, Any]], *, width: int = 40, scale: float = 1.0
) -> dict[str, Any]:
    """Render explicit trace spans as aligned ASCII timeline bars."""
    if not isinstance(spans, (list, tuple)):
        _fail("E003", "spans must be a sequence")
    columns = _nonnegative_int(width, "width")
    if columns < 1:
        _fail("E003", "width must be at least 1")
    factor = _finite(scale, "scale")
    if factor <= 0:
        _fail("E003", "scale must be positive")
    prepared: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            _fail("E003", "each timeline span must be a mapping")
        start = _finite(span.get("start"), "span.start")
        end = _finite(span.get("end"), "span.end")
        if end < start:
            _fail("E003", "span end must not precede start")
        prepared.append(
            {
                "label": _text(str(span.get("label", "span")), "span.label"),
                "start": start,
                "end": end,
            }
        )
    total = max((span["end"] for span in prepared), default=0.0)
    lines: list[str] = []
    for span in prepared:
        first = min(columns - 1, int(span["start"] / max(total, 1e-9) * columns))
        size = max(1, int((span["end"] - span["start"]) * factor))
        size = min(size, columns - first)
        bar = "." * first + "#" * size + "." * (columns - first - size)
        lines.append(f"{span['label']:<12} | {bar}")
    return {"lines": lines, "text": "\n".join(lines), "width": columns, "total": total}


def dviz_diff_lines(before: str, after: str, *, context: int = 1) -> dict[str, Any]:
    """Produce a deterministic unified line diff of two explicit sources."""
    first = _text(before, "before")
    second = _text(after, "after")
    padding = _nonnegative_int(context, "context")
    lines = list(
        difflib.unified_diff(
            first.splitlines(),
            second.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=padding,
        )
    )
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "added": sum(
            1 for line in lines if line.startswith("+") and not line.startswith("+++")
        ),
        "removed": sum(
            1 for line in lines if line.startswith("-") and not line.startswith("---")
        ),
        "hunks": sum(1 for line in lines if line.startswith("@@")),
    }


def dviz_diff_spans(before: str, after: str) -> dict[str, Any]:
    """Locate changed character ranges using explicit source strings."""
    first = _text(before, "before")
    second = _text(after, "after")
    matcher = difflib.SequenceMatcher(None, first, second, autojunk=False)
    spans: list[dict[str, Any]] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag != "equal":
            spans.append(
                {
                    "tag": tag,
                    "start": old_start,
                    "end": old_end,
                    "replacement": second[new_start:new_end],
                    "text": first[old_start:old_end],
                }
            )
    return {
        "spans": spans,
        "changed": sum(item["end"] - item["start"] for item in spans),
        "ratio": matcher.ratio(),
    }


def dviz_minimap(view: Mapping[str, Any], *, columns: int = 40) -> dict[str, Any]:
    """Render annotated spans as a coarse per-line character grid."""
    existing = _view(view)
    size = _nonnegative_int(columns, "columns")
    if size < 1:
        _fail("E003", "columns must be at least 1")
    length = max(1, len(existing["source"]))
    rows: list[dict[str, Any]] = []
    for line in dviz_line_map(existing)["lines"]:
        cells = ["."] * size
        for span in existing["spans"]:
            start, end = span["span"]
            if start > line["end"] or end < line["start"]:
                continue
            low = max(start, line["start"]) / length * size
            high = min(end, line["end"]) / length * size
            first = max(0, min(size - 1, int(low)))
            last = max(first + 1, min(size, int(high) + 1))
            for column in range(first, last):
                cells[column] = "#"
        rows.append({"line": line["line"], "text": "".join(cells)})
    return {"rows": rows, "columns": size, "line_count": len(rows)}


def dviz_view_digest(view: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint a view with SHA-256 over its canonical JSON form."""
    existing = _view(view)
    payload = _canonical_json(existing).encode("utf-8")
    return {
        "digest": hashlib.sha256(payload).hexdigest(),
        "algorithm": "sha256",
        "bytes": len(payload),
        "schema": _VIEW_SCHEMA,
    }


def dviz_view_report(
    view: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] = (),
    *,
    include_diff: str | None = None,
) -> dict[str, Any]:
    """Assemble a complete structured report from one explicit view."""
    existing = _view(view)
    if not isinstance(records, (list, tuple)):
        _fail("E003", "records must be a sequence")
    report: dict[str, Any] = {
        "schema": "polyglot-visualizer-report/v2",
        "title": existing["title"],
        "phase": existing["phase"],
        "revision": existing["revision"],
        "gutter": dviz_source_gutter(existing),
        "annotations": dviz_annotate(existing),
        "minimap": dviz_minimap(existing),
        "digest": dviz_view_digest(existing),
        "span_count": len(existing["spans"]),
        "tree": dviz_tree_from_records(records) if records else None,
    }
    if include_diff is not None:
        report["diff"] = dviz_diff_lines(existing["source"], include_diff)
    return report
