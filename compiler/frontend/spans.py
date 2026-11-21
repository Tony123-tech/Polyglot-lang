"""Pure source-span helpers for the frontend framework rollout.

A span is a half-open ``[start, end)`` pair of Python string (character)
indices.  Byte offsets are UTF-8 byte offsets derived from encodings of source
prefixes, and line/column pairs are one-based.  API misuse raises
:class:`~compiler.frontend.errors.PolyglotError` with ``E003``; byte offsets
that are out of range or split a multibyte character raise ``E008``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError
from compiler.frontend.framework import (
    _char_to_byte,
    _is_ast_node,
    _line_col_to_offset,
    _make_record,
    _offset_to_line_col,
    _token_end_char_offset,
)
from lexer import Token

__all__ = [
    "span_byte_to_char",
    "span_char_to_byte",
    "span_clamp",
    "span_compare",
    "span_contains",
    "span_format",
    "span_from_ast_node",
    "span_from_token",
    "span_intersect",
    "span_line_col_to_offset",
    "span_line_range",
    "span_make",
    "span_merge_adjacent",
    "span_offset_to_line_col",
    "span_overlaps",
    "span_shift",
    "span_text",
    "span_union",
    "span_utf8_validate",
    "span_validate",
]


def _fail(message: str, *, code: str = "E003", line: int = 0, col: int = 0) -> NoReturn:
    """Raise a structured frontend diagnostic at an optional source location."""
    raise PolyglotError(code, message, line=line, col=col)


def _as_source(value: object, name: str = "source") -> str:
    """Require *value* to be source text, raising ``E003`` otherwise."""
    if not isinstance(value, str):
        _fail(f"{name} must be str, not {type(value).__name__}")
    return value


def _as_int(value: object, name: str) -> int:
    """Require *value* to be a plain integer (``bool`` is not accepted)."""
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{name} must be int, not {type(value).__name__}")
    return value


def _as_int_pair(value: object, name: str = "span") -> tuple[int, int]:
    """Require a two-element integer sequence without checking endpoint order."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{name} must be a two-integer sequence")
    if len(value) != 2:
        _fail(f"{name} must contain exactly two integers")
    return _as_int(value[0], f"{name}[0]"), _as_int(value[1], f"{name}[1]")


def _as_span(value: object, name: str = "span") -> tuple[int, int]:
    """Require a valid two-integer, non-decreasing half-open span."""
    start, end = _as_int_pair(value, name)
    if start > end:
        _fail(f"{name} must be non-decreasing")
    return start, end


def _is_char_boundary(source: str, offset: int) -> bool:
    """True when a numeric offset is a boundary in the UTF-8 byte stream.

    A numeric offset that lands on a continuation byte of a multibyte character
    splits that character even though it may be a valid Python string index.
    """
    if offset < 0 or offset > len(source):
        return False
    if offset == 0 or offset == len(source):
        return True
    cursor = 0
    for ch in source:
        cursor += len(ch.encode("utf-8"))
        if cursor == offset:
            return True
        if cursor > offset:
            return False
    return False


def span_offset_to_line_col(source: str, offset: int) -> list[int]:
    """Map a character offset to a one-based ``[line, col]`` pair.

    The offset is clamped into ``[0, len(source)]`` before the conversion, so
    negative and past-the-end offsets still produce a meaningful location.
    """
    source = _as_source(source)
    offset = _as_int(offset, "offset")
    line, col = _offset_to_line_col(source, offset)
    return [line, col]


def span_line_col_to_offset(source: str, line: int, col: int) -> int:
    """Map a one-based ``(line, col)`` pair back to a clamped character offset.

    Lines or columns below one resolve to offset zero; locations past the last
    line or column resolve to ``len(source)``.
    """
    source = _as_source(source)
    line = _as_int(line, "line")
    col = _as_int(col, "col")
    return _line_col_to_offset(source, line, col)


def span_char_to_byte(source: str, offset: int) -> int:
    """Return the UTF-8 byte offset corresponding to a clamped char offset."""
    source = _as_source(source)
    offset = _as_int(offset, "offset")
    return _char_to_byte(source, offset)


def span_byte_to_char(source: str, offset: int) -> int:
    """Invert :func:`span_char_to_byte` for exact UTF-8 byte offsets.

    ``E008`` is raised when the byte offset is negative, past the encoded text,
    or lands inside a multibyte character rather than on a character boundary.
    """
    source = _as_source(source)
    offset = _as_int(offset, "offset")
    if offset < 0:
        line, col = _offset_to_line_col(source, 0)
        _fail(
            f"byte offset {offset} is out of range",
            code="E008",
            line=line,
            col=col,
        )
    cursor = 0
    for index, ch in enumerate(source):
        width = len(ch.encode("utf-8"))
        if offset < cursor + width:
            if offset == cursor:
                return index
            line, col = _offset_to_line_col(source, index)
            _fail(
                f"byte offset {offset} splits a multibyte character",
                code="E008",
                line=line,
                col=col,
            )
        cursor += width
    if offset == cursor:
        return len(source)
    line, col = _offset_to_line_col(source, len(source))
    _fail(
        f"byte offset {offset} is out of range",
        code="E008",
        line=line,
        col=col,
    )


def span_make(source: str, start: int, end: int) -> dict:
    """Build the standard span record for ``[start, end)`` in *source*.

    Endpoints are clamped to the source length and swapped when necessary so
    the recorded span is always non-decreasing.  The returned record carries
    character, UTF-8 byte and one-based line/column spans plus the sliced text.
    """
    source = _as_source(source)
    start = _as_int(start, "start")
    end = _as_int(end, "end")
    start, end = min(start, end), max(start, end)
    start = max(0, min(start, len(source)))
    end = max(start, min(end, len(source)))
    start_line, start_col = _offset_to_line_col(source, start)
    end_line, end_col = _offset_to_line_col(source, end)
    return {
        "char_span": [start, end],
        "byte_span": [_char_to_byte(source, start), _char_to_byte(source, end)],
        "line_col_span": [
            [start_line, start_col],
            [end_line, end_col],
        ],
        "text": source[start:end],
    }


def span_union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Return the smallest span covering both validated spans *a* and *b*."""
    first_start, first_end = _as_span(a, "a")
    second_start, second_end = _as_span(b, "b")
    return [min(first_start, second_start), max(first_end, second_end)]


def span_intersect(a: Sequence[int], b: Sequence[int]) -> list[int] | None:
    """Return the overlap of *a* and *b*, or ``None`` when disjoint/touching."""
    first_start, first_end = _as_span(a, "a")
    second_start, second_end = _as_span(b, "b")
    start = max(first_start, second_start)
    end = min(first_end, second_end)
    return [start, end] if start < end else None


def span_contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    """True when *outer* fully contains *inner*, shared boundaries included."""
    outer_start, outer_end = _as_span(outer, "outer")
    inner_start, inner_end = _as_span(inner, "inner")
    return outer_start <= inner_start and inner_end <= outer_end


def span_overlaps(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when two half-open spans share at least one character."""
    first_start, first_end = _as_span(a, "a")
    second_start, second_end = _as_span(b, "b")
    return first_start < second_end and second_start < first_end


def span_compare(a: Sequence[int], b: Sequence[int]) -> int:
    """Order spans by start offset, breaking ties by end offset."""
    first_start, first_end = _as_span(a, "a")
    second_start, second_end = _as_span(b, "b")
    if first_start != second_start:
        return -1 if first_start < second_start else 1
    if first_end != second_end:
        return -1 if first_end < second_end else 1
    return 0


def span_shift(span: Sequence[int], delta: int) -> list[int]:
    """Move both endpoints by *delta*, clamping negative results to zero."""
    start, end = _as_span(span)
    delta = _as_int(delta, "delta")
    return [max(0, start + delta), max(0, end + delta)]


def span_clamp(span: Sequence[int], low: int, high: int) -> list[int]:
    """Restrict a span to ``[low, high]``, collapsing fully outside spans."""
    start, end = _as_span(span)
    low = _as_int(low, "low")
    high = _as_int(high, "high")
    if low > high:
        _fail("low must not exceed high")
    if end <= low:
        return [low, low]
    if start >= high:
        return [high, high]
    return [max(start, low), min(end, high)]


def span_merge_adjacent(spans: Sequence[Sequence[int]]) -> list[list[int]]:
    """Sort spans by start and merge overlapping or touching ranges."""
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        _fail("spans must be a sequence of two-integer spans")
    ordered = sorted(
        (_as_span(item, "spans item") for item in spans),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def span_from_token(source: str, token: Token) -> dict:
    """Build a token record whose end comes from the raw lexeme in *source*.

    Decoded token values can be shorter than their source text (escapes,
    unicode), so the exclusive end is recovered with the lexer-aware
    ``_token_end_char_offset`` helper.  The record also includes token ``type``
    and ``value`` fields alongside the standard span keys.
    """
    source = _as_source(source)
    if not isinstance(token, Token):
        _fail(f"token must be a Token, not {type(token).__name__}")
    end = _token_end_char_offset(source, token)
    record: dict[str, Any] = _make_record(
        source, token.type.name, token.value, token.line, token.col, token.pos, end
    )
    return record


def span_from_ast_node(node: object) -> list[int] | None:
    """Return an AST dataclass node's two-integer span, or ``None`` if absent.

    Ordinary non-node objects and nodes without a span yield ``None`` without
    raising.  A dataclass node that *has* a malformed span attribute is API
    misuse and raises ``E003``.
    """
    if not _is_ast_node(node):
        return None
    value = getattr(node, "span", None)
    if value is None:
        return None
    start, end = _as_int_pair(value, "node.span")
    return [start, end]


def span_line_range(source: str, start_line: int, end_line: int) -> list[int]:
    """Return the character span covering whole lines ``start_line..end_line``.

    Covered lines include their trailing newline, except the last line of the
    file and when *end_line* is the last line.  Line numbers beyond the end of
    the source are clamped to the final line.
    """
    source = _as_source(source)
    start_line = _as_int(start_line, "start_line")
    end_line = _as_int(end_line, "end_line")
    if start_line < 1 or end_line < 1 or start_line > end_line:
        _fail("line numbers must be one-based and non-decreasing")
    lines = source.split("\n")
    first = min(start_line, len(lines))
    last = min(end_line, len(lines))
    start = sum(len(lines[index]) + 1 for index in range(first - 1))
    if last == len(lines):
        end = len(source)
    else:
        end = sum(len(lines[index]) + 1 for index in range(last))
    return [min(start, len(source)), min(end, len(source))]


def span_text(source: str, span: Sequence[int]) -> str:
    """Return the exact source slice for a validated, clamped span."""
    source = _as_source(source)
    start, end = _as_span(span)
    start = max(0, min(start, len(source)))
    end = max(start, min(end, len(source)))
    return source[start:end]


def span_utf8_validate(source: str, span: Sequence[int]) -> dict:
    """Report whether both numeric endpoints are UTF-8 byte-stream boundaries.

    An endpoint that lands inside a multibyte character's continuation bytes is
    reported with ``valid=False`` and ``reason='splits multibyte character'``
    instead of raising.  Bad argument types still raise ``E003``.
    """
    source = _as_source(source)
    start, end = _as_int_pair(span)
    start = max(0, min(start, len(source)))
    end = max(0, min(end, len(source)))
    if start > end:
        start, end = end, start
    valid = _is_char_boundary(source, start) and _is_char_boundary(source, end)
    return {
        "valid": valid,
        "reason": None if valid else "splits multibyte character",
        "char_span": [start, end],
        "byte_span": [_char_to_byte(source, start), _char_to_byte(source, end)],
    }


def span_format(span: Sequence[int]) -> str:
    """Format a validated span as a canonical exclusive-end ``'[a, b)'``."""
    start, end = _as_span(span)
    return f"[{start}, {end})"


def span_validate(source: str, span: Sequence[int]) -> bool:
    """Return whether *span* is valid for *source*, without ever raising.

    Valid spans contain two plain integers with ``0 <= start <= end`` within
    the source length, and both numeric endpoints must be boundaries of the
    UTF-8 encoded byte stream (not interior bytes of a multibyte character).
    """
    try:
        if not isinstance(source, str):
            return False
        if isinstance(span, (str, bytes)) or not isinstance(span, Sequence):
            return False
        if len(span) != 2:
            return False
        start, end = span
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (start, end)
        ):
            return False
        if not 0 <= start <= end <= len(source):
            return False
        return _is_char_boundary(source, start) and _is_char_boundary(source, end)
    except Exception:
        return False
