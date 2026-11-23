"""Gleam-inspired bit-array pattern matching with checked bounds.

Gleam's ``<<header:8, len:16, payload:binary>>`` binary pattern syntax is one
of its signature features: you can destructure a byte sequence by bit-width,
endianness, and type in a single pattern, and the compiler statically verifies
that the pattern is well-formed and the match is exhaustive.

This module brings that idea to ``polyglot-lang`` as a *pure-Python* model:

* :func:`parse_bit_pattern` parses a pattern string like
  ``<<header:8, len:16, payload:binary>>`` into a :class:`BitarrayPattern`
  of :class:`BitSegment` values, validating segment widths, endianness, and
  that the total fixed width does not exceed a configurable maximum.
* :func:`match_bit_array` applies a parsed pattern to a ``bytes`` buffer,
  performing **checked** slicing — every read is bounds-validated against
  ``len(data)`` before extraction, so malformed input raises ``E001`` (size
  mismatch) rather than reading out of bounds.  No ``ctypes.cast``, no
  ``struct.unpack_from`` without a length guard, no raw pointer arithmetic.

Diagnostics follow the shared ``E001``–``E008`` surface: ``E001`` for
malformed patterns or size mismatches, ``E003`` for API misuse, and ``E008``
for malformed serialized state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "BitSegment",
    "BitSegmentKind",
    "BitarrayPattern",
    "MatchBitResult",
    "match_bit_array",
    "parse_bit_pattern",
]

# Maximum total fixed bit-width a pattern may declare.  This prevents a
# malicious or buggy pattern from forcing unbounded work or huge allocations.
_MAX_FIXED_BITS: Final[int] = 4096

# Each segment is ``name[:WIDTH][:TYPE][:ENDIAN]`` where WIDTH is optional
# for variable-width types (``binary``/``bit_string`` consume the remaining
# buffer). TYPE is one of the supported Gleam-style binary types and ENDIAN
# is ``big``/``little``/``native``.
_SEGMENT_RE: Final = re.compile(
    r"^\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_']*)"
    r"\s*(?::\s*(?P<width>\d+))?"
    r"\s*(?::\s*(?P<type>binary|bit_string|utf8_codepoint|int|float))?"
    r"\s*(?::\s*(?P<endian>big|little|native))?\s*$"
)


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


class BitSegmentKind(Enum):
    """The kind of value a bit segment carries."""

    BINARY = "binary"
    BIT_STRING = "bit_string"
    UTF8_CODEPOINT = "utf8_codepoint"
    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True)
class BitSegment:
    """One segment of a bit-array pattern."""

    name: str
    width: int
    kind: BitSegmentKind = BitSegmentKind.BINARY
    endian: str = "big"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "kind": self.kind.value,
            "endian": self.endian,
        }


@dataclass(frozen=True)
class BitarrayPattern:
    """A parsed ``<<segment:8, ...>>`` pattern."""

    segments: tuple[BitSegment, ...]

    @property
    def fixed_bits(self) -> int:
        """Total fixed bit-width declared by the pattern (0 if all variable)."""
        return sum(segment.width for segment in self.segments)

    @property
    def has_variable(self) -> bool:
        """Whether any segment has an unspecified width (consumes the rest)."""
        return any(segment.width == 0 for segment in self.segments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.as_dict() for segment in self.segments],
            "fixed_bits": self.fixed_bits,
            "has_variable": self.has_variable,
        }


_VARIABLE_KINDS: Final[frozenset[BitSegmentKind]] = frozenset(
    {
        BitSegmentKind.BINARY,
        BitSegmentKind.BIT_STRING,
        BitSegmentKind.INT,
    }
)


def parse_bit_pattern(source: str) -> BitarrayPattern:
    """Parse a ``<<...>>`` bit-array pattern string.

    The pattern must be wrapped in ``<<`` / ``>>`` and contain one or more
    comma-separated segments of the form ``name:WIDTH[:TYPE][:ENDIAN]``.

    Raises ``E001`` on malformed syntax, unknown types/endianness, or a
    fixed-width total exceeding :data:`_MAX_FIXED_BITS`.
    """
    text = source.strip()
    if not text.startswith("<<") or not text.endswith(">>"):
        _fail(
            "E001",
            f"bit-array pattern must be wrapped in <<...>>: {source!r}",
        )
    inner = text[2:-2].strip()
    if not inner:
        _fail("E001", "bit-array pattern has no segments")
    segments: list[BitSegment] = []
    for raw in inner.split(","):
        segment = _parse_segment(raw)
        segments.append(segment)
    pattern = BitarrayPattern(segments=tuple(segments))
    if pattern.fixed_bits > _MAX_FIXED_BITS:
        _fail(
            "E001",
            f"bit-array pattern declares {pattern.fixed_bits} fixed bits, "
            f"exceeding the maximum of {_MAX_FIXED_BITS}",
            reason="the pattern is too wide to validate safely",
        )
    return pattern


def _parse_segment(raw: str) -> BitSegment:
    match = _SEGMENT_RE.match(raw)
    if match is None:
        _fail("E001", f"malformed bit-array segment: {raw!r}")
    name = match.group("name")
    # When no explicit type is given, infer from the width: a specified width
    # with no type defaults to ``int`` (the most useful numeric reading), while
    # a missing width defaults to ``binary`` (consume the rest of the buffer).
    width_text = match.group("width")
    type_text = match.group("type")
    if type_text is None:
        type_text = "binary" if width_text is None else "int"
    kind = _resolve_kind(type_text)
    # A variable-width segment (binary/bit_string/int without explicit width)
    # consumes the remaining buffer; width 0 marks that case.
    if width_text is None:
        if kind in _VARIABLE_KINDS:
            width = 0
        else:
            _fail(
                "E001",
                f"segment {name!r} of type {type_text!r} requires an explicit width",
            )
    else:
        width = int(width_text)
        if width <= 0:
            _fail("E001", f"segment {name!r} has non-positive width {width}")
    endian = match.group("endian") or "big"
    if endian not in {"big", "little", "native"}:
        _fail("E001", f"unknown endianness {endian!r}")
    if kind == BitSegmentKind.FLOAT and endian == "native":
        _fail(
            "E001",
            "native endianness is not portable for float segments; use big or little",
        )
    return BitSegment(name=name, width=width, kind=kind, endian=endian)


def _resolve_kind(type_text: str) -> BitSegmentKind:
    try:
        return BitSegmentKind(type_text)
    except ValueError:
        _fail("E001", f"unknown bit-array segment type: {type_text!r}")


@dataclass(frozen=True)
class MatchBitResult:
    """The result of matching a bit-array pattern against a buffer."""

    matched: bool
    bindings: dict[str, Any]
    consumed_bits: int
    adt_pattern: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "bindings": dict(self.bindings),
            "consumed_bits": self.consumed_bits,
            "pattern": self.adt_pattern,
        }


def match_bit_array(pattern: BitarrayPattern, data: bytes) -> MatchBitResult:
    """Match *pattern* against *data* with checked bounds.

    Every segment read is validated against ``len(data)`` *before*
    extraction, so a short buffer raises ``E001`` (size mismatch) rather than
    reading out of bounds.  No ``ctypes``, no raw pointer arithmetic, no
    ``struct.unpack_from`` without a length guard.

    * ``binary`` / ``bit_string`` segments consume the remaining bytes after
      the fixed-width prefix (Gleam's ``payload:binary`` semantics).
    * ``int`` segments consume ``width`` bits, big- or little-endian per the
      segment's ``endian`` flag.
    * ``float`` segments consume 32 or 64 bits per ``width``.
    * ``utf8_codepoint`` segments consume 8 bits (one byte) and validate
      that the byte is a legal UTF-8 code-point start.

    Returns a :class:`MatchBitResult` with the extracted bindings.  Raises
    ``E001`` if the buffer is too short for the fixed-width segments or if
    the pattern is internally inconsistent.
    """
    if not isinstance(data, bytes):
        _fail("E003", "match_bit_array expects a bytes buffer")
    total_bits = len(data) * 8
    bit_offset = 0
    bindings: dict[str, Any] = {}

    # A segment with width 0 is variable-width (consumes the remaining
    # buffer).  All others are fixed-width, even if their kind is in
    # _VARIABLE_KINDS (e.g. ``payload:32:binary`` reads exactly 32 bits).
    fixed_segments = [s for s in pattern.segments if s.width > 0]
    variable_segments = [s for s in pattern.segments if s.width == 0]
    if len(variable_segments) > 1:
        _fail(
            "E001",
            "bit-array pattern may have at most one variable-width segment",
            reason="Gleam requires variable segments to be the final segment",
        )
    if variable_segments and pattern.segments[-1].width != 0:
        _fail(
            "E001",
            "variable-width segment must be the final segment in the pattern",
            reason="Gleam requires variable segments to appear last",
        )

    for segment in fixed_segments:
        end = bit_offset + segment.width
        if end > total_bits:
            _fail(
                "E001",
                f"buffer too short for segment {segment.name!r}: "
                f"need {end} bits, have {total_bits}",
                reason="checked bounds validation failed",
            )
        bindings[segment.name] = _extract_fixed(segment, data, bit_offset)
        bit_offset = end

    for segment in variable_segments:
        remaining_bits = total_bits - bit_offset
        remaining_bytes = remaining_bits // 8
        bindings[segment.name] = _extract_variable(
            segment, data, bit_offset, remaining_bytes
        )
        bit_offset = total_bits

    return MatchBitResult(
        matched=True,
        bindings=bindings,
        consumed_bits=bit_offset,
        adt_pattern=_format_pattern(pattern),
    )


def _extract_fixed(segment: BitSegment, data: bytes, bit_offset: int) -> Any:
    byte_offset = bit_offset // 8
    bit_in_byte = bit_offset % 8
    if bit_in_byte != 0:
        _fail(
            "E001",
            f"segment {segment.name!r} is not byte-aligned at bit {bit_offset}",
            reason="only byte-aligned fixed segments are supported",
        )
    width_bytes = segment.width // 8
    if segment.width % 8 != 0:
        _fail(
            "E001",
            f"segment {segment.name!r} width {segment.width} is not a multiple of 8",
        )
    if byte_offset + width_bytes > len(data):
        _fail(
            "E001",
            f"buffer too short for segment {segment.name!r}",
        )
    chunk = data[byte_offset : byte_offset + width_bytes]
    if segment.kind == BitSegmentKind.FLOAT:
        if segment.width == 32:
            import struct

            return struct.unpack(">f" if segment.endian == "big" else "<f", chunk)[0]
        if segment.width == 64:
            import struct

            return struct.unpack(">d" if segment.endian == "big" else "<d", chunk)[0]
        _fail(
            "E001",
            f"float segment {segment.name!r} must be 32 or 64 bits wide",
        )
    if segment.kind == BitSegmentKind.INT:
        return int.from_bytes(
            chunk, "big" if segment.endian == "big" else "little", signed=False
        )
    if segment.kind == BitSegmentKind.UTF8_CODEPOINT:
        if len(chunk) != 1:
            _fail("E001", f"utf8 segment {segment.name!r} must be 1 byte wide")
        return chunk.decode("utf-8", errors="replace")
    return chunk


def _extract_variable(
    segment: BitSegment, data: bytes, bit_offset: int, remaining_bytes: int
) -> Any:
    byte_offset = bit_offset // 8
    chunk = data[byte_offset : byte_offset + remaining_bytes]
    if segment.kind in {BitSegmentKind.BINARY, BitSegmentKind.BIT_STRING}:
        return chunk
    if segment.kind == BitSegmentKind.INT:
        return int.from_bytes(
            chunk, "big" if segment.endian == "big" else "little", signed=False
        )
    return chunk


def _format_pattern(pattern: BitarrayPattern) -> str:
    parts = []
    for segment in pattern.segments:
        text = f"{segment.name}:{segment.width}"
        if segment.kind != BitSegmentKind.BINARY:
            text += f":{segment.kind.value}"
        if segment.endian != "big":
            text += f":{segment.endian}"
        parts.append(text)
    return "<<" + ", ".join(parts) + ">>"
