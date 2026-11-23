"""Zig-exclusive primitive semantics: saturating arithmetic & arbitrary bit-width integers."""

from __future__ import annotations

import re

_ARB_INT_RE = re.compile(r"^([iu])(\d+)$")


def parse_arbitrary_int_type(type_name: str) -> tuple[int, bool] | None:
    """Parse Zig arbitrary bit-width integer types like `u1`, `u3`, `i19`, `u24`."""
    match = _ARB_INT_RE.match(type_name)
    if not match:
        return None
    signed = match.group(1) == "i"
    bits = int(match.group(2))
    if not 0 <= bits <= 65535:
        raise ValueError(f"Zig integer bit-width must be 0..=65535, got {bits}")
    return bits, signed


def int_bounds(type_name: str) -> tuple[int, int]:
    """Return (min_val, max_val) for standard or arbitrary bit-width integers."""
    parsed = parse_arbitrary_int_type(type_name)
    if parsed is None:
        raise TypeError(f"not an integer type: {type_name}")
    bits, signed = parsed
    if bits == 0:
        return (0, 0)
    if signed:
        return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return (0, (1 << bits) - 1)


def wrap_arbitrary_int(value: int, type_name: str) -> int:
    """Wrap/truncate an integer to arbitrary bit-width `iN` or `uN`."""
    parsed = parse_arbitrary_int_type(type_name)
    if parsed is None:
        return value
    bits, signed = parsed
    if bits == 0:
        return 0
    mask = (1 << bits) - 1
    wrapped = int(value) & mask
    if signed and wrapped >= (1 << (bits - 1)):
        wrapped -= 1 << bits
    return wrapped


def saturating_op(left: int, right: int, op: str, type_name: str = "i64") -> int:
    """Zig saturating arithmetic (+|, -|, *|) clamping at integer type bounds."""
    if op in {"+", "+|", "+|="}:
        raw = left + right
    elif op in {"-", "-|", "-|="}:
        raw = left - right
    elif op in {"*", "*|", "*|="}:
        raw = left * right
    else:
        raise ValueError(f"unsupported saturating operator: {op!r}")

    lo, hi = int_bounds(type_name)
    if raw < lo:
        return lo
    if raw > hi:
        return hi
    return raw


# Markers that make an example Zig-only. Both the parity suite and the example
# runner read this one list, so a new Zig-exclusive construct cannot be
# forgotten in one of them.
ZIG_ONLY_SOURCE_MARKERS = (
    "+|",
    "-|",
    "*|",
    "SimdN<",
    "#[parallel]",
    "#[packed]",
    "#[unchecked]",
    "popcount!(",
    "clz!(",
    "ctz!(",
    "bswap!(",
    "bitreverse!(",
    "bitcast!(",
    "bitsizeof!(",
    "sizeof!(",
)


def is_zig_only_source(source: str) -> bool:
    """True when a program needs `--backend zig` (SPEC 8.12.5-8.12.8, 8.12.15-8.12.18)."""
    return any(marker in source for marker in ZIG_ONLY_SOURCE_MARKERS)
