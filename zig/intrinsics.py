"""Zig-exclusive intrinsic functions exposed to polyglot programs.

Every intrinsic here lowers 1:1 onto a Zig builtin, which is why they belong to
the Zig backend alone and stay outside the Python/native parity contract
(SPEC 8.12.1). The Python side does the part Zig cannot do well from a
source-level call: pin the operand's width, reject a float where an integer is
required, and reject a `bitCast` whose two sides have different bit counts.

Bounded model (SPEC 8.12.18, 10.52): the operand width is taken from the
inferred polyglot type, defaulting to `i64` for an unannotated identifier.
`usize`/`isize` are modelled as 64-bit, which matches the supported platforms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INT_WIDTHS: dict[str, int] = {
    "bool": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "i128": 128,
    "isize": 64,
    "u8": 8,
    "u16": 16,
    "u32": 32,
    "u64": 64,
    "u128": 128,
    "usize": 64,
}

FLOAT_WIDTHS: dict[str, int] = {
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "f128": 128,
}

#: integer intrinsics: polyglot macro name -> Zig builtin
VALUE_INTRINSICS: dict[str, str] = {
    "popcount": "@popCount",
    "clz": "@clz",
    "ctz": "@ctz",
    "bswap": "@byteSwap",
    "bitreverse": "@bitReverse",
}

#: type-level intrinsics: polyglot macro name -> Zig builtin
TYPE_INTRINSICS: dict[str, str] = {
    "bitsizeof": "@bitSizeOf",
    "sizeof": "@sizeOf",
    "alignof": "@alignOf",
}

#: overflow-reporting intrinsics; the flag is the tuple's second element
OVERFLOW_INTRINSICS: dict[str, str] = {
    "add_with_overflow": "@addWithOverflow",
    "mul_with_overflow": "@mulWithOverflow",
}

#: wrapping intrinsics reuse the same builtin and take the tuple's first element
WRAPPING_INTRINSICS: dict[str, str] = {
    "wrapping_add": "@addWithOverflow",
    "wrapping_mul": "@mulWithOverflow",
}

#: rotate intrinsics: polyglot macro name -> std.math function
ROTATE_INTRINSICS: dict[str, str] = {
    "rotl": "rotl",
    "rotr": "rotr",
}

#: float/integer conversion intrinsics: polyglot macro name -> Zig builtin
CONVERSION_INTRINSICS: dict[str, str] = {
    "int_from_float": "@intFromFloat",
    "float_from_int": "@floatFromInt",
}

#: slice/pointer memory intrinsics: polyglot macro name -> Zig builtin
MEMORY_INTRINSICS: dict[str, str] = {
    "memcpy_bytes": "@memcpy",
    "memset_bytes": "@memset",
}

_ARBITRARY_RE = re.compile(r"^([iu])(\d+)$")


class IntrinsicError(Exception):
    """A rejected intrinsic call, reported by the backend as ``E001``/``E003``."""


@dataclass(frozen=True)
class IntrinsicShape:
    """What the operand must be, and what the Zig builtin needs."""

    name: str
    builtin: str
    requires_byte_multiple: bool = False


def bit_width(type_name: str) -> int | None:
    """Bit width of a Zig type this backend models, or ``None``."""
    if type_name in INT_WIDTHS:
        return INT_WIDTHS[type_name]
    if type_name in FLOAT_WIDTHS:
        return FLOAT_WIDTHS[type_name]
    match = _ARBITRARY_RE.match(type_name)
    if match:
        bits = int(match.group(2))
        if 0 <= bits <= 65535:
            return bits
    return None


def is_float(type_name: str) -> bool:
    return type_name in FLOAT_WIDTHS


def shape_for(name: str) -> IntrinsicShape | None:
    """Look up a value intrinsic, rejecting a byte-width mismatch later."""
    builtin = VALUE_INTRINSICS.get(name)
    if builtin is None:
        return None
    return IntrinsicShape(
        name=name,
        builtin=builtin,
        requires_byte_multiple=name == "bswap",
    )


def require_value_operand(name: str, type_name: str) -> IntrinsicShape:
    """Validate the operand of an integer intrinsic and return its shape."""
    shape = shape_for(name)
    if shape is None:
        raise IntrinsicError(f"unknown Zig intrinsic {name!r}")
    if is_float(type_name):
        raise IntrinsicError(
            f"{name}! needs an integer operand, got {type_name}; "
            "use a bitCast or an integer type"
        )
    width = bit_width(type_name)
    if width is None:
        raise IntrinsicError(
            f"{name}! cannot size operand type {type_name!r}; "
            "annotate the value with a fixed-width integer type"
        )
    if shape.requires_byte_multiple and width % 8 != 0:
        raise IntrinsicError(
            f"bswap! needs a byte-sized operand, {type_name} is {width} bits"
        )
    return shape


def require_type_operand(name: str, type_name: str) -> str:
    """Validate the type argument of a type-level intrinsic."""
    builtin = TYPE_INTRINSICS.get(name)
    if builtin is None:
        raise IntrinsicError(f"unknown Zig intrinsic {name!r}")
    if bit_width(type_name) is None:
        raise IntrinsicError(
            f"{name}! needs a type with a known bit size, got {type_name!r}"
        )
    return builtin


def int_shape(type_name: str) -> tuple[int, bool] | None:
    """``(width, signed)`` for an integer type this backend models, or ``None``."""
    if type_name == "bool":
        return 1, False
    if type_name in INT_WIDTHS:
        return INT_WIDTHS[type_name], type_name.startswith("i")
    match = _ARBITRARY_RE.match(type_name)
    if match:
        bits = int(match.group(2))
        if 0 <= bits <= 65535:
            return bits, match.group(1) == "i"
    return None


def require_arithmetic_operand(name: str, type_name: str) -> tuple[int, bool]:
    """Validate an operand that must be a real integer (not float, not ``bool``)."""
    if is_float(type_name):
        raise IntrinsicError(
            f"{name}! needs an integer operand, got {type_name}; "
            "use bitcast! to read the raw bits instead"
        )
    if type_name == "bool":
        raise IntrinsicError(
            f"{name}! needs an integer wider than one bit, got bool"
        )
    shape = int_shape(type_name)
    if shape is None:
        raise IntrinsicError(
            f"{name}! cannot size operand type {type_name!r}; "
            "annotate the value with a fixed-width integer type"
        )
    if shape[0] == 0:
        raise IntrinsicError(f"{name}! needs a non-zero bit width, got {type_name}")
    return shape


def require_matching_operands(name: str, left: str, right: str) -> None:
    """Both operands of an overflow or rotate intrinsic must share one type."""
    if left != right:
        raise IntrinsicError(
            f"{name}! needs both operands to have the same type, got {left} and {right}"
        )


def require_float_operand(name: str, type_name: str) -> int:
    """Validate the float side of a conversion intrinsic."""
    width = FLOAT_WIDTHS.get(type_name)
    if width is None:
        raise IntrinsicError(
            f"{name}! needs a float operand, got {type_name}; "
            "use float_from_int! to convert an integer"
        )
    return width


# -- Python mirrors, used by the tests to cross-check Zig's answers --------


def _signed_value(wrapped: int, width: int, signed: bool) -> int:
    if not signed:
        return wrapped
    sign_bit = 1 << (width - 1)
    return wrapped - (1 << width) if wrapped & sign_bit else wrapped


def mirror_overflow(
    name: str, left: int, right: int, type_name: str
) -> tuple[int, int]:
    """``(value, flag)`` for ``add_with_overflow!`` / ``mul_with_overflow!``."""
    shape = int_shape(type_name)
    if shape is None or shape[0] == 0:
        raise IntrinsicError(f"no overflow mirror for {type_name!r}")
    width, signed = shape
    if name in {"add_with_overflow", "wrapping_add"}:
        raw = left + right
    elif name in {"mul_with_overflow", "wrapping_mul"}:
        raw = left * right
    else:
        raise IntrinsicError(f"no overflow mirror for {name!r}")
    mask = (1 << width) - 1
    value = _signed_value(raw & mask, width, signed)
    if signed:
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    else:
        low, high = 0, mask
    return value, int(raw < low or raw > high)


def mirror_rotate(name: str, value: int, count: int, type_name: str) -> int:
    """Rotate within the declared width, matching ``std.math.rotl``/``rotr``."""
    shape = int_shape(type_name)
    if shape is None or shape[0] == 0:
        raise IntrinsicError(f"no rotate mirror for {type_name!r}")
    width, signed = shape
    mask = (1 << width) - 1
    unsigned = int(value) & mask
    turns = count % width
    if name == "rotl":
        result = ((unsigned << turns) | (unsigned >> (width - turns))) & mask
    elif name == "rotr":
        result = ((unsigned >> turns) | (unsigned << (width - turns))) & mask
    else:
        raise IntrinsicError(f"no rotate mirror for {name!r}")
    return _signed_value(result, width, signed)


# -- Python mirrors, used by the tests to cross-check Zig's answers --------


def mirror(name: str, value: int, type_name: str) -> int:
    """Compute the intrinsic in Python, matching Zig for unsigned widths."""
    width = bit_width(type_name)
    if width is None or width == 0:
        raise IntrinsicError(f"no mirror for {type_name!r}")
    mask = (1 << width) - 1
    unsigned = int(value) & mask
    if name == "popcount":
        return unsigned.bit_count()
    if name == "clz":
        return width - unsigned.bit_length()
    if name == "ctz":
        return width if unsigned == 0 else (unsigned & -unsigned).bit_length() - 1
    if name == "bswap":
        if width % 8 != 0:
            raise IntrinsicError(f"bswap! needs a byte-sized operand, got {width} bits")
        raw = unsigned.to_bytes(width // 8, "big")
        return int.from_bytes(raw[::-1], "big")
    if name == "bitreverse":
        return int(f"{unsigned:0{width}b}"[::-1], 2)
    raise IntrinsicError(f"no mirror for {name!r}")
