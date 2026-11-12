"""Runtime helpers used by the Python backend."""

from __future__ import annotations


def panic(msg: str = "") -> None:
    raise RuntimeError(msg or "panic")


def assert_eq(a, b) -> None:
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")


def assert_ne(a, b) -> None:
    if a == b:
        raise AssertionError(f"{a!r} == {b!r}")


# ---------------------------------------------------------------------------
# N7: usize / isize boundary helpers
# ---------------------------------------------------------------------------

import sys

# Target pointer width (default: host). Can be overridden per compile via
# set_target_width(); SPEC 2.4.10-2.4.12.
HOST_TARGET_WIDTH = 64 if sys.maxsize > 2**32 else 32
_TARGET_WIDTH = HOST_TARGET_WIDTH


def set_target_width(width: int) -> None:
    """Set the simulated target width (32 or 64). SPEC 2.4.12(c)."""
    global _TARGET_WIDTH
    if width not in (32, 64):
        raise ValueError(f"target width must be 32 or 64, got {width}")
    _TARGET_WIDTH = width


def get_target_width() -> int:
    """Return the current target width."""
    return _TARGET_WIDTH


def _validated_width(width: int | None) -> int:
    target_width = _TARGET_WIDTH if width is None else width
    if target_width not in (32, 64):
        raise ValueError(f"target width must be 32 or 64, got {target_width}")
    return target_width


def _usize_max(width: int | None = None) -> int:
    return 2**_validated_width(width) - 1


def _isize_min(width: int | None = None) -> int:
    return -(2 ** (_validated_width(width) - 1))


def _isize_max(width: int | None = None) -> int:
    return 2 ** (_validated_width(width) - 1) - 1


def cast_usize(value: int, *, target_width: int | None = None) -> int:
    """`expr as usize`: range check, raise OverflowError if out of range.

    SPEC 2.4.6: no implicit conversion.
    SPEC 2.4.9: OverflowError only at an explicit `as` boundary.
    """
    if not isinstance(value, int):
        raise TypeError(f"cannot cast {type(value).__name__} to usize")
    width = _validated_width(target_width)
    if value < 0 or value > _usize_max(width):
        raise OverflowError(
            f"value {value} does not fit in usize "
            f"(target width {width}-bit)"
        )
    return value


def cast_isize(value: int, *, target_width: int | None = None) -> int:
    """`expr as isize`: range check, raise OverflowError if out of range."""
    if not isinstance(value, int):
        raise TypeError(f"cannot cast {type(value).__name__} to isize")
    width = _validated_width(target_width)
    if value < _isize_min(width) or value > _isize_max(width):
        raise OverflowError(
            f"value {value} does not fit in isize "
            f"(target width {width}-bit)"
        )
    return value


def cast_i64(value: int) -> int:
    """`expr as i64`: wrap to signed 64-bit range (matches Rust semantics)."""
    if not isinstance(value, int):
        raise TypeError(f"cannot cast {type(value).__name__} to i64")
    mask = (1 << 64) - 1
    v = value & mask
    if v >= (1 << 63):
        v -= 1 << 64
    return v
