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


def cast_i64(value) -> int:
    """`expr as i64`: wrap to signed 64-bit range (matches Rust semantics)."""
    if isinstance(value, bool):
        value = int(value)
    elif isinstance(value, str):
        if len(value) == 1:
            value = ord(value)
        else:
            raise TypeError("cannot cast str to i64")
    elif isinstance(value, float):
        import math
        if math.isnan(value) or math.isinf(value):
            return 0
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"cannot cast {type(value).__name__} to i64")
    mask = (1 << 64) - 1
    v = value & mask
    if v >= (1 << 63):
        v -= 1 << 64
    return v


# ---------------------------------------------------------------------------
# M4: additional integer casts
# ---------------------------------------------------------------------------

_INT_SPECS = {
    "i8": (8, True),
    "i16": (16, True),
    "i32": (32, True),
    "i64": (64, True),
    "i128": (128, True),
    "u8": (8, False),
    "u16": (16, False),
    "u32": (32, False),
    "u64": (64, False),
    "u128": (128, False),
}


def _int_range(name: str, target_width: int | None = None):
    if name == "isize":
        w = _validated_width(target_width)
        return -(1 << (w - 1)), (1 << (w - 1)) - 1
    if name == "usize":
        w = _validated_width(target_width)
        return 0, (1 << w) - 1
    bits, signed = _INT_SPECS[name]
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def cast_int(value, ty: str, *, target_width: int | None = None, literal_out_of_range: bool = False) -> int:
    """Cast integer to target type.

    For Python backend (M4): if literal_out_of_range is True, raise
    OverflowError at the `as` boundary. Otherwise apply Rust-like wrapping
    for fixed-width integers and range checking for pointer-sized types.
    """
    if isinstance(value, bool):
        value = int(value)
    elif isinstance(value, str):
        if len(value) == 1:
            value = ord(value)
        else:
            raise TypeError(f"cannot cast str to {ty}")
    elif isinstance(value, float):
        import math
        if math.isnan(value) or math.isinf(value):
            value = 0
        else:
            value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"cannot cast {type(value).__name__} to {ty}")
    if literal_out_of_range:
        lo, hi = _int_range(ty, target_width)
        raise OverflowError(
            f"value {value} does not fit in {ty} (range {lo}..={hi})"
        )
    if ty == "usize":
        # explicit as usize: range check per existing behavior
        return cast_usize(value, target_width=target_width)
    if ty == "isize":
        return cast_isize(value, target_width=target_width)
    if ty not in _INT_SPECS:
        raise TypeError(f"unsupported integer cast target {ty}")
    bits, signed = _INT_SPECS[ty]
    mask = (1 << bits) - 1
    v = value & mask
    if signed:
        sign = 1 << (bits - 1)
        if v >= sign:
            v -= 1 << bits
    return v


def cast_i8(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "i8", literal_out_of_range=literal_out_of_range)

def cast_i16(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "i16", literal_out_of_range=literal_out_of_range)

def cast_i32(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "i32", literal_out_of_range=literal_out_of_range)

def cast_i128(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "i128", literal_out_of_range=literal_out_of_range)

def cast_u8(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "u8", literal_out_of_range=literal_out_of_range)

def cast_u16(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "u16", literal_out_of_range=literal_out_of_range)

def cast_u32(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "u32", literal_out_of_range=literal_out_of_range)

def cast_u64(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "u64", literal_out_of_range=literal_out_of_range)

def cast_u128(value: int, *, literal_out_of_range: bool = False) -> int:
    return cast_int(value, "u128", literal_out_of_range=literal_out_of_range)


def int_in_range(value: int, ty: str, target_width: int | None = None) -> bool:
    lo, hi = _int_range(ty, target_width)
    return lo <= value <= hi


# ---------------------------------------------------------------------------
# M3: BTreeMap and HashSet helpers (Python backend)
# ---------------------------------------------------------------------------

class _BTreeMap(dict):
    """BTreeMap representation: dict with sorted iteration semantics."""

    def sorted_items(self):
        return [(k, self[k]) for k in sorted(self.keys())]


def btreemap_new():
    return _BTreeMap()


def btreemap_insert(m, k, v):
    prev = m.get(k)
    m[k] = v
    if prev is None and k not in m:
        # shouldn't happen
        return None
    return prev


def btreemap_insert_with_prev(m, k, v):
    prev = m.get(k)
    m[k] = v
    return prev


def btreemap_get(m, k):
    return m.get(k)


def btreemap_remove(m, k):
    if k in m:
        return m.pop(k)
    return None


def btreemap_contains(m, k):
    return k in m


def btreemap_len(m):
    return len(m)


def btreemap_is_empty(m):
    return len(m) == 0


def btreemap_keys(m):
    return [k for k in sorted(m.keys())]


def btreemap_values(m):
    return [m[k] for k in sorted(m.keys())]


def btreemap_iter(m):
    return [(k, m[k]) for k in sorted(m.keys())]


class _BTreeEntry:
    def __init__(self, mapping, key):
        self.mapping = mapping
        self.key = key

    def or_insert(self, value):
        if self.key not in self.mapping:
            self.mapping[self.key] = value
            return value
        return self.mapping[self.key]

    def or_insert_with(self, factory):
        if self.key not in self.mapping:
            self.mapping[self.key] = factory()
        return self.mapping[self.key]

    def or_insert_with_key(self, factory):
        if self.key not in self.mapping:
            self.mapping[self.key] = factory(self.key)
        return self.mapping[self.key]

    def and_modify(self, f):
        if self.key in self.mapping:
            f(self.mapping[self.key])
        return self


class _BTreeSlot:
    def __init__(self, mapping, key):
        self.mapping = mapping
        self.key = key

    def get(self):
        return self.mapping[self.key]

    def set(self, value):
        self.mapping[self.key] = value


class _Option:
    def __init__(self, value, is_some):
        self._value = value
        self._is_some = is_some

    @staticmethod
    def some(v):
        return _Option(v, True)

    @staticmethod
    def none():
        return _Option(None, False)

    def is_some(self):
        return self._is_some

    def is_none(self):
        return not self._is_some

    def unwrap(self):
        if not self._is_some:
            raise RuntimeError("called `Option::unwrap()` on a `None` value")
        return self._value

    def unwrap_or(self, default):
        return self._value if self._is_some else default

    def unwrap_or_else(self, f):
        return self._value if self._is_some else f()


def btreemap_entry(m, k):
    return _BTreeEntry(m, k)


def btreemap_get_mut(m, k):
    if k in m:
        return _Option.some(_BTreeSlot(m, k))
    return _Option.none()


def hashset_new():
    return set()


def hashset_insert(s, v):
    before = len(s)
    s.add(v)
    return len(s) != before


def hashset_contains(s, v):
    return v in s


def hashset_remove(s, v):
    if v in s:
        s.remove(v)
        return True
    return False


def hashset_len(s):
    return len(s)


def hashset_is_empty(s):
    return len(s) == 0


def hashset_iter(s):
    return list(s)
