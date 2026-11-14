"""
Polyglot translator: Rust AST -> Python AST -> callable.

Pipeline:
    source -> Lexer -> Parser -> Program
    Program -> Translator -> Python ast.Module
    ast.Module -> compile() -> exec() -> namespace

Covers roadmap layer 3:

- literals, idents, paths
- unary, binary, assignment, compound assignment
- call, method call, field, index
- tuple, array, repeat array
- struct literal -> dataclass constructor call
- if expr -> IfExp / If statement
- match -> if/elif chain
- closure -> lambda
- range -> range() call
- cast -> runtime no-op
- macro: println!/print!/eprintln!/format!/vec! -> print/format/list
- let / return / break / continue
- fn -> FunctionDef / AsyncFunctionDef
- struct -> @dataclass ClassDef
- enum -> Enum ClassDef
- impl -> methods injected into the target ClassDef
- trait -> class with NotImplementedError stubs
- use / type alias -> skipped
- const / static -> module-level assignment

Assign / CompoundAssign are translated in statement position only.
In expression position they raise TranslateError (Rust semantics).
"""

from __future__ import annotations

import ast
import itertools
import math
import struct
import sys
from copy import copy
from functools import cmp_to_key, reduce

from ast_nodes import (
    ArrayExpr,
    Assign,
    Await,
    Binary,
    Block,
    Break,
    Call,
    Cast,
    Closure,
    CompoundAssign,
    ConstDecl,
    Continue,
    EnumDecl,
    Expr,
    ExprStmt,
    Field,
    FnDecl,
    FnParam,
    For,
    Ident,
    IfExpr,
    IfLet,
    ImplBlock,
    Index,
    Item,
    ItemStmt,
    LetStmt,
    LitBool,
    LitChar,
    LitFloat,
    LitInt,
    LitStr,
    Loop,
    MacroCall,
    MatchExpr,
    MethodCall,
    ModDecl,
    OptionNone,
    OptionSome,
    Paren,
    PatBinding,
    PatBox,
    PatConst,
    PatEnum,
    Path,
    PatIdent,
    PatLiteral,
    PatMacro,
    PatOr,
    PatPath,
    PatRange,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    Pattern,
    PatTuple,
    PatWildcard,
    Program,
    Range,
    RepeatArray,
    ResultErr,
    ResultOk,
    Return,
    Stmt,
    StructDecl,
    StructLiteral,
    TraitDecl,
    Try,
    TupleExpr,
    Type,
    TypeAlias,
    TypeArray,
    TypeDynTrait,
    TypeFn,
    TypeImplTrait,
    TypeName,
    TypeRef,
    TypeSlice,
    TypeTuple,
    TypeUnit,
    Unary,
    UseDecl,
    While,
    is_i64_type,
    is_isize_type,
    is_usize_type,
)
from backends.base import BackendBase
from backends.errors import TranslateError
from backends.python.runtime import (
    _iflet_match,
    HOST_TARGET_WIDTH,
    btreemap_contains,
    btreemap_entry,
    btreemap_get,
    btreemap_get_mut,
    btreemap_insert_with_prev,
    btreemap_is_empty,
    btreemap_iter,
    btreemap_keys,
    btreemap_len,
    btreemap_new,
    btreemap_remove,
    btreemap_values,
    cast_i8,
    cast_i16,
    cast_i32,
    cast_i64,
    cast_i128,
    cast_isize,
    cast_u8,
    cast_u16,
    cast_u32,
    cast_u64,
    cast_u128,
    cast_usize,
    get_target_width,
    hashset_contains,
    hashset_insert,
    hashset_is_empty,
    hashset_iter,
    hashset_len,
    hashset_new,
    hashset_remove,
)
from match_diagnostics import missing_match_variants, warn_match_issues
from struct_order import order_struct_declarations

_SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "object",
    "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "RuntimeError",
    "AssertionError", "IndexError", "KeyError", "StopIteration",
    "NotImplementedError", "ZeroDivisionError",
    "__build_class__",  # required for class definitions
})


def _sandbox_builtins() -> dict:
    import builtins

    return {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTINS
        if hasattr(builtins, name)
    }
# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TypeErrorForBoundary(Exception):
    """Raised when an integer crosses a pointer-width boundary implicitly."""

    code = "E001"

    def __init__(
        self,
        source_type: str,
        expected_type: str,
        line: int,
        col: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.source_type = source_type
        self.expected_type = expected_type
        self.line = line
        self.col = col
        self.span = span
        super().__init__(self.format())

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{self.code}]: type mismatch at line {self.line}, column {self.col}\n"
            f"  found:    {self.source_type}\n"
            f"  expected: {self.expected_type}\n"
            f"  help: write `as {self.expected_type}` here\n"
            "  note: `as` may wrap or truncate the value; check the range first\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "i8": "int", "i16": "int", "i32": "int", "i64": "int",
    "i128": "int", "isize": "int",
    "u8": "int", "u16": "int", "u32": "int", "u64": "int",
    "u128": "int", "usize": "int",
    "f32": "float", "f64": "float",
    "bool": "bool",
    "char": "str",
    "str": "str",
    "String": "str",
    "Vec": "list",
    "HashMap": "dict",
    "HashSet": "set",
    "Option": "object",
    "Result": "object",
}

# Bit width and signedness for Rust integer primitives. Pointer-width casts
# use the selected target width via the runtime helpers below.
_INT_TYPE_BITS: dict[str, tuple[int, bool]] = {
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

_FLOAT_TYPE_NAMES = frozenset({"f32", "f64"})
_PRIMITIVE_NUMERIC_TYPES = frozenset(_INT_TYPE_BITS) | _FLOAT_TYPE_NAMES | {"usize", "isize"}

# Each compiled program gets a unique module name so consecutive compilations
# do not clobber one another in ``sys.modules``.
_MODULE_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Macro mapping
# ---------------------------------------------------------------------------

MACRO_MAP: dict[str, str] = {
    "println": "print",
    "print": "print",
    "eprintln": "print",
    "eprint": "print",
    "format": "format",
    "vec": "list",
    "panic": "_panic",
    "assert": "assert",
    "assert_eq": "_assert_eq",
    "assert_ne": "_assert_ne",
}


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _runtime_panic(msg: str = "") -> None:
    raise RuntimeError(msg or "panic")


def _runtime_non_exhaustive_match(scrutinee_type: str) -> None:
    raise RuntimeError(
        f"non-exhaustive {scrutinee_type} match reached at runtime"
    )


def _runtime_assert_eq(a, b) -> None:
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")


def _runtime_assert_ne(a, b) -> None:
    if a == b:
        raise AssertionError(f"{a!r} == {b!r}")


def _runtime_rust_len(value) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(value)


def _runtime_rust_slice(value, start, end):
    if isinstance(value, str):
        data = value.encode("utf-8")
        start = 0 if start is None else start
        end = len(data) if end is None else end
        if start > end:
            raise ValueError(
                f"slice index starts at {start} but ends at {end}"
            )
        return data[start:end].decode("utf-8")
    start = 0 if start is None else start
    end = len(value) if end is None else end
    if start > end:
        raise ValueError(
            f"slice index starts at {start} but ends at {end}"
        )
    return value[start:end]


def _runtime_vec_retain(values: list, predicate) -> None:
    values[:] = [value for value in values if predicate(value)]


def _runtime_vec_dedup(values: list) -> None:
    if values:
        values[:] = [
            values[0],
            *(value for previous, value in itertools.pairwise(values)
              if value != previous),
        ]


def _runtime_vec_windows(values: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("window size must be positive")
    return [
        values[start:start + size]
        for start in range(max(0, len(values) - size + 1))
    ]


def _runtime_vec_chunks(values: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [
        values[start:start + size]
        for start in range(0, len(values), size)
    ]


def _runtime_filter(values, predicate):
    if isinstance(values, _Option):
        return values.filter(predicate)
    return [value for value in values if predicate(value)]


def _runtime_vec_get(values, index):
    if 0 <= index < len(values):
        return _Option.some(values[index])
    return _Option.none()


def _runtime_vec_truncate(values, length):
    if length >= 0:
        del values[length:]


def _runtime_vec_rotate(values, amount, right=False):
    if not 0 <= amount <= len(values):
        raise ValueError("rotation amount exceeds vector length")
    if amount:
        split = len(values) - amount if right else amount
        values[:] = values[split:] + values[:split]


def _runtime_vec_fill(values, value):
    for index in range(len(values)):
        values[index] = value


def _runtime_vec_swap(values, left, right):
    if not 0 <= left < len(values) or not 0 <= right < len(values):
        raise IndexError("vector swap index out of range")
    values[left], values[right] = values[right], values[left]


def _runtime_vec_extend_from_slice(values, other):
    values.extend(other)


def _runtime_vec_resize(values, length, value):
    if length < 0:
        raise ValueError("resize length must be non-negative")
    if length < len(values):
        del values[length:]
    else:
        values.extend([value] * (length - len(values)))


def _runtime_vec_split_off(values, index):
    if not 0 <= index <= len(values):
        raise IndexError("split index out of range")
    tail = values[index:]
    del values[index:]
    return tail


def _runtime_vec_partition(values, predicate):
    matching = []
    remaining = []
    for value in values:
        (matching if predicate(value) else remaining).append(value)
    return matching, remaining


class _MapSlot:
    """A mutable view of one HashMap entry.

    Rust's ``get_mut`` and entry API hand out ``&mut V``. Python values are
    not reference-like, so the slot records the map and key and writes back on
    mutation.
    """

    __slots__ = ("_key", "_mapping")

    def __init__(self, mapping, key):
        self._mapping = mapping
        self._key = key

    def get(self):
        return self._mapping[self._key]

    def set(self, value):
        self._mapping[self._key] = value
        return value

    def _apply(self, other, op):
        if isinstance(other, _MapSlot):
            other = other.get()
        self.set(op(self.get(), other))
        return self

    def __iadd__(self, other):
        return self._apply(other, lambda left, right: left + right)

    def __isub__(self, other):
        return self._apply(other, lambda left, right: left - right)

    def __imul__(self, other):
        return self._apply(other, lambda left, right: left * right)

    def __itruediv__(self, other):
        return self._apply(other, lambda left, right: left / right)

    def __imod__(self, other):
        return self._apply(other, lambda left, right: left % right)

    def __eq__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() == other

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() < other

    def __le__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() <= other

    def __gt__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() > other

    def __ge__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() >= other

    def __add__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() + other

    def __radd__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return other + self.get()

    def __sub__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() - other

    def __rsub__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return other - self.get()

    def __mul__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() * other

    def __rmul__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return other * self.get()

    def __truediv__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() / other

    def __mod__(self, other):
        if isinstance(other, _MapSlot):
            other = other.get()
        return self.get() % other

    def __int__(self):
        return int(self.get())

    def __index__(self):
        return int(self.get())

    def __bool__(self):
        return bool(self.get())

    def __hash__(self):
        return hash(self.get())

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())


class _MapEntry:
    """The subset of Rust's Entry API used by or_insert/and_modify chains."""

    __slots__ = ("_key", "_mapping")

    def __init__(self, mapping, key):
        self._mapping = mapping
        self._key = key

    def _slot(self):
        return _MapSlot(self._mapping, self._key)

    def or_insert(self, value):
        if self._key not in self._mapping:
            self._mapping[self._key] = value
        return self._slot()

    def or_insert_with(self, factory):
        if self._key not in self._mapping:
            self._mapping[self._key] = factory()
        return self._slot()

    def or_insert_with_key(self, factory):
        if self._key not in self._mapping:
            self._mapping[self._key] = factory(self._key)
        return self._slot()

    def key(self):
        return self._key

    def and_modify(self, update):
        if self._key in self._mapping:
            update(self._slot())
        return self


def _runtime_map_insert(mapping, key, value):
    if key in mapping:
        previous = mapping[key]
        mapping[key] = value
        return _Option.some(previous)
    mapping[key] = value
    return _Option.none()


def _runtime_map_get(mapping, key):
    if key in mapping:
        return _Option.some(mapping[key])
    return _Option.none()


def _runtime_map_get_mut(mapping, key):
    if key in mapping:
        return _Option.some(_MapSlot(mapping, key))
    return _Option.none()


def _runtime_map_remove(mapping, key):
    if key in mapping:
        return _Option.some(mapping.pop(key))
    return _Option.none()


def _runtime_map_keys(mapping):
    return list(mapping.keys())


def _runtime_map_values(mapping):
    return list(mapping.values())


def _runtime_map_iter(mapping):
    return list(mapping.items())


def _runtime_map_entry(mapping, key):
    return _MapEntry(mapping, key)


def _runtime_map_slot_set(slot, value):
    slot.set(value)
    return value


def _runtime_map_slot_aug(slot, value, op):
    current = slot.get()
    if op == "+":
        slot.set(current + value)
    elif op == "-":
        slot.set(current - value)
    elif op == "*":
        slot.set(current * value)
    elif op == "/":
        slot.set(_runtime_int_div(current, value))
    elif op == "%":
        slot.set(_runtime_int_rem(current, value))
    else:
        raise ValueError(f"unsupported map slot operation {op!r}")
    return slot


def _runtime_cast(value, ty, *, target_width: int | None = None):
    """Emulate Rust's `as` cast semantics.

    Integer casts truncate to the target width and reinterpret the sign bit;
    float-to-integer casts truncate toward zero and saturate like Rust's
    saturating casts; `f32` casts round through single precision. Casts to
    unknown types are identity operations.
    """
    if ty == "bool":
        return bool(value)
    if ty == "char":
        if isinstance(value, str):
            return value
        code = int(value)
        if not 0 <= code <= 0x10FFFF or 0xD800 <= code <= 0xDFFF:
            raise ValueError("char cast out of range")
        return chr(code)
    if ty in _FLOAT_TYPE_NAMES:
        number = float(value)
        if ty == "f32":
            try:
                return struct.unpack("<f", struct.pack("<f", number))[0]
            except OverflowError:
                return math.copysign(math.inf, number)
        return number
    if ty == "usize":
        width = get_target_width() if target_width is None else target_width
        return int(value) & ((1 << width) - 1)
    if ty == "isize":
        width = get_target_width() if target_width is None else target_width
        number = int(value) & ((1 << width) - 1)
        return number - (1 << width) if number >= (1 << (width - 1)) else number
    if ty == "i64" and isinstance(value, int) and not isinstance(value, bool):
        return cast_i64(value)
    spec = _INT_TYPE_BITS.get(ty)
    if spec is None:
        return value
    width, signed = spec
    if isinstance(value, str):
        number = ord(value)
    elif isinstance(value, float):
        if math.isnan(value):
            number = 0
        elif math.isinf(value):
            if signed:
                number = (
                    (1 << (width - 1)) - 1
                    if value > 0
                    else -(1 << (width - 1))
                )
            else:
                number = (1 << width) - 1 if value > 0 else 0
        else:
            number = int(value)
    else:
        number = int(value)
    modulus = 1 << width
    number %= modulus
    if signed and number >= (modulus >> 1):
        number -= modulus
    return number


def _runtime_int_div(left, right):
    """Rust integer division truncates toward zero; floats divide normally."""
    if (
        isinstance(left, int)
        and isinstance(right, int)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        quotient = abs(left) // abs(right)
        return quotient if (left < 0) == (right < 0) else -quotient
    return left / right


def _runtime_int_rem(left, right):
    """Rust `%` keeps the dividend's sign; floats use C `fmod`."""
    if (
        isinstance(left, int)
        and isinstance(right, int)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        quotient = abs(left) // abs(right)
        quotient = quotient if (left < 0) == (right < 0) else -quotient
        return left - quotient * right
    try:
        return math.fmod(left, right)
    except ValueError:
        return math.nan


def _runtime_string_repeat(value, count):
    if count < 0:
        raise ValueError("negative repeat count")
    return value * count


def _runtime_string_find(value, needle):
    position = value.find(needle)
    if position < 0:
        return _Option.none()
    return _Option.some(len(value[:position].encode("utf-8")))


def _runtime_string_rfind(value, needle):
    position = value.rfind(needle)
    if position < 0:
        return _Option.none()
    return _Option.some(len(value[:position].encode("utf-8")))


def _runtime_string_strip(value, part, *, prefix):
    if prefix:
        if not value.startswith(part):
            return _Option.none()
        return _Option.some(value[len(part):])
    if not value.endswith(part):
        return _Option.none()
    return _Option.some(value[:len(value) - len(part)])


def _runtime_string_split_once(value, separator, *, reverse):
    position = (
        value.rfind(separator) if reverse else value.find(separator)
    )
    if position < 0:
        return _Option.none()
    return _Option.some((
        value[:position],
        value[position + len(separator):],
    ))


def _runtime_string_lines(value):
    if not value:
        return []
    lines = value.split("\n")
    if lines[-1] == "":
        lines.pop()
    for index in range(len(lines) - 1):
        lines[index] = lines[index].removesuffix("\r")
    return lines


def _runtime_string_is_char_boundary(value, index):
    encoded = value.encode("utf-8")
    return (
        0 <= index <= len(encoded)
        and (
            index == len(encoded)
            or encoded[index] & 0xC0 != 0x80
        )
    )


def _runtime_string_to_ascii_lowercase(value):
    return value.translate(
        str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    )


def _runtime_string_to_ascii_uppercase(value):
    return value.translate(
        str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )


def _runtime_parse(value):
    try:
        return _Result.from_ok(int(value))
    except ValueError as exc:
        return _Result.from_err(str(exc))


class _ResultErrSentinel(Exception):
    def __init__(self, result):
        self.result = result


class _Option:
    __slots__ = ("_is_some", "_value")

    def __init__(self, is_some, value=None):
        self._is_some = is_some
        self._value = value

    @classmethod
    def some(cls, value):
        return cls(True, value)

    @classmethod
    def none(cls):
        return cls(False, None)

    def is_some(self):
        return self._is_some

    def is_none(self):
        return not self._is_some

    def unwrap(self):
        if not self._is_some:
            raise ValueError("called `Option::unwrap()` on a `None` value")
        return self._value

    def unwrap_or(self, default):
        return self._value if self._is_some else default

    def unwrap_or_else(self, f):
        return self._value if self._is_some else f()

    def and_then(self, f):
        return f(self._value) if self._is_some else _Option.none()

    def or_else(self, f):
        return self if self._is_some else f()

    def filter(self, predicate):
        return self if self._is_some and predicate(self._value) else _Option.none()

    def ok_or(self, error):
        return (
            _Result.from_ok(self._value)
            if self._is_some
            else _Result.from_err(error)
        )

    def ok_or_else(self, error_factory):
        return (
            _Result.from_ok(self._value)
            if self._is_some
            else _Result.from_err(error_factory())
        )

    def __eq__(self, other):
        if not isinstance(other, _Option):
            return NotImplemented
        if self._is_some != other._is_some:
            return False
        return self._value == other._value

    def __repr__(self):
        if self._is_some:
            return f"Some({self._value!r})"
        return "None"

class _Result:
    __slots__ = ("_is_ok", "_value")

    def __init__(self, is_ok, value=None):
        self._is_ok = is_ok
        self._value = value

    @classmethod
    def from_ok(cls, value):
        return cls(True, value)

    @classmethod
    def from_err(cls, value):
        return cls(False, value)

    def is_ok(self):
        return self._is_ok

    def is_err(self):
        return not self._is_ok

    def ok(self):
        return _Option.some(self._value) if self._is_ok else _Option.none()

    def err(self):
        return _Option.none() if self._is_ok else _Option.some(self._value)

    def and_then(self, f):
        return f(self._value) if self._is_ok else self

    def or_else(self, f):
        return self if self._is_ok else f(self._value)

    def __try__(self):
        if not self._is_ok:
            raise _ResultErrSentinel(self)
        return self._value
    def map_err(self, f):
        if self._is_ok:
            return self
        return _Result.from_err(f(self._value))
    def unwrap(self):
        if not self._is_ok:
            raise RuntimeError(f"called `Result::unwrap()` on an `Err` value: {self._value!r}")
        return self._value

    def unwrap_or(self, default):
        return self._value if self._is_ok else default

    def unwrap_or_else(self, f):
        return self._value if self._is_ok else f(self._value)
    def __eq__(self, other):
        if not isinstance(other, _Result):
            return NotImplemented
        if self._is_ok != other._is_ok:
            return False
        return self._value == other._value

    def __repr__(self):
        if self._is_ok:
            return f"Ok({self._value!r})"
        return f"Err({self._value!r})"


class _PythonEnumVariant:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *values):
        return (self, *values)


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_dataclass():
    from dataclasses import dataclass
    return dataclass


def _import_enum():
    from enum import Enum
    return Enum


def _import_auto():
    from enum import auto
    return auto


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class Translator:
    def __init__(self) -> None:
        self.namespace: dict = {}
        self.struct_names: set[str] = set()
        self.struct_fields: dict[str, set[str]] = {}
        self.struct_field_types: dict[str, dict[str, str]] = {}
        self._current_impl_type: str | None = None
        self.enum_names: set[str] = set()
        self.enum_variants: dict[str, set[str]] = {}
        self.constant_names: set[str] = set()
        self.impl_blocks: dict[str, list[ImplBlock]] = {}
        self._function_signatures: dict[str, list[str | None]] = {}
        self._method_return_types: dict[tuple[str, str], str | None] = {}
        self.generic_struct_names: set[str] = set()
        self.generic_type_names: set[str] = set()
        self._temp_counter = 0
        self._binding_types: dict[str, str] = {}
        self._slot_bindings: set[str] = set()
        self._numeric_types: dict[str, str] = {}
        self._return_numeric_type: str | None = None
        self._return_type_name: str | None = None
        self.target_width: int = HOST_TARGET_WIDTH

    # ---- entry -------------------------------------------------------------

    def translate_program(self, program: Program) -> ast.Module:
        self.struct_names.clear()
        self.struct_fields.clear()
        self.struct_field_types.clear()
        self._current_impl_type = None
        self.enum_names.clear()
        self.enum_variants.clear()
        self.constant_names.clear()
        self.generic_struct_names.clear()
        self.generic_type_names.clear()
        self._temp_counter = 0
        self._binding_types.clear()
        self._slot_bindings.clear()
        self._numeric_types.clear()
        self._return_numeric_type = None
        self._return_type_name = None
        self._function_signatures.clear()
        self._method_return_types.clear()
        self.impl_blocks.clear()
        # Pass 1: collect struct / enum / impl names.
        for item in program.items:
            if isinstance(item, StructDecl):
                self.struct_names.add(item.name)
                self.struct_fields[item.name] = {
                    name for name, _ty, _is_pub in item.fields
                }
                self.struct_field_types[item.name] = {
                    name: ty.name
                    for name, ty, _is_pub in item.fields
                    if isinstance(ty, TypeName)
                }
                if item.generics:
                    self.generic_struct_names.add(item.name)
                    self.generic_type_names.update(
                        param.name for param in item.generics
                    )
            elif isinstance(item, EnumDecl):
                self.enum_names.add(item.name)
                self.enum_variants[item.name] = {
                    variant.name for variant in item.variants
                }
            elif isinstance(item, ConstDecl):
                self.constant_names.add(item.name)
            elif isinstance(item, FnDecl):
                self._function_signatures[item.name] = [
                    self._binding_type_name(param.ty)
                    for param in item.params
                ]
            elif isinstance(item, ImplBlock):
                target = self._type_name_of(item.ty)
                self.impl_blocks.setdefault(target, []).append(item)
                for method in item.items:
                    if isinstance(method, FnDecl):
                        self._method_return_types[
                            (target, method.name)
                        ] = self._binding_type_name(method.ret)

        # Pass 2: emit statements. impl blocks are consumed here.
        body: list[ast.stmt] = []
        emitted_classes: set[str] = set()
        struct_items = [
            item for item in program.items if isinstance(item, StructDecl)
        ]
        for item in order_struct_declarations(struct_items):
            body.append(self.translate_struct(item))
            emitted_classes.add(item.name)

        for item in program.items:
            if isinstance(item, ImplBlock):
                continue
            if isinstance(item, StructDecl):
                continue
            elif isinstance(item, EnumDecl):
                body.append(self.translate_enum(item))
                emitted_classes.add(item.name)
            else:
                body.extend(self.translate_item(item))

        # Orphan impl blocks.
        for target, impls in self.impl_blocks.items():
            if target in emitted_classes:
                continue
            methods: list[ast.stmt] = []
            for impl in impls:
                methods.extend(self._impl_methods(impl))
            if not methods:
                methods = [ast.Pass()]
            body.append(ast.ClassDef(
                name=target, bases=[], keywords=[], body=methods,
                decorator_list=[],
            ))

        return ast.fix_missing_locations(
            ast.Module(body=body, type_ignores=[])
        )

    def compile_rust(
        self,
        program: Program,
        *,
        sandbox: bool = False,
        target_width: int | None = None,
    ) -> dict:
        import types

        if target_width is not None and target_width not in (32, 64):
            raise ValueError(
                f"target width must be 32 or 64, got {target_width}"
            )
        self.target_width = HOST_TARGET_WIDTH if target_width is None else target_width
        module = self.translate_program(program)
        code = compile(module, "<polyglot>", "exec")
        module_name = f"_polyglot_generated_{next(_MODULE_COUNTER)}"
        ns: dict = {
            "__name__": module_name,
            "dataclass": _import_dataclass(),
            "Enum": _import_enum(),
            "auto": _import_auto(),
        }
        ns["_panic"] = _runtime_panic
        ns["_runtime_non_exhaustive_match"] = _runtime_non_exhaustive_match
        ns["_assert_eq"] = _runtime_assert_eq
        ns["_assert_ne"] = _runtime_assert_ne
        ns["_rust_len"] = _runtime_rust_len
        ns["_rust_slice"] = _runtime_rust_slice
        ns["_Option"] = _Option
        ns["_iflet_match"] = _iflet_match
        ns["_Result"] = _Result
        ns["_ResultErrSentinel"] = _ResultErrSentinel
        ns["_PythonEnumVariant"] = _PythonEnumVariant
        ns["_clone"] = copy
        ns["_cmp_to_key"] = cmp_to_key
        ns["_reduce"] = reduce
        ns["_vec_retain"] = _runtime_vec_retain
        ns["_vec_dedup"] = _runtime_vec_dedup
        ns["_vec_windows"] = _runtime_vec_windows
        ns["_vec_chunks"] = _runtime_vec_chunks
        ns["_filter"] = _runtime_filter
        ns["_vec_get"] = _runtime_vec_get
        ns["_vec_truncate"] = _runtime_vec_truncate
        ns["_vec_rotate"] = _runtime_vec_rotate
        ns["_vec_fill"] = _runtime_vec_fill
        ns["_vec_swap"] = _runtime_vec_swap
        ns["_vec_extend_from_slice"] = _runtime_vec_extend_from_slice
        ns["_vec_resize"] = _runtime_vec_resize
        ns["_vec_split_off"] = _runtime_vec_split_off
        ns["_vec_partition"] = _runtime_vec_partition
        ns["_string_repeat"] = _runtime_string_repeat
        ns["_string_find"] = _runtime_string_find
        ns["_string_rfind"] = _runtime_string_rfind
        ns["_string_strip"] = _runtime_string_strip
        ns["_string_split_once"] = _runtime_string_split_once
        ns["_string_lines"] = _runtime_string_lines
        ns["_string_is_char_boundary"] = _runtime_string_is_char_boundary
        ns["_string_to_ascii_lowercase"] = _runtime_string_to_ascii_lowercase
        ns["_string_to_ascii_uppercase"] = _runtime_string_to_ascii_uppercase
        ns["_result_parse"] = _runtime_parse
        ns["_map_insert"] = _runtime_map_insert
        ns["_map_get"] = _runtime_map_get
        ns["_map_get_mut"] = _runtime_map_get_mut
        ns["_map_remove"] = _runtime_map_remove
        ns["_map_keys"] = _runtime_map_keys
        ns["_map_values"] = _runtime_map_values
        ns["_map_iter"] = _runtime_map_iter
        ns["_map_entry"] = _runtime_map_entry
        ns["_map_slot_set"] = _runtime_map_slot_set
        ns["_map_slot_aug"] = _runtime_map_slot_aug
        # M3 helpers
        ns["_btreemap_new"] = btreemap_new
        ns["_btreemap_insert"] = btreemap_insert_with_prev
        ns["_btreemap_get"] = btreemap_get
        ns["_btreemap_remove"] = btreemap_remove
        ns["_btreemap_contains"] = btreemap_contains
        ns["_btreemap_len"] = btreemap_len
        ns["_btreemap_is_empty"] = btreemap_is_empty
        ns["_btreemap_keys"] = btreemap_keys
        ns["_btreemap_values"] = btreemap_values
        ns["_btreemap_iter"] = btreemap_iter
        ns["_btreemap_entry"] = btreemap_entry
        ns["_btreemap_get_mut"] = btreemap_get_mut
        ns["_hashset_new"] = hashset_new
        ns["_hashset_insert"] = hashset_insert
        ns["_hashset_contains"] = hashset_contains
        ns["_hashset_remove"] = hashset_remove
        ns["_hashset_len"] = hashset_len
        ns["_hashset_is_empty"] = hashset_is_empty
        ns["_hashset_iter"] = hashset_iter
        if target_width is None:
            ns["_cast"] = _runtime_cast
            ns["_cast_isize"] = cast_isize
            ns["_cast_usize"] = cast_usize
        else:
            ns["_cast"] = lambda value, ty: _runtime_cast(
                value, ty, target_width=target_width,
            )
            ns["_cast_isize"] = lambda value: cast_isize(
                value, target_width=target_width,
            )
            ns["_cast_usize"] = lambda value: cast_usize(
                value, target_width=target_width,
            )
        for _name, _fn in {
            "i8": cast_i8, "i16": cast_i16, "i32": cast_i32, "i64": cast_i64, "i128": cast_i128,
            "u8": cast_u8, "u16": cast_u16, "u32": cast_u32, "u64": cast_u64, "u128": cast_u128,
        }.items():
            ns[f"_cast_{_name}"] = _fn
        ns["_div"] = _runtime_int_div
        ns["_mod"] = _runtime_int_rem

        # Register the namespace as a module so @dataclass can look it up
        # via sys.modules[cls.__module__]. A unique name avoids clobbering
        # previously compiled programs.
        mod = types.ModuleType(module_name)
        mod.__dict__.update(ns)
        mod.__dict__["__name__"] = module_name
        sys.modules[module_name] = mod
        if sandbox:
            mod.__dict__["__builtins__"] = _sandbox_builtins()
        exec(code, mod.__dict__)  # noqa: S102
        self.namespace = mod.__dict__
        return mod.__dict__

    # ---- items -------------------------------------------------------------

    def translate_item(self, item: Item) -> list[ast.stmt]:
        if isinstance(item, FnDecl):
            return [self.translate_fn(item)]
        if isinstance(item, StructDecl):
            return [self.translate_struct(item)]
        if isinstance(item, EnumDecl):
            return [self.translate_enum(item)]
        if isinstance(item, ImplBlock):
            target = self._type_name_of(item.ty)
            methods: list[ast.stmt] = self._impl_methods(item)
            if not methods:
                methods = [ast.Pass()]
            cls = ast.ClassDef(
                name=target, bases=[], keywords=[], body=methods,
                decorator_list=[],
            )
            out: list[ast.stmt] = [cls]
            return out
        if isinstance(item, ConstDecl):
            return [self.translate_const(item)]
        if isinstance(item, UseDecl):
            return []
        if isinstance(item, ModDecl):
            out2: list[ast.stmt] = []
            for sub in item.items or []:
                out2.extend(self.translate_item(sub))
            return out2
        if isinstance(item, TypeAlias):
            return []
        if isinstance(item, TraitDecl):
            return [self.translate_trait(item)]
        raise TranslateError(
            f"unsupported item {type(item).__name__}",
            getattr(item, "line", 0), getattr(item, "col", 0),
        )

    # ---- fn ----------------------------------------------------------------

    def translate_fn(
        self, fn: FnDecl, *, is_method: bool = False,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        from termination import check_function

        check_function(fn)
        params: list[ast.arg] = [self.translate_fn_param(p) for p in fn.params]
        has_self = bool(params) and params[0].arg == "self"
        if is_method and not has_self:
            params.insert(0, ast.arg(arg="self", annotation=None))
        args = ast.arguments(
            posonlyargs=[], args=params, vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        )
        body: list[ast.stmt] = []
        previous_binding_types = self._binding_types
        self._binding_types = {
            param.pattern.name: name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            for name in [self._binding_type_name(param.ty)]
            if name is not None
        }
        previous_slot_bindings = self._slot_bindings
        self._slot_bindings = set(previous_slot_bindings)
        previous_numeric_types = self._numeric_types
        self._numeric_types = {
            param.pattern.name: numeric
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            for numeric in [self._numeric_type_name(param.ty)]
            if numeric is not None
        }
        previous_return_numeric = self._return_numeric_type
        self._return_numeric_type = self._numeric_type_name(fn.ret)
        previous_return_type_name = self._return_type_name
        self._return_type_name = self._binding_type_name(fn.ret)
        try:
            if fn.body is not None:
                body = self.translate_block_as_fn_body(fn.body)
        finally:
            self._binding_types = previous_binding_types
            self._slot_bindings = previous_slot_bindings
            self._numeric_types = previous_numeric_types
            self._return_numeric_type = previous_return_numeric
            self._return_type_name = previous_return_type_name
        if not body:
            body = [ast.Pass()]
        if (
            isinstance(fn.ret, TypeName)
            and fn.ret.name == "Result"
            and len(fn.ret.args) == 2
        ):
            body = [ast.Try(
                body=body,
                handlers=[ast.ExceptHandler(
                    type=ast.Name(
                        id="_ResultErrSentinel", ctx=ast.Load(),
                    ),
                    name="_polyglot_result_error",
                    body=[ast.Return(value=ast.Attribute(
                        value=ast.Name(
                            id="_polyglot_result_error", ctx=ast.Load(),
                        ),
                        attr="result",
                        ctx=ast.Load(),
                    ))],
                )],
                orelse=[],
                finalbody=[],
            )]
        ret: ast.expr | None = (
            self.translate_type(fn.ret) if fn.ret is not None else None
        )
        decorators: list[ast.expr] = []
        # impl method without self -> associated function, expose as
        # @staticmethod so `Point.new(x, y)` binds correctly.
        if is_method and not has_self:
            decorators.append(
                ast.Name(id="staticmethod", ctx=ast.Load())
            )
        if fn.is_async:
            return ast.AsyncFunctionDef(
                name=fn.name, args=args, body=body,
                decorator_list=decorators, returns=ret,
                type_comment=None,
            )
        return ast.FunctionDef(
            name=fn.name, args=args, body=body,
            decorator_list=decorators, returns=ret,
            type_comment=None,
        )

    def translate_fn_param(self, p: FnParam) -> ast.arg:
        pat = p.pattern
        if isinstance(pat, PatRef):
            pat = pat.inner
        if isinstance(pat, PatIdent):
            name = pat.name
        elif isinstance(pat, PatWildcard):
            name = "_"
        else:
            raise TranslateError(
                f"unsupported fn parameter pattern {type(pat).__name__}",
                p.line, p.col,
            )
        if name == "self":
            return ast.arg(arg="self", annotation=None)
        return ast.arg(arg=name, annotation=self._type_or_object(p.ty))

    def translate_block_as_fn_body(self, block: Block) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        for stmt in block.stmts:
            body.extend(self.translate_stmt(stmt))
        if block.tail is not None:
            self._check_type_boundary(
                self._known_expr_type(block.tail),
                self._return_type_name,
                block.tail.line,
                block.tail.col,
            )
            body.append(ast.Return(
                value=self._coerce_return_value(
                    self.translate_expr(block.tail),
                ),
            ))
        return body

    # ---- struct ------------------------------------------------------------

    def translate_struct(self, s: StructDecl) -> ast.ClassDef:
        decorators: list[ast.expr] = [ast.Name(id="dataclass", ctx=ast.Load())]
        body: list[ast.stmt] = []
        for fname, fty, _pub in s.fields:
            body.append(ast.AnnAssign(
                target=ast.Name(id=fname, ctx=ast.Store()),
                annotation=self._type_or_object(fty),
                value=None,
                simple=1,
            ))
        for impl in self.impl_blocks.get(s.name, []):
            body.extend(self._impl_methods(impl))
        if not body:
            body = [ast.Pass()]
        return ast.ClassDef(
            name=s.name, bases=[], keywords=[], body=body,
            decorator_list=decorators,
        )

    def _impl_methods(self, impl: ImplBlock) -> list[ast.stmt]:
        previous_impl_type = self._current_impl_type
        self._current_impl_type = self._type_name_of(impl.ty)
        try:
            methods: list[ast.stmt] = []
            for sub in impl.items:
                if isinstance(sub, FnDecl):
                    methods.append(
                        self.translate_fn(
                            sub, is_method=self._fn_is_method(sub),
                        )
                    )
            return methods
        finally:
            self._current_impl_type = previous_impl_type

    def _fn_is_method(self, fn: FnDecl) -> bool:
        if not fn.params:
            return False
        pat = fn.params[0].pattern
        if isinstance(pat, PatRef):
            pat = pat.inner
        return isinstance(pat, PatIdent) and pat.name == "self"

    # ---- enum --------------------------------------------------------------

    def translate_enum(self, e: EnumDecl) -> ast.ClassDef:
        body: list[ast.stmt] = []
        has_payloads = any(v.fields is not None for v in e.variants)
        for v in e.variants:
            body.append(ast.Assign(
                targets=[
                    ast.Name(
                        id=self._python_identifier(v.name),
                        ctx=ast.Store(),
                    ),
                ],
                value=(
                    ast.Call(
                        func=ast.Name(
                            id="_PythonEnumVariant", ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=v.name)],
                        keywords=[],
                    )
                    if has_payloads
                    else ast.Call(
                        func=ast.Name(id="auto", ctx=ast.Load()),
                        args=[], keywords=[],
                    )
                ),
            ))
        for impl in self.impl_blocks.get(e.name, []):
            body.extend(self._impl_methods(impl))
        if not body:
            body = [ast.Pass()]
        bases: list[ast.expr] = (
            [] if has_payloads else [ast.Name(id="Enum", ctx=ast.Load())]
        )
        return ast.ClassDef(
            name=e.name, bases=bases, keywords=[], body=body,
            decorator_list=[],
        )

    # ---- impl (standalone) -------------------------------------------------

    def translate_impl(self, impl: ImplBlock) -> list[ast.stmt]:
        target = self._type_name_of(impl.ty)
        methods: list[ast.stmt] = self._impl_methods(impl)
        if not methods:
            methods = [ast.Pass()]
        cls = ast.ClassDef(
            name=target, bases=[], keywords=[], body=methods,
            decorator_list=[],
        )
        out: list[ast.stmt] = [cls]
        return out

    def _type_name_of(self, ty: Type) -> str:
        if isinstance(ty, TypeName):
            return ty.name
        if isinstance(ty, TypeRef):
            return self._type_name_of(ty.inner)
        raise TranslateError(
            f"cannot get type name from {type(ty).__name__}",
            getattr(ty, "line", 0), getattr(ty, "col", 0),
        )

    def _numeric_type_name(self, ty: Type | None) -> str | None:
        """Return the primitive numeric type name for ``ty``, if any."""
        if isinstance(ty, TypeRef):
            return self._numeric_type_name(ty.inner)
        if isinstance(ty, TypeName) and ty.name in _PRIMITIVE_NUMERIC_TYPES:
            return ty.name
        return None

    def _binding_type_name(self, ty: Type | None) -> str | None:
        """Return the named type behind a parameter type, if any."""
        if isinstance(ty, TypeRef):
            return self._binding_type_name(ty.inner)
        if isinstance(ty, TypeName):
            return ty.name
        return None

    def _coerce_numeric(self, value: ast.expr, ty: str) -> ast.expr:
        return ast.Call(
            func=ast.Name(id="_cast", ctx=ast.Load()),
            args=[value, ast.Constant(value=ty)],
            keywords=[],
        )

    def _binary_with_op(
        self, op: str, left: ast.expr, right: ast.expr,
    ) -> ast.expr:
        builtin: dict[str, type[ast.operator]] = {
            "+": ast.Add,
            "-": ast.Sub,
            "*": ast.Mult,
        }
        if op in builtin:
            return ast.BinOp(left=left, op=builtin[op](), right=right)
        helper = {"/": "_div", "%": "_mod"}.get(op)
        if helper is not None:
            return ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=[left, right],
                keywords=[],
            )
        raise TranslateError(f"unknown binary op {op}")

    # ---- trait -------------------------------------------------------------

    def translate_trait(self, t: TraitDecl) -> ast.ClassDef:
        body: list[ast.stmt] = []
        for item in t.items:
            if item.fn is not None:
                fn = self.translate_fn(item.fn, is_method=True)
                fn.body = [ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                        args=[], keywords=[],
                    ),
                    cause=None,
                )]
                body.append(fn)
        if not body:
            body = [ast.Pass()]
        return ast.ClassDef(
            name=t.name, bases=[], keywords=[], body=body,
            decorator_list=[],
        )

    # ---- const / static ----------------------------------------------------

    def translate_const(self, c: ConstDecl) -> ast.Assign:
        value = self.translate_expr(c.value)
        return ast.Assign(
            targets=[ast.Name(id=c.name, ctx=ast.Store())],
            value=value,
        )

    # ---- statements --------------------------------------------------------

    def translate_stmt(self, s: Stmt) -> list[ast.stmt]:
        if isinstance(s, LetStmt):
            return [self.translate_let(s)]
        if isinstance(s, ExprStmt):
            expr = s.expr
            if isinstance(expr, Assign):
                return [self._translate_assign_stmt(expr)]
            if isinstance(expr, CompoundAssign):
                return [self._translate_compound_assign_stmt(expr)]
            if isinstance(expr, MacroCall) and expr.path == "write":
                return [self._translate_write_macro(expr)]
            if isinstance(expr, IfExpr):
                return [self.translate_if_stmt(expr)]
            if isinstance(expr, IfLet):
                return self.translate_if_let_stmt(expr)
            if isinstance(expr, While):
                return [self.translate_while(expr)]
            if isinstance(expr, For):
                return [self.translate_for(expr)]
            if isinstance(expr, Loop):
                return [self.translate_loop(expr)]
            if isinstance(expr, MatchExpr):
                return [self.translate_match_stmt(expr)]
            if isinstance(expr, Return):
                if expr.value is not None:
                    self._check_type_boundary(
                        self._known_expr_type(expr.value),
                        self._return_type_name,
                        expr.line,
                        expr.col,
                    )
                return [ast.Return(
                    value=self._coerce_return_value(
                        self.translate_opt_expr(expr.value),
                    ),
                )]
            if isinstance(expr, Break):
                return [ast.Break()]
            if isinstance(expr, Continue):
                return [ast.Continue()]
            if isinstance(expr, Block):
                return self.translate_block_as_body(expr)
            return [ast.Expr(value=self.translate_expr(expr))]
        if isinstance(s, ItemStmt):
            return self.translate_item(s.item)
        raise TranslateError(
            f"unsupported statement {type(s).__name__}",
            getattr(s, "line", 0), getattr(s, "col", 0),
        )

    def _translate_write_macro(self, macro: MacroCall) -> ast.AugAssign:
        if not isinstance(macro.args, list) or len(macro.args) < 2:
            raise TranslateError(
                "write! expects a mutable string and format string",
                macro.line, macro.col,
            )
        target_expr = macro.args[0]
        if isinstance(target_expr, Unary) and target_expr.op in ("&", "&mut"):
            target_expr = target_expr.operand
        if not isinstance(target_expr, Ident):
            raise TranslateError(
                "write! target must be an identifier",
                macro.line, macro.col,
            )
        format_value = self.translate_expr(macro.args[1])
        values = [self.translate_expr(arg) for arg in macro.args[2:]]
        if values:
            format_value = ast.Call(
                func=ast.Attribute(
                    value=format_value, attr="format", ctx=ast.Load(),
                ),
                args=values,
                keywords=[],
            )
        return ast.AugAssign(
            target=ast.Name(id=target_expr.name, ctx=ast.Store()),
            op=ast.Add(),
            value=format_value,
        )

    def _translate_assign_stmt(self, e: Assign) -> ast.stmt:
        value = self.translate_expr(e.value)
        if isinstance(e.target, Unary) and e.target.op == "*":
            if not self._is_map_slot_expr(e.target.operand):
                raise TranslateError(
                    "dereference assignment target is not a mutable map slot",
                    e.line, e.col,
                )
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_map_slot_set", ctx=ast.Load()),
                    args=[self.translate_expr(e.target.operand), value],
                    keywords=[],
                ),
            )
        if (
            isinstance(e.target, Ident)
            and e.target.name in self._numeric_types
        ):
            self._check_type_boundary(
                self._known_expr_type(e.value),
                self._numeric_types[e.target.name],
                e.line,
                e.col,
            )
            value = self._coerce_numeric(
                value, self._numeric_types[e.target.name],
            )
        target = self._translate_lvalue(e.target)
        return ast.Assign(targets=[target], value=value)

    def _translate_compound_assign_stmt(
        self, e: CompoundAssign,
    ) -> ast.stmt:
        value = self.translate_expr(e.value)
        op_map: dict[str, type[ast.operator]] = {
            "+=": ast.Add, "-=": ast.Sub,
            "*=": ast.Mult, "/=": ast.Div, "%=": ast.Mod,
        }
        if e.op not in op_map:
            raise TranslateError(
                f"unsupported compound assign op {e.op!r}",
                e.line, e.col,
            )
        if isinstance(e.target, Unary) and e.target.op == "*":
            if not self._is_map_slot_expr(e.target.operand):
                raise TranslateError(
                    "dereference assignment target is not a mutable map slot",
                    e.line, e.col,
                )
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_map_slot_aug", ctx=ast.Load()),
                    args=[
                        self.translate_expr(e.target.operand),
                        value,
                        ast.Constant(value=e.op[0]),
                    ],
                    keywords=[],
                ),
            )
        if (
            isinstance(e.target, Ident)
            and e.target.name in self._numeric_types
        ):
            combined = self._binary_with_op(
                e.op[0],
                ast.Name(id=e.target.name, ctx=ast.Load()),
                value,
            )
            return ast.Assign(
                targets=[ast.Name(id=e.target.name, ctx=ast.Store())],
                value=self._coerce_numeric(
                    combined, self._numeric_types[e.target.name],
                ),
            )
        target = self._translate_lvalue(e.target)
        return ast.AugAssign(target=target, op=op_map[e.op](), value=value)

    def translate_block_as_body(self, block: Block) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        for stmt in block.stmts:
            body.extend(self.translate_stmt(stmt))
        if block.tail is not None:
            body.append(ast.Expr(value=self.translate_expr(block.tail)))
        if not body:
            body = [ast.Pass()]
        return body

    def _binding_type_with_nested(self, ty: TypeName) -> str:
        """Encode a TypeName preserving nested collection kinds (one level)."""
        if ty.name in {"HashMap", "BTreeMap"} and len(ty.args) >= 2:
            value_ty = ty.args[1]
            if isinstance(value_ty, TypeName) and value_ty.name in {
                "HashMap", "BTreeMap", "HashSet", "Vec",
            }:
                return f"{ty.name}:{value_ty.name}"
        return ty.name

    def translate_let(self, s: LetStmt) -> ast.stmt:
        if not isinstance(s.pattern, PatIdent):
            raise TranslateError(
                f"unsupported let pattern {type(s.pattern).__name__}",
                s.line, s.col,
            )
        target = ast.Name(id=s.pattern.name, ctx=ast.Store())
        numeric_type = self._numeric_type_name(s.ty)
        inferred_type = (
            self._known_expr_type(s.value) if s.value is not None else None
        )
        self._check_type_boundary(
            inferred_type,
            self._binding_type_name(s.ty),
            s.line,
            s.col,
            getattr(s.value, "span", None) or getattr(s, "span", None),
        )
        if numeric_type is None and inferred_type in _PRIMITIVE_NUMERIC_TYPES:
            numeric_type = inferred_type
        if numeric_type is not None:
            self._numeric_types[s.pattern.name] = numeric_type
        else:
            self._numeric_types.pop(s.pattern.name, None)
        if s.value is None:
            return ast.AnnAssign(
                target=target,
                annotation=self._type_or_object(s.ty),
                value=None, simple=1,
            )
        value = self.translate_expr(s.value)
        if numeric_type is not None:
            value = self._coerce_numeric(value, numeric_type)
        binding_type = (
            self._binding_type_with_nested(s.ty)
            if isinstance(s.ty, TypeName)
            else self._expr_variant_type(s.value)
        )
        if (
            binding_type is None
            and s.value is not None
            and self._is_vec_expr(s.value)
        ):
            binding_type = "Vec"
        if binding_type in {"Option", "Result", "HashMap"} or binding_type is not None:
            self._binding_types[s.pattern.name] = binding_type
        if self._is_map_slot_expr(s.value):
            self._slot_bindings.add(s.pattern.name)
        else:
            self._slot_bindings.discard(s.pattern.name)
        return ast.Assign(targets=[target], value=value)

    def translate_while(self, w: While) -> ast.While:
        test = self.translate_expr(w.cond)
        body = self.translate_block_as_body(w.body)
        return ast.While(test=test, body=body, orelse=[])

    def translate_for(self, f: For) -> ast.For:
        if isinstance(f.pattern, PatIdent):
            target: ast.expr = ast.Name(
                id=f.pattern.name, ctx=ast.Store(),
            )
        elif isinstance(f.pattern, PatTuple):
            target = ast.Tuple(
                elts=[self._for_pattern_target(p) for p in f.pattern.elems],
                ctx=ast.Store(),
            )
        else:
            raise TranslateError(
                f"unsupported for pattern {type(f.pattern).__name__}",
                f.line, f.col,
            )
        iter_expr = self.translate_expr(f.iter)
        if isinstance(f.pattern, PatIdent):
            name = f.pattern.name
            previous_numeric_type = self._numeric_types.get(name)
            previous_binding_type = self._binding_types.get(name)
            loop_type = self._known_expr_type(f.iter)
            self._numeric_types.pop(name, None)
            self._binding_types.pop(name, None)
            if loop_type in _PRIMITIVE_NUMERIC_TYPES:
                self._numeric_types[name] = loop_type
                self._binding_types[name] = loop_type
            elif loop_type is not None:
                self._binding_types[name] = loop_type
            try:
                body = self.translate_block_as_body(f.body)
            finally:
                self._restore_binding(self._numeric_types, name, previous_numeric_type)
                self._restore_binding(self._binding_types, name, previous_binding_type)
        else:
            body = self.translate_block_as_body(f.body)
        return ast.For(target=target, iter=iter_expr, body=body, orelse=[])

    @staticmethod
    def _restore_binding(
        bindings: dict[str, str],
        name: str,
        value: str | None,
    ) -> None:
        if value is None:
            bindings.pop(name, None)
        else:
            bindings[name] = value

    def _for_pattern_target(self, pattern: Pattern) -> ast.expr:
        if isinstance(pattern, PatIdent):
            return ast.Name(id=pattern.name, ctx=ast.Store())
        if isinstance(pattern, PatTuple):
            return ast.Tuple(
                elts=[self._for_pattern_target(p) for p in pattern.elems],
                ctx=ast.Store(),
            )
        raise TranslateError(
            f"unsupported for pattern {type(pattern).__name__}",
            pattern.line, pattern.col,
        )

    def translate_loop(self, lp: Loop) -> ast.While:
        body = self.translate_block_as_body(lp.body)
        return ast.While(test=ast.Constant(value=True), body=body, orelse=[])

    def translate_if_stmt(self, ifx: IfExpr) -> ast.If:
        test = self.translate_expr(ifx.cond)
        body = self.translate_block_as_body(ifx.then_block)
        orelse: list[ast.stmt] = []
        if isinstance(ifx.else_branch, Block):
            orelse = self.translate_block_as_body(ifx.else_branch)
        elif isinstance(ifx.else_branch, IfExpr):
            orelse = [self.translate_if_stmt(ifx.else_branch)]
        return ast.If(test=test, body=body, orelse=orelse)

    def _translate_iflet_expr(self, ifx: IfLet) -> ast.expr:
        if not (
            isinstance(ifx.pattern, PatEnum)
            and ifx.pattern.path.split("::")[-1] == "Some"
            and len(ifx.pattern.args) == 1
            and isinstance(ifx.pattern.args[0], PatIdent)
        ):
            raise TranslateError(
                "if-let expression currently supports `Some(binding)` only",
                ifx.pattern.line, ifx.pattern.col,
            )
        binding = ifx.pattern.args[0]
        if ifx.then_block.tail is None or ifx.then_block.stmts:
            raise TranslateError(
                "if-let expression requires a single tail expression in the then-branch",
                ifx.then_block.line, ifx.then_block.col,
            )
        if isinstance(ifx.else_branch, Block) and (
            ifx.else_branch.tail is None or ifx.else_branch.stmts
        ):
            raise TranslateError(
                "if-let expression requires a single tail expression in the else-branch",
                ifx.else_branch.line, ifx.else_branch.col,
            )
        then_expr = self.translate_expr(ifx.then_block.tail)
        else_expr = self._translate_else_expr(ifx.else_branch)
        return ast.Call(
            func=ast.Name(id="_iflet_match", ctx=ast.Load()),
            args=[
                self.translate_expr(ifx.value),
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[], args=[ast.arg(arg=binding.name)],
                        kwonlyargs=[], kw_defaults=[], defaults=[],
                    ),
                    body=then_expr,
                ),
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[], args=[],
                        kwonlyargs=[], kw_defaults=[], defaults=[],
                    ),
                    body=else_expr,
                ),
            ],
            keywords=[],
        )

    def translate_if_let_stmt(self, ifx: IfLet) -> list[ast.stmt]:
        temp_name = f"_polyglot_if_let_{self._temp_counter}"
        self._temp_counter += 1
        temp = ast.Name(id=temp_name, ctx=ast.Load())
        if isinstance(ifx.pattern, PatEnum):
            variant = ifx.pattern.path.split("::")[-1]
            if variant != "Some" or len(ifx.pattern.args) != 1:
                raise TranslateError(
                    "if let currently supports Some(binding)",
                    ifx.pattern.line, ifx.pattern.col,
                )
            binding = ifx.pattern.args[0]
            if not isinstance(binding, PatIdent):
                raise TranslateError(
                    "if let Some pattern requires an identifier binding",
                    binding.line, binding.col,
                )
            test = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Call(
                        func=ast.Name(id="isinstance", ctx=ast.Load()),
                        args=[temp, ast.Name(id="_Option", ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Call(
                        func=ast.Attribute(
                            value=temp, attr="is_some", ctx=ast.Load(),
                        ),
                        args=[], keywords=[],
                    ),
                ],
            )
            body = [ast.Assign(
                targets=[ast.Name(id=binding.name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=temp, attr="unwrap", ctx=ast.Load(),
                    ),
                    args=[], keywords=[],
                ),
            )]
            body.extend(self.translate_block_as_body(ifx.then_block))
        else:
            raise TranslateError(
                "if let currently supports Some(binding)",
                ifx.pattern.line, ifx.pattern.col,
            )
        if isinstance(ifx.else_branch, Block):
            orelse = self.translate_block_as_body(ifx.else_branch)
        elif isinstance(ifx.else_branch, IfLet):
            orelse = self.translate_if_let_stmt(ifx.else_branch)
        elif isinstance(ifx.else_branch, IfExpr):
            orelse = [self.translate_if_stmt(ifx.else_branch)]
        else:
            orelse = []
        return [
            ast.Assign(
                targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                value=self.translate_expr(ifx.value),
            ),
            ast.If(test=test, body=body, orelse=orelse),
        ]

    def translate_match_stmt(self, m: MatchExpr) -> ast.If:
        scrutinee_type = self._match_scrutinee_type(m)
        warn_match_issues(
            m, scrutinee_type=scrutinee_type,
        )
        subject = self.translate_expr(m.scrutinee)
        missing = missing_match_variants(m, scrutinee_type)
        node: ast.stmt | None = (
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(
                        id="_runtime_non_exhaustive_match",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=scrutinee_type)],
                    keywords=[],
                ),
            )
            if missing else None
        )
        for arm in reversed(m.arms):
            self._validate_pattern_bindings(arm.pattern)
            test, bindings = self._pattern_test_bindings(arm.pattern, subject)
            body: list[ast.stmt] = self._translate_match_body(arm.body)
            for name, value in reversed(list(bindings.items())):
                body.insert(0, ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=value,
                ))
            if arm.guard is not None:
                guarded_body: ast.stmt = ast.If(
                    test=self.translate_expr(arm.guard),
                    body=body,
                    orelse=[node] if node is not None else [],
                )
                body = [guarded_body]
            if test is None:
                node = ast.If(
                    test=ast.Constant(value=True), body=body,
                    orelse=[node] if node is not None else [],
                )
            else:
                node = ast.If(
                    test=test, body=body,
                    orelse=[node] if node is not None else [],
                )
        if node is None:
            return ast.If(
                test=ast.Constant(value=False),
                body=[ast.Pass()], orelse=[],
            )
        return node

    def _pattern_test(
        self, pat: Pattern, subject: ast.expr,
    ) -> ast.expr | None:
        return self._pattern_test_bindings(pat, subject)[0]

    def _pattern_test_bindings(
        self, pat: Pattern, subject: ast.expr,
    ) -> tuple[ast.expr, dict[str, ast.expr]]:
        if isinstance(pat, PatWildcard):
            return ast.Constant(value=True), {}
        if isinstance(pat, PatIdent):
            return ast.Constant(value=True), {pat.name: subject}
        if isinstance(pat, PatLiteral):
            return ast.Compare(
                left=subject, ops=[ast.Eq()],
                comparators=[self.translate_expr(pat.value)],
            ), {}
        if isinstance(pat, PatRange):
            tests: list[ast.expr] = []
            if pat.start is not None:
                tests.append(ast.Compare(
                    left=subject, ops=[ast.GtE()],
                    comparators=[self.translate_expr(pat.start)],
                ))
            if pat.end is not None:
                tests.append(ast.Compare(
                    left=subject,
                    ops=[ast.LtE() if pat.inclusive else ast.Lt()],
                    comparators=[self.translate_expr(pat.end)],
                ))
            return self._and_pattern_tests(tests), {}
        if isinstance(pat, PatTuple):
            tests: list[ast.expr] = [
                ast.Call(
                    func=ast.Name(id="isinstance", ctx=ast.Load()),
                    args=[subject, ast.Name(id="tuple", ctx=ast.Load())],
                    keywords=[],
                ),
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[subject], keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=len(pat.elems))],
                ),
            ]
            bindings: dict[str, ast.expr] = {}
            for index, child in enumerate(pat.elems):
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Subscript(
                        value=subject, slice=ast.Constant(value=index),
                        ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatSlice):
            rest_index = next(
                (
                    index for index, elem in enumerate(pat.elems)
                    if isinstance(elem, PatRest)
                    or (
                        isinstance(elem, PatBinding)
                        and isinstance(elem.inner, PatRest)
                    )
                ),
                None,
            )
            has_rest = rest_index is not None
            fixed_count = len(pat.elems) - int(has_rest)
            tests: list[ast.expr] = [
                ast.Call(
                    func=ast.Name(id="isinstance", ctx=ast.Load()),
                    args=[
                        subject,
                        ast.Tuple(
                            elts=[
                                ast.Name(id="list", ctx=ast.Load()),
                                ast.Name(id="tuple", ctx=ast.Load()),
                            ],
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                ),
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[subject], keywords=[],
                    ),
                    ops=[ast.GtE() if has_rest else ast.Eq()],
                    comparators=[ast.Constant(value=fixed_count)],
                ),
            ]
            bindings: dict[str, ast.expr] = {}
            trailing_count = (
                len(pat.elems) - rest_index - 1
                if rest_index is not None else 0
            )
            for index, child in enumerate(pat.elems):
                is_rest = isinstance(child, PatRest) or (
                    isinstance(child, PatBinding)
                    and isinstance(child.inner, PatRest)
                )
                if is_rest:
                    upper = (
                        ast.UnaryOp(
                            op=ast.USub(),
                            operand=ast.Constant(value=trailing_count),
                        )
                        if trailing_count else None
                    )
                    rest_value = ast.Subscript(
                        value=subject,
                        slice=ast.Slice(
                            lower=ast.Constant(value=rest_index),
                            upper=upper,
                            step=None,
                        ),
                        ctx=ast.Load(),
                    )
                    if isinstance(child, PatBinding):
                        bindings[child.name] = rest_value
                    continue
                sequence_index = index
                if rest_index is not None and index > rest_index:
                    sequence_index = index - len(pat.elems)
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Subscript(
                        value=subject,
                        slice=ast.Constant(value=sequence_index),
                        ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatStruct):
            class_name = pat.path.split("::")[-1]
            declared_fields = self.struct_fields.get(class_name)
            if declared_fields is not None:
                used_fields = {name for name, _child in pat.fields}
                extra_fields = used_fields - declared_fields
                if extra_fields:
                    raise TranslateError(
                        f"struct pattern {pat.path} has unknown fields: "
                        f"{', '.join(sorted(extra_fields))}",
                        pat.line, pat.col,
                    )
                missing_fields = declared_fields - used_fields
                if missing_fields and not pat.rest:
                    raise TranslateError(
                        f"struct pattern {pat.path} is missing fields; "
                        "include '..'",
                        pat.line, pat.col,
                    )
            tests = [ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[subject, ast.Name(id=class_name, ctx=ast.Load())],
                keywords=[],
            )]
            bindings = {}
            for name, child in pat.fields:
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Attribute(
                        value=subject, attr=name, ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatBinding):
            if isinstance(pat.inner, PatRest):
                return ast.Constant(value=True), {pat.name: subject}
            if not pat.name.isidentifier():
                raise TranslateError(
                    f"invalid binding name {pat.name!r}",
                    pat.line, pat.col,
                )
            test, bindings = self._pattern_test_bindings(pat.inner, subject)
            bindings[pat.name] = subject
            return test, bindings
        if isinstance(pat, PatBox):
            return self._pattern_test_bindings(pat.inner, subject)
        if isinstance(pat, PatConst):
            parts = pat.path.split("::")
            resolved = (
                parts[0] in self.constant_names
                if len(parts) == 1
                else (
                    parts[0] in self.enum_variants
                    and parts[-1] in self.enum_variants[parts[0]]
                )
            )
            if not resolved:
                raise TranslateError(
                    f"unresolved constant pattern {pat.path!r}",
                    pat.line, pat.col,
                )
            return ast.Compare(
                left=subject,
                ops=[ast.Eq()],
                comparators=[self._path_to_expr(pat.path)],
            ), {}
        if isinstance(pat, PatPath):
            return ast.Compare(
                left=subject,
                ops=[ast.Eq()],
                comparators=[self._path_to_expr(pat.path)],
            ), {}
        if isinstance(pat, PatMacro):
            raise TranslateError(
                f"pattern macro {pat.path}! is not supported",
                pat.line, pat.col,
            )
        if isinstance(pat, PatRef):
            return self._pattern_test_bindings(pat.inner, subject)
        if isinstance(pat, PatOr):
            alternatives = [
                self._pattern_test_bindings(alt, subject)
                for alt in pat.alts
            ]
            binding_sets = [
                frozenset(bindings) for _test, bindings in alternatives
            ]
            if binding_sets and any(
                names != binding_sets[0] for names in binding_sets[1:]
            ):
                raise TranslateError(
                    "or-pattern alternatives must bind the same names",
                    pat.line, pat.col,
                )
            tests = [test for test, _bindings in alternatives]
            names = set().union(*(bindings for _test, bindings in alternatives))
            merged: dict[str, ast.expr] = {}
            for name in names:
                value = alternatives[-1][1][name]
                for test, bindings in reversed(alternatives[:-1]):
                    value = ast.IfExp(
                        test=test, body=bindings[name], orelse=value,
                    )
                merged[name] = value
            return ast.BoolOp(op=ast.Or(), values=tests), merged
        if isinstance(pat, PatEnum):
            enum_name, variant = self._split_path(pat.path)
            comparator: ast.expr
            if enum_name is not None:
                comparator = ast.Attribute(
                    value=ast.Name(id=enum_name, ctx=ast.Load()),
                    attr=self._python_identifier(variant),
                    ctx=ast.Load(),
                )
            else:
                comparator = ast.Name(
                    id=self._python_identifier(variant),
                    ctx=ast.Load(),
                )
            if pat.args:
                tests: list[ast.expr] = [
                    ast.Call(
                        func=ast.Name(id="isinstance", ctx=ast.Load()),
                        args=[subject, ast.Name(id="tuple", ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Compare(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[subject], keywords=[],
                        ),
                        ops=[ast.Eq()],
                        comparators=[
                            ast.Constant(value=len(pat.args) + 1),
                        ],
                    ),
                    ast.Compare(
                        left=ast.Subscript(
                            value=subject, slice=ast.Constant(value=0),
                            ctx=ast.Load(),
                        ),
                        ops=[ast.Eq()],
                        comparators=[comparator],
                    ),
                ]
                bindings: dict[str, ast.expr] = {}
                for index, child in enumerate(pat.args, start=1):
                    child_test, child_bindings = self._pattern_test_bindings(
                        child,
                        ast.Subscript(
                            value=subject, slice=ast.Constant(value=index),
                            ctx=ast.Load(),
                        ),
                    )
                    tests.append(child_test)
                    bindings.update(child_bindings)
                return self._and_pattern_tests(tests), bindings
            return ast.Compare(
                left=subject, ops=[ast.Eq()], comparators=[comparator],
            ), {}
        raise TranslateError(
            f"unsupported match pattern {type(pat).__name__}",
            getattr(pat, "line", 0), getattr(pat, "col", 0),
        )

    def _validate_pattern_bindings(self, pattern: Pattern) -> set[str]:
        if isinstance(pattern, PatIdent):
            return {pattern.name}
        if isinstance(pattern, PatBinding):
            names = self._validate_pattern_bindings(pattern.inner)
            if pattern.name in names:
                self._duplicate_binding_error(pattern, pattern.name)
            return names | {pattern.name}
        if isinstance(pattern, PatOr):
            alternatives = [
                self._validate_pattern_bindings(alt)
                for alt in pattern.alts
            ]
            if alternatives and any(
                names != alternatives[0] for names in alternatives[1:]
            ):
                raise TranslateError(
                    "or-pattern alternatives must bind the same names in "
                    f"{self._pattern_text(pattern)}",
                    pattern.line, pattern.col,
                )
            return alternatives[0] if alternatives else set()
        if isinstance(pattern, PatRef | PatBox):
            return self._validate_pattern_bindings(pattern.inner)
        if isinstance(pattern, (PatTuple, PatSlice)):
            children = pattern.elems
        elif isinstance(pattern, PatEnum):
            children = pattern.args
        elif isinstance(pattern, PatStruct):
            children = [child for _name, child in pattern.fields]
        else:
            return set()
        names: set[str] = set()
        for child in children:
            child_names = self._validate_pattern_bindings(child)
            duplicates = names & child_names
            if duplicates:
                self._duplicate_binding_error(
                    pattern, min(duplicates),
                )
            names.update(child_names)
        return names

    def _duplicate_binding_error(self, pattern: Pattern, name: str) -> None:
        raise TranslateError(
            f"duplicate binding {name!r} in pattern "
            f"{self._pattern_text(pattern)}",
            pattern.line, pattern.col,
        )

    def _pattern_text(self, pattern: Pattern) -> str:
        if isinstance(pattern, PatIdent):
            return pattern.name
        if isinstance(pattern, PatWildcard):
            return "_"
        if isinstance(pattern, PatLiteral):
            value = pattern.value
            if hasattr(value, "raw"):
                return value.raw
            return str(getattr(value, "value", value))
        if isinstance(pattern, PatTuple):
            inner = ", ".join(self._pattern_text(item) for item in pattern.elems)
            if len(pattern.elems) == 1:
                inner += ","
            return f"({inner})"
        if isinstance(pattern, PatSlice):
            return f"[{', '.join(self._pattern_text(item) for item in pattern.elems)}]"
        if isinstance(pattern, PatOr):
            return " | ".join(self._pattern_text(item) for item in pattern.alts)
        if isinstance(pattern, PatBinding):
            return f"{pattern.name} @ {self._pattern_text(pattern.inner)}"
        if isinstance(pattern, PatRef):
            return f"&{self._pattern_text(pattern.inner)}"
        if isinstance(pattern, PatRest):
            return ".."
        if isinstance(pattern, PatEnum):
            args = ", ".join(self._pattern_text(item) for item in pattern.args)
            return f"{pattern.path}({args})"
        if isinstance(pattern, PatStruct):
            fields = ", ".join(
                f"{name}: {self._pattern_text(child)}"
                for name, child in pattern.fields
            )
            return f"{pattern.path} {{ {fields} }}"
        return type(pattern).__name__

    def _and_pattern_tests(self, tests: list[ast.expr]) -> ast.expr:
        if not tests:
            return ast.Constant(value=True)
        if len(tests) == 1:
            return tests[0]
        return ast.BoolOp(op=ast.And(), values=tests)

    def _split_path(self, path: str) -> tuple[str | None, str]:
        parts = path.split("::")
        if len(parts) == 1:
            return None, parts[0]
        return parts[-2], parts[-1]

    def _path_to_expr(self, path: str) -> ast.expr:
        parts = path.split("::")
        expr: ast.expr = ast.Name(
            id=self._python_identifier(parts[0]), ctx=ast.Load(),
        )
        for part in parts[1:]:
            expr = ast.Attribute(
                value=expr,
                attr=self._python_identifier(part),
                ctx=ast.Load(),
            )
        return expr

    def _python_identifier(self, name: str) -> str:
        if name in {"None", "True", "False"}:
            return f"{name}_"
        return name

    def _translate_match_body(self, body: Expr) -> list[ast.stmt]:
        if isinstance(body, Block):
            return self.translate_block_as_fn_body(body)
        return [ast.Return(value=self.translate_expr(body))]

    # ---- expressions -------------------------------------------------------

    def translate_opt_expr(self, e: Expr | None) -> ast.expr | None:
        if e is None:
            return None
        return self.translate_expr(e)

    def _coerce_return_value(
        self, value: ast.expr | None,
    ) -> ast.expr | None:
        if (
            value is None
            or self._return_numeric_type is None
            or self._return_numeric_type in {"usize", "isize"}
        ):
            return value
        return self._coerce_numeric(value, self._return_numeric_type)

    def infer_type(self, expr: Expr) -> str | None:
        if isinstance(expr, MethodCall):
            receiver_type = self._known_expr_type(expr.receiver)
            if expr.method == "len":
                if receiver_type in {"Vec", "String", "HashMap", "BTreeMap", "HashSet"}:
                    return "usize"
                return self._method_return_types.get(
                    (receiver_type, expr.method),
                )
            # M3: nested collection method return types
            if receiver_type and receiver_type.split(":", 1)[0] in {"HashMap", "BTreeMap"}:
                if expr.method == "get":
                    return self._nested_get_value_type(expr.receiver)
                if expr.method == "get_mut":
                    return self._nested_get_value_type(expr.receiver)
                if expr.method in {"keys", "values", "iter"}:
                    return "Vec"
            if receiver_type == "Option" and expr.method == "unwrap":
                return self._option_unwrap_type(expr.receiver)
            return self._method_return_types.get(
                (receiver_type, expr.method),
            )
        if isinstance(expr, Cast) and isinstance(expr.ty, TypeName):
            return expr.ty.name
        if isinstance(expr, Range):
            return (
                self._known_expr_type(expr.end)
                if expr.end is not None
                else None
            ) or (
                self._known_expr_type(expr.start)
                if expr.start is not None
                else None
            )
        if isinstance(expr, Binary):
            left_type = self._known_expr_type(expr.left)
            right_type = self._known_expr_type(expr.right)
            if left_type is not None and right_type is not None:
                self._check_type_boundary(
                    right_type, left_type, expr.line, expr.col,
                )
            return left_type or right_type
        return None

    def _nested_get_value_type(self, receiver: Expr) -> str | None:
        """Return the language type of the value stored in a map receiver."""
        if isinstance(receiver, Ident):
            binding = self._binding_types.get(receiver.name)
            if binding and ":" in binding:
                value_kind = binding.split(":", 1)[1]
                if value_kind in {"HashMap", "BTreeMap", "HashSet", "Vec"}:
                    return value_kind
            return None
        if isinstance(receiver, Paren):
            return self._nested_get_value_type(receiver.inner)
        return None

    def _option_unwrap_type(self, receiver: Expr) -> str | None:
        """Return the payload type of Option::unwrap from a get chain."""
        if isinstance(receiver, MethodCall):
            if receiver.method in {"get", "get_mut"}:
                return self._nested_get_value_type(receiver.receiver)
            if receiver.method == "unwrap":
                return self._option_unwrap_type(receiver.receiver)
        if isinstance(receiver, Paren):
            return self._option_unwrap_type(receiver.inner)
        return None

    def _known_expr_type(self, expr: Expr) -> str | None:
        inferred = self.infer_type(expr)
        if inferred is not None:
            return inferred
        if isinstance(expr, Ident):
            return self._numeric_types.get(
                expr.name,
                self._binding_types.get(expr.name),
            )
        if isinstance(expr, StructLiteral):
            return expr.path.split("::")[-1]
        if isinstance(expr, Field):
            receiver_type = self._known_expr_type(expr.receiver)
            return self.struct_field_types.get(receiver_type or "", {}).get(
                expr.name,
            )
        if isinstance(expr, Paren):
            return self._known_expr_type(expr.inner)
        return None

    def _check_call_arguments(self, call: Call) -> None:
        if isinstance(call.func, Ident):
            name = call.func.name
        elif isinstance(call.func, Path):
            name = call.func.segments[-1]
        else:
            return
        parameter_types = self._function_signatures.get(name)
        if parameter_types is None:
            return
        for argument, expected_type in zip(call.args, parameter_types):
            self._check_type_boundary(
                self._known_expr_type(argument),
                expected_type,
                argument.line,
                argument.col,
            )

    def _is_integer_type_name(self, ty: str | None) -> bool:
        if ty is None:
            return False
        return ty in _INT_TYPE_BITS or ty in {"isize", "usize"}

    def _check_type_boundary(
        self,
        source_type: str | None,
        expected_type: str | None,
        line: int,
        col: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        if source_type is None or expected_type is None:
            return
        # M4: implicit conversion between different integer types requires `as`
        if self._is_integer_type_name(source_type) and self._is_integer_type_name(expected_type):
            if source_type != expected_type:
                raise TypeErrorForBoundary(
                    source_type, expected_type, line, col, span,
                )
            return
        # fallback for pointer-sized cases
        source_ty = TypeName(line, col, source_type)
        expected_ty = TypeName(line, col, expected_type)
        source_is_pointer_sized = (
            is_usize_type(source_ty) or is_isize_type(source_ty)
        )
        expected_is_pointer_sized = (
            is_usize_type(expected_ty) or is_isize_type(expected_ty)
        )
        if (
            (source_is_pointer_sized and is_i64_type(expected_ty))
            or (expected_is_pointer_sized and is_i64_type(source_ty))
            or (
                source_is_pointer_sized
                and expected_is_pointer_sized
                and source_type != expected_type
            )
        ):
            raise TypeErrorForBoundary(
                source_type, expected_type, line, col, span,
            )

    def _check_usize_method_args(self, expr: MethodCall) -> None:
        if expr.method in {"repeat", "is_char_boundary"}:
            indexes = (0,)
        elif self._is_vec_expr(expr.receiver):
            indexes = {
                "get": (0,),
                "truncate": (0,),
                "rotate_left": (0,),
                "rotate_right": (0,),
                "swap": (0, 1),
                "resize": (0,),
                "split_off": (0,),
                "windows": (0,),
                "chunks": (0,),
                "insert": (0,),
                "remove": (0,),
            }.get(expr.method, ())
        else:
            indexes = ()
        for index in indexes:
            if index < len(expr.args):
                arg = expr.args[index]
                self._check_type_boundary(
                    self._known_expr_type(arg), "usize", arg.line, arg.col,
                )

    def _is_vec_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            return self._binding_types.get(expr.name) == "Vec"
        if isinstance(expr, (ArrayExpr, RepeatArray)):
            return True
        if isinstance(expr, MacroCall) and expr.path == "vec":
            return True
        if isinstance(expr, MethodCall):
            return self._is_vec_expr(expr.receiver)
        return False

    def _unwrap_paren(self, expr: Expr) -> Expr:
        while isinstance(expr, Paren):
            expr = expr.inner
        return expr

    def _integer_literal_value(self, expr: Expr) -> int | None:
        expr = self._unwrap_paren(expr)
        if isinstance(expr, LitInt):
            return int(expr.value)
        if isinstance(expr, Unary) and expr.op == "-":
            inner = self._integer_literal_value(expr.operand)
            return -inner if inner is not None else None
        return None

    def _int_range(self, ty: str) -> tuple[int, int] | None:
        if ty in {"isize", "usize"}:
            w = self.target_width
            if ty == "usize":
                return (0, (1 << w) - 1)
            return (-(1 << (w - 1)), (1 << (w - 1)) - 1)
        if ty not in _INT_TYPE_BITS:
            return None
        bits, signed = _INT_TYPE_BITS[ty]
        if signed:
            return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        return (0, (1 << bits) - 1)

    def translate_cast(self, node: Cast) -> ast.expr:
        inner = self.translate_expr(node.expr)
        if isinstance(node.ty, TypeName):
            target = node.ty.name
            # Pointer-sized casts keep existing helpers (range-checked)
            if target in {"isize", "usize"}:
                helper = "_cast_isize" if target == "isize" else "_cast_usize"
                return ast.Call(
                    func=ast.Name(id=helper, ctx=ast.Load()),
                    args=[inner],
                    keywords=[],
                )
            # All fixed-width integer casts
            if target in _INT_TYPE_BITS:
                lit = self._integer_literal_value(node.expr)
                literal_oob = False
                if lit is not None:
                    rng = self._int_range(target)
                    if rng is not None and not (rng[0] <= lit <= rng[1]):
                        literal_oob = True
                helper = f"_cast_{target}"
                call = ast.Call(
                    func=ast.Name(id=helper, ctx=ast.Load()),
                    args=[inner],
                    keywords=[],
                )
                if literal_oob:
                    call.keywords.append(
                        ast.keyword(arg="literal_out_of_range", value=ast.Constant(value=True))
                    )
                return call
            return self._coerce_numeric(inner, target)
        return inner

    def translate_expr(self, e: Expr) -> ast.expr:
        if isinstance(e, LitInt):
            return ast.Constant(value=e.value)
        if isinstance(e, LitFloat):
            return ast.Constant(value=e.value)
        if isinstance(e, LitStr):
            return ast.Constant(value=e.value)
        if isinstance(e, LitChar):
            return ast.Constant(value=e.value)
        if isinstance(e, LitBool):
            return ast.Constant(value=e.value)
        if isinstance(e, Ident):
            return ast.Name(id=e.name, ctx=ast.Load())
        if isinstance(e, Path):
            # A::B::C -> if A is a known struct/enum, emit A.B (or A.B.C)
            if len(e.segments) >= 2:
                first = e.segments[0]
                if first in self.struct_names or first in self.enum_names:
                    node: ast.expr = ast.Name(id=first, ctx=ast.Load())
                    for seg in e.segments[1:]:
                        node = ast.Attribute(
                            value=node,
                            attr=self._python_identifier(seg),
                            ctx=ast.Load(),
                        )
                    return node
            return ast.Name(
                id=self._python_identifier(e.segments[-1]),
                ctx=ast.Load(),
            )
        if isinstance(e, OptionSome):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_Option", ctx=ast.Load()),
                    attr="some",
                    ctx=ast.Load(),
                ),
                args=[self.translate_expr(e.value)],
                keywords=[],
            )
        if isinstance(e, OptionNone):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_Option", ctx=ast.Load()),
                    attr="none",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        if isinstance(e, ResultOk):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_Result", ctx=ast.Load()),
                    attr="from_ok",
                    ctx=ast.Load(),
                ),
                args=[self.translate_expr(e.value)],
                keywords=[],
            )
        if isinstance(e, ResultErr):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_Result", ctx=ast.Load()),
                    attr="from_err",
                    ctx=ast.Load(),
                ),
                args=[self.translate_expr(e.value)],
                keywords=[],
            )

        if isinstance(e, Paren):
            return self.translate_expr(e.inner)
        if isinstance(e, Unary):
            if (
                e.op == "-"
                and isinstance(e.operand, Cast)
                and isinstance(e.operand.ty, TypeName)
                and e.operand.ty.name in {"isize", "usize"}
            ):
                return self.translate_cast(Cast(
                    e.operand.line,
                    e.operand.col,
                    Unary(
                        e.line,
                        e.col,
                        "-",
                        e.operand.expr,
                    ),
                    e.operand.ty,
                ))
            return self._translate_unary(e)
        if isinstance(e, Binary):
            return self._translate_binary(e)
        if isinstance(e, (Assign, CompoundAssign)):
            raise TranslateError(
                "Assign cannot appear in expression position",
                e.line, e.col,
            )
        if isinstance(e, Call):
            if (
                isinstance(e.func, Path)
                and e.func.segments == ["String", "from"]
                and len(e.args) == 1
            ):
                return ast.Call(
                    func=ast.Name(id="str", ctx=ast.Load()),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if (
                isinstance(e.func, Path)
                and e.func.segments == ["HashMap", "new"]
                and not e.args
            ):
                return ast.Dict(keys=[], values=[])
            if (
                isinstance(e.func, Path)
                and e.func.segments == ["HashSet", "new"]
                and not e.args
            ):
                return ast.Call(func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[])
            if (
                isinstance(e.func, Path)
                and e.func.segments == ["BTreeMap", "new"]
                and not e.args
            ):
                return ast.Call(func=ast.Name(id="_btreemap_new", ctx=ast.Load()), args=[], keywords=[])
            self._check_call_arguments(e)
            func = self.translate_expr(e.func)
            args = [self.translate_expr(a) for a in e.args]
            return ast.Call(func=func, args=args, keywords=[])
        if isinstance(e, MethodCall):
            self._check_usize_method_args(e)
            recv = self.translate_expr(e.receiver)
            if e.method == "into_iter" and not e.args:
                return recv
            if self._is_hashmap_expr(e.receiver):
                if e.method == "insert" and len(e.args) == 2:
                    return ast.Call(
                        func=ast.Name(id="_map_insert", ctx=ast.Load()),
                        args=[
                            recv,
                            self.translate_expr(e.args[0]),
                            self.translate_expr(e.args[1]),
                        ],
                        keywords=[],
                    )
                if e.method == "get" and len(e.args) == 1:
                    return ast.Call(
                        func=ast.Name(id="_map_get", ctx=ast.Load()),
                        args=[recv, self.translate_expr(e.args[0])],
                        keywords=[],
                    )
                if e.method == "get_mut" and len(e.args) == 1:
                    return ast.Call(
                        func=ast.Name(id="_map_get_mut", ctx=ast.Load()),
                        args=[recv, self.translate_expr(e.args[0])],
                        keywords=[],
                    )
                if e.method == "remove" and len(e.args) == 1:
                    return ast.Call(
                        func=ast.Name(id="_map_remove", ctx=ast.Load()),
                        args=[recv, self.translate_expr(e.args[0])],
                        keywords=[],
                    )
                if e.method == "contains_key" and len(e.args) == 1:
                    return ast.Compare(
                        left=self.translate_expr(e.args[0]),
                        ops=[ast.In()],
                        comparators=[recv],
                    )
                if e.method == "entry" and len(e.args) == 1:
                    return ast.Call(
                        func=ast.Name(id="_map_entry", ctx=ast.Load()),
                        args=[recv, self.translate_expr(e.args[0])],
                        keywords=[],
                    )
                if e.method == "keys" and not e.args:
                    return ast.Call(
                        func=ast.Name(id="_map_keys", ctx=ast.Load()),
                        args=[recv],
                        keywords=[],
                    )
                if e.method == "values" and not e.args:
                    return ast.Call(
                        func=ast.Name(id="_map_values", ctx=ast.Load()),
                        args=[recv],
                        keywords=[],
                    )
                if e.method == "iter" and not e.args:
                    return ast.Call(
                        func=ast.Name(id="_map_iter", ctx=ast.Load()),
                        args=[recv],
                        keywords=[],
                    )
            if self._is_hashset_expr(e.receiver):
                if e.method == "insert" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_hashset_insert", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "contains" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_hashset_contains", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "remove" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_hashset_remove", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "len" and not e.args:
                    return ast.Call(func=ast.Name(id="_hashset_len", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "is_empty" and not e.args:
                    return ast.Call(func=ast.Name(id="_hashset_is_empty", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "iter" and not e.args:
                    return ast.Call(func=ast.Name(id="_hashset_iter", ctx=ast.Load()), args=[recv], keywords=[])
            if self._is_btreemap_expr(e.receiver):
                if e.method == "insert" and len(e.args) == 2:
                    return ast.Call(func=ast.Name(id="_btreemap_insert", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0]), self.translate_expr(e.args[1])], keywords=[])
                if e.method == "get" and len(e.args) == 1:
                    v = ast.Call(func=ast.Name(id="_btreemap_get", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                    # return Option-like for parity? but easier to return value/None; keep simple
                    return v
                if e.method == "remove" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_btreemap_remove", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "contains_key" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_btreemap_contains", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "len" and not e.args:
                    return ast.Call(func=ast.Name(id="_btreemap_len", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "is_empty" and not e.args:
                    return ast.Call(func=ast.Name(id="_btreemap_is_empty", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "keys" and not e.args:
                    return ast.Call(func=ast.Name(id="_btreemap_keys", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "values" and not e.args:
                    return ast.Call(func=ast.Name(id="_btreemap_values", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "iter" and not e.args:
                    return ast.Call(func=ast.Name(id="_btreemap_iter", ctx=ast.Load()), args=[recv], keywords=[])
                if e.method == "entry" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_btreemap_entry", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
                if e.method == "get_mut" and len(e.args) == 1:
                    return ast.Call(func=ast.Name(id="_btreemap_get_mut", ctx=ast.Load()), args=[recv, self.translate_expr(e.args[0])], keywords=[])
            if e.method == "map_err" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="map_err", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "unwrap" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="unwrap", ctx=ast.Load(),
                    ),
                    args=[], keywords=[],
                )
            if e.method == "unwrap_or" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="unwrap_or", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "unwrap_or_else" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="unwrap_or_else", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("windows", "chunks") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(
                        id=f"_vec_{e.method}", ctx=ast.Load(),
                    ),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "retain" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_retain", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "dedup" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_vec_dedup", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "remove" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="pop", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "fold" and len(e.args) == 2:
                return ast.Call(
                    func=ast.Name(id="_reduce", ctx=ast.Load()),
                    args=[
                        self.translate_expr(e.args[1]),
                        recv,
                        self.translate_expr(e.args[0]),
                    ],
                    keywords=[],
                )
            if e.method == "map" and len(e.args) == 1:
                option_kind = self._option_result_type(e.receiver)
                if option_kind is not None:
                    raise TranslateError(
                        f"{option_kind}.map is unsupported; use match or "
                        "if let instead",
                        e.line, e.col,
                    )
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "Vec.map expects a single identifier parameter",
                        e.line, e.col,
                    )
                target = ast.Name(
                    id=closure.params[0][0].name, ctx=ast.Store(),
                )
                body = (
                    self.translate_expr(closure.body.tail)
                    if isinstance(closure.body, Block)
                    and closure.body.tail is not None
                    else self.translate_expr(closure.body)
                )
                return ast.ListComp(
                    elt=body,
                    generators=[
                        ast.comprehension(
                            target=target,
                            iter=recv,
                            ifs=[],
                            is_async=0,
                        ),
                    ],
                )
            if e.method == "filter" and len(e.args) == 1:
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "filter expects a single identifier parameter",
                        e.line, e.col,
                    )
                return ast.Call(
                    func=ast.Name(id="_filter", ctx=ast.Load()),
                    args=[recv, self._translate_closure(closure)],
                    keywords=[],
                )
            if e.method == "partition" and len(e.args) == 1:
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "partition expects a single identifier parameter",
                        e.line, e.col,
                    )
                return ast.Call(
                    func=ast.Name(id="_vec_partition", ctx=ast.Load()),
                    args=[recv, self._translate_closure(closure)],
                    keywords=[],
                )
            if e.method == "sort_by_key" and len(e.args) == 1:
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "sort_by_key expects a single identifier parameter",
                        e.line, e.col,
                    )
                return ast.Call(
                    func=ast.Attribute(value=recv, attr="sort", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="key",
                            value=self._translate_closure(closure),
                        ),
                    ],
                )
            if e.method == "is_sorted" and not e.args:
                return ast.Compare(
                    left=recv,
                    ops=[ast.Eq()],
                    comparators=[
                        ast.Call(
                            func=ast.Name(id="sorted", ctx=ast.Load()),
                            args=[recv],
                            keywords=[],
                        ),
                    ],
                )
            if e.method == "sort_by" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="sort", ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="key",
                            value=ast.Call(
                                func=ast.Name(
                                    id="_cmp_to_key", ctx=ast.Load(),
                                ),
                                args=[self.translate_expr(e.args[0])],
                                keywords=[],
                            ),
                        ),
                    ],
                )
            if e.method == "contains" and len(e.args) == 1:
                return ast.Compare(
                    left=self.translate_expr(e.args[0]),
                    ops=[ast.In()],
                    comparators=[recv],
                )
            if e.method in ("first", "last") and not e.args:
                index = ast.Constant(value=0 if e.method == "first" else -1)
                return ast.IfExp(
                    test=ast.Compare(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[recv],
                            keywords=[],
                        ),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0)],
                    ),
                    body=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="_Option", ctx=ast.Load()),
                            attr="some",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Subscript(
                            value=recv,
                            slice=index,
                            ctx=ast.Load(),
                        )],
                        keywords=[],
                    ),
                    orelse=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="_Option", ctx=ast.Load()),
                            attr="none",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                )
            if e.method == "get" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_get", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "find" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_string_find", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "rfind" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_string_rfind", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("strip_prefix", "strip_suffix") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_string_strip", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_expr(e.args[0]),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="prefix",
                            value=ast.Constant(
                                value=e.method == "strip_prefix",
                            ),
                        ),
                    ],
                )
            if e.method in ("split_once", "rsplit_once") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_string_split_once", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_expr(e.args[0]),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="reverse",
                            value=ast.Constant(value=e.method == "rsplit_once"),
                        ),
                    ],
                )
            if e.method == "truncate" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_truncate", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("rotate_left", "rotate_right") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_rotate", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_expr(e.args[0]),
                        ast.Constant(value=e.method == "rotate_right"),
                    ],
                    keywords=[],
                )
            if e.method == "fill" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_fill", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "swap" and len(e.args) == 2:
                return ast.Call(
                    func=ast.Name(id="_vec_swap", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_expr(e.args[0]),
                        self.translate_expr(e.args[1]),
                    ],
                    keywords=[],
                )
            if e.method == "extend_from_slice" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(
                        id="_vec_extend_from_slice", ctx=ast.Load(),
                    ),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "resize" and len(e.args) == 2:
                return ast.Call(
                    func=ast.Name(id="_vec_resize", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_expr(e.args[0]),
                        self.translate_expr(e.args[1]),
                    ],
                    keywords=[],
                )
            if e.method == "split_off" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_split_off", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("starts_with", "ends_with") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv,
                        attr="startswith" if e.method == "starts_with" else "endswith",
                        ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("trim_start", "trim_end") and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv,
                        attr="lstrip" if e.method == "trim_start" else "rstrip",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            if e.method == "is_ascii" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="isascii", ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            if e.method == "lines" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_string_lines", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "split_whitespace" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="split", ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            if e.method == "sort_unstable" and not e.args:
                return ast.Call(
                    func=ast.Attribute(value=recv, attr="sort", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            if e.method == "as_bytes" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="encode", ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            if e.method == "is_char_boundary" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(
                        id="_string_is_char_boundary", ctx=ast.Load(),
                    ),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method in ("to_ascii_lowercase", "to_ascii_uppercase") and not e.args:
                return ast.Call(
                    func=ast.Name(
                        id=f"_string_{e.method}", ctx=ast.Load(),
                    ),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "is_empty" and not e.args:
                return ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[recv],
                        keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                )
            if e.method == "repeat" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_string_repeat", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "clone" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_clone", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "as_slice" and not e.args:
                return recv
            if e.method == "to_vec" and not e.args:
                return recv
            if e.method == "len" and not e.args:
                receiver_type = self._known_expr_type(e.receiver)
                if (
                    receiver_type is None
                    or (receiver_type, "len") not in self._method_return_types
                ):
                    return ast.Call(
                        func=ast.Name(id="_rust_len", ctx=ast.Load()),
                        args=[recv],
                        keywords=[],
                    )
            if e.method == "chars" and not e.args:
                return ast.Call(
                    func=ast.Name(id="list", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "bytes" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="encode", ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value="utf-8")],
                    keywords=[],
                )
            if e.method == "to_string" and not e.args:
                return ast.Call(
                    func=ast.Name(id="str", ctx=ast.Load()),
                    args=[recv], keywords=[],
                )
            if e.method == "parse" and not e.args:
                return ast.Call(
                    func=ast.Name(id="int", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "split" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="split", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "join" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=self.translate_expr(e.args[0]),
                        attr="join",
                        ctx=ast.Load(),
                    ),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "trim" and not e.args:
                method = "strip"
                args: list[ast.expr] = []
            elif e.method in ("to_uppercase", "to_lowercase"):
                method = {
                    "to_uppercase": "upper",
                    "to_lowercase": "lower",
                }[e.method]
                args = []
            elif e.method in ("replace", "starts_with", "ends_with"):
                method = {
                    "starts_with": "startswith",
                    "ends_with": "endswith",
                }.get(e.method, e.method)
                args = [self.translate_expr(arg) for arg in e.args]
            else:
                method = ""
                args = []
            if method:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr=method, ctx=ast.Load(),
                    ),
                    args=args,
                    keywords=[],
                )
            attr = ast.Attribute(value=recv, attr=e.method, ctx=ast.Load())
            args = [self.translate_expr(a) for a in e.args]
            return ast.Call(func=attr, args=args, keywords=[])
        if isinstance(e, Field):
            recv = self.translate_expr(e.receiver)
            return ast.Attribute(value=recv, attr=e.name, ctx=ast.Load())
        if isinstance(e, Index):
            recv = self.translate_expr(e.receiver)
            if isinstance(e.index, Range):
                return ast.Call(
                    func=ast.Name(id="_rust_slice", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_opt_expr(e.index.start)
                        or ast.Constant(value=None),
                        self.translate_opt_expr(e.index.end)
                        or ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            index_type = self._known_expr_type(e.index)
            self._check_type_boundary(
                index_type, "usize", e.index.line, e.index.col,
            )
            idx = self.translate_expr(e.index)
            return ast.Subscript(value=recv, slice=idx, ctx=ast.Load())
        if isinstance(e, TupleExpr):
            elts: list[ast.expr] = [self.translate_expr(x) for x in e.elems]
            return ast.Tuple(elts=elts, ctx=ast.Load())
        if isinstance(e, ArrayExpr):
            elts: list[ast.expr] = [self.translate_expr(x) for x in e.elems]
            return ast.List(elts=elts, ctx=ast.Load())
        if isinstance(e, RepeatArray):
            elem = self.translate_expr(e.elem)
            count = self.translate_expr(e.count)
            return ast.BinOp(
                left=ast.List(elts=[elem], ctx=ast.Load()),
                op=ast.Mult(), right=count,
            )
        if isinstance(e, StructLiteral):
            return self._translate_struct_literal(e)
        if isinstance(e, IfLet):
            return self._translate_iflet_expr(e)
        if isinstance(e, IfExpr):
            if e.then_block.stmts or (
                isinstance(e.else_branch, Block) and e.else_branch.stmts
            ):
                raise TranslateError(
                    "if-expression body must be a single expression; "
                    "use an if-statement instead",
                    e.line, e.col,
                )
            return ast.IfExp(
                test=self.translate_expr(e.cond),
                body=self.translate_expr(e.then_block.tail)
                if e.then_block.tail is not None
                else ast.Constant(value=None),
                orelse=self._translate_else_expr(e.else_branch),
            )
        if isinstance(e, MatchExpr):
            return self._translate_match_expr(e)
        if isinstance(e, Closure):
            return self._translate_closure(e)
        if isinstance(e, Range):
            args: list[ast.expr] = []
            if e.start is not None:
                args.append(self.translate_expr(e.start))
            else:
                args.append(ast.Constant(value=0))
            if e.end is not None:
                end = self.translate_expr(e.end)
                if e.inclusive:
                    end = ast.BinOp(
                        left=end, op=ast.Add(),
                        right=ast.Constant(value=1),
                    )
                args.append(end)
            return ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=args, keywords=[],
            )
        if isinstance(e, Cast):
            return self.translate_cast(e)
        if isinstance(e, Return):
            raise TranslateError("Return is a statement here", e.line, e.col)
        if isinstance(e, MacroCall):
            return self._translate_macro(e)
        if isinstance(e, Await):
            raise TranslateError(
                "Await requires async context", e.line, e.col,
            )
        if isinstance(e, Try):
            if isinstance(e.expr, MethodCall) and e.expr.method == "parse":
                value = self.translate_expr(e.expr.receiver)
                result = ast.Call(
                    func=ast.Name(id="_result_parse", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                )
            else:
                result = self.translate_expr(e.expr)
            return ast.Call(
                func=ast.Attribute(
                    value=result, attr="__try__", ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        if isinstance(e, Block):
            if e.tail is not None:
                return self.translate_expr(e.tail)
            return ast.Constant(value=None)
        raise TranslateError(
            f"unsupported expression {type(e).__name__}",
            getattr(e, "line", 0), getattr(e, "col", 0),
        )

    def _translate_unary(self, e: Unary) -> ast.expr:
        operand = self.translate_expr(e.operand)
        if e.op == "-":
            return ast.UnaryOp(op=ast.USub(), operand=operand)
        if e.op == "!":
            return ast.UnaryOp(op=ast.Not(), operand=operand)
        if e.op in ("&", "&mut", "*"):
            return operand
        raise TranslateError(f"unknown unary op {e.op}", e.line, e.col)

    def _translate_binary(self, e: Binary) -> ast.expr:
        left = self.translate_expr(e.left)
        right = self.translate_expr(e.right)
        if e.op == "+":
            return ast.BinOp(left=left, op=ast.Add(), right=right)
        if e.op == "-":
            return ast.BinOp(left=left, op=ast.Sub(), right=right)
        if e.op == "*":
            return ast.BinOp(left=left, op=ast.Mult(), right=right)
        if e.op == "/":
            return ast.Call(
                func=ast.Name(id="_div", ctx=ast.Load()),
                args=[left, right], keywords=[],
            )
        if e.op == "%":
            return ast.Call(
                func=ast.Name(id="_mod", ctx=ast.Load()),
                args=[left, right], keywords=[],
            )
        if e.op == "==":
            return ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
        if e.op == "!=":
            return ast.Compare(left=left, ops=[ast.NotEq()], comparators=[right])
        if e.op == "<":
            return ast.Compare(left=left, ops=[ast.Lt()], comparators=[right])
        if e.op == ">":
            return ast.Compare(left=left, ops=[ast.Gt()], comparators=[right])
        if e.op == "<=":
            return ast.Compare(left=left, ops=[ast.LtE()], comparators=[right])
        if e.op == ">=":
            return ast.Compare(left=left, ops=[ast.GtE()], comparators=[right])
        if e.op == "&&":
            return ast.BoolOp(op=ast.And(), values=[left, right])
        if e.op == "||":
            return ast.BoolOp(op=ast.Or(), values=[left, right])
        if e.op == "&":
            return ast.BinOp(left=left, op=ast.BitAnd(), right=right)
        if e.op == "|":
            return ast.BinOp(left=left, op=ast.BitOr(), right=right)
        raise TranslateError(f"unknown binary op {e.op}", e.line, e.col)

    def _translate_struct_literal(self, e: StructLiteral) -> ast.expr:
        name = e.path.split("::")[-1]
        if name in self.struct_names:
            keywords: list[ast.keyword] = [
                ast.keyword(arg=k, value=self.translate_expr(v))
                for k, v in e.fields
            ]
            return ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[], keywords=keywords,
            )
        keys: list[ast.expr | None] = [
            ast.Constant(value=k) for k, _ in e.fields
        ]
        values: list[ast.expr] = [
            self.translate_expr(v) for _, v in e.fields
        ]
        return ast.Dict(keys=keys, values=values)

    def _translate_lvalue(
        self, e: Expr,
    ) -> ast.Name | ast.Attribute | ast.Subscript:
        if isinstance(e, Ident):
            return ast.Name(id=e.name, ctx=ast.Store())
        if isinstance(e, Field):
            recv = self.translate_expr(e.receiver)
            return ast.Attribute(value=recv, attr=e.name, ctx=ast.Store())
        if isinstance(e, Index):
            recv = self.translate_expr(e.receiver)
            idx = self.translate_expr(e.index)
            return ast.Subscript(value=recv, slice=idx, ctx=ast.Store())
        raise TranslateError(
            f"unsupported lvalue {type(e).__name__}",
            getattr(e, "line", 0), getattr(e, "col", 0),
        )

    def _translate_else_expr(self, branch) -> ast.expr:
        if branch is None:
            return ast.Constant(value=None)
        if isinstance(branch, Block):
            if branch.tail is not None:
                return self.translate_expr(branch.tail)
            return ast.Constant(value=None)
        if isinstance(branch, IfExpr):
            return self.translate_expr(branch)
        return self.translate_expr(branch)

    def _translate_match_expr(self, m: MatchExpr) -> ast.expr:
        scrutinee_type = self._match_scrutinee_type(m)
        warn_match_issues(
            m, scrutinee_type=scrutinee_type,
        )
        subject = self.translate_expr(m.scrutinee)
        missing = missing_match_variants(m, scrutinee_type)
        node: ast.expr = (
            ast.Call(
                func=ast.Name(
                    id="_runtime_non_exhaustive_match",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=scrutinee_type)],
                keywords=[],
            )
            if missing else ast.Constant(value=None)
        )
        for arm in reversed(m.arms):
            self._validate_pattern_bindings(arm.pattern)
            test, bindings = self._pattern_test_bindings(arm.pattern, subject)
            if isinstance(arm.body, Block):
                body: ast.expr = (
                    self.translate_expr(arm.body.tail)
                    if arm.body.tail is not None
                    else ast.Constant(value=None)
                )
            else:
                body = self.translate_expr(arm.body)
            if bindings:
                guarded = body
                if arm.guard is not None:
                    guarded = ast.IfExp(
                        test=self.translate_expr(arm.guard),
                        body=body, orelse=node,
                    )
                body = ast.Call(
                    func=ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg=name) for name in bindings],
                            kwonlyargs=[], kw_defaults=[], defaults=[],
                        ),
                        body=guarded,
                    ),
                    args=list(bindings.values()),
                    keywords=[],
                )
                node = ast.IfExp(test=test, body=body, orelse=node)
            else:
                if arm.guard is not None:
                    test = self._and_pattern_tests([
                        test, self.translate_expr(arm.guard),
                    ])
                node = ast.IfExp(test=test, body=body, orelse=node)
        return node

    def _match_scrutinee_type(self, match: MatchExpr) -> str | None:
        if isinstance(match.scrutinee, Ident):
            return self._binding_types.get(match.scrutinee.name)
        return self._expr_variant_type(match.scrutinee)

    def _option_result_type(self, expr: Expr) -> str | None:
        """Return ``"Option"``/``"Result"`` when the receiver kind is known."""
        if isinstance(expr, Ident):
            kind = self._binding_types.get(expr.name)
            return kind if kind in {"Option", "Result"} else None
        if isinstance(expr, Paren):
            return self._option_result_type(expr.inner)
        kind = self._expr_variant_type(expr)
        return kind if kind in {"Option", "Result"} else None

    def _expr_variant_type(self, expr: Expr) -> str | None:
        if isinstance(expr, (OptionSome, OptionNone)):
            return "Option"
        if isinstance(expr, (ResultOk, ResultErr)):
            return "Result"
        if isinstance(expr, MethodCall) and expr.method in {"pop", "next"}:
            return "Option"
        if isinstance(expr, Path) and expr.segments[:1] == ["HashMap"]:
            return "HashMap"
        if (
            isinstance(expr, Call)
            and isinstance(expr.func, Path)
            and expr.func.segments == ["HashMap", "new"]
        ):
            return "HashMap"
        if isinstance(expr, MethodCall):
            inferred = self.infer_type(expr)
            if inferred is not None:
                return inferred
            if isinstance(expr.receiver, MethodCall):
                return self._expr_variant_type(expr.receiver)
        return None

    def _expr_struct_type(self, expr: Expr) -> str | None:
        """Resolve the struct type of an expression when it is known."""
        if isinstance(expr, Ident):
            if expr.name == "self" and self._current_impl_type is not None:
                return self._current_impl_type
            return self._binding_types.get(expr.name)
        if isinstance(expr, Paren):
            return self._expr_struct_type(expr.inner)
        if isinstance(expr, Field):
            owner = self._expr_struct_type(expr.receiver)
            if owner is None:
                return None
            return self.struct_field_types.get(owner, {}).get(expr.name)
        return None

    def _is_hashmap_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            binding = self._binding_types.get(expr.name)
            return bool(binding) and binding.split(":", 1)[0] == "HashMap"
        if isinstance(expr, Field):
            owner = self._expr_struct_type(expr.receiver)
            if owner is None:
                return False
            fields = self.struct_field_types.get(owner, {})
            return fields.get(expr.name) == "HashMap"
        if (
            isinstance(expr, Call)
            and isinstance(expr.func, Path)
            and expr.func.segments == ["HashMap", "new"]
        ):
            return True
        if isinstance(expr, Paren):
            return self._is_hashmap_expr(expr.inner)
        return False

    def _is_hashset_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            return self._binding_types.get(expr.name) == "HashSet"
        if (
            isinstance(expr, Call)
            and isinstance(expr.func, Path)
            and expr.func.segments == ["HashSet", "new"]
        ):
            return True
        if isinstance(expr, Paren):
            return self._is_hashset_expr(expr.inner)
        return False

    def _is_btreemap_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            return self._binding_types.get(expr.name) == "BTreeMap"
        if (
            isinstance(expr, Call)
            and isinstance(expr.func, Path)
            and expr.func.segments == ["BTreeMap", "new"]
        ):
            return True
        if isinstance(expr, Paren):
            return self._is_btreemap_expr(expr.inner)
        return False

    def _is_map_entry_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Paren):
            return self._is_map_entry_expr(expr.inner)
        if isinstance(expr, MethodCall):
            if expr.method == "entry":
                return self._is_hashmap_expr(expr.receiver)
            if expr.method == "and_modify":
                return self._is_map_entry_expr(expr.receiver)
        return False

    def _is_map_get_mut_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Paren):
            return self._is_map_get_mut_expr(expr.inner)
        return (
            isinstance(expr, MethodCall)
            and expr.method == "get_mut"
            and self._is_hashmap_expr(expr.receiver)
        )

    def _is_map_slot_expr(self, expr: Expr) -> bool:
        """Return whether ``expr`` evaluates to a mutable map slot.

        Python has no ``&mut V``; slots are represented by ``_MapSlot``.
        Only expressions in this set may legally appear under a ``*``
        dereference that mutates in place.
        """
        if isinstance(expr, Ident):
            return expr.name in self._slot_bindings
        if isinstance(expr, Paren):
            return self._is_map_slot_expr(expr.inner)
        if isinstance(expr, MethodCall):
            if expr.method in {"unwrap", "expect"}:
                return self._is_map_get_mut_expr(expr.receiver)
            if expr.method in {"or_insert", "or_insert_with"}:
                return self._is_map_entry_expr(expr.receiver)
        return False

    def _slot_mutation_expr(self, expr: Expr) -> ast.expr | None:
        if not isinstance(expr, (Assign, CompoundAssign)):
            return None
        if not (isinstance(expr.target, Unary) and expr.target.op == "*"):
            return None
        if not self._is_map_slot_expr(expr.target.operand):
            raise TranslateError(
                "dereference assignment target is not a mutable map slot",
                expr.line, expr.col,
            )
        if isinstance(expr, CompoundAssign) and expr.op[0] not in "+-*/":
            raise TranslateError(
                f"unsupported compound assign op {expr.op!r}",
                expr.line, expr.col,
            )
        operand = self.translate_expr(expr.target.operand)
        value = self.translate_expr(expr.value)
        name = "_map_slot_set" if isinstance(expr, Assign) else "_map_slot_aug"
        values = [operand, value]
        if isinstance(expr, CompoundAssign):
            values.append(ast.Constant(value=expr.op[0]))
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=values,
            keywords=[],
        )

    def _translate_closure(self, c: Closure) -> ast.Lambda:
        args: list[ast.arg] = []
        param_names: list[str] = []
        for pat, _ty in c.params:
            if not isinstance(pat, PatIdent):
                raise TranslateError(
                    f"unsupported closure param {type(pat).__name__}",
                    c.line, c.col,
                )
            args.append(ast.arg(arg=pat.name, annotation=None))
            param_names.append(pat.name)
        previous_slot_bindings = self._slot_bindings
        # Closure parameters passed to the entry API (`and_modify`) are
        # mutable map slots, so `*value += 1` must resolve against them.
        self._slot_bindings = set(previous_slot_bindings)
        self._slot_bindings.update(param_names)
        try:
            body_expr: ast.expr
            if isinstance(c.body, Block):
                if c.body.tail is not None:
                    body_expr = self.translate_expr(c.body.tail)
                elif len(c.body.stmts) == 1:
                    stmt = c.body.stmts[0]
                    mutation = (
                        self._slot_mutation_expr(stmt.expr)
                        if isinstance(stmt, ExprStmt)
                        else None
                    )
                    body_expr = (
                        mutation
                        if mutation is not None
                        else ast.Constant(value=None)
                    )
                else:
                    body_expr = ast.Constant(value=None)
            else:
                mutation = self._slot_mutation_expr(c.body)
                body_expr = (
                    mutation
                    if mutation is not None
                    else self.translate_expr(c.body)
                )
        finally:
            self._slot_bindings = previous_slot_bindings
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=args, vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=body_expr,
        )

    # ---- macros ------------------------------------------------------------

    def _translate_macro(self, m: MacroCall) -> ast.expr:
        if m.path == "vec":
            if isinstance(m.args, str):
                from lexer import lex as _lex, TokenType
                from parser import Parser as _Parser
                tokens = _lex(m.args)
                parser = _Parser(tokens)
                elems = []
                while not parser.at_end():
                    elems.append(parser.parse_expr())
                    if not parser.eat(TokenType.COMMA):
                        break
                return ast.List(
                    elts=[self.translate_expr(x) for x in elems],
                    ctx=ast.Load(),
                )
            return ast.List(
                elts=[self.translate_expr(a) for a in m.args],
                ctx=ast.Load(),
            )
        target = MACRO_MAP.get(m.path)
        if target is None:
            raise TranslateError(
                f"unknown macro {m.path}!", m.line, m.col,
            )
        if isinstance(m.args, str):
            raise TranslateError(
                f"macro {m.path}! with token-stream args not supported",
                m.line, m.col,
            )
        if m.path == "assert":
            return ast.Call(
                func=ast.Name(id="assert", ctx=ast.Load()),
                args=[], keywords=[],
            )
        args: list[ast.expr] = [self.translate_expr(a) for a in m.args]
        if m.path in ("println", "print", "eprintln", "eprint", "format"):
            if not args:
                return ast.Call(
                    func=ast.Name(id=target, ctx=ast.Load()),
                    args=[ast.Constant(value="")], keywords=[],
                )
            fmt = args[0]
            rest = args[1:]
            if rest:
                fmt = ast.Call(
                    func=ast.Attribute(
                        value=fmt, attr="format", ctx=ast.Load(),
                    ),
                    args=rest, keywords=[],
                )
            return ast.Call(
                func=ast.Name(id=target, ctx=ast.Load()),
                args=[fmt], keywords=[],
            )
        return ast.Call(
            func=ast.Name(id=target, ctx=ast.Load()),
            args=args, keywords=[],
        )

    # ---- types -------------------------------------------------------------

    def _type_or_object(self, ty: Type | None) -> ast.expr:
        """Return a non-None type expression, falling back to `object`."""
        if ty is None:
            return ast.Name(id="object", ctx=ast.Load())
        result = self.translate_type(ty)
        if result is None:
            return ast.Name(id="object", ctx=ast.Load())
        return result

    def translate_type(self, ty: Type | None) -> ast.expr | None:
        if ty is None:
            return None
        if isinstance(ty, TypeName):
            if ty.name in self.generic_type_names:
                return ast.Name(id="int", ctx=ast.Load())
            base = TYPE_MAP.get(ty.name, ty.name)
            if ty.name in self.generic_struct_names:
                return ast.Name(id=base, ctx=ast.Load())
            if ty.args:
                if len(ty.args) > 1:
                    slice_multi: ast.expr = ast.Tuple(
                        elts=[self._type_or_object(a) for a in ty.args],
                        ctx=ast.Load(),
                    )
                    slice_ = slice_multi
                else:
                    slice_ = self._type_or_object(ty.args[0])
                return ast.Subscript(
                    value=ast.Name(id=base, ctx=ast.Load()),
                    slice=slice_,
                    ctx=ast.Load(),
                )
            return ast.Name(id=base, ctx=ast.Load())
        if isinstance(ty, TypeRef):
            return self.translate_type(ty.inner)
        if isinstance(ty, TypeSlice):
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=self._type_or_object(ty.inner),
                ctx=ast.Load(),
            )
        if isinstance(ty, TypeArray):
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=self._type_or_object(ty.inner),
                ctx=ast.Load(),
            )
        if isinstance(ty, TypeTuple):
            elts: list[ast.expr] = [
                self._type_or_object(x) for x in ty.elems
            ]
            return ast.Tuple(elts=elts, ctx=ast.Load())
        if isinstance(ty, TypeUnit):
            return ast.Constant(value=None)
        if isinstance(ty, (TypeFn, TypeImplTrait, TypeDynTrait)):
            return ast.Name(id="object", ctx=ast.Load())
        raise TranslateError(
            f"unsupported type {type(ty).__name__}",
            getattr(ty, "line", 0), getattr(ty, "col", 0),
        )


# ---------------------------------------------------------------------------
# Backend wrapper
# ---------------------------------------------------------------------------
class PythonBackend(BackendBase):
    """Backend that translates Rust AST to Python ast and execs it."""

    name = "python"

    def __init__(
        self,
        *,
        trusted: bool = True,
        sandbox: bool = False,
        target_width: int | None = None,
    ) -> None:
        super().__init__(trusted=trusted)
        if target_width is not None and target_width not in (32, 64):
            raise ValueError(
                f"target width must be 32 or 64, got {target_width}"
            )
        self.sandbox = sandbox
        self.target_width = (
            HOST_TARGET_WIDTH if target_width is None else target_width
        )
        self.translator = Translator()

    def compile(self, program: Program) -> dict:
        self._check_trust()
        return self.translator.compile_rust(
            program,
            sandbox=self.sandbox,
            target_width=self.target_width,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_rust(program: Program) -> dict:
    """Translate a parsed Rust Program to Python and exec it."""
    return Translator().compile_rust(program)


def translate_rust(program: Program) -> ast.Module:
    """Translate a parsed Rust Program to a Python ast.Module (no exec)."""
    return Translator().translate_program(program)
