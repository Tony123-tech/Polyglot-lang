"""A small proof-carrying IR with interval, nullability, and effect domains.

The IR is intentionally tiny and explicit.  Nodes have stable ids, typed
inputs, optional refinement predicates, and optional effect labels.  A
certificate records a predicate, the domains used, and a digest of the node
shape it was attached to.

Abstract interpretation is a forward fixpoint over the three domains:

* intervals (with sound widening and an explicit ``Unknown`` top),
* nullability (``nonnull`` / ``null`` / ``maybe``),
* linearity/effects (use counts and effect labels).

No pass output is trusted: :mod:`compiler.proof_ir.certificates` recomputes
the abstract state and re-checks every certificate after each transformation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NoReturn

from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
DomainName = Literal["interval", "nullability", "linearity", "effects"]
Nullability = Literal["nonnull", "null", "maybe"]
Number = int | float

_INT_RANGES: dict[str, tuple[int, int]] = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "isize": (-(2**63), 2**63 - 1),
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "usize": (0, 2**64 - 1),
}
_FLOAT_TYPES = frozenset({"f32", "f64", "float"})
_BOOL_TYPES = frozenset({"bool", "boolean"})


def _fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    span: Span = None,
) -> NoReturn:
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "strengthen the refinement, declare the dependency, or raise the bound",
        note="compiler.proof_ir/ir",
        span=span,
    )


def _text(value: object, label: str, *, span: Span = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("E003", f"{label} must be a non-empty string", span=span)
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A sound integer/float interval; ``lo > hi`` denotes bottom."""

    lo: Number = -math.inf
    hi: Number = math.inf

    @property
    def is_empty(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo == -math.inf and self.hi == math.inf

    @property
    def is_unbounded(self) -> bool:
        return self.lo == -math.inf or self.hi == math.inf

    @property
    def singleton(self) -> Number | None:
        if not self.is_empty and self.lo == self.hi:
            return self.lo
        return None

    def contains(self, value: Number) -> bool:
        return not self.is_empty and self.lo <= value <= self.hi

    def contains_zero(self) -> bool:
        return self.contains(0)

    def subset_of(self, other: Interval) -> bool:
        if self.is_empty:
            return True
        if other.is_empty:
            return False
        return self.lo >= other.lo and self.hi <= other.hi

    def intersects(self, other: Interval) -> bool:
        return (
            not self.is_empty
            and not other.is_empty
            and max(self.lo, other.lo) <= min(self.hi, other.hi)
        )

    def join(self, other: Interval) -> Interval:
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return Interval(lo, hi) if lo <= hi else Interval(1, 0)

    def widen(self, other: Interval) -> Interval:
        if self.is_empty:
            return other
        lo = -math.inf if other.lo < self.lo else self.lo
        hi = math.inf if other.hi > self.hi else self.hi
        return Interval(lo, hi)

    def clamp(self, type_name: str) -> Interval:
        if type_name in _INT_RANGES:
            low, high = _INT_RANGES[type_name]
            return self.meet(Interval(low, high))
        if type_name in _BOOL_TYPES:
            return self.meet(Interval(0, 1))
        return self

    def as_dict(self) -> dict[str, Any]:
        if self.is_empty:
            return {"empty": True}
        return {
            "lo": None if self.lo == -math.inf else self.lo,
            "hi": None if self.hi == math.inf else self.hi,
        }

    def __str__(self) -> str:
        if self.is_empty:
            return "empty"
        lo = "-inf" if self.lo == -math.inf else str(self.lo)
        hi = "inf" if self.hi == math.inf else str(self.hi)
        return f"[{lo}, {hi}]"


def _add_bound(left: Number, right: Number, fallback: Number) -> Number:
    value = left + right
    return fallback if isinstance(value, float) and math.isnan(value) else value


def _mul_bound(left: Number, right: Number) -> Number:
    if left == 0 or right == 0:
        return 0
    return left * right


def _div_bound(left: Number, right: Number) -> Number:
    if right == 0:
        return 0
    return left / right


def interval_add(left: Interval, right: Interval) -> Interval:
    if left.is_empty or right.is_empty:
        return Interval(1, 0)
    return Interval(
        _add_bound(left.lo, right.lo, -math.inf),
        _add_bound(left.hi, right.hi, math.inf),
    )


def interval_neg(value: Interval) -> Interval:
    if value.is_empty:
        return value
    return Interval(-value.hi, -value.lo)


def interval_sub(left: Interval, right: Interval) -> Interval:
    return interval_add(left, interval_neg(right))


def interval_mul(left: Interval, right: Interval) -> Interval:
    if left.is_empty or right.is_empty:
        return Interval(1, 0)
    if left.singleton == 0 or right.singleton == 0:
        return Interval(0, 0)
    if left.is_unbounded and left.contains_zero():
        return Interval()
    if right.is_unbounded and right.contains_zero():
        return Interval()
    products = (
        _mul_bound(left.lo, right.lo),
        _mul_bound(left.lo, right.hi),
        _mul_bound(left.hi, right.lo),
        _mul_bound(left.hi, right.hi),
    )
    return Interval(min(products), max(products))


def interval_div(left: Interval, right: Interval) -> Interval:
    if left.is_empty or right.is_empty or right.contains_zero():
        return Interval()
    quotients = (
        _div_bound(left.lo, right.lo),
        _div_bound(left.lo, right.hi),
        _div_bound(left.hi, right.lo),
        _div_bound(left.hi, right.hi),
    )
    return Interval(min(quotients), max(quotients))


# ---------------------------------------------------------------------------
# Expressions and predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarExpr:
    name: str


@dataclass(frozen=True)
class ConstExpr:
    value: Number


@dataclass(frozen=True)
class AddExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class SubExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class MulExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class DivExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class NegExpr:
    value: Expr


type Expr = VarExpr | ConstExpr | AddExpr | SubExpr | MulExpr | DivExpr | NegExpr


@dataclass(frozen=True)
class EqExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class NeExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class LeExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class LtExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class GeExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class GtExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class And:
    parts: tuple[Predicate, ...]


@dataclass(frozen=True)
class Or:
    parts: tuple[Predicate, ...]


@dataclass(frozen=True)
class Not:
    part: Predicate


@dataclass(frozen=True)
class NotNull:
    var: str


@dataclass(frozen=True)
class IsNull:
    var: str


@dataclass(frozen=True)
class LinearOnce:
    resource: str


@dataclass(frozen=True)
class LinearAtMostOnce:
    resource: str


@dataclass(frozen=True)
class EffectAbsent:
    resource: str
    effect: str


@dataclass(frozen=True)
class Truth:
    value: bool


type Predicate = (
    EqExpr
    | NeExpr
    | LeExpr
    | LtExpr
    | GeExpr
    | GtExpr
    | And
    | Or
    | Not
    | NotNull
    | IsNull
    | LinearOnce
    | LinearAtMostOnce
    | EffectAbsent
    | Truth
)


def expr_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, VarExpr):
        return {"kind": "var", "name": expr.name}
    if isinstance(expr, ConstExpr):
        return {"kind": "const", "value": expr.value}
    if isinstance(expr, NegExpr):
        return {"kind": "neg", "value": expr_to_dict(expr.value)}
    if isinstance(expr, AddExpr | SubExpr | MulExpr | DivExpr):
        return {
            "kind": type(expr).__name__.removesuffix("Expr").lower(),
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    _fail("E003", f"unknown expression {type(expr).__name__}")


def predicate_to_dict(predicate: Predicate) -> dict[str, Any]:
    if isinstance(predicate, EqExpr | NeExpr | LeExpr | LtExpr | GeExpr | GtExpr):
        return {
            "kind": type(predicate).__name__.removesuffix("Expr").lower(),
            "left": expr_to_dict(predicate.left),
            "right": expr_to_dict(predicate.right),
        }
    if isinstance(predicate, And | Or):
        return {
            "kind": type(predicate).__name__.lower(),
            "parts": [predicate_to_dict(part) for part in predicate.parts],
        }
    if isinstance(predicate, Not):
        return {"kind": "not", "part": predicate_to_dict(predicate.part)}
    if isinstance(predicate, NotNull):
        return {"kind": "not_null", "var": predicate.var}
    if isinstance(predicate, IsNull):
        return {"kind": "is_null", "var": predicate.var}
    if isinstance(predicate, LinearOnce):
        return {"kind": "linear_once", "resource": predicate.resource}
    if isinstance(predicate, LinearAtMostOnce):
        return {"kind": "linear_at_most_once", "resource": predicate.resource}
    if isinstance(predicate, EffectAbsent):
        return {
            "kind": "effect_absent",
            "resource": predicate.resource,
            "effect": predicate.effect,
        }
    if isinstance(predicate, Truth):
        return {"kind": "truth", "value": predicate.value}
    _fail("E003", f"unknown predicate {type(predicate).__name__}")


def expr_vars(expr: Expr) -> frozenset[str]:
    if isinstance(expr, VarExpr):
        return frozenset({expr.name})
    if isinstance(expr, ConstExpr):
        return frozenset()
    if isinstance(expr, NegExpr):
        return expr_vars(expr.value)
    return expr_vars(expr.left) | expr_vars(expr.right)


def predicate_vars(predicate: Predicate) -> frozenset[str]:
    if isinstance(predicate, EqExpr | NeExpr | LeExpr | LtExpr | GeExpr | GtExpr):
        return expr_vars(predicate.left) | expr_vars(predicate.right)
    if isinstance(predicate, And | Or):
        result: frozenset[str] = frozenset()
        for part in predicate.parts:
            result |= predicate_vars(part)
        return result
    if isinstance(predicate, Not):
        return predicate_vars(predicate.part)
    if isinstance(predicate, NotNull | IsNull):
        return frozenset({predicate.var})
    return frozenset()


def eval_expr_interval(
    expr: Expr,
    values: Mapping[str, AbstractValue],
) -> Interval:
    if isinstance(expr, VarExpr):
        value = values.get(expr.name)
        return value.interval if value is not None else Interval()
    if isinstance(expr, ConstExpr):
        return Interval(expr.value, expr.value)
    if isinstance(expr, NegExpr):
        return interval_neg(eval_expr_interval(expr.value, values))
    left = eval_expr_interval(expr.left, values)
    right = eval_expr_interval(expr.right, values)
    if isinstance(expr, AddExpr):
        return interval_add(left, right)
    if isinstance(expr, SubExpr):
        return interval_sub(left, right)
    if isinstance(expr, MulExpr):
        return interval_mul(left, right)
    if isinstance(expr, DivExpr):
        return interval_div(left, right)
    _fail("E003", f"unknown expression {type(expr).__name__}")


def eval_expr_concrete(
    expr: Expr,
    values: Mapping[str, Any],
) -> Any:
    if isinstance(expr, VarExpr):
        return values.get(expr.name)
    if isinstance(expr, ConstExpr):
        return expr.value
    if isinstance(expr, NegExpr):
        return -eval_expr_concrete(expr.value, values)
    left = eval_expr_concrete(expr.left, values)
    right = eval_expr_concrete(expr.right, values)
    if isinstance(expr, AddExpr):
        return left + right
    if isinstance(expr, SubExpr):
        return left - right
    if isinstance(expr, MulExpr):
        return left * right
    if isinstance(expr, DivExpr):
        return (
            left // right
            if isinstance(left, int) and isinstance(right, int)
            else left / right
        )
    _fail("E003", f"unknown expression {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Abstract values and states
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractValue:
    interval: Interval = field(default_factory=Interval)
    nullability: Nullability = "maybe"
    type_name: str = "i64"

    def join(self, other: AbstractValue) -> AbstractValue:
        if self.nullability == other.nullability:
            nullability = self.nullability
        else:
            nullability = "maybe"
        return AbstractValue(
            interval=self.interval.join(other.interval),
            nullability=nullability,
            type_name=self.type_name
            if self.type_name == other.type_name
            else "unknown",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval.as_dict(),
            "nullability": self.nullability,
            "type_name": self.type_name,
        }


@dataclass(frozen=True)
class EffectState:
    counts: Mapping[str, int] = field(default_factory=dict)
    events: frozenset[str] = frozenset()

    def join(self, other: EffectState) -> EffectState:
        keys = set(self.counts) | set(other.counts)
        counts = {
            key: max(self.counts.get(key, 0), other.counts.get(key, 0)) for key in keys
        }
        return EffectState(counts=counts, events=self.events | other.events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "events": sorted(self.events),
        }


@dataclass(frozen=True)
class AbstractState:
    values: Mapping[str, AbstractValue] = field(default_factory=dict)
    effects: EffectState = field(default_factory=EffectState)
    track_effects: bool = False
    reachable: bool = True

    def value(self, name: str) -> AbstractValue:
        return self.values.get(name, AbstractValue())

    def join(self, other: AbstractState) -> AbstractState:
        if not self.reachable:
            return other
        if not other.reachable:
            return self
        names = set(self.values) | set(other.values)
        values = {name: self.value(name).join(other.value(name)) for name in names}
        return AbstractState(
            values=values,
            effects=self.effects.join(other.effects),
            track_effects=self.track_effects and other.track_effects,
            reachable=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": {
                name: value.as_dict() for name, value in sorted(self.values.items())
            },
            "effects": self.effects.as_dict(),
            "track_effects": self.track_effects,
            "reachable": self.reachable,
        }


def default_value(type_name: str) -> AbstractValue:
    if type_name.endswith("?") or type_name.lower() in {"option", "optional"}:
        return AbstractValue(Interval(), "maybe", type_name)
    if type_name in _INT_RANGES:
        low, high = _INT_RANGES[type_name]
        return AbstractValue(Interval(low, high), "nonnull", type_name)
    if type_name in _BOOL_TYPES:
        return AbstractValue(Interval(0, 1), "nonnull", type_name)
    if type_name in _FLOAT_TYPES:
        return AbstractValue(Interval(), "nonnull", type_name)
    if type_name in {"str", "String", "string"}:
        return AbstractValue(Interval(), "nonnull", type_name)
    return AbstractValue(Interval(), "maybe", type_name)


def _with_interval(
    state: AbstractState, name: str, interval: Interval
) -> AbstractState:
    values = dict(state.values)
    current = values.get(name, AbstractValue())
    values[name] = replace(current, interval=interval)
    return replace(state, values=values)


def _with_nullability(
    state: AbstractState, name: str, nullability: Nullability
) -> AbstractState:
    values = dict(state.values)
    current = values.get(name, AbstractValue())
    values[name] = replace(current, nullability=nullability)
    return replace(state, values=values)


def _const_value(expr: Expr) -> Number | None:
    if isinstance(expr, ConstExpr):
        return expr.value
    if isinstance(expr, NegExpr):
        inner = _const_value(expr.value)
        return None if inner is None else -inner
    return None


def refine_state(state: AbstractState, predicate: Predicate) -> AbstractState | None:
    """Refine *state* by *predicate*; ``None`` means unreachable."""
    if not state.reachable:
        return state
    if isinstance(predicate, Truth):
        return state if predicate.value else None
    if isinstance(predicate, And):
        current: AbstractState | None = state
        for part in predicate.parts:
            if current is None:
                return None
            current = refine_state(current, part)
        return current
    if isinstance(predicate, Or):
        return state
    if isinstance(predicate, Not):
        inner = predicate.part
        if isinstance(inner, NotNull):
            return _with_nullability(state, inner.var, "null")
        if isinstance(inner, IsNull):
            return _with_nullability(state, inner.var, "nonnull")
        if isinstance(inner, Truth):
            return state if not inner.value else None
        return state
    if isinstance(predicate, NotNull):
        return _with_nullability(state, predicate.var, "nonnull")
    if isinstance(predicate, IsNull):
        return _with_nullability(state, predicate.var, "null")
    if isinstance(predicate, EqExpr):
        value = _const_value(predicate.right)
        if isinstance(predicate.left, VarExpr) and value is not None:
            return _with_interval(state, predicate.left.name, Interval(value, value))
        value = _const_value(predicate.left)
        if isinstance(predicate.right, VarExpr) and value is not None:
            return _with_interval(state, predicate.right.name, Interval(value, value))
        return state
    if isinstance(predicate, NeExpr):
        value = _const_value(predicate.right)
        if isinstance(predicate.left, VarExpr) and value is not None:
            current = state.value(predicate.left.name).interval
            if current.singleton == value:
                return None
        value = _const_value(predicate.left)
        if isinstance(predicate.right, VarExpr) and value is not None:
            current = state.value(predicate.right.name).interval
            if current.singleton == value:
                return None
        return state
    if isinstance(predicate, LeExpr | LtExpr | GeExpr | GtExpr):
        left_var = predicate.left if isinstance(predicate.left, VarExpr) else None
        right_var = predicate.right if isinstance(predicate.right, VarExpr) else None
        left_const = _const_value(predicate.left)
        right_const = _const_value(predicate.right)
        if left_var is not None and right_const is not None:
            current = state.value(left_var.name).interval
            if isinstance(predicate, LeExpr):
                return _with_interval(
                    state, left_var.name, current.meet(Interval(-math.inf, right_const))
                )
            if isinstance(predicate, LtExpr):
                bound = right_const - 1 if isinstance(right_const, int) else right_const
                return _with_interval(
                    state, left_var.name, current.meet(Interval(-math.inf, bound))
                )
            if isinstance(predicate, GeExpr):
                return _with_interval(
                    state, left_var.name, current.meet(Interval(right_const, math.inf))
                )
            if isinstance(predicate, GtExpr):
                bound = right_const + 1 if isinstance(right_const, int) else right_const
                return _with_interval(
                    state, left_var.name, current.meet(Interval(bound, math.inf))
                )
        if right_var is not None and left_const is not None:
            current = state.value(right_var.name).interval
            if isinstance(predicate, LeExpr):
                return _with_interval(
                    state, right_var.name, current.meet(Interval(left_const, math.inf))
                )
            if isinstance(predicate, LtExpr):
                bound = left_const + 1 if isinstance(left_const, int) else left_const
                return _with_interval(
                    state, right_var.name, current.meet(Interval(bound, math.inf))
                )
            if isinstance(predicate, GeExpr):
                return _with_interval(
                    state, right_var.name, current.meet(Interval(-math.inf, left_const))
                )
            if isinstance(predicate, GtExpr):
                bound = left_const - 1 if isinstance(left_const, int) else left_const
                return _with_interval(
                    state, right_var.name, current.meet(Interval(-math.inf, bound))
                )
        return state
    return state


def eval_predicate_abstract(predicate: Predicate, state: AbstractState) -> Truth:
    if not state.reachable:
        return "true"
    if isinstance(predicate, Truth):
        return "true" if predicate.value else "false"
    if isinstance(predicate, And):
        truths = [eval_predicate_abstract(part, state) for part in predicate.parts]
        if any(item == "false" for item in truths):
            return "false"
        return "true" if all(item == "true" for item in truths) else "unknown"
    if isinstance(predicate, Or):
        truths = [eval_predicate_abstract(part, state) for part in predicate.parts]
        if any(item == "true" for item in truths):
            return "true"
        return "false" if all(item == "false" for item in truths) else "unknown"
    if isinstance(predicate, Not):
        inner = eval_predicate_abstract(predicate.part, state)
        if inner == "true":
            return "false"
        if inner == "false":
            return "true"
        return "unknown"
    if isinstance(predicate, NotNull):
        nullability = state.value(predicate.var).nullability
        return (
            "true"
            if nullability == "nonnull"
            else "false"
            if nullability == "null"
            else "unknown"
        )
    if isinstance(predicate, IsNull):
        nullability = state.value(predicate.var).nullability
        return (
            "true"
            if nullability == "null"
            else "false"
            if nullability == "nonnull"
            else "unknown"
        )
    if isinstance(predicate, LinearOnce | LinearAtMostOnce):
        if not state.track_effects:
            return "unknown"
        count = state.effects.counts.get(predicate.resource, 0)
        if isinstance(predicate, LinearOnce):
            return "true" if count == 1 else "false"
        return "true" if count <= 1 else "false"
    if isinstance(predicate, EffectAbsent):
        if not state.track_effects:
            return "unknown"
        label = f"{predicate.effect}:{predicate.resource}"
        return "false" if label in state.effects.events else "true"
    if isinstance(predicate, EqExpr):
        left = eval_expr_interval(predicate.left, state.values)
        right = eval_expr_interval(predicate.right, state.values)
        if left.is_empty or right.is_empty:
            return "false"
        if not left.intersects(right):
            return "false"
        left_singleton, right_singleton = left.singleton, right.singleton
        if left_singleton is not None and right_singleton is not None:
            return "true" if left_singleton == right_singleton else "false"
        return "unknown"
    if isinstance(predicate, NeExpr):
        inner = eval_predicate_abstract(EqExpr(predicate.left, predicate.right), state)
        return "false" if inner == "true" else "true" if inner == "false" else "unknown"
    if isinstance(predicate, LeExpr | LtExpr | GeExpr | GtExpr):
        left = eval_expr_interval(predicate.left, state.values)
        right = eval_expr_interval(predicate.right, state.values)
        if left.is_empty or right.is_empty:
            return "true"
        if isinstance(predicate, LeExpr):
            if left.hi <= right.lo:
                return "true"
            return "false" if left.lo > right.hi else "unknown"
        if isinstance(predicate, LtExpr):
            if left.hi < right.lo:
                return "true"
            return "false" if left.lo >= right.hi else "unknown"
        if isinstance(predicate, GeExpr):
            if left.lo >= right.hi:
                return "true"
            return "false" if left.hi < right.lo else "unknown"
        if isinstance(predicate, GtExpr):
            if left.lo > right.hi:
                return "true"
            return "false" if left.hi <= right.lo else "unknown"
    _fail("E003", f"unknown predicate {type(predicate).__name__}")


# ---------------------------------------------------------------------------
# IR nodes and certificates
# ---------------------------------------------------------------------------


type Op = Literal[
    "const",
    "param",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "ite",
    "phi",
    "assign",
    "assume",
    "assert",
    "call",
    "load",
    "store",
    "cast",
    "return",
    "effect",
    "nop",
]
_VALID_OPS: frozenset[str] = frozenset(
    {
        "const",
        "param",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "ite",
        "phi",
        "assign",
        "assume",
        "assert",
        "call",
        "load",
        "store",
        "cast",
        "return",
        "effect",
        "nop",
    }
)


@dataclass(frozen=True)
class InvariantCertificate:
    """A machine-checkable invariant attached to one IR node."""

    node_id: str
    predicate: Predicate
    domains: tuple[DomainName, ...]
    node_digest: str
    evidence: str
    preconditions: tuple[Predicate, ...] = ()
    postconditions: tuple[Predicate, ...] = ()
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "predicate": predicate_to_dict(self.predicate),
            "domains": list(self.domains),
            "node_digest": self.node_digest,
            "evidence": self.evidence,
            "preconditions": [predicate_to_dict(item) for item in self.preconditions],
            "postconditions": [predicate_to_dict(item) for item in self.postconditions],
            "span": list(self.span) if self.span is not None else None,
        }


@dataclass(frozen=True)
class IRNode:
    node_id: str
    op: Op
    inputs: tuple[str, ...] = ()
    value: Any = None
    type_name: str = "i64"
    predicate: Predicate | None = None
    effects: tuple[str, ...] = ()
    certificate: InvariantCertificate | None = None
    span: Span = None

    def __post_init__(self) -> None:
        _text(self.node_id, "node_id", span=self.span)
        if self.op not in _VALID_OPS:
            _fail("E003", f"unknown IR op {self.op!r}", span=self.span)
        for item in self.inputs:
            _text(item, "input node id", span=self.span)
        for item in self.effects:
            _text(item, "effect label", span=self.span)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "inputs": list(self.inputs),
            "value": _jsonable(self.value),
            "type_name": self.type_name,
            "predicate": (
                predicate_to_dict(self.predicate)
                if self.predicate is not None
                else None
            ),
            "effects": list(self.effects),
            "certificate": (
                self.certificate.as_dict() if self.certificate is not None else None
            ),
            "span": list(self.span) if self.span is not None else None,
        }


def node_signature(node: IRNode) -> str:
    payload = {
        "node_id": node.node_id,
        "op": node.op,
        "inputs": list(node.inputs),
        "value": _jsonable(node.value),
        "type_name": node.type_name,
        "predicate": (
            predicate_to_dict(node.predicate) if node.predicate is not None else None
        ),
        "effects": list(node.effects),
        "span": list(node.span) if node.span is not None else None,
    }
    return _digest(payload)


def make_certificate(
    node: IRNode,
    predicate: Predicate,
    *,
    domains: Sequence[DomainName],
    evidence: str,
    preconditions: Sequence[Predicate] = (),
    postconditions: Sequence[Predicate] = (),
) -> InvariantCertificate:
    return InvariantCertificate(
        node_id=node.node_id,
        predicate=predicate,
        domains=tuple(domains),
        node_digest=node_signature(node),
        evidence=evidence,
        preconditions=tuple(preconditions),
        postconditions=tuple(postconditions),
        span=node.span,
    )


@dataclass(frozen=True)
class IRPass:
    """A transformation over a whole IR node sequence."""

    name: str
    transform: Callable[[tuple[IRNode, ...]], Sequence[IRNode]]
    description: str = ""

    def __post_init__(self) -> None:
        _text(self.name, "pass name")
        if not callable(self.transform):
            _fail("E003", "pass transform must be callable")


def identity_pass() -> IRPass:
    return IRPass("identity", lambda nodes: tuple(nodes), "no-op pass")


def constant_fold_pass() -> IRPass:
    def transform(nodes: tuple[IRNode, ...]) -> tuple[IRNode, ...]:
        by_id = {node.node_id: node for node in nodes}
        output: list[IRNode] = []
        for node in nodes:
            if node.op in {"add", "sub", "mul", "div", "neg"}:
                values: list[Number] = []
                for input_id in node.inputs:
                    source = by_id.get(input_id)
                    if source is None or source.op != "const":
                        values = []
                        break
                    if not isinstance(source.value, (int, float)):
                        values = []
                        break
                    values.append(source.value)
                if values:
                    try:
                        if node.op == "add":
                            result = values[0] + values[1]
                        elif node.op == "sub":
                            result = values[0] - values[1]
                        elif node.op == "mul":
                            result = values[0] * values[1]
                        elif node.op == "div":
                            if values[1] == 0:
                                raise ZeroDivisionError
                            result = values[0] / values[1]
                        else:
                            result = -values[0]
                    except IndexError, ZeroDivisionError, TypeError:
                        output.append(node)
                        continue
                    output.append(
                        IRNode(
                            node.node_id,
                            "const",
                            value=result,
                            type_name=node.type_name,
                            span=node.span,
                        )
                    )
                    continue
            output.append(node)
        return tuple(output)

    return IRPass(
        "constant-fold",
        transform,
        "fold arithmetic on constant inputs; stale certificates are dropped",
    )


def dead_code_elimination_pass() -> IRPass:
    def transform(nodes: tuple[IRNode, ...]) -> tuple[IRNode, ...]:
        by_id = {node.node_id: node for node in nodes}
        sinks = [
            node
            for node in nodes
            if node.op in {"return", "assert", "effect", "store", "call"}
        ]
        live: set[str] = set()
        stack = [node.node_id for node in sinks]
        while stack:
            current = stack.pop()
            if current in live:
                continue
            live.add(current)
            node = by_id.get(current)
            if node is not None:
                stack.extend(node.inputs)
        return tuple(node for node in nodes if node.node_id in live)

    return IRPass(
        "dead-code-elimination",
        transform,
        "remove nodes not reachable from a return/assert/effect/store/call sink",
    )


# ---------------------------------------------------------------------------
# Abstract interpretation
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    states: dict[str, AbstractState]
    before: dict[str, AbstractState]
    iterations: int
    converged: bool
    diagnostics: list[dict[str, str]]


def _validate_nodes(nodes: Sequence[IRNode], *, span: Span = None) -> dict[str, IRNode]:
    by_id: dict[str, IRNode] = {}
    for node in nodes:
        if not isinstance(node, IRNode):
            _fail("E003", "nodes must contain IRNode values", span=span)
        if node.node_id in by_id:
            _fail(
                "E008",
                f"duplicate IR node id {node.node_id!r}",
                span=node.span or span,
            )
        by_id[node.node_id] = node
    for node in nodes:
        for input_id in node.inputs:
            if input_id not in by_id:
                _fail(
                    "E006",
                    f"IR node {node.node_id!r} references missing input {input_id!r}",
                    span=node.span or span,
                )
    return by_id


def _apply_effects(state: AbstractState, effects: Sequence[str]) -> AbstractState:
    if not state.track_effects or not effects:
        return state
    counts = dict(state.effects.counts)
    events = set(state.effects.events)
    for label in effects:
        if ":" not in label:
            _fail("E003", f"effect label {label!r} must have the form action:resource")
        action, resource = label.split(":", 1)
        if not action or not resource:
            _fail("E003", f"effect label {label!r} must have the form action:resource")
        if action in {"use", "dup"}:
            counts[resource] = counts.get(resource, 0) + 1
        if action in {"close", "release", "dup", "drop"}:
            events.add(label)
    return replace(state, effects=EffectState(counts=counts, events=frozenset(events)))


def _transfer(
    node: IRNode,
    state: AbstractState,
    known: Mapping[str, AbstractState],
    previous: Mapping[str, AbstractState],
    *,
    span: Span,
) -> AbstractState:
    if not state.reachable:
        return state
    values = state.values
    if node.op == "const":
        if not isinstance(node.value, (int, float)):
            _fail(
                "E008",
                f"const node {node.node_id!r} needs a numeric value",
                span=node.span or span,
            )
        result = AbstractValue(
            Interval(node.value, node.value), "nonnull", node.type_name
        )
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op == "param":
        result = default_value(node.type_name)
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op in {"add", "sub", "mul", "div", "neg"}:
        if node.op == "neg":
            if len(node.inputs) != 1:
                _fail(
                    "E003",
                    f"neg node {node.node_id!r} needs one input",
                    span=node.span or span,
                )
            expression: Expr = NegExpr(VarExpr(node.inputs[0]))
        else:
            if len(node.inputs) != 2:
                _fail(
                    "E003",
                    f"{node.op} node {node.node_id!r} needs two inputs",
                    span=node.span or span,
                )
            left, right = node.inputs
            expression = {
                "add": AddExpr,
                "sub": SubExpr,
                "mul": MulExpr,
                "div": DivExpr,
            }[node.op](VarExpr(left), VarExpr(right))
        interval = eval_expr_interval(expression, values).clamp(node.type_name)
        result = AbstractValue(interval, "nonnull", node.type_name)
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op in {"assign", "cast", "return", "store", "nop"}:
        source = node.inputs[0] if node.inputs else None
        base = (
            state.value(source) if source is not None else default_value(node.type_name)
        )
        if node.op == "cast":
            base = replace(
                base,
                interval=base.interval.clamp(node.type_name),
                type_name=node.type_name,
            )
        result = replace(base, type_name=node.type_name)
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op == "phi":
        if not node.inputs:
            result = default_value(node.type_name)
        else:
            result = state.value(node.inputs[0])
            for input_id in node.inputs[1:]:
                result = result.join(state.value(input_id))
            for input_id in node.inputs:
                if input_id not in known:
                    prior = previous.get(input_id)
                    if prior is not None:
                        result = result.join(prior.value(input_id))
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op == "ite":
        if node.predicate is None or len(node.inputs) != 2:
            _fail(
                "E003",
                f"ite node {node.node_id!r} needs a predicate and two inputs",
                span=node.span or span,
            )
        then_id, else_id = node.inputs
        truth = eval_predicate_abstract(node.predicate, state)
        if truth == "true":
            result = state.value(then_id)
        elif truth == "false":
            result = state.value(else_id)
        else:
            result = state.value(then_id).join(state.value(else_id))
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op == "assume":
        if node.predicate is None:
            _fail(
                "E003",
                f"assume node {node.node_id!r} needs a predicate",
                span=node.span or span,
            )
        refined = refine_state(state, node.predicate)
        if refined is None:
            return replace(state, reachable=False)
        return _apply_effects(refined, node.effects)
    if node.op == "assert":
        if node.predicate is None:
            _fail(
                "E003",
                f"assert node {node.node_id!r} needs a predicate",
                span=node.span or span,
            )
        return _apply_effects(state, node.effects)
    if node.op in {"call", "load"}:
        result = default_value(node.type_name)
        return _apply_effects(
            replace(state, values={**values, node.node_id: result}), node.effects
        )
    if node.op == "effect":
        return _apply_effects(state, node.effects)
    _fail("E003", f"unsupported IR op {node.op!r}", span=node.span or span)


def infer_abstract_states(
    nodes: Sequence[IRNode],
    *,
    assumptions: Mapping[str, AbstractValue] | None = None,
    domains: Sequence[DomainName] = ("interval", "nullability", "effects"),
    max_iterations: int = 64,
    span: Span = None,
) -> InferenceResult:
    """Run a forward abstract interpretation to a bounded fixpoint."""
    if max_iterations <= 0:
        _fail("E003", "max_iterations must be positive", span=span)
    _validate_nodes(nodes, span=span)
    initial = AbstractState(
        values=dict(assumptions or {}),
        track_effects="effects" in domains or "linearity" in domains,
    )
    previous: dict[str, AbstractState] = {}
    diagnostics: list[dict[str, str]] = []
    for iteration in range(1, max_iterations + 1):
        states: dict[str, AbstractState] = {}
        before: dict[str, AbstractState] = {}
        current = initial
        for node in nodes:
            before[node.node_id] = current
            computed = _transfer(
                node, current, states, previous, span=node.span or span
            )
            prior = previous.get(node.node_id)
            if prior is not None and node.op in {"phi"}:
                computed = prior.join(computed)
            states[node.node_id] = computed
            if node.op == "assert" and node.predicate is not None:
                truth = eval_predicate_abstract(node.predicate, computed)
                if truth == "false":
                    diagnostics.append(
                        {
                            "code": "E002",
                            "message": f"assertion {node.node_id!r} is definitely false",
                        }
                    )
                elif truth == "unknown":
                    diagnostics.append(
                        {
                            "code": "E002",
                            "message": f"assertion {node.node_id!r} is Unknown",
                        }
                    )
            current = computed
        if states == previous:
            return InferenceResult(states, before, iteration, True, diagnostics)
        previous = states
    return InferenceResult(previous, {}, max_iterations, False, diagnostics)


__all__ = [
    "AbstractState",
    "AbstractValue",
    "AddExpr",
    "And",
    "ConstExpr",
    "DivExpr",
    "DomainName",
    "EffectAbsent",
    "EffectState",
    "EqExpr",
    "Expr",
    "GeExpr",
    "GtExpr",
    "IRNode",
    "IRPass",
    "InferenceResult",
    "Interval",
    "InvariantCertificate",
    "IsNull",
    "LeExpr",
    "LinearAtMostOnce",
    "LinearOnce",
    "LtExpr",
    "MulExpr",
    "NeExpr",
    "NegExpr",
    "Not",
    "NotNull",
    "Op",
    "Or",
    "Predicate",
    "SubExpr",
    "Truth",
    "VarExpr",
    "constant_fold_pass",
    "dead_code_elimination_pass",
    "default_value",
    "eval_expr_concrete",
    "eval_expr_interval",
    "eval_predicate_abstract",
    "expr_to_dict",
    "identity_pass",
    "infer_abstract_states",
    "interval_add",
    "interval_div",
    "interval_mul",
    "interval_neg",
    "interval_sub",
    "make_certificate",
    "node_signature",
    "predicate_to_dict",
    "predicate_vars",
    "refine_state",
]
