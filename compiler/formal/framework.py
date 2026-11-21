"""Frame H — bounded symbolic execution, model checking, and refinement.

These analyses are deliberately *bounded* and sound in the following narrow
sense:

* a path is pruned only when an interval solver proves its constraints
  infeasible; otherwise it is kept with an explicit ``unknown`` feasibility;
* a divergence witness is returned only after concrete evaluation confirms it;
* a bounded refinement proof is claimed only when every explored path of both
  programs agrees and no bound, timeout, or unsupported boundary was hit; and
* step limits, loop unroll limits, path limits, and solver time budgets are
  always reported as structured diagnostics (``E008`` bounds, ``E004``
  timeouts, ``E003`` unsupported constructs, ``E002`` proof failures).

Nothing here proves unrestricted program equivalence, universal termination,
or absence of aliasing.  :func:`formal_require_bounded_proof` converts an
honest ``unknown`` report into a ``PillarError`` for callers that need a hard
gate.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ast_nodes import (
    Assign,
    Binary,
    Cast,
    CompoundAssign,
    ExprStmt,
    FnDecl,
    Ident,
    IfExpr,
    LetStmt,
    LitBool,
    LitInt,
    Loop,
    Paren,
    PatIdent,
    Return,
    Unary,
    While,
)
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError
from tools.diff_fuzz import normalize_error

Span = tuple[int, int] | None
Feasibility = Literal["feasible", "infeasible", "unknown"]

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
DEFAULT_MAX_STEPS = 64
DEFAULT_MAX_UNROLL = 4
DEFAULT_MAX_PATHS = 64
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_SAMPLES = 24

_COMPARE_OPS = {"<", "<=", ">", ">=", "==", "!="}
_BINARY_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
    "&&": "and",
    "||": "or",
}


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "reduce the bound, split the proof, or inspect the reported witness",
        note="compiler.formal/frame-h",
        span=span,
    )


# ---------------------------------------------------------------------------
# Symbolic expressions and intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymExpr:
    """A small symbolic term over i64 variables."""

    op: str
    value: int | str | bool | None = None
    args: tuple[SymExpr, ...] = ()

    @staticmethod
    def var(name: str) -> SymExpr:
        return SymExpr("var", value=name)

    @staticmethod
    def const(value: int) -> SymExpr:
        return SymExpr("const", value=value)

    @staticmethod
    def boolean(value: bool) -> SymExpr:
        return SymExpr("bool", value=value)

    def render(self) -> str:
        if self.op in {"var", "const", "bool"}:
            return str(self.value)
        if self.op == "neg":
            return f"(-{self.args[0].render()})"
        if self.op == "not":
            return f"(!{self.args[0].render()})"
        if len(self.args) == 2:
            symbol = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
                "eq": "==",
                "ne": "!=",
                "and": "&&",
                "or": "||",
            }.get(self.op, self.op)
            return f"({self.args[0].render()} {symbol} {self.args[1].render()})"
        return f"{self.op}({', '.join(arg.render() for arg in self.args)})"

    def variables(self) -> frozenset[str]:
        if self.op == "var":
            return frozenset({str(self.value)})
        result: set[str] = set()
        for arg in self.args:
            result.update(arg.variables())
        return frozenset(result)

    def evaluate(self, env: Mapping[str, int]) -> Any:
        """Concrete evaluation used only to verify emitted witnesses."""
        if self.op == "var":
            return env[str(self.value)]
        if self.op in {"const", "bool"}:
            return self.value
        if self.op == "neg":
            return -int(self.args[0].evaluate(env))
        if self.op == "not":
            return not bool(self.args[0].evaluate(env))
        left = self.args[0].evaluate(env)
        right = self.args[1].evaluate(env)
        if self.op == "add":
            return left + right
        if self.op == "sub":
            return left - right
        if self.op == "mul":
            return left * right
        if self.op == "lt":
            return left < right
        if self.op == "le":
            return left <= right
        if self.op == "gt":
            return left > right
        if self.op == "ge":
            return left >= right
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        if self.op == "and":
            return bool(left) and bool(right)
        if self.op == "or":
            return bool(left) or bool(right)
        raise ValueError(f"unknown symbolic operator {self.op!r}")


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    def render(self) -> str:
        return f"[{self.lo}, {self.hi}]"


def _substitute(expr: SymExpr, env: Mapping[str, SymExpr]) -> SymExpr:
    if expr.op == "var":
        return env.get(str(expr.value), expr)
    if not expr.args:
        return expr
    return SymExpr(expr.op, expr.value, tuple(_substitute(a, env) for a in expr.args))


# ---------------------------------------------------------------------------
# Structured reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormalDiagnostic:
    code: str
    message: str
    kind: str
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "kind": self.kind,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class SymbolicPath:
    path_id: str
    constraints: tuple[str, ...]
    result: str
    steps: int
    unrolls: dict[str, int]
    feasibility: Feasibility
    truncated: bool = False
    # Keep symbolic atoms for witness minimization; never serialize them.
    constraint_terms: tuple[SymExpr, ...] = field(default=(), repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "constraints": list(self.constraints),
            "result": self.result,
            "steps": self.steps,
            "unrolls": dict(self.unrolls),
            "feasibility": self.feasibility,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class BoundedExecutionReport:
    function: str
    bounds: dict[str, int]
    paths: tuple[SymbolicPath, ...]
    diagnostics: tuple[FormalDiagnostic, ...]
    branch_decisions: tuple[dict[str, Any], ...] = ()
    explored: int = 0
    pruned: int = 0

    @property
    def complete(self) -> bool:
        return not any(diag.kind in {"bound", "timeout"} for diag in self.diagnostics)

    @property
    def proved(self) -> bool:
        return (
            self.complete
            and bool(self.paths)
            and all(
                path.feasibility != "unknown" and not path.truncated
                for path in self.paths
            )
        )

    def digest(self) -> str:
        payload = {
            "function": self.function,
            "paths": [path.as_dict() for path in self.paths],
            "diagnostics": [diag.as_dict() for diag in self.diagnostics],
        }
        return hashlib.sha256(
            __import__("json").dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "bounds": dict(self.bounds),
            "paths": [path.as_dict() for path in self.paths],
            "diagnostics": [diag.as_dict() for diag in self.diagnostics],
            "branch_decisions": [dict(item) for item in self.branch_decisions],
            "explored": self.explored,
            "pruned": self.pruned,
            "complete": self.complete,
            "proved": self.proved,
            "digest": self.digest(),
        }


@dataclass(frozen=True)
class Divergence:
    inputs: dict[str, int]
    left: Any
    right: Any
    left_backend: str = "left"
    right_backend: str = "right"

    def as_dict(self) -> dict[str, Any]:
        return {
            "inputs": dict(self.inputs),
            "left": self.left,
            "right": self.right,
            "left_backend": self.left_backend,
            "right_backend": self.right_backend,
        }


@dataclass(frozen=True)
class RelationalReport:
    left: str
    right: str
    depth: int
    inputs: dict[str, tuple[int, int]]
    checked: int
    proved_bounded: bool
    divergences: tuple[Divergence, ...] = ()
    left_report: BoundedExecutionReport | None = None
    right_report: BoundedExecutionReport | None = None
    diagnostics: tuple[FormalDiagnostic, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "depth": self.depth,
            "inputs": {name: list(bounds) for name, bounds in self.inputs.items()},
            "checked": self.checked,
            "proved_bounded": self.proved_bounded,
            "divergences": [item.as_dict() for item in self.divergences],
            "left_report": self.left_report.as_dict() if self.left_report else None,
            "right_report": self.right_report.as_dict() if self.right_report else None,
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "caveats": (
                "bounded equivalence up to the declared unroll/step depth only",
                "unbounded inputs and loops remain undecidable",
            ),
        }


@dataclass(frozen=True)
class WitnessExplanation:
    path_id: str
    constraints: tuple[str, ...]
    intervals: dict[str, str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "constraints": list(self.constraints),
            "intervals": dict(self.intervals),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InfeasiblePathReport:
    function: str
    witnesses: tuple[WitnessExplanation, ...]
    dead_branches: tuple[dict[str, Any], ...]
    execution: BoundedExecutionReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "witnesses": [item.as_dict() for item in self.witnesses],
            "dead_branches": [dict(item) for item in self.dead_branches],
            "execution": self.execution.as_dict(),
        }


# ---------------------------------------------------------------------------
# Interval solver (conservative over-approximation)
# ---------------------------------------------------------------------------


def _flip(op: str) -> str:
    """Swap operands (``x < y`` becomes ``y > x``)."""
    return {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}[op]


def _negate(op: str) -> str:
    """Negate a comparison (``x < y`` becomes ``x >= y``)."""
    return {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq"}[op]


def _narrow_one(op: str, name: str, value: int, ranges: dict[str, Interval]) -> bool:
    current = ranges.get(name, Interval(I64_MIN, I64_MAX))
    lo, hi = current.lo, current.hi
    if op == "lt":
        hi = min(hi, value - 1)
    elif op == "le":
        hi = min(hi, value)
    elif op == "gt":
        lo = max(lo, value + 1)
    elif op == "ge":
        lo = max(lo, value)
    elif op == "eq":
        lo, hi = max(lo, value), min(hi, value)
    elif op == "ne":
        if lo == hi == value:
            return False
    else:
        return True
    if lo > hi:
        return False
    ranges[name] = Interval(max(lo, I64_MIN), min(hi, I64_MAX))
    return True


def _analyze(expr: SymExpr, ranges: dict[str, Interval]) -> tuple[Feasibility, dict]:
    """Refine ``ranges``; only ``infeasible`` is a proof of contradiction."""
    if expr.op == "bool":
        if expr.value:
            return "feasible", dict(ranges)
        return "infeasible", dict(ranges)
    if expr.op == "not":
        inner = expr.args[0]
        if inner.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
            return _analyze(SymExpr(_negate(inner.op), args=inner.args), dict(ranges))
        return "unknown", dict(ranges)
    if expr.op == "and":
        current = dict(ranges)
        saw_unknown = False
        for arg in expr.args:
            status, current = _analyze(arg, current)
            if status == "infeasible":
                return "infeasible", current
            saw_unknown = saw_unknown or status == "unknown"
        return ("unknown" if saw_unknown else "feasible"), current
    if expr.op == "or":
        statuses: list[Feasibility] = []
        branches: list[dict[str, Interval]] = []
        for arg in expr.args:
            status, refined = _analyze(arg, dict(ranges))
            statuses.append(status)
            branches.append(refined)
        if all(status == "infeasible" for status in statuses):
            return "infeasible", dict(ranges)
        if any(status == "unknown" for status in statuses):
            return "unknown", dict(ranges)
        merged: dict[str, Interval] = {}
        for branch in branches:
            for name, interval in branch.items():
                if name not in merged:
                    merged[name] = interval
                else:
                    old = merged[name]
                    merged[name] = Interval(
                        min(old.lo, interval.lo), max(old.hi, interval.hi)
                    )
        for name, interval in ranges.items():
            merged.setdefault(name, interval)
        return "feasible", merged
    if expr.op in {"lt", "le", "gt", "ge", "eq", "ne"} and len(expr.args) == 2:
        left, right = expr.args
        op = expr.op
        if right.op == "var" and left.op == "const":
            left, right = right, left
            op = _flip(op)
        if left.op == "var" and right.op in {"const", "bool"}:
            refined = dict(ranges)
            if _narrow_one(op, str(left.value), int(right.value), refined):
                return "feasible", refined
            return "infeasible", refined
    return "unknown", dict(ranges)


def _find_witness(
    constraints: Sequence[SymExpr], ranges: Mapping[str, Interval]
) -> dict[str, int] | None:
    """Return a *verified* concrete assignment, or ``None`` (unknown)."""
    names = sorted({name for c in constraints for name in c.variables()} | set(ranges))
    if not names:
        env: dict[str, int] = {}
        if all(bool(c.evaluate(env)) for c in constraints):
            return env
        return None

    def verify(values: dict[str, int]) -> dict[str, int] | None:
        if all(bool(c.evaluate(values)) for c in constraints):
            return values
        return None

    baseline: dict[str, int] = {}
    for name in names:
        interval = ranges.get(name, Interval(-8, 8))
        baseline[name] = interval.lo if abs(interval.lo) <= 1 << 40 else 0
    found = verify(baseline)
    if found is not None:
        return found
    for name in names:
        interval = ranges.get(name, Interval(-8, 8))
        for candidate in {interval.lo, interval.hi, 0, 1, -1}:
            attempt = dict(baseline)
            attempt[name] = candidate
            found = verify(attempt)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Lowering a restricted Polyglot AST subset into the formal IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stmt:
    kind: Literal["assign", "if", "loop", "return"]
    target: str = ""
    expr: SymExpr | None = None
    cond: SymExpr | None = None
    then_body: tuple[Stmt, ...] = ()
    else_body: tuple[Stmt, ...] = ()
    span: Span = None


def _span(node: Any, fallback: Span) -> Span:
    value = getattr(node, "span", None)
    return value if value is not None else fallback


def _lower_expr(expr: Any, span: Span) -> SymExpr:
    if isinstance(expr, Paren):
        return _lower_expr(expr.inner, span)
    if isinstance(expr, LitInt):
        return SymExpr.const(int(expr.value))
    if isinstance(expr, LitBool):
        return SymExpr.boolean(bool(expr.value))
    if isinstance(expr, Ident):
        return SymExpr.var(str(expr.name))
    if isinstance(expr, Unary) and expr.op == "-":
        return SymExpr("neg", args=(_lower_expr(expr.operand, span),))
    if isinstance(expr, Unary) and expr.op in {"!", "not"}:
        return SymExpr("not", args=(_lower_expr(expr.operand, span),))
    if isinstance(expr, Cast):
        return _lower_expr(expr.expr, span)
    if isinstance(expr, Binary) and expr.op in _BINARY_OPS:
        return SymExpr(
            _BINARY_OPS[expr.op],
            args=(_lower_expr(expr.left, span), _lower_expr(expr.right, span)),
        )
    raise _fail(
        "E003",
        f"frame H cannot symbolically lower {type(expr).__name__} expression",
        span=span,
        reason="the formal frame supports a bounded i64/bool expression subset",
        help_text="lower the construct to scalar i64/bool expressions before analysis",
    )


def _lower_block_of_expression(expr: Any, span: Span) -> tuple[Stmt, ...]:
    """Lower an expression used as a statement (parsers wrap these in ExprStmt)."""
    if isinstance(expr, Return):
        return (
            Stmt(
                "return",
                expr=_lower_expr(expr.value, span) if expr.value is not None else None,
                span=span,
            ),
        )
    if isinstance(expr, Assign):
        return (
            Stmt(
                "assign",
                target=_target_name(expr.target, span),
                expr=_lower_expr(expr.value, span),
                span=span,
            ),
        )
    if isinstance(expr, CompoundAssign):
        op = {"+=": "add", "-=": "sub", "*=": "mul"}.get(str(expr.op))
        if op is None:
            raise _fail(
                "E003",
                f"frame H cannot lower compound operator {expr.op!r}",
                span=span,
            )
        return (
            Stmt(
                "assign",
                target=_target_name(expr.target, span),
                expr=SymExpr(
                    op,
                    args=(
                        SymExpr.var(_target_name(expr.target, span)),
                        _lower_expr(expr.value, span),
                    ),
                ),
                span=span,
            ),
        )
    if isinstance(expr, IfExpr):
        return (_lower_if(expr, span),)
    if isinstance(expr, While):
        return (
            Stmt(
                "loop",
                cond=_lower_expr(expr.cond, span),
                then_body=_lower_block(expr.body, span),
                span=span,
            ),
        )
    if isinstance(expr, Loop):
        return (
            Stmt(
                "loop",
                cond=SymExpr.boolean(True),
                then_body=_lower_block(expr.body, span),
                span=span,
            ),
        )
    raise _fail(
        "E003",
        f"frame H cannot lower {type(expr).__name__} as a statement",
        span=span,
    )


def _target_name(target: Any, span: Span) -> str:
    if isinstance(target, Paren):
        return _target_name(target.inner, span)
    if isinstance(target, Ident):
        return str(target.name)
    raise _fail(
        "E003",
        f"frame H cannot symbolically assign to {type(target).__name__}",
        span=span,
        reason="only scalar named bindings are tracked by the interval solver",
    )


def _lower_block(block: Any, fallback: Span) -> tuple[Stmt, ...]:
    statements: list[Stmt] = []
    for stmt in block.stmts:
        span = _span(stmt, fallback)
        if isinstance(stmt, LetStmt):
            if not isinstance(stmt.pattern, PatIdent) or stmt.value is None:
                raise _fail(
                    "E003",
                    "frame H requires a named let binding with an initializer",
                    span=span,
                )
            statements.append(
                Stmt(
                    "assign",
                    target=str(stmt.pattern.name),
                    expr=_lower_expr(stmt.value, span),
                    span=span,
                )
            )
        elif isinstance(stmt, Assign):
            statements.append(
                Stmt(
                    "assign",
                    target=_target_name(stmt.target, span),
                    expr=_lower_expr(stmt.value, span),
                    span=span,
                )
            )
        elif isinstance(stmt, CompoundAssign):
            base = SymExpr.var(_target_name(stmt.target, span))
            value = _lower_expr(stmt.value, span)
            op = {
                "+=": "add",
                "-=": "sub",
                "*=": "mul",
            }.get(str(stmt.op))
            if op is None:
                raise _fail(
                    "E003",
                    f"frame H cannot lower compound operator {stmt.op!r}",
                    span=span,
                )
            statements.append(
                Stmt(
                    "assign",
                    target=_target_name(stmt.target, span),
                    expr=SymExpr(op, args=(base, value)),
                    span=span,
                )
            )
        elif isinstance(stmt, IfExpr):
            statements.append(_lower_if(stmt, fallback))
        elif isinstance(stmt, While):
            statements.append(
                Stmt(
                    "loop",
                    cond=_lower_expr(stmt.cond, span),
                    then_body=_lower_block(stmt.body, span),
                    span=span,
                )
            )
        elif isinstance(stmt, Return):
            statements.append(
                Stmt(
                    "return",
                    expr=_lower_expr(stmt.value, span)
                    if stmt.value is not None
                    else None,
                    span=span,
                )
            )
        elif isinstance(stmt, ExprStmt):
            inner = stmt.expr
            if isinstance(inner, (Return, IfExpr, While, Loop, Assign, CompoundAssign)):
                statements.extend(_lower_block_of_expression(inner, span))
            elif isinstance(
                inner,
                (
                    Ident,
                    LitInt,
                    LitBool,
                    Binary,
                    Unary,
                    Paren,
                    Cast,
                ),
            ):
                # Other scalar expressions have no tracked side effect.
                continue
            else:
                raise _fail(
                    "E003",
                    f"frame H cannot symbolically lower {type(inner).__name__} expression statement",
                    span=span,
                    reason="this construct requires an explicit backend boundary",
                )
        else:
            raise _fail(
                "E003",
                f"frame H cannot symbolically lower {type(stmt).__name__} statement",
                span=span,
                reason="this construct requires an explicit backend boundary",
            )
    if block.tail is not None:
        statements.append(
            Stmt("return", expr=_lower_expr(block.tail, fallback), span=fallback)
        )
    return tuple(statements)


def _lower_if(expr: IfExpr, fallback: Span) -> Stmt:
    span = _span(expr, fallback)
    then_branch = expr.then_block
    else_branch = expr.else_branch
    then_body = (
        _lower_block(then_branch, span)
        if hasattr(then_branch, "stmts")
        else (Stmt("return", expr=_lower_expr(then_branch, span), span=span),)
    )
    if else_branch is None:
        else_body: tuple[Stmt, ...] = ()
    elif hasattr(else_branch, "stmts"):
        else_body = _lower_block(else_branch, span)
    elif isinstance(else_branch, IfExpr):
        else_body = (_lower_if(else_branch, span),)
    else:
        else_body = (Stmt("return", expr=_lower_expr(else_branch, span), span=span),)
    return Stmt(
        "if",
        cond=_lower_expr(expr.cond, span),
        then_body=then_body,
        else_body=else_body,
        span=span,
    )


def _select_function(source: str | FnDecl, function: str, span: Span) -> FnDecl:
    if isinstance(source, FnDecl):
        return source
    program = parse(lex(source))
    functions = [item for item in program.items if isinstance(item, FnDecl)]
    if not functions:
        raise _fail("E006", "no function available for symbolic execution", span=span)
    for fn in functions:
        if fn.name == function:
            return fn
    names = ", ".join(sorted(fn.name for fn in functions))
    raise _fail(
        "E006",
        f"function {function!r} not found (available: {names})",
        span=span,
    )


def _function_body(fn: FnDecl, span: Span) -> tuple[Stmt, ...]:
    if fn.body is None:
        raise _fail(
            "E003",
            f"function {fn.name!r} has no body to execute symbolically",
            span=span,
        )
    return _lower_block(fn.body, span)


def _parameter_intervals(fn: FnDecl, span: Span) -> dict[str, Interval]:
    result: dict[str, Interval] = {}
    for param in fn.params:
        if not isinstance(param.pattern, PatIdent):
            raise _fail(
                "E003",
                "frame H requires named scalar parameters",
                span=span,
            )
        result[str(param.pattern.name)] = Interval(I64_MIN, I64_MAX)
    return result


# ---------------------------------------------------------------------------
# Bounded symbolic executor
# ---------------------------------------------------------------------------


@dataclass
class _State:
    env: dict[str, SymExpr] = field(default_factory=dict)
    constraints: tuple[SymExpr, ...] = ()
    result: SymExpr | None = None
    steps: int = 0
    unrolls: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    returned: bool = False
    path_id: str = "p0"


class _Context:
    def __init__(
        self,
        *,
        max_steps: int,
        max_unroll: int,
        max_paths: int,
        timeout_ms: int,
        span: Span,
    ) -> None:
        self.max_steps = max_steps
        self.max_unroll = max_unroll
        self.max_paths = max_paths
        self.deadline = time.monotonic() + timeout_ms / 1000
        self.span = span
        self.diagnostics: list[FormalDiagnostic] = []
        self.branch_decisions: list[dict[str, Any]] = []
        self.explored = 0
        self.pruned = 0
        self.path_counter = 0
        self.timeout_hit = False

    def new_path(self) -> str:
        self.path_counter += 1
        return f"p{self.path_counter}"

    def check_timeout(self) -> bool:
        if self.timeout_hit:
            return True
        if time.monotonic() > self.deadline:
            self.timeout_hit = True
            self.diagnostics.append(
                FormalDiagnostic(
                    "E004",
                    "solver time budget exceeded; bounded result is incomplete",
                    "timeout",
                    self.span,
                )
            )
            return True
        return False

    def add_bound(self, message: str, kind: str = "bound") -> None:
        self.diagnostics.append(FormalDiagnostic("E008", message, kind, self.span))


def _clone_state(state: _State, path_id: str) -> _State:
    return _State(
        env=dict(state.env),
        constraints=tuple(state.constraints),
        result=state.result,
        steps=state.steps,
        unrolls=dict(state.unrolls),
        truncated=state.truncated,
        returned=state.returned,
        path_id=path_id,
    )


def _feasibility(
    ctx: _Context, constraints: Sequence[SymExpr]
) -> tuple[Feasibility, dict]:
    if ctx.check_timeout():
        return "unknown", {}
    ranges: dict[str, Interval] = {}
    status = "feasible"
    for condition in constraints:
        step_status, ranges = _analyze(condition, ranges)
        if step_status == "infeasible":
            return "infeasible", ranges
        if step_status == "unknown":
            status = "unknown"
    return status, ranges


def _exec_block(stmts: Sequence[Stmt], states: list[_State], ctx: _Context) -> None:
    for stmt in stmts:
        if not states or all(state.truncated or state.returned for state in states):
            return
        if ctx.check_timeout():
            for state in states:
                if not state.returned:
                    state.truncated = True
            return
        if stmt.kind == "assign":
            for state in states:
                if state.truncated or state.returned:
                    continue
                state.steps += 1
                if state.steps > ctx.max_steps:
                    ctx.add_bound(f"symbolic step bound {ctx.max_steps} exhausted")
                    state.truncated = True
                    continue
                assert stmt.expr is not None
                state.env[stmt.target] = _substitute(stmt.expr, state.env)
            continue
        if stmt.kind == "return":
            for state in states:
                if state.truncated or state.returned:
                    continue
                state.steps += 1
                assert stmt.expr is not None or state.result is not None
                if stmt.expr is not None:
                    state.result = _substitute(stmt.expr, state.env)
                state.returned = True
            continue
        if stmt.kind == "if":
            _exec_if(stmt, states, ctx)
            continue
        if stmt.kind == "loop":
            _exec_loop(stmt, states, ctx)
            continue
        raise _fail(
            "E003", f"unknown formal statement kind {stmt.kind!r}", span=ctx.span
        )


def _exec_if(stmt: Stmt, states: list[_State], ctx: _Context) -> None:
    assert stmt.cond is not None
    next_states: list[_State] = []
    for state in states:
        if state.truncated or state.returned:
            next_states.append(state)
            continue
        state.steps += 1
        if state.steps > ctx.max_steps:
            ctx.add_bound(f"symbolic step bound {ctx.max_steps} exhausted")
            state.truncated = True
            next_states.append(state)
            continue
        condition = _substitute(stmt.cond, state.env)
        if condition.op == "bool":
            branch = "then" if condition.value else "else"
            ctx.branch_decisions.append(
                {
                    "condition": condition.render(),
                    "branch": branch,
                    "feasibility": "feasible",
                    "path_id": state.path_id,
                }
            )
            selected = stmt.then_body if condition.value else stmt.else_body
            child = _clone_state(state, state.path_id)
            children = [child]
            _exec_block(selected, children, ctx)
            next_states.extend(children)
            continue
        negated = SymExpr("not", args=(condition,))
        then_status, _ = _feasibility(ctx, (*state.constraints, condition))
        else_status, _ = _feasibility(ctx, (*state.constraints, negated))
        ctx.branch_decisions.append(
            {
                "condition": condition.render(),
                "branch": "then",
                "feasibility": then_status,
                "path_id": state.path_id,
            }
        )
        ctx.branch_decisions.append(
            {
                "condition": negated.render(),
                "branch": "else",
                "feasibility": else_status,
                "path_id": state.path_id,
            }
        )
        if then_status == "infeasible":
            ctx.pruned += 1
        if else_status == "infeasible":
            ctx.pruned += 1
        if then_status == "infeasible" and else_status == "infeasible":
            ctx.add_bound("both branches of a guard are infeasible", "proof")
            dead = _clone_state(state, ctx.new_path())
            dead.constraints = (*dead.constraints, condition, negated)
            next_states.append(dead)
            continue
        for feasible, condition_expr, body in (
            (then_status, condition, stmt.then_body),
            (else_status, negated, stmt.else_body),
        ):
            if feasible == "infeasible":
                # Keep the dead guard as an explicit infeasible path so witness
                # synthesis can explain it instead of silently dropping code.
                dead = _clone_state(state, ctx.new_path())
                dead.constraints = (*dead.constraints, condition_expr)
                dead.returned = True
                dead.result = SymExpr.const(0)
                next_states.append(dead)
                continue
            child = _clone_state(state, ctx.new_path())
            child.constraints = (*child.constraints, condition_expr)
            children = [child]
            _exec_block(body, children, ctx)
            next_states.extend(children)
    states[:] = next_states


def _exec_loop(stmt: Stmt, states: list[_State], ctx: _Context) -> None:
    assert stmt.cond is not None
    next_states: list[_State] = []
    for state in states:
        if state.truncated or state.returned:
            next_states.append(state)
            continue
        state.steps += 1
        if state.steps > ctx.max_steps:
            ctx.add_bound(f"symbolic step bound {ctx.max_steps} exhausted")
            state.truncated = True
            next_states.append(state)
            continue
        loop_id = str(stmt.span) if stmt.span is not None else "loop"
        condition = _substitute(stmt.cond, state.env)
        hit_unroll_bound = False
        active = [state]
        while active:
            if ctx.check_timeout():
                for item in active:
                    item.truncated = True
                    next_states.append(item)
                break
            current = active.pop(0)
            current.steps += 1
            if current.steps > ctx.max_steps:
                ctx.add_bound(f"symbolic step bound {ctx.max_steps} exhausted")
                current.truncated = True
                next_states.append(current)
                continue
            cond = _substitute(stmt.cond, current.env)
            if cond.op == "bool" and not cond.value:
                next_states.append(current)
                continue
            if cond.op == "bool" and cond.value:
                status: Feasibility = "feasible"
            else:
                status, _ = _feasibility(ctx, (*current.constraints, cond))
            if status == "infeasible":
                next_states.append(current)
                continue
            if current.unrolls.get(loop_id, 0) >= ctx.max_unroll:
                ctx.add_bound(
                    f"loop unroll bound {ctx.max_unroll} exhausted at "
                    f"{condition.render()}"
                )
                current.truncated = True
                next_states.append(current)
                hit_unroll_bound = True
                continue
            current.unrolls[loop_id] = current.unrolls.get(loop_id, 0) + 1
            current.constraints = (*current.constraints, cond)
            child = _clone_state(current, ctx.new_path())
            children = [child]
            _exec_block(stmt.then_body, children, ctx)
            for derived in children:
                if derived.returned or derived.truncated:
                    next_states.append(derived)
                else:
                    active.append(derived)
        if hit_unroll_bound:
            ctx.branch_decisions.append(
                {
                    "condition": condition.render(),
                    "branch": "unroll-bound",
                    "feasibility": "unknown",
                    "path_id": state.path_id,
                }
            )
    states[:] = next_states


def _execute(
    source: str | FnDecl,
    *,
    function: str,
    max_steps: int,
    max_unroll: int,
    max_paths: int,
    timeout_ms: int,
    span: Span,
) -> tuple[BoundedExecutionReport, tuple[str, ...]]:
    fn = _select_function(source, function, span)
    body = _function_body(fn, span)
    names = tuple(_parameter_intervals(fn, span))
    ctx = _Context(
        max_steps=max_steps,
        max_unroll=max_unroll,
        max_paths=max_paths,
        timeout_ms=timeout_ms,
        span=span,
    )
    initial = _State(
        env={name: SymExpr.var(name) for name in names},
        path_id="p0",
    )
    states = [initial]
    _exec_block(body, states, ctx)
    if len(states) > max_paths:
        ctx.add_bound(f"symbolic path bound {max_paths} exhausted")
        states = states[:max_paths]
    paths: list[SymbolicPath] = []
    for state in states:
        ctx.explored += 1
        status, _ranges = _feasibility(ctx, state.constraints)
        paths.append(
            SymbolicPath(
                path_id=state.path_id,
                constraints=tuple(item.render() for item in state.constraints),
                result=state.result.render() if state.result else "void",
                steps=state.steps,
                unrolls=dict(state.unrolls),
                feasibility=status,
                truncated=state.truncated,
                constraint_terms=state.constraints,
            )
        )
        if status == "infeasible":
            ctx.pruned += 1
    report = BoundedExecutionReport(
        function=fn.name,
        bounds={
            "max_steps": max_steps,
            "max_unroll": max_unroll,
            "max_paths": max_paths,
            "timeout_ms": timeout_ms,
        },
        paths=tuple(paths),
        diagnostics=tuple(ctx.diagnostics),
        branch_decisions=tuple(ctx.branch_decisions),
        explored=ctx.explored,
        pruned=ctx.pruned,
    )
    return report, names


def _maybe_raise(report: BoundedExecutionReport, strict: bool, span: Span) -> None:
    if not strict or report.complete:
        return
    diagnostic = next(
        item for item in report.diagnostics if item.kind in {"bound", "timeout"}
    )
    raise _fail(diagnostic.code, diagnostic.message, span=span or diagnostic.span)


def formal_sym_exec_bounded_paths(
    source: str | FnDecl,
    *,
    function: str = "main",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_unroll: int = DEFAULT_MAX_UNROLL,
    max_paths: int = DEFAULT_MAX_PATHS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    strict: bool = True,
    span: Span = None,
) -> BoundedExecutionReport:
    """Symbolically execute one function over bounded paths.

    ``strict=True`` (default) raises ``E008`` on an exhausted step/unroll/path
    bound and ``E004`` on a solver timeout.  Pass ``strict=False`` to inspect
    the partial report, whose ``complete``/``proved`` flags remain honest.
    """
    if min(max_steps, max_unroll, max_paths, timeout_ms) <= 0:
        raise _fail("E003", "frame H bounds must be positive", span=span)
    report, _ = _execute(
        source,
        function=function,
        max_steps=max_steps,
        max_unroll=max_unroll,
        max_paths=max_paths,
        timeout_ms=timeout_ms,
        span=span or _span(source if isinstance(source, FnDecl) else None, None),
    )
    _maybe_raise(report, strict, span)
    return report


def formal_require_bounded_proof(
    report: BoundedExecutionReport, *, span: Span = None
) -> None:
    """Raise ``E002`` unless the report is a complete, proved bounded result."""
    if report.proved:
        return
    if not report.complete:
        _maybe_raise(report, True, span)
    raise _fail(
        "E002",
        "bounded symbolic proof is unknown or incomplete",
        span=span,
        reason=f"complete={report.complete}, proved={report.proved}",
        help_text="refine guards or increase the declared bounds and re-check",
    )


# ---------------------------------------------------------------------------
# Concrete evaluation and bounded relational refinement
# ---------------------------------------------------------------------------


def _concrete_run(stmts: Sequence[Stmt], args: Mapping[str, int]) -> Any:
    env = dict(args)

    def run_block(body: Sequence[Stmt]) -> Any:
        for stmt in body:
            if stmt.kind == "assign":
                assert stmt.expr is not None
                env[stmt.target] = int(stmt.expr.evaluate(env))
            elif stmt.kind == "return":
                return stmt.expr.evaluate(env) if stmt.expr is not None else None
            elif stmt.kind == "if":
                assert stmt.cond is not None
                if bool(stmt.cond.evaluate(env)):
                    value = run_block(stmt.then_body)
                else:
                    value = run_block(stmt.else_body)
                if value is not None:
                    return value
            elif stmt.kind == "loop":
                assert stmt.cond is not None
                guard = 0
                while bool(stmt.cond.evaluate(env)):
                    guard += 1
                    if guard > 512:
                        return "<concrete-loop-bound>"
                    value = run_block(stmt.then_body)
                    if value is not None:
                        return value
        return None

    return run_block(stmts)


def _normalize_outcome(value: Any) -> dict[str, str] | None:
    return normalize_error(value)


def _sample_inputs(
    names: Sequence[str],
    intervals: Mapping[str, tuple[int, int]],
    samples: int,
    seed: int,
) -> tuple[dict[str, int], ...]:
    rng = random.Random(f"polyglot-formal-rel:{seed}")
    results: list[dict[str, int]] = []
    for index in range(max(samples, 0)):
        assignment: dict[str, int] = {}
        for name in names:
            lo, hi = intervals.get(name, (-8, 8))
            assignment[name] = rng.randint(lo, hi)
        results.append(assignment)
    return tuple(results)


def _program_label(value: str | FnDecl) -> str:
    return value if isinstance(value, str) else value.name


def formal_check_relational_refinement(
    left: str | FnDecl,
    right: str | FnDecl,
    *,
    inputs: Mapping[str, tuple[int, int]] | None = None,
    runners: Mapping[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_unroll: int = DEFAULT_MAX_UNROLL,
    max_paths: int = DEFAULT_MAX_PATHS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    strict: bool = False,
    span: Span = None,
) -> RelationalReport:
    """Compare two programs/backends on bounded inputs and symbolic paths.

    With ``runners`` supplied, each callable is invoked on every sampled input
    and outcomes are normalized with :func:`tools.diff_fuzz.normalize_error`.
    Without ``runners``, both source programs are lowered to the formal IR and
    compared both symbolically and by concrete bounded evaluation.  A witness
    is emitted only after a concrete divergence is observed.
    """
    left_report, left_names = _execute(
        left,
        function="main",
        max_steps=max_steps,
        max_unroll=max_unroll,
        max_paths=max_paths,
        timeout_ms=timeout_ms,
        span=span,
    )
    right_report, right_names = _execute(
        right,
        function="main",
        max_steps=max_steps,
        max_unroll=max_unroll,
        max_paths=max_paths,
        timeout_ms=timeout_ms,
        span=span,
    )
    names = tuple(left_names or right_names)
    declared = {
        name: tuple(inputs[name]) if name in inputs else (-8, 8) for name in names
    }
    for name, bounds in declared.items():
        if bounds[0] > bounds[1]:
            raise _fail("E003", f"invalid input interval for {name!r}", span=span)

    divergences: list[Divergence] = []
    seen_divergences: set[tuple[tuple[tuple[str, int], ...], str, str]] = set()
    checked = 0
    left_fn = _select_function(left, "main", span)
    right_fn = _select_function(right, "main", span)
    left_body = _function_body(left_fn, span)
    right_body = _function_body(right_fn, span)
    assignments = _sample_inputs(names, declared, samples, seed)
    backend_names = tuple(runners) if runners else ("left", "right")
    for assignment in assignments:
        checked += 1
        if runners:
            observed = {
                name: _normalize_outcome(runner(**assignment))
                for name, runner in runners.items()
            }
            distinct = {repr(value) for value in observed.values()}
            if len(distinct) > 1:
                first, second = backend_names[:2]
                key = (
                    tuple(sorted(assignment.items())),
                    repr(observed[first]),
                    repr(observed[second]),
                )
                if key not in seen_divergences:
                    seen_divergences.add(key)
                    divergences.append(
                        Divergence(
                            assignment, observed[first], observed[second], first, second
                        )
                    )
        else:
            first_value = _concrete_run(left_body, assignment)
            second_value = _concrete_run(right_body, assignment)
            if first_value != second_value:
                key = (
                    tuple(sorted(assignment.items())),
                    repr(first_value),
                    repr(second_value),
                )
                if key not in seen_divergences:
                    seen_divergences.add(key)
                    divergences.append(
                        Divergence(assignment, first_value, second_value)
                    )

    left_signature = {
        (path.constraints, path.result)
        for path in left_report.paths
        if path.feasibility != "infeasible" and not path.truncated
    }
    right_signature = {
        (path.constraints, path.result)
        for path in right_report.paths
        if path.feasibility != "infeasible" and not path.truncated
    }
    symbolic_agreement = (
        left_signature == right_signature
        and bool(left_signature)
        and left_report.proved
        and right_report.proved
    )
    diagnostics = tuple(
        dict.fromkeys(left_report.diagnostics + right_report.diagnostics, None)
    )
    report = RelationalReport(
        left=_program_label(left),
        right=_program_label(right),
        depth=max_unroll,
        inputs=dict(declared),
        checked=checked,
        proved_bounded=symbolic_agreement and not divergences and checked > 0,
        divergences=tuple(divergences),
        left_report=left_report,
        right_report=right_report,
        diagnostics=tuple(diagnostics),
    )
    if strict and divergences:
        first = divergences[0]
        raise _fail(
            "E002",
            "bounded relational refinement found a concrete divergence",
            span=span,
            reason=(
                f"inputs={first.inputs}, {first.left_backend}={first.left!r}, "
                f"{first.right_backend}={first.right!r}"
            ),
            help_text="inspect the witness and align the two lowerings",
        )
    return report


# ---------------------------------------------------------------------------
# Infeasible path / dead guard witnesses
# ---------------------------------------------------------------------------


def _minimal_conflict(constraints: Sequence[SymExpr]) -> tuple[SymExpr, ...]:
    """Shrink an infeasible constraint set greedily (sound: re-check each drop)."""
    current = list(constraints)
    for candidate in list(current):
        trial = [item for item in current if item is not candidate]
        status, _ = _analyze(
            SymExpr("and", args=tuple(trial)) if trial else SymExpr.boolean(True), {}
        )
        if status == "infeasible":
            current = trial
    return tuple(current)


def _explain_intervals(constraints: Sequence[SymExpr]) -> dict[str, str]:
    """Show which intersection empties each variable's interval."""
    ranges: dict[str, Interval] = {}
    for expr in constraints:
        status, refined = _analyze(expr, ranges)
        if status == "infeasible":
            details: dict[str, str] = {}
            for name in sorted(expr.variables()):
                previous = ranges.get(name)
                details[name] = (
                    f"empty after {expr.render()}"
                    f" (previous {previous.render() if previous else 'unbounded'})"
                )
            return details
        ranges = refined
    return {name: interval.render() for name, interval in ranges.items()}


def formal_synthesize_infeasible_path_witness(
    source: str | FnDecl,
    *,
    function: str = "main",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_unroll: int = DEFAULT_MAX_UNROLL,
    max_paths: int = DEFAULT_MAX_PATHS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    strict: bool = False,
    span: Span = None,
) -> InfeasiblePathReport:
    """Explain unreachable branches and impossible guard combinations.

    Dead branches are findings, not failures.  ``strict`` applies only to
    bound/timeout diagnostics from the underlying symbolic execution.
    """
    execution, _ = _execute(
        source,
        function=function,
        max_steps=max_steps,
        max_unroll=max_unroll,
        max_paths=max_paths,
        timeout_ms=timeout_ms,
        span=span,
    )
    _maybe_raise(execution, strict, span)
    fn = _select_function(source, function, span)
    witnesses: list[WitnessExplanation] = []
    for path in execution.paths:
        if path.feasibility != "infeasible":
            continue
        constraints = path.constraint_terms
        minimal = _minimal_conflict(constraints)
        witnesses.append(
            WitnessExplanation(
                path_id=path.path_id,
                constraints=tuple(item.render() for item in minimal)
                or path.constraints,
                intervals=_explain_intervals(minimal or constraints),
                reason="interval intersection is empty for the recorded path guards",
            )
        )
    dead_branches = [
        dict(decision)
        for decision in execution.branch_decisions
        if decision.get("feasibility") == "infeasible"
    ]
    return InfeasiblePathReport(
        function=fn.name,
        witnesses=tuple(witnesses),
        dead_branches=tuple(dead_branches),
        execution=execution,
    )


__all__ = [
    "BoundedExecutionReport",
    "Divergence",
    "Feasibility",
    "FormalDiagnostic",
    "InfeasiblePathReport",
    "Interval",
    "RelationalReport",
    "SymExpr",
    "SymbolicPath",
    "WitnessExplanation",
    "formal_check_relational_refinement",
    "formal_require_bounded_proof",
    "formal_sym_exec_bounded_paths",
    "formal_synthesize_infeasible_path_witness",
]
