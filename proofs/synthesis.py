"""Bi-directional formal-synthesis foundations.

The sound core here is deliberately small and honest:

* a controlled invariant language (a Python-expression subset) is parsed into
  a symbolic tree;
* integer expressions are checked with an i64 interval domain and a
  polynomial identity normalizer;
* candidates are enumerated from a finite grammar and accepted only when the
  postcondition is *proved* under the declared parameter ranges, never merely
  sampled;
* every accepted candidate carries a certificate that an independent checker
  re-validates; and
* the emitted ``.poly`` function embeds a runtime proof guard so a backend or
  hand edit that breaks the proof fails loudly.

Unrestricted natural language, arbitrary loops, heap aliasing, and "100%
verified for every possible specification" are out of scope.  Use the
controlled grammar in :func:`interpret_spec` or construct ``SynthesisSpec``
directly.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ast_nodes import (
    ArrayExpr,
    Binary,
    Block,
    Expr,
    ExprStmt,
    FnDecl,
    FnParam,
    Ident,
    Index,
    LetStmt,
    LitInt,
    MacroCall,
    Node,
    PatIdent,
    Program,
    RepeatArray,
    TypeName,
    Unary,
)
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


# ---------------------------------------------------------------------------
# Symbolic expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("empty interval")

    def render(self) -> str:
        return f"[{self.lo}, {self.hi}]"


@dataclass(frozen=True)
class ExprNode:
    op: str
    value: int | str | None = None
    args: tuple[ExprNode, ...] = ()

    @staticmethod
    def var(name: str) -> ExprNode:
        return ExprNode("var", value=name)

    @staticmethod
    def const(value: int) -> ExprNode:
        return ExprNode("const", value=value)

    def children(self) -> tuple[ExprNode, ...]:
        return self.args

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.args)

    def render(self) -> str:
        if self.op == "var":
            return str(self.value)
        if self.op == "const":
            return str(self.value)
        if self.op == "neg":
            return f"(-{self.args[0].render()})"
        if self.op == "floordiv":
            return f"({self.args[0].render()} / {self.args[1].render()})"
        if self.op == "mod":
            return f"({self.args[0].render()} % {self.args[1].render()})"
        symbol = {"add": "+", "sub": "-", "mul": "*"}.get(self.op)
        if symbol is None:
            return f"{self.op}({', '.join(arg.render() for arg in self.args)})"
        return f"({self.args[0].render()} {symbol} {self.args[1].render()})"

    def to_polyglot(self, line: int = 1, col: int = 1) -> Expr:
        if self.op == "var":
            return Ident(line, col, str(self.value))
        if self.op == "const":
            assert isinstance(self.value, int)
            return LitInt(line, col, self.value, str(self.value))
        if self.op == "neg":
            return Unary(line, col, "-", self.args[0].to_polyglot(line, col))
        if self.op in {"add", "sub", "mul", "floordiv", "mod"}:
            symbol = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "floordiv": "/",
                "mod": "%",
            }[self.op]
            return Binary(
                line,
                col,
                symbol,
                self.args[0].to_polyglot(line, col),
                self.args[1].to_polyglot(line, col),
            )
        raise PillarError(
            "E003",
            f"cannot emit symbolic operator {self.op!r} as Polyglot code",
        )

    def poly(self) -> dict[tuple[str, ...], int] | None:
        """Canonical integer polynomial, or ``None`` outside the ring fragment."""
        if self.op == "const":
            assert isinstance(self.value, int)
            return {} if self.value == 0 else {(): self.value}
        if self.op == "var":
            assert isinstance(self.value, str)
            return {(self.value,): 1}
        if self.op == "neg":
            inner = self.args[0].poly()
            return (
                None if inner is None else {key: -value for key, value in inner.items()}
            )
        if self.op not in {"add", "sub", "mul"}:
            return None
        left = self.args[0].poly()
        right = self.args[1].poly()
        if left is None or right is None:
            return None
        if self.op in {"add", "sub"}:
            result = dict(left)
            sign = 1 if self.op == "add" else -1
            for key, value in right.items():
                result[key] = result.get(key, 0) + sign * value
            return {key: value for key, value in result.items() if value}
        result: dict[tuple[str, ...], int] = {}
        for left_key, left_value in left.items():
            for right_key, right_value in right.items():
                key = tuple(sorted(left_key + right_key))
                result[key] = result.get(key, 0) + left_value * right_value
        return {key: value for key, value in result.items() if value}

    def interval(self, env: dict[str, Interval]) -> Interval | None:
        if self.op == "const":
            assert isinstance(self.value, int)
            if not I64_MIN <= self.value <= I64_MAX:
                return None
            return Interval(self.value, self.value)
        if self.op == "var":
            return env.get(str(self.value))
        if self.op == "neg":
            inner = self.args[0].interval(env)
            if inner is None:
                return None
            return _checked(-inner.hi, -inner.lo)
        if self.op in {"add", "sub", "mul", "floordiv", "mod"}:
            left = self.args[0].interval(env)
            right = self.args[1].interval(env)
            if left is None or right is None:
                return None
            if self.op == "add":
                return _checked(left.lo + right.lo, left.hi + right.hi)
            if self.op == "sub":
                return _checked(left.lo - right.hi, left.hi - right.lo)
            if self.op == "mul":
                products = [
                    left.lo * right.lo,
                    left.lo * right.hi,
                    left.hi * right.lo,
                    left.hi * right.hi,
                ]
                return _checked(min(products), max(products))
            if right.lo == right.hi and right.lo != 0:
                divisor = right.lo
                if divisor > 0:
                    return _checked(left.lo // divisor, left.hi // divisor)
                return _checked(left.hi // divisor, left.lo // divisor)
            return None
        return None


def _checked(lo: int, hi: int) -> Interval | None:
    if lo < I64_MIN or hi > I64_MAX:
        return None
    return Interval(lo, hi)


def _from_python(node: ast.AST) -> ExprNode:
    if isinstance(node, ast.Name):
        return ExprNode.var(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return ExprNode.const(1 if node.value else 0)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return ExprNode.const(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ExprNode("neg", args=(_from_python(node.operand),))
    if isinstance(node, ast.BinOp):
        mapping = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
        }
        for py_type, op in mapping.items():
            if isinstance(node.op, py_type):
                return ExprNode(
                    op, args=(_from_python(node.left), _from_python(node.right))
                )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"min", "max"}
        and len(node.args) == 2
        and not node.keywords
    ):
        return ExprNode(
            node.func.id,
            args=tuple(_from_python(arg) for arg in node.args),
        )
    raise PillarError(
        "E003",
        "unsupported expression in a formal condition",
        reason=f"node {type(node).__name__} is outside the controlled grammar",
        help_text="use integer variables, constants, + - * / %, comparisons, and and/or/not",
    )


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cond:
    op: str
    left: Cond | ExprNode | None = None
    right: Cond | ExprNode | None = None

    def render(self) -> str:
        if self.op in {"true", "false"}:
            return self.op
        if self.op == "not":
            assert isinstance(self.left, Cond)
            return f"!({self.left.render()})"
        if self.op in {"and", "or"}:
            assert isinstance(self.left, Cond) and isinstance(self.right, Cond)
            symbol = "&&" if self.op == "and" else "||"
            return f"({self.left.render()} {symbol} {self.right.render()})"
        assert isinstance(self.left, ExprNode) and isinstance(self.right, ExprNode)
        return f"({self.left.render()} {self.op} {self.right.render()})"

    def to_polyglot(self, line: int = 1, col: int = 1) -> Expr:
        if self.op in {"true", "false"}:
            from ast_nodes import LitBool

            return LitBool(line, col, self.op == "true")
        if self.op == "not":
            assert isinstance(self.left, Cond)
            return Unary(line, col, "!", self.left.to_polyglot(line, col))
        if self.op in {"and", "or"}:
            assert isinstance(self.left, Cond) and isinstance(self.right, Cond)
            return Binary(
                line,
                col,
                "&&" if self.op == "and" else "||",
                self.left.to_polyglot(line, col),
                self.right.to_polyglot(line, col),
            )
        assert isinstance(self.left, ExprNode) and isinstance(self.right, ExprNode)
        return Binary(
            line,
            col,
            self.op,
            self.left.to_polyglot(line, col),
            self.right.to_polyglot(line, col),
        )

    def substitute(self, name: str, replacement: ExprNode) -> Cond:
        if self.op in {"true", "false"}:
            return self
        if self.op == "not":
            assert isinstance(self.left, Cond)
            return Cond("not", left=self.left.substitute(name, replacement))
        if self.op in {"and", "or"}:
            assert isinstance(self.left, Cond) and isinstance(self.right, Cond)
            return Cond(
                self.op,
                left=self.left.substitute(name, replacement),
                right=self.right.substitute(name, replacement),
            )
        assert isinstance(self.left, ExprNode) and isinstance(self.right, ExprNode)
        return Cond(
            self.op,
            left=_substitute_expr(self.left, name, replacement),
            right=_substitute_expr(self.right, name, replacement),
        )

    def check(self, env: dict[str, Interval]) -> str:
        if self.op == "true":
            return "proved"
        if self.op == "false":
            return "failed"
        if self.op == "not":
            assert isinstance(self.left, Cond)
            result = self.left.check(env)
            return {"proved": "failed", "failed": "proved"}.get(result, "unknown")
        if self.op == "and":
            assert isinstance(self.left, Cond) and isinstance(self.right, Cond)
            results = {self.left.check(env), self.right.check(env)}
            if "failed" in results:
                return "failed"
            return "proved" if results == {"proved"} else "unknown"
        if self.op == "or":
            assert isinstance(self.left, Cond) and isinstance(self.right, Cond)
            results = {self.left.check(env), self.right.check(env)}
            if "proved" in results:
                return "proved"
            return "failed" if results == {"failed"} else "unknown"
        assert isinstance(self.left, ExprNode) and isinstance(self.right, ExprNode)
        if self.op in {"==", "!="}:
            left_poly = self.left.poly()
            right_poly = self.right.poly()
            if left_poly is not None and right_poly is not None:
                identical = left_poly == right_poly
                if self.op == "==" and identical:
                    return "proved"
                if self.op == "!=" and identical:
                    return "failed"
        left = self.left.interval(env)
        right = self.right.interval(env)
        if left is None or right is None:
            return "unknown"
        if self.op == "==":
            if left.hi < right.lo or right.hi < left.lo:
                return "failed"
            return "unknown"
        if self.op == "!=":
            if left.hi < right.lo or right.hi < left.lo:
                return "proved"
            return "unknown"
        if self.op == "<":
            if left.hi < right.lo:
                return "proved"
            if left.lo >= right.hi:
                return "failed"
            return "unknown"
        if self.op == "<=":
            if left.hi <= right.lo:
                return "proved"
            if left.lo > right.hi:
                return "failed"
            return "unknown"
        if self.op == ">":
            return Cond("<", left=right, right=left).check(env)
        if self.op == ">=":
            return Cond("<=", left=right, right=left).check(env)
        raise PillarError("E003", f"unsupported condition operator {self.op!r}")


def _substitute_expr(node: ExprNode, name: str, replacement: ExprNode) -> ExprNode:
    if node.op == "var" and node.value == name:
        return replacement
    if not node.args:
        return node
    return ExprNode(
        node.op,
        value=node.value,
        args=tuple(_substitute_expr(arg, name, replacement) for arg in node.args),
    )


def parse_condition(text: str) -> Cond:
    normalised = text.strip()
    normalised = normalised.replace("&&", " and ").replace("||", " or ")
    try:
        tree = ast.parse(normalised, mode="eval").body
    except SyntaxError as error:
        raise PillarError(
            "E003",
            "formal condition is not parseable",
            reason=str(error),
            help_text="use the controlled grammar, e.g. `result == a + b`",
            line=error.lineno or 0,
            col=error.offset or 0,
        ) from error
    return _cond_from_python(tree)


def _cond_from_python(node: ast.AST) -> Cond:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return Cond("true" if node.value else "false")
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            return Cond(
                "and",
                left=_cond_from_python(
                    ast.Compare(
                        left=node.left,
                        ops=node.ops[:1],
                        comparators=node.comparators[:1],
                    )
                ),
                right=_cond_from_python(
                    ast.Compare(
                        left=node.comparators[0],
                        ops=node.ops[1:],
                        comparators=node.comparators[1:],
                    )
                ),
            )
        op = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }.get(type(node.ops[0]))
        if op is None:
            raise PillarError("E003", "unsupported comparison operator")
        return Cond(
            op,
            left=_from_python(node.left),
            right=_from_python(node.comparators[0]),
        )
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        result = _cond_from_python(node.values[0])
        for value in node.values[1:]:
            result = Cond(op, left=result, right=_cond_from_python(value))
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return Cond("not", left=_cond_from_python(node.operand))
    raise PillarError(
        "E003",
        "unsupported top-level formal condition",
        reason=f"node {type(node).__name__} is outside the controlled grammar",
        help_text="use comparisons, `and`, `or`, `not`, or a boolean literal",
    )


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    name: str
    ty: str = "i64"
    lo: int | None = None
    hi: int | None = None


@dataclass(frozen=True)
class SynthesisSpec:
    name: str
    params: tuple[ParamSpec, ...]
    postcondition: str
    preconditions: tuple[str, ...] = ()
    max_depth: int = 2

    def condition(self) -> Cond:
        return parse_condition(self.postcondition)


def _refine(cond: Cond, env: dict[str, Interval]) -> None:
    if cond.op == "and":
        assert isinstance(cond.left, Cond) and isinstance(cond.right, Cond)
        _refine(cond.left, env)
        _refine(cond.right, env)
        return
    if cond.op not in {"==", "!=", "<", "<=", ">", ">="}:
        return
    left, right = cond.left, cond.right
    assert isinstance(left, ExprNode) and isinstance(right, ExprNode)
    if left.op == "var" and right.op == "const" and isinstance(right.value, int):
        name, value, op = str(left.value), right.value, cond.op
    elif right.op == "var" and left.op == "const" and isinstance(left.value, int):
        name, value = str(right.value), left.value
        op = {
            "==": "==",
            "!=": "!=",
            "<": ">",
            "<=": ">=",
            ">": "<",
            ">=": "<=",
        }[cond.op]
    else:
        return
    current = env.get(name)
    if current is None:
        return
    lo, hi = current.lo, current.hi
    if op == ">=":
        lo = max(lo, value)
    elif op == ">":
        lo = max(lo, value + 1)
    elif op == "<=":
        hi = min(hi, value)
    elif op == "<":
        hi = min(hi, value - 1)
    elif op == "==":
        lo = hi = value
    else:
        return
    if lo > hi:
        raise PillarError(
            "E002",
            f"preconditions are unsatisfiable for `{name}`",
            reason=f"refined interval would be [{lo}, {hi}]",
        )
    env[name] = Interval(lo, hi)


def environment(spec: SynthesisSpec) -> dict[str, Interval]:
    env: dict[str, Interval] = {}
    for param in spec.params:
        lo = I64_MIN if param.lo is None else param.lo
        hi = I64_MAX if param.hi is None else param.hi
        if lo > hi:
            raise PillarError(
                "E002",
                f"parameter `{param.name}` has an empty range",
            )
        env[param.name] = Interval(lo, hi)
    for text in spec.preconditions:
        _refine(parse_condition(text), env)
    return env


_NL_FUNCTION = re.compile(
    r"^\s*function\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)\s*$"
)
_NL_ASSUME = re.compile(
    r"^\s*assume\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+is\s+between\s+"
    r"(?P<lo>-?\d+)\s+and\s+(?P<hi>-?\d+)\s*$"
)
_NL_ENSURE = re.compile(r"^\s*(?:ensure|postcondition)\s*:?\s*(?P<expr>.+)$")
_NL_FOR_ALL = re.compile(
    r"^\s*for\s+all\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+"
    r"(?P<lo>-?\d+)\s*\.\.\s*(?P<hi>-?\d+)\s*,\s*(?P<expr>.+)$"
)


def interpret_spec(text: str) -> SynthesisSpec:
    """Interpret a controlled natural-language specification.

    Supported lines::

        function add(a: i64, b: i64)
        assume a is between 0 and 100
        assume b is between 0 and 100
        ensure result == a + b
        for all a in 0..100, result == a + 1
    """
    name: str | None = None
    params: list[ParamSpec] = []
    postcondition: str | None = None
    preconditions: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _NL_FUNCTION.match(line)
        if match:
            name = match.group("name")
            for chunk in match.group("params").split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if ":" not in chunk:
                    raise PillarError(
                        "E003",
                        f"parameter {chunk!r} is missing a type",
                        help_text="write `a: i64`",
                    )
                param_name, ty = (part.strip() for part in chunk.split(":", 1))
                params.append(ParamSpec(param_name, ty or "i64"))
            continue
        match = _NL_ASSUME.match(line)
        if match:
            pname = match.group("name")
            lo, hi = int(match.group("lo")), int(match.group("hi"))
            for index, param in enumerate(params):
                if param.name == pname:
                    params[index] = dataclasses.replace(param, lo=lo, hi=hi)
                    break
            else:
                params.append(ParamSpec(pname, "i64", lo, hi))
            continue
        match = _NL_ENSURE.match(line)
        if match:
            postcondition = match.group("expr").strip()
            continue
        match = _NL_FOR_ALL.match(line)
        if match:
            pname = match.group("name")
            lo, hi = int(match.group("lo")), int(match.group("hi"))
            if hi < lo:
                raise PillarError("E002", "`for all` range is empty")
            for index, param in enumerate(params):
                if param.name == pname:
                    params[index] = dataclasses.replace(param, lo=lo, hi=hi)
                    break
            else:
                params.append(ParamSpec(pname, "i64", lo, hi))
            postcondition = match.group("expr").strip()
            continue
        preconditions.append(line)
    if name is None:
        raise PillarError(
            "E003",
            "specification has no `function <name>(...)` line",
        )
    if postcondition is None:
        raise PillarError(
            "E003",
            "specification has no `ensure <condition>` line",
        )
    return SynthesisSpec(
        name=name,
        params=tuple(params),
        postcondition=postcondition,
        preconditions=tuple(preconditions),
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofStep:
    rule: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class ProofCertificate:
    spec: SynthesisSpec
    candidate: str
    interval: tuple[int, int]
    steps: tuple[ProofStep, ...]
    digest: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "spec": dataclasses.asdict(self.spec),
            "candidate": self.candidate,
            "interval": list(self.interval),
            "steps": [dataclasses.asdict(step) for step in self.steps],
        }

    def to_json(self) -> str:
        payload = self.body()
        payload["digest"] = self.digest
        return json.dumps(payload, indent=2, sort_keys=True)


def _expr_from_text(text: str) -> ExprNode:
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as error:
        raise PillarError(
            "E003",
            "candidate expression is not parseable",
            reason=str(error),
        ) from error
    return _from_python(tree)


def _candidate_space(
    params: tuple[ParamSpec, ...],
    max_depth: int,
    *,
    limit: int = 4000,
) -> list[ExprNode]:
    atoms = [ExprNode.var(param.name) for param in params] + [
        ExprNode.const(value) for value in (0, 1, 2)
    ]
    atoms = list(dict.fromkeys(atoms))
    seen: dict[str, ExprNode] = {atom.render(): atom for atom in atoms}
    frontier = list(atoms)
    all_exprs = list(atoms)
    for _depth in range(1, max_depth + 1):
        produced: list[ExprNode] = []
        for left in frontier:
            for right in all_exprs:
                for op in ("add", "sub", "mul"):
                    expr = ExprNode(op, args=(left, right))
                    key = expr.render()
                    if key in seen or expr.node_count() > 2 * max_depth + 3:
                        continue
                    seen[key] = expr
                    produced.append(expr)
                    if len(seen) >= limit:
                        break
                if len(seen) >= limit:
                    break
            if len(seen) >= limit:
                break
        all_exprs.extend(produced)
        frontier = produced
        if len(seen) >= limit:
            break
    return sorted(
        seen.values(),
        key=lambda expr: (expr.node_count(), len(expr.render()), expr.render()),
    )


def synthesize(spec: SynthesisSpec) -> tuple[ExprNode, ProofCertificate]:
    env = environment(spec)
    postcondition = spec.condition()
    for candidate in _candidate_space(spec.params, spec.max_depth):
        interval = candidate.interval(env)
        if interval is None:
            continue
        obligation = postcondition.substitute("result", candidate)
        if obligation.check(env) != "proved":
            continue
        steps = [
            ProofStep(
                "parameter_range",
                {
                    "name": param.name,
                    "lo": env[param.name].lo,
                    "hi": env[param.name].hi,
                },
            )
            for param in spec.params
        ]
        steps.append(
            ProofStep(
                "candidate_interval",
                {
                    "expression": candidate.render(),
                    "lo": interval.lo,
                    "hi": interval.hi,
                },
            )
        )
        steps.append(
            ProofStep(
                "obligation",
                {
                    "postcondition": postcondition.render(),
                    "substituted": obligation.render(),
                    "status": "proved",
                },
            )
        )
        body = {
            "spec": dataclasses.asdict(spec),
            "candidate": candidate.render(),
            "interval": [interval.lo, interval.hi],
            "steps": [dataclasses.asdict(step) for step in steps],
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return candidate, ProofCertificate(
            spec=spec,
            candidate=candidate.render(),
            interval=(interval.lo, interval.hi),
            steps=tuple(steps),
            digest=digest,
        )
    raise PillarError(
        "E002",
        f"no candidate satisfies `{spec.postcondition}` within the grammar",
        reason="the bounded enumerator exhausted its search space",
        help_text="widen parameter ranges, simplify the postcondition, or raise max_depth",
    )


def check_certificate(certificate: ProofCertificate, *, strict: bool = True) -> bool:
    """Independently re-check a certificate.  Never trusts the stored status."""
    body = {
        "spec": dataclasses.asdict(certificate.spec),
        "candidate": certificate.candidate,
        "interval": list(certificate.interval),
        "steps": [dataclasses.asdict(step) for step in certificate.steps],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if digest != certificate.digest:
        if strict:
            raise PillarError(
                "E008",
                "proof certificate failed its digest check",
                help_text="the certificate was modified after synthesis",
            )
        return False
    env = environment(certificate.spec)
    candidate = _expr_from_text(certificate.candidate)
    interval = candidate.interval(env)
    if interval is None or (interval.lo, interval.hi) != certificate.interval:
        if strict:
            raise PillarError(
                "E002",
                "certificate interval does not match the candidate",
            )
        return False
    obligation = certificate.spec.condition().substitute("result", candidate)
    proved = obligation.check(env) == "proved"
    if not proved and strict:
        raise PillarError(
            "E002",
            "certificate obligation is not proved",
            reason=f"postcondition `{certificate.spec.postcondition}` is not implied",
        )
    return proved


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit_source(spec: SynthesisSpec, candidate: ExprNode) -> str:
    params = ", ".join(f"{param.name}: {param.ty}" for param in spec.params)
    postcondition = spec.condition().render()
    return (
        f"fn {spec.name}({params}) -> i64 {{\n"
        f"    let result = {candidate.render()};\n"
        f"    assert!({postcondition});\n"
        f"    result\n"
        f"}}\n"
    )


def _require_matching_certificate(
    spec: SynthesisSpec,
    candidate: ExprNode,
    certificate: ProofCertificate,
) -> None:
    if certificate.spec != spec:
        raise PillarError(
            "E003",
            "proof certificate belongs to a different specification",
            help_text="re-run synthesize() for this spec",
        )
    if certificate.candidate != candidate.render():
        raise PillarError(
            "E003",
            "proof certificate does not match the candidate being emitted",
            help_text="emit the candidate returned by synthesize()",
        )
    if not check_certificate(certificate):
        raise PillarError("E002", "refusing to emit an unproved function")


def emit_program(
    spec: SynthesisSpec,
    candidate: ExprNode,
    certificate: ProofCertificate,
) -> Program:
    _require_matching_certificate(spec, candidate, certificate)
    source = emit_source(spec, candidate)
    return parse(lex(source))


def emit_function_ast(
    spec: SynthesisSpec,
    candidate: ExprNode,
    certificate: ProofCertificate,
) -> FnDecl:
    """Construct the synthesized ``FnDecl`` directly (no source round-trip)."""
    _require_matching_certificate(spec, candidate, certificate)
    line = col = 1
    params = [
        FnParam(
            line,
            col,
            PatIdent(line, col, param.name),
            TypeName(line, col, param.ty),
        )
        for param in spec.params
    ]
    condition = spec.condition().to_polyglot(line, col)
    body = Block(
        line,
        col,
        [
            LetStmt(
                line,
                col,
                PatIdent(line, col, "result"),
                None,
                candidate.to_polyglot(line, col),
            ),
            ExprStmt(
                line,
                col,
                MacroCall(line, col, "assert", [condition]),
            ),
        ],
        Ident(line, col, "result"),
    )
    return FnDecl(
        line,
        col,
        spec.name,
        params,
        TypeName(line, col, "i64"),
        body,
    )


def compile_both(
    program: Program,
    *,
    native: bool = False,
    trusted: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compile the synthesized program with the Python and (optionally) native backend."""
    from backends import get_backend

    namespaces = {
        "python": get_backend("python", trusted=trusted).compile(program),
    }
    if native:
        namespaces["native"] = get_backend("native", trusted=trusted).compile(program)
    return namespaces


# ---------------------------------------------------------------------------
# Architectural invariants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofObligation:
    identifier: str
    description: str
    status: str
    span: tuple[int, int] | None = None
    detail: str = ""


def _walk(value: Any):
    if isinstance(value, Node):
        yield value
        for info in dataclasses.fields(value):
            if info.name in {"line", "col", "span"}:
                continue
            yield from _walk(getattr(value, info.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)


def verify_bounds(program: Program) -> list[ProofObligation]:
    """Prove constant array-index bounds in straight-line code.

    This is intentionally a small abstract interpreter: it tracks array
    lengths from ``let`` bindings and checks literal indices.  Dynamic indices
    and loop-derived indices are reported as ``unknown`` rather than silently
    accepted.
    """
    obligations: list[ProofObligation] = []
    lengths: dict[str, int] = {}
    for node in _walk(program):
        if isinstance(node, LetStmt) and isinstance(node.pattern, PatIdent):
            if isinstance(node.value, ArrayExpr):
                lengths[node.pattern.name] = len(node.value.elems)
            elif isinstance(node.value, RepeatArray) and isinstance(
                node.value.count, LitInt
            ):
                lengths[node.pattern.name] = node.value.count.value
        if isinstance(node, Index) and isinstance(node.receiver, Ident):
            name = node.receiver.name
            if name not in lengths:
                continue
            if isinstance(node.index, LitInt):
                index = node.index.value
                status = "proved" if 0 <= index < lengths[name] else "failed"
                obligations.append(
                    ProofObligation(
                        identifier=f"bounds:{name}[{index}]",
                        description=f"index {index} is within {name}.len() == {lengths[name]}",
                        status=status,
                        span=getattr(node, "span", None),
                    )
                )
            else:
                obligations.append(
                    ProofObligation(
                        identifier=f"bounds:{name}:dynamic",
                        description=f"dynamic index into {name} is unproved",
                        status="unknown",
                        span=getattr(node, "span", None),
                    )
                )
    return obligations


@dataclass(frozen=True)
class LockOrder:
    """A declared lock-acquisition order; a cycle means deadlock is possible."""

    edges: tuple[tuple[str, str], ...]
    span: tuple[int, int] | None = None

    def check(self) -> bool:
        return self.cycle() is None

    def cycle(self) -> tuple[str, ...] | None:
        """Return one deterministic cycle witness, if the order is cyclic."""
        adjacency: dict[str, set[str]] = {}
        nodes: set[str] = set()
        for before, after in self.edges:
            nodes.update((before, after))
            adjacency.setdefault(before, set()).add(after)
            adjacency.setdefault(after, set())

        path: list[str] = []
        positions: dict[str, int] = {}
        visited: set[str] = set()

        def visit(node: str) -> tuple[str, ...] | None:
            if node in positions:
                return tuple(path[positions[node] :] + [node])
            if node in visited:
                return None
            positions[node] = len(path)
            path.append(node)
            for neighbour in sorted(adjacency.get(node, ())):
                cycle = visit(neighbour)
                if cycle is not None:
                    return cycle
            path.pop()
            del positions[node]
            visited.add(node)
            return None

        for node in sorted(nodes):
            cycle = visit(node)
            if cycle is not None:
                return cycle
        return None


def verify_lock_order(order: LockOrder) -> ProofObligation:
    cycle = order.cycle()
    if cycle is not None:
        witness = " -> ".join(cycle)
        raise PillarError(
            "E002",
            f"declared lock order contains a cycle: {witness}",
            reason=f"the acquisition path {witness} permits circular wait",
            help_text="choose a total order and remove the back edge",
            span=order.span,
        )
    return ProofObligation(
        identifier="deadlock:lock-order",
        description=f"{len(order.edges)} lock-order edges are acyclic",
        status="proved",
    )


@dataclass(frozen=True)
class Counterexample:
    """A reproducible failing obligation rendered as a test seed."""

    condition: str
    status: str
    detail: str

    def render(self) -> str:
        return f"counterexample for `{self.condition}`: {self.status} ({self.detail})"


def extract_counterexample_trace(
    condition: Cond,
    env: dict[str, Interval] | None = None,
) -> Counterexample:
    """Return the smallest reproducible failure record for a condition.

    The checker is interval-based, so the result is an interval violation
    rather than a concrete SMT model.  A proved condition has no counterexample
    and is rejected with E002.
    """
    status = condition.check(env or {})
    if status == "proved":
        raise PillarError(
            "E002",
            "the condition is proved and has no counterexample",
            help_text="pass an unproved condition to extract a trace",
        )
    return Counterexample(
        condition=condition.render(),
        status=status,
        detail="interval or polynomial evidence was insufficient to prove the obligation",
    )


__all__ = [
    "Cond",
    "Counterexample",
    "ExprNode",
    "Interval",
    "LockOrder",
    "ParamSpec",
    "ProofCertificate",
    "ProofObligation",
    "ProofStep",
    "SynthesisSpec",
    "check_certificate",
    "compile_both",
    "emit_function_ast",
    "emit_program",
    "emit_source",
    "environment",
    "interpret_spec",
    "parse_condition",
    "synthesize",
    "verify_bounds",
    "verify_lock_order",
]
