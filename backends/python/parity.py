"""Closed AST evaluator and parity helpers for the Python backend rollout.

All evaluation goes through :func:`_eval`, a small interpreter over genuine
:mod:`ast_nodes` trees.  It supports literals, identifiers, arithmetic,
comparisons, calls to a fixed whitelist of pure builtins, blocks, conditionals,
matches, casts, and returns.  User source text is never passed to ``eval`` or
``exec``; unsupported AST nodes raise ``E002`` rather than being executed by
Python.  Parity records compare outcomes as values, including normalised
exception payloads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NoReturn

import compiler.core.ast_nodes as astn
from compiler.frontend.errors import PolyglotError

__all__ = [
    "pyparity_build_case",
    "pyparity_compare_outcomes",
    "pyparity_compare_values",
    "pyparity_diff_report",
    "pyparity_eval_binary",
    "pyparity_eval_block",
    "pyparity_eval_call",
    "pyparity_eval_expr",
    "pyparity_eval_function",
    "pyparity_eval_if",
    "pyparity_eval_literal",
    "pyparity_eval_match",
    "pyparity_eval_pattern",
    "pyparity_eval_program",
    "pyparity_eval_unary",
    "pyparity_normalize_error",
    "pyparity_run_case",
    "pyparity_run_matrix",
    "pyparity_trace_summary",
    "pyparity_validate_oracle",
]

_SAFE_FUNCTIONS: Final[dict[str, Any]] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
    "sum": sum,
}
_LITERALS: Final[tuple[type[astn.Node], ...]] = (
    astn.LitInt,
    astn.LitFloat,
    astn.LitStr,
    astn.LitChar,
    astn.LitBool,
)
_BINARY: Final[frozenset[str]] = frozenset(
    {
        "%",
        "&",
        "*",
        "+",
        "-",
        "/",
        "<",
        "<=",
        "==",
        ">",
        ">=",
        "^",
        "!=",
        "|",
        "&&",
        "||",
    }
)
_MAX_STEPS: Final[int] = 512


# ---------------------------------------------------------------------------
# Internal helpers and evaluator
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured parity diagnostic."""
    raise PolyglotError(code, message)


def _text(value: object, label: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty str")
    return value


def _node(
    value: object, expected: type[astn.Node] | tuple[type[astn.Node], ...], label: str
) -> Any:
    """Require a genuine project AST node of an expected type."""
    if not isinstance(value, expected):
        _fail("E003", f"{label} must be a {getattr(expected, '__name__', 'AST node')}")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a string-keyed mapping."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E003", f"{label} keys must be str")
        result[key] = item
    return result


def _kind(value: object) -> str:
    """Return a stable kind label for an evaluated value."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


@dataclass
class _State:
    """Mutable evaluation budget and trace shared by a single evaluation."""

    trace: list[dict[str, Any]] = field(default_factory=list)
    budget: int = _MAX_STEPS

    def step(self, kind: str, detail: str) -> None:
        """Record one evaluation event or raise when the budget is exhausted."""
        self.budget -= 1
        if self.budget < 0:
            _fail("E002", "parity evaluation step budget exhausted")
        self.trace.append({"kind": kind, "detail": detail, "remaining": self.budget})


class _ReturnSignal(Exception):
    """Internal control-flow signal used by evaluated ``return`` expressions."""

    def __init__(self, value: Any) -> None:
        super().__init__("return")
        self.value = value


def _literal_value(node: object) -> Any:
    """Return the Python value of a literal AST node used in a pattern."""
    if isinstance(node, astn.LitInt):
        return int(node.value)
    if isinstance(node, astn.LitFloat):
        return float(node.value)
    if isinstance(node, (astn.LitStr, astn.LitChar)):
        return node.value
    if isinstance(node, astn.LitBool):
        return bool(node.value)
    if isinstance(node, astn.OptionNone):
        return ("None",)
    _fail("E002", f"unsupported literal pattern: {type(node).__name__}")


def _match_pattern(
    pattern: astn.Node, value: Any, state: _State, env: dict[str, Any]
) -> dict[str, Any] | None:
    """Return pattern bindings, or ``None`` when a pattern does not match."""
    if isinstance(pattern, astn.PatWildcard):
        return {}
    if isinstance(pattern, astn.PatIdent):
        return {_text(pattern.name, "binding name"): value}
    if isinstance(pattern, astn.PatLiteral):
        return {} if _literal_value(pattern.value) == value else None
    if isinstance(pattern, (astn.PatEnum, astn.PatPath, astn.PatConst)):
        tag = _text(pattern.path, "variant path").split("::")[-1]
        if not isinstance(value, tuple) or not value or value[0] != tag:
            return None
        if isinstance(pattern, astn.PatEnum) and len(value) - 1 != len(pattern.args):
            return None
        bindings: dict[str, Any] = {}
        if isinstance(pattern, astn.PatEnum):
            for item, payload in zip(pattern.args, value[1:]):
                found = _match_pattern(item, payload, state, env)
                if found is None:
                    return None
                bindings.update(found)
        return bindings
    if isinstance(pattern, astn.PatTuple):
        if not isinstance(value, tuple) or len(value) != len(pattern.elems):
            return None
        bindings = {}
        for item, payload in zip(pattern.elems, value):
            found = _match_pattern(item, payload, state, env)
            if found is None:
                return None
            bindings.update(found)
        return bindings
    if isinstance(pattern, (astn.PatRef, astn.PatBox)):
        return _match_pattern(pattern.inner, value, state, env)
    if isinstance(pattern, astn.PatBinding):
        found = _match_pattern(pattern.inner, value, state, env)
        if found is None:
            return None
        return {_text(pattern.name, "binding name"): value, **found}
    if isinstance(pattern, astn.PatOr):
        for alternative in pattern.alts:
            found = _match_pattern(alternative, value, state, env)
            if found is not None:
                return found
        return None
    if isinstance(pattern, astn.PatRange):
        low = None if pattern.start is None else _eval(pattern.start, state, env)
        high = None if pattern.end is None else _eval(pattern.end, state, env)
        if low is not None and value < low:
            return None
        if high is not None and (value > high if pattern.inclusive else value >= high):
            return None
        return {}
    if isinstance(pattern, astn.PatSlice):
        if not isinstance(value, (list, tuple)) or len(value) != len(pattern.elems):
            return None
        bindings = {}
        for item, payload in zip(pattern.elems, value):
            found = _match_pattern(item, payload, state, env)
            if found is None:
                return None
            bindings.update(found)
        return bindings
    _fail("E002", f"unsupported pattern node: {type(pattern).__name__}")


def _eval(node: astn.Node, state: _State, env: dict[str, Any]) -> Any:
    """Evaluate one project AST expression inside a closed environment."""
    if not isinstance(node, astn.Node):
        _fail("E003", "expression must be a project AST node")
    state.step("eval", type(node).__name__)
    if isinstance(node, _LITERALS):
        return _literal_value(node)
    if isinstance(node, astn.Ident):
        name = _text(node.name, "identifier")
        if name not in env:
            _fail("E002", f"unbound parity identifier: {name}")
        return env[name]
    if isinstance(node, astn.Paren):
        return _eval(node.inner, state, env)
    if isinstance(node, astn.Unary):
        operand = _eval(node.operand, state, env)
        if node.op == "-":
            return -operand
        if node.op == "+":
            return +operand
        if node.op in ("!", "not"):
            return not operand
        _fail("E002", f"unsupported unary operator: {node.op}")
    if isinstance(node, astn.Binary):
        operator = _text(node.op, "operator")
        if operator not in _BINARY:
            _fail("E002", f"unsupported binary operator: {operator}")
        left = _eval(node.left, state, env)
        right = _eval(node.right, state, env)
        return _apply_binary(operator, left, right)
    if isinstance(node, astn.TupleExpr):
        return tuple(_eval(item, state, env) for item in node.elems)
    if isinstance(node, astn.ArrayExpr):
        return [_eval(item, state, env) for item in node.elems]
    if isinstance(node, (astn.OptionSome, astn.ResultOk, astn.ResultErr)):
        tag = (
            "Some"
            if isinstance(node, astn.OptionSome)
            else ("Ok" if isinstance(node, astn.ResultOk) else "Err")
        )
        return (tag, _eval(node.value, state, env))
    if isinstance(node, astn.OptionNone):
        return ("None",)
    if isinstance(node, astn.Block):
        return _eval_block(node, state, env)
    if isinstance(node, astn.IfExpr):
        condition = _eval(node.cond, state, env)
        if condition:
            return _eval_branch(node.then_block, state, env)
        return (
            _eval_branch(node.else_branch, state, env)
            if node.else_branch is not None
            else None
        )
    if isinstance(node, astn.MatchExpr):
        subject = _eval(node.scrutinee, state, env)
        for arm in node.arms:
            if not isinstance(arm, astn.MatchArm):
                _fail("E003", "match arms must be MatchArm nodes")
            found = _match_pattern(arm.pattern, subject, state, env)
            if found is None:
                continue
            if arm.guard is not None:
                guarded = dict(env, **found)
                if not _eval(arm.guard, state, guarded):
                    continue
            return _eval(arm.body, state, dict(env, **found))
        _fail("E002", "non-exhaustive parity match")
    if isinstance(node, astn.Call):
        if not isinstance(node.func, astn.Ident):
            _fail("E002", "only direct calls to safe builtins are supported")
        name = _text(node.func.name, "function name")
        function = _SAFE_FUNCTIONS.get(name)
        if function is None:
            _fail("E002", f"function is not in the safe parity whitelist: {name}")
        arguments = [_eval(argument, state, env) for argument in node.args]
        return function(*arguments)
    if isinstance(node, astn.Cast):
        value = _eval(node.expr, state, env)
        target = node.ty
        if not isinstance(target, astn.TypeName):
            _fail("E002", "only named cast targets are supported")
        if target.name in ("f32", "f64"):
            return float(value)
        if target.name in ("bool",):
            return bool(value)
        if target.name in (
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
        ):
            return int(value)
        _fail("E002", f"unsupported cast target: {target.name}")
    if isinstance(node, astn.Return):
        raise _ReturnSignal(
            None if node.value is None else _eval(node.value, state, env)
        )
    _fail("E002", f"unsupported expression node: {type(node).__name__}")


def _apply_binary(operator: str, left: Any, right: Any) -> Any:
    """Apply one whitelisted binary operator to already-evaluated values."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator in ("/", "%") and right == 0:
        _fail("E002", "division by zero in parity evaluation")
    if operator == "/":
        return (
            int(left / right)
            if isinstance(left, int) and isinstance(right, int)
            else left / right
        )
    if operator == "%":
        return left % right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "&&":
        return bool(left) and bool(right)
    if operator == "||":
        return bool(left) or bool(right)
    if operator == "&":
        return left & right
    if operator == "|":
        return left | right
    if operator == "^":
        return left ^ right
    _fail("E002", f"unsupported binary operator: {operator}")


def _eval_branch(branch: astn.Node | None, state: _State, env: dict[str, Any]) -> Any:
    """Evaluate an if/else branch that may be an expression or a block."""
    if branch is None:
        return None
    return _eval(branch, state, env)


def _eval_block(block: astn.Block, state: _State, env: dict[str, Any]) -> Any:
    """Execute a block in a child scope and return its tail value."""
    _node(block, astn.Block, "block")
    local = dict(env)
    result: Any = None
    try:
        for statement in block.stmts:
            if isinstance(statement, astn.LetStmt):
                if not isinstance(statement.pattern, astn.PatIdent):
                    _fail("E002", "only identifier let-bindings are supported")
                if statement.value is None:
                    _fail("E002", "let-binding requires a value")
                local[_text(statement.pattern.name, "binding name")] = _eval(
                    statement.value, state, local
                )
            elif isinstance(statement, astn.ExprStmt):
                _eval(statement.expr, state, local)
            else:
                _fail("E002", f"unsupported statement node: {type(statement).__name__}")
        if block.tail is not None:
            result = _eval(block.tail, state, local)
    except _ReturnSignal as signal:
        result = signal.value
    return result


def _outcome(value: Any = None, error: str | None = None) -> dict[str, Any]:
    """Build a stable success or error outcome record."""
    if error is None:
        return {"ok": True, "value": value, "error": None}
    return {"ok": False, "value": None, "error": error}


def _canonical(value: Any) -> str:
    """Serialize a value into a canonical comparison string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Public parity helpers
# ---------------------------------------------------------------------------


def pyparity_eval_literal(node: astn.Node) -> dict[str, Any]:
    """Evaluate one literal AST node into its Python value."""
    value = _literal_value(_node(node, _LITERALS, "literal"))
    return {"value": value, "kind": _kind(value), "node": type(node).__name__}


def pyparity_eval_expr(
    node: astn.Node, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate a closed project AST expression with a supplied environment."""
    state = _State()
    value = _eval(node, state, _mapping(env or {}, "env"))
    return {
        "value": value,
        "kind": _kind(value),
        "steps": len(state.trace),
        "trace": list(state.trace),
    }


def pyparity_eval_binary(
    left: astn.Node,
    operator: str,
    right: astn.Node,
    *,
    env: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one binary operation from two project AST expressions."""
    op = _text(operator, "operator")
    if op not in _BINARY:
        _fail("E002", f"unsupported binary operator: {op}")
    state = _State()
    scope = _mapping(env or {}, "env")
    value = _apply_binary(op, _eval(left, state, scope), _eval(right, state, scope))
    return {
        "value": value,
        "operator": op,
        "kind": _kind(value),
        "steps": len(state.trace),
    }


def pyparity_eval_unary(
    operand: astn.Node, operator: str, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate one unary operation from a project AST expression."""
    op = _text(operator, "operator")
    if op not in ("-", "+", "!", "not"):
        _fail("E002", f"unsupported unary operator: {op}")
    state = _State()
    value = _eval(astn.Unary(0, 0, op, operand), state, _mapping(env or {}, "env"))
    return {
        "value": value,
        "operator": op,
        "kind": _kind(value),
        "steps": len(state.trace),
    }


def pyparity_eval_call(
    call: astn.Call, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate a direct call to a whitelisted pure builtin."""
    _node(call, astn.Call, "call")
    if not isinstance(call.func, astn.Ident):
        _fail("E002", "only direct calls to safe builtins are supported")
    name = _text(call.func.name, "function name")
    function = _SAFE_FUNCTIONS.get(name)
    if function is None:
        _fail("E002", f"function is not in the safe parity whitelist: {name}")
    state = _State()
    scope = _mapping(env or {}, "env")
    arguments = [_eval(argument, state, scope) for argument in call.args]
    return {
        "value": function(*arguments),
        "function": name,
        "arguments": arguments,
        "steps": len(state.trace),
    }


def pyparity_eval_block(
    block: astn.Block, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate a block in a child scope and report its final bindings."""
    state = _State()
    base = _mapping(env or {}, "env")
    value = _eval_block(block, state, base)
    local = dict(base)
    for statement in block.stmts:
        if isinstance(statement, astn.LetStmt) and isinstance(
            statement.pattern, astn.PatIdent
        ):
            local[statement.pattern.name] = (
                _eval(statement.value, state, local)
                if statement.value is not None
                else None
            )
    return {
        "value": value,
        "bindings": local,
        "steps": len(state.trace),
        "trace": list(state.trace),
    }


def pyparity_eval_if(
    node: astn.IfExpr, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate an if expression and report which branch was selected."""
    _node(node, astn.IfExpr, "node")
    state = _State()
    scope = _mapping(env or {}, "env")
    condition = _eval(node.cond, state, scope)
    if condition:
        branch = "then"
        value = _eval_branch(node.then_block, state, scope)
    else:
        branch = "else"
        value = _eval_branch(node.else_branch, state, scope)
    return {
        "value": value,
        "branch": branch,
        "condition": condition,
        "steps": len(state.trace),
    }


def pyparity_eval_match(
    node: astn.MatchExpr, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate a match expression and report the selected arm and bindings."""
    _node(node, astn.MatchExpr, "node")
    state = _State()
    scope = _mapping(env or {}, "env")
    subject = _eval(node.scrutinee, state, scope)
    for index, arm in enumerate(node.arms):
        found = _match_pattern(arm.pattern, subject, state, scope)
        if found is None:
            continue
        guarded = dict(scope, **found)
        if arm.guard is not None and not _eval(arm.guard, state, guarded):
            continue
        value = _eval(arm.body, state, guarded)
        return {
            "value": value,
            "arm": index,
            "bindings": found,
            "subject": subject,
            "steps": len(state.trace),
        }
    _fail("E002", "non-exhaustive parity match")


def pyparity_eval_pattern(
    pattern: astn.Node, value: Any, *, env: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Match one AST pattern against a value without executing source text."""
    state = _State()
    found = _match_pattern(pattern, value, state, _mapping(env or {}, "env"))
    return {
        "matched": found is not None,
        "bindings": found or {},
        "pattern": type(pattern).__name__,
    }


def pyparity_eval_function(
    function: astn.FnDecl, *, arguments: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate one parsed function declaration with explicit arguments."""
    _node(function, astn.FnDecl, "function")
    if function.body is None:
        _fail("E002", "function has no body to evaluate")
    supplied = _mapping(arguments or {}, "arguments")
    parameters = [
        _text(parameter.pattern.name, "parameter name")
        for parameter in function.params
        if isinstance(parameter.pattern, astn.PatIdent)
    ]
    if len(parameters) != len(function.params):
        _fail("E002", "only identifier parameters are supported")
    unknown = sorted(set(supplied) - set(parameters))
    if unknown:
        _fail("E002", f"unknown arguments: {', '.join(unknown)}")
    missing = [name for name in parameters if name not in supplied]
    if missing:
        _fail("E002", f"missing arguments: {', '.join(missing)}")
    state = _State()
    env = dict(supplied)
    value = _eval_block(function.body, state, env)
    return {
        "function": function.name,
        "value": value,
        "bindings": env,
        "steps": len(state.trace),
        "trace": list(state.trace),
    }


def pyparity_eval_program(
    program: astn.Program,
    *,
    entry: str | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate zero-argument functions or one named entry function."""
    _node(program, astn.Program, "program")
    functions = [item for item in program.items if isinstance(item, astn.FnDecl)]
    names = [function.name for function in functions]
    if entry is not None:
        selected = [
            function for function in functions if function.name == _text(entry, "entry")
        ]
        if not selected:
            _fail("E006", f"no such parity entry function: {entry}")
    else:
        selected = [function for function in functions if not function.params]
    results: dict[str, Any] = {}
    steps = 0
    trace: list[dict[str, Any]] = []
    for function in selected:
        result = pyparity_eval_function(function, arguments=arguments)
        results[function.name] = result["value"]
        steps += result["steps"]
        trace.extend(result["trace"])
    return {
        "functions": names,
        "results": results,
        "steps": steps,
        "trace": trace,
    }


def pyparity_compare_values(left: Any, right: Any) -> dict[str, Any]:
    """Compare two evaluated values using a canonical, type-aware encoding."""
    equal = _canonical(left) == _canonical(right)
    return {
        "equal": equal,
        "left": left,
        "right": right,
        "reason": "canonical encodings match"
        if equal
        else "canonical encodings differ",
    }


def pyparity_compare_outcomes(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two success/error outcome records as parity values."""
    first, second = _mapping(left, "left"), _mapping(right, "right")
    for label, record in (("left", first), ("right", second)):
        if "ok" not in record:
            _fail("E003", f"{label} outcome is missing the ok field")
        if not isinstance(record["ok"], bool):
            _fail("E003", f"{label} outcome ok field must be bool")
    if first["ok"] != second["ok"]:
        return {
            "equal": False,
            "left_kind": "ok" if first["ok"] else "error",
            "right_kind": "ok" if second["ok"] else "error",
            "reason": "success/error kinds differ",
        }
    if first["ok"]:
        comparison = pyparity_compare_values(first.get("value"), second.get("value"))
    else:
        comparison = pyparity_compare_values(
            pyparity_normalize_error(first.get("error")),
            pyparity_normalize_error(second.get("error")),
        )
    return {
        "equal": comparison["equal"],
        "left_kind": "ok" if first["ok"] else "error",
        "right_kind": "ok" if second["ok"] else "error",
        "reason": comparison["reason"],
    }


def pyparity_build_case(
    function: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic parity case descriptor with a stable case id."""
    name = _text(function, "function")
    values = _mapping(arguments or {}, "arguments")
    description = label if label is not None else name
    payload = json.dumps(
        {"function": name, "arguments": values, "label": description},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "function": name,
        "arguments": dict(sorted(values.items())),
        "label": _text(description, "label"),
        "case_id": hashlib.sha256(payload).hexdigest()[:16],
    }


def pyparity_run_case(program: astn.Program, case: Mapping[str, Any]) -> dict[str, Any]:
    """Run one parity case, treating evaluation failures as error outcomes."""
    _node(program, astn.Program, "program")
    descriptor = _mapping(case, "case")
    for field_name in ("function", "arguments"):
        if field_name not in descriptor:
            _fail("E003", f"case is missing {field_name}")
    try:
        result = pyparity_eval_program(
            program,
            entry=_text(descriptor["function"], "function"),
            arguments=_mapping(descriptor["arguments"], "arguments"),
        )
        outcome = _outcome(value=result["results"][descriptor["function"]])
        steps = result["steps"]
        trace = result["trace"]
    except PolyglotError as error:
        outcome = _outcome(error=pyparity_normalize_error(error)["signature"])
        steps = 0
        trace = []
    return {
        "function": descriptor["function"],
        "arguments": dict(
            sorted(_mapping(descriptor["arguments"], "arguments").items())
        ),
        "outcome": outcome,
        "steps": steps,
        "trace": trace,
    }


def pyparity_run_matrix(
    program: astn.Program,
    cases: Sequence[Mapping[str, Any]],
    *,
    reference_program: astn.Program | None = None,
) -> dict[str, Any]:
    """Run a matrix of cases and compare each outcome with an oracle program."""
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        _fail("E003", "cases must be a sequence of case mappings")
    results: list[dict[str, Any]] = []
    matched = 0
    for case in cases:
        actual = pyparity_run_case(program, case)
        expected = (
            pyparity_run_case(reference_program, case)
            if reference_program is not None
            else pyparity_run_case(program, case)
        )
        comparison = pyparity_compare_outcomes(expected["outcome"], actual["outcome"])
        if comparison["equal"]:
            matched += 1
        results.append(
            {
                "function": actual["function"],
                "outcome": actual["outcome"],
                "expected": expected["outcome"],
                "equal": comparison["equal"],
            }
        )
    return {
        "cases": results,
        "matched": matched,
        "mismatched": len(results) - matched,
        "total": len(results),
    }


def pyparity_diff_report(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a structured diff between two sequences of outcomes."""
    if isinstance(expected, (str, bytes)) or isinstance(actual, (str, bytes)):
        _fail("E003", "expected and actual must be sequences of outcomes")
    left, right = list(expected), list(actual)
    total = max(len(left), len(right))
    differences: list[dict[str, Any]] = []
    same = 0
    for index in range(total):
        if index >= len(left) or index >= len(right):
            differences.append({"index": index, "reason": "missing outcome"})
            continue
        comparison = pyparity_compare_outcomes(left[index], right[index])
        if comparison["equal"]:
            same += 1
        else:
            differences.append({"index": index, "reason": comparison["reason"]})
    return {"differences": differences, "same": same, "total": total}


def pyparity_normalize_error(error: object) -> dict[str, Any]:
    """Normalise an exception, diagnostic, or message into a stable record."""
    if isinstance(error, BaseException):
        kind = type(error).__name__
        message = str(error)
    elif isinstance(error, Mapping) and "code" in error:
        record = _mapping(error, "error")
        kind = _text(record["code"], "code")
        message = _text(record.get("message", kind), "message")
    elif isinstance(error, str):
        if ": " in error:
            kind, message = error.split(": ", 1)
        else:
            kind = "ValueError"
            message = error
    else:
        _fail("E003", f"unsupported error payload: {type(error).__name__}")
    signature = f"{kind}: {message}" if message else kind
    return {"kind": kind, "message": message, "signature": signature}


def pyparity_validate_oracle(
    oracle: Mapping[str, Any], *, expected: Sequence[str] = ()
) -> dict[str, Any]:
    """Validate an oracle mapping without invoking any of its callables."""
    functions = _mapping(oracle, "oracle")
    valid: list[str] = []
    invalid: list[str] = []
    for name, candidate in functions.items():
        valid_name = name.isidentifier() if isinstance(name, str) else False
        if not valid_name or not callable(candidate):
            invalid.append(str(name))
        else:
            valid.append(name)
    missing = [name for name in expected if name not in functions]
    return {
        "functions": sorted(valid),
        "invalid": sorted(invalid),
        "missing": list(missing),
        "callable": not invalid and not missing,
        "count": len(valid),
    }


def pyparity_trace_summary(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise an evaluation trace without exposing environment values."""
    if isinstance(trace, (str, bytes)) or not isinstance(trace, Sequence):
        _fail("E003", "trace must be a sequence of event mappings")
    kinds: dict[str, int] = {}
    events: list[str] = []
    functions: list[str] = []
    for event in trace:
        record = _mapping(event, "trace event")
        kind = _text(record.get("kind", "event"), "kind")
        detail = str(record.get("detail", ""))
        kinds[kind] = kinds.get(kind, 0) + 1
        events.append(f"{kind}:{detail}")
        if detail in _SAFE_FUNCTIONS:
            functions.append(detail)
    return {
        "steps": len(events),
        "events": events,
        "kinds": dict(sorted(kinds.items())),
        "functions": list(dict.fromkeys(functions)),
    }
