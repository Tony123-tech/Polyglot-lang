"""Frontend boundary validation shared by every backend, run before WAT lowering.

The WASM backend deliberately executes a restricted value representation, but
*diagnostics* are not a backend-specific concern.  A malformed pointer-width
cast, an unprovable ``#[terminates]`` annotation, a bad host-seam arity, or an
invalid integer literal must surface with the same normalized ``E001``–``E008``
code and message as the Python and native paths.  This module reuses the exact
error types used by those paths instead of reimplementing diagnostic wording.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from compiler.core.ast_nodes import (
    Call,
    Cast,
    FnDecl,
    Ident,
    LetStmt,
    LitBool,
    LitInt,
    LitStr,
    MethodCall,
    Node,
    Program,
    TypeName,
)
from backends.errors import InvalidIntegerParseLiteralError, check_host_call_arity
from backends.python.translator import TypeErrorForBoundary
from compiler.analysis.termination import check_function, check_program

_INTEGER_TYPES = {
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
}


def _type_name(ty: Any) -> str | None:
    if isinstance(ty, TypeName):
        return ty.name
    return None


def _known_type(expr: Any, locals_: dict[str, str]) -> str | None:
    """Return the narrow static type used by the shared boundary check."""
    if isinstance(expr, LitInt):
        return "i64"
    if isinstance(expr, LitBool):
        return "bool"
    if isinstance(expr, LitStr):
        return "String"
    if isinstance(expr, Ident):
        return locals_.get(expr.name)
    if isinstance(expr, Cast):
        return _type_name(expr.ty)
    if isinstance(expr, MethodCall):
        if expr.method in {"len", "is_empty"}:
            return "usize"
        return None
    if isinstance(expr, Call) and isinstance(expr.func, Ident):
        return locals_.get(f"@return:{expr.func.name}")
    return None


def _children(node: Any) -> list[Any]:
    result: list[Any] = []
    for field in fields(node):
        value = getattr(node, field.name, None)
        if isinstance(value, Node):
            result.append(value)
        elif isinstance(value, (list, tuple)):
            result.extend(item for item in value if isinstance(item, Node))
    return result


def _validate_call(call: Call) -> None:
    from backends.errors import HOST_CALL_ARITIES

    name: str | None = None
    if isinstance(call.func, Ident):
        name = call.func.name
    elif hasattr(call.func, "segments") and call.func.segments:
        name = call.func.segments[-1]
    if name is not None and name in HOST_CALL_ARITIES:
        check_host_call_arity(
            name, call.args, line=call.line, col=call.col, span=call.span
        )


def _validate_parse(call: MethodCall) -> None:
    if call.method != "parse":
        return
    receiver = call.receiver
    if isinstance(receiver, Node) and hasattr(receiver, "inner"):
        receiver = receiver.inner
    if not isinstance(receiver, LitStr):
        return
    try:
        int(receiver.value, 10)
    except ValueError:
        raise InvalidIntegerParseLiteralError(
            receiver.value,
            receiver.line,
            receiver.col,
            receiver.span,
        ) from None


def _validate_let(stmt: LetStmt, locals_: dict[str, str]) -> None:
    expected = _type_name(stmt.ty)
    if expected is None or stmt.value is None:
        return
    source = _known_type(stmt.value, locals_)
    if source is None:
        return
    if source in _INTEGER_TYPES and expected in _INTEGER_TYPES and source != expected:
        raise TypeErrorForBoundary(
            source,
            expected,
            stmt.line,
            stmt.col,
            stmt.span or getattr(stmt.value, "span", None),
        )


def _validate_item(item: Any, proven_terminating: set[str] | None = None) -> None:
    if isinstance(item, FnDecl):
        if item.terminates:
            # TerminationError carries ``reason`` with the stable message the
            # differential harness normalizes to E002.
            check_function(item, proven_terminating=proven_terminating)
        locals_: dict[str, str] = {}
        for param in item.params:
            name = getattr(param.pattern, "name", None)
            type_name = _type_name(param.ty)
            if name and type_name:
                locals_[name] = type_name
        if item.ret is not None:
            locals_[f"@return:{item.name}"] = _type_name(item.ret) or ""
        _walk(item, locals_)


def _walk(node: Any, locals_: dict[str, str]) -> None:
    if isinstance(node, LetStmt):
        _validate_let(node, locals_)
        name = getattr(node.pattern, "name", None)
        declared = _type_name(node.ty)
        inferred = _known_type(node.value, locals_) if node.value is not None else None
        if name and (declared or inferred):
            locals_[name] = declared or inferred or ""  # type: ignore[assignment]
    if isinstance(node, Call):
        _validate_call(node)
    if isinstance(node, MethodCall):
        _validate_parse(node)
    if not is_dataclass(node):
        return
    for child in _children(node):
        _walk(child, locals_)


def wasm_preflight_validate(program: Program) -> None:
    """Run the shared frontend checks before any WASM-specific lowering."""
    proven = check_program(program)
    for item in program.items:
        if isinstance(item, FnDecl):
            _validate_item(item, proven)
        elif hasattr(item, "items"):
            for nested in item.items:
                _validate_item(nested, proven)


__all__ = ["wasm_preflight_validate"]
