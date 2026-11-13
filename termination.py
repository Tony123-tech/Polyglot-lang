"""Conservative termination checking for annotated functions."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from ast_nodes import (
    Assign,
    Binary,
    Call,
    CompoundAssign,
    Continue,
    Expr,
    ExprStmt,
    FnDecl,
    For,
    Ident,
    IfExpr,
    IfLet,
    Index,
    LetStmt,
    LitInt,
    Loop,
    MacroCall,
    MethodCall,
    Node,
    Paren,
    PatIdent,
    Range,
    TypeName,
    TypeRef,
    Unary,
    While,
)


class TerminationError(Exception):
    """Raised when an annotated function has no supported termination proof."""

    code = "E002"

    def __init__(
        self,
        fn_name: str,
        reason: str,
        suggestion: str,
        line: int,
        col: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.fn_name = fn_name
        self.reason = reason
        self.suggestion = suggestion
        self.line = line
        self.col = col
        self.span = span
        super().__init__(self.format())

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{self.code}]: cannot prove termination of `{self.fn_name}`\n"
            f"  reason: {self.reason}\n"
            f"  help: {self.suggestion}\n"
            "  note: this function is annotated #[terminates]\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()


def _children(value: object) -> list[object]:
    if not is_dataclass(value):
        return []
    return [
        getattr(value, item.name)
        for item in fields(value)
        if item.name not in {"line", "col"}
    ]


def _walk(value: object):
    if isinstance(value, Node):
        yield value
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)
    elif is_dataclass(value):
        for child in _children(value):
            yield from _walk(child)


def _unwrap(expr: Expr) -> Expr:
    while isinstance(expr, Paren):
        expr = expr.inner
    return expr


def _positive_integer(expr: Expr) -> int | None:
    expr = _unwrap(expr)
    if isinstance(expr, Unary) and expr.op == "-":
        return None
    if not isinstance(expr, LitInt):
        return None
    raw = expr.raw.replace("_", "")
    try:
        value = int(raw, 0)
    except ValueError:
        try:
            value = int(raw, 10)
        except ValueError:
            return None
    return value if value > 0 else None


def _condition_lower_bounds(expr: Expr) -> dict[str, int]:
    expr = _unwrap(expr)
    if isinstance(expr, Binary) and expr.op == "&&":
        return {
            **_condition_lower_bounds(expr.left),
            **_condition_lower_bounds(expr.right),
        }
    if not isinstance(expr, Binary):
        return {}
    left = _unwrap(expr.left)
    right = _unwrap(expr.right)
    if isinstance(left, Ident) and isinstance(right, LitInt):
        try:
            bound = int(right.raw.replace("_", ""), 0)
        except ValueError:
            bound = right.value
        if expr.op == ">" and bound >= 0:
            return {left.name: bound + 1}
        if expr.op == ">=" and bound >= 0:
            return {left.name: bound}
    if isinstance(right, Ident) and isinstance(left, LitInt):
        try:
            bound = int(left.raw.replace("_", ""), 0)
        except ValueError:
            bound = left.value
        if expr.op == "<" and bound >= 0:
            return {right.name: bound + 1}
        if expr.op == "<=" and bound >= 0:
            return {right.name: bound}
    return {}


def _decreasing_integer_assignment(expr: Expr, name: str) -> bool:
    expr = _unwrap(expr)
    if not isinstance(expr, Binary) or expr.op != "-":
        return False
    left = _unwrap(expr.left)
    return (
        isinstance(left, Ident)
        and left.name == name
        and _positive_integer(expr.right) is not None
    )


def _assignment_target_name(expr: Expr) -> str | None:
    target = _unwrap(expr)
    return target.name if isinstance(target, Ident) else None


def _while_variant(loop: While) -> str | None:
    lower_bounds = _condition_lower_bounds(loop.cond)
    body = loop.body
    direct_updates: dict[str, list[bool]] = {}
    shadowed_names = {
        node.pattern.name
        for node in _walk(body)
        if isinstance(node, LetStmt)
        and isinstance(node.pattern, PatIdent)
    }
    for stmt in body.stmts:
        if not isinstance(stmt, ExprStmt):
            continue
        expr = _unwrap(stmt.expr)
        if isinstance(expr, Assign):
            name = _assignment_target_name(expr.target)
            if name is not None:
                direct_updates.setdefault(name, []).append(
                    _decreasing_integer_assignment(expr.value, name)
                )
        elif isinstance(expr, CompoundAssign):
            name = _assignment_target_name(expr.target)
            if name is not None:
                positive = _positive_integer(expr.value) is not None
                direct_updates.setdefault(name, []).append(
                    expr.op == "-=" and positive
                )
    for name, bound in lower_bounds.items():
        if bound < 0 or name in shadowed_names:
            continue
        updates = direct_updates.get(name, [])
        all_updates: list[bool] = []
        for node in _walk(body):
            if isinstance(node, Assign):
                target_name = _assignment_target_name(node.target)
                if target_name == name:
                    all_updates.append(
                        _decreasing_integer_assignment(node.value, name)
                    )
            elif isinstance(node, CompoundAssign):
                target_name = _assignment_target_name(node.target)
                if target_name == name:
                    all_updates.append(
                        node.op == "-="
                        and _positive_integer(node.value) is not None
                    )
        if updates and all(updates) and all(all_updates) and not any(
            isinstance(node, Continue)
            for stmt in body.stmts
            for node in _walk(stmt)
        ):
            return name
    return None


def _parameter_info(fn: FnDecl) -> list[tuple[str | None, str | None]]:
    result: list[tuple[str | None, str | None]] = []
    for param in fn.params:
        pattern = param.pattern
        name = pattern.name if isinstance(pattern, PatIdent) else None
        ty = param.ty
        while isinstance(ty, TypeRef):
            ty = ty.inner
        type_name = ty.name if isinstance(ty, TypeName) else None
        result.append((name, type_name))
    return result


def _slice_decreases(expr: Expr, name: str, positive_names: set[str]) -> bool:
    expr = _unwrap(expr)
    if isinstance(expr, MethodCall) and expr.method == "to_vec" and not expr.args:
        expr = _unwrap(expr.receiver)
    if not isinstance(expr, Index):
        return False
    receiver = _unwrap(expr.receiver)
    index = _unwrap(expr.index)
    if not isinstance(receiver, Ident) or receiver.name != name:
        return False
    if not isinstance(index, Range) or index.end is not None:
        return False
    start = index.start
    if start is None:
        return False
    start = _unwrap(start)
    if _positive_integer(start) is not None:
        return True
    return isinstance(start, Ident) and start.name in positive_names


def _recursive_call_decreases(
    fn: FnDecl,
    call: Call,
    lower_bounds: dict[str, int],
    parameters: list[tuple[str | None, str | None]],
) -> bool:
    decreasing = False
    for index, (parameter_name, parameter_type) in enumerate(parameters):
        if parameter_name is None or index >= len(call.args):
            continue
        argument = _unwrap(call.args[index])
        if isinstance(argument, Ident) and argument.name == parameter_name:
            continue
        if parameter_type == "i64":
            if (
                _decreasing_integer_assignment(argument, parameter_name)
                and lower_bounds.get(parameter_name, -1) >= 0
            ):
                decreasing = True
                continue
            return False
        if parameter_type in {"Vec", "String"}:
            positive_names = {
                name for name, lower_bound in lower_bounds.items()
                if lower_bound > 0
            }
            if _slice_decreases(argument, parameter_name, positive_names):
                decreasing = True
                continue
            return False
        return False
    return decreasing


def check_function(fn: FnDecl) -> None:
    """Raise TerminationError if an annotated function cannot be proven total."""
    if not fn.terminates:
        return
    if fn.body is None:
        raise TerminationError(
            fn.name,
            "the function has no body to analyze",
            "provide a body with a supported decreasing measure",
            fn.line,
            fn.col,
            getattr(fn, "span", None),
        )
    if fn.is_async:
        raise TerminationError(
            fn.name,
            "async suspension is outside the supported termination proof",
            "remove async from the annotated function",
            fn.line,
            fn.col,
            getattr(fn, "span", None),
        )

    parameters = _parameter_info(fn)
    parameter_types = {
        name: ty for name, ty in parameters if name is not None
    }
    parameter_names = set(parameter_types)
    body_nodes = list(_walk(fn.body))
    recursive_calls = [
        node for node in body_nodes
        if isinstance(node, Call)
        and isinstance(node.func, Ident)
        and node.func.name == fn.name
    ]
    if recursive_calls:
        mutated = {
            _assignment_target_name(node.target)
            for node in body_nodes
            if isinstance(node, (Assign, CompoundAssign))
        } & parameter_names
        shadowed = {
            node.pattern.name
            for node in body_nodes
            if isinstance(node, LetStmt)
            and isinstance(node.pattern, PatIdent)
        } & parameter_names
        shadows_function = (
            fn.name in parameter_names
            or any(
                isinstance(node, FnDecl) and node is not fn
                and node.name == fn.name
                for node in body_nodes
            )
            or fn.name in {
                node.pattern.name
                for node in body_nodes
                if isinstance(node, LetStmt)
                and isinstance(node.pattern, PatIdent)
            }
        )
        if mutated or shadowed or shadows_function:
            node = recursive_calls[0]
            raise TerminationError(
                fn.name,
                "a recursive parameter is reassigned or shadowed",
                "keep the decreasing parameter unchanged except in the "
                "recursive argument",
                node.line,
                node.col,
                getattr(fn, "span", None),
            )

    def fail(reason: str, suggestion: str, node: Node) -> None:
        raise TerminationError(
            fn.name, reason, suggestion, node.line, node.col,
            getattr(fn, "span", None),
        )

    def visit(value: object, lower_bounds: dict[str, int]) -> None:
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child, lower_bounds)
            return
        if not isinstance(value, Node):
            return

        if isinstance(value, Loop):
            fail(
                "an unbounded loop has no supported termination proof",
                "use a bounded `for` loop or a decreasing `while` variant",
                value,
            )
        if isinstance(value, While):
            variant = _while_variant(value)
            if variant is None:
                fail(
                    "the while condition and body do not prove a decreasing "
                    "variant with a lower bound",
                    "guard a variable with a non-negative lower bound and "
                    "decrease it on every iteration",
                    value,
                )
            visit(value.cond, lower_bounds)
            body_bounds = {
                **lower_bounds,
                **_condition_lower_bounds(value.cond),
            }
            visit(value.body, body_bounds)
            return
        if isinstance(value, For):
            if not isinstance(_unwrap(value.iter), Range):
                fail(
                    "the iterator is not a statically bounded range",
                    "iterate over a finite `start..end` range",
                    value,
                )
            range_expr = _unwrap(value.iter)
            if range_expr.start is None or range_expr.end is None:
                fail(
                    "the range does not have finite explicit bounds",
                    "provide explicit start and end bounds",
                    value,
                )
            visit(value.iter, lower_bounds)
            visit(value.body, lower_bounds)
            return
        if isinstance(value, IfExpr):
            visit(value.cond, lower_bounds)
            then_bounds = dict(lower_bounds)
            for name, bound in _condition_lower_bounds(value.cond).items():
                then_bounds[name] = max(then_bounds.get(name, bound), bound)
            visit(value.then_block, then_bounds)
            if value.else_branch is not None:
                visit(value.else_branch, lower_bounds)
            return
        if isinstance(value, IfLet):
            visit(value.value, lower_bounds)
            visit(value.then_block, lower_bounds)
            if value.else_branch is not None:
                visit(value.else_branch, lower_bounds)
            return
        if isinstance(value, Call):
            if isinstance(value.func, Ident) and value.func.name == fn.name:
                if not _recursive_call_decreases(
                    fn, value, lower_bounds, parameters,
                ):
                    fail(
                        "a recursive call has no provably decreasing "
                        "argument",
                        "recurse with a smaller argument under a matching "
                        "lower-bound guard",
                        value,
                    )
            else:
                fail(
                    "a non-recursive function call is outside the supported "
                    "proof subset",
                    "inline the call or provide a proof supported by the "
                    "termination checker",
                    value,
                )
            for child in _children(value):
                visit(child, lower_bounds)
            return
        if isinstance(value, MethodCall):
            receiver = _unwrap(value.receiver)
            receiver_type = (
                parameter_types.get(receiver.name)
                if isinstance(receiver, Ident)
                else None
            )
            is_slice_to_vec = (
                value.method == "to_vec"
                and isinstance(receiver, Index)
                and isinstance(_unwrap(receiver.index), Range)
            )
            is_known_length = (
                value.method in {"len", "is_empty"}
                and receiver_type in {"Vec", "String"}
            )
            if not (is_slice_to_vec or is_known_length):
                fail(
                    f"method call `{value.method}` is outside the supported "
                    "proof subset",
                    "use only a known finite operation in an annotated "
                    "function",
                    value,
                )
            for child in _children(value):
                visit(child, lower_bounds)
            return

        if isinstance(value, MacroCall):
            if value.path not in {"panic", "assert", "assert_eq", "assert_ne"}:
                fail(
                    f"macro call `{value.path}!` is outside the supported "
                    "proof subset",
                    "remove the macro or use a known terminating operation",
                    value,
                )
            for child in _children(value):
                visit(child, lower_bounds)
            return

        for child in _children(value):
            visit(child, lower_bounds)

    visit(fn.body, {})
