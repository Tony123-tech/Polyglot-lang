"""Zero-intrusion AST lowering pass for Zig-exclusive features."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any

from compiler.core.ast_nodes import Binary, Call, Ident, Program

_SAT_FN_MAP = {
    "+|": "_zig_sat_add",
    "-|": "_zig_sat_sub",
    "*|": "_zig_sat_mul",
}


def lower_zig_ast(program: Program) -> Program:
    """Return a copy of `program` with Zig operators lowered to standard Calls."""
    cloned = deepcopy(program)
    _rewrite_node(cloned)
    return cloned


def _rewrite_node(node: Any) -> Any:
    if isinstance(node, Binary) and node.op in _SAT_FN_MAP:
        left = _rewrite_node(node.left)
        right = _rewrite_node(node.right)
        fn_name = _SAT_FN_MAP[node.op]
        return Call(
            line=node.line,
            col=node.col,
            func=Ident(node.line, node.col, fn_name),
            args=[left, right],
            span=node.span,
        )

    if is_dataclass(node):
        for f in fields(node):
            if f.name in {"line", "col", "span"}:
                continue
            val = getattr(node, f.name)
            if isinstance(val, list):
                setattr(node, f.name, [_rewrite_node(item) for item in val])
            elif is_dataclass(val):
                setattr(node, f.name, _rewrite_node(val))

    return node
