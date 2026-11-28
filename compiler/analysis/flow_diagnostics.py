"""Shared, lightweight control-flow diagnostics."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

from compiler.core.ast_nodes import Block, ExprStmt, Node, Program, Return

_SKIP_FIELDS = frozenset({"line", "col", "span", "source_lines"})


def _walk(value: object, seen: set[int] | None = None) -> Iterator[Node]:
    if seen is None:
        seen = set()
    if isinstance(value, Node):
        if id(value) in seen:
            return
        seen.add(id(value))
        yield value
        for field_name in getattr(value, "__dataclass_fields__", {}):
            if field_name in _SKIP_FIELDS:
                continue
            yield from _walk(getattr(value, field_name), seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, seen)


def _is_return(stmt: object) -> bool:
    return isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Return)


def warn_unreachable_after_return(program: Program) -> None:
    """Warn once for code that can never run because a `return` precedes it."""
    for node in _walk(program):
        if not isinstance(node, Block):
            continue
        return_index = next(
            (index for index, stmt in enumerate(node.stmts) if _is_return(stmt)),
            None,
        )
        if return_index is None:
            continue
        unreachable = (
            node.stmts[return_index + 1]
            if return_index + 1 < len(node.stmts)
            else node.tail
        )
        if unreachable is not None:
            warnings.warn(
                f"unreachable code after return at "
                f"{unreachable.line}:{unreachable.col}",
                UserWarning,
                stacklevel=4,
            )


__all__ = ["warn_unreachable_after_return"]
