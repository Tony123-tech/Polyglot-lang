"""Typed splicing of macro holes.

A macro is a template plus a parameter type list. Expansion fills every
:class:`~compiler.macro_engine.ast.Hole` with the caller's argument after
checking the argument's inferred type against the parameter's declared type.
Only the small type set ``i64``/``str``/``bool``/``unknown`` is modelled, and
``unknown`` is accepted for names because the surrounding checker owns their
type.

Bounded model (SPEC 8.19, 10.49): recursion is capped, holes must be filled
exactly once, and leftover holes are an error rather than a partially expanded
tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .ast import (
    Add,
    Block,
    Call,
    Hole,
    If,
    Let,
    LitBool,
    LitInt,
    LitStr,
    Node,
    Return,
)


def infer_type(node: Node) -> str:
    """Best-effort type of a node in the miniature type language."""
    if isinstance(node, LitInt):
        return "i64"
    if isinstance(node, LitStr):
        return "str"
    if isinstance(node, LitBool):
        return "bool"
    if isinstance(node, Add):
        left, right = infer_type(node.left), infer_type(node.right)
        if left == "unknown" or right == "unknown":
            return "unknown"
        if left != "i64" or right != "i64":
            raise PillarError(
                "E001",
                f"'add' needs i64 operands, got {left} and {right}",
                help_text="cast or compare with i64 values before adding",
            )
        return "i64"
    if isinstance(node, Call):
        if node.callee == "println":
            return "unknown"
        return "unknown"
    if isinstance(node, If):
        return infer_type(node.then)
    if isinstance(node, Block):
        return infer_type(node.statements[-1]) if node.statements else "unknown"
    return "unknown"


@dataclass(frozen=True)
class ExpandResult:
    """The spliced tree and how much work the expansion did."""

    ast: Node
    holes_filled: int
    depth: int
    param_types: tuple[str, ...]


def _substitute(node: Node, args: tuple[Node, ...], counter: list[int]) -> Node:
    if isinstance(node, Hole):
        if node.index >= len(args):
            raise PillarError(
                "E002",
                f"macro hole ${node.index} has no matching argument",
                help_text=f"pass {node.index + 1} arguments to this macro",
            )
        counter[0] += 1
        return args[node.index]
    if isinstance(node, Add):
        return Add(
            _substitute(node.left, args, counter),
            _substitute(node.right, args, counter),
        )
    if isinstance(node, Call):
        return Call(
            node.callee, tuple(_substitute(arg, args, counter) for arg in node.args)
        )
    if isinstance(node, If):
        return If(
            _substitute(node.cond, args, counter),
            _substitute(node.then, args, counter),
            _substitute(node.otherwise, args, counter),
        )
    if isinstance(node, Let):
        return Let(
            node.name,
            _substitute(node.value, args, counter),
            _substitute(node.body, args, counter),
        )
    if isinstance(node, Return):
        return Return(_substitute(node.value, args, counter))
    if isinstance(node, Block):
        return Block(
            tuple(
                _substitute(statement, args, counter) for statement in node.statements
            )
        )
    return node


def _count_holes(node: Node) -> int:
    return sum(
        1 if isinstance(child, Hole) else _count_holes(child)
        for child in node.children()
    )


def macro_expand_typed_ast_splice(
    template: Node,
    args: tuple[Node, ...],
    *,
    param_types: tuple[str, ...],
    depth: int = 0,
    max_depth: int = 64,
) -> ExpandResult:
    """Fill holes in ``template`` with ``args`` after checking their types."""
    if max_depth <= 0:
        raise PillarError("E003", "macro expansion depth limit must be positive")
    if depth > max_depth:
        raise PillarError(
            "E002",
            f"macro expansion exceeded the depth limit of {max_depth}",
            help_text="a macro is expanding itself; break the recursion",
        )
    if len(args) != len(param_types):
        raise PillarError(
            "E002",
            f"macro expects {len(param_types)} argument(s), got {len(args)}",
            help_text="match the parameter list declared with the macro",
        )
    for position, (arg, declared) in enumerate(zip(args, param_types, strict=True)):
        actual = infer_type(arg)
        if declared == "unknown" or actual == "unknown":
            continue
        if actual != declared:
            raise PillarError(
                "E001",
                f"macro argument {position} has type {actual}, expected {declared}",
                help_text="pass an argument of the declared type",
            )

    required = _count_holes(template)
    if required != len(args):
        raise PillarError(
            "E002",
            f"macro template has {required} hole(s) but {len(args)} argument(s) were supplied",
            help_text="every hole needs exactly one argument",
        )

    counter = [0]
    spliced = _substitute(template, args, counter)
    if _count_holes(spliced) != 0:
        raise PillarError("E002", "macro expansion left an unfilled hole")
    return ExpandResult(
        ast=spliced,
        holes_filled=counter[0],
        depth=depth,
        param_types=param_types,
    )
