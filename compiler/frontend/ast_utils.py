"""Read-only inspection helpers for :mod:`ast_nodes` trees.

Every public function in this module works on genuine :class:`ast_nodes.Node`
objects produced by the parser (or assembled from them by hand).  Traversal is
deterministic: nodes are visited in pre-order, following dataclass field
declaration order and, inside containers, list/tuple order and dict insertion
order.  The ``line``, ``col``, ``span`` and ``source_lines`` metadata fields are
never traversed as children.  Invalid inputs are reported as ``E003``
:class:`~compiler.frontend.errors.PolyglotError` diagnostics rather than as
generic Python exceptions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from typing import Any

import ast_nodes as ast
from compiler.frontend.errors import PolyglotError
from compiler.frontend.framework import _ast_to_dict as _framework_ast_to_dict

__all__ = [
    "ast_collect_literals",
    "ast_collect_symbols",
    "ast_depth",
    "ast_find_by_predicate",
    "ast_find_by_type",
    "ast_node_at_offset",
    "ast_node_count",
    "ast_node_kinds",
    "ast_to_json",
    "ast_walk",
]

_METADATA_FIELDS = frozenset({"line", "col", "span", "source_lines"})
_LITERAL_NODES = (ast.LitInt, ast.LitFloat, ast.LitBool, ast.LitStr, ast.LitChar)


def _api_error(message: str) -> PolyglotError:
    """Build the standard E003 API-misuse diagnostic used by this module."""
    return PolyglotError("E003", message)


def _is_node(value: object) -> bool:
    """Return whether *value* is a concrete dataclass AST node."""
    return isinstance(value, ast.Node) and is_dataclass(value)


def _container_items(value: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> list[Any]:
    """Return a container's children in deterministic iteration order."""
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _contains_node(value: object) -> bool:
    """Report whether any AST node occurs in *value* or one of its containers."""
    if _is_node(value):
        return True
    if isinstance(value, (list, tuple, dict)):
        return any(_contains_node(item) for item in _container_items(value))
    return False


def _validate_walk_input(node: object) -> None:
    """Reject values that are neither an AST node nor a container of nodes."""
    if _is_node(node):
        return
    if isinstance(node, (list, tuple, dict)):
        if not _contains_node(node):
            raise _api_error("expected an AST node or a container containing AST nodes")
        return
    raise _api_error(f"expected an AST node or container, got {type(node).__name__}")


def _walk_pairs(node: object, depth: int = 1) -> Iterator[tuple[object, int]]:
    """Yield ``(node, depth)`` pairs in pre-order; containers add no depth."""
    if _is_node(node):
        yield node, depth
        for field in fields(node):
            if field.name in _METADATA_FIELDS:
                continue
            yield from _walk_pairs(getattr(node, field.name), depth + 1)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_pairs(item, depth)
    elif isinstance(node, dict):
        for item in node.values():
            yield from _walk_pairs(item, depth)


def _span_list(node: object) -> list[int] | None:
    """Return ``[start, end]`` for a well-formed span, otherwise ``None``."""
    span = getattr(node, "span", None)
    if not isinstance(span, (tuple, list)) or len(span) != 2:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in span):
        return None
    return [span[0], span[1]]


def _type_symbol(ty: ast.Type | None) -> str | None:
    """Return the declared name of a type used as an impl target."""
    if ty is None:
        return None
    if isinstance(ty, ast.TypeName):
        return ty.name
    return ast.type_display_name(ty)


def ast_walk(node: object) -> list[object]:
    """Return every AST node reachable from *node* in deterministic pre-order.

    *node* may be a single dataclass AST node or a list, tuple or dict
    containing nodes.  Source metadata fields (``line``, ``col``, ``span`` and
    ``source_lines``) and scalar field values are not visited, so the result
    contains only AST nodes and always includes *node* itself when it is one.
    """
    _validate_walk_input(node)
    return [child for child, _ in _walk_pairs(node)]


def ast_node_count(node: object) -> int:
    """Count the AST nodes reachable from *node*, including *node* itself."""
    _validate_walk_input(node)
    return sum(1 for _ in _walk_pairs(node))


def ast_depth(node: object) -> int:
    """Return the maximum AST nesting depth rooted at *node*.

    A single node has depth one; each nested node field increases the depth by
    one.  Containers (lists, tuples and dicts) do not count as tree levels.
    """
    _validate_walk_input(node)
    return max((depth for _, depth in _walk_pairs(node)), default=0)


def ast_find_by_type(node: object, type_name: str) -> list[object]:
    """Find nodes whose concrete class name is exactly *type_name*.

    The matches are returned in the same pre-order used by :func:`ast_walk`.
    An unknown class name simply produces an empty list.
    """
    _validate_walk_input(node)
    if not isinstance(type_name, str):
        raise _api_error("type_name must be a str")
    return [
        child for child, _ in _walk_pairs(node) if type(child).__name__ == type_name
    ]


def ast_find_by_predicate(
    node: object, predicate: Callable[[object], bool]
) -> list[object]:
    """Find nodes for which *predicate* returns a truthy value.

    The predicate is invoked once per node in pre-order, including *node*
    itself; matching nodes are returned in that same order.
    """
    _validate_walk_input(node)
    if not callable(predicate):
        raise _api_error("predicate must be callable")
    return [child for child, _ in _walk_pairs(node) if predicate(child)]


def ast_to_json(node: object) -> dict:
    """Serialize an AST node into a JSON-safe dictionary.

    The shape follows the framework convention: ``kind`` holds the class name,
    then each non-metadata field follows, and finally ``line``, ``col`` and an
    optional ``char_span`` derived from ``node.span``.
    """
    if not _is_node(node):
        raise _api_error(f"expected an AST node, got {type(node).__name__}")
    result = _framework_ast_to_dict(node)
    if not isinstance(result, dict):
        raise _api_error("AST node did not serialize to a dictionary")
    return result


def ast_node_kinds(node: object) -> dict[str, int]:
    """Count reachable AST nodes per concrete class name, sorted by name."""
    _validate_walk_input(node)
    counts: Counter[str] = Counter(
        type(child).__name__ for child, _ in _walk_pairs(node)
    )
    return dict(sorted(counts.items()))


def ast_collect_symbols(program: ast.Program) -> list[str]:
    """Collect declared names from *program* in deterministic order.

    Included are function names (trait and impl methods too), struct, enum and
    trait names, impl target type names, constant and static names, and the
    names bound by ``let`` statements inside function bodies.  Names are
    unique: a repeated declaration keeps the position of its first occurrence.
    """
    if not isinstance(program, ast.Program):
        raise _api_error(f"expected an ast.Program, got {type(program).__name__}")

    seen: set[str] = set()
    symbols: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            symbols.append(name)

    for node, _ in _walk_pairs(program):
        if isinstance(node, (ast.FnDecl, ast.StructDecl, ast.EnumDecl, ast.TraitDecl)):
            add(node.name)
        elif isinstance(node, ast.ImplBlock):
            add(_type_symbol(node.ty))
        elif isinstance(node, ast.ConstDecl):
            add(node.name)
        elif isinstance(node, ast.LetStmt):
            for bound, _ in _walk_pairs(node.pattern):
                if isinstance(bound, (ast.PatIdent, ast.PatBinding)):
                    add(bound.name)
    return symbols


def ast_collect_literals(program: ast.Program) -> list[dict]:
    """Collect literal nodes from *program* with their values and spans.

    Integer, float, boolean, string and character literals are reported in
    pre-order as dictionaries with ``kind``, ``value``, ``line``, ``col`` and
    ``char_span`` keys; ``char_span`` is ``None`` when the node has no valid
    span.
    """
    if not isinstance(program, ast.Program):
        raise _api_error(f"expected an ast.Program, got {type(program).__name__}")

    literals: list[dict] = []
    for node, _ in _walk_pairs(program):
        if isinstance(node, _LITERAL_NODES):
            literals.append(
                {
                    "kind": type(node).__name__,
                    "value": node.value,
                    "line": node.line,
                    "col": node.col,
                    "char_span": _span_list(node),
                }
            )
    return literals


def ast_node_at_offset(program: ast.Program, offset: int) -> object | None:
    """Find the narrowest AST node whose span covers the character *offset*.

    Spans are treated as half-open ``[start, end)`` intervals.  Among nodes
    covering *offset*, the one with the shortest span wins; equal spans are
    broken in favour of the deeper tree node and finally pre-order position.
    Nodes without a valid span are ignored, and ``None`` is returned when no
    node covers *offset*.
    """
    if not isinstance(program, ast.Program):
        raise _api_error(f"expected an ast.Program, got {type(program).__name__}")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise _api_error("offset must be an int")
    if offset < 0:
        raise _api_error("offset must be non-negative")

    best: object | None = None
    best_key: tuple[int, int, int] | None = None
    for index, (node, depth) in enumerate(_walk_pairs(program)):
        span = _span_list(node)
        if span is None or not span[0] <= offset < span[1]:
            continue
        key = (span[1] - span[0], -depth, index)
        if best_key is None or key < best_key:
            best_key = key
            best = node
    return best
