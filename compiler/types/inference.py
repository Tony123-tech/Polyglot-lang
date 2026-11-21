"""Deterministic type-inference helpers for the FRAME 2 rollout.

Types crossing this module's API are either :mod:`ast_nodes` type nodes or
canonical strings such as ``Vec<T>``, ``&'a mut i64`` and ``fn(i64) -> bool``.
Strings are parsed into a small structural representation, so unification,
substitution, generalization and alias expansion all operate on the same shape
and never depend on incidental whitespace in the caller's spelling.

Diagnostics follow the shared E001-E008 surface: E001 marks incompatible types,
E003 marks framework API misuse, E005 marks occurs-check cycles and E008 marks
malformed encoded type expressions.  Everything here is pure Python and does
not consult the native backend.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn

import ast_nodes as ast
from compiler.frontend.errors import PolyglotError

__all__ = [
    "infer_alias_expand",
    "infer_array_element",
    "infer_binop_result",
    "infer_call_signature",
    "infer_cast_type",
    "infer_coerce",
    "infer_constraint_solve",
    "infer_expr_type",
    "infer_expression_fingerprint",
    "infer_free_variables",
    "infer_function_signature",
    "infer_generalize",
    "infer_generic_arguments",
    "infer_instantiate",
    "infer_join_branches",
    "infer_let_binding",
    "infer_literal_type",
    "infer_method_receiver",
    "infer_occurs",
    "infer_pattern_bindings",
    "infer_reference_depth",
    "infer_scheme_arity",
    "infer_solve_program",
    "infer_substitute",
    "infer_tuple_elements",
    "infer_type_class_context",
    "infer_unary_type",
    "infer_unify",
    "infer_unknown_fresh",
    "infer_widen",
]

_BUILTIN_TYPES: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "char",
        "f32",
        "f64",
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
        "String",
        "str",
        "Vec",
        "Option",
        "Result",
        "Box",
        "HashMap",
        "HashSet",
        "Range",
        "Self",
        "dyn",
        "fn",
        "impl",
        "mut",
    }
)
_NUMERIC_RANK: Final[dict[str, int]] = {
    "bool": 0,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 3,
    "u32": 3,
    "f32": 4,
    "i64": 5,
    "u64": 5,
    "f64": 6,
    "i128": 7,
    "u128": 7,
    "isize": 5,
    "usize": 5,
}
_FLOAT_TYPES: Final[frozenset[str]] = frozenset({"f32", "f64"})
_CONTAINER_TYPES: Final[frozenset[str]] = frozenset(
    {"Vec", "Option", "Result", "Box", "HashMap", "HashSet"}
)
_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"[$?]?[A-Za-z_][A-Za-z0-9_]*")
_LIFETIME_RE: Final[re.Pattern[str]] = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*")
_SCHEME_RE: Final[re.Pattern[str]] = re.compile(
    r"^forall\s+([$?A-Za-z][$?A-Za-z0-9_]*(?:\s+[$?A-Za-z][$?A-Za-z0-9_]*)*)\s*\.\s*(.+)$",
    re.DOTALL,
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*(->|'[A-Za-z_][A-Za-z0-9_]*|[$?]?[A-Za-z_][A-Za-z0-9_]*|"
    r"[0-9][A-Za-z0-9_]*|[<>,&\[\];()])"
)
_ARITHMETIC_OPS: Final[frozenset[str]] = frozenset({"+", "-", "*", "/", "%"})
_COMPARISON_OPS: Final[frozenset[str]] = frozenset({"==", "!=", "<", ">", "<=", ">="})
_LOGICAL_OPS: Final[frozenset[str]] = frozenset({"&&", "||", "&", "|", "^"})
_LITERAL_SUFFIXES: Final[tuple[str, ...]] = (
    "i128",
    "u128",
    "isize",
    "usize",
    "i64",
    "u64",
    "i32",
    "u32",
    "i16",
    "u16",
    "i8",
    "u8",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured diagnostic attributed to the type-inference phase."""
    raise PolyglotError(code, message, phase="types")


def _require_mapping(value: object, label: str) -> Mapping[Any, Any]:
    """Validate that *value* is a mapping, reporting E003 otherwise."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _require_sequence(value: object, label: str) -> list[Any]:
    """Validate that *value* is a list/tuple, reporting E003 otherwise."""
    if not isinstance(value, (list, tuple)):
        _fail("E003", f"{label} must be a list or tuple, got {type(value).__name__}")
    return list(value)


def _tokenize(text: str) -> list[str]:
    """Split a canonical type expression into parser tokens."""
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            if text[position].isspace():
                position += 1
                continue
            _fail("E008", f"invalid character {text[position]!r} in type {text!r}")
        token = match.group(1)
        if token is None:
            break
        tokens.append(token)
        position = match.end()
    return tokens


class _TypeParser:
    """A tiny recursive-descent parser for canonical type strings."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        """Return the next token without consuming it."""
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def take(self, expected: str | None = None) -> str:
        """Consume one token, optionally requiring an exact spelling."""
        token = self.peek()
        if token is None:
            _fail("E008", "unexpected end of type expression")
        if expected is not None and token != expected:
            _fail("E008", f"expected {expected!r}, found {token!r}")
        self.index += 1
        return token

    def parse(self) -> tuple:
        """Parse the complete token stream into a structural type node."""
        node = self._parse_type()
        extra = self.peek()
        if extra is not None:
            _fail("E008", f"unexpected token {extra!r} in type expression")
        return node

    def _parse_type(self) -> tuple:
        token = self.peek()
        if token is None:
            _fail("E008", "expected a type expression")
        if token == "&":
            return self._parse_ref()
        if token == "fn":
            return self._parse_fn()
        if token == "(":
            return self._parse_tuple()
        if token == "[":
            return self._parse_sequence()
        if token.startswith("'"):
            _fail("E008", f"lifetime {token!r} must follow '&'")
        name = self.take()
        if not _IDENT_RE.fullmatch(name):
            _fail("E008", f"invalid type name {name!r}")
        args: list[tuple] = []
        if self.peek() == "<":
            self.take("<")
            if self.peek() == ">":
                _fail("E008", "empty generic argument list")
            while True:
                args.append(self._parse_type())
                following = self.peek()
                if following == ",":
                    self.take(",")
                    continue
                if following == ">":
                    self.take(">")
                    break
                _fail("E008", "expected ',' or '>' in generic arguments")
        return ("name", name, tuple(args))

    def _parse_ref(self) -> tuple:
        self.take("&")
        lifetime: str | None = None
        token = self.peek()
        if token is not None and token.startswith("'"):
            if not _LIFETIME_RE.fullmatch(token):
                _fail("E008", f"invalid lifetime {token!r}")
            lifetime = token
            self.take()
        mutable = False
        if self.peek() == "mut":
            self.take()
            mutable = True
        return ("ref", lifetime, mutable, self._parse_type())

    def _parse_fn(self) -> tuple:
        self.take("fn")
        self.take("(")
        params: list[tuple] = []
        if self.peek() != ")":
            while True:
                params.append(self._parse_type())
                if self.peek() == ",":
                    self.take(",")
                    continue
                break
        self.take(")")
        result: tuple | None = None
        if self.peek() == "->":
            self.take("->")
            result = self._parse_type()
        return ("fn", tuple(params), result)

    def _parse_tuple(self) -> tuple:
        self.take("(")
        if self.peek() == ")":
            self.take(")")
            return ("tuple", ())
        items: list[tuple] = []
        trailing = False
        while True:
            items.append(self._parse_type())
            if self.peek() == ",":
                self.take(",")
                trailing = True
                if self.peek() == ")":
                    break
                continue
            trailing = False
            break
        self.take(")")
        if len(items) == 1 and not trailing:
            return items[0]
        return ("tuple", tuple(items))

    def _parse_sequence(self) -> tuple:
        self.take("[")
        inner = self._parse_type()
        if self.peek() == ";":
            self.take(";")
            size = self.take()
            if not (_IDENT_RE.fullmatch(size) or size.isdigit()):
                _fail("E008", f"invalid array size {size!r}")
            self.take("]")
            return ("array", inner, size)
        self.take("]")
        return ("slice", inner)


def _node(text: str) -> tuple:
    """Parse a canonical type string into a structural node."""
    return _TypeParser(_tokenize(text)).parse()


def _format(node: tuple) -> str:
    """Render a structural type node back to canonical source-like spelling."""
    tag = node[0]
    if tag == "name":
        if not node[2]:
            return node[1]
        return f"{node[1]}<{', '.join(_format(arg) for arg in node[2])}>"
    if tag == "ref":
        prefix = f"{node[1]} " if node[1] else ""
        mutability = "mut " if node[2] else ""
        return f"&{prefix}{mutability}{_format(node[3])}"
    if tag == "slice":
        return f"[{_format(node[1])}]"
    if tag == "array":
        return f"[{_format(node[1])}; {node[2]}]"
    if tag == "tuple":
        if not node[1]:
            return "()"
        if len(node[1]) == 1:
            return f"({_format(node[1][0])},)"
        return f"({', '.join(_format(item) for item in node[1])})"
    if tag == "fn":
        params = ", ".join(_format(param) for param in node[1])
        result = _format(node[2]) if node[2] is not None else "()"
        return f"fn({params}) -> {result}"
    _fail("E008", f"unsupported type node {tag!r}")


def _canonical(value: object) -> str:
    """Normalize an AST type node or type string to canonical spelling."""
    if isinstance(value, ast.Type):
        text = ast.type_display_name(value)
    elif isinstance(value, str):
        text = value
    else:
        _fail("E003", f"expected a type expression, got {type(value).__name__}")
    if not text.strip():
        _fail("E003", "type expression must not be empty")
    return _format(_node(text))


def _map_type(node: tuple, transform: Any) -> tuple:
    """Rebuild *node*, applying *transform* to each visited node."""
    tag = node[0]
    if tag == "name":
        rebuilt = ("name", node[1], tuple(_map_type(arg, transform) for arg in node[2]))
    elif tag == "ref":
        rebuilt = ("ref", node[1], node[2], _map_type(node[3], transform))
    elif tag == "slice":
        rebuilt = ("slice", _map_type(node[1], transform))
    elif tag == "array":
        rebuilt = ("array", _map_type(node[1], transform), node[2])
    elif tag == "tuple":
        rebuilt = ("tuple", tuple(_map_type(item, transform) for item in node[1]))
    elif tag == "fn":
        params = tuple(_map_type(param, transform) for param in node[1])
        result = _map_type(node[2], transform) if node[2] is not None else None
        rebuilt = ("fn", params, result)
    else:
        rebuilt = node
    return transform(rebuilt)


def _is_var(node: tuple) -> bool:
    """Return whether *node* is a bare inference variable name."""
    return node[0] == "name" and not node[2] and node[1] not in _BUILTIN_TYPES


def _vars(node: tuple) -> set[str]:
    """Collect every free inference-variable name in a structural type."""
    found: set[str] = set()

    def visit(current: tuple) -> tuple:
        if _is_var(current):
            found.add(current[1])
        return current

    _map_type(node, visit)
    return found


def _resolve(
    node: tuple, subst: Mapping[str, tuple], seen: frozenset[str] = frozenset()
) -> tuple:
    """Follow substitution chains without looping on a cyclic binding."""
    if _is_var(node) and node[1] in subst:
        if node[1] in seen:
            return node
        return _resolve(subst[node[1]], subst, seen | {node[1]})
    tag = node[0]
    if tag == "name":
        return ("name", node[1], tuple(_resolve(arg, subst, seen) for arg in node[2]))
    if tag == "ref":
        return ("ref", node[1], node[2], _resolve(node[3], subst, seen))
    if tag == "slice":
        return ("slice", _resolve(node[1], subst, seen))
    if tag == "array":
        return ("array", _resolve(node[1], subst, seen), node[2])
    if tag == "tuple":
        return ("tuple", tuple(_resolve(item, subst, seen) for item in node[1]))
    if tag == "fn":
        params = tuple(_resolve(param, subst, seen) for param in node[1])
        result = _resolve(node[2], subst, seen) if node[2] is not None else None
        return ("fn", params, result)
    return node


def _occurrences(
    name: str, node: tuple, subst: Mapping[str, tuple] | None = None
) -> int:
    """Count occurrences of an inference variable, resolving bindings first."""
    current = _resolve(node, subst) if subst is not None else node
    if _is_var(current) and current[1] == name:
        return 1
    if current[0] == "name":
        return sum(_occurrences(name, arg, None) for arg in current[2])
    if current[0] == "ref":
        return _occurrences(name, current[3], None)
    if current[0] in {"slice", "array"}:
        return _occurrences(name, current[1], None)
    if current[0] == "tuple":
        return sum(_occurrences(name, item, None) for item in current[1])
    if current[0] == "fn":
        total = sum(_occurrences(name, param, None) for param in current[1])
        if current[2] is not None:
            total += _occurrences(name, current[2], None)
        return total
    return 0


def _compiled(mapping: Mapping[Any, Any]) -> dict[str, tuple]:
    """Validate and parse a substitution mapping into structural nodes."""
    result: dict[str, tuple] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            _fail("E003", "substitution keys must be non-empty strings")
        result[key] = _node(_canonical(value))
    return result


def _substitute_nodes(
    node: tuple, replacements: Mapping[str, tuple], seen: frozenset[str] = frozenset()
) -> tuple:
    """Apply an already-compiled structural substitution with cycle protection."""

    def transform(current: tuple) -> tuple:
        if _is_var(current) and current[1] in replacements and current[1] not in seen:
            return _substitute_nodes(
                replacements[current[1]], replacements, seen | {current[1]}
            )
        return current

    return _map_type(node, transform)


def _substitute(node: tuple, mapping: Mapping[Any, Any]) -> tuple:
    """Apply a variable-to-type substitution repeatedly with cycle protection."""
    return _substitute_nodes(node, _compiled(mapping))


def _unify_nodes(
    left: tuple,
    right: tuple,
    subst: dict[str, tuple],
    variables: set[str],
    trail: list[str],
) -> None:
    """Unify two structural types, accumulating bindings in *subst*."""
    left = _resolve(left, subst)
    right = _resolve(right, subst)
    if left == right:
        return
    if _is_var(left) and left[1] in variables:
        if _occurrences(left[1], right, subst):
            _fail("E005", f"occurs check failed: {left[1]} occurs in {_format(right)}")
        subst[left[1]] = right
        trail.append(left[1])
        return
    if _is_var(right) and right[1] in variables:
        if _occurrences(right[1], left, subst):
            _fail("E005", f"occurs check failed: {right[1]} occurs in {_format(left)}")
        subst[right[1]] = left
        trail.append(right[1])
        return
    if left[0] != right[0]:
        _fail("E001", f"cannot unify {_format(left)} with {_format(right)}")
    tag = left[0]
    if tag == "name":
        if left[1] != right[1] or len(left[2]) != len(right[2]):
            _fail("E001", f"cannot unify {_format(left)} with {_format(right)}")
        for first, second in zip(left[2], right[2]):
            _unify_nodes(first, second, subst, variables, trail)
        return
    if tag == "ref":
        if left[2] != right[2]:
            _fail("E001", "reference mutability differs")
        _unify_nodes(left[3], right[3], subst, variables, trail)
        return
    if tag == "slice":
        _unify_nodes(left[1], right[1], subst, variables, trail)
        return
    if tag == "array":
        if left[2] != right[2]:
            _fail("E001", f"array sizes {left[2]} and {right[2]} differ")
        _unify_nodes(left[1], right[1], subst, variables, trail)
        return
    if tag == "tuple" and len(left[1]) == len(right[1]):
        for first, second in zip(left[1], right[1]):
            _unify_nodes(first, second, subst, variables, trail)
        return
    if (
        tag == "fn"
        and len(left[1]) == len(right[1])
        and (left[2] is None) == (right[2] is None)
    ):
        for first, second in zip(left[1], right[1]):
            _unify_nodes(first, second, subst, variables, trail)
        if left[2] is not None and right[2] is not None:
            _unify_nodes(left[2], right[2], subst, variables, trail)
        return
    _fail("E001", f"cannot unify {_format(left)} with {_format(right)}")


def _match_pattern(
    pattern: tuple,
    concrete: tuple,
    subst: dict[str, tuple],
    variables: set[str],
) -> None:
    """Match a type pattern against a concrete type, binding its variables."""
    if _is_var(pattern) and pattern[1] in variables:
        existing = subst.get(pattern[1])
        if existing is None:
            subst[pattern[1]] = concrete
        elif _resolve(existing, subst) != concrete:
            _fail("E001", f"conflicting bindings for {pattern[1]}")
        return
    if pattern[0] != concrete[0]:
        _fail("E001", f"pattern {_format(pattern)} does not match {_format(concrete)}")
    tag = pattern[0]
    if tag == "name":
        if pattern[1] != concrete[1] or len(pattern[2]) != len(concrete[2]):
            _fail(
                "E001", f"pattern {_format(pattern)} does not match {_format(concrete)}"
            )
        for first, second in zip(pattern[2], concrete[2]):
            _match_pattern(first, second, subst, variables)
        return
    if tag == "ref":
        if pattern[2] != concrete[2]:
            _fail("E001", "reference mutability differs")
        _match_pattern(pattern[3], concrete[3], subst, variables)
        return
    if tag == "slice":
        _match_pattern(pattern[1], concrete[1], subst, variables)
        return
    if tag == "array":
        if pattern[2] != concrete[2]:
            _fail("E001", f"array sizes {pattern[2]} and {concrete[2]} differ")
        _match_pattern(pattern[1], concrete[1], subst, variables)
        return
    if tag == "tuple" and len(pattern[1]) == len(concrete[1]):
        for first, second in zip(pattern[1], concrete[1]):
            _match_pattern(first, second, subst, variables)
        return
    if (
        tag == "fn"
        and len(pattern[1]) == len(concrete[1])
        and (pattern[2] is None) == (concrete[2] is None)
    ):
        for first, second in zip(pattern[1], concrete[1]):
            _match_pattern(first, second, subst, variables)
        if pattern[2] is not None and concrete[2] is not None:
            _match_pattern(pattern[2], concrete[2], subst, variables)
        return
    _fail("E001", f"pattern {_format(pattern)} does not match {_format(concrete)}")


def _rank(node: tuple) -> int | None:
    """Return the numeric widening rank of a type, or ``None`` if non-numeric."""
    if node[0] == "name" and not node[2]:
        return _NUMERIC_RANK.get(node[1])
    return None


def _widened(left: tuple, right: tuple) -> tuple | None:
    """Return the common numeric type of two types, or ``None`` when invalid."""
    left_rank, right_rank = _rank(left), _rank(right)
    if left_rank is None or right_rank is None:
        return None
    if left == right:
        return left
    if left[1] in _FLOAT_TYPES or right[1] in _FLOAT_TYPES:
        return _node("f64")
    winner = left if left_rank >= right_rank else right
    return winner


def _iter_items(items: Sequence[ast.Item]) -> list[ast.Item]:
    """Flatten inline modules into a deterministic list of items."""
    flattened: list[ast.Item] = []
    for item in items:
        flattened.append(item)
        if isinstance(item, ast.ModDecl) and item.items is not None:
            flattened.extend(_iter_items(item.items))
    return flattened


def _pattern_names(pattern: ast.Pattern) -> list[str]:
    """Collect bound identifier names from a binding pattern in order."""
    if isinstance(pattern, ast.PatIdent):
        return [pattern.name]
    if isinstance(pattern, ast.PatBinding):
        return [pattern.name, *_pattern_names(pattern.inner)]
    if isinstance(pattern, ast.PatRef):
        return _pattern_names(pattern.inner)
    if isinstance(pattern, ast.PatTuple):
        return [name for item in pattern.elems for name in _pattern_names(item)]
    if isinstance(pattern, ast.PatOr) and pattern.alts:
        return _pattern_names(pattern.alts[0])
    return []


def _pattern_type(pattern: ast.Pattern, ty: tuple, index: int = 0) -> dict[str, str]:
    """Bind pattern names to structural component types."""
    if isinstance(pattern, ast.PatIdent):
        return {pattern.name: _format(ty)}
    if isinstance(pattern, ast.PatRef):
        inner = ty[3] if ty[0] == "ref" else ty
        return _pattern_type(pattern.inner, inner, index)
    if isinstance(pattern, ast.PatBinding):
        return {pattern.name: _format(ty), **_pattern_type(pattern.inner, ty, index)}
    if (
        isinstance(pattern, ast.PatTuple)
        and ty[0] == "tuple"
        and len(ty[1]) == len(pattern.elems)
    ):
        bindings: dict[str, str] = {}
        for position, (subpattern, component) in enumerate(zip(pattern.elems, ty[1])):
            bindings.update(_pattern_type(subpattern, component, position))
        return bindings
    return {}


def _parse_scheme(scheme: str) -> tuple[tuple[str, ...], tuple]:
    """Split a ``forall`` scheme into quantified names and its body type."""
    match = _SCHEME_RE.match(scheme.strip())
    if match is None:
        return (), _node(_canonical(scheme))
    variables = tuple(match.group(1).split())
    return variables, _node(match.group(2))


def _fresh(used: set[str], prefix: str) -> str:
    """Return the first deterministic fresh name not present in *used*."""
    index = 0
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _expr_type(
    expr: ast.Expr, env: Mapping[str, str], expected: str | None
) -> tuple[str | None, int, list[str]]:
    """Infer one expression's canonical type with a visited count and unknowns."""
    if not isinstance(expr, ast.Expr):
        _fail("E003", f"expected an AST expression, got {type(expr).__name__}")
    visited = 1
    unknown: list[str] = []
    scope = dict(env)

    def recurse(child: ast.Expr, child_expected: str | None = None) -> str | None:
        nonlocal visited
        result, count, names = _expr_type(child, scope, child_expected)
        visited += count
        unknown.extend(names)
        return result

    if isinstance(expr, ast.Paren):
        return recurse(expr.inner, expected), visited, unknown
    if isinstance(expr, ast.LitInt):
        return infer_literal_type(expr)["type"], visited, unknown
    if isinstance(expr, ast.LitFloat):
        return "f64", visited, unknown
    if isinstance(expr, ast.LitStr):
        return "String", visited, unknown
    if isinstance(expr, ast.LitChar):
        return "char", visited, unknown
    if isinstance(expr, ast.LitBool):
        return "bool", visited, unknown
    if isinstance(expr, ast.Ident):
        if expr.name in scope:
            return _canonical(scope[expr.name]), visited, unknown
        unknown.append(expr.name)
        return None, visited, unknown
    if isinstance(expr, ast.Path):
        head = expr.segments[0]
        if expr.generic_args:
            args = ", ".join(_canonical(arg) for arg in expr.generic_args)
            return f"{head}<{args}>", visited, unknown
        if head in scope:
            return _canonical(scope[head]), visited, unknown
        unknown.append(head)
        return None, visited, unknown
    if isinstance(expr, ast.Unary):
        operand = recurse(expr.operand)
        if operand is None:
            return None, visited, unknown
        return (
            infer_unary_type(expr.op, operand, lifetime=None, mutable=expr.mutable)[
                "type"
            ],
            visited,
            unknown,
        )
    if isinstance(expr, ast.Binary):
        left, right = recurse(expr.left), recurse(expr.right)
        if left is None or right is None:
            return None, visited, unknown
        return infer_binop_result(expr.op, left, right)["type"], visited, unknown
    if isinstance(expr, ast.Cast):
        return _canonical(expr.ty), visited, unknown
    if isinstance(expr, ast.TupleExpr):
        parts = [recurse(item) for item in expr.elems]
        if any(part is None for part in parts):
            return None, visited, unknown
        return (
            infer_tuple_elements([str(part) for part in parts])["tuple"],
            visited,
            unknown,
        )
    if isinstance(expr, (ast.ArrayExpr, ast.RepeatArray)):
        if isinstance(expr, ast.ArrayExpr):
            parts = [recurse(item) for item in expr.elems]
        else:
            parts = [recurse(expr.elem)]
        if expected is not None and _canonical(expected).startswith("Vec<"):
            return _canonical(expected), visited, unknown
        found = infer_array_element([part for part in parts if part is not None])
        return found["array"], visited, unknown
    if isinstance(expr, ast.StructLiteral):
        name = expr.path.split("::")[-1]
        args = [_canonical(arg) for arg in expr.generic_args]
        if not args and expected is not None:
            expected_node = _node(_canonical(expected))
            if expected_node[0] == "name" and expected_node[1] == name:
                args = [_format(arg) for arg in expected_node[2]]
        return (
            _format(("name", name, tuple(_node(arg) for arg in args))),
            visited,
            unknown,
        )
    if isinstance(expr, ast.OptionSome):
        inner = recurse(expr.value)
        return (f"Option<{inner}>" if inner else None), visited, unknown
    if isinstance(expr, ast.OptionNone):
        return "Option<?T0>", visited, unknown
    if isinstance(expr, ast.ResultOk):
        inner = recurse(expr.value)
        return (f"Result<{inner}, ?E0>" if inner else None), visited, unknown
    if isinstance(expr, ast.ResultErr):
        inner = recurse(expr.value)
        return (f"Result<?T0, {inner}>" if inner else None), visited, unknown
    if isinstance(expr, ast.MethodCall):
        receiver = recurse(expr.receiver)
        if receiver is None:
            return None, visited, unknown
        args = [recurse(arg) for arg in expr.args]
        if any(arg is None for arg in args):
            return None, visited, unknown
        return (
            infer_method_receiver(expr.method, receiver, [str(arg) for arg in args])[
                "type"
            ],
            visited,
            unknown,
        )
    if isinstance(expr, ast.Call):
        function = recurse(expr.func)
        args = [recurse(arg) for arg in expr.args]
        if function is None or any(arg is None for arg in args):
            return None, visited, unknown
        signature = _signature_from_value(function)
        return (
            infer_call_signature(signature, [str(arg) for arg in args])["return"],
            visited,
            unknown,
        )
    if isinstance(expr, ast.Block):
        for stmt in expr.stmts:
            if isinstance(stmt, ast.LetStmt):
                if stmt.value is not None:
                    annotation = _canonical(stmt.ty) if stmt.ty is not None else None
                    recurse(stmt.value, annotation)
                binding = infer_let_binding(stmt, scope)
                for name, binding_type in binding["bindings"].items():
                    scope[name] = binding_type
            elif isinstance(stmt, ast.ExprStmt):
                recurse(stmt.expr)
        return (recurse(expr.tail) if expr.tail is not None else "()"), visited, unknown
    if isinstance(expr, ast.IfExpr):
        recurse(expr.cond)
        then_type = recurse(expr.then_block)
        other = expr.else_branch
        if other is None:
            return then_type, visited, unknown
        if isinstance(other, ast.Block):
            else_type = recurse(other)
        else:
            else_type = recurse(other)
        if then_type is None or else_type is None:
            return then_type or else_type, visited, unknown
        return infer_join_branches([then_type, else_type])["type"], visited, unknown
    if isinstance(expr, ast.IfLet):
        value = recurse(expr.value)
        then_type = recurse(expr.then_block)
        if expr.else_branch is None:
            return "()" if value else None, visited, unknown
        if isinstance(expr.else_branch, ast.Block):
            else_type = recurse(expr.else_branch)
        else:
            else_type = recurse(expr.else_branch)
        if then_type and else_type:
            return infer_join_branches([then_type, else_type])["type"], visited, unknown
        return then_type or else_type, visited, unknown
    if isinstance(expr, ast.Range):
        endpoint = expr.start if expr.start is not None else expr.end
        if endpoint is None:
            return "Range<i64>", visited, unknown
        found = recurse(endpoint)
        return (f"Range<{found}>" if found else None), visited, unknown
    if isinstance(expr, ast.Await):
        return recurse(expr.expr, expected), visited, unknown
    if isinstance(expr, ast.Try):
        inner = recurse(expr.expr)
        if inner is None:
            return None, visited, unknown
        node = _node(inner)
        if node[0] == "name" and node[1] in {"Option", "Result"} and node[2]:
            return _format(node[2][0]), visited, unknown
        return inner, visited, unknown
    unknown.append(type(expr).__name__)
    return None, visited, unknown


def _signature_from_value(value: object) -> dict[str, Any]:
    """Normalize an AST function, signature dict or ``fn(...)`` string."""
    if isinstance(value, ast.FnDecl):
        return infer_function_signature(value)
    if isinstance(value, Mapping):
        if "params" not in value or "return" not in value:
            _fail("E003", "signature mapping requires 'params' and 'return' keys")
        params: list[dict[str, str]] = []
        for param in _require_sequence(value["params"], "signature params"):
            if isinstance(param, Mapping):
                params.append(
                    {
                        "name": str(param.get("name", "")),
                        "type": _canonical(param["type"]),
                    }
                )
            else:
                params.append({"name": "", "type": _canonical(param)})
        return {
            "name": str(value.get("name", "<anonymous>")),
            "params": params,
            "return": _canonical(value["return"]),
            "generics": list(_require_sequence(value.get("generics", []), "generics")),
            "arity": len(params),
            "is_async": bool(value.get("is_async", False)),
        }
    if isinstance(value, str):
        node = _node(_canonical(value))
        if node[0] != "fn":
            _fail("E003", "signature string must describe a function type")
        return {
            "name": "<anonymous>",
            "params": [{"name": "", "type": _format(param)} for param in node[1]],
            "return": _format(node[2]) if node[2] is not None else "()",
            "generics": sorted(_vars(node)),
            "arity": len(node[1]),
            "is_async": False,
        }
    _fail("E003", f"unsupported signature value {type(value).__name__}")


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------


def infer_literal_type(
    literal: object, *, hint: object | None = None
) -> dict[str, Any]:
    """Infer the canonical type of a Python or AST literal value.

    Integer suffixes such as ``42u8`` are honored; unsuffixed integers default
    to ``i64``.  An explicit *hint* overrides the default for numeric and string
    literals, and an incompatible hint raises ``E001``.
    """
    if isinstance(literal, bool):
        category, text, raw = "bool", "bool", str(literal)
    elif isinstance(literal, ast.LitBool):
        category, text, raw = "bool", "bool", str(literal.value)
    elif isinstance(literal, int):
        category, text, raw = "integer", "i64", str(literal)
    elif isinstance(literal, ast.LitInt):
        category, text, raw = "integer", "i64", literal.raw
    elif isinstance(literal, (float, ast.LitFloat)):
        category, text = "float", "f64"
        raw = literal.raw if isinstance(literal, ast.LitFloat) else str(literal)
    elif isinstance(literal, str):
        category, text, raw = "string", "String", literal
    elif isinstance(literal, ast.LitStr):
        category, text, raw = "string", "String", literal.value
    elif isinstance(literal, ast.LitChar):
        category, text, raw = "char", "char", literal.value
    else:
        _fail("E003", f"unsupported literal value {type(literal).__name__}")
    if category == "integer":
        lowered = raw.lower()
        for suffix in _LITERAL_SUFFIXES:
            if lowered.endswith(suffix):
                text = suffix
                break
    hint_text = _canonical(hint) if hint is not None else None
    if hint_text is not None:
        hint_node = _node(hint_text)
        if category in {"integer", "float"} and _rank(hint_node) is None:
            _fail("E001", f"{hint_text} is not a numeric literal hint")
        if category == "string" and hint_text not in {"String", "str"}:
            _fail("E001", f"{hint_text} is not a string literal hint")
        text = "str" if category == "string" and hint_text == "str" else hint_text
    confidence = (
        "default"
        if (
            category == "integer"
            and text == "i64"
            and not any(raw.lower().endswith(s) for s in _LITERAL_SUFFIXES)
        )
        else "exact"
    )
    return {
        "type": text,
        "category": category,
        "raw": raw,
        "hint": hint_text,
        "confidence": confidence,
    }


def infer_unary_type(
    op: str,
    operand: object,
    *,
    lifetime: str | None = None,
    mutable: bool = False,
) -> dict[str, Any]:
    """Infer the result of a unary operator applied to *operand*.

    Borrow operators build a reference type (including an optional lifetime),
    dereference strips one reference or slice layer, ``!`` requires ``bool``
    and negation requires a numeric operand.
    """
    if not isinstance(op, str) or not op:
        _fail("E003", "op must be a non-empty string")
    operand_text = _canonical(operand)
    operand_node = _node(operand_text)
    normalized = "&mut" if op in {"&mut", "& mut"} else op
    if normalized in {"&", "&mut"}:
        region = f"{lifetime} " if lifetime else ""
        mutability = "mut " if mutable or normalized == "&mut" else ""
        result = f"&{region}{mutability}{operand_text}"
        kind = "borrow"
    elif normalized == "!":
        if operand_text != "bool":
            _fail("E001", f"logical not requires bool, got {operand_text}")
        result, kind = "bool", "logical"
    elif normalized == "-":
        if _rank(operand_node) is None:
            _fail("E001", f"negation requires a numeric type, got {operand_text}")
        result, kind = operand_text, "negation"
    elif normalized == "*":
        if operand_node[0] not in {"ref", "slice"}:
            _fail("E001", f"cannot dereference {operand_text}")
        result, kind = (
            _format(
                operand_node[-1] if operand_node[0] == "slice" else operand_node[3]
            ),
            "deref",
        )
    else:
        _fail("E003", f"unsupported unary operator {op!r}")
    return {
        "type": result,
        "op": normalized,
        "operand": operand_text,
        "kind": kind,
        "lifetime": lifetime,
    }


def infer_binop_result(op: str, left: object, right: object) -> dict[str, Any]:
    """Infer a binary operator's result type using deterministic numeric widening.

    Comparisons and logical operators yield ``bool``; arithmetic and bitwise
    operators require numeric operands and use :func:`infer_widen` to select a
    common representation.
    """
    if not isinstance(op, str) or not op:
        _fail("E003", "op must be a non-empty string")
    left_text, right_text = _canonical(left), _canonical(right)
    left_node, right_node = _node(left_text), _node(right_text)
    if op in _COMPARISON_OPS:
        if left_text != right_text and _widened(left_node, right_node) is None:
            _fail("E001", f"cannot compare {left_text} with {right_text}")
        result, kind = "bool", "comparison"
    elif op in {"&&", "||"}:
        if left_text != "bool" or right_text != "bool":
            _fail("E001", f"{op} requires bool operands")
        result, kind = "bool", "logical"
    elif op in _ARITHMETIC_OPS | _LOGICAL_OPS:
        common = _widened(left_node, right_node)
        if common is None:
            _fail(
                "E001", f"{op} requires numeric operands, got {left_text}, {right_text}"
            )
        result, kind = (
            _format(common),
            "arithmetic" if op in _ARITHMETIC_OPS else "bitwise",
        )
    else:
        _fail("E003", f"unsupported binary operator {op!r}")
    return {
        "type": result,
        "op": op,
        "left": left_text,
        "right": right_text,
        "kind": kind,
        "coerced": left_text != right_text,
    }


def infer_cast_type(source: object, target: object) -> dict[str, Any]:
    """Validate a cast from *source* to *target* and describe its precision.

    Numeric and boolean/char casts are supported, as are reference-to-reference
    casts whose innermost types are compatible.  Unsupported casts raise E001.
    """
    source_text, target_text = _canonical(source), _canonical(target)
    source_node, target_node = _node(source_text), _node(target_text)
    if source_text == target_text:
        kind, lossless = "identity", True
    elif _rank(source_node) is not None and _rank(target_node) is not None:
        source_rank = _rank(source_node) or 0
        target_rank = _rank(target_node) or 0
        kind, lossless = "numeric", target_rank >= source_rank
    elif source_text in {"bool", "char"} and _rank(target_node) is not None:
        kind, lossless = "numeric", False
    elif source_node[0] == "ref" and target_node[0] == "ref":
        if source_node[2] != target_node[2]:
            _fail("E001", "cannot cast between shared and mutable references")
        inner = _widened(source_node[3], target_node[3])
        if inner is None and _format(source_node[3]) != _format(target_node[3]):
            _fail("E001", f"cannot cast {source_text} to {target_text}")
        kind, lossless = "reference", True
    else:
        _fail("E001", f"cannot cast {source_text} to {target_text}")
    return {
        "from": source_text,
        "to": target_text,
        "kind": kind,
        "lossless": lossless,
    }


def infer_expr_type(
    expr: ast.Expr,
    env: Mapping[str, object] | None = None,
    *,
    expected: object | None = None,
) -> dict[str, Any]:
    """Infer an AST expression's type against an optional name environment.

    The environment maps identifiers to canonical types.  Unknown names are
    listed instead of raising because a partial inference result is useful for
    editors and diagnostics; malformed API inputs still raise E003.
    """
    if env is not None:
        _require_mapping(env, "env")
    bindings = {str(name): _canonical(ty) for name, ty in (env or {}).items()}
    expected_text = _canonical(expected) if expected is not None else None
    found, visited, unknown = _expr_type(expr, bindings, expected_text)
    return {
        "type": found,
        "env": dict(sorted(bindings.items())),
        "visited": visited,
        "unresolved": sorted(set(unknown)),
        "expected": expected_text,
    }


def infer_pattern_bindings(pattern: ast.Pattern, ty: object) -> dict[str, Any]:
    """Destructure *pattern* against *ty* into a mapping of bound names."""
    if not isinstance(pattern, ast.Pattern):
        _fail("E003", f"expected an AST pattern, got {type(pattern).__name__}")
    ty_text = _canonical(ty)
    node = _node(ty_text)
    bindings = _pattern_type(pattern, node)
    if (
        isinstance(pattern, ast.PatTuple)
        and (node[0] != "tuple" or len(node[1]) != len(pattern.elems))
        and (node[0] != "name" or not _is_var(node))
    ):
        _fail("E001", f"pattern arity does not match {ty_text}")
    if isinstance(pattern, ast.PatOr) and len(pattern.alts) > 1:
        first = _pattern_type(pattern.alts[0], node)
        for alternative in pattern.alts[1:]:
            if _pattern_type(alternative, node) != first and not _is_var(node):
                _fail("E001", "alternative patterns bind different types")
    return {
        "bindings": dict(sorted(bindings.items())),
        "source": ty_text,
        "bound": len(bindings),
    }


def infer_let_binding(
    stmt: ast.LetStmt, env: Mapping[str, object] | None = None
) -> dict[str, Any]:
    """Infer a ``let`` statement's binding types and reconcile its annotation."""
    if not isinstance(stmt, ast.LetStmt):
        _fail("E003", f"expected an ast.LetStmt, got {type(stmt).__name__}")
    bindings_env = (
        {str(name): _canonical(ty) for name, ty in env.items()}
        if env is not None
        else {}
    )
    annotation = _canonical(stmt.ty) if stmt.ty is not None else None
    inferred = (
        infer_expr_type(stmt.value, bindings_env, expected=annotation)["type"]
        if stmt.value is not None
        else None
    )
    final = annotation or inferred
    coerced = False
    if annotation and inferred and annotation != inferred:
        common = _widened(_node(inferred), _node(annotation))
        if common is not None:
            final = _format(common)
            coerced = True
        else:
            _fail("E001", f"annotation {annotation} conflicts with inferred {inferred}")
    pattern_result = (
        infer_pattern_bindings(stmt.pattern, final) if final else {"bindings": {}}
    )
    names = _pattern_names(stmt.pattern)
    return {
        "name": names[0] if names else None,
        "type": final,
        "annotated": annotation,
        "inferred": inferred,
        "coerced": coerced,
        "bindings": pattern_result["bindings"],
        "code": None,
    }


def infer_function_signature(fn: ast.FnDecl) -> dict[str, Any]:
    """Extract a deterministic signature record from a function declaration."""
    if not isinstance(fn, ast.FnDecl):
        _fail("E003", f"expected an ast.FnDecl, got {type(fn).__name__}")
    params: list[dict[str, str]] = []
    for position, param in enumerate(fn.params):
        names = _pattern_names(param.pattern)
        param_type = _canonical(param.ty) if param.ty is not None else f"?T{position}"
        params.append(
            {"name": names[0] if names else f"arg{position}", "type": param_type}
        )
    return {
        "name": fn.name,
        "params": params,
        "return": _canonical(fn.ret) if fn.ret is not None else "()",
        "generics": [param.name for param in fn.generics],
        "arity": len(params),
        "is_async": bool(fn.is_async),
    }


def infer_call_signature(signature: object, args: Sequence[object]) -> dict[str, Any]:
    """Unify call arguments with a function signature and infer its result.

    *signature* may be an :class:`ast.FnDecl`, a signature dictionary produced
    by :func:`infer_function_signature`, or a canonical ``fn(...) -> R`` string.
    Wrong arity raises E001 and unsupported values raise E003.
    """
    arguments = [_canonical(arg) for arg in _require_sequence(args, "args")]
    record = _signature_from_value(signature)
    if len(arguments) != record["arity"]:
        _fail(
            "E001",
            f"{record['name']} expects {record['arity']} argument(s), got {len(arguments)}",
        )
    subst: dict[str, tuple] = {}
    trail: list[str] = []
    variables = {str(name) for name in record["generics"]}
    variables.update(
        name for param in record["params"] for name in _vars(_node(param["type"]))
    )
    for param, argument in zip(record["params"], arguments):
        _unify_nodes(_node(param["type"]), _node(argument), subst, variables, trail)
    result = _substitute_nodes(_node(record["return"]), subst)
    return {
        "name": record["name"],
        "return": _format(result),
        "substitution": dict(
            sorted((key, _format(value)) for key, value in subst.items())
        ),
        "matched": arguments,
        "arity": len(arguments),
    }


def infer_method_receiver(
    method: str, receiver: object, args: Sequence[object] = ()
) -> dict[str, Any]:
    """Infer a built-in method's result type from its receiver and arguments.

    Common container and string methods have fixed result types; ``clone``
    returns the receiver, ``unwrap`` extracts the first generic argument, and
    ``get`` wraps it in ``Option``.  An unknown method raises E003 while an
    unsupported receiver or wrong arity raises E001.
    """
    if not isinstance(method, str) or not method:
        _fail("E003", "method must be a non-empty string")
    receiver_text = _canonical(receiver)
    arguments = [_canonical(arg) for arg in _require_sequence(args, "args")]
    node = _node(receiver_text)
    arity = 1
    if method in {"len", "chars_count", "byte_len"}:
        result, arity = "usize", 0
    elif method == "is_empty":
        result, arity = "bool", 0
    elif method in {"contains", "starts_with", "ends_with"}:
        result = "bool"
    elif method in {"to_string", "trim", "to_uppercase", "to_lowercase"}:
        result, arity = "String", 0
    elif method == "clone":
        result, arity = receiver_text, 0
    elif method == "unwrap":
        if node[0] != "name" or not node[2]:
            _fail("E001", f"unwrap requires a generic receiver, got {receiver_text}")
        result, arity = _format(node[2][0]), 0
    elif method in {"push", "insert"}:
        result = "()"
    elif method == "get":
        element = _format(node[2][0]) if node[0] == "name" and node[2] else "?T0"
        result = f"Option<{element}>"
    elif method == "map":
        if node[0] == "name" and node[1] == "Result" and len(node[2]) > 1:
            result = f"Result<?U0, {_format(node[2][1])}>"
        else:
            result = "Option<?U0>"
    else:
        _fail("E003", f"unknown method {method!r}")
    if len(arguments) != arity:
        _fail("E001", f"{method} expects {arity} argument(s), got {len(arguments)}")
    return {
        "method": method,
        "receiver": receiver_text,
        "args": arguments,
        "arity": arity,
        "type": result,
    }


def infer_generic_arguments(
    params: Sequence[object],
    args: Sequence[object],
    declared: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Match type patterns (for example ``Vec<T>``) against concrete arguments."""
    patterns = [_node(_canonical(item)) for item in _require_sequence(params, "params")]
    concretes = [_node(_canonical(item)) for item in _require_sequence(args, "args")]
    if len(patterns) != len(concretes):
        _fail(
            "E001",
            f"expected {len(patterns)} generic argument(s), got {len(concretes)}",
        )
    variables = (
        {str(name) for name in declared}
        if declared is not None
        else {name for pattern in patterns for name in _vars(pattern)}
    )
    subst: dict[str, tuple] = {}
    for pattern, concrete in zip(patterns, concretes):
        _match_pattern(pattern, concrete, subst, variables)
    resolved = {key: _format(_resolve(value, subst)) for key, value in subst.items()}
    return {
        "substitution": dict(sorted(resolved.items())),
        "unbound": sorted(variables - set(subst)),
        "matched": len(patterns),
        "conflicts": [],
    }


def infer_unify(
    left: object, right: object, *, variables: Sequence[str] | None = None
) -> dict[str, Any]:
    """Unify two type expressions and return their most general common type."""
    left_node, right_node = _node(_canonical(left)), _node(_canonical(right))
    names = (
        {str(name) for name in variables}
        if variables is not None
        else _vars(left_node) | _vars(right_node)
    )
    subst: dict[str, tuple] = {}
    trail: list[str] = []
    _unify_nodes(left_node, right_node, subst, names, trail)
    return {
        "type": _format(_resolve(left_node, subst)),
        "substitution": dict(
            sorted((key, _format(value)) for key, value in subst.items())
        ),
        "steps": len(trail),
        "variables": sorted(names),
    }


def infer_substitute(ty: object, mapping: Mapping[str, object]) -> dict[str, Any]:
    """Apply a variable substitution to a type expression without mutating input."""
    canonical = _canonical(ty)
    _require_mapping(mapping, "mapping")
    replacements = dict(
        sorted((str(key), _canonical(value)) for key, value in mapping.items())
    )
    result = _substitute(_node(canonical), replacements)
    return {
        "type": _format(result),
        "replacements": [f"{key} -> {value}" for key, value in replacements.items()],
        "unchanged": _format(result) == canonical,
        "mapping": replacements,
    }


def infer_free_variables(ty: object) -> dict[str, Any]:
    """List the free inference variables of a type expression in sorted order."""
    canonical = _canonical(ty)
    names = sorted(_vars(_node(canonical)))
    return {"variables": names, "count": len(names), "type": canonical}


def infer_occurs(variable: str, ty: object) -> dict[str, Any]:
    """Report whether *variable* occurs inside *ty* and how deeply."""
    if not isinstance(variable, str) or not _IDENT_RE.fullmatch(variable):
        _fail("E003", "variable must be a valid identifier")
    canonical = _canonical(ty)
    node = _node(canonical)

    def depth(current: tuple, level: int) -> int:
        if _is_var(current) and current[1] == variable:
            return level
        children: list[tuple] = []
        if current[0] == "name":
            children = list(current[2])
        elif current[0] == "ref":
            children = [current[3]]
        elif current[0] in {"slice", "array"}:
            children = [current[1]]
        elif current[0] == "tuple":
            children = list(current[1])
        elif current[0] == "fn":
            children = list(current[1]) + ([current[2]] if current[2] else [])
        return max((depth(child, level + 1) for child in children), default=-1)

    count = _occurrences(variable, node)
    return {
        "occurs": count > 0,
        "variable": variable,
        "type": canonical,
        "positions": count,
        "depth": depth(node, 0),
    }


def infer_generalize(ty: object, *, env_types: Sequence[object] = ()) -> dict[str, Any]:
    """Quantify variables that are not free in the surrounding environment."""
    canonical = _canonical(ty)
    variables = _vars(_node(canonical))
    for item in _require_sequence(env_types, "env_types"):
        variables -= _vars(_node(_canonical(item)))
    quantified = sorted(variables)
    scheme = f"forall {' '.join(quantified)}. {canonical}" if quantified else canonical
    return {
        "scheme": scheme,
        "quantified": quantified,
        "mono": canonical,
        "env": sorted(_canonical(item) for item in env_types),
    }


def infer_instantiate(scheme: str, *, prefix: str = "?T") -> dict[str, Any]:
    """Replace a scheme's quantified variables with deterministic fresh names."""
    if not isinstance(scheme, str) or not isinstance(prefix, str) or not prefix:
        _fail("E003", "scheme and prefix must be non-empty strings")
    quantified, body = _parse_scheme(scheme)
    used = _vars(body) | set(quantified)
    renamed: dict[str, str] = {}
    for name in quantified:
        fresh = _fresh(used, prefix)
        used.add(fresh)
        renamed[name] = fresh
    result = _substitute(body, renamed)
    return {
        "type": _format(result),
        "renamed": dict(sorted(renamed.items())),
        "scheme": scheme,
        "quantified": sorted(quantified),
    }


def infer_widen(left: object, right: object) -> dict[str, Any]:
    """Return the wider of two numeric types, or raise E001 when incompatible."""
    left_text, right_text = _canonical(left), _canonical(right)
    common = _widened(_node(left_text), _node(right_text))
    if common is None:
        _fail("E001", f"cannot widen {left_text} and {right_text}")
    return {
        "type": _format(common),
        "left": left_text,
        "right": right_text,
        "rank": _rank(common) or 0,
    }


def infer_join_branches(branches: Sequence[object | None]) -> dict[str, Any]:
    """Join control-flow branch types into one common type or list conflicts."""
    items = _require_sequence(branches, "branches")
    texts = [_canonical(item) if item is not None else "()" for item in items]
    conflicts: list[dict[str, str]] = []
    common: tuple | None = None
    for text in texts:
        if common is None:
            common = _node(text)
            continue
        widened = _widened(common, _node(text))
        if widened is not None:
            common = widened
        elif _format(common) != text:
            conflicts.append({"code": "E001", "left": _format(common), "right": text})
    return {
        "type": _format(common) if common is not None else "()",
        "unanimous": not conflicts,
        "inputs": texts,
        "conflicts": conflicts,
    }


def infer_coerce(source: object, target: object) -> dict[str, Any]:
    """Describe whether *source* coerces to *target* without changing its value."""
    source_text, target_text = _canonical(source), _canonical(target)
    source_node, target_node = _node(source_text), _node(target_text)
    if source_text == target_text:
        kind, code = "identity", ""
    elif _widened(source_node, target_node) is not None:
        kind, code = "widening", ""
    elif target_node[0] == "ref" and _format(target_node[3]) == source_text:
        kind, code = "borrow", ""
    elif source_node[0] == "name" and target_node[0] in {"impl", "dyn"}:
        kind, code = "trait_upcast", ""
    else:
        kind, code = "none", "E001"
    return {
        "coercible": code == "",
        "kind": kind,
        "from": source_text,
        "to": target_text,
        "code": code,
    }


def infer_constraint_solve(
    constraints: Sequence[object], *, max_rounds: int = 16
) -> dict[str, Any]:
    """Solve a list of ``(variable, type)`` obligations deterministically."""
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 1
    ):
        _fail("E003", "max_rounds must be a positive integer")
    pairs: list[tuple[str, str]] = []
    for raw in _require_sequence(constraints, "constraints"):
        if isinstance(raw, Mapping):
            if "left" not in raw or "right" not in raw:
                _fail("E003", "constraint mapping requires left and right keys")
            left, right = raw["left"], raw["right"]
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            left, right = raw
        else:
            _fail("E003", "constraint must be a pair or a mapping")
        pairs.append((_canonical(left), _canonical(right)))
    subst: dict[str, tuple] = {}
    issues: list[dict[str, str]] = []
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        changed = False
        for left_text, right_text in pairs:
            left = _resolve(_node(left_text), subst)
            right = _resolve(_node(right_text), subst)
            if left == right:
                continue
            if _is_var(left) and _occurrences(left[1], right, subst) == 0:
                subst[left[1]] = right
                changed = True
            elif _is_var(right) and _occurrences(right[1], left, subst) == 0:
                subst[right[1]] = left
                changed = True
            elif _is_var(left) or _is_var(right):
                variable = left[1] if _is_var(left) else right[1]
                issues.append(
                    {
                        "code": "E005",
                        "variable": variable,
                        "message": f"cyclic obligation {left_text} = {right_text}",
                    }
                )
            else:
                issues.append(
                    {
                        "code": "E001",
                        "left": left_text,
                        "right": right_text,
                        "message": f"cannot satisfy {left_text} = {right_text}",
                    }
                )
        if not changed:
            break
    else:
        issues.append({"code": "E005", "message": "constraint solver did not converge"})
    resolved = {left: _format(_resolve(_node(left), subst)) for left, _ in pairs}
    return {
        "substitution": dict(
            sorted((key, _format(value)) for key, value in subst.items())
        ),
        "resolved": dict(sorted(resolved.items())),
        "rounds": rounds,
        "issues": issues,
        "solved": not issues,
    }


def infer_type_class_context(
    ty: object,
    traits: Sequence[str],
    impls: Mapping[str, Sequence[object]],
) -> dict[str, Any]:
    """Determine which trait obligations a type can satisfy from known impls."""
    canonical = _canonical(ty)
    required = sorted({str(name) for name in _require_sequence(traits, "traits")})
    _require_mapping(impls, "impls")
    known: dict[str, list[tuple]] = {
        str(trait): [
            _node(_canonical(target))
            for target in _require_sequence(targets, "impls targets")
        ]
        for trait, targets in impls.items()
    }

    def satisfies(text: str, trait: str) -> bool:
        node = _node(text)
        if (
            node[0] == "name"
            and node[1] in _CONTAINER_TYPES
            and trait
            in {
                "Clone",
                "Copy",
                "Debug",
                "PartialEq",
                "Eq",
                "Hash",
                "Send",
                "Sync",
            }
        ):
            return bool(node[2]) and all(
                satisfies(_format(arg), trait) for arg in node[2]
            )
        for target in known.get(trait, []):
            subst: dict[str, tuple] = {}
            try:
                _match_pattern(target, node, subst, _vars(target))
            except PolyglotError:
                continue
            return True
        return False

    satisfied = [trait for trait in required if satisfies(canonical, trait)]
    missing = [trait for trait in required if trait not in satisfied]
    context = f"{canonical}: {' + '.join(required)}" if required else ""
    return {
        "type": canonical,
        "required": required,
        "satisfied": satisfied,
        "missing": missing,
        "context": context,
    }


def infer_array_element(elements: Sequence[object]) -> dict[str, Any]:
    """Unify array or vector element types into one element type."""
    texts = [_canonical(item) for item in _require_sequence(elements, "elements")]
    conflicts: list[dict[str, str]] = []
    element: tuple | None = None
    for index, text in enumerate(texts):
        node = _node(text)
        if element is None:
            element = node
            continue
        common = _widened(element, node)
        if common is not None:
            element = common
        elif _format(element) != text:
            conflicts.append(
                {
                    "code": "E001",
                    "index": str(index),
                    "found": text,
                    "expected": _format(element),
                }
            )
    chosen = _format(element) if element is not None else "?T0"
    return {
        "element": chosen,
        "array": _format(("slice", _node(chosen))),
        "inputs": texts,
        "conflicts": conflicts,
    }


def infer_tuple_elements(elements: Sequence[object]) -> dict[str, Any]:
    """Build a canonical tuple type from its element types."""
    texts = [_canonical(item) for item in _require_sequence(elements, "elements")]
    node = ("tuple", tuple(_node(text) for text in texts))
    return {
        "tuple": _format(node),
        "elements": texts,
        "arity": len(texts),
        "unit": not texts,
    }


def infer_reference_depth(ty: object) -> dict[str, Any]:
    """Count nested reference layers and collect their lifetimes."""
    canonical = _canonical(ty)
    node = _node(canonical)
    depth = 0
    lifetimes: list[str] = []
    mutable_outer = False
    current = node
    while current[0] == "ref":
        depth += 1
        if depth == 1:
            mutable_outer = current[2]
        if current[1]:
            lifetimes.append(current[1])
        current = current[3]
    return {
        "depth": depth,
        "innermost": _format(current),
        "mutable_outer": mutable_outer,
        "lifetimes": sorted(lifetimes),
        "type": canonical,
    }


def infer_alias_expand(ty: object, aliases: Mapping[str, object]) -> dict[str, Any]:
    """Expand named type aliases with cycle detection (E005 on a cycle)."""
    canonical = _canonical(ty)
    _require_mapping(aliases, "aliases")
    replacements = {str(name): _canonical(value) for name, value in aliases.items()}
    current = canonical
    expanded: list[str] = []
    active: set[str] = set()
    while True:
        node = _node(current)
        if not _is_var(node) or node[1] not in replacements:
            break
        name = node[1]
        if name in active:
            _fail("E005", f"type alias cycle through {name!r}")
        active.add(name)
        current = replacements[name]
        expanded.append(f"{name} -> {current}")
    return {
        "type": current,
        "expanded": expanded,
        "steps": len(expanded),
        "aliases": sorted(replacements),
    }


def infer_scheme_arity(scheme: str) -> dict[str, Any]:
    """Describe a scheme's quantified variables, parameter count and result."""
    if not isinstance(scheme, str):
        _fail("E003", "scheme must be a string")
    quantified, body = _parse_scheme(scheme)
    is_function = body[0] == "fn"
    return {
        "quantified": sorted(quantified),
        "params": len(body[1]) if is_function else 0,
        "return": _format(body[2])
        if is_function and body[2] is not None
        else _format(body),
        "is_function": is_function,
        "scheme": scheme,
    }


def infer_unknown_fresh(
    existing: Sequence[str], *, prefix: str = "?T"
) -> dict[str, Any]:
    """Allocate a fresh inference variable name not already in use."""
    used = [str(item) for item in _require_sequence(existing, "existing")]
    if not isinstance(prefix, str) or not prefix:
        _fail("E003", "prefix must be a non-empty string")
    name = _fresh(set(used), prefix)
    return {
        "name": name,
        "used": sorted(used),
        "prefix": prefix,
        "index": int(name[len(prefix) :]),
    }


def infer_expression_fingerprint(
    expr: ast.Expr, env: Mapping[str, object] | None = None
) -> dict[str, Any]:
    """Hash an expression's inferred type and unresolved names deterministically."""
    result = infer_expr_type(expr, env)
    payload = f"{result['type']}|{','.join(result['unresolved'])}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "sha256": digest,
        "type": result["type"],
        "node_count": result["visited"],
        "algorithm": "sha256",
    }


def infer_solve_program(program: ast.Program) -> dict[str, Any]:
    """Infer signatures, parameter bindings and let bindings for a whole program."""
    if not isinstance(program, ast.Program):
        _fail("E003", f"expected an ast.Program, got {type(program).__name__}")
    functions: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    for item in _iter_items(program.items):
        if not isinstance(item, ast.FnDecl):
            continue
        signature = infer_function_signature(item)
        env = {param["name"]: param["type"] for param in signature["params"]}
        bindings: dict[str, Any] = {}
        for stmt in item.body.stmts if item.body is not None else []:
            if not isinstance(stmt, ast.LetStmt):
                continue
            try:
                binding = infer_let_binding(stmt, env)
            except PolyglotError as error:
                issues.append(
                    {
                        "code": error.code,
                        "item": item.name,
                        "message": error.diagnostic.message,
                    }
                )
                continue
            bindings[binding["name"] or f"binding{len(bindings)}"] = binding
            if binding["type"]:
                env[binding["name"] or f"arg{len(env)}"] = binding["type"]
        try:
            body = (
                infer_expr_type(item.body, env)
                if item.body is not None
                else {"type": "()"}
            )
        except PolyglotError as error:
            body = {"type": None}
            issues.append(
                {
                    "code": error.code,
                    "item": item.name,
                    "message": error.diagnostic.message,
                }
            )
        functions[item.name] = {
            "signature": signature,
            "return": signature["return"],
            "bindings": bindings,
            "body_type": body["type"],
        }
    return {
        "functions": dict(sorted(functions.items())),
        "issues": issues,
        "checked": len(functions),
        "names": sorted(functions),
    }
