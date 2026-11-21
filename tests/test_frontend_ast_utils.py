"""Focused tests for the ten public AST utilities in ``compiler.frontend``."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

import ast_nodes as ast
from compiler.frontend import ast_utils
from compiler.frontend.errors import PolyglotError
from lexer import lex
from parser import parse

SOURCE = """fn main() -> i64 {
    let a = 1;
    let b = 2.5;
    let ok = true;
    let ch = 'x';
    let text = "hi";
    a
}
struct Point { x: i64 }
enum Color { Red, Green }
trait Shape { fn area(self) -> i64; }
impl Point { fn zero() -> Point { let p = Point { x: 0 }; p } }
const LIMIT: i64 = 7;
static FLAG: bool = false;
"""

EXPECTED_KINDS = {
    "Block": 2,
    "ConstDecl": 2,
    "EnumDecl": 1,
    "EnumVariant": 2,
    "FnDecl": 3,
    "FnParam": 1,
    "Ident": 2,
    "ImplBlock": 1,
    "LetStmt": 6,
    "LitBool": 2,
    "LitChar": 1,
    "LitFloat": 1,
    "LitInt": 3,
    "LitStr": 1,
    "PatIdent": 7,
    "Program": 1,
    "StructDecl": 1,
    "StructLiteral": 1,
    "TraitDecl": 1,
    "TraitItem": 1,
    "TypeName": 7,
}


@pytest.fixture(scope="module")
def program() -> ast.Program:
    """Parse a program containing every declaration form used by the tests."""
    return parse(lex(SOURCE))


def _expect_e003(function: Callable[..., object], *args: object) -> None:
    """Assert that a call fails with a structured E003 PolyglotError."""
    with pytest.raises(PolyglotError) as exc:
        function(*args)
    assert exc.value.code == "E003"


def test_ast_walk_preorder_containers_and_validation(program: ast.Program) -> None:
    nodes = ast_utils.ast_walk(program)
    assert len(nodes) == 47
    assert nodes[0] is program
    assert nodes[1] is program.items[0]
    assert all(isinstance(node, ast.Node) for node in nodes)
    assert [node.span for node in nodes if isinstance(node, ast.LitInt)] == [
        (31, 32),
        (258, 259),
        (288, 289),
    ]

    container = [program.items[1], {"nested": (program.items[2],)}]
    assert ast_utils.ast_walk(container) == [
        program.items[1],
        program.items[1].fields[0][1],
        program.items[2],
        *program.items[2].variants,
    ]
    hand_built = ast.Program(1, 1, [], {1: "fn main() {}"}, span=(0, 12))
    assert ast_utils.ast_walk(hand_built) == [hand_built]

    for bad in ("not a node", 42, [], {}, [1, 2]):
        _expect_e003(ast_utils.ast_walk, bad)


def test_ast_node_count_exact_and_validation(program: ast.Program) -> None:
    assert ast_utils.ast_node_count(program) == 47
    assert ast_utils.ast_node_count(program.items[0]) == 19
    assert ast_utils.ast_node_count(ast.LitInt(1, 1, 5, "5")) == 1

    for bad in (None, 3.5, ["x"], ()):
        _expect_e003(ast_utils.ast_node_count, bad)


def test_ast_depth_nesting_and_validation(program: ast.Program) -> None:
    assert ast_utils.ast_depth(program) == 7
    leaf = ast.LitInt(1, 1, 5, "5")
    block = ast.Block(1, 1, [ast.ExprStmt(1, 1, leaf)])
    assert ast_utils.ast_depth(block) == 3
    assert ast_utils.ast_depth([block]) == 3

    for bad in ("node", 7, {"k": "v"}):
        _expect_e003(ast_utils.ast_depth, bad)


def test_ast_find_by_type_exact_spans_and_argument_checks(
    program: ast.Program,
) -> None:
    ints = ast_utils.ast_find_by_type(program, "LitInt")
    assert [type(node).__name__ for node in ints] == ["LitInt"] * 3
    assert [node.value for node in ints] == [1, 0, 7]
    assert [node.span for node in ints] == [(31, 32), (258, 259), (288, 289)]
    assert ast_utils.ast_find_by_type(program, "NoSuchNode") == []

    _expect_e003(ast_utils.ast_find_by_type, program, 123)
    _expect_e003(ast_utils.ast_find_by_type, "not a node", "LitInt")


def test_ast_find_by_predicate_preorder_and_callable_check(
    program: ast.Program,
) -> None:
    shared_span = ast_utils.ast_find_by_predicate(
        program, lambda node: getattr(node, "span", None) == (31, 32)
    )
    assert [type(node).__name__ for node in shared_span] == ["LetStmt", "LitInt"]
    line_two = ast_utils.ast_find_by_predicate(program, lambda node: node.line == 2)
    assert [type(node).__name__ for node in line_two] == [
        "LetStmt",
        "PatIdent",
        "LitInt",
    ]
    assert (
        ast_utils.ast_find_by_predicate(
            program, lambda node: getattr(node, "raw", "") == "7"
        )[0].value
        == 7
    )

    _expect_e003(ast_utils.ast_find_by_predicate, program, "not callable")
    _expect_e003(ast_utils.ast_find_by_predicate, 5, bool)


def test_ast_to_json_conventions_and_validation(program: ast.Program) -> None:
    literal = ast.LitInt(2, 5, 7, "7", span=(10, 11))
    assert ast_utils.ast_to_json(literal) == {
        "kind": "LitInt",
        "value": 7,
        "raw": "7",
        "line": 2,
        "col": 5,
        "char_span": [10, 11],
    }
    encoded = ast_utils.ast_to_json(program)
    assert encoded["kind"] == "Program"
    assert encoded["char_span"] == [0, 316]
    assert encoded["items"][0]["kind"] == "FnDecl"
    assert "span" not in encoded and "source_lines" not in encoded
    assert json.loads(json.dumps(encoded))["kind"] == "Program"

    for bad in ([program], {"node": program}, "node"):
        _expect_e003(ast_utils.ast_to_json, bad)


def test_ast_node_kinds_exact_counts_and_validation(program: ast.Program) -> None:
    kinds = ast_utils.ast_node_kinds(program)
    assert kinds == EXPECTED_KINDS
    assert sum(kinds.values()) == 47
    assert list(kinds) == sorted(kinds)

    for bad in (42, "not a node", []):
        _expect_e003(ast_utils.ast_node_kinds, bad)


def test_ast_collect_symbols_declaration_order_and_validation(
    program: ast.Program,
) -> None:
    assert ast_utils.ast_collect_symbols(program) == [
        "main",
        "a",
        "b",
        "ok",
        "ch",
        "text",
        "Point",
        "Color",
        "Shape",
        "area",
        "zero",
        "p",
        "LIMIT",
        "FLAG",
    ]
    _expect_e003(ast_utils.ast_collect_symbols, program.items[0])
    _expect_e003(ast_utils.ast_collect_symbols, [program])


def test_ast_collect_literals_values_spans_and_validation(
    program: ast.Program,
) -> None:
    assert ast_utils.ast_collect_literals(program) == [
        {"kind": "LitInt", "value": 1, "line": 2, "col": 13, "char_span": [31, 32]},
        {
            "kind": "LitFloat",
            "value": 2.5,
            "line": 3,
            "col": 13,
            "char_span": [46, 49],
        },
        {
            "kind": "LitBool",
            "value": True,
            "line": 4,
            "col": 14,
            "char_span": [64, 68],
        },
        {
            "kind": "LitChar",
            "value": "x",
            "line": 5,
            "col": 14,
            "char_span": [83, 86],
        },
        {
            "kind": "LitStr",
            "value": "hi",
            "line": 6,
            "col": 16,
            "char_span": [103, 107],
        },
        {"kind": "LitInt", "value": 0, "line": 12, "col": 54, "char_span": [258, 259]},
        {"kind": "LitInt", "value": 7, "line": 13, "col": 20, "char_span": [288, 289]},
        {
            "kind": "LitBool",
            "value": False,
            "line": 14,
            "col": 21,
            "char_span": [311, 316],
        },
    ]
    _expect_e003(ast_utils.ast_collect_literals, "not a program")
    _expect_e003(ast_utils.ast_collect_literals, program.items[0])


def test_ast_node_at_offset_smallest_span_and_tie_break(
    program: ast.Program,
) -> None:
    literal = ast_utils.ast_node_at_offset(program, 31)
    assert isinstance(literal, ast.LitInt)
    assert (literal.value, literal.span) == (1, (31, 32))

    trait_method = ast_utils.ast_node_at_offset(program, 200)
    assert isinstance(trait_method, ast.FnDecl)
    assert trait_method.name == "area"
    assert ast_utils.ast_node_at_offset(program, 300) is program
    assert ast_utils.ast_node_at_offset(program, 500) is None

    leaf = ast.LitInt(1, 1, 9, "9", span=(0, 1))
    expression = ast.ExprStmt(1, 1, leaf, span=(0, 1))
    block = ast.Block(1, 1, [expression], span=(0, 1))
    function = ast.FnDecl(1, 1, "f", [], None, block, span=(0, 1))
    same_spans = ast.Program(1, 1, [function], span=(0, 1))
    assert ast_utils.ast_node_at_offset(same_spans, 0) is leaf

    no_span = ast.LitInt(1, 1, 4, "4")
    without_span = ast.Program(1, 1, [no_span], span=(0, 1))
    assert ast_utils.ast_node_at_offset(without_span, 0) is without_span

    _expect_e003(ast_utils.ast_node_at_offset, program, -1)
    _expect_e003(ast_utils.ast_node_at_offset, program, 2.5)
    _expect_e003(ast_utils.ast_node_at_offset, program, True)
    _expect_e003(ast_utils.ast_node_at_offset, program.items[0], 0)
