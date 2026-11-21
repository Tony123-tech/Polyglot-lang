"""Behavioral tests for the frontend parsing frame (35 public functions)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import ast_nodes as ast
from compiler.frontend import parsing
from compiler.frontend.errors import PolyglotError
from lexer import lex


def _expect_error(
    code: str, function: Callable[[str], Any], source: str
) -> PolyglotError:
    """Assert that *function* rejects *source* with diagnostic *code*."""
    with pytest.raises(PolyglotError) as caught:
        function(source)
    assert caught.value.code == code
    return caught.value


def test_parse_source_program_builds_real_program_and_reports_parse_errors():
    source = "fn main() -> i64 { 1 }"
    program = parsing.parse_source_program(source)

    assert isinstance(program, ast.Program)
    assert len(program.items) == 1
    assert isinstance(program.items[0], ast.FnDecl)
    assert program.items[0].name == "main"
    assert program.span == (0, len(source))
    assert lex(source)[0].value == "fn"
    assert program.source_lines[1] == source

    error = _expect_error("E003", parsing.parse_source_program, "fn f( {")
    assert "expected pattern" in error.reason
    assert (error.line, error.col, error.span) == (1, 7, (6, 7))


def test_parse_items_converts_each_real_item_to_a_dictionary():
    items = parsing.parse_items("fn a() {}\nstruct S {}")

    assert [item["kind"] for item in items] == ["FnDecl", "StructDecl"]
    assert [item["name"] for item in items] == ["a", "S"]
    assert [item["line"] for item in items] == [1, 2]
    assert items[0]["char_span"] == [0, 9]
    assert parsing.parse_items("") == []
    _expect_error("E003", parsing.parse_items, "fn f( {")


def test_parse_function_signature_handles_body_and_declaration_forms():
    declaration = parsing.parse_function_signature(
        "pub async fn f<T: Clone>(a: i64, b: &mut T) -> Vec<u8> { a }"
    )
    assert declaration == {
        "name": "f",
        "params": [
            {"name": "a", "type": "i64"},
            {"name": "b", "type": "&mut T"},
        ],
        "return_type": "Vec<u8>",
        "generic_params": ["T"],
        "is_async": True,
        "is_pub": True,
        "char_span": [0, 60],
    }
    assert parsing.parse_function_signature("fn g(x: i64) -> i64;") == {
        "name": "g",
        "params": [{"name": "x", "type": "i64"}],
        "return_type": "i64",
        "generic_params": [],
        "is_async": False,
        "is_pub": False,
        "char_span": [0, 20],
    }
    _expect_error("E001", parsing.parse_function_signature, "struct S {}")


def test_parse_struct_fields_reports_visibility_types_and_exact_spans():
    source = "struct S { pub a: i64, b: Vec<u8> }"
    fields = parsing.parse_struct_fields(source)

    assert fields == [
        {"name": "a", "type": "i64", "is_pub": True, "char_span": [11, 21]},
        {"name": "b", "type": "Vec<u8>", "is_pub": False, "char_span": [23, 33]},
    ]
    assert source[11:21] == "pub a: i64"
    assert source[23:33] == "b: Vec<u8>"
    _expect_error("E001", parsing.parse_struct_fields, "fn f() {}")


def test_parse_enum_variants_reports_tuple_named_and_unit_forms():
    variants = parsing.parse_enum_variants("enum E { A, B(i64, u8), C { x: bool } }")

    assert variants == [
        {"name": "A", "fields": None, "discriminant": None},
        {"name": "B", "fields": ["i64", "u8"], "discriminant": None},
        {"name": "C", "fields": ["bool"], "discriminant": None},
    ]
    _expect_error("E001", parsing.parse_enum_variants, "fn f() {}")


def test_parse_trait_methods_distinguishes_signatures_from_default_bodies():
    methods = parsing.parse_trait_methods(
        "trait Tr { fn a(&self) -> i64; fn b(&self) -> i64 { 42 } type X = i64; }"
    )

    assert methods == [
        {"name": "a", "signature": "fn a(&self) -> i64", "has_default": False},
        {"name": "b", "signature": "fn b(&self) -> i64", "has_default": True},
    ]
    _expect_error("E001", parsing.parse_trait_methods, "struct S {}")


def test_parse_impl_methods_keeps_declaration_order_and_body_flag():
    methods = parsing.parse_impl_methods(
        "impl S { pub fn new() -> S { S } fn go(&self, x: i64) {} }"
    )

    assert methods == [
        {"name": "new", "signature": "fn new() -> S", "has_body": True},
        {
            "name": "go",
            "signature": "fn go(&self, x: i64)",
            "has_body": True,
        },
    ]
    _expect_error("E001", parsing.parse_impl_methods, "enum E {}")


def test_parse_type_expr_classifies_named_generic_ref_and_slice_types():
    generic = parsing.parse_type_expr("Vec<u8>")
    assert generic == {
        "text": "Vec<u8>",
        "kind": "generic",
        "name": "Vec",
        "args": [
            {
                "text": "u8",
                "kind": "named",
                "name": "u8",
                "args": [],
                "char_span": [4, 6],
            }
        ],
        "char_span": [0, 7],
    }
    reference = parsing.parse_type_expr("&mut [i64]")
    assert reference["text"] == "&mut [i64]"
    assert reference["kind"] == "ref"
    assert reference["name"] == "&mut"
    assert reference["args"][0]["kind"] == "slice"
    assert reference["args"][0]["text"] == "[i64]"
    assert reference["char_span"] == [0, 10]
    assert parsing.parse_type_expr("fn(i64) -> u8")["kind"] == "fn"
    _expect_error("E003", parsing.parse_type_expr, "i64 + 1")


def test_parse_generic_args_splits_nested_angle_brackets_and_lifetimes():
    source = "<i64, Vec<u8>, 'a>"
    args = parsing.parse_generic_args(source)

    assert args == [
        {"text": "i64", "char_span": [1, 4]},
        {"text": "Vec<u8>", "char_span": [6, 13]},
        {"text": "'a", "char_span": [15, 17]},
    ]
    assert source[6:13] == "Vec<u8>"
    assert parsing.parse_generic_args("i64") == []
    _expect_error("E003", parsing.parse_generic_args, "<i64")


def test_parse_where_predicates_splits_bounds_and_preserves_spans():
    source = "where T: Clone, U: Debug + Send"
    predicates = parsing.parse_where_predicates(source)

    assert predicates == [
        {"name": "T", "bounds": ["Clone"], "char_span": [6, 14]},
        {"name": "U", "bounds": ["Debug", "Send"], "char_span": [16, 31]},
    ]
    assert source[16:31] == "U: Debug + Send"
    assert parsing.parse_where_predicates("fn f() {}") == []
    error = _expect_error("E006", parsing.parse_where_predicates, "where T Clone")
    assert error.col == 7


def test_parse_attributes_reads_real_attr_nodes_and_raw_argument_text():
    source = "#[derive(Debug, Clone)]\nfn f() {}"
    assert parsing.parse_attributes(source) == [
        {"name": "derive", "args": "Debug, Clone", "char_span": [0, 23]}
    ]
    assert source[0:23].endswith("]")
    assert parsing.parse_attributes("#[cfg(test)] fn f() {}") == [
        {"name": "cfg", "args": "test", "char_span": [0, 12]}
    ]
    assert parsing.parse_attributes("fn f() {}") == []
    _expect_error("E003", parsing.parse_attributes, "#[derive(")


def test_parse_visibility_inspects_the_leading_token_only():
    assert parsing.parse_visibility("pub fn f() {}") == "pub"
    assert parsing.parse_visibility("pub") == "pub"
    assert parsing.parse_visibility("fn f() {}") == "private"
    assert parsing.parse_visibility("async fn f() {}") == "private"
    _expect_error("E006", parsing.parse_visibility, "")


def test_parse_let_binding_returns_name_mutability_type_and_initializer():
    assert parsing.parse_let_binding("let mut x: i64 = 1 + 2;") == {
        "name": "x",
        "mutable": True,
        "type": "i64",
        "init": "1 + 2",
        "char_span": [0, 23],
    }
    assert parsing.parse_let_binding("let y = foo(a);") == {
        "name": "y",
        "mutable": False,
        "type": None,
        "init": "foo(a)",
        "char_span": [0, 15],
    }
    _expect_error("E003", parsing.parse_let_binding, "let mut = 1;")


def test_parse_match_arms_simple_accepts_full_match_and_bare_arm_lists():
    source = "match x { Some(y) if y > 0 => y, _ => 0 }"
    assert parsing.parse_match_arms_simple(source) == [
        {
            "pattern": "Some(y)",
            "guard": "y > 0",
            "body": "y",
            "char_span": [10, 31],
        },
        {"pattern": "_", "guard": None, "body": "0", "char_span": [33, 39]},
    ]
    assert parsing.parse_match_arms_simple("Some(y) if y > 0 => y, _ => 0") == [
        {
            "pattern": "Some(y)",
            "guard": "y > 0",
            "body": "y",
            "char_span": [0, 21],
        },
        {"pattern": "_", "guard": None, "body": "0", "char_span": [23, 29]},
    ]
    _expect_error("E003", parsing.parse_match_arms_simple, "nonsense")


def test_parse_if_chain_returns_every_condition_body_and_branch_span():
    source = "if a { 1 } else if b { 2 } else { 3 }"
    branches = parsing.parse_if_chain(source)

    assert branches == [
        {"condition": "a", "body": "{\n    1\n}", "char_span": [0, 10]},
        {"condition": "b", "body": "{\n    2\n}", "char_span": [16, 26]},
        {"condition": None, "body": "{\n    3\n}", "char_span": [32, 37]},
    ]
    assert source[16:26] == "if b { 2 }"
    _expect_error("E001", parsing.parse_if_chain, "foo()")


def test_parse_loop_body_covers_for_while_and_loop_headers():
    assert parsing.parse_loop_body("for i in 0..10 { foo(i); }") == {
        "kind": "for",
        "header": "for i in 0..10",
        "body": "{\n    foo(i);\n}",
        "char_span": [0, 26],
    }
    assert parsing.parse_loop_body("while x > 0 { x = x - 1; }") == {
        "kind": "while",
        "header": "while x > 0",
        "body": "{\n    x = x - 1;\n}",
        "char_span": [0, 26],
    }
    assert parsing.parse_loop_body("loop { break; }") == {
        "kind": "loop",
        "header": "loop",
        "body": "{\n    break;\n}",
        "char_span": [0, 15],
    }
    _expect_error("E001", parsing.parse_loop_body, "foo()")


def test_parse_call_expr_uses_pratt_operands_and_exact_span():
    source = "f(a, b)"
    assert parsing.parse_call_expr(source) == {
        "callee": "f",
        "args": ["a", "b"],
        "char_span": [0, 7],
    }
    assert source[0:7] == "f(a, b)"
    _expect_error("E001", parsing.parse_call_expr, "a + b")


def test_parse_path_expr_splits_real_path_segments():
    assert parsing.parse_path_expr("a::b::C") == {
        "segments": ["a", "b", "C"],
        "text": "a::b::C",
        "char_span": [0, 7],
    }
    assert parsing.parse_path_expr("solo") == {
        "segments": ["solo"],
        "text": "solo",
        "char_span": [0, 4],
    }
    _expect_error("E001", parsing.parse_path_expr, "1 + 2")


def test_parse_range_expr_distinguishes_inclusive_from_exclusive():
    assert parsing.parse_range_expr("a..=b") == {
        "start": "a",
        "end": "b",
        "inclusive": True,
        "char_span": [0, 5],
    }
    assert parsing.parse_range_expr("a..b") == {
        "start": "a",
        "end": "b",
        "inclusive": False,
        "char_span": [0, 4],
    }
    _expect_error("E001", parsing.parse_range_expr, "a + b")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "42",
            {"kind": "int", "raw": "42", "value": "42", "char_span": [0, 2]},
        ),
        (
            "1.5",
            {"kind": "float", "raw": "1.5", "value": "1.5", "char_span": [0, 3]},
        ),
        (
            "true",
            {"kind": "bool", "raw": "true", "value": "true", "char_span": [0, 4]},
        ),
        (
            '"hi"',
            {"kind": "string", "raw": '"hi"', "value": "hi", "char_span": [0, 4]},
        ),
        (
            "'a'",
            {"kind": "char", "raw": "'a'", "value": "a", "char_span": [0, 3]},
        ),
    ],
)
def test_parse_literal_expr_reports_kind_raw_lexeme_and_value(
    source: str, expected: dict
):
    assert parsing.parse_literal_expr(source) == expected
    _expect_error("E001", parsing.parse_literal_expr, "a + b")


def test_parse_unary_expr_handles_reference_and_mutability():
    assert parsing.parse_unary_expr("&mut x") == {
        "op": "&mut",
        "operand": "x",
        "char_span": [0, 6],
    }
    assert parsing.parse_unary_expr("-x") == {
        "op": "-",
        "operand": "x",
        "char_span": [0, 2],
    }
    _expect_error("E001", parsing.parse_unary_expr, "a + b")


def test_parse_binary_expr_uses_framework_precedence_table():
    result = parsing.parse_binary_expr("a + b * c")

    assert result == {
        "op": "+",
        "left": "a",
        "right": "b * c",
        "precedence": 10,
        "char_span": [0, 9],
    }
    assert parsing.parse_binary_expr("a == b")["precedence"] == 5
    _expect_error("E001", parsing.parse_binary_expr, "a")


def test_parse_block_expr_reports_statements_and_tail_expression():
    assert parsing.parse_block_expr("{ let x = 1; x + 2 }") == {
        "statements": ["let x = 1;"],
        "tail": "x + 2",
        "char_span": [0, 20],
    }
    assert parsing.parse_block_expr("{}") == {
        "statements": [],
        "tail": None,
        "char_span": [0, 2],
    }
    _expect_error("E003", parsing.parse_block_expr, "let x = 1;")


def test_parse_lambda_expr_handles_the_grammar_closure_form():
    assert parsing.parse_lambda_expr("|x, y| x + y") == {
        "params": ["x", "y"],
        "body": "x + y",
        "char_span": [0, 12],
    }
    assert parsing.parse_lambda_expr("|x: i64| -> i64 x")["params"] == ["x: i64"]
    _expect_error("E001", parsing.parse_lambda_expr, "a")


def test_parse_return_expr_handles_value_and_bare_return():
    assert parsing.parse_return_expr("return x;") == {
        "value": "x",
        "char_span": [0, 9],
    }
    assert parsing.parse_return_expr("return;") == {
        "value": None,
        "char_span": [0, 7],
    }
    _expect_error("E001", parsing.parse_return_expr, "foo()")


def test_parse_tuple_expr_requires_at_least_two_elements():
    assert parsing.parse_tuple_expr("(a, b, c)") == {
        "elements": ["a", "b", "c"],
        "char_span": [0, 9],
    }
    _expect_error("E003", parsing.parse_tuple_expr, "()")
    _expect_error("E001", parsing.parse_tuple_expr, "(a)")


def test_parse_array_expr_rejects_repeat_forms():
    assert parsing.parse_array_expr("[a, b, c]") == {
        "elements": ["a", "b", "c"],
        "char_span": [0, 9],
    }
    assert parsing.parse_array_expr("[]") == {
        "elements": [],
        "char_span": [0, 2],
    }
    _expect_error("E001", parsing.parse_array_expr, "[a; 2]")


def test_parse_index_expr_returns_receiver_and_index_text():
    assert parsing.parse_index_expr("a[i]") == {
        "receiver": "a",
        "index": "i",
        "char_span": [0, 4],
    }
    _expect_error("E001", parsing.parse_index_expr, "a.b")


def test_parse_field_expr_returns_receiver_and_field_name():
    assert parsing.parse_field_expr("a.b") == {
        "receiver": "a",
        "field": "b",
        "char_span": [0, 3],
    }
    _expect_error("E001", parsing.parse_field_expr, "a[0]")


def test_parse_try_expr_reports_its_operand():
    assert parsing.parse_try_expr("a?") == {
        "operand": "a",
        "char_span": [0, 2],
    }
    _expect_error("E001", parsing.parse_try_expr, "a")


def test_parse_cast_expr_reports_canonical_target_type():
    assert parsing.parse_cast_expr("a as i64") == {
        "expr": "a",
        "type": "i64",
        "char_span": [0, 8],
    }
    assert parsing.parse_cast_expr("a as Vec<u8>")["type"] == "Vec<u8>"
    _expect_error("E001", parsing.parse_cast_expr, "a")


def test_parse_pattern_simple_uses_real_pattern_node_kinds():
    assert parsing.parse_pattern_simple("Some(x)") == {
        "kind": "PatEnum",
        "text": "Some(x)",
        "char_span": [0, 7],
    }
    assert parsing.parse_pattern_simple("(a, b)") == {
        "kind": "PatTuple",
        "text": "(a, b)",
        "char_span": [0, 6],
    }
    _expect_error("E003", parsing.parse_pattern_simple, "!")


def test_parse_arm_guard_simple_parses_only_a_leading_if_guard():
    assert parsing.parse_arm_guard_simple("if x > 0") == {
        "condition": "x > 0",
        "char_span": [3, 8],
    }
    assert parsing.parse_arm_guard_simple("x > 0") is None
    _expect_error("E003", parsing.parse_arm_guard_simple, "if")


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("let x = 1;", "let"),
        ("foo();", "expr"),
        ("return x;", "return"),
        ("break;", "break"),
        ("continue;", "continue"),
        ("fn f() {}", "item"),
        ("pub struct S {}", "item"),
        ("errdefer cleanup();", "errdefer"),
        ("}", "unknown"),
        ("let = ;", "unknown"),
    ],
)
def test_parse_statement_kind_classifies_real_statements(source: str, kind: str):
    assert parsing.parse_statement_kind(source) == kind
    _expect_error("E003", parsing.parse_statement_kind, 123)  # type: ignore[arg-type]


def test_parse_expression_depth_counts_nested_pratt_nodes():
    assert parsing.parse_expression_depth("x") == 0
    assert parsing.parse_expression_depth("a + b * c") == 2
    assert parsing.parse_expression_depth("(a + b) * c") == 3
    assert parsing.parse_expression_depth("f([a, b])") == 2
    _expect_error("E003", parsing.parse_expression_depth, "return x")
