"""Behavioral tests for FRAME 1 (lexing, parsing and source spans)."""

from __future__ import annotations

import pytest

import compiler.core.ast_nodes as ast
from compiler.frontend import framework
from compiler.core.lexer import lex
from compiler.core.parser import parse


def test_lex_source_with_spans_utf8_bytes_and_line_columns():
    source = 'let café = "é";\nlet x = 1;'
    result = framework.lex_source_with_spans(source)

    ident = next(token for token in result["tokens"] if token["value"] == "café")
    assert ident["char_span"] == [4, 8]
    assert source[4:8] == "café"
    assert ident["byte_span"] == [4, 9]
    assert ident["line_col_span"] == [[1, 5], [1, 9]]

    x_token = next(token for token in result["tokens"] if token["value"] == "x")
    assert x_token["line_col_span"][0] == [2, 5]
    assert result["byte_length"] == len(source.encode("utf-8"))
    assert result["line_count"] == 2

    eof = result["tokens"][-1]
    assert eof["type"] == "EOF"
    assert eof["char_span"] == [len(source), len(source)]


def test_lex_raw_and_interpolated_strings_handles_hashes_and_nested_braces():
    source = 'let a = r#"C:\\temp\\n {raw}"#; let b = "x\\n{a + {b}}";'
    result = framework.lex_raw_and_interpolated_strings(source)

    assert result["diagnostics"] == []
    raw, interpolated = result["tokens"]
    assert raw["kind"] == "raw"
    assert raw["value"] == r"C:\temp\n {raw}"
    assert raw["interpolations"] == []
    assert source[raw["char_span"][0] : raw["char_span"][1]].startswith('r#"')

    assert interpolated["kind"] == "interpolated"
    assert interpolated["interpolations"][0]["expression"] == "a + {b}"
    assert interpolated["interpolations"][0]["terminated"] is True
    assert [escape["value"] for escape in interpolated["escapes"]] == ["\n"]
    assert interpolated["value"] == "x\n{}"


def test_lex_raw_and_interpolated_strings_reports_bad_escapes_and_braces():
    result = framework.lex_raw_and_interpolated_strings('"bad \\q {oops"')

    messages = [diagnostic["message"] for diagnostic in result["diagnostics"]]
    assert any(message.startswith("unknown escape") for message in messages)
    assert "unterminated interpolation" in messages
    assert all(diagnostic["code"] == "E001" for diagnostic in result["diagnostics"])


def test_lex_recover_unterminated_token_continues_after_e001():
    source = 'let ok = 1;\nlet bad = "oops\nlet tail = 2;'
    result = framework.lex_recover_unterminated_token(source)

    assert result["diagnostic_count"] == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["code"] == "E001"
    assert diagnostic["message"] == "unterminated string"
    assert diagnostic["line"] == 2
    assert diagnostic["char_span"][0] == source.index('"oops')
    assert diagnostic["byte_span"][0] == len(
        source[: diagnostic["char_span"][0]].encode("utf-8")
    )
    assert [token["value"] for token in result["tokens"]][-5:-1] == [
        "let",
        "tail",
        "=",
        "2",
    ]


def test_lex_recover_unterminated_token_recovers_from_block_comment():
    result = framework.lex_recover_unterminated_token(
        "let a = 1; /* broken\nlet b = 2;"
    )

    assert any(
        diagnostic["kind"] == "COMMENT" and diagnostic["code"] == "E001"
        for diagnostic in result["diagnostics"]
    )
    assert [token["value"] for token in result["tokens"]][-5:-2] == ["let", "b", "="]


def test_parse_pratt_expr_with_precedence_respects_binding_power():
    result = framework.parse_pratt_expr_with_precedence("a + b * c - d")
    node = result["ast"]

    assert node["op"] == "-"
    assert node["left"]["op"] == "+"
    assert node["left"]["right"]["op"] == "*"
    assert node["right"]["value"] == "d"
    assert node["char_span"] == [0, len("a + b * c - d")]
    assert framework.PRECEDENCE_TABLE["*"] > framework.PRECEDENCE_TABLE["+"]
    assert result["canonical"] == "a + b * c - d"


def test_parse_pratt_expr_with_precedence_handles_right_assoc_postfix_and_tokens():
    assignment = framework.parse_pratt_expr_with_precedence("a = b = c")["ast"]
    assert assignment["op"] == "="
    assert assignment["right"]["op"] == "="

    postfix = framework.parse_pratt_expr_with_precedence("-f(x).y? + 1")["ast"]
    assert postfix["op"] == "+"
    assert postfix["left"]["op"] == "-"
    assert postfix["left"]["operand"]["kind"] == "try"
    assert postfix["left"]["operand"]["operand"]["kind"] == "field"

    from_tokens = framework.parse_pratt_expr_with_precedence(lex("x + 1"))
    assert from_tokens["ast"]["kind"] == "binary"
    assert from_tokens["ast"]["left"]["value"] == "x"


def test_parse_generic_type_params_bounds_defaults_and_const_params():
    params = framework.parse_generic_type_params(
        "<T: Clone + Debug, U = Vec<T>, const N: usize = 4>"
    )

    assert [param["name"] for param in params] == ["T", "U", "N"]
    assert [bound["name"] for bound in params[0]["bounds"]] == ["Clone", "Debug"]
    assert params[0]["char_span"] == [1, 17]
    assert params[1]["default"]["text"] == "Vec<T>"
    assert params[2]["kind"] == "const"
    assert params[2]["type"]["text"] == "usize"
    assert params[2]["default"]["text"] == "4"
    assert framework.parse_generic_type_params("fn f() {}") == []


def test_parse_pattern_match_arms_nested_patterns_guards_and_full_match():
    result = framework.parse_pattern_match_arms(
        "Some(Point { x, y: 0 }) if x > 0 => x, (a, b) | (b, a) => a + b, _ => 0"
    )

    assert result["arm_count"] == 3
    first, second, third = result["arms"]
    assert first["pattern"]["kind"] == "PatEnum"
    assert first["pattern"]["args"][0]["kind"] == "PatStruct"
    assert [name for name, _pattern in first["pattern"]["args"][0]["fields"]] == [
        "x",
        "y",
    ]
    assert first["guard_text"] == "x > 0"
    assert first["char_span"][0] == 0
    assert second["pattern"]["kind"] == "PatOr"
    assert second["pattern_text"] == "(a, b) | (b, a)"
    assert third["pattern"]["kind"] == "PatWildcard"

    full = framework.parse_pattern_match_arms(
        "match value { Ok(x) if x > 1 => x, Err(_) => 0 }"
    )
    assert full["scrutinee"]["kind"] == "Ident"
    assert full["arms"][0]["pattern"]["kind"] == "PatEnum"
    assert full["arms"][1]["guard"] is None


def test_parse_macro_invocations_delimiters_context_and_nesting():
    result = framework.parse_macro_invocations(
        'println!("hi"); let v = vec![1, 2, 3]; assert!(v.len() == 3, "bad");'
    )
    macros = result["macros"]

    assert [macro["path"] for macro in macros] == ["println", "vec", "assert"]
    assert macros[0]["delimiter"] == "()"
    assert macros[0]["kind"] == "statement"
    assert macros[1]["delimiter"] == "[]"
    assert macros[1]["kind"] == "expression"
    assert len(macros[1]["args"]) == 3
    assert macros[1]["args"][0]["value"] == "1"
    assert len(macros[2]["args"]) == 2

    nested = framework.parse_macro_invocations("assert!(vec![1].len() == 1);")
    assert {macro["path"] for macro in nested["macros"]} == {"assert", "vec"}

    broken = framework.parse_macro_invocations("foo!(1, 2")
    assert broken["diagnostics"][0]["message"] == "unterminated macro invocation"
    assert broken["macros"][0]["terminated"] is False


def test_parse_error_sync_recovery_collects_multiple_syntax_diagnostics():
    result = framework.parse_error_sync_recovery(
        "fn f() { let a = ; let b = ; let c = 3; }"
    )

    assert result["diagnostic_count"] >= 2
    assert all(diagnostic["code"] == "SYNTAX" for diagnostic in result["diagnostics"])
    assert len(result["sync_points"]) >= 2
    assert result["recovered_count"] >= 1
    assert all(diagnostic["char_span"][0] >= 0 for diagnostic in result["diagnostics"])


def test_ast_node_span_normalize_fills_leaves_and_container_ranges():
    source = "fn f(a: i64) -> i64 { a + 1 }"
    program = parse(lex(source))
    normalized = framework.ast_node_span_normalize(program, source)

    assert normalized is not program
    assert normalized.items[0].span == (0, len(source))
    assert normalized.items[0].body.span == (source.index("{"), len(source))

    manual = ast.Binary(
        1,
        1,
        "+",
        ast.LitInt(1, 1, 1, "1"),
        ast.LitInt(1, 5, 2, "2"),
    )
    assert framework.ast_node_span_normalize(manual, "1 + 2").span == (0, 5)
    assert manual.span is None


def test_ast_pretty_print_canonical_round_trips_program_and_prints_types():
    source = "fn add(a: i64, b: i64) -> i64 {\n    let x = a + b * 2;\n    x\n}\n"
    program = parse(lex(source))
    text = framework.ast_pretty_print_canonical(program)

    assert text == source
    assert framework.ast_pretty_print_canonical(parse(lex(text))) == text
    assert (
        framework.ast_pretty_print_canonical(
            ast.TypeName(1, 1, "Vec", [ast.TypeName(1, 1, "i64")])
        )
        == "Vec<i64>"
    )
    with pytest.raises(TypeError):
        framework.ast_pretty_print_canonical(object())


def test_registry_resolves_every_frame_one_function():
    from framework_functions import iter_functions

    for item in iter_functions(1):
        assert callable(item.callable)
        assert item.callable.__doc__
