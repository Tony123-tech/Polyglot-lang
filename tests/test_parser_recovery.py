"""Panic-mode parser recovery tests."""

from __future__ import annotations

from ast_nodes import FnDecl
from lexer import lex
from parser import ParseError, parse, parse_recovering


def _function_names(program) -> list[str]:
    return [item.name for item in program.items if isinstance(item, FnDecl)]


def test_recovering_parser_reports_multiple_errors() -> None:
    source = """
fn first() -> i64 { 1 }
fn broken( { }
fn second() -> i64 { 2 }
struct Bad { field: }
fn third() -> i64 { 3 }
"""
    program, errors = parse_recovering(lex(source))

    assert len(errors) == 2
    assert all(isinstance(error, ParseError) for error in errors)
    assert _function_names(program) == ["first", "second", "third"]


def test_recovering_parser_keeps_clean_program() -> None:
    source = """
fn one() -> i64 { 1 }
fn two() -> i64 { one() + 1 }
"""
    program, errors = parse_recovering(lex(source))

    assert errors == []
    assert _function_names(program) == ["one", "two"]


def test_recovering_parser_terminates_on_stray_tokens() -> None:
    source = """
fn good() -> i64 { 1 }
let broken = ;
fn also_good() -> i64 { 2 }
"""
    program, errors = parse_recovering(lex(source))

    assert len(errors) == 1
    assert _function_names(program) == ["good", "also_good"]


def test_plain_parse_still_stops_at_first_error() -> None:
    source = """
fn broken( { }
fn later() -> i64 { 1 }
"""
    try:
        parse(lex(source))
    except ParseError as error:
        assert error.line == 2
    else:
        raise AssertionError("parse() should raise on the first syntax error")


def test_check_all_errors_reports_each_error(tmp_path, capsys) -> None:
    from api.cli import main

    path = tmp_path / "broken.poly"
    path.write_text(
        "fn broken( { }\nfn other() -> i64 { 1 }\nstruct Bad { field: }\n",
        encoding="utf-8",
    )

    assert main(["check", str(path), "--all-errors"]) == 1
    stderr = capsys.readouterr().err
    assert stderr.count("error:") == 2


def test_if_statement_followed_by_parenthesized_expression() -> None:
    # Regression: `(a - 1) / 2` after a block-form `if` used to parse as a call
    # on the if-expression, producing a translation error.
    source = """
fn f(a: i64) -> i64 {
    if a == 0 { return 0; }
    (a - 1) / 2
}
"""
    program = parse(lex(source))
    body = program.items[0].body
    assert len(body.stmts) == 1
    assert body.tail is not None
    assert type(body.tail).__name__ == "Binary"
