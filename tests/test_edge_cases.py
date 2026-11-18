"""Edge-case regressions found while auditing the JIT backend.

These cover frontend precedence, the `#[terminates]` unsigned-underflow hole,
include-path handling, and Python/JIT cross-backend parity for scalar
semantics.  Most of the file is platform independent; parity tests that use
the JIT skip on non-Apple-Silicon hosts.
"""

from __future__ import annotations

import platform

import pytest

from ast_nodes import Binary
from backends import get_backend
from backends.jit.memory import is_supported_host
from include_expander import IncludeNotFoundError, load_program
from lexer import lex
from parser import parse
from termination import TerminationError, check_function


def _compile(source: str, backend: str = "python"):
    return get_backend(backend, trusted=True).compile(parse(lex(source)))


def _expression_tree(expression: str) -> object:
    source = f"fn f(a: i64, b: i64, c: i64) -> i64 {{ {expression} }}"
    return parse(lex(source)).items[0].body.tail


# ---------------------------------------------------------------------------
# Bitwise operator precedence
# ---------------------------------------------------------------------------


def test_bitwise_or_binds_tighter_than_equality():
    tree = _expression_tree("a == b | c")
    assert isinstance(tree, Binary) and tree.op == "=="
    assert isinstance(tree.right, Binary) and tree.right.op == "|"


def test_bitwise_and_binds_tighter_than_equality():
    tree = _expression_tree("a & b == c")
    assert isinstance(tree, Binary) and tree.op == "=="
    assert isinstance(tree.left, Binary) and tree.left.op == "&"


def test_bitwise_and_binds_tighter_than_or_and_xor_between_them():
    tree = _expression_tree("a | b & c")
    assert isinstance(tree, Binary) and tree.op == "|"
    assert isinstance(tree.right, Binary) and tree.right.op == "&"

    tree = _expression_tree("a & b ^ c")
    # Rust precedence: `&` binds tighter than `^`, so this is `(a & b) ^ c`.
    assert isinstance(tree, Binary) and tree.op == "^"
    assert isinstance(tree.left, Binary) and tree.left.op == "&"


def test_arithmetic_binds_tighter_than_bitwise_operators():
    tree = _expression_tree("a & b + c")
    assert isinstance(tree, Binary) and tree.op == "&"
    assert isinstance(tree.right, Binary) and tree.right.op == "+"


def test_bitwise_operators_evaluate_correctly():
    assert _compile("fn main() -> i64 { 6 | 1 & 3 }")["main"]() == 7
    assert _compile("fn main() -> i64 { 1 | 2 & 4 }")["main"]() == 1
    assert _compile("fn main() -> i64 { 1 + 2 & 3 }")["main"]() == 3
    assert _compile("fn main() -> i64 { 2 * 3 | 1 }")["main"]() == 7
    assert _compile("fn main() -> i64 { 6 ^ 3 }")["main"]() == 5


# ---------------------------------------------------------------------------
# Termination checker: unsigned subtraction wraps
# ---------------------------------------------------------------------------


def _check_termination(source: str) -> None:
    function = parse(lex(source)).items[0]
    check_function(function)


def test_signed_condition_at_zero_is_a_valid_termination_proof():
    _check_termination(
        """
#[terminates]
fn f(mut i: i64) -> i64 {
    while i >= 0 { i = i - 1; }
    i
}
"""
    )


@pytest.mark.parametrize(
    "source",
    [
        # u64: `i >= 0` is always true and 0 - 1 wraps to u64::MAX.
        """
#[terminates]
fn f(mut i: u64) -> i64 {
    while i >= 0 { i = i - 1; }
    0
}
""",
        # u32 declared through a let binding.
        """
#[terminates]
fn f() -> i64 {
    let mut i: u32 = 5;
    while i >= 0 { i = i - 1; }
    0
}
""",
        # usize parameter.
        """
#[terminates]
fn f(mut i: usize) -> i64 {
    while i >= 0 { i = i - 1; }
    0
}
""",
        # A step larger than one can jump past zero.
        """
#[terminates]
fn f(mut i: u64) -> i64 {
    while i > 1 { i = i - 2; }
    0
}
""",
    ],
)
def test_unsigned_self_decrement_is_rejected_when_it_can_wrap(source):
    with pytest.raises(TerminationError) as error:
        _check_termination(source)
    assert error.value.code == "E002"


@pytest.mark.parametrize(
    "source",
    [
        """
#[terminates]
fn f(mut i: u64) -> i64 {
    while i > 0 { i = i - 1; }
    0
}
""",
        """
#[terminates]
fn f(mut i: usize) -> i64 {
    while i >= 1 { i -= 1; }
    0
}
""",
    ],
)
def test_unsigned_unit_decrement_from_a_positive_bound_is_accepted(source):
    _check_termination(source)


# ---------------------------------------------------------------------------
# Include handling
# ---------------------------------------------------------------------------


def test_include_target_that_is_a_directory_is_rejected(tmp_path):
    main = tmp_path / "main.poly"
    main.write_text('include!("sub");\nfn main() -> i64 { 1 }\n', encoding="utf-8")
    (tmp_path / "sub").mkdir()
    with pytest.raises(IncludeNotFoundError) as error:
        load_program(main)
    assert error.value.code == "E006"
    assert "not a file" in str(error.value)


def test_include_missing_target_still_reports_e006(tmp_path):
    main = tmp_path / "main.poly"
    main.write_text(
        'include!("nope.poly");\nfn main() -> i64 { 1 }\n', encoding="utf-8"
    )
    with pytest.raises(IncludeNotFoundError) as error:
        load_program(main)
    assert error.value.code == "E006"


def test_include_parent_segments_are_resolved_relative_to_the_includer(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "lib.poly").write_text("fn from_lib() -> i64 { 7 }\n", encoding="utf-8")
    main = sub / "main.poly"
    main.write_text(
        'include!("../lib.poly");\nfn main() -> i64 { from_lib() }\n',
        encoding="utf-8",
    )
    program = load_program(main)
    names = {item.name for item in program.items if hasattr(item, "name")}
    assert "from_lib" in names and "main" in names


def test_absolute_include_paths_are_allowed(tmp_path):
    library = tmp_path / "library.poly"
    library.write_text("fn helper() -> i64 { 3 }\n", encoding="utf-8")
    main = tmp_path / "main.poly"
    main.write_text(
        f'include!("{library}");\nfn main() -> i64 {{ helper() }}\n',
        encoding="utf-8",
    )
    program = load_program(main)
    names = {item.name for item in program.items if hasattr(item, "name")}
    assert "helper" in names


# ---------------------------------------------------------------------------
# Constant folding must match runtime Rust semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("const A: i64 = -17 % 5; fn main() -> i64 { A }", -2),
        ("const A: i64 = 17 % -5; fn main() -> i64 { A }", 2),
        ("const A: i64 = -17 / 5; fn main() -> i64 { A }", -3),
        ("const A: i64 = 17 / -5; fn main() -> i64 { A }", -3),
        ("const A: i32 = 2147483648; fn main() -> i64 { A as i64 }", -(1 << 31)),
        ("const A: u8 = 260; fn main() -> u64 { A as u64 }", 4),
    ],
)
def test_jit_constant_folding_uses_rust_truncation(source, expected):
    if not is_supported_host():
        pytest.skip("JIT backend requires macOS on Apple Silicon")
    assert _compile(source, "jit")["main"]() == expected


def test_jit_constant_division_by_zero_raises():
    if not is_supported_host():
        pytest.skip("JIT backend requires macOS on Apple Silicon")
    with pytest.raises(ZeroDivisionError):
        _compile("const A: i64 = 1 / 0; fn main() -> i64 { A }", "jit")


# ---------------------------------------------------------------------------
# Python/JIT parity for scalar semantics
# ---------------------------------------------------------------------------

PARITY_PROGRAMS = [
    ("fn main() -> i64 { -17 / 5 }", -3),
    ("fn main() -> i64 { -17 % 5 }", -2),
    ("fn main() -> i64 { 17 % -5 }", 2),
    ("fn main() -> i64 { let a: i8 = 127; let b: i8 = 1; (a + b) as i64 }", -128),
    ("fn main() -> u64 { let a: u8 = 255; let b: u8 = 1; (a + b) as u64 }", 0),
    (
        "fn main() -> u64 { let a: u64 = 18446744073709551615u64; a / 2u64 }",
        (1 << 63) - 1,
    ),
    ("fn main() -> i64 { let a: u64 = 18446744073709551615u64; a as i32 as i64 }", -1),
    (
        """fn main() -> i64 {
            let mut i: i64 = 0;
            let mut total: i64 = 0;
            while i < 10 { total = total + i; i = i + 1; }
            total
        }""",
        45,
    ),
    (
        """fn main() -> i64 {
            let x: i64 = 4;
            match x { 0 => 1, 1 | 2 => 2, 3..=4 => 3, _ => 4 }
        }""",
        3,
    ),
    (
        """fn main() -> bool {
            let a: u64 = 18446744073709551615u64;
            let b: i64 = -1;
            a > 1u64 && b < 1
        }""",
        True,
    ),
    ("fn main() -> i64 { 6 ^ 3 | 1 }", 5),
]


@pytest.mark.parametrize(("source", "expected"), PARITY_PROGRAMS)
def test_backend_parity_for_scalar_semantics(source, expected):
    if not is_supported_host():
        pytest.skip("JIT backend requires macOS on Apple Silicon")
    python_result = _compile(source, "python")["main"]()
    jit_result = _compile(source, "jit")["main"]()
    assert python_result == expected
    assert jit_result == expected


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_jit_requires_apple_silicon_or_reports_unsupported():
    backend = get_backend("jit", trusted=True)
    source = "fn main() -> i64 { 1 }"
    if platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }:
        assert backend.compile(parse(lex(source)))["main"]() == 1
    else:
        from backends.jit.errors import JITUnsupportedError

        with pytest.raises(JITUnsupportedError):
            backend.compile(parse(lex(source)))


def test_jit_backend_composes_with_parse_and_lex_errors():
    from lexer import LexError
    from parser import ParseError

    with pytest.raises(LexError):
        parse(lex("fn main() -> i64 { ` }"))
    with pytest.raises(ParseError):
        parse(lex("fn main( -> i64 { 1 }"))
