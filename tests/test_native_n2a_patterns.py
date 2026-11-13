"""N2a match-pattern expansion tests."""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
import warnings

import pytest

from backends.errors import UnsupportedNativePatternError
from backends.python.translator import TranslateError, compile_rust
from backends.rust.compiler import RustCompileError
from lexer import lex
from parser import ParseError, parse

SOURCES = {
    "pattern_or": """
fn pattern_or(x: i64) -> i64 {
    match x {
        1 | 3 | 5 => 1,
        _ => 0,
    }
}
""",
    "pattern_range_exclusive": """
fn pattern_range_exclusive(x: i64) -> i64 {
    match x {
        1..4 => 1,
        _ => 0,
    }
}
""",
    "pattern_range_inclusive": """
fn pattern_range_inclusive(x: i64) -> i64 {
    match x {
        1..=4 => 1,
        _ => 0,
    }
}
""",
    "pattern_tuple": """
fn pattern_tuple(pair: (i64, i64)) -> i64 {
    match pair {
        (1, 2) => 12,
        _ => 0,
    }
}
""",
    "pattern_nested_tuple": """
fn pattern_nested_tuple(pair: (i64, (i64, i64))) -> i64 {
    match pair {
        (1, (2, 3)) => 123,
        _ => 0,
    }
}
""",
    "pattern_unit_tuple": """
fn pattern_unit_tuple() -> i64 {
    match () {
        () => 1,
    }
}
""",
    "pattern_single_tuple": """
fn pattern_single_tuple(value: (i64,)) -> i64 {
    match value {
        (7,) => 1,
        _ => 0,
    }
}
""",
    "pattern_struct_explicit": """
struct PatternPair {
    left: i64,
    right: i64,
}

impl PatternPair {
    fn new(left: i64, right: i64) -> PatternPair {
        PatternPair { left: left, right: right }
    }
}

fn pattern_struct_explicit(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left: 3, right: 4 } => 1,
        _ => 0,
    }
}
""",
    "pattern_struct_shorthand": """
fn pattern_struct_shorthand(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left, .. } => left,
    }
}
""",
    "pattern_struct_rest": """
fn pattern_struct_rest(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { right: 4, .. } => 1,
        _ => 0,
    }
}
""",
    "pattern_binding": """
fn pattern_binding(x: i64) -> i64 {
    match x {
        whole @ 4 => whole,
        _ => 0,
    }
}
""",
    "pattern_range_open_start": """
fn pattern_range_open_start(x: i64) -> i64 {
    match x {
        ..=3 => 1,
        _ => 0,
    }
}
""",
    "pattern_range_open_end": """
fn pattern_range_open_end(x: i64) -> i64 {
    match x {
        3.. => 1,
        _ => 0,
    }
}
""",
    "pattern_range_negative": """
fn pattern_range_negative(x: i64) -> i64 {
    match x {
        -3..=-1 => 1,
        _ => 0,
    }
}
""",
    "pattern_or_range": """
fn pattern_or_range(x: i64) -> i64 {
    match x {
        1..=2 | 4 => 1,
        _ => 0,
    }
}
""",
    "pattern_tuple_binding": """
fn pattern_tuple_binding(pair: (i64, i64)) -> i64 {
    match pair {
        (1, value) => value,
        _ => 0,
    }
}
""",
    "pattern_nested_binding": """
fn pattern_nested_binding(pair: (i64, i64)) -> (i64, i64) {
    match pair {
        whole @ (1, 2) => whole,
        _ => (0, 0),
    }
}
""",
    "pattern_struct_nested": """
fn pattern_struct_nested(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left: value @ 3, .. } => value,
        _ => 0,
    }
}
""",
    "pattern_ref_literal": """
fn pattern_ref_literal(x: i64) -> i64 {
    match &x {
        &4 => 1,
        _ => 0,
    }
}
""",
    "pattern_ref_binding": """
fn pattern_ref_binding(x: i64) -> i64 {
    match &x {
        &value => value,
    }
}
""",
    "pattern_slice_fixed": """
fn pattern_slice_fixed(xs: Vec<i64>) -> i64 {
    match xs {
        [a, b] => *a + *b,
        _ => 0,
    }
}
""",
    "pattern_slice_rest_middle": """
fn pattern_slice_rest_middle(xs: Vec<i64>) -> i64 {
    match xs {
        [1, .., 3] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_rest_end": """
fn pattern_slice_rest_end(xs: Vec<i64>) -> i64 {
    match xs {
        [1, 2, ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_rest_start": """
fn pattern_slice_rest_start(xs: Vec<i64>) -> i64 {
    match xs {
        [.., 2, 3] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_all_rest": """
fn pattern_slice_all_rest(xs: Vec<i64>) -> i64 {
    match xs {
        [..] => 1,
    }
}
""",
    "pattern_slice_empty": """
fn pattern_slice_empty(xs: Vec<i64>) -> i64 {
    match xs {
        [] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_literals": """
fn pattern_slice_literals(xs: Vec<i64>) -> i64 {
    match xs {
        [1, 2, 3] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_bindings": """
fn pattern_slice_bindings(xs: Vec<i64>) -> i64 {
    match xs {
        [first, .., last] => *first + *last,
        _ => 0,
    }
}
""",
    "pattern_slice_tuple": """
fn pattern_slice_tuple(xs: Vec<(i64, i64)>) -> i64 {
    match xs {
        [(1, 2), ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_or": """
fn pattern_slice_or(xs: Vec<i64>) -> i64 {
    match xs {
        [(1 | 2), ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_range": """
fn pattern_slice_range(xs: Vec<i64>) -> i64 {
    match xs {
        [1..=5, ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_ref": """
fn pattern_slice_ref(xs: Vec<i64>) -> i64 {
    match xs {
        [&1, ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_at": """
fn pattern_slice_at(xs: Vec<i64>) -> i64 {
    match xs {
        [value @ 1..=5, ..] => *value,
        _ => 0,
    }
}
""",
    "pattern_slice_guard": """
fn pattern_slice_guard(xs: Vec<i64>) -> i64 {
    match xs {
        [first, ..] if *first > 0 => 1,
        _ => 0,
    }
}
""",
    "pattern_slice_tuple_context": """
fn pattern_slice_tuple_context() -> i64 {
    let values = [1, 2];
    let pair = (values.as_slice(), 3);
    match pair {
        ([a, b], c) => *a + *b + c,
        _ => 0,
    }
}
""",
    "pattern_slice_rest_binding": """
fn pattern_slice_rest_binding(xs: Vec<i64>) -> i64 {
    match xs {
        [first, rest @ ..] => *first + rest.len() as i64,
        _ => 0,
    }
}
""",
    "pattern_slice_only_rest_binding": """
fn pattern_slice_only_rest_binding(xs: Vec<i64>) -> i64 {
    match xs {
        [rest @ ..] => rest.len() as i64,
    }
}
""",
    "pattern_const_bare": """
const LIMIT: i64 = 7;

fn pattern_const_bare(x: i64) -> i64 {
    match x {
        const LIMIT => 1,
        _ => 0,
    }
}
""",
    "pattern_const_path": """
const MAX_VALUE: i64 = 9;

fn pattern_const_path(x: i64) -> i64 {
    match x {
        const MAX_VALUE => 1,
        _ => 0,
    }
}
""",
    "pattern_const_qualified": """
enum Choice {
    A,
    B,
}

fn pattern_const_qualified() -> i64 {
    let x = Choice::A;
    match x {
        const Choice::A => 1,
        Choice::B => 0,
    }
}
""",
    "pattern_const_binding": """
fn pattern_const_binding(x: i64) -> i64 {
    match x {
        value @ const MAX_VALUE => value,
        _ => 0,
    }
}
""",
    "pattern_const_tuple": """
fn pattern_const_tuple(pair: (i64, i64)) -> i64 {
    match pair {
        (const LIMIT, y) => y,
        _ => 0,
    }
}
""",
    "pattern_path_constant": """
fn pattern_path_constant() -> i64 {
    let x = Choice::B;
    match x {
        Choice::B => 1,
        _ => 0,
    }
}
""",
    "pattern_path_or": """
fn pattern_path_or() -> i64 {
    let x = Choice::A;
    match x {
        Choice::A | Choice::B => 1,
    }
}
""",
    "pattern_struct_rest_batch3": """
fn pattern_struct_rest_batch3(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left, .. } => left,
    }
}
""",
    "pattern_or_tuple": """
fn pattern_or_tuple(pair: (i64, i64)) -> i64 {
    match pair {
        (1, value) | (2, value) => value,
        _ => 0,
    }
}
""",
    "pattern_tuple_or": """
fn pattern_tuple_or(pair: (i64, i64)) -> i64 {
    match pair {
        (1 | 2, value) => value,
        _ => 0,
    }
}
""",
    "pattern_slice_or_inner": """
fn pattern_slice_or_inner(xs: Vec<i64>) -> i64 {
    match xs {
        [1 | 2, ..] => 1,
        _ => 0,
    }
}
""",
    "pattern_struct_or_inner": """
fn pattern_struct_or_inner(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left: 1 | 2, .. } => 1,
        _ => 0,
    }
}
""",
    "pattern_binding_or": """
fn pattern_binding_or(x: i64) -> i64 {
    match x {
        value @ (1 | 2) => value,
        _ => 0,
    }
}
""",
    "pattern_guard_or": """
fn pattern_guard_or(x: i64) -> i64 {
    match x {
        1 | 2 if x > 0 => 1,
        _ => 0,
    }
}
""",
    "pattern_guard_tuple": """
fn pattern_guard_tuple(pair: (i64, i64)) -> i64 {
    match pair {
        (a, b) if a > b => a - b,
        _ => 0,
    }
}
""",
    "pattern_guard_slice": """
fn pattern_guard_slice(xs: Vec<i64>) -> i64 {
    match xs {
        [a, ..] if *a > 0 => 1,
        _ => 0,
    }
}
""",
    "pattern_guard_struct": """
fn pattern_guard_struct(pair: PatternPair) -> i64 {
    match pair {
        PatternPair { left, .. } if left > 0 => left,
        _ => 0,
    }
}
""",
    "pattern_guard_binding": """
fn pattern_guard_binding(x: i64) -> i64 {
    match x {
        value @ 1..=5 if value > 2 => value,
        _ => 0,
    }
}
""",
    "pattern_guard_or_expr": """
fn pattern_guard_or_expr(x: i64) -> i64 {
    match x {
        1 | 2 if x > 0 | x < 0 => 1,
        _ => 0,
    }
}
""",
    "pattern_many_arms": """
fn pattern_many_arms(x: i64) -> i64 {
    match x {
        1 => 10,
        2 => 20,
        value => value,
    }
}
""",
    "pattern_arm_bindings": """
fn pattern_arm_bindings(x: i64) -> i64 {
    match x {
        value @ 1 => value,
        other @ 2 => other,
        _ => 0,
    }
}
""",
    "pattern_option_or_bindings": """
enum MatchOption {
    Some(i64),
    None,
}

fn pattern_option_or_bindings() -> i64 {
    let value = MatchOption::Some(8);
    match value {
        MatchOption::Some(x) | MatchOption::Some(x) => x,
        MatchOption::None => 0,
    }
}
""",
    "pattern_slice_wrong_length": """
fn pattern_slice_wrong_length(xs: Vec<i64>) -> i64 {
    match xs {
        [a, b] => *a + *b,
        _ => 0,
    }
}
""",
    "pattern_nested_or_tuple": """
fn pattern_nested_or_tuple(value: (i64, (i64, i64))) -> i64 {
    match value {
        (1 | 2, (3 | 4, x)) => x,
        _ => 0,
    }
}
""",
}

PYTHON_ONLY_SOURCES = {
    "pattern_slice_struct_context": """
struct Grid {
    coords: Vec<i64>,
}

fn pattern_slice_struct_context(grid: Grid) -> i64 {
    match grid {
        Grid { coords: [x, y] } => x + y,
        _ => 0,
    }
}
""",
    "pattern_slice_enum_context": """
enum MaybeGrid {
    Values(Vec<i64>),
    Empty,
}

fn pattern_slice_enum_context(value: MaybeGrid) -> i64 {
    match value {
        MaybeGrid::Values([x, y]) => x + y,
        _ => 0,
    }
}
""",
    "pattern_slice_nested": """
fn pattern_slice_nested(values: Vec<Vec<i64>>) -> i64 {
    match values {
        [[a], [b]] => a + b,
        _ => 0,
    }
}
""",
    "pattern_box": """
fn pattern_box(x: i64) -> i64 {
    match x {
        box 1 => 1,
        _ => 0,
    }
}
""",
    "pattern_box_tuple": """
fn pattern_box_tuple(x: (i64, i64)) -> i64 {
    match x {
        box (a, b) => a + b,
        _ => 0,
    }
}
""",
    "pattern_box_nested": """
fn pattern_box_nested(x: i64) -> i64 {
    match x {
        box box value => value,
    }
}
""",
    "pattern_box_binding": """
fn pattern_box_binding(x: i64) -> i64 {
    match x {
        box value @ 1..=5 => value,
        _ => 0,
    }
}
""",
    "pattern_box_ident": """
fn pattern_box_ident(x: i64) -> i64 {
    match x {
        box value => value,
    }
}
""",
    "pattern_or_enum_payload": """
enum PythonMatchOption {
    Some(i64),
    None,
}

fn pattern_or_enum_payload(value: PythonMatchOption) -> i64 {
    match value {
        PythonMatchOption::Some(x) | PythonMatchOption::Some(x) => x,
        _ => 0,
    }
}
""",
    "pattern_tuple_wrong_arity_python": """
fn pattern_tuple_wrong_arity_python(value: (i64, i64, i64)) -> i64 {
    match value {
        (a, b) => a + b,
        _ => 0,
    }
}
""",
}


@pytest.mark.native
def test_pattern_or(native_ns):
    assert native_ns["pattern_or"](3) == 1
    assert native_ns["pattern_or"](2) == 0


@pytest.mark.native
def test_pattern_range_exclusive(native_ns):
    assert native_ns["pattern_range_exclusive"](1) == 1
    assert native_ns["pattern_range_exclusive"](3) == 1
    assert native_ns["pattern_range_exclusive"](4) == 0


@pytest.mark.native
def test_pattern_range_inclusive(native_ns):
    assert native_ns["pattern_range_inclusive"](4) == 1
    assert native_ns["pattern_range_inclusive"](5) == 0


@pytest.mark.native
def test_pattern_tuple(native_ns):
    assert native_ns["pattern_tuple"]((1, 2)) == 12
    assert native_ns["pattern_tuple"]((1, 3)) == 0


@pytest.mark.native
def test_pattern_nested_tuple(native_ns):
    assert native_ns["pattern_nested_tuple"]((1, (2, 3))) == 123
    assert native_ns["pattern_nested_tuple"]((1, (2, 4))) == 0


@pytest.mark.native
def test_pattern_unit_tuple(native_ns):
    assert native_ns["pattern_unit_tuple"]() == 1


@pytest.mark.native
def test_pattern_single_tuple(native_ns):
    assert native_ns["pattern_single_tuple"]((7,)) == 1
    assert native_ns["pattern_single_tuple"]((8,)) == 0


@pytest.mark.native
def test_pattern_struct_explicit(native_ns):
    pair = native_ns["PatternPair"](3, 4)
    assert native_ns["pattern_struct_explicit"](pair) == 1


@pytest.mark.native
def test_pattern_struct_shorthand(native_ns):
    pair = native_ns["PatternPair"](9, 4)
    assert native_ns["pattern_struct_shorthand"](pair) == 9


@pytest.mark.native
def test_pattern_struct_rest(native_ns):
    pair = native_ns["PatternPair"](9, 4)
    assert native_ns["pattern_struct_rest"](pair) == 1


@pytest.mark.native
def test_pattern_binding(native_ns):
    assert native_ns["pattern_binding"](4) == 4
    assert native_ns["pattern_binding"](5) == 0


@pytest.mark.native
def test_pattern_range_open_start(native_ns):
    assert native_ns["pattern_range_open_start"](3) == 1
    assert native_ns["pattern_range_open_start"](4) == 0


@pytest.mark.native
def test_pattern_range_open_end(native_ns):
    assert native_ns["pattern_range_open_end"](3) == 1
    assert native_ns["pattern_range_open_end"](2) == 0


@pytest.mark.native
def test_pattern_range_negative(native_ns):
    assert native_ns["pattern_range_negative"](-2) == 1
    assert native_ns["pattern_range_negative"](0) == 0


@pytest.mark.native
def test_pattern_or_range(native_ns):
    assert native_ns["pattern_or_range"](2) == 1
    assert native_ns["pattern_or_range"](4) == 1
    assert native_ns["pattern_or_range"](3) == 0


@pytest.mark.native
def test_pattern_tuple_binding(native_ns):
    assert native_ns["pattern_tuple_binding"]((1, 8)) == 8
    assert native_ns["pattern_tuple_binding"]((2, 8)) == 0


@pytest.mark.native
def test_pattern_nested_binding(native_ns):
    assert native_ns["pattern_nested_binding"]((1, 2)) == (1, 2)
    assert native_ns["pattern_nested_binding"]((1, 3)) == (0, 0)


@pytest.mark.native
def test_pattern_struct_nested(native_ns):
    pair = native_ns["PatternPair"](3, 4)
    assert native_ns["pattern_struct_nested"](pair) == 3


@pytest.mark.native
def test_pattern_ref_literal(native_ns):
    assert native_ns["pattern_ref_literal"](4) == 1
    assert native_ns["pattern_ref_literal"](3) == 0


@pytest.mark.native
def test_pattern_ref_binding(native_ns):
    assert native_ns["pattern_ref_binding"](9) == 9


@pytest.mark.native
def test_pattern_slice_fixed(native_ns):
    assert native_ns["pattern_slice_fixed"]([3, 4]) == 7
    assert native_ns["pattern_slice_fixed"]([3]) == 0


@pytest.mark.native
def test_pattern_slice_rest_middle(native_ns):
    assert native_ns["pattern_slice_rest_middle"]([1, 2, 3]) == 1
    assert native_ns["pattern_slice_rest_middle"]([1, 2, 4]) == 0


@pytest.mark.native
def test_pattern_slice_rest_end(native_ns):
    assert native_ns["pattern_slice_rest_end"]([1, 2]) == 1
    assert native_ns["pattern_slice_rest_end"]([1, 2, 3]) == 1
    assert native_ns["pattern_slice_rest_end"]([1, 3]) == 0


@pytest.mark.native
def test_pattern_slice_rest_start(native_ns):
    assert native_ns["pattern_slice_rest_start"]([2, 3]) == 1
    assert native_ns["pattern_slice_rest_start"]([1, 2, 3]) == 1
    assert native_ns["pattern_slice_rest_start"]([1, 2, 4]) == 0


@pytest.mark.native
def test_pattern_slice_all_rest(native_ns):
    assert native_ns["pattern_slice_all_rest"]([]) == 1
    assert native_ns["pattern_slice_all_rest"]([1, 2]) == 1


@pytest.mark.native
def test_pattern_slice_empty(native_ns):
    assert native_ns["pattern_slice_empty"]([]) == 1
    assert native_ns["pattern_slice_empty"]([1]) == 0


@pytest.mark.native
def test_pattern_slice_literals(native_ns):
    assert native_ns["pattern_slice_literals"]([1, 2, 3]) == 1
    assert native_ns["pattern_slice_literals"]([1, 2]) == 0


@pytest.mark.native
def test_pattern_slice_bindings(native_ns):
    assert native_ns["pattern_slice_bindings"]([4, 5, 6]) == 10
    assert native_ns["pattern_slice_bindings"]([]) == 0


@pytest.mark.native
def test_pattern_slice_tuple(native_ns):
    assert native_ns["pattern_slice_tuple"]([(1, 2)]) == 1
    assert native_ns["pattern_slice_tuple"]([(1, 3)]) == 0


@pytest.mark.native
def test_pattern_slice_or(native_ns):
    assert native_ns["pattern_slice_or"]([2]) == 1
    assert native_ns["pattern_slice_or"]([3]) == 0


@pytest.mark.native
def test_pattern_slice_range(native_ns):
    assert native_ns["pattern_slice_range"]([3]) == 1
    assert native_ns["pattern_slice_range"]([6]) == 0


@pytest.mark.native
def test_pattern_slice_ref(native_ns):
    assert native_ns["pattern_slice_ref"]([1]) == 1
    assert native_ns["pattern_slice_ref"]([2]) == 0


@pytest.mark.native
def test_pattern_slice_at(native_ns):
    assert native_ns["pattern_slice_at"]([5]) == 5
    assert native_ns["pattern_slice_at"]([8]) == 0


@pytest.mark.native
def test_pattern_slice_guard(native_ns):
    assert native_ns["pattern_slice_guard"]([2]) == 1
    assert native_ns["pattern_slice_guard"]([-1]) == 0


@pytest.mark.native
def test_pattern_slice_tuple_context(native_ns):
    assert native_ns["pattern_slice_tuple_context"]() == 6


@pytest.mark.native
def test_pattern_slice_rest_binding(native_ns):
    assert native_ns["pattern_slice_rest_binding"]([3, 4, 5]) == 5
    assert native_ns["pattern_slice_rest_binding"]([]) == 0


@pytest.mark.native
def test_pattern_slice_only_rest_binding(native_ns):
    assert native_ns["pattern_slice_only_rest_binding"]([]) == 0
    assert native_ns["pattern_slice_only_rest_binding"]([1, 2]) == 2


@pytest.mark.parametrize(
    "source",
    [
        "fn f(x: Box<i64>) -> i64 { match x { box value => *value } }",
        "fn f(x: Box<i64>) -> i64 { match x { box 1 => 1, _ => 0 } }",
        "fn f(x: Box<(i64, i64)>) -> i64 { match x { box (a, b) => a + b } }",
        "fn f(x: Box<Box<i64>>) -> i64 { match x { box box value => *value } }",
        "fn f(x: Box<i64>) -> i64 { match x { name @ box value => *value } }",
    ],
)
def test_pattern_box_native_rejected(source, native_backend):
    with pytest.raises(UnsupportedNativePatternError, match="Box patterns"):
        native_backend.compile(parse(lex(source)))


@pytest.mark.native
def test_pattern_const_bare(native_ns):
    assert native_ns["pattern_const_bare"](7) == 1
    assert native_ns["pattern_const_bare"](8) == 0


@pytest.mark.native
def test_pattern_const_path(native_ns):
    assert native_ns["pattern_const_path"](9) == 1
    assert native_ns["pattern_const_path"](8) == 0


@pytest.mark.native
def test_pattern_const_qualified(native_ns):
    assert native_ns["pattern_const_qualified"]() == 1


@pytest.mark.native
def test_pattern_const_binding(native_ns):
    assert native_ns["pattern_const_binding"](9) == 9


@pytest.mark.native
def test_pattern_const_tuple(native_ns):
    assert native_ns["pattern_const_tuple"]((7, 5)) == 5
    assert native_ns["pattern_const_tuple"]((8, 5)) == 0


def _macro_pattern_source(pattern: str) -> str:
    return f"""
fn test_macro_pattern() -> i64 {{
    match 1 {{
        {pattern} => 1,
        _ => 0,
    }}
}}
"""


@pytest.mark.parametrize(
    "pattern",
    ["m!()", "m!(1, 2)", "m! { 1 }", "m!(x @ 1..=5)"],
)
def test_pattern_macro_parses(pattern):
    program = parse(lex(_macro_pattern_source(pattern)))
    assert program.items[0].body.tail.arms[0].pattern.path == "m"


@pytest.mark.native
def test_pattern_macro_nested_in_tuple(native_backend):
    source = _macro_pattern_source("(m!(), y)")
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_path_constant(native_ns):
    assert native_ns["pattern_path_constant"]() == 1


@pytest.mark.native
def test_pattern_path_binding_rejected(native_backend):
    source = """
enum Choice { A, B }
fn invalid_path_binding() -> i64 {
    let value = Choice::A;
    match value {
        Choice::A @ x => 1,
        _ => 0,
    }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_path_or(native_ns):
    assert native_ns["pattern_path_or"]() == 1


def test_pattern_standalone_rest_rejected():
    source = "fn invalid() -> i64 { match 1 { .. => 1, _ => 0 } }"
    with pytest.raises(ParseError, match="rest pattern is only valid") as error:
        parse(lex(source))
    assert error.value.line == 1
    assert error.value.col > 0


@pytest.mark.native
def test_pattern_struct_rest_batch3(native_ns):
    assert native_ns["pattern_struct_rest_batch3"](
        native_ns["PatternPair"](11, 12),
    ) == 11


@pytest.mark.native
def test_pattern_or_tuple(native_ns):
    assert native_ns["pattern_or_tuple"]((1, 8)) == 8
    assert native_ns["pattern_or_tuple"]((2, 9)) == 9
    assert native_ns["pattern_or_tuple"]((3, 9)) == 0


@pytest.mark.native
def test_pattern_tuple_or(native_ns):
    assert native_ns["pattern_tuple_or"]((2, 8)) == 8
    assert native_ns["pattern_tuple_or"]((3, 8)) == 0


@pytest.mark.native
def test_pattern_slice_or_inner(native_ns):
    assert native_ns["pattern_slice_or_inner"]([2]) == 1
    assert native_ns["pattern_slice_or_inner"]([3]) == 0


@pytest.mark.native
def test_pattern_struct_or_inner(native_ns):
    assert native_ns["pattern_struct_or_inner"](
        native_ns["PatternPair"](2, 0),
    ) == 1
    assert native_ns["pattern_struct_or_inner"](
        native_ns["PatternPair"](3, 0),
    ) == 0


@pytest.mark.native
def test_pattern_binding_or(native_ns):
    assert native_ns["pattern_binding_or"](2) == 2
    assert native_ns["pattern_binding_or"](3) == 0


@pytest.mark.native
def test_pattern_guard_or(native_ns):
    assert native_ns["pattern_guard_or"](2) == 1
    assert native_ns["pattern_guard_or"](-1) == 0


@pytest.mark.native
def test_pattern_guard_tuple(native_ns):
    assert native_ns["pattern_guard_tuple"]((7, 3)) == 4
    assert native_ns["pattern_guard_tuple"]((2, 5)) == 0


@pytest.mark.native
def test_pattern_guard_slice(native_ns):
    assert native_ns["pattern_guard_slice"]([5]) == 1
    assert native_ns["pattern_guard_slice"]([-1]) == 0


@pytest.mark.native
def test_pattern_guard_struct(native_ns):
    assert native_ns["pattern_guard_struct"](
        native_ns["PatternPair"](5, 0),
    ) == 5
    assert native_ns["pattern_guard_struct"](
        native_ns["PatternPair"](-5, 0),
    ) == 0


@pytest.mark.native
def test_pattern_guard_binding(native_ns):
    assert native_ns["pattern_guard_binding"](4) == 4
    assert native_ns["pattern_guard_binding"](2) == 0


@pytest.mark.native
def test_pattern_guard_or_expr(native_ns):
    assert native_ns["pattern_guard_or_expr"](1) == 1
    assert native_ns["pattern_guard_or_expr"](2) == 1


@pytest.mark.native
def test_pattern_many_arms(native_ns):
    assert native_ns["pattern_many_arms"](1) == 10
    assert native_ns["pattern_many_arms"](2) == 20
    assert native_ns["pattern_many_arms"](3) == 3


@pytest.mark.native
def test_pattern_arm_bindings(native_ns):
    assert native_ns["pattern_arm_bindings"](1) == 1
    assert native_ns["pattern_arm_bindings"](2) == 2
    assert native_ns["pattern_arm_bindings"](3) == 0


@pytest.mark.native
def test_pattern_or_binding_consistency(native_ns):
    assert native_ns["pattern_option_or_bindings"]() == 8


@pytest.mark.native
def test_pattern_or_disjoint_bindings_rejected(native_backend):
    source = """
enum Maybe { Some(i64), None }
fn invalid_or_binding() -> i64 {
    let value = Maybe::Some(1);
    match value {
        Maybe::Some(x) | Maybe::None => 1,
    }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_struct_missing_rest_rejected(native_backend):
    source = """
struct Point { x: i64, y: i64 }
fn invalid_struct_pattern(point: Point) -> i64 {
    match point { Point { x } => x }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_struct_extra_field_rejected(native_backend):
    source = """
struct Point { x: i64, y: i64 }
fn invalid_struct_pattern(point: Point) -> i64 {
    match point { Point { x, y, z } => x + y + z }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_tuple_wrong_arity_rejected(native_backend):
    source = """
fn invalid_tuple_pattern() -> i64 {
    let value = (1, 2, 3);
    match value { (a, b) => a + b }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "error[E004]:" in error.value.pretty()


@pytest.mark.native
def test_pattern_slice_wrong_length(native_ns):
    assert native_ns["pattern_slice_wrong_length"]([1, 2]) == 3
    assert native_ns["pattern_slice_wrong_length"]([1, 2, 3]) == 0


@pytest.mark.native
def test_pattern_nested_or_tuple(native_ns):
    assert native_ns["pattern_nested_or_tuple"]((2, (4, 9))) == 9
    assert native_ns["pattern_nested_or_tuple"]((3, (4, 9))) == 0


def test_python_or_enum_payload_binding():
    program = parse(lex(PYTHON_ONLY_SOURCES["pattern_or_enum_payload"]))
    namespace = compile_rust(program)
    tagged_value = (namespace["PythonMatchOption"].Some, 12)
    assert namespace["pattern_or_enum_payload"](tagged_value) == 12


def test_python_tuple_pattern_wrong_arity_does_not_match():
    program = parse(
        lex(PYTHON_ONLY_SOURCES["pattern_tuple_wrong_arity_python"]),
    )
    namespace = compile_rust(program)
    assert namespace["pattern_tuple_wrong_arity_python"]((1, 2, 3)) == 0


def test_python_or_pattern_binding_mismatch():
    source = """
enum Maybe { Some(i64), None }
fn invalid_or_binding(value: Maybe) -> i64 {
    match value {
        Maybe::Some(x) | Maybe::None => 1,
    }
}
"""
    with pytest.raises(TranslateError, match="or-pattern alternatives"):
        compile_rust(parse(lex(source)))


def test_python_nested_or_guard_parity():
    source = "\n".join([*SOURCES.values(), *PYTHON_ONLY_SOURCES.values()])
    namespace = compile_rust(parse(lex(source)))
    pair = namespace["PatternPair"](5, 2)
    pair_small = namespace["PatternPair"](2, 0)
    cases = [
        ("pattern_or_tuple", ((2, 9),), 9),
        ("pattern_tuple_or", ((2, 9),), 9),
        ("pattern_slice_or_inner", ([2],), 1),
        ("pattern_struct_or_inner", (pair_small,), 1),
        ("pattern_binding_or", (2,), 2),
        ("pattern_guard_or", (2,), 1),
        ("pattern_guard_tuple", ((7, 3),), 4),
        ("pattern_guard_slice", ([5],), 1),
        ("pattern_guard_struct", (pair,), 5),
        ("pattern_guard_binding", (4,), 4),
        ("pattern_guard_or_expr", (1,), 1),
        ("pattern_many_arms", (2,), 20),
        ("pattern_arm_bindings", (2,), 2),
        ("pattern_slice_wrong_length", ([1, 2, 3],), 0),
        ("pattern_nested_or_tuple", ((2, (4, 9)),), 9),
    ]
    for function_name, args, expected in cases:
        assert namespace[function_name](*args) == expected


def test_python_struct_pattern_shape_errors():
    missing = """
struct Point { x: i64, y: i64 }
fn invalid(point: Point) -> i64 {
    match point { Point { x } => x }
}
"""
    extra = """
struct Point { x: i64, y: i64 }
fn invalid(point: Point) -> i64 {
    match point { Point { x, y, z } => x + y + z }
}
"""
    with pytest.raises(TranslateError, match="missing fields"):
        compile_rust(parse(lex(missing)))
    with pytest.raises(TranslateError, match="unknown fields"):
        compile_rust(parse(lex(extra)))


def test_python_box_pattern_parity():
    cases = [
        ("pattern_box_ident", (7,), 7),
        ("pattern_box", (1,), 1),
        ("pattern_box_tuple", ((2, 3),), 5),
        ("pattern_box_nested", (8,), 8),
        ("pattern_box_binding", (3,), 3),
    ]
    source = "\n".join(PYTHON_ONLY_SOURCES.values())
    namespace = compile_rust(parse(lex(source)))
    for function_name, args, expected in cases:
        assert namespace[function_name](*args) == expected


def test_python_pattern_macro_raises_translate_error():
    program = parse(lex(_macro_pattern_source("m!()")))
    with pytest.raises(TranslateError, match="pattern macro m!"):
        compile_rust(program)


@pytest.mark.skip(
    reason="Nested enum payload slices are not represented by the Python enum runtime yet",
)
def test_pattern_slice_enum_context():
    # TODO: add payload-carrying enum values to the Python backend runtime.
    program = parse(lex(PYTHON_ONLY_SOURCES["pattern_slice_enum_context"]))
    compile_rust(program)


@pytest.mark.skip(
    reason="Native matching cannot apply slice patterns to Vec-valued struct fields yet",
)
def test_native_pattern_slice_struct_context():
    # TODO: lower nested Vec field patterns through field-to-slice matching.
    pass


@pytest.mark.skip(
    reason="Nested Vec slice patterns require native nested slice lowering",
)
def test_native_pattern_slice_nested():
    # TODO: lower each nested Vec to a slice before matching.
    pass


def test_pattern_slice_struct_context():
    source = "\n".join(PYTHON_ONLY_SOURCES.values())
    namespace = compile_rust(parse(lex(source)))
    assert namespace["pattern_slice_struct_context"](
        namespace["Grid"]([3, 4]),
    ) == 7


def test_pattern_slice_nested():
    program = parse(lex(PYTHON_ONLY_SOURCES["pattern_slice_nested"]))
    namespace = compile_rust(program)
    assert namespace["pattern_slice_nested"]([[1], [2]]) == 3
    assert namespace["pattern_slice_nested"]([[1], [2, 3]]) == 0


@pytest.mark.native
def test_error_duplicate_binding_native(native_backend):
    source = """
fn duplicate_binding(value: (i64, i64)) -> i64 {
    match value { (x, x) => x }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "bound more than once" in error.value.pretty()


@pytest.mark.native
def test_error_duplicate_binding_or_different_types(native_backend):
    source = """
enum Value { Number(i64), Text(String) }
fn duplicate_or_binding() -> i64 {
    let value = Value::Number(1);
    match value {
        Value::Number(x) | Value::Text(x) => 1,
    }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "mismatched types" in error.value.pretty()


def test_error_rest_outside_slice():
    source = "fn invalid() -> i64 { match 1 { .. => 1 } }"
    with pytest.raises(ParseError, match="rest pattern") as error:
        parse(lex(source))
    assert error.value.line == 1
    assert error.value.col > 0


def test_error_empty_or_pattern():
    source = "fn invalid() -> i64 { match 1 { | => 1 } }"
    with pytest.raises(ParseError, match="expected pattern"):
        parse(lex(source))


def test_error_at_binding_without_inner_pattern():
    source = "fn invalid() -> i64 { match 1 { x @ => 1 } }"
    with pytest.raises(ParseError, match="expected pattern"):
        parse(lex(source))


def test_error_box_without_inner_pattern():
    source = "fn invalid() -> i64 { match 1 { box => 1 } }"
    with pytest.raises(ParseError, match="expected pattern"):
        parse(lex(source))


@pytest.mark.native
def test_error_unresolved_const(native_backend):
    source = """
fn unresolved_constant(value: i64) -> i64 {
    match value { const MISSING => 1, _ => 0 }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "MISSING" in error.value.pretty()
    with pytest.raises(TranslateError, match="unresolved constant pattern"):
        compile_rust(parse(lex(source)))


@pytest.mark.native
def test_error_unsupported_macro(native_backend):
    source = _macro_pattern_source("unsupported!()")
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "unsupported" in error.value.pretty()


def test_error_two_slice_rests():
    source = "fn invalid() -> i64 { match [] { [.., ..] => 1 } }"
    with pytest.raises(ParseError, match="only one rest") as error:
        parse(lex(source))
    assert error.value.line == 1
    assert error.value.col > 0


def test_error_two_struct_rests():
    source = """
struct Point { x: i64 }
fn invalid(point: Point) -> i64 {
    match point { Point { .., .. } => 1 }
}
"""
    with pytest.raises(ParseError, match="only one rest") as error:
        parse(lex(source))
    assert error.value.line > 0
    assert error.value.col > 0


def test_warning_unreachable_arm():
    source = """
fn warning(value: i64) -> i64 {
    match value { _ => 0, 1 => 1 }
}
"""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        compile_rust(parse(lex(source)))
    assert len(captured) == 1
    assert "unreachable match arm" in str(captured[0].message)


def test_warning_duplicate_literal_arm():
    source = """
fn warning(value: i64) -> i64 {
    match value { 1 => 0, 1 => 1, 1 => 2, _ => 3 }
}
"""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        compile_rust(parse(lex(source)))
    assert len(captured) == 1
    assert "duplicate literal" in str(captured[0].message)


def test_warning_guard_shadows_binding():
    source = """
fn warning(value: i64) -> i64 {
    match value { value if value > 0 => 1, _ => 0 }
}
"""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        compile_rust(parse(lex(source)))
    assert len(captured) == 1
    assert "shadows scrutinee" in str(captured[0].message)


@pytest.mark.native
def test_error_non_exhaustive_native_match(native_backend):
    source = """
fn non_exhaustive(value: i64) -> i64 {
    match value { 0 => 0 }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    assert "non-exhaustive patterns" in error.value.pretty()


def test_error_source_location():
    source = "fn invalid() -> i64 {\n    match 1 { .. => 1 }\n}"
    with pytest.raises(ParseError, match="rest pattern") as error:
        parse(lex(source))
    assert (error.value.line, error.value.col) == (2, 15)
    assert "at 2:15" in str(error.value)


def test_error_duplicate_binding_contains_pattern():
    source = """
fn invalid(value: (i64, i64)) -> i64 {
    match value { (x, x) => x }
}
"""
    with pytest.raises(TranslateError) as error:
        compile_rust(parse(lex(source)))
    assert "(x, x)" in str(error.value)
    assert "at: line 3," in str(error.value)


def test_python_unsupported_pattern_has_translate_error():
    with pytest.raises(TranslateError, match="pattern macro unsupported!") as error:
        compile_rust(parse(lex(_macro_pattern_source("unsupported!()"))))
    assert error.value.line == 4
    assert error.value.col > 0


@pytest.mark.native
def test_native_compile_error_pretty_has_pattern(native_backend):
    source = """
fn unresolved(value: i64) -> i64 {
    match value { const NOT_DEFINED => 1, _ => 0 }
}
"""
    with pytest.raises(RustCompileError) as error:
        native_backend.compile(parse(lex(source)))
    pretty = error.value.pretty()
    assert "error[E004]:" in pretty
    assert "NOT_DEFINED" in pretty
    assert error.value.error_line is not None
    assert error.value.error_col is not None


@pytest.mark.parametrize(
    ("source", "expected_type", "message"),
    [
        (
            "fn invalid() -> i64 { match 1 { .. => 1 } }",
            ParseError,
            "rest pattern",
        ),
        (
            (
                "fn invalid(value: (i64, i64)) -> i64 { "
                "match value { (x, x) => x } }"
            ),
            TranslateError,
            "duplicate binding",
        ),
    ],
)
def test_error_type_and_message_contract(source, expected_type, message):
    with pytest.raises(expected_type, match=message):
        compile_rust(parse(lex(source)))


def test_match_warnings_emitted_once_per_expression():
    source = """
fn warning(value: i64) -> i64 {
    match value {
        _ => 0,
        1 => 1,
        1 => 2,
        value if value > 0 => 3,
    }
}
"""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        compile_rust(parse(lex(source)))
    messages = [str(item.message) for item in captured]
    assert sum("unreachable match arm" in msg for msg in messages) == 1
    assert sum("duplicate literal" in msg for msg in messages) == 1
    assert sum("shadows scrutinee" in msg for msg in messages) == 1
    assert len(messages) == 3


def test_python_backend_pattern_parity():
    source = "\n".join([*SOURCES.values(), *PYTHON_ONLY_SOURCES.values()])
    namespace = compile_rust(parse(lex(source)))
    pair = namespace["PatternPair"](3, 4)
    cases = [
        ("pattern_or", (3,), 1),
        ("pattern_range_exclusive", (4,), 0),
        ("pattern_range_inclusive", (4,), 1),
        ("pattern_tuple", ((1, 2),), 12),
        ("pattern_nested_tuple", ((1, (2, 3)),), 123),
        ("pattern_unit_tuple", (), 1),
        ("pattern_single_tuple", ((7,),), 1),
        ("pattern_struct_explicit", (pair,), 1),
        ("pattern_struct_shorthand", (pair,), 3),
        ("pattern_struct_rest", (pair,), 1),
        ("pattern_binding", (4,), 4),
        ("pattern_range_open_start", (3,), 1),
        ("pattern_range_open_end", (3,), 1),
        ("pattern_range_negative", (-2,), 1),
        ("pattern_or_range", (4,), 1),
        ("pattern_tuple_binding", ((1, 8),), 8),
        ("pattern_nested_binding", ((1, 2),), (1, 2)),
        ("pattern_struct_nested", (pair,), 3),
        ("pattern_ref_literal", (4,), 1),
        ("pattern_ref_binding", (9,), 9),
        ("pattern_slice_fixed", ([3, 4],), 7),
        ("pattern_slice_rest_middle", ([1, 2, 3],), 1),
        ("pattern_slice_rest_end", ([1, 2, 3],), 1),
        ("pattern_slice_rest_start", ([1, 2, 3],), 1),
        ("pattern_slice_all_rest", ([1, 2],), 1),
        ("pattern_slice_empty", ([],), 1),
        ("pattern_slice_literals", ([1, 2, 3],), 1),
        ("pattern_slice_bindings", ([4, 5, 6],), 10),
        ("pattern_slice_tuple", ([(1, 2)],), 1),
        ("pattern_slice_or", ([2],), 1),
        ("pattern_slice_range", ([3],), 1),
        ("pattern_slice_ref", ([1],), 1),
        ("pattern_slice_at", ([5],), 5),
        ("pattern_slice_guard", ([2],), 1),
        ("pattern_slice_tuple_context", (), 6),
        ("pattern_slice_rest_binding", ([3, 4, 5],), 5),
        ("pattern_slice_only_rest_binding", ([1, 2],), 2),
    ]

    for function_name, args, expected in cases:
        assert namespace[function_name](*args) == expected

    grid_ns = compile_rust(
        parse(lex(PYTHON_ONLY_SOURCES["pattern_slice_struct_context"])),
    )
    assert grid_ns["pattern_slice_struct_context"](
        grid_ns["Grid"]([3, 4]),
    ) == 7

    nested_ns = compile_rust(
        parse(lex(PYTHON_ONLY_SOURCES["pattern_slice_nested"])),
    )
    assert nested_ns["pattern_slice_nested"]([[1], [2]]) == 3
