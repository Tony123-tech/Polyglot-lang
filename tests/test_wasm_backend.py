"""WASM backend tests: WAT generation and wasmtime execution."""

from __future__ import annotations

import pytest

from backends import get_backend
from backends.wasm.codegen import WasmCodegen, WasmUnsupportedError
from lexer import lex
from parser import parse


def _compile(source: str) -> dict:
    pytest.importorskip("wasmtime")
    program = parse(lex(source))
    return get_backend("wasm", trusted=True).compile(program)


def test_add_function_runs_in_wasmtime() -> None:
    namespace = _compile("fn add(a: i64, b: i64) -> i64 { a + b }")
    assert namespace["add"](2, 3) == 5


def test_recursive_function_and_if_expression() -> None:
    source = "fn fact(n: i64) -> i64 { if n <= 1 { 1 } else { n * fact(n - 1) } }"
    namespace = _compile(source)
    assert namespace["fact"](5) == 120


def test_while_loop_and_compound_assignment() -> None:
    source = """
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total += i;
        i += 1;
    }
    total
}
"""
    namespace = _compile(source)
    assert namespace["sum_to"](5) == 10


def test_bool_arguments_and_logical_operators() -> None:
    source = """
fn even_positive(x: i64) -> bool { x > 0 && x % 2 == 0 }
fn flip(x: bool) -> bool { !x }
"""
    namespace = _compile(source)
    assert namespace["even_positive"](4) is True
    assert namespace["even_positive"](3) is False
    assert namespace["flip"](True) is False


def test_early_return_and_void_function() -> None:
    source = """
fn abs(x: i64) -> i64 {
    if x < 0 {
        return -x;
    }
    x
}
fn noop() { let unused: i64 = 1; }
fn main() { noop(); }
"""
    namespace = _compile(source)
    assert namespace["abs"](-7) == 7
    assert namespace["main"]() is None


def test_codegen_emits_wat_module() -> None:
    program = parse(lex("fn main() -> i64 { 1 + 2 }"))
    wat = WasmCodegen().generate(program)

    assert wat.startswith("(module\n")
    assert '(export "main")' in wat
    assert "i64.add" in wat


def test_unsupported_aggregate_abi_raises_with_position() -> None:
    program = parse(lex("struct Point { x: i64 }\nfn get(p: Point) -> i64 { p.x }"))

    with pytest.raises(WasmUnsupportedError) as info:
        WasmCodegen().generate(program)

    assert "unsupported parameter type 'Point'" in str(info.value)
    assert info.value.line == 2


def test_struct_literal_is_specialized_into_scalar_wat() -> None:
    source = """
struct Point { x: i64 }
fn main() -> i64 {
    let p: Point = Point { x: 7 };
    p.x + 1
}
"""
    namespace = _compile(source)
    assert namespace["main"]() == 8


def test_vec_and_enum_aggregates_are_specialized() -> None:
    source = """
enum Tree { Leaf(i64), Node(Vec<i64>) }
fn main() -> i64 {
    let nested: Vec<Vec<i64>> = vec![vec![4, 2], vec![4]];
    let tree: Tree = Tree::Node(vec![3, 1]);
    match tree {
        Tree::Leaf(value) => value,
        Tree::Node(children) => children.len() as i64 + nested.len() as i64,
    }
}
"""
    namespace = _compile(source)
    assert namespace["main"]() == 4


def test_bounded_generic_method_is_inlined() -> None:
    source = """
struct Box<T: Clone> { value: T }
impl<T: Clone> Box<T> {
    fn new(value: T) -> Box<T> { Box { value: value } }
    fn get(&self) -> T { self.value.clone() }
}
fn main() -> i64 {
    let item: Box<i64> = Box::new(9);
    item.get() + 1
}
"""
    namespace = _compile(source)
    assert namespace["main"]() == 10


def test_build_writes_wat_file(tmp_path, capsys) -> None:
    from api.cli import main

    source = tmp_path / "add.poly"
    source.write_text("fn add(a: i64, b: i64) -> i64 { a + b }\n", encoding="utf-8")

    assert main(["build", str(source), "--backend", "wasm"]) == 0

    artifact = tmp_path / "add.wat"
    assert artifact.exists()
    assert '(export "add")' in artifact.read_text(encoding="utf-8")
    assert str(artifact) in capsys.readouterr().out
