# pyright: reportAttributeAccessIssue=false

"""Trait bounds, generic argument preservation, and monomorphization tests."""

from __future__ import annotations

import shutil

import pytest

from backends import get_backend
from backends.jit.errors import JITUnsupportedError
from backends.jit.memory import is_supported_host
from compiler.analysis.monomorph import MonomorphizationError, monomorphize
from compiler.core.ast_nodes import TypeName
from compiler.core.lexer import lex
from compiler.core.parser import parse

requires_apple_silicon = pytest.mark.skipif(
    not is_supported_host(),
    reason="the JIT backend requires macOS on Apple Silicon (arm64)",
)

BACKENDS = [
    "python",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            shutil.which("cargo") is None, reason="cargo is unavailable"
        ),
    ),
]


def _namespace(source: str, backend: str):
    return get_backend(backend, trusted=True).compile(parse(lex(source)))


def _main(source: str, backend: str):
    return _namespace(source, backend)["main"]()


def _names(source: str) -> set[str]:
    program = monomorphize(parse(lex(source)))
    return {item.name for item in program.items if hasattr(item, "name")}


GENERIC_BOX = """
struct Box<T: Clone> { value: T }
impl<T: Clone> Box<T> {
    fn new(value: T) -> Box<T> {
        Box { value: value }
    }
    fn get(&self) -> T {
        self.value.clone()
    }
}
fn main() -> i64 {
    let a: Box<i64> = Box::new(7);
    let b: Box<String> = Box::new("hello".to_string());
    a.get() * 10 + (b.get().len() as i64)
}
"""

IDENTITY = """
fn identity<T>(x: T) -> T { x }
fn main() -> i64 {
    identity::<i64>(20) + identity(21) + 1
}
"""


# ---------------------------------------------------------------------------
# Parser: bounds and generic arguments
# ---------------------------------------------------------------------------


def test_trait_bounds_are_parsed_on_functions():
    program = parse(lex("fn clone_it<T: Display + Clone>(value: T) -> T { value }"))
    function = program.items[0]
    assert function.generics[0].name == "T"
    assert [bound.name for bound in function.generics[0].bounds] == [
        "Display",
        "Clone",
    ]


def test_trait_bounds_are_parsed_on_structs():
    program = parse(lex("struct Wrapper<T: Clone> { value: T }"))
    declaration = program.items[0]
    assert declaration.generics[0].name == "T"
    assert [bound.name for bound in declaration.generics[0].bounds] == ["Clone"]


def test_type_name_keeps_nested_generic_arguments():
    program = parse(lex("fn take(value: Box<Vec<i64>>) -> Box<Vec<i64>> { value }"))
    function = program.items[0]
    outer = function.params[0].ty
    assert isinstance(outer, TypeName)
    assert outer.name == "Box"
    assert isinstance(outer.args[0], TypeName)
    assert outer.args[0].name == "Vec"
    assert outer.args[0].args[0].name == "i64"


def test_turbofish_is_parsed_on_calls():
    program = parse(lex("fn main() -> i64 { identity::<i64>(1) }"))
    call = program.items[0].body.tail
    assert call.func.generic_args[0].name == "i64"


# ---------------------------------------------------------------------------
# Monomorphization pass
# ---------------------------------------------------------------------------


def test_generic_structs_become_concrete_declarations():
    names = _names(GENERIC_BOX)
    assert "Box_i64" in names
    assert "Box_String" in names


def test_monomorphized_ast_has_no_type_parameters():
    program = monomorphize(parse(lex(GENERIC_BOX)))
    for item in program.items:
        assert not getattr(item, "generics", []), type(item).__name__


def test_repeated_instantiations_are_deduplicated():
    names = _names(
        """
fn identity<T>(x: T) -> T { x }
fn main() -> i64 {
    identity::<i64>(1) + identity::<i64>(2) + identity(3)
}
"""
    )
    assert names == {"identity_i64", "main"}


def test_infinite_generic_expansion_hits_the_recursion_limit():
    source = """
fn grow<T>() -> i64 { grow::<Vec<T>>() }
fn main() -> i64 { grow::<i64>() }
"""
    with pytest.raises(MonomorphizationError) as error:
        monomorphize(parse(lex(source)))
    assert error.value.code == "E003"
    assert "recursion limit" in str(error.value)


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_box_i64_and_box_string_parity(backend):
    namespace = _namespace(GENERIC_BOX, backend)
    assert namespace["main"]() == 75
    assert "Box_i64" in namespace
    assert "Box_String" in namespace


@pytest.mark.parametrize("backend", BACKENDS)
def test_generic_identity_explicit_and_inferred(backend):
    assert _main(IDENTITY, backend) == 42


@requires_apple_silicon
def test_generic_identity_runs_in_the_jit():
    assert _main(IDENTITY, "jit") == 42


@requires_apple_silicon
def test_jit_reports_unsupported_aggregate_generics():
    with pytest.raises(JITUnsupportedError):
        _namespace(GENERIC_BOX, "jit")


@pytest.mark.parametrize("backend", BACKENDS)
def test_concrete_struct_literal_without_constructor(backend):
    source = """
struct Pair<T> { first: T, second: T }
fn main() -> i64 {
    let pair: Pair<i64> = Pair { first: 20, second: 22 };
    pair.first + pair.second
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_generic_enum_constructor_and_patterns(backend):
    source = """
enum Maybe<T> { Some(T), None }
fn main() -> i64 {
    let value: Maybe<i64> = Maybe::Some(41);
    match value {
        Maybe::Some(inner) => inner + 1,
        Maybe::None => 0,
    }
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_concrete_and_generic_impls_do_not_collide(backend):
    source = """
struct Box<T> { value: T }
impl<T> Box<T> {
    fn new(value: T) -> Box<T> {
        Box { value: value }
    }
}
impl Box<i64> {
    fn double(&self) -> i64 {
        self.value * 2
    }
}
fn main() -> i64 {
    let boxed: Box<i64> = Box::new(21);
    boxed.double()
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_impl_can_omit_the_generic_parameter_list(backend):
    source = """
struct Box<T> { value: T }
impl Box<T> {
    fn get(&self) -> T {
        self.value.clone()
    }
}
fn main() -> i64 {
    let boxed: Box<i64> = Box { value: 42 };
    boxed.get()
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_generic_instantiations(backend):
    source = """
struct Box<T> { value: T }
fn wrap<T>(value: T) -> Box<T> {
    Box { value: value }
}
fn main() -> i64 {
    let nested: Box<Box<i64>> = wrap(wrap(41));
    nested.value.value + 1
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_multiple_type_parameters(backend):
    source = """
fn count_pair<A, B>(first: A, second: B) -> i64 {
    42
}
fn main() -> i64 {
    count_pair::<i64, String>(1, "two".to_string())
}
"""
    assert _main(source, backend) == 42


@pytest.mark.parametrize("backend", BACKENDS)
def test_impl_on_a_specialized_generic_type(backend):
    source = """
struct Box<T> { value: T }
impl Box<Vec<T>> {
    fn count(&self) -> i64 {
        self.value.len() as i64
    }
}
fn main() -> i64 {
    let boxed: Box<Vec<i64>> = Box { value: vec![1, 2, 3] };
    boxed.count()
}
"""
    assert _main(source, backend) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_recursive_generic_struct_terminates(backend):
    source = """
struct Node<T> {
    value: T,
    children: Vec<Node<T>>,
}
fn main() -> i64 {
    let leaf: Node<i64> = Node { value: 1, children: vec![] };
    let root: Node<i64> = Node { value: 2, children: vec![leaf] };
    root.value + root.children.len() as i64
}
"""
    assert _main(source, backend) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_recursive_generic_enum_terminates(backend):
    source = """
enum Tree<T> {
    Leaf(T),
    Node(Vec<Tree<T>>),
}
fn main() -> i64 {
    let tree: Tree<i64> = Tree::Node(vec![Tree::Leaf(1), Tree::Leaf(2)]);
    match tree {
        Tree::Leaf(value) => value,
        Tree::Node(children) => children.len() as i64,
    }
}
"""
    assert _main(source, backend) == 2


def test_generic_struct_expansion_hits_the_recursion_limit():
    source = """
struct Wrap<T> {
    value: T,
    next: Option<Wrap<Vec<T>>>,
}
fn main() -> i64 { 0 }
"""
    with pytest.raises(MonomorphizationError) as error:
        monomorphize(parse(lex(source)))
    assert error.value.code == "E003"
    assert "recursion limit" in str(error.value)
