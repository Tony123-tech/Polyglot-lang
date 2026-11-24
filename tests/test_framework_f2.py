"""Behavioral tests for FRAME 2: type checking, traits, monomorphization.

Every framework function is exercised against real Polyglot source parsed by
the project's lexer/parser.  Where a passing test is more convincing than an
AST assertion, the program is also compiled and executed through the Python
backend.  No native toolchain or platform-specific feature is required.
"""

from __future__ import annotations

import pytest

import compiler.core.ast_nodes as ast
import framework_functions
from backends import get_backend
from compiler.analysis.monomorph import monomorphize
from compiler.core.lexer import lex
from compiler.core.parser import parse
from compiler.types.framework import (
    TypeConstraint,
    monomorph_cache_placeholder_register,
    monomorph_emit_inflation_report,
    monomorph_propagate_vec_elem_type,
    monomorph_prune_dead_specializations,
    monomorph_resolve_recursive_enum,
    monomorph_resolve_recursive_struct,
    monomorph_unify_cache_keys,
    typecheck_detect_occurs_check_cycle,
    typecheck_enforce_trait_bounds,
    typecheck_infer_expressions,
)
from runtime.diagnostics import PillarError


def _program(source: str) -> ast.Program:
    return parse(lex(source))


def _main(source: str):
    return get_backend("python", trusted=True).compile(_program(source))["main"]()


def _display(ty: ast.Type | None) -> str:
    return ast.type_display_name(ty)


def test_frame_two_registry_resolves_every_callable():
    names = {item.name for item in framework_functions.iter_functions(frame=2)}
    assert names == {
        "typecheck_infer_expressions",
        "typecheck_enforce_trait_bounds",
        "typecheck_detect_occurs_check_cycle",
        "monomorph_cache_placeholder_register",
        "monomorph_resolve_recursive_enum",
        "monomorph_resolve_recursive_struct",
        "monomorph_unify_cache_keys",
        "monomorph_propagate_vec_elem_type",
        "monomorph_prune_dead_specializations",
        "monomorph_emit_inflation_report",
    }
    for name in names:
        assert callable(framework_functions.get_function(name).callable)


# 11 - typecheck_infer_expressions -----------------------------------------


def test_infer_propagates_bindings_calls_and_control_flow():
    source = """
fn identity<T>(x: T) -> T { x }
fn choose(flag: bool) -> i64 {
    let base: i64 = identity(40);
    let text: String = identity("hello".to_string());
    let joined = if flag { base } else { 2 };
    let empty: Vec<String> = vec![];
    joined
}
"""
    program = _program(source)
    result = typecheck_infer_expressions(program)
    assert result.diagnostics == ()
    assert _display(result.binding("base", "choose")) == "i64"
    assert _display(result.binding("text", "choose")) == "String"
    assert _display(result.binding("joined", "choose")) == "i64"
    assert _display(result.binding("empty", "choose")) == "Vec<String>"

    choose = next(
        item for item in program.items if getattr(item, "name", None) == "choose"
    )
    lets = [stmt for stmt in choose.body.stmts if isinstance(stmt, ast.LetStmt)]
    assert [_display(result.type_of(stmt.value)) for stmt in lets] == [
        "i64",
        "String",
        "i64",
        "Vec<String>",
    ]
    assert _display(result.type_of(choose.body.tail)) == "i64"
    assert result.type_of(None) is None


def test_infer_reports_control_flow_type_mismatch():
    program = _program(
        'fn main(flag: bool) -> i64 { if flag { 1 } else { "x".to_string() } }'
    )
    result = typecheck_infer_expressions(program)
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["E001"]
    assert "type mismatch" in result.diagnostics[0].message
    assert result.diagnostics[0].line == 1


# 12 - typecheck_enforce_trait_bounds --------------------------------------


def test_enforce_trait_bounds_accepts_satisfied_and_flags_missing_impls():
    source = """
trait Pretty {}
impl Pretty for i64 {}
struct Box<T: Pretty> { value: T }
fn show<U: Pretty>(x: U) -> i64 { 0 }
fn main() -> i64 {
    let good: Box<i64> = Box { value: 1 };
    show::<i64>(1);
    let bad: Box<String> = Box { value: "x".to_string() };
    0
}
"""
    report = typecheck_enforce_trait_bounds(_program(source), assume_builtins=False)
    assert len(report.checks) == 3
    assert len(report.satisfied) == 2
    assert [(check.item, check.parameter) for check in report.satisfied] == [
        ("Box", "T"),
        ("show", "U"),
    ]
    violations = report.violations
    assert len(violations) == 1
    assert violations[0].item == "Box"
    assert violations[0].trait_name == "Pretty"
    assert violations[0].argument_text == "String"
    assert violations[0].code == "E001"
    assert report.as_dict()["violations"][0]["argument"] == "String"


def test_enforce_trait_bounds_propagates_through_builtin_containers():
    source = """
struct Wrap<T: Clone> { value: T }
fn main() -> i64 {
    let wrapped: Wrap<Vec<i64>> = Wrap { value: vec![1] };
    0
}
"""
    report = typecheck_enforce_trait_bounds(_program(source))
    assert len(report.checks) == 1
    assert report.violations == ()
    assert report.checks[0].argument_text == "Vec<i64>"


# 13 - typecheck_detect_occurs_check_cycle ---------------------------------


def test_occurs_check_direct_indirect_reflexive_and_acyclic_cases():
    ty = ast.TypeName
    constraints = [
        TypeConstraint("A", ty(0, 0, "B"), origin="let a"),
        TypeConstraint("B", ty(0, 0, "Vec", [ty(0, 0, "A")]), origin="let b"),
        TypeConstraint("T", ty(0, 0, "Vec", [ty(0, 0, "T")]), origin="let t"),
        TypeConstraint("V", ty(0, 0, "V"), origin="tautology"),
        TypeConstraint("W", ty(0, 0, "i64"), origin="let w"),
    ]
    diagnostics = typecheck_detect_occurs_check_cycle(constraints)
    assert [(item.code, item.cycle) for item in diagnostics] == [
        ("E005-OC1", ("T",)),
        ("E005-OC2", ("A", "B")),
    ]
    assert diagnostics[0].variable == "T"
    assert "occurs inside Vec<T>" in diagnostics[0].message
    assert diagnostics[1].origin == "let a"
    assert all(item.variable not in {"V", "W"} for item in diagnostics)


def test_occurs_check_deduplicates_cycles_seen_from_multiple_variables():
    ty = ast.TypeName
    constraints = [
        TypeConstraint("A", ty(0, 0, "Vec", [ty(0, 0, "B")])),
        TypeConstraint("B", ty(0, 0, "A")),
        TypeConstraint("C", ty(0, 0, "B")),
    ]
    diagnostics = typecheck_detect_occurs_check_cycle(constraints)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E005-OC2"
    assert diagnostics[0].cycle == ("A", "B")


# 14 - monomorph_cache_placeholder_register --------------------------------


def test_placeholder_register_keeps_first_instance_for_recursive_expansion():
    key = ("enum", "Tree", ("i64",))
    cache: dict[tuple[str, str, tuple[str, ...]], ast.Item] = {}
    placeholder = ast.EnumDecl(0, 0, "Tree_i64", [])
    registered = monomorph_cache_placeholder_register(cache, key, placeholder)
    assert registered is placeholder

    registered.variants.append(
        ast.EnumVariant(
            0,
            0,
            "Node",
            [ast.TypeName(0, 0, "Vec", [ast.TypeName(0, 0, registered.name)])],
        )
    )
    duplicate = ast.EnumDecl(0, 0, "Tree_i64_duplicate", [])
    assert monomorph_cache_placeholder_register(cache, key, duplicate) is placeholder
    assert len(cache) == 1
    field_type = registered.variants[0].fields[0]
    assert isinstance(field_type, ast.TypeName)
    assert _display(field_type.args[0]) == "Tree_i64"


# 15 - monomorph_resolve_recursive_enum ------------------------------------


def test_recursive_enum_resolution_survives_default_and_explicit_arguments():
    source = """
enum Tree<T> { Leaf(T), Node(Vec<Tree<T>>) }
fn main() -> i64 {
    let tree: Tree<i64> = Tree::Node(vec![Tree::Leaf(1), Tree::Leaf(2)]);
    match tree {
        Tree::Leaf(value) => value,
        Tree::Node(children) => children.len() as i64,
    }
}
"""
    program = _program(source)
    default = monomorph_resolve_recursive_enum(program, "Tree")
    assert default.concrete_name == "Tree"
    assert default.has_self_reference()
    assert "Vec" in default.referenced_names
    enum_decl = next(
        item for item in default.declarations if isinstance(item, ast.EnumDecl)
    )
    node = next(variant for variant in enum_decl.variants if variant.name == "Node")
    assert _display(node.fields[0]) == "Vec<Tree>"

    explicit = monomorph_resolve_recursive_enum(
        program, "Tree", [ast.TypeName(0, 0, "String")]
    )
    assert explicit.concrete_name == "Tree_String"
    assert explicit.has_self_reference()
    assert "Tree_String" in explicit.referenced_names
    explicit_enum = next(
        item for item in explicit.declarations if isinstance(item, ast.EnumDecl)
    )
    explicit_node = next(
        variant for variant in explicit_enum.variants if variant.name == "Node"
    )
    assert _display(explicit_node.fields[0]) == "Vec<Tree_String>"
    assert _main(source) == 2


def test_recursive_enum_resolution_reports_unknown_name():
    with pytest.raises(PillarError) as error:
        monomorph_resolve_recursive_enum(_program("fn main() -> i64 { 0 }"), "Missing")
    assert error.value.code == "E003"
    assert "Missing" in error.value.reason


# 16 - monomorph_resolve_recursive_struct ----------------------------------


def test_recursive_struct_resolution_and_python_backend():
    source = """
struct Node<T> { value: T, children: Vec<Node<T>> }
fn main() -> i64 {
    let leaf: Node<i64> = Node { value: 1, children: vec![] };
    let root: Node<i64> = Node { value: 2, children: vec![leaf] };
    root.value + root.children.len() as i64
}
"""
    resolution = monomorph_resolve_recursive_struct(_program(source), "Node")
    assert resolution.concrete_name == "Node"
    assert resolution.has_self_reference()
    assert any(isinstance(item, ast.StructDecl) for item in resolution.declarations)
    assert _main(source) == 3


def test_mutually_recursive_generic_structs_both_resolve():
    source = """
struct A<T> { other: Vec<B<T>>, value: T }
struct B<T> { back: Vec<A<T>>, value: T }
fn main() -> i64 { 7 }
"""
    program = _program(source)
    left = monomorph_resolve_recursive_struct(program, "A")
    right = monomorph_resolve_recursive_struct(program, "B")
    assert {left.concrete_name, right.concrete_name} <= (
        left.referenced_names | right.referenced_names
    )
    assert "B_i64" in left.referenced_names
    assert "A" in right.referenced_names
    left_struct = next(
        item
        for item in left.declarations
        if isinstance(item, ast.StructDecl) and item.name == left.concrete_name
    )
    right_struct = next(
        item
        for item in right.declarations
        if isinstance(item, ast.StructDecl) and item.name == right.concrete_name
    )
    left_fields = {name: ty for name, ty, _pub in left_struct.fields}
    right_fields = {name: ty for name, ty, _pub in right_struct.fields}
    assert _display(left_fields["other"]) == "Vec<B_i64>"
    assert _display(right_fields["back"]) == "Vec<A>"
    assert _main(source) == 7


# 17 - monomorph_unify_cache_keys ------------------------------------------


def test_unify_cache_keys_merges_alias_chains_and_keeps_canonical_value():
    canonical = ("struct", "Box", ("i64",))
    alias = ("struct", "Box", ("alias",))
    chained = ("struct", "Box", ("chain",))
    canonical_item = ast.StructDecl(0, 0, "Box_i64", [])
    duplicate = ast.StructDecl(0, 0, "Box_i64_duplicate", [])
    cache: dict[tuple[str, str, tuple[str, ...]], ast.Item] = {
        canonical: canonical_item,
        alias: duplicate,
    }
    changed = monomorph_unify_cache_keys(cache, [(canonical, alias), (alias, chained)])
    assert changed == 2
    assert cache[canonical] is canonical_item
    assert cache[alias] is canonical_item
    assert cache[chained] is canonical_item


# 18 - monomorph_propagate_vec_elem_type ------------------------------------


def test_vec_propagation_uses_annotations_nesting_and_reports_conflicts():
    source = """
fn main() -> i64 {
    let empty: Vec<String> = vec![];
    let nested = vec![vec![1], vec![2]];
    let mixed = vec![1, "x".to_string()];
    0
}
"""
    program = _program(source)
    main = program.items[0]
    lets = [stmt for stmt in main.body.stmts if isinstance(stmt, ast.LetStmt)]
    assert len(lets) == 3

    empty = monomorph_propagate_vec_elem_type(lets[0].value, lets[0].ty)
    assert empty.element_type is not None
    assert _display(empty.vec_type) == "Vec<String>"
    assert empty.conflicts == ()

    nested = monomorph_propagate_vec_elem_type(lets[1].value)
    assert _display(nested.vec_type) == "Vec<Vec<i64>>"

    mixed = monomorph_propagate_vec_elem_type(lets[2].value)
    assert _display(mixed.vec_type) == "Vec<i64>"
    assert mixed.conflicts == ("E001: element 1 has type String, expected i64",)

    env = {"names": ast.TypeName(0, 0, "String")}
    from_env = monomorph_propagate_vec_elem_type(
        _program("fn f() { vec![names] }").items[0].body.tail, env=env
    )
    assert _display(from_env.vec_type) == "Vec<String>"


# 19 - monomorph_prune_dead_specializations ---------------------------------


def test_prune_dead_specializations_keeps_executable_program():
    source = """
fn identity<T>(x: T) -> T { x }
fn other<T>(x: T) -> T { x }
fn main() -> i64 {
    let dead = other("x".to_string());
    identity(42)
}
"""
    program = monomorphize(_program(source))
    main = next(
        item
        for item in program.items
        if isinstance(item, ast.FnDecl) and item.name == "main"
    )
    main.body.stmts = [
        stmt
        for stmt in main.body.stmts
        if not (
            isinstance(stmt, ast.LetStmt)
            and getattr(stmt.pattern, "name", "") == "dead"
        )
    ]
    assert "other_String" in {getattr(item, "name", "") for item in program.items}

    result = monomorph_prune_dead_specializations(program)
    assert result.removed == ("other_String",)
    assert set(result.kept) == {"main", "identity_i64"}
    assert "other_String" not in {
        getattr(item, "name", "") for item in result.program.items
    }
    assert get_backend("python", trusted=True).compile(result.program)["main"]() == 42
    assert result.as_dict()["removed"] == ["other_String"]


# 20 - monomorph_emit_inflation_report -------------------------------------


def test_inflation_report_counts_specializations_and_flags_growth():
    source = """
fn pick<T>(x: T) -> T { x }
fn main() -> i64 { pick(1) + pick::<String>("x".to_string()).len() as i64 }
"""
    report = monomorph_emit_inflation_report(
        _program(source), max_instantiations=1, max_growth_nodes=0
    )
    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.family == "pick"
    assert entry.instantiations == 2
    assert entry.generic_nodes > 0
    assert entry.specialized_nodes > entry.generic_nodes
    assert entry.growth_nodes > 0
    assert report.total_instantiations == 2
    assert report.total_specialized_nodes == entry.specialized_nodes
    assert report.total_growth_nodes == entry.growth_nodes
    assert [warning.split(":")[0] for warning in report.warnings] == [
        "E002-INF1",
        "E002-INF2",
    ]
    summary = report.as_dict()
    assert summary["families"][0]["family"] == "pick"
    assert summary["warnings"] == list(report.warnings)


def test_inflation_report_is_empty_for_non_generic_source():
    report = monomorph_emit_inflation_report(_program("fn main() -> i64 { 1 }"))
    assert report.entries == ()
    assert report.warnings == ()
    assert report.as_dict()["total_instantiations"] == 0
