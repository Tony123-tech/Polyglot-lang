"""Behavioral tests for Batch 2 of the 1,000-function framework rollout.

The ten legacy FRAME 2 functions receive a compact smoke test because their
full behavior is covered by ``tests/test_framework_f2.py``.  Every new helper
is exercised with a real behavioral assertion, and the structured E001-E008
error paths are checked where they apply.
"""

from __future__ import annotations

import importlib

import pytest

import compiler.core.ast_nodes as ast
from compiler.core.lexer import lex
from compiler.core.parser import parse
from compiler.frontend.errors import PolyglotError
from compiler.types import inference as infer
from compiler.types import lifetimes as life
from compiler.types import monomorphization as mono
from compiler.types import traits as trait
from compiler.types.framework import (
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
from compiler.types.rollout import FRAMEWORK_ROWS

_LEGACY_NAMES = [
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
]
_MODULES = (
    (infer, "infer_", 30, "inference"),
    (trait, "trait_", 20, "traits"),
    (life, "life_", 20, "lifetimes"),
    (mono, "mono_", 20, "monomorphization"),
)


def _program(source: str) -> ast.Program:
    """Parse a Polyglot source string into an AST program."""
    return parse(lex(source))


def _function(source: str, name: str = "main") -> ast.FnDecl:
    """Find a named function declaration in parsed source."""
    return next(
        item
        for item in _program(source).items
        if isinstance(item, ast.FnDecl) and item.name == name
    )


def _error_code(code: str) -> pytest.RaisesExc:
    """Return a pytest context manager checking a specific diagnostic code."""
    return pytest.raises(PolyglotError, match=code)


# ---------------------------------------------------------------------------
# Registry metadata
# ---------------------------------------------------------------------------


def test_rollout_has_one_hundred_resolvable_documented_rows():
    assert len(FRAMEWORK_ROWS) == 100
    assert [row[0] for row in FRAMEWORK_ROWS[:10]] == _LEGACY_NAMES
    assert all(row[3] == "tests.test_types_rollout" for row in FRAMEWORK_ROWS)
    for name, module_name, _category, _test_module in FRAMEWORK_ROWS:
        function = getattr(importlib.import_module(module_name), name)
        assert callable(function)
        assert function.__doc__
    new_names = [row[0] for row in FRAMEWORK_ROWS[10:]]
    assert len(new_names) == len(set(new_names)) == 90
    expected = {
        name
        for module, _prefix, _count, _category in _MODULES
        for name in module.__all__
    }
    assert set(new_names) == expected
    for row in FRAMEWORK_ROWS[:3]:
        assert row[2] == "typechecking"
    for row in FRAMEWORK_ROWS[3:10]:
        assert row[2] == "monomorphization"


def test_each_new_module_has_the_exact_public_surface():
    for module, prefix, count, category in _MODULES:
        assert len(module.__all__) == count
        assert module.__all__ == sorted(module.__all__)
        assert all(name.startswith(prefix) for name in module.__all__)
        categories = {
            row[2] for row in FRAMEWORK_ROWS[10:] if row[1] == module.__name__
        }
        assert categories == {category}


# ---------------------------------------------------------------------------
# Legacy FRAME 2 functions (smoke only)
# ---------------------------------------------------------------------------


def test_legacy_framework_functions_are_callable():
    empty = _program("fn main() -> i64 { 0 }")
    assert typecheck_infer_expressions(empty).diagnostics == ()
    assert typecheck_enforce_trait_bounds(empty).checks == ()
    assert typecheck_detect_occurs_check_cycle([]) == ()
    assert monomorph_cache_placeholder_register({}, "key", "value") == "value"

    enum_program = _program(
        "enum Tree<T> { Leaf(T), Node(Vec<Tree<T>>) }\nfn main() -> i64 { 0 }"
    )
    enum_result = monomorph_resolve_recursive_enum(enum_program, "Tree")
    assert enum_result.has_self_reference()

    struct_program = _program(
        "struct Node<T> { value: T, children: Vec<Node<T>> }\nfn main() -> i64 { 0 }"
    )
    struct_result = monomorph_resolve_recursive_struct(struct_program, "Node")
    assert struct_result.has_self_reference()

    cache: dict[str, int] = {"canonical": 1}
    assert monomorph_unify_cache_keys(cache, [("canonical", "alias")]) == 1
    assert cache["alias"] == 1

    array = ast.ArrayExpr(0, 0, [ast.LitInt(0, 0, 1, "1")])
    propagation = monomorph_propagate_vec_elem_type(array)
    assert ast.type_display_name(propagation.element_type) == "i64"

    pruned = monomorph_prune_dead_specializations(
        _program("fn main() -> i64 { 0 }\nfn dead() -> i64 { 1 }")
    )
    assert pruned.removed == ("dead",)

    report = monomorph_emit_inflation_report(
        _program("fn identity<T>(x: T) -> T { x }\nfn main() -> i64 { identity(1) }")
    )
    assert report.total_instantiations == 1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def test_infer_literal_type_and_unary_operators():
    assert infer.infer_literal_type(7)["type"] == "i64"
    assert infer.infer_literal_type(7)["confidence"] == "default"
    assert infer.infer_literal_type(ast.LitInt(0, 0, 255, "7u8"))["type"] == "u8"
    assert infer.infer_unary_type("&", "i64", lifetime="'a")["type"] == "&'a i64"
    assert infer.infer_unary_type("*", "&mut i64")["type"] == "i64"
    with _error_code("E001"):
        infer.infer_unary_type("!", "i64")
    with _error_code("E003"):
        infer.infer_literal_type(object())


def test_infer_binop_cast_widen_and_coerce():
    assert infer.infer_binop_result("+", "i32", "i64")["type"] == "i64"
    assert infer.infer_binop_result("==", "i64", "i64")["type"] == "bool"
    assert infer.infer_cast_type("i32", "i64")["lossless"] is True
    assert infer.infer_widen("f32", "i64")["type"] == "f64"
    assert infer.infer_coerce("i32", "i64")["kind"] == "widening"
    assert infer.infer_coerce("i64", "String")["code"] == "E001"
    with _error_code("E001"):
        infer.infer_binop_result("+", "i64", "String")
    with _error_code("E003"):
        infer.infer_binop_result("??", "i64", "i64")


def test_infer_expr_type_and_let_binding_reconcile_annotations():
    fn = _function("fn f(x: i64) -> i64 { let y: i64 = x + 1; y }", "f")
    assert infer.infer_expr_type(fn.body, {"x": "i64"})["type"] == "i64"
    let_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.LetStmt))
    binding = infer.infer_let_binding(let_stmt, {"x": "i64"})
    assert binding["name"] == "y"
    assert binding["type"] == "i64"
    bad = _function("fn f() -> i64 { let x: String = 1; 0 }", "f")
    bad_let = next(stmt for stmt in bad.body.stmts if isinstance(stmt, ast.LetStmt))
    with _error_code("E001"):
        infer.infer_let_binding(bad_let)


def test_infer_pattern_bindings_and_function_signature():
    pattern = ast.PatTuple(0, 0, [ast.PatIdent(0, 0, "a"), ast.PatIdent(0, 0, "b")])
    assert infer.infer_pattern_bindings(pattern, "(i64, bool)")["bindings"] == {
        "a": "i64",
        "b": "bool",
    }
    signature = infer.infer_function_signature(
        _function("fn identity<T>(x: T) -> T { x }", "identity")
    )
    assert signature["params"] == [{"name": "x", "type": "T"}]
    assert signature["generics"] == ["T"]
    with _error_code("E001"):
        infer.infer_pattern_bindings(pattern, "i64")


def test_infer_call_signature_and_method_receiver():
    called = infer.infer_call_signature("fn(T) -> T", ["i64"])
    assert called["return"] == "i64"
    assert called["substitution"] == {"T": "i64"}
    assert infer.infer_method_receiver("len", "Vec<i64>")["type"] == "usize"
    assert infer.infer_method_receiver("unwrap", "Option<i64>")["type"] == "i64"
    with _error_code("E001"):
        infer.infer_call_signature("fn(T) -> T", ["i64", "bool"])
    with _error_code("E003"):
        infer.infer_method_receiver("teleport", "i64")


def test_infer_generic_unification_substitution_and_occurs():
    matched = infer.infer_generic_arguments(["Vec<T>"], ["Vec<i64>"])
    assert matched["substitution"] == {"T": "i64"}
    assert infer.infer_unify("T", "Vec<i64>", variables=["T"])["type"] == "Vec<i64>"
    assert (
        infer.infer_substitute("fn(T) -> Vec<T>", {"T": "i64"})["type"]
        == "fn(i64) -> Vec<i64>"
    )
    assert infer.infer_free_variables("fn(T) -> Vec<U>")["variables"] == ["T", "U"]
    assert infer.infer_occurs("T", "Vec<T>")["occurs"] is True
    with _error_code("E005"):
        infer.infer_unify("T", "Vec<T>", variables=["T"])
    with _error_code("E001"):
        infer.infer_unify("i64", "bool")


def test_infer_generalize_instantiate_scheme_and_fresh_names():
    scheme = infer.infer_generalize("fn(T) -> T", env_types=["i64"])
    assert scheme["scheme"] == "forall T. fn(T) -> T"
    assert infer.infer_instantiate(scheme["scheme"])["type"] == "fn(?T0) -> ?T0"
    assert infer.infer_scheme_arity("forall T. fn(T) -> T")["params"] == 1
    assert infer.infer_unknown_fresh(["?T0"])["name"] == "?T1"


def test_infer_join_array_tuple_and_reference_helpers():
    joined = infer.infer_join_branches(["i32", "i64"])
    assert joined["type"] == "i64" and joined["unanimous"]
    assert (
        infer.infer_join_branches(["i64", "String"])["conflicts"][0]["code"] == "E001"
    )
    assert infer.infer_array_element(["i32", "i64"])["array"] == "[i64]"
    assert infer.infer_tuple_elements(["i64", "bool"])["tuple"] == "(i64, bool)"
    depth = infer.infer_reference_depth("&&'a mut i64")
    assert depth["depth"] == 2 and depth["lifetimes"] == ["'a"]


def test_infer_aliases_constraints_context_and_program():
    assert (
        infer.infer_alias_expand("Alias", {"Alias": "Vec<i64>"})["type"] == "Vec<i64>"
    )
    solved = infer.infer_constraint_solve([("T", "i64"), ("U", "T")])
    assert solved["substitution"] == {"T": "i64", "U": "i64"}
    context = infer.infer_type_class_context(
        "i64", ["Clone", "Debug"], {"Clone": ["i64"], "Debug": ["i64"]}
    )
    assert context["satisfied"] == ["Clone", "Debug"]
    program = infer.infer_solve_program(
        _program("fn main() -> i64 { let x: i64 = 1; x }")
    )
    assert program["checked"] == 1
    assert program["functions"]["main"]["bindings"]["x"]["type"] == "i64"
    with _error_code("E005"):
        infer.infer_alias_expand("A", {"A": "B", "B": "A"})
    cyclic = infer.infer_constraint_solve([("T", "Vec<T>")])
    assert cyclic["issues"][0]["code"] == "E005"


def test_infer_expression_fingerprint_is_stable():
    first = infer.infer_expression_fingerprint(ast.LitInt(0, 0, 1, "1"))
    second = infer.infer_expression_fingerprint(ast.LitInt(9, 9, 1, "1"))
    assert first["sha256"] == second["sha256"]
    assert len(first["sha256"]) == 64
    with _error_code("E003"):
        infer.infer_expr_type(ast.LitInt(0, 0, 1, "1"), {"x": 123})


# ---------------------------------------------------------------------------
# Trait helpers
# ---------------------------------------------------------------------------


def test_trait_parse_bounds_and_fingerprints():
    parsed = trait.trait_parse_bound("T: Clone + Into<String>")
    assert [item["name"] for item in parsed["traits"]] == ["Clone", "Into"]
    assert (
        trait.trait_bound_fingerprint(["T: Clone + Debug"])["sha256"]
        == (trait.trait_bound_fingerprint(["T: Debug + Clone"])["sha256"])
    )
    bounds = trait.trait_collect_bounds(
        _function("fn f<T: Clone>(x: T) -> T { x }", "f")
    )
    assert bounds["count"] == 1
    assert bounds["bounds"][0]["param"] == "T"


def test_trait_collect_resolve_and_check_bounds():
    program = _program(
        "trait Marker {}\nimpl Marker for i64 {}\nfn main() -> i64 { 0 }"
    )
    collected = trait.trait_collect_impls(program)
    assert collected["traits"] == ["Marker"]
    resolved = trait.trait_resolve_impl("Marker", "i64", collected["impls"])
    assert resolved["found"] is True
    assert trait.trait_check_bound("T: Marker", "i64", collected["impls"])["satisfied"]
    missing = trait.trait_check_bound("T: Marker", "bool", collected["impls"])
    assert missing["code"] == "E001"
    report = trait.trait_check_program(program)
    assert report["checked"] >= 0
    assert report["code"] == ""


def test_trait_supertraits_cycles_and_linearization():
    declarations = {"A": ["B"], "B": ["C"], "C": []}
    assert trait.trait_supertraits("A", declarations)["ancestors"] == ["B", "C"]
    assert trait.trait_linearize("A", declarations)["order"] == ["C", "B", "A"]
    cyclic = {"A": ["B"], "B": ["A"]}
    assert trait.trait_detect_cycle(cyclic)["cycles"][0]["code"] == "E005"
    with _error_code("E005"):
        trait.trait_supertraits("A", cyclic)
    with _error_code("E005"):
        trait.trait_linearize("A", cyclic)


def test_trait_impl_inspection_completeness_and_safety():
    program = _program(
        'trait Shape { fn area(&self) -> i64; fn name(&self) -> String { "s".to_string() } }'
        "\nimpl Shape for i64 { fn area(&self) -> i64 { 1 } }"
        "\nimpl Shape for bool {}"
    )
    decl = next(item for item in program.items if isinstance(item, ast.TraitDecl))
    impls = [item for item in program.items if isinstance(item, ast.ImplBlock)]
    required = trait.trait_required_methods(decl)
    assert required["required"] == ["area"]
    assert required["defaults"] == ["name"]
    assert trait.trait_default_body_map(decl)["defaults"] == {
        "area": False,
        "name": True,
    }
    assert trait.trait_impl_methods(impls[0])["target"] == "i64"
    completeness = trait.trait_check_impl_completeness(decl, [impls[0], impls[1]])
    assert completeness["complete"] is False
    assert completeness["impls"][1]["missing"] == ["area"]
    unsafe = trait.trait_object_safety(
        next(
            item
            for item in _program("trait Bad { fn make<T>(x: T) -> T; }").items
            if isinstance(item, ast.TraitDecl)
        )
    )
    assert unsafe["unsafe_methods"] == ["make"]
    assert (
        trait.trait_orphan_check(
            {"trait": "foreign::Trait", "target": "foreign::Type"},
            local_crates=["mine"],
        )["code"]
        == "E001"
    )


def test_trait_coherence_substitution_associated_and_dispatch():
    impls = [
        {"trait": "Clone", "target": "Box<i64>", "methods": [{"name": "clone"}]},
        {"trait": "Clone", "target": "Box<i64>", "methods": [{"name": "clone"}]},
    ]
    assert trait.trait_coherence_check(impls)["conflicts"][0]["code"] == "E001"
    assert trait.trait_substitute_bound("T: Into<Vec<T>>", {"T": "i64"})["bound"] == (
        "T: Into<Vec<i64>>"
    )
    assert (
        trait.trait_resolve_associated_type(
            "Iter",
            "Item",
            impls + [{"trait": "Iter", "target": "i64", "associated": {"Item": "u8"}}],
        )["type"]
        == "u8"
    )
    assert trait.trait_resolve_associated_type("Iter", "Missing", [])["code"] == "E006"
    program = _program(
        "trait Marker {}\nimpl Marker for i64 { fn go(&self) -> i64 { 1 } }"
    )
    assert trait.trait_dispatch_table(program)["count"] == 1


# ---------------------------------------------------------------------------
# Lifetime helpers
# ---------------------------------------------------------------------------


def test_life_parse_extract_rewrite_and_fingerprint():
    assert life.life_parse("'a")["bare"] == "a"
    assert life.life_extract("&'a mut Vec<&'b i64>")["lifetimes"] == ["'a", "'b"]
    assert life.life_freeze("&'a mut i64")["type"] == "&'static mut i64"
    assert life.life_erase("&'a mut i64")["type"] == "&mut i64"
    assert life.life_substitute("&'a i64", {"'a": "'static"})["type"] == "&'static i64"
    assert len(life.life_region_fingerprint("&'a i64")["sha256"]) == 64
    with _error_code("E003"):
        life.life_parse("not_a_lifetime")


def test_life_borrow_elision_and_program_check():
    assert life.life_infer_borrow(ast.LitInt(0, 0, 1, "1"))["lifetime"] == "'static"
    borrowed = life.life_infer_borrow(
        ast.Unary(0, 0, "&", ast.Ident(0, 0, "x")), {"x": "'a"}
    )
    assert borrowed["lifetime"] == "'a"
    assert life.life_elide("fn(&'a i64) -> i64")["rule"] == "sole-input"
    program = life.life_check_program(_program("fn f(x: &i64) -> &i64 { x }"))
    assert program["issues"][0]["code"] == "E001"


def test_life_constraints_solution_and_region_choice():
    assert life.life_constraint("'a", "'b")["relation"] == "outlives"
    solution = life.life_solve([("'a", "'b"), ("'b", "'c")])
    assert "'c" in solution["outlives"]["'a"]
    assert life.life_check_constraints([("'a", "'c")], solution["outlives"])["ok"]
    assert life.life_shorten("'a", "'c", solution["outlives"])["valid"]
    assert life.life_unify_regions("'a", "'b", solution["outlives"])["region"] == "'b"
    assert life.life_solve([("'a", "'b"), ("'b", "'a")])["cycles"][0]["code"] == "E005"


def test_life_environment_extend_generativity_liveness_and_drop():
    env = life.life_environment({"p": "&'a i64", "v": "i64"})
    assert env["lifetimes"] == {"p": "'a", "v": "'local"}
    assert life.life_extend(env["lifetimes"], "q", "'static")["added"] is True
    assert life.life_generativity(2, used=["'g2_0"])["name"] == "'g2_1"
    assert life.life_liveness(["let x = 1", "use x", "drop x"])["live"]["x"] == [0, 2]
    assert life.life_drop_order(["a", "b"])["order"] == ["b", "a"]


def test_life_borrow_check_reports_overlapping_mutable_borrows():
    invalid = life.life_borrow_check(["borrow x mut", "borrow x mut", "release x"])
    assert invalid["violations"][0]["code"] == "E001"
    assert life.life_borrow_check(["borrow x mut", "release x", "borrow x"])["ok"]


# ---------------------------------------------------------------------------
# Monomorphization helpers
# ---------------------------------------------------------------------------


def test_mono_keys_mangling_and_fingerprints():
    assert mono.mono_key("Vec", ["i64"])["key"] == "Vec<i64>"
    assert mono.mono_parse_key("Vec<i64, bool>")["args"] == ["i64", "bool"]
    assert mono.mono_fresh_name("Vec", ["i64"])["name"] == "Vec__i64"
    assert len(mono.mono_fingerprint("Vec<i64>")["sha256"]) == 64
    with _error_code("E008"):
        mono.mono_parse_key("123")
    with _error_code("E003"):
        mono.mono_key("", ["i64"])


def test_mono_registration_resolution_substitution_and_params():
    cache: dict[str, str] = {}
    assert mono.mono_register(cache, "Vec<i64>", "first")["created"] is True
    assert mono.mono_register(cache, "Vec<i64>", "second")["value"] == "first"
    assert mono.mono_resolve(cache, "Vec<i64>")["value"] == "first"
    assert mono.mono_substitute("Vec<T>", {"T": "i64"})["type"] == "Vec<i64>"
    assert mono.mono_default_arguments(["T", "U"])["args"] == ["i64", "i64"]
    generic = next(
        item
        for item in _program("struct Box<T> { value: T }").items
        if isinstance(item, ast.StructDecl)
    )
    assert mono.mono_collect_params(generic)["params"] == ["T"]
    with _error_code("E006"):
        mono.mono_resolve(cache, "Vec<bool>")


def test_mono_dependency_topology_cycles_and_pruning():
    graph = {"A": ["B"], "B": []}
    assert mono.mono_dependency_graph(graph)["edges"] == [["A", "B"]]
    assert mono.mono_topological_order(graph)["order"] == ["B", "A"]
    assert mono.mono_detect_cycle(graph)["acyclic"]
    assert mono.mono_prune(graph, ["A"])["kept"] == ["A", "B"]
    cyclic = {"A": ["B"], "B": ["A"]}
    assert mono.mono_detect_cycle(cyclic)["cycles"][0]["code"] == "E005"
    with _error_code("E005"):
        mono.mono_topological_order(cyclic)


def test_mono_recursive_expansion_and_vec_propagation():
    cache: dict[str, object] = {}
    expansion = mono.mono_expand_recursive(
        {
            "name": "Tree",
            "args": ["T"],
            "dependencies": [{"name": "Tree", "args": ["T"]}],
        },
        cache,
    )
    assert expansion["self_references"] == 1
    assert expansion["created"] is True
    assert mono.mono_propagate_vec(["i32", "i64"])["vec"] == "Vec<i64>"
    expected = mono.mono_propagate_vec(["String"], expected="Vec<String>")
    assert expected["element"] == "String"
    conflict = mono.mono_propagate_vec(["String"], expected="Vec<i64>")
    assert conflict["conflicts"][0]["code"] == "E001"


def test_mono_cache_unification_validation_stats_and_inflation():
    cache: dict[str, object] = {"Vec<i64>": "instance"}
    unified = mono.mono_unify_keys(cache, [("Vec<i64>", "Alias<i64>")])
    assert unified["changed"] == 1
    assert cache["Alias<i64>"] == "instance"
    assert mono.mono_cache_stats({"Vec<i64>": {"kind": "placeholder"}})["kinds"] == {
        "placeholder": 1
    }
    invalid = mono.mono_validate_cache(
        {"123": "bad", "Vec<i64>": {"dependencies": ["Missing"]}}
    )
    assert {issue["code"] for issue in invalid["issues"]} == {"E008", "E006"}
    report = mono.mono_inflation_report({"Vec": 9}, max_instantiations=8)
    assert report["warnings"][0]["code"] == "E002-INF1"


def test_mono_program_report_uses_the_real_monomorphizer():
    report = mono.mono_program_report(
        _program("fn identity<T>(x: T) -> T { x }\nfn main() -> i64 { identity(1) }")
    )
    assert "identity" in report["generics"]
    assert report["generated"]["functions"] >= 1
