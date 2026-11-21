"""Behavioral tests for Batch 4: Python backend and CPython interop rollout."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from importlib import import_module
from types import FrameType
from typing import Any

import pytest

import ast_nodes as astn
from backends.python import emit as em
from backends.python import interop as it
from backends.python import parity as pa
from backends.python import runtime_helpers as rt
from backends.python.framework import (
    py_bind_native_extension_module,
    py_emit_macro_expansions,
    py_emit_match_lowering,
    py_emit_module_header,
    py_emit_struct_and_enum_classes,
    py_emit_trait_dispatch_adapters,
    py_format_with_ruff_compatible_style,
    py_guard_frame_and_trace_state,
    py_marshal_result_and_errors,
    py_run_parity_harness,
)
from backends.python.rollout import FRAMEWORK_ROWS
from compiler.frontend.ast_utils import ast_find_by_type
from compiler.frontend.errors import PolyglotError
from lexer import lex
from parser import parse

_SOURCE = """
struct Point { x: i64, y: i64 }
enum Shape { Circle(f64), Empty }
enum Flag { On, Off }
trait Speak { fn say(&self) -> i64; }
impl Point { fn sum(&self) -> i64 { self.x + self.y } }
fn add(a: i64, b: i64) -> i64 { a + b }
fn zero() -> i64 { 2 + 3 * 4 }
fn choose(x: i64) -> i64 { if x > 2 { x * 2 } else { 0 } }
fn dispatch(x: i64) -> i64 { match x { 1 => 10, n => n + 1 } }
"""
_PROGRAM = parse(lex(_SOURCE))
_STRUCT = _PROGRAM.items[0]
_ENUM = _PROGRAM.items[1]
_FLAG = _PROGRAM.items[2]
_TRAIT = _PROGRAM.items[3]
_IMPL = _PROGRAM.items[4]
_ADD = _PROGRAM.items[5]
_ZERO = _PROGRAM.items[6]
_IF = ast_find_by_type(_PROGRAM, "IfExpr")[0]
_MATCH = ast_find_by_type(_PROGRAM, "MatchExpr")[0]
_LEGACY_NAMES = (
    "py_emit_module_header",
    "py_emit_struct_and_enum_classes",
    "py_emit_trait_dispatch_adapters",
    "py_emit_match_lowering",
    "py_emit_macro_expansions",
    "py_bind_native_extension_module",
    "py_marshal_result_and_errors",
    "py_guard_frame_and_trace_state",
    "py_format_with_ruff_compatible_style",
    "py_run_parity_harness",
)


def _expect(code: str, call: Callable[[], Any]) -> None:
    """Assert that *call* raises a PolyglotError with diagnostic *code*."""
    with pytest.raises(PolyglotError) as error:
        call()
    assert error.value.code == code


def _parsed(source: str) -> ast.Module:
    """Parse emitted source so malformed emission is caught immediately."""
    return ast.parse(source)


def _literal(kind: str, value: Any) -> astn.Node:
    """Build a project literal node of a chosen kind."""
    constructors = {
        "int": lambda: astn.LitInt(1, 1, value, str(value)),
        "float": lambda: astn.LitFloat(1, 1, value, str(value)),
        "str": lambda: astn.LitStr(1, 1, value),
        "bool": lambda: astn.LitBool(1, 1, value),
    }
    return constructors[kind]()


# --- rollout metadata -------------------------------------------------------


def test_rollout_has_100_resolvable_documented_rows():
    assert len(FRAMEWORK_ROWS) == 100
    assert tuple(row[0] for row in FRAMEWORK_ROWS[:10]) == _LEGACY_NAMES
    assert [row[2] for row in FRAMEWORK_ROWS[:10]] == [
        "emission",
        "emission",
        "emission",
        "emission",
        "emission",
        "interop",
        "interop",
        "interop",
        "formatting",
        "parity",
    ]
    assert len({row[0] for row in FRAMEWORK_ROWS}) == 100
    modules = {row[1]: import_module(row[1]) for row in FRAMEWORK_ROWS}
    for name, module_name, category, test_module in FRAMEWORK_ROWS:
        assert isinstance((name, module_name, category, test_module), tuple)
        assert test_module == "tests.test_python_rollout"
        function = getattr(modules[module_name], name)
        assert callable(function)
        assert function.__doc__
    assert len({row[1] for row in FRAMEWORK_ROWS[10:]}) == 4


def test_legacy_framework_functions_smoke():
    header = py_emit_module_header(version="4.0-test")
    assert "__polyglot__" in header
    assert "class Point" in py_emit_struct_and_enum_classes([_STRUCT])
    assert "_TRAIT_METHODS" in py_emit_trait_dispatch_adapters([_TRAIT])
    assert "if " in py_emit_match_lowering(_MATCH)
    assert "_macro_result" in py_emit_macro_expansions([])
    binding = py_bind_native_extension_module(
        "no_such_ext_xyz", fallback={"add": lambda left, right: left + right}
    )
    assert binding.functions["add"](1, 2) == 3
    marshalled = py_marshal_result_and_errors(("Ok", 7))
    assert marshalled.ok and marshalled.value == 7
    guard = py_guard_frame_and_trace_state(suspend_trace=False)
    with guard as snapshot:
        assert snapshot.implementation == "cpython"
    assert py_format_with_ruff_compatible_style("x=1") == "x = 1\n"
    report = py_run_parity_harness("fn f() -> i64 { 1 }", functions=["f"])
    assert report.cases == ()


# --- emit.py ---------------------------------------------------------------


def test_pygen_module_header_and_import_block_emit_valid_python():
    first = em.pygen_module_header(
        "demo", imports=["import math"], target_width=32, version="1.2.3"
    )
    assert first == em.pygen_module_header(
        "demo", imports=["import math"], target_width=32, version="1.2.3"
    )
    assert first["module"] == "demo" and first["target_width"] == 32
    assert '"version": "1.2.3"' in first["source"]
    _parsed(first["source"])
    assert "math" in first["bindings"]
    _expect("E001", lambda: em.pygen_module_header(target_width=16))
    imports = em.pygen_import_block(["import math", "from os import sep"])
    assert imports["statements"] == ["from os import sep", "import math"]
    assert imports["bindings"] == ["math", "sep"]
    _expect("E001", lambda: em.pygen_import_block(["from os import *"]))


def test_pygen_struct_enum_trait_and_impl_emitters():
    struct = em.pygen_struct_dataclass(_STRUCT)
    assert struct["class_name"] == "Point"
    assert [field["name"] for field in struct["fields"]] == ["x", "y"]
    assert "slots=True" in struct["source"]
    _parsed(struct["source"])
    tagged = em.pygen_enum_union(_ENUM)
    assert tagged["representation"] == "tagged"
    assert tagged["variants"] == ["Circle", "Empty"]
    assert "def make_Circle" in tagged["source"]
    _parsed(tagged["source"])
    unit = em.pygen_enum_union(_FLAG)
    assert unit["representation"] == "enum" and "auto()" in unit["source"]
    _expect(
        "E001",
        lambda: em.pygen_enum_union(
            astn.EnumDecl(
                1,
                1,
                "Dup",
                [astn.EnumVariant(1, 1, "A", None), astn.EnumVariant(1, 1, "A", None)],
            )
        ),
    )
    protocol = em.pygen_trait_protocol(_TRAIT)
    assert protocol["protocol"] == "Speak"
    assert [method["name"] for method in protocol["methods"]] == ["say"]
    assert "@runtime_checkable" in protocol["source"]
    mixin = em.pygen_impl_methods(_IMPL)
    assert mixin["methods"] == ["sum"] and mixin["target"] == "Point"
    assert "def sum(self)" in mixin["source"]
    _parsed(mixin["source"])
    _expect("E003", lambda: em.pygen_struct_dataclass(_ENUM))


def test_pygen_signature_body_dispatch_and_pattern_guard():
    signature = em.pygen_function_signature(_ADD)
    assert signature["source"] == "def add(a: int, b: int) -> int:"
    assert [param["name"] for param in signature["params"]] == ["a", "b"]
    body = em.pygen_function_body(["acc = a + b"], tail="acc")
    assert body["indent"] == 1 and body["has_tail"]
    _parsed("def f() -> int:\n" + body["source"] + "\n")
    dispatch = em.pygen_match_dispatch(_MATCH)
    assert dispatch["subject"] == "_match_subject"
    assert [arm["bindings"] for arm in dispatch["arms"]] == [[], ["n"]]
    assert "elif True:" in dispatch["source"]
    _parsed("x = 1\n" + dispatch["source"])
    literal_guard = em.pygen_pattern_guard(_MATCH.arms[0].pattern)
    assert literal_guard["source"] == "_subject == 1"
    assert literal_guard["kind"] == "PatLiteral"
    binding_guard = em.pygen_pattern_guard(_MATCH.arms[1].pattern)
    assert binding_guard["bindings"] == ["n"]
    _expect(
        "E002",
        lambda: em.pygen_pattern_guard(astn.PatMacro(1, 1, "m!", "tokens")),
    )
    _expect("E001", lambda: em.pygen_match_dispatch(_MATCH, subject="9bad"))


def test_pygen_literals_types_calls_and_statements():
    assert em.pygen_literal_source(_literal("int", 7))["source"] == "7"
    assert em.pygen_literal_source(_literal("float", 1.5))["kind"] == "float"
    assert em.pygen_literal_source(_literal("str", 'a"b'))["source"] == '"a\\"b"'
    assert em.pygen_literal_source(astn.LitBool(1, 1, True))["source"] == "True"
    _expect(
        "E008",
        lambda: em.pygen_literal_source(astn.LitFloat(1, 1, float("inf"), "inf")),
    )
    vector_type = astn.TypeName(1, 1, "Vec", [astn.TypeName(1, 1, "i64")])
    annotation = em.pygen_type_annotation(vector_type)
    assert annotation["source"] == "list[int]"
    assert annotation["generic_args"] == ["int"]
    assert em.pygen_type_annotation(None)["source"] == "None"
    call = em.pygen_call_expression(
        "fn", arguments=["1", "2"], keywords={"flag": "True"}
    )
    assert call["arity"] == 2 and call["keywords"] == ["flag"]
    assert call["source"] == "fn(1, 2, flag=True)"
    _expect("E001", lambda: em.pygen_call_expression("fn", keywords={"9bad": "1"}))
    assignment = em.pygen_assignment("x", "1 + 2", annotation="int")
    assert assignment["annotated"] and assignment["operator"] == "="
    assert assignment["source"] == "x: int = 1 + 2"
    assert em.pygen_return_statement()["source"] == "return"
    assert em.pygen_return_statement("x")["has_value"]


def test_pygen_control_flow_fields_and_stubs():
    conditional = em.pygen_conditional(
        [("x > 0", ["y = 1"]), ("x < 0", ["y = -1"])], else_lines=["y = 0"]
    )
    assert conditional["branches"] == 2 and conditional["has_else"]
    _parsed(conditional["source"])
    loop = em.pygen_loop_block(
        "for", body=["total += i"], target="i", iterator="range(3)"
    )
    assert loop["kind"] == "for" and "total += i" in loop["source"]
    guarded = em.pygen_exception_guard(
        ["risky()"], handlers=[("ValueError", "err", ["message = str(err)"])]
    )
    assert guarded["handlers"][0]["alias"] == "err" and not guarded["has_finally"]
    _parsed(guarded["source"])
    assert em.pygen_dataclass_field("x", "int", default="1")["has_default"]
    decorator = em.pygen_decorator("dataclass", arguments=["slots=True"])
    assert decorator["arguments"] == ["slots=True"]
    assert decorator["source"] == "@dataclass(slots=True)"
    doc = em.pygen_docstring_block('say """hi"""')
    assert doc["escaped"] and '\\"\\"\\"' in doc["source"]
    constructor = em.pygen_enum_variant_constructor("Shape", _ENUM.variants[0])
    assert constructor["name"] == "make_Circle" and constructor["arity"] == 1
    _parsed(constructor["source"])
    macro = em.pygen_macro_call_source(
        astn.MacroCall(1, 1, "assert!", [_literal("bool", True)])
    )
    assert macro["source"] == "_assert(True)"
    _expect(
        "E001",
        lambda: em.pygen_macro_call_source(astn.MacroCall(1, 1, "assert!", [])),
    )
    _expect(
        "E002",
        lambda: em.pygen_macro_call_source(astn.MacroCall(1, 1, "nope!", [])),
    )
    _expect(
        "E003",
        lambda: em.pygen_macro_call_source(astn.MacroCall(1, 1, "vec!", "raw")),
    )
    exports = em.pygen_module_exports(["b", "a", "b"])
    assert exports["names"] == ["b", "a"] and exports["count"] == 2
    stub = em.pygen_native_binding_stub(
        "native.ext", symbols=["add"], fallback_module="pure"
    )
    assert stub["symbols"] == ["add"] and "def bind_fallback" in stub["source"]
    _parsed(stub["source"])


# --- interop.py ------------------------------------------------------------


def test_interop_abi_descriptors_layouts_and_digest():
    descriptor = it.pyinterop_describe_abi(
        platform_name="darwin", features=["pyo3", "pyo3"]
    )
    assert descriptor["abi"] == "darwin-64"
    assert descriptor["features"] == ["pyo3"]
    assert "pyo3" in descriptor["flags"]
    _expect("E007", lambda: it.pyinterop_describe_abi(platform_name="plan9"))
    symbol = it.pyinterop_validate_symbol_name("native.mod.add")
    assert symbol["module"] == "native.mod" and symbol["attribute"] == "add"
    _expect("E001", lambda: it.pyinterop_validate_symbol_name("bad-name"))
    width = it.pyinterop_abi_pointer_width(32)
    assert width["bytes"] == 4 and width["max_signed"] == 2**31 - 1
    _expect("E001", lambda: it.pyinterop_abi_pointer_width(128))
    layout = it.pyinterop_abi_layout([("head", 1, 1), ("word", 8, 8)])
    assert layout["size"] == 16 and layout["alignment"] == 8
    assert layout["fields"][1]["offset"] == 8
    _expect("E001", lambda: it.pyinterop_abi_layout([("bad", 4, 3)]))
    left = it.pyinterop_describe_abi(platform_name="linux")
    right = it.pyinterop_describe_abi(platform_name="darwin")
    comparison = it.pyinterop_check_abi_compatibility(left, right)
    assert not comparison["compatible"]
    assert any("platform" in reason for reason in comparison["reasons"])
    assert it.pyinterop_check_abi_compatibility(left, left)["compatible"]
    digest = it.pyinterop_symbol_digest("native.mod", ["add"], abi=left)
    assert digest["algorithm"] == "sha256" and len(digest["digest"]) == 64
    assert digest == it.pyinterop_symbol_digest("native.mod", ["add"], abi=left)


def test_interop_binding_planning_and_fallbacks():
    native = {"add": lambda left, right: left + right}
    fallback = {"sub": lambda left, right: left - right}
    plan = it.pyinterop_plan_binding(
        "native.mod",
        requested=["add", "sub", "missing"],
        registry=native,
        fallback=fallback,
        aliases={"math_add": "add"},
    )
    assert plan["native"] == ["add"] and plan["fallback"] == ["sub"]
    assert plan["missing"] == ["missing"]
    assert [entry["source"] for entry in plan["plan"]] == [
        "native",
        "fallback",
        "missing",
    ]
    _expect(
        "E005",
        lambda: it.pyinterop_plan_binding(
            "native.mod", requested=["a"], aliases={"a": "b", "b": "a"}
        ),
    )
    bound = it.pyinterop_bind_callable("add", native["add"], arity=2)
    assert bound["bound"] and bound["arity"] == 2
    _expect("E001", lambda: it.pyinterop_bind_callable("add", native["add"], arity=3))
    _expect("E003", lambda: it.pyinterop_bind_callable("add", 42))
    registry = it.pyinterop_bind_registry(
        native, fallback=fallback, names=["add", "sub", "gone"]
    )
    assert registry["count"] == 2 and registry["missing"] == ["gone"]
    assert registry["bindings"]["sub"]["source"] == "fallback"
    resolved = it.pyinterop_resolve_fallback("sub", native=native, fallback=fallback)
    assert resolved["used_fallback"] and resolved["source"] == "fallback"
    _expect(
        "E006",
        lambda: it.pyinterop_resolve_fallback("gone", native=native, strict=True),
    )
    probe = it.pyinterop_native_module_probe("native.mod", available=["native.mod"])
    assert probe["present"] and probe["origin"] == "inventory"
    _expect(
        "E006",
        lambda: it.pyinterop_native_module_probe("missing.mod", strict=True),
    )


def test_interop_wire_formats_and_manifest():
    scalar = it.pyinterop_marshal_scalar(7)
    assert scalar["kind"] == "int" and scalar["encoded"]["value"] == 7
    assert it.pyinterop_unmarshal_scalar(scalar["encoded"])["value"] == 7
    _expect("E001", lambda: it.pyinterop_marshal_scalar(2**70))
    sequence = it.pyinterop_marshal_sequence([1, "a", True])
    decoded = it.pyinterop_unmarshal_sequence(sequence["encoded"])
    assert sequence["length"] == 3 and decoded["values"] == [1, "a", True]
    _expect(
        "E008",
        lambda: it.pyinterop_unmarshal_sequence([{"type": "int", "value": True}]),
    )
    ok_result = it.pyinterop_marshal_result(("Ok", 5))
    err_result = it.pyinterop_marshal_result(RuntimeError("bad"))
    assert ok_result["tag"] == "ok" and err_result["tag"] == "err"
    assert it.pyinterop_unmarshal_result(ok_result["encoded"])["value"] == 5
    assert it.pyinterop_unmarshal_result(err_result["encoded"])["error"] == "bad"
    _expect("E008", lambda: it.pyinterop_unmarshal_result({"tag": "ok", "payload": []}))
    _expect("E003", lambda: it.pyinterop_marshal_result(1, error_type=str))
    plan = it.pyinterop_call_plan("native.mod.add", arguments=[1, 2], budget=1)
    assert not plan["allowed"] and plan["budget"] == 1
    _expect("E001", lambda: it.pyinterop_call_plan("f", budget=-1))
    translated = it.pyinterop_error_translation("E004", "tool missing")
    assert translated["diagnostic"]["payload"] is None
    assert translated["attribution"] == "cpython-interop"
    _expect("E003", lambda: it.pyinterop_error_translation("E009", "bad"))
    manifest = it.pyinterop_extension_manifest(
        "native.mod", symbols=["add"], registry={"add": len}
    )
    assert (
        manifest["module"] == "native.mod"
        and manifest["manifest"]["fallback_policy"] == "fallback"
    )
    _expect(
        "E004",
        lambda: it.pyinterop_extension_manifest("native.mod", toolchain_error="no cc"),
    )
    _expect(
        "E001",
        lambda: it.pyinterop_extension_manifest("native.mod", fallback_policy="magic"),
    )


# --- parity.py -------------------------------------------------------------


def test_parity_closed_evaluator_values_and_calls():
    assert pa.pyparity_eval_literal(_literal("int", 7))["value"] == 7
    expression = pa.pyparity_eval_expr(_ZERO.body.tail)
    assert expression["value"] == 14 and expression["steps"] == 5
    assert expression["trace"][0]["kind"] == "eval"
    binary = pa.pyparity_eval_binary(_literal("int", 2), "*", _literal("int", 3))
    assert binary["value"] == 6 and binary["operator"] == "*"
    unary = pa.pyparity_eval_unary(_literal("int", 2), "-")
    assert unary["value"] == -2
    call_node = astn.Call(
        1,
        1,
        astn.Ident(1, 1, "abs"),
        [astn.Unary(1, 1, "-", _literal("int", 3))],
    )
    call = pa.pyparity_eval_call(call_node)
    assert call["value"] == 3 and call["function"] == "abs"
    _expect(
        "E002",
        lambda: pa.pyparity_eval_call(
            astn.Call(1, 1, astn.Ident(1, 1, "eval"), [_literal("str", "x")])
        ),
    )
    _expect(
        "E002", lambda: pa.pyparity_eval_expr(astn.Loop(1, 1, astn.Block(1, 1, [])))
    )


def test_parity_blocks_control_flow_and_functions():
    block = astn.Block(
        1,
        1,
        [
            astn.LetStmt(
                1,
                1,
                astn.PatIdent(1, 1, "x"),
                None,
                _literal("int", 4),
            )
        ],
        astn.Ident(1, 1, "x"),
    )
    evaluated = pa.pyparity_eval_block(block)
    assert evaluated["value"] == 4 and evaluated["bindings"]["x"] == 4
    assert pa.pyparity_eval_if(_IF, env={"x": 3})["branch"] == "then"
    assert pa.pyparity_eval_if(_IF, env={"x": 1})["value"] == 0
    matched = pa.pyparity_eval_match(_MATCH, env={"x": 1})
    assert matched["value"] == 10 and matched["arm"] == 0
    assert pa.pyparity_eval_match(_MATCH, env={"x": 4})["value"] == 5
    pattern = pa.pyparity_eval_pattern(_MATCH.arms[1].pattern, 8)
    assert pattern["matched"] and pattern["bindings"] == {"n": 8}
    function = pa.pyparity_eval_function(_ADD, arguments={"a": 2, "b": 5})
    assert function["value"] == 7 and function["function"] == "add"
    program = pa.pyparity_eval_program(_PROGRAM)
    assert program["results"] == {"zero": 14}
    assert "add" in program["functions"]
    _expect(
        "E002",
        lambda: pa.pyparity_eval_function(_ADD, arguments={"a": 1}),
    )
    _expect("E006", lambda: pa.pyparity_eval_program(_PROGRAM, entry="missing"))


def test_parity_comparisons_cases_and_reports():
    assert not pa.pyparity_compare_values(1, 1.0)["equal"]
    assert pa.pyparity_compare_values([1, 2], [1, 2])["equal"]
    assert pa.pyparity_compare_outcomes(
        {"ok": False, "error": ValueError("bad")},
        {"ok": False, "error": "ValueError: bad"},
    )["equal"]
    assert not pa.pyparity_compare_outcomes(
        {"ok": True, "value": 1}, {"ok": False, "error": "x"}
    )["equal"]
    _expect("E003", lambda: pa.pyparity_compare_outcomes({"value": 1}, {"value": 1}))
    case = pa.pyparity_build_case("add", arguments={"b": 3, "a": 2})
    assert case["arguments"] == {"a": 2, "b": 3} and len(case["case_id"]) == 16
    assert case == pa.pyparity_build_case("add", arguments={"a": 2, "b": 3})
    result = pa.pyparity_run_case(_PROGRAM, case)
    assert result["outcome"]["ok"] and result["outcome"]["value"] == 5
    missing = pa.pyparity_run_case(_PROGRAM, {"function": "missing", "arguments": {}})
    assert not missing["outcome"]["ok"] and "E006" in missing["outcome"]["error"]
    other = parse(lex("fn zero() -> i64 { 99 }"))
    zero_case = pa.pyparity_build_case("zero")
    matrix = pa.pyparity_run_matrix(_PROGRAM, [zero_case], reference_program=other)
    assert matrix["total"] == 1 and matrix["mismatched"] == 1
    assert pa.pyparity_run_matrix(_PROGRAM, [zero_case])["matched"] == 1
    report = pa.pyparity_diff_report(
        [{"ok": True, "value": 1}], [{"ok": True, "value": 2}]
    )
    assert report["same"] == 0 and report["differences"][0]["index"] == 0
    assert (
        pa.pyparity_normalize_error(RuntimeError("bad"))["signature"]
        == "RuntimeError: bad"
    )
    _expect("E003", lambda: pa.pyparity_normalize_error(42))
    oracle = pa.pyparity_validate_oracle({"add": len, "bad": 42}, expected=["add"])
    assert oracle["functions"] == ["add"] and oracle["invalid"] == ["bad"]
    assert not oracle["callable"]
    summary = pa.pyparity_trace_summary(result["trace"])
    assert summary["steps"] == result["steps"] and "Binary" in summary["events"][0]


# --- runtime_helpers.py ----------------------------------------------------


def test_runtime_frame_model_respects_pep667():
    snapshot = rt.pyruntime_frame_snapshot(
        {"a": 1}, globals_view={"g": 2}, lineno=7, code_name="demo"
    )
    assert snapshot["locals"] == {"a": 1} and snapshot["writethrough"]
    assert snapshot["code"]["name"] == "demo"
    real_frame: FrameType = sys._getframe()
    from_real = rt.pyruntime_frame_snapshot(real_frame)
    assert isinstance(from_real["locals"], dict) and from_real["version"] == 1
    assert (
        from_real["code"]["name"] == test_runtime_frame_model_respects_pep667.__name__
    )
    view = rt.pyruntime_frame_proxy_view(snapshot)
    assert list(view["view"].keys()) == ["a"] and not view["writethrough"]
    with pytest.raises(TypeError):
        view["view"]["a"] = 9  # type: ignore[index]
    updated = rt.pyruntime_frame_snapshot({"a": 2, "b": 3})
    plan = rt.pyruntime_frame_restore_plan(snapshot, updated)
    assert plan["writes"] == {"a": 2, "b": 3} and plan["count"] == 2
    locals_model: dict[str, Any] = {"a": 1, "old": 0}
    applied = rt.pyruntime_frame_apply_proxied(
        locals_model, plan["writes"], deletes=["old"]
    )
    assert locals_model == {"a": 2, "b": 3}
    assert applied["written"] == ["a", "b"] and applied["deleted"] == ["old"]
    _expect("E006", lambda: rt.pyruntime_frame_apply_proxied({}, {}, deletes=["gone"]))
    _expect(
        "E007",
        lambda: rt.pyruntime_frame_apply_proxied(real_frame, {"a": 2}),
    )
    flags = rt.pyruntime_frame_trace_flags(snapshot)
    assert flags["mask"] == 0 and not flags["trace_enabled"]
    identity = rt.pyruntime_frame_identity(snapshot)
    assert len(identity["id"]) == 16 and identity["components"]["lineno"] == 7
    check = rt.pyruntime_frame_model_check(snapshot)
    assert check["valid"] and not check["frame_type"]
    _expect("E007", lambda: rt.pyruntime_frame_model_check(real_frame))
    _expect("E003", lambda: rt.pyruntime_frame_snapshot([1, 2]))


def test_runtime_option_and_result_tagging():
    some = rt.pyruntime_option_some(9)
    none = rt.pyruntime_option_none()
    assert some["encoded"] == ("Some", 9) and some["is_some"]
    assert rt.pyruntime_option_unwrap(some)["value"] == 9
    assert none["encoded"] == ("None",)
    _expect("E002", lambda: rt.pyruntime_option_unwrap(none))
    _expect("E008", lambda: rt.pyruntime_option_unwrap(("Maybe", 1)))
    ok = rt.pyruntime_result_ok("yes")
    err = rt.pyruntime_result_err(RuntimeError("no"))
    assert ok["encoded"] == ("Ok", "yes") and ok["ok"]
    assert err["encoded"] == ("Err", "no") and not err["ok"]
    assert rt.pyruntime_result_unwrap(ok)["value"] == "yes"
    _expect("E002", lambda: rt.pyruntime_result_unwrap(err))


def test_runtime_checked_integer_helpers():
    cast = rt.pyruntime_int_cast(200, "u8")
    assert cast["value"] == 200 and cast["width"] == 8
    assert rt.pyruntime_int_cast(300, "u8", wrap=True)["value"] == 44
    _expect("E001", lambda: rt.pyruntime_int_cast(300, "u8"))
    _expect("E001", lambda: rt.pyruntime_int_cast(1, "i7"))
    division = rt.pyruntime_int_div(-7, 2)
    assert division["value"] == -3
    assert rt.pyruntime_int_rem(-7, 2)["value"] == -1
    _expect("E002", lambda: rt.pyruntime_int_div(1, 0))
    _expect("E002", lambda: rt.pyruntime_int_rem(1, 0))
    total = rt.pyruntime_checked_add(2, 3, width=8)
    product = rt.pyruntime_checked_mul(2, 3, width=8)
    assert total["value"] == 5 and not total["overflow"]
    assert product["value"] == 6 and product["signed"]
    assert rt.pyruntime_checked_add(127, 1, width=8, wrap=True)["value"] == -128
    _expect("E001", lambda: rt.pyruntime_checked_mul(100, 2, width=8))


def test_runtime_assertions_panic_match_and_enum_codecs():
    assert rt.pyruntime_assert_true(True)["ok"]
    assert rt.pyruntime_assert_eq(2, 2)["comparison"] == "eq"
    assert rt.pyruntime_assert_ne(2, 3)["comparison"] == "ne"
    _expect("E002", lambda: rt.pyruntime_assert_true(False))
    _expect("E002", lambda: rt.pyruntime_assert_eq(1, 2))
    _expect("E002", lambda: rt.pyruntime_assert_ne(1, 1))
    _expect("E002", lambda: rt.pyruntime_panic("boom"))
    _expect("E003", lambda: rt.pyruntime_panic("boom", code="E009"))
    coverage = rt.pyruntime_match_exhaustiveness(["A"], variants=["A", "B"])
    assert not coverage["exhaustive"] and coverage["missing"] == ["B"]
    assert rt.pyruntime_match_exhaustiveness(["A"], wildcard=True)["exhaustive"]
    _expect("E001", lambda: rt.pyruntime_match_exhaustiveness(["A", "A"]))
    encoded = rt.pyruntime_enum_encode("Circle", [2.5])
    assert encoded["encoded"] == ("Circle", 2.5) and encoded["arity"] == 1
    decoded = rt.pyruntime_enum_decode(encoded["encoded"], variants=["Circle"])
    assert decoded["tag"] == "Circle" and decoded["payload"] == (2.5,)
    _expect("E008", lambda: rt.pyruntime_enum_decode(()))
    _expect("E002", lambda: rt.pyruntime_enum_decode(("Other",), variants=["Circle"]))
