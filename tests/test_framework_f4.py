"""Behavioral tests for FRAME 4: Python backend and CPython interop."""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from typing import Any

import pytest

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
from backends.python.translator import RuntimeFailure, _Result, _ResultErrSentinel
from compiler.core.ast_nodes import (
    EnumDecl,
    ImplBlock,
    LitBool,
    LitInt,
    LitStr,
    MacroCall,
    StructDecl,
    TraitDecl,
)
from compiler.core.lexer import lex
from compiler.core.parser import parse
from framework_functions import get_function

SOURCES = {
    "f4_parity_native": """
fn f4_native_add(a: i64, b: i64) -> i64 { a + b }
fn f4_native_fallible(flag: bool) -> Result<i64, String> {
    if flag { Ok(11) } else { Err("f4 missing") }
}
""",
}


def _program(source: str):
    """Parse trusted Polyglot source into an AST."""
    return parse(lex(source))


def _items(program: Any, *kinds: type):
    """Return program items whose type is one of ``kinds``."""
    return [item for item in program.items if isinstance(item, kinds)]


def _exec_generated(body: str) -> dict[str, Any]:
    """Execute generated source after the standard FRAME 4 header."""
    namespace: dict[str, Any] = {}
    source = py_emit_module_header(version="4.0-test") + "\n" + body
    exec(compile(source, "<frame4-generated>", "exec"), namespace)  # noqa: S102
    return namespace


# --- 31: module header ------------------------------------------------------


def test_module_header_is_deterministic_and_exposes_version_and_runtime_hooks():
    kwargs = {
        "module_name": "f4_module",
        "version": "4.0.0",
        "target_width": 32,
        "imports": ["import math"],
    }
    first = py_emit_module_header(**kwargs)
    assert first == py_emit_module_header(**kwargs)
    assert '"version": "4.0.0"' in first
    assert '"target_width": 32' in first
    namespace: dict[str, Any] = {}
    exec(compile(first, "<frame4-header>", "exec"), namespace)  # noqa: S102
    assert namespace["__polyglot__"]["module"] == "f4_module"
    assert namespace["_cast"](5, "i64") == 5
    assert namespace["_Result"].from_ok(3).is_ok()
    with pytest.raises(ValueError, match="target_width"):
        py_emit_module_header(target_width=16)
    with pytest.raises(ValueError, match="unsupported import"):
        py_emit_module_header(imports=["import os; import sys"])


# --- 32: structs and enums --------------------------------------------------


def test_struct_and_enum_classes_keep_slots_and_tagged_union_semantics():
    program = _program(
        """
        struct Point { x: i64, y: i64 }
        impl Point { fn sum(&self) -> i64 { self.x + self.y } }
        enum Shape { Circle(f64), Rect(f64, f64), Empty }
        """
    )
    source = py_emit_struct_and_enum_classes(
        _items(program, StructDecl, EnumDecl),
        impls=_items(program, ImplBlock),
    )
    namespace = _exec_generated(source)
    point = namespace["Point"](2, 3)
    assert point.sum() == 5
    assert not hasattr(point, "__dict__")
    circle = namespace["Shape"].make_Circle(2.5)
    assert circle[0] is namespace["Shape"].Circle
    assert circle[1] == 2.5
    assert namespace["Shape"].is_Circle(circle)
    assert not namespace["Shape"].is_Rect(circle)
    decoded = namespace["Shape"].decode(circle)
    assert decoded.tag == "Circle"
    assert decoded.payload == (2.5,)
    empty = namespace["Shape"].make_Empty()
    assert empty is namespace["Shape"].Empty
    assert namespace["Shape"].is_Empty(empty)


# --- 33: trait dispatch -----------------------------------------------------


def test_trait_dispatch_adapters_use_protocols_bounds_and_impl_tables():
    program = _program(
        """
        trait Drawable { fn draw(&self) -> str; }
        trait Printable<T: Drawable> { fn render(&self) -> str; }
        struct Widget { label: String }
        impl Drawable for Widget { fn draw(&self) -> str { self.label } }
        """
    )
    source = py_emit_struct_and_enum_classes(
        _items(program, StructDecl),
        impls=_items(program, ImplBlock),
    )
    source += "\n\n" + py_emit_trait_dispatch_adapters(
        _items(program, TraitDecl),
        impls=_items(program, ImplBlock),
    )
    namespace = _exec_generated(source)
    widget = namespace["Widget"]("hello")
    assert namespace["supports_trait"]("Drawable", widget)
    assert namespace["dispatch_trait"]("Drawable", widget, "draw") == "hello"
    assert namespace["_TRAIT_BOUNDS"]["Printable"] == ("Drawable",)
    assert namespace["_TRAIT_IMPLS"]["Drawable"]["Widget"] == ("draw",)
    with pytest.raises(KeyError, match="unknown trait"):
        namespace["dispatch_trait"]("Missing", widget, "draw")
    with pytest.raises(TypeError, match="not a method"):
        namespace["dispatch_trait"]("Drawable", widget, "missing")


# --- 34: match lowering -----------------------------------------------------


def test_match_lowering_binds_before_guards_and_handles_result_variants():
    classifier = _program(
        "fn classify(x: i64) -> i64 { "
        "match x { 0 => 10, n if n > 2 => n * 2, _ => -1 } }"
    ).items[0]
    body = py_emit_match_lowering(classifier.body.tail)
    assert "if n > 2:" in body
    namespace = _exec_generated(
        "def classify(x: int) -> int:\n" + textwrap.indent(body, "    ")
    )
    assert [namespace["classify"](value) for value in (0, 1, 3)] == [10, -1, 6]

    picker = _program(
        "fn pick(value: Result<i64, String>) -> i64 { "
        "match value { Ok(n) => n, Err(_error) => -1 } }"
    ).items[0]
    pick_body = py_emit_match_lowering(picker.body.tail)
    picker_ns = _exec_generated(
        "def pick(value: object) -> int:\n" + textwrap.indent(pick_body, "    ")
    )
    assert picker_ns["pick"](_Result.from_ok(7)) == 7
    assert picker_ns["pick"](_Result.from_err("bad")) == -1


# --- 35: macro expansion ----------------------------------------------------


def test_macro_expansions_produce_executable_python_expressions():
    calls = [
        MacroCall(1, 0, "vec", "1, 2 + 3"),
        MacroCall(
            1,
            0,
            "format",
            [LitStr(1, 0, "n={}"), LitInt(1, 0, 7, "7")],
        ),
        MacroCall(1, 0, "assert", [LitBool(1, 0, True)]),
    ]
    namespace = _exec_generated(py_emit_macro_expansions(calls))
    assert namespace["_macro_result_0"] == [1, 5]
    assert namespace["_macro_result_1"] == "n=7"
    assert namespace["_macro_result_2"] is None
    assert len(namespace["_macro_result_all"]) == 3

    failing = py_emit_macro_expansions(
        [MacroCall(1, 0, "assert", [LitBool(1, 0, False)])]
    )
    with pytest.raises(RuntimeFailure):
        exec(  # noqa: S102
            compile(
                py_emit_module_header() + "\n" + failing,
                "<frame4-failing-macro>",
                "exec",
            ),
            {},
        )


# --- 36: native extension binding ------------------------------------------


def test_native_binding_uses_fallback_and_reports_missing_symbols():
    fallback = {"double": lambda value: value * 2}
    binding = py_bind_native_extension_module(
        "_f4_native_module_that_does_not_exist",
        fallback=fallback,
        required=["double"],
    )
    assert binding.loaded is False
    assert binding.reason is not None
    assert binding.fallbacks == ("double",)
    assert binding.call("double", 4) == 8
    assert binding["double"] is fallback["double"]

    loaded = py_bind_native_extension_module("operator", required=["add"])
    assert loaded.loaded is True
    assert loaded.call("add", 2, 3) == 5
    with pytest.raises(RuntimeError, match="missing required symbols"):
        py_bind_native_extension_module(
            "_f4_native_module_that_does_not_exist",
            required=["absent"],
        )


# --- 37: result and error marshalling --------------------------------------


def test_marshal_result_and_errors_converts_variants_to_structured_values():
    ok = py_marshal_result_and_errors(_Result.from_ok(7))
    assert ok.ok and ok.variant == "Ok" and ok.unwrap() == 7

    error = py_marshal_result_and_errors(("Err", "bad"))
    assert not error.ok and error.variant == "Err"
    assert isinstance(error.error, RuntimeError)
    assert error.error.polyglot_payload == "bad"
    with pytest.raises(RuntimeError, match="bad"):
        error.unwrap()

    sentinel = py_marshal_result_and_errors(
        _ResultErrSentinel(_Result.from_err("sent"))
    )
    assert not sentinel.ok and str(sentinel.error) == "sent"
    native_error = py_marshal_result_and_errors(ValueError("native"))
    assert native_error.variant == "ValueError" and not native_error.ok
    some = py_marshal_result_and_errors(("Some", 4), variant="Custom")
    assert some.ok and some.variant == "Custom" and some.value == 4
    none = py_marshal_result_and_errors(("None",))
    assert none.ok and none.value is None and none.variant == "None"


# --- 38: frame and trace guard ---------------------------------------------


def test_frame_guard_suspends_and_restores_trace_state_on_all_paths():
    def tracer(frame, event, arg):
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        with py_guard_frame_and_trace_state() as snapshot:
            secret = "do-not-leak"
            assert sys.gettrace() is None
            summary = snapshot.summary(sys._getframe())
            assert (
                summary["function"]
                == "test_frame_guard_suspends_and_restores_trace_state_on_all_paths"
            )
            assert "secret" in summary["locals"]
            assert secret not in repr(summary)
        assert sys.gettrace() is tracer
        with pytest.raises(ValueError, match="boom"), py_guard_frame_and_trace_state():
            raise ValueError("boom")
        assert sys.gettrace() is tracer
    finally:
        sys.settrace(previous)


# --- 39: ruff-compatible formatting ----------------------------------------


def test_ruff_compatible_formatter_is_deterministic_and_wraps_long_calls():
    messy = "class A:\n    def f(self): return 'x'\ndef g(a,b): return pair(a,b)\n"
    formatted = py_format_with_ruff_compatible_style(messy)
    ast.parse(formatted)
    assert '"x"' in formatted
    assert "\n\n\ndef g" in formatted
    assert all(not line.endswith(" ") for line in formatted.splitlines())
    assert formatted.endswith("\n")
    assert py_format_with_ruff_compatible_style(formatted) == formatted

    arguments = ", ".join(f"value_{index}" for index in range(20))
    long_source = f"def h() -> None:\n    collect({arguments})\n"
    wrapped = py_format_with_ruff_compatible_style(long_source, line_length=72)
    ast.parse(wrapped)
    assert max(len(line) for line in wrapped.splitlines()) <= 72
    assert wrapped.count("collect(") == 1


# --- 40: parity harness -----------------------------------------------------


def test_parity_harness_compares_results_and_detects_mismatches():
    source = "fn f4_twice(x: i64) -> i64 { x * 2 }"
    report = py_run_parity_harness(
        source,
        inputs=[{"x": 3}],
        functions=["f4_twice"],
        reference={"f4_twice": lambda x: x * 2},
    )
    assert report.passed
    assert report.case_count == 1
    assert report.cases[0].python == 6
    assert "fn f4_twice" in report.canonical_source

    bad_report = py_run_parity_harness(
        source,
        inputs=[{"x": 3}],
        functions=["f4_twice"],
        reference={"f4_twice": lambda x: 99},
    )
    assert not bad_report.passed
    assert bad_report.mismatches
    assert not bad_report.cases[0].matches_reference


@pytest.mark.native
def test_parity_harness_matches_compiled_pyo3_backend(native_ns):
    source = SOURCES["f4_parity_native"]
    add_report = py_run_parity_harness(
        source,
        inputs=[{"a": 2, "b": 3}],
        functions=["f4_native_add"],
        native={"f4_native_add": native_ns["f4_native_add"]},
    )
    fallible_report = py_run_parity_harness(
        source,
        inputs=[{"flag": True}, {"flag": False}],
        functions=["f4_native_fallible"],
        native={"f4_native_fallible": native_ns["f4_native_fallible"]},
    )
    assert add_report.passed, add_report.mismatches
    assert fallible_report.passed, fallible_report.mismatches
    assert add_report.case_count + fallible_report.case_count == 3
    assert fallible_report.cases[1].python[0] == "raises"


# --- registry contract ------------------------------------------------------


def test_frame4_registry_entries_resolve_to_the_ten_documented_callables():
    expected = (
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
    for number, name in enumerate(expected, start=31):
        entry = get_function(number)
        assert entry.name == name
        assert entry.frame == 4
        assert callable(entry.callable)
        assert inspect.signature(entry.callable)
        assert (entry.callable.__doc__ or "").strip()
