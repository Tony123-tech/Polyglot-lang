"""Paired Python/native tests for bounded native codegen additions."""

import pytest

from backends.errors import TranslateError
from backends.python.translator import compile_rust
from backends.rust.codegen import RustCodegen
from compiler.core.lexer import lex
from compiler.core.parser import parse

SOURCES = {
    "borrowed_containers": """
fn codegen_borrowed_len(values: &Vec<i64>) -> i64 {
    values.len() as i64
}
fn codegen_borrowed_sum(values: &Vec<i64>) -> i64 {
    values.fold(0, |total, value| total + value)
}
fn codegen_borrowed_map(values: &Vec<i64>) -> Vec<i64> {
    values.map(|value| value + 1)
}
fn codegen_borrowed_filter(values: &Vec<i64>) -> Vec<i64> {
    values.filter(|value| *value > 2)
}
fn codegen_borrowed_local(values: &Vec<i64>) -> i64 {
    let view: &Vec<i64> = values;
    view.len() as i64
}
fn codegen_borrowed_text(value: &String) -> i64 {
    value.len() as i64
}
fn codegen_borrowed_text_trim(value: &String) -> String {
    value.trim()
}
fn codegen_borrowed_text_slice(value: &String) -> String {
    &value[1..3]
}
""",
    "attributes": """
#[inline]
fn codegen_inline(value: i64) -> i64 {
    value + 1
}
#[cold]
fn codegen_cold() -> i64 {
    7
}
#[must_use]
fn codegen_result(value: i64) -> Result<i64, String> {
    Ok(value)
}
#[must_use]
fn codegen_plain_must_use(value: i64) -> i64 {
    value
}
#[repr(transparent)]
struct CodegenWrapper { value: i64 }
fn codegen_wrapper_new(value: i64) -> CodegenWrapper {
    CodegenWrapper { value: value }
}
fn codegen_wrapper_value(wrapper: CodegenWrapper) -> i64 {
    wrapper.value
}
""",
    "constant_folding": """
fn codegen_const_less() -> bool {
    1 < 2
}
fn codegen_const_equal() -> bool {
    "a" == "a"
}
fn codegen_folded_guard(value: i64) -> bool {
    2 <= 2 && value == 4
}
""",
}


def _generated(key: str) -> str:
    return RustCodegen().generate(parse(lex(SOURCES[key])))


@pytest.mark.native
def test_native_borrowed_containers(native_ns):
    assert native_ns["codegen_borrowed_len"]([1, 2, 3]) == 3
    assert native_ns["codegen_borrowed_sum"]([1, 2, 3]) == 6
    assert native_ns["codegen_borrowed_map"]([1, 2]) == [2, 3]
    assert native_ns["codegen_borrowed_filter"]([1, 2, 3, 4]) == [3, 4]
    assert native_ns["codegen_borrowed_local"]([1, 2, 3]) == 3
    assert native_ns["codegen_borrowed_text"]("hello") == 5
    assert native_ns["codegen_borrowed_text_trim"]("  hi  ") == "hi"
    assert native_ns["codegen_borrowed_text_slice"]("abcdef") == "bc"


def test_python_borrowed_containers():
    namespace = compile_rust(parse(lex(SOURCES["borrowed_containers"])))

    assert namespace["codegen_borrowed_len"]([1, 2, 3]) == 3
    assert namespace["codegen_borrowed_sum"]([1, 2, 3]) == 6
    assert namespace["codegen_borrowed_map"]([1, 2]) == [2, 3]
    assert namespace["codegen_borrowed_filter"]([1, 2, 3, 4]) == [3, 4]
    assert namespace["codegen_borrowed_local"]([1, 2, 3]) == 3
    assert namespace["codegen_borrowed_text"]("hello") == 5
    assert namespace["codegen_borrowed_text_trim"]("  hi  ") == "hi"
    assert namespace["codegen_borrowed_text_slice"]("abcdef") == "bc"


def test_borrowed_vec_uses_an_owned_pyo3_boundary_and_slice_body():
    source = _generated("borrowed_containers")

    assert "fn codegen_borrowed_len(values: Vec<i64>) -> i64" in source
    assert (
        "fn __polyglot_slice_codegen_borrowed_len(mut values: &[i64]) -> i64" in source
    )
    assert "fn codegen_borrowed_text(mut value: &str) -> i64" in source


@pytest.mark.native
def test_native_attributes_and_transparent_struct(native_ns):
    assert native_ns["codegen_inline"](1) == 2
    assert native_ns["codegen_cold"]() == 7
    assert native_ns["codegen_result"](3) == 3
    wrapper = native_ns["codegen_wrapper_new"](9)
    assert native_ns["codegen_wrapper_value"](wrapper) == 9


def test_generated_function_attributes_are_bounded():
    source = _generated("attributes")

    assert "#[inline]\n#[pyfunction]\nfn codegen_inline" in source
    assert "#[cold]\n#[pyfunction]\nfn codegen_cold" in source
    assert "#[must_use]\n#[pyfunction]\nfn codegen_result" in source
    assert source.count("#[must_use]") == 1
    assert "#[repr(transparent)]\n#[pyclass]" in source


def test_repr_transparent_rejects_multi_field_structs():
    source = "#[repr(transparent)] struct Bad { left: i64, right: i64 }"

    with pytest.raises(TranslateError, match=r"#\[repr\(transparent\)\] requires"):
        RustCodegen().generate(parse(lex(source)))


def test_runtime_helpers_carry_local_dead_code_allow():
    source = _generated("constant_folding")

    assert "#[allow(dead_code)]\nfn polyglot_sleep_ms" in source
    assert "#[allow(dead_code)]\nfn polyglot_reactor" in source


@pytest.mark.native
def test_native_literal_comparisons(native_ns):
    assert native_ns["codegen_const_less"]() is True
    assert native_ns["codegen_const_equal"]() is True
    assert native_ns["codegen_folded_guard"](4) is True
    assert native_ns["codegen_folded_guard"](5) is False


def test_python_literal_comparisons():
    namespace = compile_rust(parse(lex(SOURCES["constant_folding"])))

    assert namespace["codegen_const_less"]() is True
    assert namespace["codegen_const_equal"]() is True
    assert namespace["codegen_folded_guard"](4) is True
    assert namespace["codegen_folded_guard"](5) is False


def test_literal_comparisons_are_folded_in_generated_rust():
    source = _generated("constant_folding")

    assert "fn codegen_const_less() -> bool {\n    true\n}" in source
    assert "fn codegen_const_equal() -> bool {\n    true\n}" in source
    assert (
        "fn codegen_folded_guard(mut value: i64) -> bool {\n    (true && (value == 4))\n}"
        in source
    )
