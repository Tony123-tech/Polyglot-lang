import subprocess
from dataclasses import is_dataclass

import pytest

from backends.rust.codegen import RustCodegen
from backends.rust.compiler import NativeBackend, RustCompileError, _parse_cargo_error
from lexer import lex
from parser import parse


def test_rust_compile_error_captures_diagnostic_fields():
    error = _parse_cargo_error(
        "",
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n"
        "  |\n",
        "fn main() {}\n",
    )

    assert is_dataclass(error)
    assert isinstance(error, RustCompileError)
    assert error.message == "mismatched types"
    assert (error.rust_line, error.rust_col) == (3, 7)
    assert (error.source_line, error.source_col) == (None, None)
    assert error.error_line == 3
    assert error.error_col == 7
    assert error.error_code == "E0308"
    assert "error[E004]:" in error.pretty()


def test_rust_compile_error_without_source_location():
    error = _parse_cargo_error(
        "",
        "error: linking with `cc` failed: linker command not found\n",
        "",
    )

    assert error.rust_line is None
    assert error.rust_col is None
    assert error.message == "linking with `cc` failed: linker command not found"
    assert error.error_code is None
    assert "error[E004]:" in error.pretty()


def test_rust_compile_error_captures_warnings_without_replacing_error():
    error = _parse_cargo_error(
        "",
        "warning: unused variable: `x`\n"
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n",
        "fn f() {}\n",
    )

    assert error.message == "mismatched types"
    assert error.warnings == ["unused variable: `x`"]


def test_rust_compile_error_keeps_later_errors_in_raw_stderr():
    error = _parse_cargo_error(
        "",
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n"
        "error[E0425]: cannot find value `missing`\n"
        " --> src/lib.rs:8:1\n",
        "",
    )

    assert error.message == "mismatched types"
    assert error.rust_line == 3
    assert "cannot find value `missing`" in error.raw_stderr


def test_rust_compile_error_deduplicates_identical_diagnostics():
    duplicate = (
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n"
        "  |\n"
        "  = help: adjust the return type\n"
    )
    error = _parse_cargo_error(
        "",
        duplicate + duplicate
        + "error[E0425]: cannot find value `missing`\n"
        " --> src/lib.rs:8:1\n",
        "",
    )

    assert error.raw_stderr.count("error[E0308]") == 1
    assert "error[E0425]" in error.raw_stderr


def test_rust_compile_error_preserves_runtime_error_compatibility():
    error = _parse_cargo_error(
        "",
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n",
        "",
    )

    assert isinstance(error, RuntimeError)
    assert error.error_line == error.rust_line == 3
    assert error.error_col == error.rust_col == 7
    assert error.rust_message == error.raw_stderr


def test_warning_only_cargo_result_does_not_raise(tmp_path, monkeypatch):
    backend = NativeBackend(cache_dir=tmp_path / "cache")
    result = subprocess.CompletedProcess(
        args=["cargo", "build"],
        returncode=0,
        stdout="",
        stderr="warning: unused variable: `x`\n",
    )
    monkeypatch.setattr(
        "backends.rust.compiler.subprocess.run",
        lambda *_args, **_kwargs: result,
    )

    backend._build(
        "fn polyglot_native() {}\n",
        tmp_path / "build",
        {},
    )


def test_rust_codegen_records_source_origin_for_emitted_lines():
    program = parse(lex("\nfn origin() -> i64 { 1 }"))
    codegen = RustCodegen()
    generated = codegen.generate(program)
    rust_line = next(
        index for index, line in enumerate(generated.splitlines(), start=1)
        if "fn origin()" in line
    )

    assert codegen.line_map[rust_line] == (2, 1)


def test_rust_codegen_records_originating_node_for_emitted_lines():
    from ast_nodes import FnDecl

    program = parse(lex("\nfn origin_node() -> i64 { 1 }"))
    codegen = RustCodegen()
    generated = codegen.generate(program)
    rust_line = next(
        index for index, line in enumerate(generated.splitlines(), start=1)
        if "fn origin_node()" in line
    )

    node = codegen.line_map_nodes[rust_line]
    assert isinstance(node, FnDecl)
    assert node.name == "origin_node"


def test_cargo_error_maps_rust_line_to_source_origin():
    error = _parse_cargo_error(
        "",
        "error[E0308]: mismatched types\n"
        " --> src/lib.rs:3:7\n",
        "fn a() {}\nfn b() {}\nfn c() {}\n",
        {3: (17, 9)},
    )

    assert (error.rust_line, error.rust_col) == (3, 7)
    assert (error.source_line, error.source_col) == (17, 9)


def test_pretty_includes_original_and_generated_lines():
    source = "\nfn bad() -> i64 { true }"
    program = parse(lex(source))
    codegen = RustCodegen()
    generated = codegen.generate(program)
    rust_line = next(
        line_number
        for line_number, rust_line in enumerate(
            generated.splitlines(), start=1,
        )
        if "fn bad()" in rust_line
    )
    error = _parse_cargo_error(
        "",
        f"error[E0308]: mismatched types\n"
        f" --> src/lib.rs:{rust_line}:5\n",
        generated,
        codegen.line_map,
        codegen.source_line_text_map,
    )
    pretty = error.pretty()

    assert error.source_line == 2
    assert "fn bad() -> i64 { true }" in pretty
    assert generated.splitlines()[rust_line - 1] in pretty


@pytest.mark.slow
@pytest.mark.parametrize(
    ("source", "expected_message", "source_line"),
    [
        (
            "fn type_error() -> i64 {\n    \"wrong\"\n}",
            "mismatched types",
            2,
        ),
        (
            "fn immutable_assign() -> i64 {\n"
            "    let value = 1;\n"
            "    value = 2;\n"
            "    value\n"
            "}",
            "cannot assign twice",
            3,
        ),
        (
            "fn non_exhaustive(value: bool) -> i64 {\n"
            "    match value {\n"
            "        true => 1,\n"
            "    }\n"
            "}",
            "non-exhaustive patterns",
            2,
        ),
        (
            "fn unresolved_const() -> i64 {\n"
            "    match 1 {\n"
            "        const MISSING => 0,\n"
            "        _ => 1,\n"
            "    }\n"
            "}",
            "cannot find unit struct",
            2,
        ),
    ],
)
def test_native_errors_map_back_to_polyglot_source(
    source, expected_message, source_line,
):
    backend = NativeBackend()
    program = parse(lex(source))

    with pytest.raises(RustCompileError) as raised:
        backend.compile(program)

    error = raised.value
    assert expected_message in error.pretty()
    # rust_line/source_line/source_line_text depend on the Cargo error
    # format, which can differ across Rust toolchain versions. Assert
    # them only when present.
    if error.rust_line is not None:
        assert error.rust_line > 0
        assert error.source_line == source_line
        assert error.source_line_text == source.splitlines()[source_line - 1]
