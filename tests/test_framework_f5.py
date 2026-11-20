"""FRAME 5 behavioral tests for the Rust / PyO3 native backend framework."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backends.rust.codegen import RustCodegen
from backends.rust.framework import (
    RsEnumSpec,
    RsFieldSpec,
    RsFunctionSpec,
    RsGuardSpec,
    RsParamSpec,
    RsPillarError,
    RsStructSpec,
    RsTraitImplSpec,
    RsVariantSpec,
    rs_emit_monomorphized_structs,
    rs_emit_raii_guards,
    rs_emit_trait_impls_and_bounds,
    rs_generate_pyo3_scaffold,
    rs_manage_gil_and_py_refs,
    rs_map_python_args_to_rust,
    rs_map_rust_results_to_py,
    rs_raise_typed_pillar_error,
    rs_run_cargo_check_and_clippy,
    rs_verify_scaffold_matches_generator,
)
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError

# Collected by conftest.py via _native_sources; key must be globally unique.
SOURCES = {
    "test_f5_native_pyo3_parity": """
fn test_f5_native_pyo3_parity(a: i64, b: i64) -> i64 { a + b }
""",
}


def _add_spec() -> RsFunctionSpec:
    return RsFunctionSpec(
        name="add",
        params=(RsParamSpec("a", "i64"), RsParamSpec("b", "i64")),
        result="i64",
        body="a + b",
        doc="Add two integers.",
    )


def _box_spec() -> RsStructSpec:
    return RsStructSpec(
        name="Box",
        fields=(
            RsFieldSpec("value", "T"),
            RsFieldSpec("tag", "i64", visibility=""),
            RsFieldSpec("items", "Vec<T>"),
        ),
        generic_params=("T",),
    )


def _write_scaffold(root: Path, **kwargs: object) -> dict[str, str]:
    generated = rs_generate_pyo3_scaffold(module_name="polyglot_f5", **kwargs)
    for relative, text in generated.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return generated


def test_generate_scaffold_is_deterministic_and_contains_pyo3_bindings():
    first = rs_generate_pyo3_scaffold(
        module_name="polyglot_f5",
        functions=[_add_spec()],
        structs=[_box_spec()],
        substitutions={"T": "i64"},
    )
    second = rs_generate_pyo3_scaffold(
        module_name="polyglot_f5",
        functions=[_add_spec()],
        structs=[_box_spec()],
        substitutions={"T": "i64"},
    )
    assert first == second
    assert set(first) == {"Cargo.toml", "pyproject.toml", "src/lib.rs"}
    assert (
        'pyo3 = { version = "0.26", features = ["extension-module"] }'
        in first["Cargo.toml"]
    )
    lib_rs = first["src/lib.rs"]
    assert "use pyo3::prelude::*;" in lib_rs
    assert "#[pyfunction]" in lib_rs
    assert (
        "fn add(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<i64>" in lib_rs
    )
    assert "missing required argument 'a' at position 0" in lib_rs
    assert "PyPillarError::new_err(error.format_pillar())" in lib_rs
    assert "m.add_function(wrap_pyfunction!(add, m)?)?;" in lib_rs
    assert "#[pymodule]\nfn polyglot_f5(" in lib_rs
    assert "pub struct BoxI64 {" in lib_rs


def test_generate_scaffold_from_program_uses_rust_codegen_metadata():
    program = parse(
        lex("fn f5_program_add(x: i64) -> i64 { x + 1 }\nstruct F5Point { x: i64 }\n")
    )
    files = rs_generate_pyo3_scaffold(module_name="polyglot_f5p", program=program)
    lib_rs = files["src/lib.rs"]
    assert "fn f5_program_add(py: Python<'_>, args: &Bound<'_, PyTuple>)" in lib_rs
    assert "x + 1" in lib_rs
    assert "struct F5Point {" in lib_rs
    assert "#[pymodule]\nfn polyglot_f5p(" in lib_rs
    canonical = RustCodegen(module_name="polyglot_f5p").generate(program)
    assert "#[pyfunction]\nfn f5_program_add(" in canonical
    assert "#[pymodule]\nfn polyglot_f5p(" in canonical


def test_generate_scaffold_collects_native_test_sources_without_cargo(tmp_path):
    source_module = tmp_path / "test_native_f5_meta.py"
    source_module.write_text(
        'SOURCES = {"test_native_f5_meta_fn": "fn '
        'test_native_f5_meta_fn(x: i64) -> i64 { x * 2 }\\n"}\n',
        encoding="utf-8",
    )
    files = rs_generate_pyo3_scaffold(
        module_name="polyglot_f5meta", native_tests_dir=tmp_path
    )
    assert "fn test_native_f5_meta_fn(py: Python<'_>" in files["src/lib.rs"]
    assert "x * 2" in files["src/lib.rs"]


def test_verify_scaffold_detects_drift_and_missing_files(tmp_path):
    generated = _write_scaffold(
        tmp_path, functions=[_add_spec()], substitutions={"T": "i64"}
    )
    clean = rs_verify_scaffold_matches_generator(
        tmp_path, module_name="polyglot_f5", generated=generated
    )
    assert clean.ok
    assert clean.matches == ("Cargo.toml", "pyproject.toml", "src/lib.rs")
    assert clean.mismatched == () and clean.missing == ()

    (tmp_path / "src/lib.rs").write_text(
        generated["src/lib.rs"] + "\n// drifted line\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").unlink()
    drifted = rs_verify_scaffold_matches_generator(
        tmp_path, module_name="polyglot_f5", generated=generated
    )
    assert not drifted.ok
    assert drifted.mismatched == ("src/lib.rs",)
    assert drifted.missing == ("pyproject.toml",)
    assert any(
        "src/lib.rs: first difference at line" in detail for detail in drifted.details
    )


def test_verify_scaffold_reports_unexpected_files_in_strict_mode(tmp_path):
    generated = _write_scaffold(tmp_path, functions=[_add_spec()])
    (tmp_path / "stale_generated.rs").write_text("stale\n", encoding="utf-8")
    permissive = rs_verify_scaffold_matches_generator(
        tmp_path, module_name="polyglot_f5", generated=generated
    )
    strict = rs_verify_scaffold_matches_generator(
        tmp_path,
        module_name="polyglot_f5",
        generated=generated,
        ignore_extra=False,
    )
    assert permissive.ok
    assert permissive.extra == ("stale_generated.rs",)
    assert not strict.ok


def test_emit_monomorphized_structs_preserves_field_order_and_visibility():
    emitted = rs_emit_monomorphized_structs([_box_spec()], substitutions={"T": "i64"})
    assert "pub struct BoxI64 {" in emitted
    assert emitted.index("pub value: i64,") < emitted.index("tag: i64,")
    assert emitted.index("tag: i64,") < emitted.index("pub items: Vec<i64>,")
    assert "    tag: i64," in emitted  # no visibility prefix
    assert "    #[pyo3(get, set)]" in emitted
    assert "    T" not in emitted

    enum = RsEnumSpec(
        name="Outcome",
        variants=(
            RsVariantSpec("Pending", ("String",)),
            RsVariantSpec("Ready", ("T",)),
            RsVariantSpec("Empty"),
        ),
        generic_params=("T",),
    )
    enum_rs = rs_emit_monomorphized_structs([], [enum], substitutions={"T": "i64"})
    assert "pub enum OutcomeI64 {" in enum_rs
    assert "Pending(String)," in enum_rs
    assert "Ready(i64)," in enum_rs
    assert "Empty," in enum_rs


def test_emit_monomorphized_structs_rejects_unsubstituted_parameter():
    with pytest.raises(RsPillarError) as error:
        rs_emit_monomorphized_structs([_box_spec()])
    assert error.value.code == "E001"
    assert "no monomorphization" in error.value.message
    assert isinstance(error.value, PillarError)


def test_emit_trait_impls_and_bounds_emits_rustc_clean_blocks():
    method = RsFunctionSpec(
        name="to_i64",
        params=(RsParamSpec("self", "&self"),),
        result="i64",
        body="self.value",
    )
    emitted = rs_emit_trait_impls_and_bounds(
        [
            RsTraitImplSpec(
                target="BoxI64",
                trait="PillarConvert",
                methods=(method,),
                bounds=("BoxI64: Clone", "BoxI64: Send"),
            ),
            RsTraitImplSpec(
                target="BoxI64",
                methods=(method,),
            ),
        ]
    )
    assert "impl PillarConvert for BoxI64" in emitted
    assert "where\n    BoxI64: Clone,\n    BoxI64: Send,\n{" in emitted
    assert "fn to_i64(&self) -> i64 {" in emitted
    assert "        self.value" in emitted
    assert emitted.count("impl BoxI64 {") == 1


def test_emit_trait_impls_rejects_malformed_where_bound():
    with pytest.raises(RsPillarError) as error:
        rs_emit_trait_impls_and_bounds(
            [RsTraitImplSpec(target="BoxI64", trait="Clone", bounds=("T Clone",))]
        )
    assert error.value.code == "E003"
    assert "must contain ':'" in error.value.message


def test_map_python_args_to_rust_emits_explicit_extraction_code():
    params = (
        RsParamSpec("a", "i64"),
        RsParamSpec("b", "Vec<String>"),
        RsParamSpec("limit", "usize", required=False, default="10"),
    )
    mapping = rs_map_python_args_to_rust(params)
    assert "let __py_args = args;" in mapping.code
    assert "__py_args.get_item(0)" in mapping.code
    assert "__arg_1.extract::<Vec<String>>()" in mapping.code
    assert (
        "PyTypeError::new_err(\"argument 'b' at position 1 must be Vec<String>\")"
        in mapping.code
    )
    assert "missing required argument 'a' at position 0" in mapping.code
    assert "Err(_) => 10," in mapping.code
    assert mapping.rust_types == {"a": "i64", "b": "Vec<String>", "limit": "usize"}


def test_map_python_args_to_rust_validates_values_and_ranges():
    params = (
        RsParamSpec("count", "u8"),
        RsParamSpec("names", "Vec<String>"),
        RsParamSpec("ratio", "f64"),
        RsParamSpec("maybe", "Option<i64>"),
    )
    mapping = rs_map_python_args_to_rust(
        params,
        values={
            "count": 7,
            "names": ["a", "b"],
            "ratio": 2,
            "maybe": None,
        },
    )
    assert mapping.values == {
        "count": 7,
        "names": ["a", "b"],
        "ratio": 2.0,
        "maybe": None,
    }
    assert "widened from int to Rust f64" in mapping.diagnostics[0]

    with pytest.raises(RsPillarError) as error:
        rs_map_python_args_to_rust((RsParamSpec("count", "u8"),), values={"count": 256})
    assert error.value.code == "E001"
    assert "outside the range of Rust u8" in error.value.message

    with pytest.raises(RsPillarError) as missing:
        rs_map_python_args_to_rust((RsParamSpec("count", "u8"),), values={})
    assert "missing required argument 'count'" in missing.value.message


def test_map_rust_results_to_py_handles_typed_errors_and_plain_values():
    typed = rs_map_rust_results_to_py("Result<i64, PillarError>")
    assert "match __result {" in typed.code
    assert "Ok(value) => Ok(value.into_pyobject(py)?.into_any().unbind())" in typed.code
    assert (
        "Err(error) => Err(PyPillarError::new_err(error.format_pillar()))" in typed.code
    )
    assert typed.ok_type == "i64"
    assert typed.python_exception == "PillarError"
    assert typed.rust_exception == "PyPillarError"

    plain = rs_map_rust_results_to_py("String")
    assert plain.code == "Ok(__result)"
    assert plain.error_type is None
    assert plain.python_exception == "None"

    string_error = rs_map_rust_results_to_py("Result<(), String>")
    assert "PyValueError::new_err(message)" in string_error.code


def test_map_rust_results_to_py_rejects_unknown_error_type():
    with pytest.raises(RsPillarError) as error:
        rs_map_rust_results_to_py("Result<i64, MysteryError>")
    assert error.value.code == "E001"
    assert "unsupported Rust error type" in error.value.message


def test_raise_typed_pillar_error_exposes_dedicated_exception_and_bridge():
    with pytest.raises(RsPillarError) as error:
        rs_raise_typed_pillar_error(
            "E004",
            "cargo failed",
            reason="missing crate",
            help_text="run cargo fetch",
            line=3,
            col=7,
        )
    assert isinstance(error.value, PillarError)
    assert error.value.code == "E004"
    assert (error.value.line, error.value.col) == (3, 7)
    assert "error[E004]: cargo failed" in str(error.value)
    assert "missing crate" in error.value.reason
    assert error.value.rust_exception_name == "PyPillarError"

    lib_rs = rs_generate_pyo3_scaffold(module_name="polyglot_f5")["src/lib.rs"]
    assert "pyo3::create_exception!(" in lib_rs
    assert "PyPillarError," in lib_rs
    assert "impl From<PillarError> for pyo3::PyErr {" in lib_rs
    assert 'm.add("PillarError", m.py().get_type::<PyPillarError>())?;' in lib_rs


def test_manage_gil_and_py_refs_releases_gil_and_blocks_borrows():
    plan = rs_manage_gil_and_py_refs(
        "compute(2)", borrowed=("value",), owned=("handle",)
    )
    assert "let __gil_owned_value = value.clone().unbind();" in plan.code
    assert "let __gil_move_handle = handle;" in plan.code
    assert "py.allow_threads(|| {" in plan.code
    assert "compute(2)" in plan.code
    assert "never call Python APIs" in " ".join(plan.rules)

    warning_plan = rs_manage_gil_and_py_refs("py.eval('1')")
    assert warning_plan.warnings
    assert "Python APIs inside allow_threads" in warning_plan.warnings[0]

    with pytest.raises(RsPillarError) as error:
        rs_manage_gil_and_py_refs("value.borrow_mut()", borrowed=("value",))
    assert error.value.code == "E007"
    assert "re-entrant GIL-release boundary" in error.value.message


def test_emit_raii_guards_release_on_every_return_path():
    emitted = rs_emit_raii_guards(
        [
            RsGuardSpec("state", "lock", "state_lock.lock().unwrap()"),
            RsGuardSpec("socket", "handle", "open_socket()", "close(__resource);"),
        ]
    )
    assert "struct PolyglotRaiiGuard<T> {" in emitted
    assert "impl<T> Drop for PolyglotRaiiGuard<T> {" in emitted
    assert "let on_drop = self.on_drop.take();" in emitted
    assert "if let (Some(resource), Some(on_drop))" in emitted
    assert "PolyglotRaiiGuard::new(" in emitted
    assert "state_lock.lock().unwrap()" in emitted
    assert "close(__resource);" in emitted
    assert emitted.index("// handle guard 'socket'") < emitted.index(
        "// lock guard 'state'"
    )

    with pytest.raises(RsPillarError) as error:
        rs_emit_raii_guards([RsGuardSpec("bad", "socket", "open()")])
    assert error.value.code == "E003"
    assert "unknown kind" in error.value.message


def test_run_cargo_check_and_clippy_uses_isolated_dirs_and_structured_result(tmp_path):
    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "f5demo"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def runner(
        command: tuple[str, ...], cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd, env))
        return subprocess.CompletedProcess(
            command, 0, stdout="warning: unused variable\n", stderr=""
        )

    base = tmp_path / "isolated"
    result = rs_run_cargo_check_and_clippy(crate, target_dir=base, runner=runner)
    assert result.ok
    assert result.check.ok and result.clippy is not None and result.clippy.ok
    assert len(calls) == 2
    check_cmd, check_cwd, check_env = calls[0]
    clippy_cmd, clippy_cwd, clippy_env = calls[1]
    assert check_cmd[:4] == ("cargo", "check", "--all-targets", "--target-dir")
    assert clippy_cmd[:4] == ("cargo", "clippy", "--all-targets", "--target-dir")
    assert check_cwd == crate and clippy_cwd == crate
    assert check_env["CARGO_TARGET_DIR"] == str(base / "check")
    assert clippy_env["CARGO_TARGET_DIR"] == str(base / "clippy")
    assert (base / "check").is_dir() and (base / "clippy").is_dir()
    assert result.check.diagnostics == ("warning: unused variable",)
    assert result.check.target_dir == str(base / "check")


def test_run_cargo_check_and_clippy_reports_failed_steps(tmp_path):
    crate = tmp_path / "broken"
    crate.mkdir()
    (crate / "Cargo.toml").write_text("[package]\nname = 'broken'\n", encoding="utf-8")

    def failing(
        command: tuple[str, ...], cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 101, "", "error[E0308]: mismatched types\n"
        )

    result = rs_run_cargo_check_and_clippy(crate, runner=failing)
    assert not result.ok
    assert result.check.returncode == 101
    assert result.clippy is not None and not result.clippy.ok
    assert "error[E0308]: mismatched types" in result.check.diagnostics


def test_run_cargo_check_and_clippy_rejects_missing_manifest(tmp_path):
    with pytest.raises(RsPillarError) as error:
        rs_run_cargo_check_and_clippy(tmp_path)
    assert error.value.code == "E006"
    assert "no Cargo.toml" in error.value.message


@pytest.mark.native
def test_native_compile_parity_with_generated_scaffold_markers(
    native_ns, native_backend
):
    """The real RustCodegen module compiles and matches our scaffold contract."""
    source = SOURCES["test_f5_native_pyo3_parity"]
    program = parse(lex(source))
    generated_rust = RustCodegen(module_name="polyglot_f5_parity").generate(program)
    scaffold = rs_generate_pyo3_scaffold(
        module_name="polyglot_f5_parity", program=program
    )["src/lib.rs"]
    for marker in ("#[pyfunction]", "#[pymodule]", "polyglot_f5_parity", "a + b"):
        assert marker in generated_rust
        assert marker in scaffold
    assert native_ns["test_f5_native_pyo3_parity"](2, 3) == 5
    assert native_ns["test_f5_native_pyo3_parity"](-4, 9) == 5
    assert native_backend is not None
