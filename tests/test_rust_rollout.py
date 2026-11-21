"""Focused tests for batch 5 of the Rust / PyO3 framework rollout."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from backends.rust import boundary, checks, framework, gil, scaffold
from backends.rust.rollout import FRAMEWORK_ROWS
from compiler.frontend.errors import PolyglotError

LEGACY_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "rs_generate_pyo3_scaffold",
        "backends.rust.framework",
        "scaffold",
        "tests.test_rust_rollout",
    ),
    (
        "rs_verify_scaffold_matches_generator",
        "backends.rust.framework",
        "scaffold",
        "tests.test_rust_rollout",
    ),
    (
        "rs_emit_monomorphized_structs",
        "backends.rust.framework",
        "scaffold",
        "tests.test_rust_rollout",
    ),
    (
        "rs_emit_trait_impls_and_bounds",
        "backends.rust.framework",
        "scaffold",
        "tests.test_rust_rollout",
    ),
    (
        "rs_map_python_args_to_rust",
        "backends.rust.framework",
        "boundary",
        "tests.test_rust_rollout",
    ),
    (
        "rs_map_rust_results_to_py",
        "backends.rust.framework",
        "boundary",
        "tests.test_rust_rollout",
    ),
    (
        "rs_raise_typed_pillar_error",
        "backends.rust.framework",
        "boundary",
        "tests.test_rust_rollout",
    ),
    (
        "rs_manage_gil_and_py_refs",
        "backends.rust.framework",
        "gil",
        "tests.test_rust_rollout",
    ),
    (
        "rs_emit_raii_guards",
        "backends.rust.framework",
        "raii",
        "tests.test_rust_rollout",
    ),
    (
        "rs_run_cargo_check_and_clippy",
        "backends.rust.framework",
        "checks",
        "tests.test_rust_rollout",
    ),
)

NEW_MODULES: dict[str, tuple[Any, str, str, int]] = {
    "backends.rust.scaffold": (scaffold, "rsc_", "scaffold", 25),
    "backends.rust.boundary": (boundary, "rsb_", "boundary", 25),
    "backends.rust.gil": (gil, "rsg_", "gil", 20),
    "backends.rust.checks": (checks, "rsk_", "checks", 20),
}

Invoke = Callable[[Path], dict[str, Any]]


def _crate_with_manifest(path: Path) -> dict[str, Any]:
    """Create a minimal crate manifest and run the modeled check steps."""
    (path / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    return checks.rsk_check_crate(path)


def _endswith(suffix: str) -> Callable[[Any], bool]:
    """Build a path assertion that does not depend on the temporary root."""

    def check(value: Any) -> bool:
        return isinstance(value, str) and value.endswith(suffix)

    return check


BEHAVIORS: tuple[tuple[str, Invoke, dict[str, Any]], ...] = (
    # 0411-0435: scaffold helpers.
    (
        "rsc_abi_settings",
        lambda _: scaffold.rsc_abi_settings(abi3=True, platform="macos"),
        {
            "abi3": True,
            "platform": "macos",
            "features": ("abi3-py314", "extension-module"),
        },
    ),
    (
        "rsc_crate_layout",
        lambda _: scaffold.rsc_crate_layout(
            {
                "Cargo.toml": "[package]\n",
                "pyproject.toml": "[build-system]\n",
                "src/lib.rs": "// generated\n",
            }
        ),
        {
            "ok": True,
            "present": ("Cargo.toml", "pyproject.toml", "src/lib.rs"),
            "file_count": 3,
        },
    ),
    (
        "rsc_crate_manifest",
        lambda _: scaffold.rsc_crate_manifest(module_name="demo_py"),
        {"path": "Cargo.toml", "module_name": "demo_py", "crate_type": "cdylib"},
    ),
    (
        "rsc_derive_line",
        lambda _: scaffold.rsc_derive_line(["Clone", "Debug"]),
        {"text": "#[derive(Clone, Debug)]", "count": 2},
    ),
    (
        "rsc_doc_comment",
        lambda _: scaffold.rsc_doc_comment("line one\nline two", indent="    "),
        {"text": "    /// line one\n    /// line two", "line_count": 2},
    ),
    (
        "rsc_enum_def",
        lambda _: scaffold.rsc_enum_def(
            {
                "name": "Choice",
                "variants": [
                    {"name": "Empty"},
                    {"name": "Value", "field_types": ["i64"]},
                ],
            }
        ),
        {"name": "Choice", "source_name": "Choice", "line_count": 6},
    ),
    (
        "rsc_feature_flags",
        lambda _: scaffold.rsc_feature_flags(
            {"alpha": ["dep_a"], "beta": ["dep_b"]}, default=["alpha"]
        ),
        {"features": ("alpha", "beta"), "default": ("alpha",), "entry_count": 3},
    ),
    (
        "rsc_field_line",
        lambda _: scaffold.rsc_field_line({"name": "value", "type": "i64"}),
        {
            "name": "value",
            "rust_type": "i64",
            "python_access": True,
            "text": "    #[pyo3(get, set)]\n    pub value: i64,",
        },
    ),
    (
        "rsc_function_def",
        lambda _: scaffold.rsc_function_def(
            {
                "name": "add",
                "params": [{"name": "a", "type": "i64"}],
                "result": "i64",
                "body": "a + 1",
                "doc": "Add one.",
            }
        ),
        {"name": "add", "result_type": "PyResult<i64>", "registration": ""},
    ),
    (
        "rsc_generic_suffix",
        lambda _: scaffold.rsc_generic_suffix("Vec<String>"),
        {"suffix": "VecString", "rust_type": "Vec<String>", "length": 9},
    ),
    (
        "rsc_layout_plan",
        lambda _: scaffold.rsc_layout_plan(
            structs=[
                {"name": "Inner", "fields": []},
                {"name": "Outer", "fields": [{"name": "inner", "type": "Inner"}]},
            ]
        ),
        {
            "names": ("Inner", "Outer"),
            "edges": {"Inner": (), "Outer": ("Inner",)},
            "types": {"Inner": "struct", "Outer": "struct"},
        },
    ),
    (
        "rsc_method_def",
        lambda _: scaffold.rsc_method_def(
            {"name": "read", "result": "i64", "body": "self.value"}
        ),
        {
            "name": "read",
            "receiver": "&self",
            "signature": "fn read(&self) -> PyResult<i64>",
            "method_attrs": ("#[pymethods]",),
        },
    ),
    (
        "rsc_module_header",
        lambda _: scaffold.rsc_module_header(
            module_name="demo", imports=["pyo3::prelude::*"], allow_unused=False
        ),
        {
            "module_name": "demo",
            "imports": ("pyo3::prelude::*",),
            "attributes": (),
            "line_count": 4,
        },
    ),
    (
        "rsc_module_init",
        lambda _: scaffold.rsc_module_init("demo", functions=["add"], classes=["Box"]),
        {
            "module_name": "demo",
            "functions": ("add",),
            "classes": ("Box",),
            "registration_count": 2,
        },
    ),
    (
        "rsc_param_decl",
        lambda _: scaffold.rsc_param_decl("count", "usize", mutable=True),
        {"text": "mut count: usize", "name": "count", "mutable": True},
    ),
    (
        "rsc_pyproject_manifest",
        lambda _: scaffold.rsc_pyproject_manifest(module_name="demo"),
        {"path": "pyproject.toml", "module_name": "demo", "requires_python": ">=3.14"},
    ),
    (
        "rsc_return_type",
        lambda _: scaffold.rsc_return_type("Result<i64, PillarError>"),
        {"text": "PyResult<i64>", "inner": "i64", "error_type": "PillarError"},
    ),
    (
        "rsc_scaffold_diff",
        lambda _: scaffold.rsc_scaffold_diff(
            {"a": "1", "b": "2"}, {"a": "1", "c": "3"}
        ),
        {"added": ("c",), "removed": ("b",), "changed": (), "comparisons": 1},
    ),
    (
        "rsc_scaffold_fingerprint",
        lambda _: scaffold.rsc_scaffold_fingerprint({"b": "2", "a": "1"}),
        {"algorithm": "sha256", "paths": ("a", "b"), "byte_length": 7},
    ),
    (
        "rsc_scaffold_manifest",
        lambda _: scaffold.rsc_scaffold_manifest(),
        {
            "module_name": "polyglot_native",
            "function_names": (),
            "class_names": (),
            "layout": {"ok": True},
        },
    ),
    (
        "rsc_scaffold_plan",
        lambda _: scaffold.rsc_scaffold_plan(
            module_name="demo",
            structs=[{"name": "Item", "fields": [{"name": "value", "type": "i64"}]}],
        ),
        {
            "module_name": "demo",
            "class_names": ("Item",),
            "function_names": (),
            "layout": {"names": ("Item",)},
        },
    ),
    (
        "rsc_scaffold_render",
        lambda _: scaffold.rsc_scaffold_render(scaffold.rsc_scaffold_plan()),
        {
            "path": "src/lib.rs",
            "section_order": (
                "header",
                "structs",
                "enums",
                "impls",
                "functions",
                "module_init",
            ),
        },
    ),
    (
        "rsc_struct_def",
        lambda _: scaffold.rsc_struct_def(
            {"name": "Point", "fields": [{"name": "x", "type": "i64"}]}
        ),
        {
            "name": "Point",
            "source_name": "Point",
            "fields": ("x",),
            "rust_types": ("i64",),
            "pyclass": True,
        },
    ),
    (
        "rsc_trait_impl",
        lambda _: scaffold.rsc_trait_impl(
            "Point", trait="Clone", bounds=["Point: Send"]
        ),
        {
            "target": "Point",
            "trait": "Clone",
            "bounds": ("Point: Send",),
            "methods": (),
        },
    ),
    (
        "rsc_type_substitute",
        lambda _: scaffold.rsc_type_substitute("Vec<T>", {"T": "i64"}),
        {"text": "Vec<i64>", "source": "Vec<T>", "unresolved": ()},
    ),
    # 0436-0460: boundary helpers.
    (
        "rsb_arg_contract",
        lambda _: boundary.rsb_arg_contract(
            [("a", "i64"), ("b", "String")], values={"a": 1, "b": "x"}
        ),
        {
            "count": 2,
            "required": ("a", "b"),
            "rust_types": ("i64", "String"),
            "validated": {"values": {"a": 1, "b": "x"}},
        },
    ),
    (
        "rsb_boundary_report",
        lambda _: boundary.rsb_boundary_report([("a", "i64")], values={"a": 1}),
        {
            "diagnostics": (),
            "safe_for_gil_release": True,
            "result": {"signature": "PyResult<i64>"},
        },
    ),
    (
        "rsb_bytes_decode",
        lambda _: boundary.rsb_bytes_decode(b"\x01\xff"),
        {"values": [1, 255], "rust_type": "Vec<u8>", "rust_expr": "vec![1, 255]"},
    ),
    (
        "rsb_call_signature",
        lambda _: boundary.rsb_call_signature("add", [("a", "i64")]),
        {
            "name": "add",
            "params": ("a",),
            "return_type": "PyResult<i64>",
            "attributes": ("#[pyfunction]",),
        },
    ),
    (
        "rsb_coerce_mapping",
        lambda _: boundary.rsb_coerce_mapping({"a": 1}, "String", "i64"),
        {"value": {"a": 1}, "rust_type": "HashMap<String, i64>", "length": 1},
    ),
    (
        "rsb_coerce_optional",
        lambda _: boundary.rsb_coerce_optional(3, "i64"),
        {
            "value": 3,
            "present": True,
            "rust_expr": "Some(3)",
            "rust_type": "Option<i64>",
        },
    ),
    (
        "rsb_coerce_scalar",
        lambda _: boundary.rsb_coerce_scalar(7, "u8"),
        {"value": 7, "rust_type": "u8", "kind": "integer", "range": [0, 255]},
    ),
    (
        "rsb_coerce_sequence",
        lambda _: boundary.rsb_coerce_sequence([1, 2], "i64"),
        {"value": [1, 2], "rust_type": "Vec<i64>", "length": 2},
    ),
    (
        "rsb_default_arg",
        lambda _: boundary.rsb_default_arg("limit", "usize", "10"),
        {"code": "10", "name": "limit", "is_literal": True},
    ),
    (
        "rsb_error_bridge",
        lambda _: boundary.rsb_error_bridge("PillarError"),
        {
            "rust_type": "PillarError",
            "python_exception": "PillarError",
            "exception_path": "PyPillarError",
        },
    ),
    (
        "rsb_exception_map",
        lambda _: boundary.rsb_exception_map("ValueError"),
        {
            "python_exception": "ValueError",
            "exception_path": "pyo3::exceptions::PyValueError",
            "requires_message": True,
        },
    ),
    (
        "rsb_extract_arg",
        lambda _: boundary.rsb_extract_arg(0, "a", "i64"),
        {
            "index": 0,
            "name": "a",
            "rust_type": "i64",
            "type_error": "argument 'a' at position 0 must be i64",
        },
    ),
    (
        "rsb_extract_plan",
        lambda _: boundary.rsb_extract_plan([("a", "i64")]),
        {"params": ("a",), "rust_types": {"a": "i64"}, "line_count": 3},
    ),
    (
        "rsb_lifetime_annotations",
        lambda _: boundary.rsb_lifetime_annotations("&str"),
        {
            "text": "&'py str",
            "lifetimes": ("'py",),
            "owned_alternative": "String",
            "borrowed": True,
        },
    ),
    (
        "rsb_none_marker",
        lambda _: boundary.rsb_none_marker("Option<i64>"),
        {
            "is_option": True,
            "inner_type": "i64",
            "none_expr": "None",
            "some_expr": "Some(value)",
        },
    ),
    (
        "rsb_numeric_narrow_check",
        lambda _: boundary.rsb_numeric_narrow_check(127, "i8"),
        {"value": 127, "min": -128, "max": 127, "fits": True},
    ),
    (
        "rsb_ownership_mode",
        lambda _: boundary.rsb_ownership_mode("&str"),
        {
            "mode": "borrow",
            "family": "reference",
            "crosses_gil_release": False,
        },
    ),
    (
        "rsb_py_ref_guard",
        lambda _: boundary.rsb_py_ref_guard("handle"),
        {
            "rust_type": "Py<PyAny>",
            "owned": True,
            "drop_line": "drop(__py_guard_handle);",
            "drop_on_scope_exit": True,
        },
    ),
    (
        "rsb_python_type_map",
        lambda _: boundary.rsb_python_type_map("int"),
        {"source": "int", "rust_type": "i64", "direction": "to_rust", "generic": False},
    ),
    (
        "rsb_result_plan",
        lambda _: boundary.rsb_result_plan("Result<i64, PillarError>"),
        {
            "ok_type": "i64",
            "error_type": "PillarError",
            "python_exception": "PillarError",
            "signature": "PyResult<i64>",
        },
    ),
    (
        "rsb_return_convert",
        lambda _: boundary.rsb_return_convert("i64"),
        {"code": "Ok(value)", "needs_py_token": False},
    ),
    (
        "rsb_rust_type_classify",
        lambda _: boundary.rsb_rust_type_classify("Vec<String>"),
        {
            "base": "Vec",
            "family": "sequence",
            "args": ("String",),
            "is_generic": True,
            "by_reference": False,
        },
    ),
    (
        "rsb_string_encode",
        lambda _: boundary.rsb_string_encode('a"b'),
        {"value": 'a"b', "escaped": True, "utf8_bytes": 3},
    ),
    (
        "rsb_type_compat",
        lambda _: boundary.rsb_type_compat("i32", "i64"),
        {"compatible": True, "coercion": "checked_cast"},
    ),
    (
        "rsb_validate_values",
        lambda _: boundary.rsb_validate_values(
            [("count", "u8"), ("names", "Vec<String>")],
            {"count": 1, "names": ["a"]},
        ),
        {
            "values": {"count": 1, "names": ["a"]},
            "checked": 2,
            "skipped_optional": 0,
            "diagnostics": (),
        },
    ),
    # 0461-0480: GIL and RAII helpers.
    (
        "rsg_allow_threads_block",
        lambda _: gil.rsg_allow_threads_block("compute(1)"),
        {
            "result_type": "i64",
            "owned": (),
            "releases_gil": True,
            "python_apis_detected": False,
        },
    ),
    (
        "rsg_async_send_check",
        lambda _: gil.rsg_async_send_check("let x = 1;"),
        {
            "send": True,
            "static": True,
            "captures": (),
            "requirements": ("Send", "'static", "no borrowed Bound<T>"),
        },
    ),
    (
        "rsg_borrow_overlap_check",
        lambda _: gil.rsg_borrow_overlap_check(["a"], ["b"]),
        {"overlap": (), "borrowed": ("a",), "owned": ("b",), "distinct": 2},
    ),
    (
        "rsg_borrow_snapshot",
        lambda _: gil.rsg_borrow_snapshot("view"),
        {
            "code": "let __gil_owned_view = view.clone().unbind();",
            "owned_type": "Py<PyAny>",
            "crosses_gil_release": True,
        },
    ),
    (
        "rsg_critical_section",
        lambda _: gil.rsg_critical_section("update()", reason="shared state"),
        {"gil_held": True, "reason": "shared state"},
    ),
    (
        "rsg_detach_attach",
        lambda _: gil.rsg_detach_attach("compute()"),
        {"reacquires_gil": True, "python_apis_in_detached_work": False},
    ),
    (
        "rsg_drop_order",
        lambda _: gil.rsg_drop_order(["a", "b", "c"]),
        {
            "reverse": ("c", "b", "a"),
            "code": "drop(c);\ndrop(b);\ndrop(a);",
            "drops": 3,
        },
    ),
    (
        "rsg_gil_contention_model",
        lambda _: gil.rsg_gil_contention_model(4, releases=3),
        {"threads": 4, "wait_steps": 9, "throughput": 0.4},
    ),
    (
        "rsg_gil_plan",
        lambda _: gil.rsg_gil_plan("compute(1)", owned=["handle"]),
        {
            "releases_gil": True,
            "blocked": (),
            "result_type": "i64",
            "rules": (
                "clone every borrowed Bound<T> into Py<T> before allow_threads",
                "never call Python APIs inside the released-GIL closure",
                "re-acquire the GIL before touching Python objects again",
            ),
        },
    ),
    (
        "rsg_gil_state",
        lambda _: gil.rsg_gil_state(["release", "acquire"]),
        {"state": "held", "held": True, "event_count": 2, "balanced": True},
    ),
    (
        "rsg_guard_release_map",
        lambda _: gil.rsg_guard_release_map(
            {"lock1": {"kind": "lock", "acquire": "lock()"}}
        ),
        {
            "names": ("lock1",),
            "release_order": ("lock1",),
            "count": 1,
            "guards": {
                "lock1": {
                    "kind": "lock",
                    "acquire": "lock()",
                    "release": "drop(__resource);",
                }
            },
        },
    ),
    (
        "rsg_lifetime_erasure",
        lambda _: gil.rsg_lifetime_erasure("view"),
        {
            "code": "let __owned_view = view.clone().unbind();",
            "from": "Bound<'py, T>",
            "to": "Py<T>",
            "erased": True,
        },
    ),
    (
        "rsg_owned_handle",
        lambda _: gil.rsg_owned_handle("handle"),
        {
            "rust_type": "Py<PyAny>",
            "increments_refcount": True,
            "drop_line": "drop(__gil_move_handle);",
        },
    ),
    (
        "rsg_ownership_audit",
        lambda _: gil.rsg_ownership_audit(
            "// before\npy.allow_threads(|| {\n    compute(1)\n});\n"
        ),
        {"ok": True, "findings": (), "code_lines": 4},
    ),
    (
        "rsg_py_ref_count",
        lambda _: gil.rsg_py_ref_count(1, clones=2, drops=1),
        {"final": 2, "balanced": False, "leaked": True},
    ),
    (
        "rsg_raii_guard_def",
        lambda _: gil.rsg_raii_guard_def(),
        {
            "guard_type": "PolyglotPyGuard",
            "resource_type": "Py<PyAny>",
            "has_drop_impl": True,
        },
    ),
    (
        "rsg_raii_guard_use",
        lambda _: gil.rsg_raii_guard_use("state"),
        {
            "name": "state",
            "kind": "lock",
            "acquire": "lock.lock().unwrap()",
            "release": "drop(__resource);",
            "drop_line": "drop(__raii_state);",
        },
    ),
    (
        "rsg_release_plan",
        lambda _: gil.rsg_release_plan(
            "compute(handle, view)", borrowed=["view"], owned=["handle"]
        ),
        {
            "safe": ("handle",),
            "blocked": ("view",),
            "overlap": (),
            "releases_gil": True,
        },
    ),
    (
        "rsg_reentrancy_check",
        lambda _: gil.rsg_reentrancy_check("compute(1)"),
        {"safe": True, "findings": (), "python_apis_detected": False},
    ),
    (
        "rsg_with_gil_block",
        lambda _: gil.rsg_with_gil_block("read()"),
        {"reacquires_gil": True, "line_count": 4},
    ),
    # 0481-0500: model-based Cargo checks.
    (
        "rsk_cargo_lock_check",
        lambda _: checks.rsk_cargo_lock_check(
            ".", lock_text='version = "3"\n\n[[package]]\nname = "demo"\n'
        ),
        {"present": True, "version": 3, "packages": 1},
    ),
    (
        "rsk_check_crate",
        lambda path: _crate_with_manifest(path),
        {"ok": True, "step_names": ("check", "clippy")},
    ),
    (
        "rsk_check_fingerprint",
        lambda _: checks.rsk_check_fingerprint(
            [{"name": "check", "ok": True}, {"name": "clippy", "ok": True}]
        ),
        {"algorithm": "sha256", "step_count": 2},
    ),
    (
        "rsk_check_report",
        lambda _: checks.rsk_check_report(
            [
                {"name": "check", "ok": True, "diagnostics": ()},
                {"name": "clippy", "ok": False, "diagnostics": ("warning: unused",)},
            ]
        ),
        {
            "ok": False,
            "total": 2,
            "names": ("check", "clippy"),
            "failed": ("clippy",),
            "diagnostics": ("warning: unused",),
        },
    ),
    (
        "rsk_check_step",
        lambda path: checks.rsk_check_step(
            "check",
            ("cargo", "check"),
            path,
            {"CARGO_TARGET_DIR": str(path / "target")},
        ),
        {"name": "check", "returncode": 0, "ok": True, "diagnostics": ()},
    ),
    (
        "rsk_command_plan",
        lambda path: checks.rsk_command_plan(
            path,
            target_dir=path / "isolated",
            clippy=True,
            deny_warnings=True,
            extra_args=["--locked"],
        ),
        {
            "clippy": True,
            "deny_warnings": True,
            "extra_args": ("--locked",),
            "check_command": (
                "cargo",
                "check",
                "--all-targets",
                "--target-dir",
                _endswith("/isolated/check"),
                "--locked",
            ),
        },
    ),
    (
        "rsk_deny_warnings",
        lambda _: checks.rsk_deny_warnings(("cargo", "clippy")),
        {
            "command": ("cargo", "clippy", "--", "-D", "warnings"),
            "source_command": ("cargo", "clippy"),
            "changed": True,
        },
    ),
    (
        "rsk_diagnostic_count",
        lambda _: checks.rsk_diagnostic_count(
            ["error[E001]: one", "warning: two", "error[E001]: three"]
        ),
        {
            "errors": 2,
            "warnings": 1,
            "total": 3,
            "by_code": {"E001": 2},
        },
    ),
    (
        "rsk_env_plan",
        lambda path: checks.rsk_env_plan(path / "target"),
        {
            "variables": {
                "CARGO_TARGET_DIR": _endswith("/target"),
                "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1",
            },
            "inherits_os_environment": False,
        },
    ),
    (
        "rsk_failure_classify",
        lambda _: checks.rsk_failure_classify(
            {"name": "check", "returncode": 0, "stderr": ""}
        ),
        {"ok": True, "code": None, "name": "check"},
    ),
    (
        "rsk_feature_matrix",
        lambda path: checks.rsk_feature_matrix(
            ["alpha", "beta"], target_dir=path / "features"
        ),
        {
            "features": ("alpha", "beta"),
            "ok": True,
            "count": 2,
            "target_dir": _endswith("/features"),
        },
    ),
    (
        "rsk_isolation_check",
        lambda path: checks.rsk_isolation_check(
            ("cargo", "check", "--target-dir", str(path / "target")),
            {"CARGO_TARGET_DIR": str(path / "target")},
        ),
        {
            "ok": True,
            "target_dir": _endswith("/target"),
            "argv_target": _endswith("/target"),
            "isolated": True,
        },
    ),
    (
        "rsk_manifest_check",
        lambda _: checks.rsk_manifest_check(
            '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n'
        ),
        {
            "name": "demo",
            "version": "0.1.0",
            "edition": "2021",
            "sections": ("package",),
            "ok": True,
        },
    ),
    (
        "rsk_model_runner",
        lambda path: checks.rsk_model_runner(
            ("cargo", "--version"), env={"CARGO_TARGET_DIR": str(path / "target")}
        ),
        {
            "model": True,
            "returncode": 0,
            "stdout": "cargo 1.90.0 (deterministic-model)\n",
            "target_dir": _endswith("/target"),
        },
    ),
    (
        "rsk_no_source_mutation",
        lambda _: checks.rsk_no_source_mutation({"a.rs": "fn main() {}\n"}),
        {"mutated": False, "changed": (), "file_count": 1},
    ),
    (
        "rsk_parse_diagnostics",
        lambda _: checks.rsk_parse_diagnostics("warning: unused\n", "error: broken\n"),
        {
            "diagnostics": ("warning: unused", "error: broken"),
            "errors": 1,
            "warnings": 1,
            "lines_scanned": 2,
        },
    ),
    (
        "rsk_replay_log",
        lambda path: checks.rsk_replay_log(
            [
                {
                    "name": "check",
                    "command": ["cargo", "check"],
                    "env": {"CARGO_TARGET_DIR": str(path / "target")},
                }
            ]
        ),
        {"ok": True, "replayed": 1, "matches": ("check",), "model_only": True},
    ),
    (
        "rsk_scaffold_check",
        lambda _: checks.rsk_scaffold_check(
            {"a.rs": "same\n"}, {"a.rs": "same\n", "b.rs": "extra\n"}, ignore_extra=True
        ),
        {
            "ok": True,
            "missing": (),
            "mismatched": (),
            "extra": ("b.rs",),
            "compared": 1,
            "ignore_extra": True,
        },
    ),
    (
        "rsk_target_dirs",
        lambda path: checks.rsk_target_dirs(path / "base"),
        {
            "base": _endswith("/base"),
            "check": _endswith("/base/check"),
            "clippy": _endswith("/base/clippy"),
            "isolated": True,
            "count": 2,
        },
    ),
    (
        "rsk_toolchain_probe",
        lambda path: checks.rsk_toolchain_probe(target_dir=path / "probe"),
        {"tool": "cargo", "version": "1.90.0", "ok": True},
    ),
)


def _assert_subset(actual: Any, expected: Any, path: str = "result") -> None:
    """Assert that *actual* contains the structured values in *expected*."""
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path} is not a mapping"
        for key, value in expected.items():
            assert key in actual, f"{path} has no {key!r}"
            _assert_subset(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)), f"{path} is not a sequence"
        assert len(actual) == len(expected), f"{path} has the wrong length"
        for index, (item, wanted) in enumerate(zip(actual, expected)):
            _assert_subset(item, wanted, f"{path}[{index}]")
    elif callable(expected):
        assert expected(actual), f"{path} failed its custom check"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_rollout_metadata_is_complete_and_ordered() -> None:
    assert len(FRAMEWORK_ROWS) == 100
    assert FRAMEWORK_ROWS[:10] == LEGACY_ROWS
    assert len({row[0] for row in FRAMEWORK_ROWS}) == 100
    assert all(row[3] == "tests.test_rust_rollout" for row in FRAMEWORK_ROWS)

    for name, module_name, _category, test_module in FRAMEWORK_ROWS:
        assert test_module == "tests.test_rust_rollout"
        module = importlib.import_module(module_name)
        function = getattr(module, name)
        assert callable(function)
        assert function.__doc__ and function.__doc__.strip()

    for module_name, (module, prefix, category, count) in NEW_MODULES.items():
        rows = [row for row in FRAMEWORK_ROWS[10:] if row[1] == module_name]
        assert len(rows) == count
        assert tuple(row[0] for row in rows) == tuple(module.__all__)
        assert all(row[0].startswith(prefix) for row in rows)
        assert all(row[2] == category for row in rows)
        assert all(len(getattr(module, row[0]).__doc__.split()) >= 5 for row in rows)


def test_legacy_scaffold_functions_remain_behavioral(tmp_path: Path) -> None:
    generated = framework.rs_generate_pyo3_scaffold(module_name="legacy_demo")
    assert set(generated) == {"Cargo.toml", "pyproject.toml", "src/lib.rs"}
    for relative, text in generated.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    verified = framework.rs_verify_scaffold_matches_generator(
        tmp_path, module_name="legacy_demo", generated=generated
    )
    assert verified.ok and verified.mismatched == ()
    emitted = framework.rs_emit_monomorphized_structs(
        [
            framework.RsStructSpec(
                name="LegacyBox",
                fields=(framework.RsFieldSpec("value", "T"),),
                generic_params=("T",),
            )
        ],
        substitutions={"T": "i64"},
    )
    assert "pub struct LegacyBoxI64 {" in emitted
    impls = framework.rs_emit_trait_impls_and_bounds(
        [{"target": "LegacyBoxI64", "trait": "Clone"}]
    )
    assert "impl Clone for LegacyBoxI64 {" in impls


def test_legacy_boundary_functions_remain_behavioral() -> None:
    mapping = framework.rs_map_python_args_to_rust([("a", "i64")], values={"a": 3})
    assert "get_item(0)" in mapping.code
    assert mapping.values == {"a": 3}
    result = framework.rs_map_rust_results_to_py("Result<i64, PillarError>")
    assert result.ok_type == "i64" and result.python_exception == "PillarError"
    with pytest.raises(framework.RsPillarError) as error:
        framework.rs_raise_typed_pillar_error("E004", "cargo failed")
    assert error.value.code == "E004"


def test_legacy_gil_and_raii_functions_remain_behavioral() -> None:
    plan = framework.rs_manage_gil_and_py_refs("compute(1)")
    assert "py.allow_threads" in plan.code
    guards = framework.rs_emit_raii_guards(
        [{"name": "state", "kind": "lock", "acquire": "lock()"}]
    )
    assert "impl<T> Drop for PolyglotRaiiGuard<T> {" in guards
    assert "state" in guards


def test_legacy_cargo_check_uses_injected_runner(tmp_path: Path) -> None:
    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "legacy"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def runner(
        command: tuple[str, ...], cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd, env))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = framework.rs_run_cargo_check_and_clippy(
        crate, target_dir=tmp_path / "target", runner=runner
    )
    assert result.ok and len(calls) == 2
    assert calls[0][1] == crate and calls[1][1] == crate
    assert calls[0][2]["CARGO_TARGET_DIR"] == str(tmp_path / "target" / "check")


@pytest.mark.parametrize(
    ("name", "invoke", "expected"),
    BEHAVIORS,
    ids=[name for name, _invoke, _expected in BEHAVIORS],
)
def test_new_function_behavior(
    name: str, invoke: Invoke, expected: dict[str, Any], tmp_path: Path
) -> None:
    assert name in {row[0] for row in FRAMEWORK_ROWS[10:]}
    actual = invoke(tmp_path)
    assert isinstance(actual, dict)
    _assert_subset(actual, expected)


ERROR_CASES: tuple[tuple[str, Invoke, str], ...] = (
    (
        "E001_scaffold_edition",
        lambda _: scaffold.rsc_crate_manifest(edition="2015"),
        "E001",
    ),
    (
        "E001_boundary_integer_range",
        lambda _: boundary.rsb_numeric_narrow_check(256, "u8"),
        "E001",
    ),
    (
        "E001_gil_owned_handle_type",
        lambda _: gil.rsg_owned_handle("handle", rust_type="String"),
        "E001",
    ),
    (
        "E003_scaffold_indent",
        lambda _: scaffold.rsc_doc_comment("doc", indent="-"),
        "E003",
    ),
    (
        "E003_boundary_index",
        lambda _: boundary.rsb_extract_arg(-1, "a", "i64"),
        "E003",
    ),
    ("E003_gil_event", lambda _: gil.rsg_gil_state(["unknown"]), "E003"),
    (
        "E004_model_tool",
        lambda path: checks.rsk_model_runner(
            ("python", "check"), env={"CARGO_TARGET_DIR": str(path / "target")}
        ),
        "E004",
    ),
    (
        "E004_failed_step",
        lambda _: checks.rsk_failure_classify(
            {"name": "check", "returncode": 1, "stderr": "error: broken\n"}
        ),
        "E004",
    ),
    ("E006_missing_manifest", lambda path: checks.rsk_check_crate(path), "E006"),
    (
        "E007_scaffold_error_type",
        lambda _: scaffold.rsc_function_def(
            {"name": "bad", "result": "Result<i64, Mystery>", "body": "1"}
        ),
        "E007",
    ),
    (
        "E007_boundary_exception",
        lambda _: boundary.rsb_exception_map("Mystery"),
        "E007",
    ),
    (
        "E007_gil_python_api",
        lambda _: gil.rsg_reentrancy_check("py.eval('1')"),
        "E007",
    ),
    (
        "E008_scaffold_version",
        lambda _: scaffold.rsc_abi_settings(pyo3_version="bad"),
        "E008",
    ),
    (
        "E008_manifest_fields",
        lambda _: checks.rsk_manifest_check('[package]\nname = "demo"\n'),
        "E008",
    ),
    ("E008_report_steps", lambda _: checks.rsk_check_report(["bad"]), "E008"),
)


@pytest.mark.parametrize(
    ("name", "invoke", "code"),
    ERROR_CASES,
    ids=[name for name, _invoke, _code in ERROR_CASES],
)
def test_new_error_paths(name: str, invoke: Invoke, code: str, tmp_path: Path) -> None:
    assert name
    with pytest.raises(PolyglotError) as error:
        invoke(tmp_path)
    assert error.value.code == code
