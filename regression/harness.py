"""Callable FRAME 10 regression harnesses (functions 94-100).

The ``test_*`` names are intentional: the registry resolves these functions by
name.  This module is deliberately not named ``test_*.py`` so pytest does not
collect the harnesses themselves; ``tests/test_framework_f10.py`` supplies the
pytest tests that call them.

Every harness returns a structured dictionary.  Native compilation is opt-in
through an injected ``native_namespace`` (the ``native_ns`` fixture in the
pytest tests) or ``allow_native_compile=True`` for a standalone run; no harness
starts Cargo unless the caller asks for it.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Native sources shared with tests/test_framework_f10.py via ``SOURCES``.
# Names and types are prefixed so the session-wide native build cannot collide
# with other test modules.
F10_SOURCES: dict[str, str] = {
    "f10_recursive_struct": """
struct F10Node<T> {
    value: T,
    children: Vec<F10Node<T>>,
}
fn f10_recursive_struct() -> i64 {
    let leaf: F10Node<i64> = F10Node { value: 1, children: vec![] };
    let root: F10Node<i64> = F10Node { value: 2, children: vec![leaf] };
    root.value + root.children.len() as i64
}
""",
    "f10_recursive_enum": """
enum F10Tree<T> {
    Leaf(T),
    Node(Vec<F10Tree<T>>),
}
fn f10_recursive_enum() -> i64 {
    let tree: F10Tree<i64> = F10Tree::Node(vec![F10Tree::Leaf(1), F10Tree::Leaf(2)]);
    match tree {
        F10Tree::Leaf(value) => value,
        F10Tree::Node(children) => children.len() as i64,
    }
}
""",
    "f10_box_bounds": """
struct F10Box<T: Clone> { value: T }
impl<T: Clone> F10Box<T> {
    fn new(value: T) -> F10Box<T> { F10Box { value: value } }
    fn get(&self) -> T { self.value.clone() }
}
fn f10_box_bounds() -> i64 {
    let a: F10Box<i64> = F10Box::new(21);
    let b: F10Box<String> = F10Box::new("abc".to_string());
    a.get() + b.get().len() as i64
}
""",
    "f10_identity_bounds": """
fn f10_identity<T>(x: T) -> T { x }
fn f10_identity_bounds() -> i64 {
    f10_identity::<i64>(20) + f10_identity(22)
}
""",
    "f10_impl_omit_generics": """
struct F10Wrap<T> { value: T }
impl F10Wrap<T> {
    fn get(&self) -> T { self.value.clone() }
}
fn f10_impl_omit_generics() -> i64 {
    let wrap: F10Wrap<i64> = F10Wrap { value: 42 };
    wrap.get()
}
""",
}

_EXPECTED = {
    "f10_recursive_struct": 3,
    "f10_recursive_enum": 2,
    "f10_box_bounds": 24,
    "f10_identity_bounds": 42,
    "f10_impl_omit_generics": 42,
}


def _error_text(error: BaseException) -> str:
    """Return a concise, deterministic exception description."""
    return f"{type(error).__name__}: {error}"


def _import(module: str) -> tuple[Any | None, str | None]:
    """Import an optional module, returning ``(module, error)``."""
    try:
        return importlib.import_module(module), None
    except Exception as error:  # optional frame modules may be absent
        return None, _error_text(error)


def _parse(source: str) -> Any:
    """Parse Polyglot source with the repository lexer/parser."""
    lexer, parser = (
        _import("compiler.core.lexer")[0],
        _import("compiler.core.parser")[0],
    )
    if lexer is None or parser is None:
        raise RuntimeError("lexer/parser are unavailable")
    return parser.parse(lexer.lex(source))


def _python_call(source: str, function: str, expected: Any) -> dict[str, Any]:
    """Compile and execute ``source`` with the Python backend."""
    try:
        backends = _import("backends")[0]
        if backends is None:
            raise RuntimeError("backends package is unavailable")
        namespace = backends.get_backend("python", trusted=True).compile(_parse(source))
        if function not in namespace or not callable(namespace[function]):
            raise KeyError(f"compiled namespace has no callable {function!r}")
        value = namespace[function]()
        return {
            "status": "ok",
            "value": value,
            "expected": expected,
            "matches": value == expected,
            "exports": sorted(name for name in namespace if name.startswith("f10_")),
        }
    except Exception as error:
        return {
            "status": "error",
            "error": _error_text(error),
            "expected": expected,
            "matches": False,
            "exports": [],
        }


def _native_call(
    source: str,
    function: str,
    expected: Any,
    namespace: Mapping[str, Any] | None,
    allow_native_compile: bool,
) -> dict[str, Any]:
    """Run a native check from an injected namespace or an opt-in build."""
    if namespace is not None:
        if function not in namespace or not callable(namespace[function]):
            return {
                "status": "error",
                "error": f"native namespace has no callable {function!r}",
                "expected": expected,
                "matches": False,
            }
        value = namespace[function]()
        return {
            "status": "ok",
            "value": value,
            "expected": expected,
            "matches": value == expected,
            "source": "injected-native-namespace",
        }
    if not allow_native_compile:
        return {
            "status": "skipped",
            "reason": "inject native_namespace or set allow_native_compile=True",
            "expected": expected,
            "matches": False,
        }
    if shutil.which("cargo") is None:
        return {
            "status": "skipped",
            "reason": "cargo is not on PATH",
            "expected": expected,
            "matches": False,
        }
    try:
        backends = _import("backends")[0]
        if backends is None:
            raise RuntimeError("backends package is unavailable")
        compiled = backends.get_backend("native", trusted=True).compile(
            _parse(source), incremental=True, cache_scope=f"f10-{function}"
        )
        if function not in compiled or not callable(compiled[function]):
            raise KeyError(f"native namespace has no callable {function!r}")
        value = compiled[function]()
        return {
            "status": "ok",
            "value": value,
            "expected": expected,
            "matches": value == expected,
            "source": "native-backend-compile",
        }
    except Exception as error:
        return {
            "status": "error",
            "error": _error_text(error),
            "expected": expected,
            "matches": False,
        }


def _monomorph_check(source: str, required_names: tuple[str, ...]) -> dict[str, Any]:
    """Verify that generic specializations resolve without residual parameters."""
    module, error = _import("compiler.analysis.monomorph")
    if module is None:
        return {"status": "unavailable", "reason": error}
    try:
        program = module.monomorphize(_parse(source))
        names = sorted(item.name for item in program.items if hasattr(item, "name"))
        residual = sorted(
            getattr(item, "name", "?")
            for item in program.items
            if getattr(item, "generics", None)
        )
        missing = [name for name in required_names if name not in names]
        return {
            "status": "ok" if not missing and not residual else "error",
            "names": names,
            "required": list(required_names),
            "missing": missing,
            "residual_generics": residual,
        }
    except Exception as failure:
        return {"status": "error", "error": _error_text(failure)}


def _parity_result(
    function: str,
    source_key: str,
    expected: Any,
    native_namespace: Mapping[str, Any] | None,
    allow_native_compile: bool,
    required_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build the common Python/native parity result for one source."""
    source = F10_SOURCES[source_key]
    python = _python_call(source, source_key, expected)
    native = _native_call(
        source, source_key, expected, native_namespace, allow_native_compile
    )
    monomorph = _monomorph_check(source, required_names)
    parity = (
        python["status"] == "ok"
        and native["status"] == "ok"
        and python["matches"]
        and native["matches"]
        and python["value"] == native["value"]
    )
    ok = parity and monomorph["status"] == "ok"
    return {
        "function": function,
        "expected": expected,
        "python": python,
        "native": native,
        "checks": {"monomorphization": monomorph},
        "parity": parity,
        "ok": ok,
        "summary": (
            f"{source_key}: python={python['status']}, native={native['status']}, "
            f"monomorph={monomorph['status']}, parity={parity}"
        ),
    }


def test_recursive_generic_structs(
    *,
    native_namespace: Mapping[str, Any] | None = None,
    allow_native_compile: bool = False,
) -> dict[str, Any]:
    """Check recursive generic struct expansion on both backends (function 94).

    ``Node<T>`` contains ``Vec<Node<T>>``; the check compiles and runs the
    probe on Python and native, and verifies that monomorphization produces a
    concrete ``F10Node`` declaration with no residual type parameters.
    """
    return _parity_result(
        "test_recursive_generic_structs",
        "f10_recursive_struct",
        _EXPECTED["f10_recursive_struct"],
        native_namespace,
        allow_native_compile,
        ("F10Node",),
    )


def test_recursive_generic_enums(
    *,
    native_namespace: Mapping[str, Any] | None = None,
    allow_native_compile: bool = False,
) -> dict[str, Any]:
    """Check recursive generic enum expansion including ``Tree<T>`` (95).

    The ``F10Tree<T>`` probe builds a node with two leaves and matches both
    variants; results and specialized names must agree across backends.
    """
    return _parity_result(
        "test_recursive_generic_enums",
        "f10_recursive_enum",
        _EXPECTED["f10_recursive_enum"],
        native_namespace,
        allow_native_compile,
        ("F10Tree",),
    )


def test_trait_bounds_and_impls(
    *,
    native_namespace: Mapping[str, Any] | None = None,
    allow_native_compile: bool = False,
) -> dict[str, Any]:
    """Check trait bounds and impl resolution on both backends (function 96).

    Covers a bounded generic struct (``T: Clone``), explicit and inferred
    generic call syntax, and an ``impl F10Wrap<T>`` block that omits the
    parameter list.  Bound parsing is checked structurally in the AST.
    """
    probes = (
        ("f10_box_bounds", ("F10Box", "F10Box_String")),
        ("f10_identity_bounds", ("f10_identity_i64",)),
        ("f10_impl_omit_generics", ("F10Wrap",)),
    )
    checks: dict[str, Any] = {}
    for key, required in probes:
        result = _parity_result(
            "test_trait_bounds_and_impls",
            key,
            _EXPECTED[key],
            native_namespace,
            allow_native_compile,
            required,
        )
        checks[key] = result

    bound_check: dict[str, Any] = {}
    try:
        ast_nodes = _import("compiler.core.ast_nodes")[0]
        program = _parse(F10_SOURCES["f10_box_bounds"])
        declaration = next(
            item
            for item in program.items
            if isinstance(item, ast_nodes.StructDecl) and item.name == "F10Box"
        )
        bounds = [bound.name for bound in declaration.generics[0].bounds]
        bound_check = {
            "status": "ok" if bounds == ["Clone"] else "error",
            "struct": "F10Box",
            "parameter": declaration.generics[0].name,
            "bounds": bounds,
        }
    except Exception as error:
        bound_check = {"status": "error", "error": _error_text(error)}

    ok = bound_check["status"] == "ok" and all(
        result["ok"] for result in checks.values()
    )
    return {
        "function": "test_trait_bounds_and_impls",
        "checks": {"bounds": bound_check, "probes": checks},
        "parity": all(result["parity"] for result in checks.values()),
        "ok": ok,
        "summary": (
            f"bounds={bound_check['status']}; "
            f"{sum(result['ok'] for result in checks.values())}/{len(checks)} "
            "probes agree across backends"
        ),
    }


def test_hotpatch_delegation_and_rollback(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    """Verify hotpatch delegation metadata and real in-process rollback (97).

    The capability registry must route the reconciled hotpatch names to
    ``runtime.hotpatch`` and the legacy shadow-reconcile name to
    ``backends.hotpatch``.  Rollback is exercised with the first-generation
    :class:`HotpatchManager` and the ``runtime.hotpatch`` journal only; no
    native extension or raw memory is used.
    """
    del root  # the check is deliberately backend-local and does not read docs
    expected_modules = {
        "hotpatch_live_dylib_trampoline": "runtime.hotpatch.trampoline",
        "hotpatch_cpython_frame_eval": "runtime.hotpatch.frames",
        "hotpatch_mio_reactor_state": "runtime.hotpatch.reactor",
        "hotpatch_rcu_ast_epoch": "runtime.hotpatch.rcu",
        "hotpatch_rollback_journal_commit": "runtime.hotpatch.journal",
        "hotpatch_shadow_heap_reconcile": "backends.hotpatch",
    }
    delegation: dict[str, Any] = {}
    capabilities, error = _import("runtime.capabilities")
    if capabilities is None:
        delegation = {"status": "unavailable", "reason": error, "entries": {}}
    else:
        entries: dict[str, Any] = {}
        for name, module in expected_modules.items():
            cap = capabilities.BY_NAME.get(name)
            entry = {
                "expected_module": module,
                "actual_module": getattr(cap, "module", None),
                "resolves": bool(
                    cap is not None and capabilities.resolve(cap) is not None
                ),
            }
            entry["status"] = (
                "ok"
                if entry["actual_module"] == module and entry["resolves"]
                else "error"
            )
            entries[name] = entry
        delegation = {
            "status": "ok"
            if all(item["status"] == "ok" for item in entries.values())
            else "error",
            "entries": entries,
        }

    old_source = "fn f10_hp_value() -> i64 { 1 }\n"
    new_source = "fn f10_hp_value() -> i64 { 2 }\n"
    manager_rollback: dict[str, Any] = {}
    try:
        backends = _import("backends")[0]
        translator = _import("backends.python.translator")[0]
        hotpatch = _import("backends.hotpatch")[0]
        if None in (backends, translator, hotpatch):
            raise RuntimeError("hotpatch/backend modules are unavailable")
        old_program = _parse(old_source)
        namespace = translator.PythonBackend(trusted=True).compile(old_program)
        manager = hotpatch.HotpatchManager(old_program, namespace)
        before = manager.call("f10_hp_value")
        applied = manager.patch(_parse(new_source))
        after = manager.call("f10_hp_value")
        restored_generation = manager.rollback()
        rolled_back = manager.call("f10_hp_value")
        values = [before, after, rolled_back]
        manager_rollback = {
            "status": "ok"
            if values == [1, 2, 1] and restored_generation == 1
            else "error",
            "values": values,
            "generation": applied.generation,
            "restored_generation": restored_generation,
        }
    except Exception as failure:
        manager_rollback = {"status": "error", "error": _error_text(failure)}

    journal_rollback: dict[str, Any] = {}
    try:
        runtime_hotpatch = _import("runtime.hotpatch")[0]
        if runtime_hotpatch is None:
            raise RuntimeError("runtime.hotpatch is unavailable")
        state = SimpleNamespace(value=1)
        journal = runtime_hotpatch.RollbackJournal()
        snapshot = journal.checkpoint(objects=[state], label="f10-rollback")
        state.value = 2
        verified = journal.verify(snapshot)
        generation = journal.rollback(snapshot)
        journal_rollback = {
            "status": "ok"
            if verified is False and state.value == 1 and generation == 1
            else "error",
            "verified_after_mutation": verified,
            "value_after_rollback": state.value,
            "generation": generation,
        }
    except Exception as failure:
        journal_rollback = {"status": "error", "error": _error_text(failure)}

    framework, framework_error = _import("runtime.hotpatch.framework")
    delegate_fn = getattr(framework, "hotpatch_delegate_legacy_backend", None)
    framework_check: dict[str, Any] = {
        "available": framework is not None,
        "callable": callable(delegate_fn),
        "reason": framework_error,
    }
    if callable(delegate_fn):
        framework_check["signature"] = str(inspect.signature(delegate_fn))

    ok = (
        delegation["status"] == "ok"
        and manager_rollback["status"] == "ok"
        and journal_rollback["status"] == "ok"
    )
    return {
        "function": "test_hotpatch_delegation_and_rollback",
        "checks": {
            "delegation": delegation,
            "manager_rollback": manager_rollback,
            "journal_rollback": journal_rollback,
            "frame6_delegate": framework_check,
        },
        "ok": ok,
        "summary": (
            f"delegation={delegation['status']}, "
            f"manager={manager_rollback['status']}, "
            f"journal={journal_rollback['status']}"
        ),
    }


def _error_fields(error: BaseException) -> dict[str, Any]:
    """Serialize a structured pillar error into comparable fields."""
    diagnostic = getattr(error, "diagnostic", None)
    return {
        "type": type(error).__name__,
        "code": getattr(error, "code", None),
        "message": getattr(diagnostic, "message", None),
        "reason": getattr(error, "reason", None),
        "help_text": getattr(error, "help_text", None),
        "note": getattr(diagnostic, "note", None),
        "line": getattr(error, "line", None),
        "col": getattr(error, "col", None),
        "span": getattr(error, "span", None),
        "formatted": str(error),
    }


def _capture_error(call: Callable[[], Any]) -> dict[str, Any]:
    """Call ``call`` and capture either its typed error or the failure to raise."""
    try:
        call()
    except Exception as error:
        return {"raised": True, "error": _error_fields(error)}
    return {"raised": False, "error": None}


def test_typed_pillar_error_parity(
    *, span: tuple[int, int] = (7, 21)
) -> dict[str, Any]:
    """Verify identical typed pillar errors on both backends (function 98).

    A rejected capability (E007), a supported capability's failed invocation
    (E008), and an invalid backend request (E003) are dispatched through the
    shared host bridge.  Every diagnostic field must match between backends.
    """
    capabilities, error = _import("runtime.capabilities")
    if capabilities is None:
        return {
            "function": "test_typed_pillar_error_parity",
            "ok": False,
            "span": list(span),
            "cases": {},
            "summary": f"runtime.capabilities unavailable: {error}",
        }

    cases: dict[str, Any] = {}
    calls: dict[str, Callable[[str], Any]] = {
        "rejected_capability": lambda backend: capabilities.invoke(
            "hotpatch_tls_slot_relocate", backend=backend, span=span
        ),
        "supported_failure": lambda backend: capabilities.invoke(
            "hotpatch_live_dylib_trampoline", backend=backend, span=span
        ),
        "invalid_backend": lambda backend: capabilities.reject(
            "hotpatch_live_dylib_trampoline", backend="wasm", span=span
        ),
    }
    expected_codes = {
        "rejected_capability": "E007",
        "supported_failure": "E008",
        "invalid_backend": "E003",
    }
    for name, call in calls.items():
        python = _capture_error(lambda call=call: call("python"))
        native = _capture_error(lambda call=call: call("native"))
        code = expected_codes[name]
        matched = (
            python["raised"]
            and native["raised"]
            and python["error"] == native["error"]
            and python["error"]["code"] == code
            and python["error"]["span"] == span
        )
        cases[name] = {
            "python": python,
            "native": native,
            "expected_code": code,
            "matches": matched,
        }
    ok = all(case["matches"] for case in cases.values())
    return {
        "function": "test_typed_pillar_error_parity",
        "ok": ok,
        "span": list(span),
        "cases": cases,
        "summary": f"{sum(case['matches'] for case in cases.values())}/{len(cases)} cases match",
    }


@contextmanager
def _workspace(tmp_dir: Path | str | None, label: str) -> Iterator[Path]:
    """Yield a scratch directory, cleaning up only when one was created here."""
    if tmp_dir is not None:
        base = Path(tmp_dir)
        base.mkdir(parents=True, exist_ok=True)
        path = base / label
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)
        return
    with tempfile.TemporaryDirectory(prefix=f"f10-{label}-") as directory:
        yield Path(directory) / label


def _normalize_rust(text: str) -> str:
    """Ignore formatting noise when comparing generated Rust sources."""
    return re.sub(r",\)", ")", re.sub(r"\s+", "", text))


def _scaffold_drift(generated: Mapping[str, str], actual_dir: Path) -> list[str]:
    """List generated files that are missing or no longer match."""
    drift: list[str] = []
    for relative, expected in generated.items():
        path = actual_dir / relative
        if not path.is_file():
            drift.append(relative)
            continue
        actual = path.read_text(encoding="utf-8")
        if _normalize_rust(actual) != _normalize_rust(expected):
            drift.append(relative)
    return drift


def test_scaffold_drift_detection(
    *,
    root: Path | str | None = None,
    tmp_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Verify that an unregenerated scaffold edit is detected (function 99).

    The expected scaffold is generated in memory from ``runtime.capabilities``;
    clean and hand-edited copies are compared in temporary directories.  If
    frame 5 is present its verifier is recorded as an independent cross-check,
    but the harness never depends on that optional module.
    """
    del root  # the generator is imported from the installed runtime package
    module, module_error = _import("runtime.native_bindings")
    if module is None:
        return {
            "function": "test_scaffold_drift_detection",
            "ok": False,
            "summary": f"runtime.native_bindings unavailable: {module_error}",
        }
    generated = {
        "Cargo.toml": module.generate_cargo_toml(),
        "src/lib.rs": module.generate_lib_rs(),
    }
    clean_drift: list[str] = []
    mutated_drift: list[str] = []
    with (
        _workspace(tmp_dir, "clean") as clean,
        _workspace(tmp_dir, "mutated") as mutated,
    ):
        module.write_crate(clean)
        module.write_crate(mutated)
        clean_drift = _scaffold_drift(generated, clean)
        lib_rs = mutated / "src" / "lib.rs"
        lib_rs.write_text(
            lib_rs.read_text(encoding="utf-8")
            + "\n// Hand-edited without regenerating the scaffold.\n",
            encoding="utf-8",
        )
        mutated_drift = _scaffold_drift(generated, mutated)

    framework, framework_error = _import("backends.rust.framework")
    verifier = getattr(framework, "rs_verify_scaffold_matches_generator", None)
    frame5_check: dict[str, Any] = {
        "available": framework is not None,
        "callable": callable(verifier),
        "reason": framework_error,
    }
    if callable(verifier):
        frame5_check["signature"] = str(inspect.signature(verifier))

    ok = not clean_drift and "src/lib.rs" in mutated_drift
    return {
        "function": "test_scaffold_drift_detection",
        "ok": ok,
        "generated_files": sorted(generated),
        "clean_drift": clean_drift,
        "mutated_drift": mutated_drift,
        "frame5_verifier": frame5_check,
        "summary": (
            f"clean drift={clean_drift}; mutated drift={mutated_drift}; detected={ok}"
        ),
    }


def test_isolated_cache_no_contamination(
    *, tmp_dir: Path | str | None = None
) -> dict[str, Any]:
    """Verify sequential cache scopes stay isolated in temporary dirs (100).

    The real :class:`NativeBackend` scope-directory and Cargo.toml writers are
    used, but Cargo is never invoked.  Two scopes and two generated sources are
    written in sequence and re-read to prove that neither manifests nor build
    keys leak across scopes.
    """
    compiler, compiler_error = _import("backends.rust.compiler")
    if compiler is None:
        return {
            "function": "test_isolated_cache_no_contamination",
            "ok": False,
            "summary": f"backends.rust.compiler unavailable: {compiler_error}",
        }
    with _workspace(tmp_dir, "cache") as work:
        backend = compiler.NativeBackend(cache_dir=work / "cache", trusted=True)
        scopes: dict[str, dict[str, Any]] = {}
        for scope, module_name in (
            ("f10-scope-a", "f10_scope_a"),
            ("f10-scope-b", "f10_scope_b"),
        ):
            scope_dir = backend._incremental_scope_dir(scope)
            scope_dir.mkdir(parents=True, exist_ok=True)
            backend._write_project_files(scope_dir, module_name)
            marker = hashlib.sha256(scope.encode("utf-8")).hexdigest()
            manifest = scope_dir / ".polyglot-manifest.json"
            manifest.write_text(
                json.dumps({"scope": scope, "marker": marker}, sort_keys=True),
                encoding="utf-8",
            )
            scopes[scope] = {
                "path": scope_dir.as_posix(),
                "module": module_name,
                "marker": marker,
                "cargo_toml": (scope_dir / "Cargo.toml").read_text(encoding="utf-8"),
                "reloaded": json.loads(manifest.read_text(encoding="utf-8")),
            }

        codegen_module = _import("backends.rust.codegen")[0]
        keys: dict[str, str] = {}
        if codegen_module is not None:
            for name, source in (
                ("f10_cache_a", "fn f10_cache_a() -> i64 { 1 }\n"),
                ("f10_cache_b", "fn f10_cache_b() -> i64 { 2 }\n"),
            ):
                codegen = codegen_module.RustCodegen(module_name=name)
                generated = codegen.generate(_parse(source))
                keys[name] = hashlib.sha256(generated.encode("utf-8")).hexdigest()[:16]

    contamination: list[str] = []
    first, second = scopes["f10-scope-a"], scopes["f10-scope-b"]
    if first["path"] == second["path"]:
        contamination.append("scope directories are identical")
    for scope, data in scopes.items():
        expected = {"scope": scope, "marker": data["marker"]}
        if data["reloaded"] != expected:
            contamination.append(f"{scope}: manifest was contaminated")
        if f'name = "{data["module"]}"' not in data["cargo_toml"]:
            contamination.append(f"{scope}: Cargo.toml lost its own module name")
        other = second if scope == "f10-scope-a" else first
        if f'name = "{other["module"]}"' in data["cargo_toml"]:
            contamination.append(f"{scope}: Cargo.toml contains the other module")
    if len(set(keys.values())) != len(keys):
        contamination.append("distinct generated sources produced the same build key")

    framework, framework_error = _import("tools.build_ci.framework")
    frame9_check = {
        "available": framework is not None,
        "reason": framework_error,
    }
    if framework is not None:
        frame9_check["callable"] = callable(
            getattr(framework, "ci_compute_cache_scope_key", None)
        )

    ok = not contamination and len(keys) == 2
    return {
        "function": "test_isolated_cache_no_contamination",
        "ok": ok,
        "scopes": scopes,
        "build_keys": keys,
        "cross_contamination": contamination,
        "frame9_helpers": frame9_check,
        "summary": (
            f"{len(scopes)} scope(s), {len(keys)} build key(s), "
            f"{len(contamination)} contamination record(s)"
        ),
    }


__all__ = [
    "F10_SOURCES",
    "test_hotpatch_delegation_and_rollback",
    "test_isolated_cache_no_contamination",
    "test_recursive_generic_enums",
    "test_recursive_generic_structs",
    "test_scaffold_drift_detection",
    "test_trait_bounds_and_impls",
    "test_typed_pillar_error_parity",
]
