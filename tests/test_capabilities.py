"""Contract tests for the unified 100-capability registry."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest

from runtime import capabilities
from runtime.capabilities import (
    BY_NAME,
    BY_SUBSYSTEM,
    CAPABILITIES,
    CAPABILITY_NAMES,
    SUBSYSTEMS,
    Capability,
    CapabilityStatus,
    capability_namespace,
    capability_signature,
    describe,
    install_capability_bridge,
    invoke,
    reject,
    resolve,
    span_of,
)
from runtime.diagnostics import ERROR_CODES, PillarError
from runtime.native_bindings import _rust_string, generate_cargo_toml, generate_lib_rs


def test_registry_has_exactly_100_unique_ordered_capabilities():
    assert len(CAPABILITIES) == 100
    assert len(CAPABILITY_NAMES) == len(set(CAPABILITY_NAMES)) == 100
    assert [cap.number for cap in CAPABILITIES] == list(range(1, 101))
    assert CAPABILITY_NAMES[0] == "hotpatch_live_dylib_trampoline"
    assert CAPABILITY_NAMES[-1] == "swarm_global_invariant_audit"


def test_each_subsystem_contributes_ten_capabilities():
    assert len(SUBSYSTEMS) == 10
    assert set(BY_SUBSYSTEM) == set(SUBSYSTEMS)
    assert all(len(BY_SUBSYSTEM[subsystem]) == 10 for subsystem in SUBSYSTEMS)


def test_every_non_rejected_capability_has_a_callable_adapter():
    missing = [
        cap.name
        for cap in CAPABILITIES
        if cap.status is not CapabilityStatus.REJECTED and resolve(cap) is None
    ]
    assert missing == []


def test_rejected_capabilities_are_uniformly_exposed():
    rejected = [cap for cap in CAPABILITIES if cap.status is CapabilityStatus.REJECTED]
    assert rejected
    for cap in rejected:
        assert cap.backends == ()
        assert resolve(cap) is None
        for backend in ("python", "native"):
            error = reject(cap.name, backend=backend)
            assert error.code == cap.code
            assert cap.name in str(error)


def test_all_rejections_preserve_code_and_span_on_both_backends():
    for cap in CAPABILITIES:
        if cap.status is not CapabilityStatus.REJECTED:
            continue
        for backend in ("python", "native"):
            with pytest.raises(PillarError) as excinfo:
                invoke(cap.name, backend=backend, span=(3, 9))
            assert excinfo.value.code == cap.code
            assert excinfo.value.span == (3, 9)


def test_span_is_preserved_in_structured_rejections():
    # A genuinely rejected capability: the reconciled hotpatch adapters are
    # now implemented and must not be used as rejection fixtures.
    error = reject("hotpatch_tls_slot_relocate", span=(17, 42))
    assert error.code == "E007"
    assert error.span == (17, 42)
    assert "span 17..42" in str(error)
    assert "hotpatch_tls_slot_relocate" in str(error)


def test_node_line_and_col_are_not_replaced_by_a_raw_span_offset():
    node = SimpleNamespace(span=(17, 42), line=7, col=4)
    error = reject("hotpatch_tls_slot_relocate", node=node)
    assert (error.line, error.col, error.span) == (7, 4, (17, 42))


def test_span_of_ignores_malformed_spans():
    assert span_of(SimpleNamespace(span=[1, 2])) == (1, 2)
    assert span_of(SimpleNamespace(span=("1", "2"))) == (1, 2)
    assert span_of(SimpleNamespace(span=(1, 2, 3))) is None
    assert span_of(SimpleNamespace(span=("x", 2))) is None
    assert span_of(None) is None


def test_unknown_capability_is_e006():
    with pytest.raises(PillarError) as excinfo:
        invoke("does_not_exist")
    assert excinfo.value.code == "E006"


@pytest.mark.parametrize("backend", ["wasm", "", None, "PYTHON"])
def test_invalid_backend_is_e003_everywhere(backend):
    cap = "hotpatch_tls_slot_relocate"
    with pytest.raises(PillarError) as excinfo:
        reject(cap, backend=backend, span=(1, 4))
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (1, 4)
    for call in (
        lambda: invoke(cap, backend=backend),
        lambda: capability_namespace(backend),
        lambda: capability_signature(cap, backend=backend),
    ):
        with pytest.raises(PillarError) as error:
            call()
        assert error.value.code == "E003"


def test_supported_capability_is_callable():
    signature = invoke("jit_type_lattice_speculate", (1, 2.0, True))
    assert signature.render() == "(int, float, bool)"


def test_span_is_forwarded_to_implementations_that_accept_it():
    from lexer import lex
    from parser import parse

    program = parse(lex("fn main() -> i64 { let xs = [1, 2, 3]; xs[0] }"))
    obligations = invoke("verify_no_panics_reachable", program)
    assert obligations[0].status == "proved"


def test_rejected_capability_raises_on_invoke_with_node_span():
    from lexer import lex
    from parser import parse

    node = parse(lex("fn main() -> i64 { 0 }"))
    with pytest.raises(PillarError) as excinfo:
        invoke("hotpatch_tls_slot_relocate", backend="native", node=node)
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == span_of(node)


def test_capability_namespace_is_identical_across_backends():
    python = capability_namespace("python")
    native = capability_namespace("native")
    assert list(python) == list(native) == list(CAPABILITY_NAMES)
    assert all(callable(fn) for fn in (*python.values(), *native.values()))


def test_namespace_returns_independent_mappings():
    first = capability_namespace()
    second = capability_namespace()
    assert first is not second
    first["jit_type_lattice_speculate"] = "mutated"
    assert callable(second["jit_type_lattice_speculate"])


def test_namespace_rejection_stubs_raise_instead_of_returning_a_error():
    for backend in ("python", "native"):
        stub = capability_namespace(backend)["hotpatch_tls_slot_relocate"]
        for call in (
            partial(stub),
            partial(stub, 1, 2, 3),
            partial(stub, span=(5, 9)),
            partial(stub, reason="explicit reason"),
        ):
            with pytest.raises(PillarError) as excinfo:
                call()
            assert excinfo.value.code == "E007"
        with pytest.raises(PillarError) as excinfo:
            stub(span=(5, 9))
        assert excinfo.value.span == (5, 9)
        with pytest.raises(PillarError) as excinfo:
            stub(reason="explicit reason")
        assert excinfo.value.reason == "explicit reason"


def test_install_bridge_does_not_overwrite_existing_symbols():
    namespace = {"jit_type_lattice_speculate": "original"}
    install_capability_bridge(namespace, overwrite=False)
    assert namespace["jit_type_lattice_speculate"] == "original"
    install_capability_bridge(namespace, overwrite=True)
    assert callable(namespace["jit_type_lattice_speculate"])


def test_describe_includes_status_signature_and_boundary():
    text = describe("sandbox_mprotect_heap_isolate")
    assert "mprotect" in text
    assert "signature:" in text
    assert "boundary:" in text
    assert BY_NAME["sandbox_mprotect_heap_isolate"].number == 71


def test_capability_signature_reflects_adapter_or_rejection():
    assert "args" in capability_signature("jit_type_lattice_speculate")
    assert "E007" in capability_signature("hotpatch_tls_slot_relocate")
    assert capability_signature("jit_type_lattice_speculate", backend="native") == (
        capability_signature("jit_type_lattice_speculate", backend="python")
    )


def test_capability_signature_handles_builtins_methods_and_no_signature(monkeypatch):
    def check(target: object, *, expected: str) -> None:
        cap = Capability(
            number=999,
            name="synthetic_cap",
            subsystem="test",
            module="unused",
            attr="unused",
            status=CapabilityStatus.FOUNDATION,
            code="E001",
            summary="",
            boundary="",
        )
        monkeypatch.setitem(BY_NAME, cap.name, cap)
        monkeypatch.setattr(capabilities, "resolve", lambda _cap: target)
        assert expected in capability_signature(cap.name)

    check(len, expected="synthetic_cap(")
    check(str.upper, expected="synthetic_cap(")

    def broken(*_args: object, **_kwargs: object) -> object:
        return None

    def raise_value_error(*_args: object, **_kwargs: object) -> object:
        raise ValueError("no signature available")

    monkeypatch.setattr(capabilities.inspect, "signature", raise_value_error)
    check(broken, expected="synthetic_cap(*args, **kwargs)")


def test_invoke_does_not_inject_span_into_a_positional_parameter(monkeypatch):
    def positional_span(span, value):
        return span, value

    cap = Capability(
        number=999,
        name="synthetic_cap",
        subsystem="test",
        module="unused",
        attr="unused",
        status=CapabilityStatus.FOUNDATION,
        code="E001",
        summary="",
        boundary="",
    )
    monkeypatch.setitem(BY_NAME, cap.name, cap)
    monkeypatch.setattr(capabilities, "resolve", lambda _cap: positional_span)
    assert invoke(cap.name, (10, 20), (30, 40), span=(1, 2)) == ((10, 20), (30, 40))


def test_resolve_distinguishes_missing_modules_from_broken_ones(monkeypatch):
    cap = Capability(
        number=999,
        name="synthetic_cap",
        subsystem="test",
        module="optional_plugin",
        attr="entry",
        status=CapabilityStatus.FOUNDATION,
        code="E001",
        summary="",
        boundary="",
    )
    resolve.cache_clear()

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda _name: None)
    assert resolve(cap) is None
    resolve.cache_clear()

    class FakeSpec:
        pass

    def boom(_name):
        raise ImportError("a dependency inside the installed module is broken")

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda _n: FakeSpec())
    monkeypatch.setattr(capabilities.importlib, "import_module", boom)
    monkeypatch.setattr(
        capabilities, "resolve", capabilities.resolve.__wrapped__
    )  # bypass the cache for the failing case
    with pytest.raises(ImportError, match="broken"):
        resolve(cap)

    def attribute_boom(_name):
        raise AttributeError("module body failed while being imported")

    monkeypatch.setattr(capabilities.importlib, "import_module", attribute_boom)
    with pytest.raises(AttributeError, match="module body failed"):
        resolve(cap)


def test_resolve_caches_adapter_lookups(monkeypatch):
    imports: list[str] = []
    real_import = capabilities.importlib.import_module

    def counting_import(name):
        imports.append(name)
        return real_import(name)

    resolve.cache_clear()
    monkeypatch.setattr(capabilities.importlib, "import_module", counting_import)
    cap = BY_NAME["jit_type_lattice_speculate"]
    first = resolve(cap)
    second = resolve(cap)
    assert first is second is not None
    assert imports == [cap.module]
    resolve.cache_clear()


# ---------------------------------------------------------------------------
# Reconciliation with runtime.hotpatch (five pillar-I capabilities)
# ---------------------------------------------------------------------------

_RECONCILED = {
    "hotpatch_live_dylib_trampoline": "runtime.hotpatch.trampoline",
    "hotpatch_cpython_frame_eval": "runtime.hotpatch.frames",
    "hotpatch_mio_reactor_state": "runtime.hotpatch.reactor",
    "hotpatch_rcu_ast_epoch": "runtime.hotpatch.rcu",
    "hotpatch_rollback_journal_commit": "runtime.hotpatch.journal",
}


@pytest.mark.parametrize(("name", "module"), _RECONCILED.items())
def test_reconciled_hotpatch_adapters_resolve(name, module):
    cap = BY_NAME[name]
    assert cap.status in (
        CapabilityStatus.FOUNDATION,
        CapabilityStatus.MODELLED,
    )
    assert cap.module == module
    assert cap.attr == name
    assert resolve(cap) is not None
    assert callable(capability_namespace()[name])


def _old_frame_target(value):
    extra = 1
    return value + extra


def _new_frame_target(value, second):
    return value + second


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("hotpatch_live_dylib_trampoline", ()),
        ("hotpatch_cpython_frame_eval", (_old_frame_target, _new_frame_target)),
        ("hotpatch_mio_reactor_state", (object(),)),
        ("hotpatch_rcu_ast_epoch", (object(),)),
        ("hotpatch_rollback_journal_commit", (object(),)),
    ],
)
def test_reconciled_hotpatch_adapters_forward_the_span(name, args):
    with pytest.raises(PillarError) as excinfo:
        invoke(name, *args, span=(2, 7))
    assert excinfo.value.code == "E008"
    assert excinfo.value.span == (2, 7)


def test_reconciled_hotpatch_boundaries_keep_the_real_e007_limits():
    trampoline_boundary = BY_NAME["hotpatch_live_dylib_trampoline"].boundary
    frame_boundary = BY_NAME["hotpatch_cpython_frame_eval"].boundary
    assert "foreign __TEXT" in trampoline_boundary
    assert "f_code" in frame_boundary

    from runtime.hotpatch.errors import HotpatchError
    from runtime.hotpatch.trampoline import LiveTrampoline

    with pytest.raises(HotpatchError) as excinfo:
        LiveTrampoline().patch_text(object(), 1, span=(4, 8))
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (4, 8)


def test_rollback_journal_adapter_returns_a_real_journal():
    from runtime.hotpatch.journal import RollbackJournal

    journal = invoke("hotpatch_rollback_journal_commit", span=(1, 3))
    assert isinstance(journal, RollbackJournal)


# ---------------------------------------------------------------------------
# Native scaffold
# ---------------------------------------------------------------------------


def test_native_binding_scaffold_covers_every_capability():
    import re
    from pathlib import Path

    source = generate_lib_rs()
    root = Path(__file__).resolve().parent.parent
    checked_in = (root / "native_pillars" / "src" / "lib.rs").read_text()

    # The checked-in file is rustfmt-formatted, so compare content rather than
    # formatting (rustfmt also drops trailing commas in short signatures).
    def normalize(text: str) -> str:
        return re.sub(r",\)", ")", re.sub(r"\s+", "", text))

    assert normalize(checked_in) == normalize(source)
    for name in CAPABILITY_NAMES:
        assert f'#[pyo3(name = "{name}")]' in source
        assert f"pub fn {name}(" in source
        assert f"wrap_pyfunction!({name}, m)" in source
    assert source.count("#[pyfunction]") == 100
    assert source.count("#[pyo3(signature = (span = None))]") == 100
    assert '"E007"' in source
    assert '"E006"' in source
    assert '"E008"' in source
    assert "extension-module" in generate_cargo_toml()


def test_rust_string_escapes_quotes_backslashes_and_control_characters():
    assert _rust_string('a"b\\c\nd\te\0f\x1f') == ('"a\\"b\\\\c\\nd\\te\\0f\\u{1f}"')
    # Non-ASCII text stays literal so rustc receives a valid UTF-8 literal.
    assert _rust_string("停顿") == '"停顿"'


def test_error_code_surface_is_fixed():
    assert set(ERROR_CODES) == {f"E{n:03d}" for n in range(1, 9)}
    with pytest.raises(TypeError):
        ERROR_CODES["E009"] = "not allowed"  # type: ignore[index]
    with pytest.raises(ValueError):
        PillarError("E009", "unknown code")


def test_pillar_error_drops_a_malformed_span_instead_of_crashing():
    error = PillarError("E003", "bad span", span=("x", 2))  # type: ignore[arg-type]
    assert error.span is None
    assert "at: line 0, column 0" in error.format()
