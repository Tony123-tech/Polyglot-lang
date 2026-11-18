"""Contract tests for the unified 100-capability registry."""

from __future__ import annotations

import pytest

from runtime.capabilities import (
    BY_NAME,
    BY_SUBSYSTEM,
    CAPABILITIES,
    CAPABILITY_NAMES,
    SUBSYSTEMS,
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
from runtime.diagnostics import PillarError
from runtime.native_bindings import generate_cargo_toml, generate_lib_rs


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
    error = reject("hotpatch_cpython_frame_eval", span=(17, 42))
    assert error.code == "E007"
    assert error.span == (17, 42)
    assert "span 17..42" in str(error)
    assert "hotpatch_cpython_frame_eval" in str(error)


def test_unknown_capability_is_e006():
    with pytest.raises(PillarError) as excinfo:
        invoke("does_not_exist")
    assert excinfo.value.code == "E006"


def test_unknown_backend_is_e003():
    with pytest.raises(PillarError) as excinfo:
        invoke("jit_type_lattice_speculate", backend="wasm")
    assert excinfo.value.code == "E003"


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
        invoke("hotpatch_cpython_frame_eval", backend="native", node=node)
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == span_of(node)


def test_capability_namespace_is_identical_across_backends():
    python = capability_namespace("python")
    native = capability_namespace("native")
    assert list(python) == list(native) == list(CAPABILITY_NAMES)
    assert all(callable(fn) for fn in (*python.values(), *native.values()))


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
    assert "E007" in capability_signature("hotpatch_live_dylib_trampoline")
    assert capability_signature("jit_type_lattice_speculate", backend="native") == (
        capability_signature("jit_type_lattice_speculate", backend="python")
    )


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
    assert '"E007"' in source
    assert '"E006"' in source
    assert '"E008"' in source
    assert "extension-module" in generate_cargo_toml()
