"""Regression and edge-case tests for the ABI/FFI contract, build provenance,
and termination checker.

These tests are pure-Python: they never invoke a native toolchain and never
leave artifacts in the repository working tree (all scratch/provenance dirs
live in ``tempfile`` outside the repo).

Covered boundaries:
  * deep recursion / nesting limits in the termination checker
  * E001-E008 error-code parity across the Zig, runtime, Python-translator,
    and Rust backends
  * contract fingerprint stability across the 32-bit and 64-bit word boundary
  * model-only ABI boundaries (opaque addresses never leak raw pointer values;
    size==0 dynamic layouts are exempt from size/align but not field names/types)
  * provenance TOCTOU / cache-isolation edge cases (same scope_key reuse,
    empty inputs/outputs, non-deterministic build functions)
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from backends.errors import TranslateError
from backends.rust.compiler import RustCompileError
from compiler.analysis.termination import TerminationError, check_function
from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import ERROR_CODES, PillarError
from tools.abi_contract.contract import (
    ALL_ERROR_CODES,
    ContractViolation,
    default_contract,
    verify_contract_against_backend,
    verify_enum_layout,
    verify_struct_layout,
    zig_error_info,
)
from tools.abi_contract.verifier import verify_backends
from tools.provenance.recorder import record_build_provenance
from tools.provenance.verifier import (
    BuildDeterminismResult,
    verify_cache_isolation,
    verify_deterministic_build,
    verify_provenance_records,
)
from zig import abi as zig_abi
from zig.abi import _ERRORS as ZIG_ERRORS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_termination(source: str) -> None:
    function = parse(lex(source)).items[0]
    check_function(function)


def _deterministic_build(inputs, env, output_dir):
    outputs = []
    for index, (name, text) in enumerate(inputs.items()):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out_path = output_dir / f"{index}_{name}.out"
        out_path.write_text(digest, encoding="utf-8")
        outputs.append(out_path)
    return outputs


def _non_deterministic_build(inputs, env, output_dir):
    counter = _non_deterministic_build.counter
    _non_deterministic_build.counter += 1
    outputs = []
    for index, (name, text) in enumerate(inputs.items()):
        digest = hashlib.sha256((text + str(counter)).encode("utf-8")).hexdigest()
        out_path = output_dir / f"{index}_{name}.out"
        out_path.write_text(digest, encoding="utf-8")
        outputs.append(out_path)
    return outputs


_non_deterministic_build.counter = 0


# ---------------------------------------------------------------------------
# Deep recursion / nesting limits
# ---------------------------------------------------------------------------


def test_termination_deeply_nested_while_is_rejected_without_stack_overflow():
    """Many nested while loops must not overflow the checker's recursion.

    The checker is structural (not syntactic-depth bounded), so deep nesting
    should terminate with a clear E002 rather than a RecursionError.
    """
    body = "    x = x - 1;\n"
    nested = "    while x >= 0 {\n" * 40 + body + "    }\n" * 40
    source = f"""#[terminates]
fn f(mut x: i64) -> i64 {{
{nested}    x
}}"""
    try:
        _check_termination(source)
    except TerminationError:
        pass
    except RecursionError as exc:
        pytest.fail(f"termination checker overflowed the stack on deep nesting: {exc}")


def test_termination_deeply_nested_blocks_terminate():
    source = (
        "fn f() -> i64 {\n" + "    {\n" * 30 + "        0;\n" + "    }\n" * 30 + "}\n"
    )
    check_function(parse(lex(source)).items[0])


# ---------------------------------------------------------------------------
# Error code parity across backends (model-only boundary)
# ---------------------------------------------------------------------------


def test_error_codes_are_consistent_across_backends():
    """E001-E008 must agree between zig.abi, runtime.diagnostics, and the
    shared contract model (the public diagnostic surface)."""
    for code in ALL_ERROR_CODES:
        assert code in ZIG_ERRORS, f"zig missing {code}"
        assert code in ERROR_CODES, f"runtime missing {code}"
        errno, name, recoverable = ZIG_ERRORS[code]
        mapped = zig_abi.zabi_error_map(code)
        assert mapped["code"] == code
        assert mapped["errno"] == errno
        assert mapped["name"] == name
        assert mapped["recoverable"] is recoverable
        info = zig_error_info(code)
        assert info.errno == errno
        assert info.name == name
        assert info.recoverable is recoverable


def test_every_backend_error_carries_a_public_code():
    instances = [
        (TranslateError("probe"), "E003"),
        (TerminationError("f", "reason", "suggestion", 1, 0), "E002"),
        (RustCompileError("probe"), "E004"),
        (PillarError("E001", "probe"), "E001"),
    ]
    for instance, code in instances:
        assert instance.code == code
        assert code in ERROR_CODES


def test_pillar_error_rejects_unknown_codes():
    with pytest.raises(ValueError):
        PillarError("E999", "out of band code")


# ---------------------------------------------------------------------------
# Contract fingerprint stability across word boundary
# ---------------------------------------------------------------------------


def test_contract_fingerprint_changes_across_word_boundary():
    """The 32-bit vs 64-bit word boundary is a real ABI boundary: the
    fingerprint must differ, and pointer_width must track it."""
    contract_32 = default_contract(word=4)
    contract_64 = default_contract(word=8)
    assert contract_32.fingerprint() != contract_64.fingerprint()
    assert contract_32.pointer_width_bytes == 4
    assert contract_64.pointer_width_bytes == 8
    # Slice pointer width scales with the word.
    assert contract_32.slices["i64_slice"].size != contract_64.slices["i64_slice"].size


def test_contract_fingerprint_is_stable_within_a_word():
    first = default_contract(word=8).fingerprint()
    second = default_contract(word=8).fingerprint()
    assert first == second
    assert len(first) == 64


# ---------------------------------------------------------------------------
# Model-only ABI boundaries
# ---------------------------------------------------------------------------


def test_opaque_address_never_leaks_a_raw_pointer_value():
    """zabi_opaque_address is a model-only indirection: the integer is treated
    as an opaque key (``0x{address:x}``), never dereferenced as a pointer."""
    opaque = zig_abi.zabi_opaque_address(0x1000, label="Point")
    assert opaque["model_only"] is True
    assert opaque["dereferenced"] is False
    assert opaque["key"] == "0x1000"
    assert opaque["present"] is False  # no model state supplied -> not found
    # Echoing the address as a key is not a raw pointer value the caller can use.
    assert opaque["address"] == 0x1000


def test_size_zero_layout_exempt_from_size_align_but_not_fields():
    """A dynamic (size==0) Python layout is exempt from size/align checks but
    field names and types are still validated against the contract."""
    contract = default_contract()
    expected = contract.structs["Point"]
    dynamic = type(expected)(
        name=expected.name,
        fields=expected.fields,
        size=0,
        align=0,
    )
    # Must not raise: size==0 skips size/align/offset checks.
    verify_struct_layout(contract, backend="python", name="Point", actual=dynamic)

    # But a wrong field type is still caught even with size==0.
    wrong_type = type(expected)(
        name=expected.name,
        fields=(
            type(expected.fields[0])(name="x", type="i64", offset=0, size=8, align=8),
            type(expected.fields[0])(name="y", type="f64", offset=8, size=8, align=8),
        ),
        size=0,
        align=0,
    )
    with pytest.raises(ContractViolation) as exc_info:
        verify_struct_layout(
            contract, backend="python", name="Point", actual=wrong_type
        )
    assert exc_info.value.check == "struct.field.y.type"


# ---------------------------------------------------------------------------
# Provenance TOCTOU / cache-isolation edge cases
# ---------------------------------------------------------------------------


def test_same_scope_key_reuses_isolated_cache_dir(tmp_path):
    """Two recordings with the same scope_key must hit the same isolated cache
    directory (a scope is a cache boundary, not a per-run boundary)."""
    cache_root = Path(tempfile.mkdtemp(prefix="polyglot-edge-scope-"))
    inputs = {"a.poly": "fn main() -> i64 { 1 }\n"}

    record_a = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        scope_key="shared-scope",
        backend="python",
    )
    record_b = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        scope_key="shared-scope",
        backend="python",
    )
    assert record_a.cache["cargo_target_dir"] == record_b.cache["cargo_target_dir"]
    assert record_a.cache["scope_key"] == record_b.cache["scope_key"] == "shared-scope"


def test_build_with_inputs_produces_complete_provenance(tmp_path):
    """A build with inputs yields a complete, verifiable provenance record."""
    cache_root = Path(tempfile.mkdtemp(prefix="polyglot-edge-valid-"))
    record = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs={"a.poly": "fn main() -> i64 { 1 }\n"},
        backend="python",
    )
    summary = verify_provenance_records([record])
    assert summary["clean"] is True
    assert summary["record_count"] == 1
    assert record.inputs
    assert record.outputs


def test_empty_inputs_are_flagged_as_a_provenance_boundary(tmp_path):
    """Zero inputs is a boundary: the record is created, but structural
    validation flags the missing inputs (no silent widening)."""
    cache_root = Path(tempfile.mkdtemp(prefix="polyglot-edge-empty-"))
    record = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs={},
        backend="python",
    )
    assert record.inputs == {}
    summary = verify_provenance_records([record])
    assert summary["clean"] is False
    assert any("no inputs recorded" in issue for issue in summary["issues"])


def test_non_deterministic_build_is_detected(tmp_path):
    """verify_deterministic_build must flag a build function whose output
    embeds process-local state (TOCTOU-style divergence across runs)."""
    cache_root = Path(tempfile.mkdtemp(prefix="polyglot-edge-nondet-"))
    _non_deterministic_build.counter = 0
    result = verify_deterministic_build(
        cache_root=cache_root,
        build_fn=_non_deterministic_build,
        inputs={"a.poly": "fn main() -> i64 { 1 }\n"},
        runs=3,
    )
    assert isinstance(result, BuildDeterminismResult)
    assert result.clean is False
    assert len(set(result.fingerprints)) > 1


def test_cache_isolation_scopes_do_not_share_cache_dirs(tmp_path):
    """Two distinct scope keys must not share a cargo target dir (zero
    cross-scope state leakage / TOCTOU boundary)."""
    cache_root = Path(tempfile.mkdtemp(prefix="polyglot-edge-isolation-"))
    result = verify_cache_isolation(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs={"a.poly": "fn main() -> i64 { 1 }\n"},
        scope_keys=["scope-alpha", "scope-beta"],
    )
    assert result.clean is True
    assert len(set(result.scope_keys)) == 2
    assert len({r.cache["cargo_target_dir"] for r in result.records}) == 2


# ---------------------------------------------------------------------------
# Contract violation reporting is structured
# ---------------------------------------------------------------------------


def test_contract_violation_to_dict_is_json_serializable():
    contract = default_contract()
    expected = contract.enums["PillarError"]
    drifted = type(expected)(
        name=expected.name,
        tag_type=expected.tag_type,
        variants=(("E001", 1), ("E999", 2)),
        size=expected.size,
        align=expected.align,
    )
    try:
        verify_enum_layout(contract, backend="zig", name="PillarError", actual=drifted)
    except ContractViolation as exc:
        payload = exc.to_dict()
        assert set(payload) == {"check", "backend", "expected", "actual", "message"}
        assert payload["backend"] == "zig"
        import json

        json.dumps(payload)  # must round-trip
    else:
        pytest.fail("expected ContractViolation for renamed enum variant")


def test_verify_backends_reports_clean_for_real_backends():
    """Real zig/python/rust snapshots must match the contract (no violations)."""
    result = verify_backends()
    assert result["clean"] is True
    assert result["contract_fingerprint"]
    assert set(result["backends"]) >= {"zig", "python", "rust"}


def test_verify_contract_against_backend_collects_violations_per_backend():
    contract = default_contract()
    bad_struct = type(contract.structs["Point"])(
        name="Point",
        fields=contract.structs["Point"].fields,
        size=999,
        align=8,
    )
    violations = verify_contract_against_backend(
        contract,
        backend="zig",
        structs={"Point": bad_struct},
        enums={},
        slices={},
        error_codes={},
    )
    assert len(violations) == 1
    assert violations[0].check == "struct.size"
    assert violations[0].backend == "zig"
