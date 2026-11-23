"""Tests for the Gleam-inspired static ADT, bit-array, and contract subsystems.

These tests are pure Python and do not require any external toolchain.  They
cover exhaustiveness witnesses, bit-array endian/bounds validation, and
cross-target capability contracts.
"""

from __future__ import annotations

import pytest

from compiler.frontend.errors import PolyglotError
from compiler.gleam_style import (
    AdtDef,
    AdtRegistry,
    BitSegmentKind,
    Capability,
    Target,
    TargetDescriptor,
    Variant,
    check_exhaustive,
    check_total_adt,
    emit_contract,
    match_bit_array,
    parse_bit_pattern,
)

# ---------------------------------------------------------------------------
# ADT exhaustiveness
# ---------------------------------------------------------------------------


def test_exhaustive_match_with_all_variants():
    adt = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
    result = check_exhaustive(adt, [("Some", ("x",)), ("None", ())])
    assert result.exhaustive is True
    assert result.missing is None
    assert set(result.covered) == {"Some", "None"}


def test_exhaustive_match_with_wildcard():
    adt = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
    result = check_exhaustive(adt, [("Some", ("x",))], has_wildcard=True)
    assert result.exhaustive is True


def test_non_exhaustive_match_reports_missing_witnesses():
    adt = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [("Some", ("x",))])
    assert exc.value.code == "E002"
    assert "None" in str(exc.value)


def test_non_exhaustive_match_reports_multiple_missing():
    adt = AdtDef(
        "Shape",
        (
            Variant("Circle", ("Float",)),
            Variant("Square", ("Float",)),
            Variant("Triangle", ("Float", "Float", "Float")),
        ),
    )
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [("Circle", ("r",))])
    assert exc.value.code == "E002"
    msg = str(exc.value)
    assert "Square" in msg
    assert "Triangle" in msg


def test_non_exhaustive_witness_includes_payload_types():
    adt = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [])
    # The missing witness for Some should include its payload type.
    assert "Some(Int)" in str(exc.value)


def test_unknown_variant_is_e001():
    adt = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [("Unknown", ())])
    assert exc.value.code == "E001"


def test_duplicate_variant_is_e001():
    # The AdtDef constructor itself does not validate; check_exhaustive does.
    adt = AdtDef("Bad", (Variant("A"), Variant("A")))
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [("A", ())])
    assert exc.value.code == "E001"


def test_empty_adt_is_e001():
    adt = AdtDef("Empty", ())
    with pytest.raises(PolyglotError) as exc:
        check_exhaustive(adt, [])
    assert exc.value.code == "E001"


def test_adt_registry_register_and_check():
    registry = AdtRegistry()
    adt = AdtDef("Bool", (Variant("True"), Variant("False")))
    registry.register(adt)
    assert registry.names() == ("Bool",)
    result = registry.check_match("Bool", [("True", ()), ("False", ())])
    assert result.exhaustive is True


def test_adt_registry_unknown_adt_is_e006():
    registry = AdtRegistry()
    with pytest.raises(PolyglotError) as exc:
        registry.check_match("Unknown", [])
    assert exc.value.code == "E006"


def test_adt_registry_duplicate_is_e001():
    registry = AdtRegistry()
    adt = AdtDef("Bool", (Variant("True"), Variant("False")))
    registry.register(adt)
    with pytest.raises(PolyglotError) as exc:
        registry.register(adt)
    assert exc.value.code == "E001"


# ---------------------------------------------------------------------------
# Total ADT / HM-style validation
# ---------------------------------------------------------------------------


def test_total_adt_simple():
    adt = AdtDef("Color", (Variant("Red"), Variant("Green"), Variant("Blue")))
    report = check_total_adt(adt)
    assert report["total"] is True
    assert report["adt"] == "Color"
    assert report["cycles"] == []


def test_total_adt_recursive_tree():
    adt = AdtDef(
        "Tree",
        (Variant("Leaf", ("Int",)), Variant("Node", ("Tree", "Tree"))),
        generics=("Int",),
    )
    report = check_total_adt(adt, bindings={"Tree": "Tree", "Int": "Int"})
    assert report["total"] is True
    # The Node variant references Tree, which is a cycle.
    assert len(report["cycles"]) > 0


def test_total_adt_budget_exceeded():
    adt = AdtDef(
        "Tree",
        (Variant("Leaf", ("Int",)), Variant("Node", ("Tree", "Tree"))),
        generics=("Int",),
    )
    with pytest.raises(PolyglotError) as exc:
        check_total_adt(adt, bindings={"Tree": "Tree", "Int": "Int"}, budget=1)
    assert exc.value.code == "E002"


# ---------------------------------------------------------------------------
# Bit-array pattern parsing and matching
# ---------------------------------------------------------------------------


def test_parse_bit_pattern_basic():
    pat = parse_bit_pattern("<<header:8, len:16, payload:binary>>")
    assert pat.fixed_bits == 24
    assert pat.has_variable is True
    assert len(pat.segments) == 3


def test_parse_bit_pattern_no_type_defaults_to_int():
    pat = parse_bit_pattern("<<header:8, len:16>>")
    assert pat.segments[0].kind == BitSegmentKind.INT
    assert pat.segments[1].kind == BitSegmentKind.INT
    assert pat.has_variable is False


def test_parse_bit_pattern_malformed_no_brackets():
    with pytest.raises(PolyglotError) as exc:
        parse_bit_pattern("header:8")
    assert exc.value.code == "E001"


def test_parse_bit_pattern_empty():
    with pytest.raises(PolyglotError) as exc:
        parse_bit_pattern("<<>>")
    assert exc.value.code == "E001"


def test_parse_bit_pattern_zero_width():
    with pytest.raises(PolyglotError) as exc:
        parse_bit_pattern("<<x:0>>")
    assert exc.value.code == "E001"


def test_match_bit_array_big_endian():
    pat = parse_bit_pattern("<<a:8, b:16>>")
    data = bytes([0x01, 0x02, 0x03])
    result = match_bit_array(pat, data)
    assert result.bindings["a"] == 1
    assert result.bindings["b"] == 0x0203


def test_match_bit_array_little_endian():
    pat = parse_bit_pattern("<<a:16:little>>")
    data = bytes([0x01, 0x02])
    result = match_bit_array(pat, data)
    assert result.bindings["a"] == 0x0201


def test_match_bit_array_variable_payload():
    pat = parse_bit_pattern("<<header:8, len:16, payload:binary>>")
    data = bytes([0xAB, 0x00, 0x05, 1, 2, 3, 4, 5])
    result = match_bit_array(pat, data)
    assert result.bindings["header"] == 0xAB
    assert result.bindings["len"] == 5
    assert result.bindings["payload"] == bytes([1, 2, 3, 4, 5])


def test_match_bit_array_too_short_raises_e001():
    pat = parse_bit_pattern("<<a:8, b:16>>")
    with pytest.raises(PolyglotError) as exc:
        match_bit_array(pat, bytes([0x01]))
    assert exc.value.code == "E001"


def test_match_bit_array_empty_buffer_variable():
    pat = parse_bit_pattern("<<payload:binary>>")
    result = match_bit_array(pat, b"")
    assert result.bindings["payload"] == b""


def test_match_bit_array_non_bytes_raises_e003():
    pat = parse_bit_pattern("<<a:8>>")
    with pytest.raises(PolyglotError) as exc:
        match_bit_array(pat, "not bytes")
    assert exc.value.code == "E003"


def test_match_bit_array_non_byte_aligned_fixed():
    # 12-bit int is not byte-aligned; should raise E001.
    pat = parse_bit_pattern("<<a:12:int>>")
    with pytest.raises(PolyglotError) as exc:
        match_bit_array(pat, bytes([0xAB, 0xCD]))
    assert exc.value.code == "E001"


def test_bit_pattern_too_wide_raises_e001():
    with pytest.raises(PolyglotError) as exc:
        parse_bit_pattern("<<" + ", ".join(f"x{i}:8" for i in range(600)) + ">>")
    assert exc.value.code == "E001"


def test_bit_pattern_variable_must_be_last():
    pat = parse_bit_pattern("<<a:binary, b:8>>")
    with pytest.raises(PolyglotError) as exc:
        match_bit_array(pat, bytes([1, 2, 3]))
    assert exc.value.code == "E001"


# ---------------------------------------------------------------------------
# Cross-target contracts
# ---------------------------------------------------------------------------


def test_contract_supported():
    contract = emit_contract("adt_test", {Capability.ADT}, Target.PYTHON)
    assert contract.supported is True
    assert contract.missing == frozenset()


def test_contract_unsupported_raises_e007():
    with pytest.raises(PolyglotError) as exc:
        emit_contract("actor_test", {Capability.OTP_ACTOR}, Target.PYTHON)
    assert exc.value.code == "E007"
    assert "otp_actor" in str(exc.value)


def test_contract_beam_supports_otp():
    contract = emit_contract("actor_test", {Capability.OTP_ACTOR}, Target.BEAM)
    assert contract.supported is True


def test_contract_override_capabilities():
    desc = TargetDescriptor(
        target=Target.PYTHON, overrides=frozenset({Capability.SIMD})
    )
    contract = emit_contract("simd_test", {Capability.SIMD}, desc)
    assert contract.supported is True


def test_contract_disable_capabilities():
    desc = TargetDescriptor(target=Target.ZIG, disabled=frozenset({Capability.SIMD}))
    with pytest.raises(PolyglotError) as exc:
        emit_contract("simd_test", {Capability.SIMD}, desc)
    assert exc.value.code == "E007"


def test_contract_empty_required_raises_e003():
    with pytest.raises(PolyglotError) as exc:
        emit_contract("empty", set(), Target.PYTHON)
    assert exc.value.code == "E003"


def test_contract_unknown_target_raises_e007():
    with pytest.raises(PolyglotError) as exc:
        emit_contract("test", {Capability.ADT}, "not_a_target")
    assert exc.value.code == "E007"


def test_target_from_name():
    assert Target.from_name("python") == Target.PYTHON
    assert Target.from_name("BEAM") == Target.BEAM


def test_capability_from_name():
    assert Capability.from_name("adt") == Capability.ADT
    assert Capability.from_name("SIMD") == Capability.SIMD
