"""Cross-backend ABI/FFI contract verification tests.

Covers struct/enum/slice layout drift detection and ``E001``-``E008``
error-code parity across the Python, Rust/PyO3, and Zig backends.
These tests are pure-Python: they never invoke a native toolchain and never
leave artifacts in the repository.
"""

from __future__ import annotations

import pytest

from tools.abi_contract.contract import (
    ALL_ERROR_CODES,
    ContractViolation,
    default_contract,
    verify_enum_layout,
    verify_error_code,
    verify_slice_layout,
    verify_struct_layout,
    zig_enum_layout,
    zig_error_info,
    zig_slice_layout,
    zig_struct_layout,
)
from tools.abi_contract.verifier import (
    capture_python_snapshot,
    capture_zig_snapshot,
    verify_backends,
    verify_error_code_parity,
)

# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def test_default_contract_has_all_error_codes():
    contract = default_contract()
    assert set(contract.error_codes) == set(ALL_ERROR_CODES)
    for code in ALL_ERROR_CODES:
        info = contract.error_codes[code]
        assert info.code == code
        assert isinstance(info.errno, int)
        assert isinstance(info.name, str)
        assert isinstance(info.recoverable, bool)
        assert info.description


def test_default_contract_has_structs_enums_slices():
    contract = default_contract()
    assert "Point" in contract.structs
    assert "Pair" in contract.structs
    assert "TaggedValue" in contract.structs
    assert "WidePoint" in contract.structs
    assert "PillarError" in contract.enums
    assert "Option" in contract.enums
    assert "Result" in contract.enums
    assert "i64_slice" in contract.slices
    assert "u8_slice" in contract.slices
    assert "bool_slice" in contract.slices


def test_contract_fingerprint_is_stable():
    first = default_contract().fingerprint()
    second = default_contract().fingerprint()
    assert first == second
    assert len(first) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Zig layout matches the contract (the Zig model is the source of truth).
# ---------------------------------------------------------------------------


def test_zig_struct_layout_matches_contract():
    contract = default_contract()
    for name, fields in (
        ("Point", (("x", "i64"), ("y", "i64"))),
        ("Pair", (("first", "i64"), ("second", "i64"))),
        ("TaggedValue", (("tag", "u8"), ("value", "i64"))),
        ("WidePoint", (("x", "i128"), ("y", "i128"))),
    ):
        actual = zig_struct_layout(name, tuple(fields))
        # Should not raise: the Zig model is the contract source.
        verify_struct_layout(contract, backend="zig", name=name, actual=actual)


def test_zig_enum_layout_matches_contract():
    contract = default_contract()
    for name, variants in (
        (
            "PillarError",
            (
                ("E001", 1),
                ("E002", 2),
                ("E003", 3),
                ("E004", 4),
                ("E005", 5),
                ("E006", 6),
                ("E007", 7),
                ("E008", 8),
            ),
        ),
        ("Option", (("None", 0), ("Some", 1))),
        ("Result", (("Ok", 0), ("Err", 1))),
    ):
        actual = zig_enum_layout(name, tuple(variants))
        verify_enum_layout(contract, backend="zig", name=name, actual=actual)


def test_zig_slice_layout_matches_contract():
    contract = default_contract()
    for element in ("i64", "u8", "bool"):
        actual = zig_slice_layout(element)
        verify_slice_layout(
            contract, backend="zig", element_type=element, actual=actual
        )


# ---------------------------------------------------------------------------
# Error code parity (E001-E008).
# ---------------------------------------------------------------------------


def test_error_code_parity_is_clean():
    violations = verify_error_code_parity()
    assert violations == [], [v.message for v in violations]


def test_all_error_codes_present_in_zig_abi():
    import zig.abi as zig_abi

    errors = getattr(zig_abi, "_ERRORS", {})
    for code in ALL_ERROR_CODES:
        assert code in errors, f"zig/abi.py _ERRORS missing {code}"
        errno, name, recoverable = errors[code]
        assert isinstance(errno, int)
        assert isinstance(name, str)
        assert isinstance(recoverable, bool)


def test_all_error_codes_present_in_runtime_diagnostics():
    from runtime.diagnostics import ERROR_CODES

    for code in ALL_ERROR_CODES:
        assert code in ERROR_CODES, f"runtime/diagnostics.py ERROR_CODES missing {code}"


def test_zig_error_info_matches_contract():
    contract = default_contract()
    for code in ALL_ERROR_CODES:
        actual = zig_error_info(code)
        # Should not raise: the model is the contract source.
        verify_error_code(contract, backend="zig", code=code, actual=actual)


def test_pillar_error_members_are_exactly_e001_to_e008():
    from zig.backend import _PILLAR_ERRORS

    assert tuple(_PILLAR_ERRORS) == ALL_ERROR_CODES


# ---------------------------------------------------------------------------
# Drift detection: deliberate violations must raise ContractViolation.
# ---------------------------------------------------------------------------


def test_struct_layout_drift_detected():
    contract = default_contract()
    expected = contract.structs["Point"]
    # Deliberately wrong offset for field "y".
    drifted = type(expected)(
        name=expected.name,
        fields=(
            type(expected.fields[0])(name="x", type="i64", offset=0, size=8, align=8),
            type(expected.fields[0])(name="y", type="i64", offset=999, size=8, align=8),
        ),
        size=expected.size,
        align=expected.align,
    )
    with pytest.raises(ContractViolation) as exc_info:
        verify_struct_layout(contract, backend="zig", name="Point", actual=drifted)
    assert exc_info.value.check == "struct.field.y.offset"
    assert exc_info.value.backend == "zig"


def test_struct_layout_type_drift_detected():
    contract = default_contract()
    expected = contract.structs["Point"]
    drifted = type(expected)(
        name=expected.name,
        fields=(
            type(expected.fields[0])(name="x", type="i64", offset=0, size=8, align=8),
            type(expected.fields[0])(
                name="y", type="f64", offset=8, size=8, align=8
            ),  # was i64
        ),
        size=expected.size,
        align=expected.align,
    )
    with pytest.raises(ContractViolation) as exc_info:
        verify_struct_layout(contract, backend="zig", name="Point", actual=drifted)
    assert exc_info.value.check == "struct.field.y.type"


def test_enum_layout_drift_detected():
    contract = default_contract()
    expected = contract.enums["PillarError"]
    drifted = type(expected)(
        name=expected.name,
        tag_type=expected.tag_type,
        variants=(
            ("E001", 1),
            ("E002", 2),
            ("E003", 3),
            ("E004", 4),
            ("E005", 5),
            ("E006", 6),
            ("E007", 7),
            ("E999", 8),
        ),  # renamed
        size=expected.size,
        align=expected.align,
    )
    with pytest.raises(ContractViolation) as exc_info:
        verify_enum_layout(contract, backend="zig", name="PillarError", actual=drifted)
    assert exc_info.value.check == "enum.variant_names"
    assert exc_info.value.backend == "zig"


def test_slice_layout_drift_detected():
    contract = default_contract()
    expected = contract.slices["i64_slice"]
    drifted = type(expected)(
        element_type=expected.element_type,
        ptr_type="[*]const u8",  # was [*]const i64
        len_type=expected.len_type,
        size=expected.size,
        align=expected.align,
    )
    with pytest.raises(ContractViolation) as exc_info:
        verify_slice_layout(contract, backend="zig", element_type="i64", actual=drifted)
    assert exc_info.value.check == "slice.i64.ptr_type"


# ---------------------------------------------------------------------------
# Full cross-backend verification.
# ---------------------------------------------------------------------------


def test_verify_backends_is_clean():
    result = verify_backends()
    assert result["clean"] is True
    assert result["contract_fingerprint"]
    for backend_name in ("zig", "python", "rust"):
        assert result["backends"][backend_name] == []
    assert result["error_code_parity"] == []


def test_capture_zig_snapshot_shape():
    snapshot = capture_zig_snapshot()
    assert set(snapshot) == {
        "structs",
        "enums",
        "slices",
        "error_codes",
        "pointer_width_bytes",
        "target_word",
    }
    assert snapshot["pointer_width_bytes"] == 8
    assert snapshot["target_word"] == 8


def test_capture_python_snapshot_has_type_map():
    snapshot = capture_python_snapshot()
    assert "python_type_map" in snapshot
    assert isinstance(snapshot["python_type_map"], dict)
    # Python is dynamic: struct/enum/slice sizes are 0 (exempt from C-layout checks).
    for layout in snapshot["structs"].values():
        assert layout.size == 0
    for layout in snapshot["enums"].values():
        assert layout.size == 0
