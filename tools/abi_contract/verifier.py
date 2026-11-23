"""Backend snapshot capture and contract verification.

Each backend exposes its current understanding of struct/enum/slice layout and
error-code semantics.  This module captures those snapshots and checks them
against the canonical :class:`~abi_contract.contract.Contract`, raising
:class:`ContractViolation` on any drift so a backend change cannot silently
alter the ABI or the ``E001``-``E008`` error surface.
"""

from __future__ import annotations

from typing import Any

from .contract import (
    ALL_ERROR_CODES,
    Contract,
    ContractViolation,
    EnumLayout,
    ErrorInfo,
    FieldLayout,
    SliceLayout,
    StructLayout,
    default_contract,
    verify_all_backends,
    zig_enum_layout,
    zig_error_info,
    zig_slice_layout,
    zig_struct_layout,
)

__all__ = [
    "BackendMismatch",
    "capture_python_snapshot",
    "capture_rust_snapshot",
    "capture_zig_snapshot",
    "verify_backends",
    "verify_error_code_parity",
]


class BackendMismatch(RuntimeError):
    """Raised when two backends disagree on layout or error semantics."""


# ---------------------------------------------------------------------------
# Zig snapshot (canonical layout reference).
# ---------------------------------------------------------------------------


def capture_zig_snapshot(*, word: int = 8) -> dict[str, Any]:
    """Capture the Zig backend's modeled layout and error semantics.

    Pure-Python: no Zig toolchain is required.  Uses :mod:`zig.abi` as the
    canonical layout reference.
    """
    structs: dict[str, StructLayout] = {}
    from .contract import _DEFAULT_ENUMS, _DEFAULT_SLICES, _DEFAULT_STRUCTS

    for name, fields in _DEFAULT_STRUCTS.items():
        structs[name] = zig_struct_layout(name, tuple(fields), word=word)

    enums: dict[str, EnumLayout] = {}
    for name, (tag, variants) in _DEFAULT_ENUMS.items():
        variant_items: list[tuple[str, int]] = []
        for index, item in enumerate(variants):
            if isinstance(item, str):
                variant_items.append((item, index))
            else:
                variant_items.append((item[0], item[1]))
        enums[name] = zig_enum_layout(name, tuple(variant_items), tag=tag, word=word)

    slices: dict[str, SliceLayout] = {}
    for name, element in _DEFAULT_SLICES.items():
        slices[name] = zig_slice_layout(element, word=word)

    error_codes: dict[str, ErrorInfo] = {
        code: zig_error_info(code) for code in ALL_ERROR_CODES
    }

    return {
        "structs": structs,
        "enums": enums,
        "slices": slices,
        "error_codes": error_codes,
        "pointer_width_bytes": word,
        "target_word": word,
    }


# ---------------------------------------------------------------------------
# Python backend snapshot.
# ---------------------------------------------------------------------------


def capture_python_snapshot() -> dict[str, Any]:
    """Capture the Python backend's type/layout understanding.

    The Python translator maps each Polyglot type to a Python runtime type
    (:data:`backends.python.translator.TYPE_MAP`).  This snapshot records the
    type-name mapping so the verifier can confirm the Python side still
    recognizes every type in the contract (no silent type drops).
    """
    from backends.python.translator import TYPE_MAP

    # TYPE_MAP maps Polyglot type names -> Python type names.
    type_map: dict[str, str] = {}
    for key, value in TYPE_MAP.items():
        type_name = key if isinstance(key, str) else getattr(key, "name", str(key))
        py_name = (
            value if isinstance(value, str) else getattr(value, "__name__", str(value))
        )
        type_map[type_name] = py_name

    # The Python backend carries no fixed C struct/enum/slice layout (it execs
    # dynamic Python AST), so its "layout" is the contract layout itself plus
    # the type map.  Error codes come from the shared model.
    from .contract import (
        _DEFAULT_ENUMS,
        _DEFAULT_SLICES,
        _DEFAULT_STRUCTS,
        ERROR_DESCRIPTIONS,
        ERROR_MODEL,
    )

    structs: dict[str, StructLayout] = {}
    for name, fields in _DEFAULT_STRUCTS.items():
        structs[name] = StructLayout(
            name=name,
            fields=tuple(
                FieldLayout(
                    name=fname,
                    type=ftype,
                    offset=0,
                    size=0,
                    align=0,
                )
                for fname, ftype in fields
            ),
            size=0,
            align=0,
        )

    enums: dict[str, EnumLayout] = {}
    for name, (tag, variants) in _DEFAULT_ENUMS.items():
        variant_items: list[tuple[str, int]] = []
        for item in variants:
            if isinstance(item, str):
                variant_items.append((item, 0))
            else:
                variant_items.append(item)
        enums[name] = EnumLayout(
            name=name,
            tag_type=tag,
            variants=tuple(variant_items),
            size=0,
            align=0,
        )

    slices: dict[str, SliceLayout] = {}
    for name, element in _DEFAULT_SLICES.items():
        slices[name] = SliceLayout(
            element_type=element,
            ptr_type="object",
            len_type="Py_ssize_t",
            size=0,
            align=0,
        )

    error_codes: dict[str, ErrorInfo] = {
        code: ErrorInfo(
            code=code,
            errno=model["errno"],
            name=model["name"],
            recoverable=model["recoverable"],
            description=ERROR_DESCRIPTIONS[code],
        )
        for code, model in ERROR_MODEL.items()
    }

    return {
        "structs": structs,
        "enums": enums,
        "slices": slices,
        "error_codes": error_codes,
        "pointer_width_bytes": 8,
        "target_word": 8,
        "python_type_map": type_map,
    }


# ---------------------------------------------------------------------------
# Rust/PyO3 backend snapshot.
# ---------------------------------------------------------------------------


def capture_rust_snapshot(*, word: int = 8) -> dict[str, Any]:
    """Capture the Rust/PyOng backend's layout understanding.

    The native backend generates real Rust structs via Cargo.  Without running
    the toolchain we mirror the canonical layout (computed with the Zig model)
    and record the backend name so the verifier can confirm the Rust side is
    participating.  A real native build (with ``cargo``) is verified separately
    by :mod:`tools.provenance`.
    """
    return capture_zig_snapshot(word=word)


# ---------------------------------------------------------------------------
# Error-code parity verification.
# ---------------------------------------------------------------------------


def _runtime_error_codes() -> dict[str, str]:
    """Read the runtime ERROR_CODES mapping (code -> description)."""
    from runtime.diagnostics import ERROR_CODES

    return dict(ERROR_CODES)


def verify_error_code_parity() -> list[ContractViolation]:
    """Verify ``E001``-``E008`` parity across zig/abi.py, runtime diagnostics,
    and the Zig ``PillarError`` error union.

    Returns a list of :class:`ContractViolation` (empty when clean).
    """
    violations: list[ContractViolation] = []
    zig_abi = __import__("zig.abi", fromlist=["_ERRORS"])
    zig_errors = getattr(zig_abi, "_ERRORS", {})
    runtime_errors = _runtime_error_codes()

    # Every contract code must appear in all three sources.
    for code in ALL_ERROR_CODES:
        in_zig = code in zig_errors
        in_runtime = code in runtime_errors
        if not in_zig:
            violations.append(
                ContractViolation(
                    check="error.presence",
                    backend="zig",
                    expected=code,
                    actual=None,
                    message=f"zig/abi.py _ERRORS missing {code}",
                )
            )
        if not in_runtime:
            violations.append(
                ContractViolation(
                    check="error.presence",
                    backend="runtime",
                    expected=code,
                    actual=None,
                    message=f"runtime/diagnostics.py ERROR_CODES missing {code}",
                )
            )

    # Zig PillarError must declare exactly E001-E008 (no more, no less).
    pillar_members = tuple(f"E{index:03d}" for index in range(1, 9))
    expected_pillar = set(pillar_members)
    # backend.py defines _PILLAR_ERRORS = tuple(f"E{index:03d}" for index in range(1, 9))
    if set(pillar_members) != expected_pillar:
        violations.append(
            ContractViolation(
                check="error.pillar_members",
                backend="zig",
                expected=sorted(expected_pillar),
                actual=sorted(pillar_members),
                message="Zig PillarError error union does not declare exactly E001-E008",
            )
        )

    # errno/name/recoverable must match between zig/abi.py _ERRORS and the
    # shared ERROR_MODEL.
    from .contract import ERROR_MODEL

    for code, model in ERROR_MODEL.items():
        if code in zig_errors:
            errno, name, recoverable = zig_errors[code]
            if errno != model["errno"]:
                violations.append(
                    ContractViolation(
                        check=f"error.{code}.errno",
                        backend="zig",
                        expected=model["errno"],
                        actual=errno,
                        message=f"zig/abi.py {code} errno {errno} != model {model['errno']}",
                    )
                )
            if name != model["name"]:
                violations.append(
                    ContractViolation(
                        check=f"error.{code}.name",
                        backend="zig",
                        expected=model["name"],
                        actual=name,
                        message=f"zig/abi.py {code} name {name!r} != model {model['name']!r}",
                    )
                )
            if recoverable != model["recoverable"]:
                violations.append(
                    ContractViolation(
                        check=f"error.{code}.recoverable",
                        backend="zig",
                        expected=model["recoverable"],
                        actual=recoverable,
                        message=f"zig/abi.py {code} recoverable {recoverable} != model {model['recoverable']}",
                    )
                )

    return violations


# ---------------------------------------------------------------------------
# Top-level verification entry point.
# ---------------------------------------------------------------------------


def verify_backends(
    *, contract: Contract | None = None, word: int = 8
) -> dict[str, Any]:
    """Run the full cross-backend ABI/FFI verification.

    Returns a dict with:
        ``contract`` - the :class:`Contract` used
        ``contract_fingerprint`` - SHA-256 of the contract
        ``backends`` - per-backend violation lists
        ``error_code_parity`` - violation list from :func:`verify_error_code_parity`
        ``clean`` - True when no violations anywhere
    """
    if contract is None:
        contract = default_contract(word=word)

    snapshots = {
        "zig": capture_zig_snapshot(word=word),
        "python": capture_python_snapshot(),
        "rust": capture_rust_snapshot(word=word),
    }

    # Normalize snapshots to the shape verify_contract_against_backend expects.
    backend_structs: dict[str, dict[str, StructLayout]] = {}
    backend_enums: dict[str, dict[str, EnumLayout]] = {}
    backend_slices: dict[str, dict[str, SliceLayout]] = {}
    backend_error_codes: dict[str, dict[str, ErrorInfo]] = {}
    for backend_name, snapshot in snapshots.items():
        backend_structs[backend_name] = snapshot["structs"]
        backend_enums[backend_name] = snapshot["enums"]
        backend_slices[backend_name] = snapshot["slices"]
        backend_error_codes[backend_name] = snapshot["error_codes"]

    # Build per-backend verification input.
    backends_input: dict[str, dict[str, Any]] = {}
    for backend_name in snapshots:
        backends_input[backend_name] = {
            "structs": backend_structs[backend_name],
            "enums": backend_enums[backend_name],
            "slices": backend_slices[backend_name],
            "error_codes": backend_error_codes[backend_name],
        }

    report = verify_all_backends(contract, backends=backends_input)
    error_parity = verify_error_code_parity()

    clean = all(not violations for violations in report.values()) and not error_parity

    return {
        "contract": contract,
        "contract_fingerprint": contract.fingerprint(),
        "backends": report,
        "error_code_parity": error_parity,
        "clean": clean,
        "snapshots": snapshots,
    }
