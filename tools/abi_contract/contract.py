"""Shared cross-backend ABI/FFI contract definitions.

The contract is the single source of truth for struct field alignment,
enum/ADT tag representation, optional/slice layout assumptions, and
``PillarError`` (``E001``-``E008``) code/payload consistency across the
Python, Rust/PyO3, and Zig backends.

The Zig model in :mod:`zig.abi` is the canonical layout reference; the Python
translator type map and the Zig ``PillarError`` error union must agree with it.
Any drift is reported as a :class:`ContractViolation` with structured
diagnostics so a backend change cannot silently alter the ABI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final

# The Zig ABI model lives in the ``zig`` package.  Import lazily so this module
# can be imported without the Zig toolchain present (the model itself is
# pure-Python and toolchain-free).
_zig_abi: Any = None


def _lazy_zig_abi() -> Any:
    global _zig_abi
    if _zig_abi is None:
        from zig import abi as _zig_abi_mod

        _zig_abi = _zig_abi_mod
    return _zig_abi


# ---------------------------------------------------------------------------
# Error codes shared across all backends (E001-E008).
# ---------------------------------------------------------------------------

ERROR_DESCRIPTIONS: Final[dict[str, str]] = {
    "E001": "type, layout, or ABI boundary",
    "E002": "semantic proof or termination obligation",
    "E003": "translation, AST, or API misuse",
    "E004": "compiler or native toolchain failure",
    "E005": "dependency cycle",
    "E006": "missing module, snapshot, or artifact",
    "E007": "unsupported native pattern or platform feature",
    "E008": "malformed serialized state or literal encoding",
}

# Canonical errno/name/recoverable model for each code, mirrored in zig/abi.py
# _ERRORS and runtime/diagnostics.py ERROR_CODES.
ERROR_MODEL: Final[dict[str, dict[str, Any]]] = {
    "E001": {"errno": 22, "name": "invalid_argument", "recoverable": True},
    "E002": {"errno": 5, "name": "unmet_obligation", "recoverable": False},
    "E003": {"errno": 22, "name": "api_misuse", "recoverable": True},
    "E004": {"errno": 74, "name": "toolchain_failure", "recoverable": True},
    "E005": {"errno": 40, "name": "dependency_cycle", "recoverable": False},
    "E006": {"errno": 2, "name": "missing_artifact", "recoverable": True},
    "E007": {"errno": 95, "name": "unsupported_feature", "recoverable": False},
    "E008": {"errno": 84, "name": "malformed_state", "recoverable": False},
}

ALL_ERROR_CODES: Final[tuple[str, ...]] = tuple(
    f"E{index:03d}" for index in range(1, 9)
)

PILLAR_ERROR_MEMBERS: Final[tuple[str, ...]] = ALL_ERROR_CODES


# ---------------------------------------------------------------------------
# Contract data model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldLayout:
    name: str
    type: str
    offset: int
    size: int
    align: int


@dataclass(frozen=True)
class StructLayout:
    name: str
    fields: tuple[FieldLayout, ...]
    size: int
    align: int
    packed: bool = False


@dataclass(frozen=True)
class EnumLayout:
    name: str
    tag_type: str
    variants: tuple[tuple[str, int], ...]
    size: int
    align: int


@dataclass(frozen=True)
class SliceLayout:
    element_type: str
    ptr_type: str
    len_type: str
    size: int
    align: int


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    errno: int
    name: str
    recoverable: bool
    description: str


@dataclass(frozen=True)
class Contract:
    """The canonical cross-backend ABI/FFI contract."""

    structs: dict[str, StructLayout] = field(default_factory=dict)
    enums: dict[str, EnumLayout] = field(default_factory=dict)
    slices: dict[str, SliceLayout] = field(default_factory=dict)
    error_codes: dict[str, ErrorInfo] = field(default_factory=dict)
    pointer_width_bytes: int = 8
    target_word: int = 8

    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the whole contract."""
        payload = json.dumps(
            {
                "structs": {
                    name: {
                        "fields": [
                            {
                                "name": f.name,
                                "type": f.type,
                                "offset": f.offset,
                                "size": f.size,
                                "align": f.align,
                            }
                            for f in layout.fields
                        ],
                        "size": layout.size,
                        "align": layout.align,
                        "packed": layout.packed,
                    }
                    for name, layout in sorted(self.structs.items())
                },
                "enums": {
                    name: {
                        "tag_type": layout.tag_type,
                        "variants": [list(v) for v in layout.variants],
                        "size": layout.size,
                        "align": layout.align,
                    }
                    for name, layout in sorted(self.enums.items())
                },
                "slices": {
                    name: {
                        "element_type": layout.element_type,
                        "ptr_type": layout.ptr_type,
                        "len_type": layout.len_type,
                        "size": layout.size,
                        "align": layout.align,
                    }
                    for name, layout in sorted(self.slices.items())
                },
                "error_codes": {
                    code: {
                        "errno": info.errno,
                        "name": info.name,
                        "recoverable": info.recoverable,
                        "description": info.description,
                    }
                    for code, info in sorted(self.error_codes.items())
                },
                "pointer_width_bytes": self.pointer_width_bytes,
                "target_word": self.target_word,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Default contract (built from the Zig model + shared error model).
# ---------------------------------------------------------------------------


def default_contract(*, word: int = 8) -> Contract:
    """Build the default contract from the shared model.

    Layouts are computed with the Zig ABI model (:mod:`zig.abi`) so the
    contract always reflects the canonical lowering.  Error codes come from
    the shared :data:`ERROR_MODEL`.
    """
    zig = _lazy_zig_abi()

    structs: dict[str, StructLayout] = {}
    for name, fields in _DEFAULT_STRUCTS.items():
        computed = zig.zabi_field_offsets(fields, word=word)
        structs[name] = StructLayout(
            name=name,
            fields=tuple(
                FieldLayout(
                    name=row["name"],
                    type=row["type"],
                    offset=row["offset"],
                    size=row["size"],
                    align=row["align"],
                )
                for row in computed["fields"]
            ),
            size=computed["size"],
            align=computed["align"],
        )

    enums: dict[str, EnumLayout] = {}
    for name, (tag, variants) in _DEFAULT_ENUMS.items():
        variant_items: list[tuple[str, int]] = []
        for index, item in enumerate(variants):
            if isinstance(item, str):
                variant_items.append((item, index))
            else:
                variant_items.append((item[0], item[1]))
        computed = zig.zabi_enum_repr(
            [
                item if isinstance(item, str) else (item[0], item[1])
                for item in variants
            ],
            tag=tag,
        )
        enums[name] = EnumLayout(
            name=name,
            tag_type=computed["tag"],
            variants=tuple(variant_items),
            size=computed["size"],
            align=computed["align"],
        )

    slices: dict[str, SliceLayout] = {}
    for name, element in _DEFAULT_SLICES.items():
        computed = zig.zabi_slice_repr(element, word=word)
        slices[name] = SliceLayout(
            element_type=element,
            ptr_type=computed["ptr_type"],
            len_type=computed["len_type"],
            size=computed["size"],
            align=computed["align"],
        )

    error_codes: dict[str, ErrorInfo] = {}
    for code in ALL_ERROR_CODES:
        model = ERROR_MODEL[code]
        error_codes[code] = ErrorInfo(
            code=code,
            errno=model["errno"],
            name=model["name"],
            recoverable=model["recoverable"],
            description=ERROR_DESCRIPTIONS[code],
        )

    return Contract(
        structs=structs,
        enums=enums,
        slices=slices,
        error_codes=error_codes,
        pointer_width_bytes=word,
        target_word=word,
    )


# Structs every backend must lay out identically.
_DEFAULT_STRUCTS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "Point": (("x", "i64"), ("y", "i64")),
    "Pair": (("first", "i64"), ("second", "i64")),
    "TaggedValue": (("tag", "u8"), ("value", "i64")),
    "WidePoint": (("x", "i128"), ("y", "i128")),
}

# Tagged enums every backend must represent identically.
_DEFAULT_ENUMS: Final[dict[str, tuple[str, tuple[str | tuple[str, int], ...]]]] = {
    "PillarError": (
        "u8",
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
    "Option": ("u8", ("None", "Some")),
    "Result": ("u8", ("Ok", "Err")),
}

# Slices every backend must represent identically (ptr/len pair).
_DEFAULT_SLICES: Final[dict[str, str]] = {
    "i64_slice": "i64",
    "u8_slice": "u8",
    "bool_slice": "bool",
}


# ---------------------------------------------------------------------------
# Violations.
# ---------------------------------------------------------------------------


class ContractViolation(RuntimeError):
    """Raised when a backend's layout or error semantics drift from the contract."""

    def __init__(
        self,
        *,
        check: str,
        backend: str,
        expected: Any,
        actual: Any,
        message: str,
    ) -> None:
        self.check = check
        self.backend = backend
        self.expected = expected
        self.actual = actual
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "backend": self.backend,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Backend layout extraction.
# ---------------------------------------------------------------------------


def zig_struct_layout(
    name: str, fields: tuple[tuple[str, str], ...], *, word: int = 8
) -> StructLayout:
    """Compute a struct layout from the Zig ABI model."""
    zig = _lazy_zig_abi()
    computed = zig.zabi_field_offsets(list(fields), word=word)
    return StructLayout(
        name=name,
        fields=tuple(
            FieldLayout(
                name=row["name"],
                type=row["type"],
                offset=row["offset"],
                size=row["size"],
                align=row["align"],
            )
            for row in computed["fields"]
        ),
        size=computed["size"],
        align=computed["align"],
    )


def zig_enum_layout(
    name: str, variants: tuple[tuple[str, int], ...], *, tag: str = "u8", word: int = 8
) -> EnumLayout:
    """Compute an enum layout from the Zig ABI model.

    ``variants`` are ``(name, explicit_value)`` pairs; the explicit values are
    preserved in the returned layout (they are not re-derived by position).
    """
    zig = _lazy_zig_abi()
    variant_items: list[str | tuple[str, int]] = []
    for variant_name, value in variants:
        variant_items.append((variant_name, value))
    computed = zig.zabi_enum_repr(variant_items, tag=tag)
    return EnumLayout(
        name=name,
        tag_type=computed["tag"],
        variants=variants,
        size=computed["size"],
        align=computed["align"],
    )


def zig_slice_layout(element_type: str, *, word: int = 8) -> SliceLayout:
    """Compute a slice layout from the Zig ABI model."""
    zig = _lazy_zig_abi()
    computed = zig.zabi_slice_repr(element_type, word=word)
    return SliceLayout(
        element_type=element_type,
        ptr_type=computed["ptr_type"],
        len_type=computed["len_type"],
        size=computed["size"],
        align=computed["align"],
    )


def zig_error_info(code: str) -> ErrorInfo:
    """Look up the modeled error info for one E001-E008 code."""
    zig = _lazy_zig_abi()
    mapped = zig.zabi_error_map(code)
    return ErrorInfo(
        code=mapped["code"],
        errno=mapped["errno"],
        name=mapped["name"],
        recoverable=mapped["recoverable"],
        description=ERROR_DESCRIPTIONS[code],
    )


# ---------------------------------------------------------------------------
# Verification.
# ---------------------------------------------------------------------------


def _require_equal(
    *, check: str, backend: str, expected: Any, actual: Any, message: str
) -> None:
    if expected != actual:
        raise ContractViolation(
            check=check,
            backend=backend,
            expected=expected,
            actual=actual,
            message=message,
        )


def verify_struct_layout(
    contract: Contract, *, backend: str, name: str, actual: StructLayout
) -> None:
    """Verify one struct's layout matches the contract.

    Backends with a dynamic object layout (size == 0, e.g. the Python
    transpiler) are exempt from offset/size/align checks; their field names
    and types are still validated against the contract.
    """
    expected = contract.structs.get(name)
    if expected is None:
        return  # not part of the contract
    expected_fields = {f.name: f for f in expected.fields}
    actual_fields = {f.name: f for f in actual.fields}
    _require_equal(
        check="struct.field_names",
        backend=backend,
        expected=tuple(sorted(expected_fields)),
        actual=tuple(sorted(actual_fields)),
        message=f"{backend}: struct {name!r} fields {tuple(sorted(actual_fields))} != contract {tuple(sorted(expected_fields))}",
    )
    for fname, exp_field in expected_fields.items():
        act_field = actual_fields.get(fname)
        if act_field is None:
            continue
        _require_equal(
            check=f"struct.field.{fname}.type",
            backend=backend,
            expected=exp_field.type,
            actual=act_field.type,
            message=f"{backend}: struct {name!r} field {fname!r}.type {act_field.type} != contract {exp_field.type}",
        )
    if actual.size != 0:
        _require_equal(
            check="struct.size",
            backend=backend,
            expected=expected.size,
            actual=actual.size,
            message=f"{backend}: struct {name!r} size {actual.size} != contract {expected.size}",
        )
        _require_equal(
            check="struct.align",
            backend=backend,
            expected=expected.align,
            actual=actual.align,
            message=f"{backend}: struct {name!r} align {actual.align} != contract {expected.align}",
        )
        for fname, exp_field in expected_fields.items():
            act_field = actual_fields[fname]
            for attr in ("offset", "size", "align"):
                exp_val = getattr(exp_field, attr)
                act_val = getattr(act_field, attr)
                _require_equal(
                    check=f"struct.field.{fname}.{attr}",
                    backend=backend,
                    expected=exp_val,
                    actual=act_val,
                    message=(
                        f"{backend}: struct {name!r} field {fname!r}.{attr} "
                        f"{act_val} != contract {exp_val}"
                    ),
                )


def verify_enum_layout(
    contract: Contract, *, backend: str, name: str, actual: EnumLayout
) -> None:
    """Verify one enum's layout matches the contract.

    Backends with a dynamic layout (size == 0) are exempt from size/align
    checks; the tag type and variant names are still validated.
    """
    expected = contract.enums.get(name)
    if expected is None:
        return
    _require_equal(
        check="enum.tag_type",
        backend=backend,
        expected=expected.tag_type,
        actual=actual.tag_type,
        message=f"{backend}: enum {name!r} tag {actual.tag_type} != contract {expected.tag_type}",
    )
    expected_names = tuple(name for name, _ in expected.variants)
    actual_names = tuple(name for name, _ in actual.variants)
    _require_equal(
        check="enum.variant_names",
        backend=backend,
        expected=expected_names,
        actual=actual_names,
        message=f"{backend}: enum {name!r} variant names {actual_names} != contract {expected_names}",
    )
    if actual.size != 0:
        _require_equal(
            check="enum.size",
            backend=backend,
            expected=expected.size,
            actual=actual.size,
            message=f"{backend}: enum {name!r} size {actual.size} != contract {expected.size}",
        )
        _require_equal(
            check="enum.align",
            backend=backend,
            expected=expected.align,
            actual=actual.align,
            message=f"{backend}: enum {name!r} align {actual.align} != contract {expected.align}",
        )
        _require_equal(
            check="enum.variants",
            backend=backend,
            expected=expected.variants,
            actual=actual.variants,
            message=f"{backend}: enum {name!r} variants {actual.variants} != contract {expected.variants}",
        )


def verify_slice_layout(
    contract: Contract, *, backend: str, element_type: str, actual: SliceLayout
) -> None:
    """Verify one slice's layout matches the contract.

    Backends with a dynamic layout (size == 0) are exempt from size/align
    checks; the element type is still validated.
    """
    expected = contract.slices.get(element_type + "_slice")
    if expected is None:
        expected = next(
            (
                layout
                for key, layout in contract.slices.items()
                if layout.element_type == element_type
            ),
            None,
        )
    if expected is None:
        return
    _require_equal(
        check=f"slice.{element_type}.element_type",
        backend=backend,
        expected=expected.element_type,
        actual=actual.element_type,
        message=f"{backend}: slice [{element_type}] element_type {actual.element_type} != contract {expected.element_type}",
    )
    if actual.size != 0:
        for attr in ("ptr_type", "len_type", "size", "align"):
            exp_val = getattr(expected, attr)
            act_val = getattr(actual, attr)
            _require_equal(
                check=f"slice.{element_type}.{attr}",
                backend=backend,
                expected=exp_val,
                actual=act_val,
                message=f"{backend}: slice [{element_type}] {attr} {act_val} != contract {exp_val}",
            )


def verify_error_code(
    contract: Contract, *, backend: str, code: str, actual: ErrorInfo
) -> None:
    """Verify one error code's payload matches the contract."""
    expected = contract.error_codes.get(code)
    if expected is None:
        return
    for attr in ("errno", "name", "recoverable", "description"):
        exp_val = getattr(expected, attr)
        act_val = getattr(actual, attr)
        _require_equal(
            check=f"error.{code}.{attr}",
            backend=backend,
            expected=exp_val,
            actual=act_val,
            message=f"{backend}: error {code} {attr} {act_val} != contract {exp_val}",
        )


def verify_contract_against_backend(
    contract: Contract,
    *,
    backend: str,
    structs: dict[str, StructLayout],
    enums: dict[str, EnumLayout],
    slices: dict[str, SliceLayout],
    error_codes: dict[str, ErrorInfo],
) -> list[ContractViolation]:
    """Run every contract check for one backend and return collected violations."""
    violations: list[ContractViolation] = []
    for name, actual in structs.items():
        try:
            verify_struct_layout(contract, backend=backend, name=name, actual=actual)
        except ContractViolation as violation:
            violations.append(violation)
    for name, actual in enums.items():
        try:
            verify_enum_layout(contract, backend=backend, name=name, actual=actual)
        except ContractViolation as violation:
            violations.append(violation)
    for element_type, actual in slices.items():
        try:
            verify_slice_layout(
                contract, backend=backend, element_type=element_type, actual=actual
            )
        except ContractViolation as violation:
            violations.append(violation)
    for code, actual in error_codes.items():
        try:
            verify_error_code(contract, backend=backend, code=code, actual=actual)
        except ContractViolation as violation:
            violations.append(violation)
    return violations


def verify_all_backends(
    contract: Contract,
    *,
    backends: dict[str, dict[str, Any]],
) -> dict[str, list[ContractViolation]]:
    """Verify every backend snapshot against the contract.

    ``backends`` maps backend name to a snapshot dict with keys ``structs``,
    ``enums``, ``slices``, ``error_codes``.
    """
    report: dict[str, list[ContractViolation]] = {}
    for backend_name, snapshot in backends.items():
        violations = verify_contract_against_backend(
            contract,
            backend=backend_name,
            structs=snapshot.get("structs", {}),
            enums=snapshot.get("enums", {}),
            slices=snapshot.get("slices", {}),
            error_codes=snapshot.get("error_codes", {}),
        )
        report[backend_name] = violations
    return report


def contract_is_clean(report: dict[str, list[ContractViolation]]) -> bool:
    """Return True when no backend reported any violation."""
    return all(not violations for violations in report.values())


def format_report(report: dict[str, list[ContractViolation]]) -> str:
    """Render a verification report as structured text."""
    lines: list[str] = ["ABI/FFI contract verification report", "=" * 60]
    if contract_is_clean(report):
        lines.append("RESULT: CLEAN - all backends match the contract")
    else:
        lines.append("RESULT: VIOLATIONS DETECTED")
        for backend_name in sorted(report):
            violations = report[backend_name]
            lines.append(f"  backend: {backend_name}")
            if not violations:
                lines.append("    (no violations)")
                continue
            for violation in violations:
                lines.append(f"    - [{violation.check}] {violation.message}")
    return "\n".join(lines)
