"""Cross-backend ABI/FFI contract verifier package.

Verifies struct field alignment, enum/ADT tag representation, optional/slice
layout assumptions, and ``PillarError`` (``E001``-``E008``) code/payload
consistency across the Python, Rust/PyO3, and Zig backends.

Public API:
    :func:`default_contract` - build the canonical contract
    :func:`verify_all_backends` - run every backend snapshot against the contract
    :class:`ContractViolation` - raised on any layout or error-semantics drift
    :func:`format_report` - render a verification report as structured text
"""

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
    format_report,
    verify_all_backends,
    verify_contract_against_backend,
    zig_enum_layout,
    zig_error_info,
    zig_slice_layout,
    zig_struct_layout,
)

__all__ = [
    "ALL_ERROR_CODES",
    "Contract",
    "ContractViolation",
    "EnumLayout",
    "ErrorInfo",
    "FieldLayout",
    "SliceLayout",
    "StructLayout",
    "default_contract",
    "format_report",
    "verify_all_backends",
    "verify_contract_against_backend",
    "zig_enum_layout",
    "zig_error_info",
    "zig_slice_layout",
    "zig_struct_layout",
]
