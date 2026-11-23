"""Bootstrap compiler core: subset front end, canonical IR, fixed point."""

from __future__ import annotations

from .diff import DiffReport, bootstrap_diff_stage_ir_hashes
from .fixed_point import FixedPointReport, bootstrap_verify_stage2_fixed_point
from .ir import parse_ir
from .stage1 import Artifact, bootstrap_stage1_compile_subset
from .subset import (
    Program,
    check_types,
    digest,
    emit_ir,
    emit_zig,
    parse_subset,
)

__all__ = [
    "Artifact",
    "DiffReport",
    "FixedPointReport",
    "Program",
    "bootstrap_diff_stage_ir_hashes",
    "bootstrap_stage1_compile_subset",
    "bootstrap_verify_stage2_fixed_point",
    "check_types",
    "digest",
    "emit_ir",
    "emit_zig",
    "parse_ir",
    "parse_subset",
]
