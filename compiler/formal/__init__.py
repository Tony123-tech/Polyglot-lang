"""Bounded formal methods frame (Frame H).

The frame exposes sound *bounded* analyses only: symbolic path execution with
explicit step/unroll bounds, bounded relational refinement checks, and
infeasible-path witnesses.  No unbounded equivalence or universal halting
claim is made; every analysis reports its bounds and `Unknown` states.
"""

from __future__ import annotations

from compiler.formal.framework import (
    BoundedExecutionReport,
    Divergence,
    Feasibility,
    FormalDiagnostic,
    InfeasiblePathReport,
    RelationalReport,
    SymbolicPath,
    WitnessExplanation,
    formal_check_relational_refinement,
    formal_require_bounded_proof,
    formal_sym_exec_bounded_paths,
    formal_synthesize_infeasible_path_witness,
)

__all__ = [
    "BoundedExecutionReport",
    "Divergence",
    "Feasibility",
    "FormalDiagnostic",
    "InfeasiblePathReport",
    "RelationalReport",
    "SymbolicPath",
    "WitnessExplanation",
    "formal_check_relational_refinement",
    "formal_require_bounded_proof",
    "formal_sym_exec_bounded_paths",
    "formal_synthesize_infeasible_path_witness",
]
