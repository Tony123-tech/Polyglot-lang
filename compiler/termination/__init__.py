"""Termination and resource-bound proof frame (Frame I).

Every result is a *proof obligation discharge* over an explicit, finite model:
decreasing variants, syntactic loop-bound certificates, and deterministic fuel
contracts.  No universal halting claim is made; unsupported recursion or loops
are reported as ``UnboundedOrUnknown`` with structured diagnostics.
"""

from __future__ import annotations

from compiler.termination.framework import (
    FuelContract,
    FuelDecision,
    LoopBoundCertificate,
    LoopBoundReport,
    TerminationDiagnostic,
    VariantCall,
    VariantReport,
    VariantSpec,
    term_check_structural_and_lexicographic_variants,
    term_enforce_resource_fuel_contracts,
    term_infer_loop_bound_certificates,
)

__all__ = [
    "FuelContract",
    "FuelDecision",
    "LoopBoundCertificate",
    "LoopBoundReport",
    "TerminationDiagnostic",
    "VariantCall",
    "VariantReport",
    "VariantSpec",
    "term_check_structural_and_lexicographic_variants",
    "term_enforce_resource_fuel_contracts",
    "term_infer_loop_bound_certificates",
]
