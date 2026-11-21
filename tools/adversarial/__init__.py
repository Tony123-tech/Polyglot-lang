"""Adversarial verification frame (Frame K).

Deterministic grammar/type-aware case generation, isolated mutation-score
gating, and a spec-oracle/full-verification report.  The frame reports exact
command results and bounded caveats; it never claims exhaustive fuzzing or a
complete proof from a passing mutation score.
"""

from __future__ import annotations

from tools.adversarial.framework import (
    AdversarialCase,
    MutantResult,
    MutantSpec,
    MutationReport,
    MutationScoreReport,
    OracleCase,
    SpecOracleReport,
    VerificationResult,
    adv_mutate_ast_and_type_constraints,
    adv_run_mutation_score_gate,
    adv_verify_spec_oracle_and_report,
)

__all__ = [
    "AdversarialCase",
    "MutantResult",
    "MutantSpec",
    "MutationReport",
    "MutationScoreReport",
    "OracleCase",
    "SpecOracleReport",
    "VerificationResult",
    "adv_mutate_ast_and_type_constraints",
    "adv_run_mutation_score_gate",
    "adv_verify_spec_oracle_and_report",
]
