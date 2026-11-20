"""FRAME 2: type checking, trait bounds, and monomorphization helpers.

The implementations live in :mod:`compiler.types.framework` so the unified
registry can resolve them by name.  They are re-exported here for callers that
prefer importing from the package root.
"""

from __future__ import annotations

from compiler.types.framework import (
    monomorph_cache_placeholder_register,
    monomorph_emit_inflation_report,
    monomorph_propagate_vec_elem_type,
    monomorph_prune_dead_specializations,
    monomorph_resolve_recursive_enum,
    monomorph_resolve_recursive_struct,
    monomorph_unify_cache_keys,
    typecheck_detect_occurs_check_cycle,
    typecheck_enforce_trait_bounds,
    typecheck_infer_expressions,
)

__all__ = [
    "monomorph_cache_placeholder_register",
    "monomorph_emit_inflation_report",
    "monomorph_propagate_vec_elem_type",
    "monomorph_prune_dead_specializations",
    "monomorph_resolve_recursive_enum",
    "monomorph_resolve_recursive_struct",
    "monomorph_unify_cache_keys",
    "typecheck_detect_occurs_check_cycle",
    "typecheck_enforce_trait_bounds",
    "typecheck_infer_expressions",
]
