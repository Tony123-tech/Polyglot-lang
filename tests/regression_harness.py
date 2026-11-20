"""Compatibility re-export for the FRAME 10 regression harnesses.

The canonical implementations now live in :mod:`regression.harness` so they are
included in installed wheels; this module keeps the in-repository import path
used by ``tests/test_framework_f10.py`` working unchanged.
"""

from __future__ import annotations

from regression.harness import (
    F10_SOURCES,
    test_hotpatch_delegation_and_rollback,
    test_isolated_cache_no_contamination,
    test_recursive_generic_enums,
    test_recursive_generic_structs,
    test_scaffold_drift_detection,
    test_trait_bounds_and_impls,
    test_typed_pillar_error_parity,
)

__all__ = [
    "F10_SOURCES",
    "test_hotpatch_delegation_and_rollback",
    "test_isolated_cache_no_contamination",
    "test_recursive_generic_enums",
    "test_recursive_generic_structs",
    "test_scaffold_drift_detection",
    "test_trait_bounds_and_impls",
    "test_typed_pillar_error_parity",
]
