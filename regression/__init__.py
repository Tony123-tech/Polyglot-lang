"""Installable home for the callable FRAME 10 regression harnesses.

The implementations live in :mod:`regression.harness` so that the 100-function
registry resolves even inside a wheel, where ``tests/`` is intentionally not
shipped.  ``tests/regression_harness.py`` remains a compatibility re-export for
the in-repository pytest suites.
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
