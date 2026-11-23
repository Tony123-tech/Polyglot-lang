"""Deterministic build provenance & cache integrity package.

Public API:
    :class:`ProvenanceRecorder` - record one build's provenance in an isolated scope
    :func:`record_build_provenance` - convenience recording helper
    :class:`BuildProvenance` - one recorded provenance record
    :func:`verify_deterministic_build` - verify repeated builds produce identical outputs
    :func:`verify_cache_isolation` - verify scopes do not leak cache state
    :class:`BuildDeterminismResult` / :class:`BuildIntegrityResult` - verification outcomes
"""

from .recorder import (
    BuildFn,
    BuildProvenance,
    ProvenanceRecorder,
    default_scope_dimensions,
    record_build_provenance,
)
from .verifier import (
    BuildDeterminismResult,
    BuildIntegrityResult,
    verify_cache_isolation,
    verify_deterministic_build,
    verify_provenance_records,
)

__all__ = [
    "BuildDeterminismResult",
    "BuildFn",
    "BuildIntegrityResult",
    "BuildProvenance",
    "ProvenanceRecorder",
    "default_scope_dimensions",
    "record_build_provenance",
    "verify_cache_isolation",
    "verify_deterministic_build",
    "verify_provenance_records",
]
