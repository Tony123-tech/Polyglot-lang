"""Diagnostics, error reporting, and telemetry for the framework pipeline.

Frame 8 reuses the shared E001-E008 error namespace from
:mod:`runtime.diagnostics` and the span helpers from :mod:`api.visualizer`.
The public callables live in :mod:`diagnostics.framework`; they are re-exported
here so callers can use either ``diagnostics.framework.diag_*`` or
``diagnostics.diag_*``.
"""

from __future__ import annotations

from .framework import (
    AggregatedDiagnostics,
    DiagnosticContext,
    DiagnosticRecord,
    diag_aggregate_without_truncating_root_cause,
    diag_create_diagnostic_context,
    diag_create_pillar_error,
    diag_export_benchmark_summary,
    diag_format_json_payload,
    diag_format_terminal_snippet,
    diag_map_error_code_e001_to_e008,
    diag_record_cache_hit_metrics,
    diag_record_compile_phase_timing,
    diag_redact_sensitive_paths,
)

__all__ = [
    "AggregatedDiagnostics",
    "DiagnosticContext",
    "DiagnosticRecord",
    "diag_aggregate_without_truncating_root_cause",
    "diag_create_diagnostic_context",
    "diag_create_pillar_error",
    "diag_export_benchmark_summary",
    "diag_format_json_payload",
    "diag_format_terminal_snippet",
    "diag_map_error_code_e001_to_e008",
    "diag_record_cache_hit_metrics",
    "diag_record_compile_phase_timing",
    "diag_redact_sensitive_paths",
]
