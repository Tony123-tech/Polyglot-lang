"""Behavioral tests for Frame 8: diagnostics, error reporting, and telemetry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from backends.errors import (
    InvalidIntegerParseLiteralError,
    TranslateError,
    UnsupportedNativePatternError,
)
from diagnostics.framework import (
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
from runtime.diagnostics import PillarError


def test_context_is_reusable_and_resolves_code_aliases():
    context = diag_create_diagnostic_context(
        "let x = @;\nlet y = 2;",
        "parse",
        "py",
        {"syntax": "E003", "types": "E001"},
        path="src/main.pg",
    )
    assert context.backend == "python"
    assert context.code_for("syntax") == "E003"
    assert context.code_for("TYPES") == "E001"
    assert context.code_for("unknown", "E004") == "E004"
    assert context.span_range((8, 9)) == {
        "start": {"line": 0, "character": 8},
        "end": {"line": 0, "character": 9},
    }
    assert context.line_text(2) == "let y = 2;"
    assert context.line_text(99) == ""
    assert context.with_phase("codegen").phase == "codegen"
    assert context["path"] == "src/main.pg"
    assert context.to_dict()["code_map"] == {"syntax": "E003", "types": "E001"}
    with pytest.raises(ValueError):
        diag_create_diagnostic_context("x", "parse", code_map={"bad": "E009"})


def test_pillar_error_carries_code_severity_span_and_phase():
    context = diag_create_diagnostic_context(
        "let x = @;", "parse", "python", {"syntax": "E003"}, path="main.pg"
    )
    error = diag_create_pillar_error(
        "syntax",
        "unexpected token",
        context=context,
        severity="warning",
        span=(8, 9),
    )
    assert type(error) is PillarError
    assert isinstance(error, PillarError)
    assert (error.code, error.severity, error.span, error.phase) == (
        "E003",
        "warning",
        (8, 9),
        "parse",
    )
    assert error.backend == "python"
    assert error.path == "main.pg"
    assert "error[E003]: unexpected token" in error.pretty()

    translated = TranslateError("bad call", 2, 4, (10, 14))
    converted = diag_create_pillar_error(translated, severity="info")
    assert converted.code == "E003"
    assert converted.severity == "info"
    assert converted.span == (10, 14)
    with pytest.raises(ValueError):
        diag_create_pillar_error("E009", "bad code")


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("lexer: unexpected token", "E003"),
        ("parser: missing semicolon", "E003"),
        ("type mismatch in let binding", "E001"),
        ("borrow checker rejected a move", "E002"),
        ("runtime panic in generated code", "E002"),
        ("codegen failed for native target", "E004"),
        ("dependency cycle detected", "E005"),
        ("missing module std::fs", "E006"),
        ("unsupported native pattern", "E007"),
        ("malformed snapshot digest", "E008"),
    ],
)
def test_error_code_mapping_covers_each_failure_family(failure: str, expected: str):
    assert diag_map_error_code_e001_to_e008(failure) == expected


def test_error_code_mapping_prefers_exception_codes_and_phase_hints():
    assert (
        diag_map_error_code_e001_to_e008(UnsupportedNativePatternError("x")) == "E007"
    )
    assert (
        diag_map_error_code_e001_to_e008(InvalidIntegerParseLiteralError("0x", 1, 1))
        == "E008"
    )
    assert diag_map_error_code_e001_to_e008("unclassified", phase="codegen") == "E004"
    assert diag_map_error_code_e001_to_e008("whatever", default="E006") == "E006"
    assert diag_map_error_code_e001_to_e008("E002") == "E002"


def test_terminal_snippet_renders_source_line_and_caret_for_span():
    context = diag_create_diagnostic_context(
        "let x = @;\nlet y = 2;", "parse", path="src/main.pg"
    )
    error = diag_create_pillar_error(
        "E003", "unexpected token", context=context, span=(8, 9)
    )
    rendered = diag_format_terminal_snippet(error, context=context)
    lines = rendered.splitlines()
    assert lines[0] == "error[E003]: unexpected token"
    assert lines[1] == "  --> src/main.pg:1:9"
    assert lines[2] == "1 | let x = @;"
    assert lines[3] == "  |         ^"
    assert "phase=parse backend=python" in rendered


def test_terminal_snippet_handles_tabs_reversed_spans_and_multiline_spans():
    tabbed = diag_format_terminal_snippet(
        {"code": "E001", "message": "bad type", "span": (9, 8)},
        "let\tx = 1;",
    )
    caret_lines = [line for line in tabbed.splitlines() if "^" in line]
    assert caret_lines == ["  | " + " " * 8 + "^"]

    clamped = diag_format_terminal_snippet(
        {"code": "E001", "message": "too wide", "span": (8, 999)},
        "let x = 1;",
    )
    assert any(line.endswith("^^") for line in clamped.splitlines())

    multiline = diag_format_terminal_snippet(
        {"code": "E002", "message": "spanning", "span": (0, 11)},
        "one\ntwo\nthree",
        max_lines=2,
    )
    assert "1 | one" in multiline
    assert "2 | two" in multiline
    assert "3 | three" not in multiline
    assert "1 more line suppressed" in multiline

    colored = diag_format_terminal_snippet(
        {"code": "E003", "message": "x"}, "x = 1;", color=True
    )
    assert "\x1b[31merror[E003]: x\x1b[0m" in colored


def test_json_payload_is_stable_editor_ready_and_accepts_many_errors():
    context = diag_create_diagnostic_context(
        "let x = @;", "parse", "python", path="src/main.pg"
    )
    error = diag_create_pillar_error(
        "E003", "unexpected token", context=context, severity="warning", span=(8, 9)
    )
    encoded = diag_format_json_payload(error, context=context)
    payload = json.loads(encoded)
    assert payload["schema"] == "polyglot-diagnostics/v1"
    assert payload["count"] == 1
    assert payload["code"] == "E003"
    detail = payload["diagnostics"][0]
    assert detail["span"] == [8, 9]
    assert (detail["line"], detail["col"]) == (1, 9)
    assert detail["excerpt"] == "let x = @;"
    assert detail["severity"] == "warning"
    assert isinstance(detail["fix"], dict)
    assert diag_format_json_payload(error, context=context, as_object=True) == payload
    assert "\n" in diag_format_json_payload(error, context=context, pretty=True)

    many = diag_format_json_payload(
        [error, {"code": "E001", "message": "second", "severity": "info"}],
        context=context,
        as_object=True,
    )
    assert many["count"] == 2
    assert "code" not in many
    assert [item["code"] for item in many["diagnostics"]] == ["E003", "E001"]


def test_aggregate_keeps_first_root_cause_even_when_truncating_cascades():
    root = {
        "code": "E001",
        "message": "root cause: " + "x" * 500,
        "severity": "error",
        "root_cause": True,
        "span": [0, 3],
    }
    cascade = lambda index: {
        "code": "E003",
        "message": f"cascade {index}",
        "caused_by": 0,
    }
    aggregate = diag_aggregate_without_truncating_root_cause(
        [root, cascade(1), cascade(2), {"code": "E004", "message": "other root"}],
        max_reported=2,
    )
    assert aggregate.total == 4
    assert len(aggregate) == 2
    assert aggregate[0].code == "E001"
    assert aggregate[0].message == root["message"]
    assert aggregate[0].span == (0, 3)
    assert aggregate.cascade_count == 2
    assert aggregate.suppressed == 2
    summary = aggregate.to_dict()
    assert summary["root_cause_preserved"] is True
    assert summary["truncated"] is True
    assert summary["root_cause"]["message"] == root["message"]
    assert [record["code"] for record in summary["diagnostics"]] == ["E001", "E003"]


def test_redaction_normalizes_home_workspace_windows_and_nested_payloads():
    options = {
        "home": "/Users/alice",
        "workspace": "/Users/alice/project",
        "placeholders": {"workspace": "PROJECT", "home": "USER", "tmp": "TEMP"},
    }
    text = (
        "read /Users/alice/project/src/main.pg and /Users/alice/notes.txt "
        r"from C:\Users\bob\secret.txt plus /private/var/folders/ab/cache/out.o"
    )
    redacted = diag_redact_sensitive_paths(text, **options)
    assert "PROJECT/src/main.pg" in redacted
    assert "USER/notes.txt" in redacted
    assert r"USER\secret.txt" in redacted
    assert "TEMP/ab/cache/out.o" in redacted
    assert "alice" not in redacted and "bob" not in redacted

    nested = diag_redact_sensitive_paths(
        {"path": ["/Users/alice/project/a.pg", "/Users/alice/b.pg"], "kind": "type"},
        **options,
    )
    assert nested == {
        "path": ["PROJECT/a.pg", "USER/b.pg"],
        "kind": "type",
    }
    assert isinstance(
        diag_redact_sensitive_paths(Path("/Users/alice/project/a.pg"), **options), str
    )
    assert (
        diag_redact_sensitive_paths("type mismatch at line 2", **options)
        == "type mismatch at line 2"
    )


def test_redaction_handles_trailing_separators_and_overlapping_prefixes():
    options = {"home": "/Users/alice", "workspace": "/Users/alice/project"}
    assert diag_redact_sensitive_paths("/Users/alice/", **options) == "~/"
    assert (
        diag_redact_sensitive_paths("/Users/alice/project", **options) == "<workspace>"
    )
    assert (
        diag_redact_sensitive_paths("<workspace>/already-redacted", **options)
        == "<workspace>/already-redacted"
    )
    assert (
        diag_redact_sensitive_paths("/home/bob/work/main.pg", **options)
        == "~/work/main.pg"
    )
    assert (
        diag_redact_sensitive_paths(r"C:/Users/bob/work/main.pg", **options)
        == r"~/work/main.pg"
    )


def test_compile_timing_uses_explicit_samples_without_a_clock():
    report = diag_record_compile_phase_timing({"lex": 0.01, "codegen": 2.0})
    assert report["phase_ms"]["lexing"] == 10.0
    assert report["phase_ms"]["codegen"] == 2000.0
    assert report["phases"]["parsing"]["seconds"] is None
    assert report["phases"]["lexing"]["alias"] == "lex"
    assert report["total_ms"] == 2010.0
    assert report["critical_phase"] == "codegen"
    assert report["measured_phases"] == ["lexing", "codegen"]

    milliseconds = diag_record_compile_phase_timing(
        {"typecheck": 5.0}, unit="ms", phases=("typecheck",)
    )
    assert milliseconds["phase_ms"]["typecheck"] == 5.0


def test_compile_timing_measures_callables_with_wrapper_or_clock():
    calls: list[str] = []

    def wrapper(phase: str, operation):
        calls.append(phase)
        sentinel = object()
        operation()
        return sentinel, 0.25

    measured = diag_record_compile_phase_timing(
        {"monomorphization": lambda: 42, "codegen": 1.0},
        wrapper=wrapper,
    )
    assert calls == ["monomorphization"]
    assert measured["phase_ms"]["monomorphization"] == 250.0
    assert measured["phase_ms"]["codegen"] == 1000.0

    ticks = iter((10.0, 11.5))
    clocked = diag_record_compile_phase_timing(
        {"lexing": lambda: None}, clock=lambda: next(ticks)
    )
    assert clocked["phase_ms"]["lexing"] == 1500.0
    assert clocked["phases"]["lexing"]["observed"] is True


def test_cache_metrics_tracks_both_backends_and_rebuild_reasons():
    report = diag_record_cache_hit_metrics(
        {
            "python": {"hits": 3, "misses": 1, "rebuild_reasons": {"stale": 1}},
            "rust": {"hits": 1, "misses": 1, "rebuild_reasons": {"feature_gate": 1}},
        }
    )
    assert report["backends"]["python"]["hit_rate"] == 0.75
    assert report["backends"]["rust"]["hit_rate"] == 0.5
    assert (report["hits"], report["misses"], report["total"]) == (4, 2, 6)
    assert report["hit_rate"] == 0.6667
    assert report["rebuild_reasons"] == {"feature_gate": 1, "stale": 1}
    assert report["rebuild_total"] == 2


def test_cache_metrics_accepts_scalar_counts_and_a_callable_lookup():
    table = {"a": True, "b": {"hit": False, "reason": "stale"}}
    report = diag_record_cache_hit_metrics(
        hits=2,
        misses=1,
        rebuild_reasons=("cold_start", "cold_start"),
        backend="native",
        wrapper=lambda key: table[key],
        keys=("a", "b"),
    )
    assert report["backends"]["rust"]["hits"] == 3
    assert report["backends"]["rust"]["misses"] == 2
    assert report["rebuild_reasons"] == {"cold_start": 2, "stale": 1}
    assert report["observations"] == 2


def test_benchmark_summary_compares_backends_and_optimization_levels():
    summary = diag_export_benchmark_summary(
        {
            "python": {"O0": 12.5, "O2": 10.0},
            "rust": {"O0": 6.25, "O2": 4.0},
        },
        baseline="python:O0",
    )
    group = summary["workloads"]["default"]
    assert summary["backends"] == ["python", "rust"]
    assert summary["opt_levels"] == ["O0", "O2"]
    assert group["baseline_ms"] == 12.5
    assert len(group["measurements"]) == 4
    comparisons = {
        (item["backend"], item["opt_level"]): item for item in group["comparisons"]
    }
    assert comparisons[("python", "O2")]["speedup"] == 1.25
    assert comparisons[("rust", "O0")]["speedup"] == 2.0
    assert comparisons[("rust", "O2")]["speedup"] == 3.125
    assert comparisons[("rust", "O2")]["delta_ms"] == -8.5
    assert group["fastest"]["backend"] == "rust"
    assert group["fastest"]["opt_level"] == "O2"


def test_benchmark_summary_groups_workloads_and_can_use_a_wrapper():
    observed: list[tuple[str, str]] = []

    def wrapper(backend: str, level: str) -> float:
        observed.append((backend, level))
        return 2.0 if backend == "rust" else 8.0

    summary = diag_export_benchmark_summary(
        {"python": {}, "rust": {}},
        wrapper=wrapper,
        opt_levels=("O0", "O2"),
        baseline="python",
    )
    assert observed == [
        ("python", "O0"),
        ("python", "O2"),
        ("rust", "O0"),
        ("rust", "O2"),
    ]
    assert summary["wrapper_used"] is True
    group = summary["workloads"]["default"]
    assert group["baseline"]["backend"] == "python"
    assert group["baseline"]["opt_level"] == "O0"
    assert [item["duration_ms"] for item in group["comparisons"]] == [8.0, 2.0, 2.0]

    records = [
        {
            "backend": "python",
            "opt_level": "O0",
            "duration_ms": 9.0,
            "workload": "warm",
        },
        {"backend": "rust", "opt_level": "O2", "duration_ms": 3.0, "workload": "cold"},
    ]
    grouped = diag_export_benchmark_summary(records, baseline="python:O0")
    assert set(grouped["workloads"]) == {"warm", "cold"}
    assert grouped["baseline_fallback_workloads"] == ["cold"]
    assert grouped["workloads"]["cold"]["baseline"]["backend"] == "rust"


def test_mapping_helpers_stay_read_only_and_all_functions_are_registered():
    from framework_functions import iter_functions

    names = {item.name for item in iter_functions(8)}
    assert names == {
        "diag_create_diagnostic_context",
        "diag_create_pillar_error",
        "diag_map_error_code_e001_to_e008",
        "diag_format_terminal_snippet",
        "diag_format_json_payload",
        "diag_aggregate_without_truncating_root_cause",
        "diag_redact_sensitive_paths",
        "diag_record_compile_phase_timing",
        "diag_record_cache_hit_metrics",
        "diag_export_benchmark_summary",
    }
    context = diag_create_diagnostic_context("x", "parse")
    assert isinstance(context.code_map, MappingProxyType)
    with pytest.raises(TypeError):
        context.code_map["new"] = "E003"  # type: ignore[index]
    assert re.fullmatch(r"E00[1-8]", diag_map_error_code_e001_to_e008("type error"))
