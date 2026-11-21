"""Behavioral tests for Batch 9: diagnostics, telemetry, and visualization."""

from __future__ import annotations

import json
from importlib import import_module

import pytest

from compiler.frontend.errors import PolyglotError
from diagnostics import context as dctx
from diagnostics import format as dfmt
from diagnostics import telemetry as dtele
from diagnostics import visualizer as dviz
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
from diagnostics.rollout import FRAMEWORK_ROWS
from runtime.diagnostics import PillarError

_SOURCE = "let x = 1;\nlet y = 2;"
_RECORD = {
    "code": "E003",
    "message": "bad token",
    "severity": "error",
    "span": [4, 5],
    "line": 1,
    "col": 5,
    "phase": "parsing",
    "backend": "python",
    "path": "src/main.pg",
    "help": "fix it",
    "note": "noted",
}
_LEGACY_NAMES = [
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
]
_NEW_MODULES = (dctx, dfmt, dtele, dviz)


def _error_code(call):
    """Call *call*, returning the structured PolyglotError code it raises."""
    with pytest.raises(PolyglotError) as caught:
        call()
    assert isinstance(caught.value, PillarError)
    return caught.value.code


def test_rollout_metadata_contains_one_hundred_resolvable_documented_functions():
    assert len(FRAMEWORK_ROWS) == 100
    assert all(len(row) == 4 for row in FRAMEWORK_ROWS)
    names = [row[0] for row in FRAMEWORK_ROWS]
    assert len(set(names)) == 100
    assert names[:10] == _LEGACY_NAMES
    assert all(row[3] == "tests.test_diagnostics_rollout" for row in FRAMEWORK_ROWS)
    assert [row[1] for row in FRAMEWORK_ROWS[:10]] == ["diagnostics.framework"] * 10
    assert [row[2] for row in FRAMEWORK_ROWS[:10]] == [
        "context",
        "context",
        "formatting",
        "formatting",
        "formatting",
        "formatting",
        "telemetry",
        "telemetry",
        "telemetry",
        "telemetry",
    ]
    assert [row[2] for row in FRAMEWORK_ROWS[10:35]] == ["context"] * 25
    assert [row[2] for row in FRAMEWORK_ROWS[35:50]] == ["formatting"] * 15
    assert [row[2] for row in FRAMEWORK_ROWS[50:75]] == ["telemetry"] * 25
    assert [row[2] for row in FRAMEWORK_ROWS[75:]] == ["visualizer"] * 25
    for name, module, _category, _test_module in FRAMEWORK_ROWS:
        function = getattr(import_module(module), name)
        assert callable(function), name
        assert function.__doc__, name


def test_all_new_module_exports_are_callable_with_docstrings():
    for module in _NEW_MODULES:
        assert len(module.__all__) == 25
        assert module.__all__ == sorted(module.__all__)
        for name in module.__all__:
            function = getattr(module, name)
            assert callable(function), name
            assert function.__doc__, name


def test_legacy_ten_functions_smoke():
    context = diag_create_diagnostic_context(_SOURCE, "parse", "py", path="main.pg")
    assert context.backend == "python"
    error = diag_create_pillar_error("E003", "bad")
    assert error.code == "E003"
    assert diag_map_error_code_e001_to_e008("borrow checker rejected a move") == "E002"
    snippet = diag_format_terminal_snippet(_RECORD, _SOURCE)
    assert "error[E003]: bad token" in snippet
    payload = diag_format_json_payload([_RECORD], as_object=True)
    assert payload["count"] == 1
    aggregate = diag_aggregate_without_truncating_root_cause([_RECORD], max_reported=1)
    assert aggregate.root_cause.message == "bad token"
    redacted = diag_redact_sensitive_paths(
        "/Users/alice/project/main.pg", home="/Users/alice", workspace="/none"
    )
    assert redacted == "~/project/main.pg"
    timing = diag_record_compile_phase_timing({"lex": 0.5})
    assert timing["phases"]["lexing"]["seconds"] == 0.5
    cache = diag_record_cache_hit_metrics(hits=3, misses=1)
    assert cache["hits"] == 3
    benchmark = diag_export_benchmark_summary(
        [
            {"backend": "python", "opt_level": "O0", "duration_ms": 10.0},
            {"backend": "rust", "opt_level": "O0", "duration_ms": 5.0},
        ]
    )
    assert benchmark["backends"] == ["python", "rust"]


# ---------------------------------------------------------------------------
# diagnostics.context
# ---------------------------------------------------------------------------


def test_dctx_new_builds_normalized_context():
    context = dctx.dctx_new(_SOURCE, "parse", "py", "src/main.pg", {"syntax": "E003"})
    assert context["phase"] == "parsing"
    assert context["backend"] == "python"
    assert context["code_map"] == {"syntax": "E003"}
    assert context["span"] is None and context["revision"] == 0
    assert _error_code(lambda: dctx.dctx_new(123)) == "E003"


def test_dctx_with_source_preserves_identity_and_bumps_revision():
    context = dctx.dctx_new(_SOURCE, "parse", "py", "main.pg")
    updated = dctx.dctx_with_source(context, "new\nsource")
    assert updated["source"] == "new\nsource"
    assert updated["phase"] == "parsing" and updated["path"] == "main.pg"
    assert updated["revision"] == 1
    assert context["source"] == _SOURCE


def test_dctx_attach_span_sets_one_based_position():
    context = dctx.dctx_new(_SOURCE)
    attached = dctx.dctx_attach_span(context, (8, 9))
    assert attached["span"] == [8, 9]
    assert (attached["line"], attached["col"]) == (1, 9)
    assert dctx.dctx_attach_span(context, (999, 1000))["span"] == [
        len(_SOURCE),
        len(_SOURCE),
    ]


def test_dctx_with_code_alias_adds_alias_without_mutation():
    context = dctx.dctx_new(_SOURCE, code_map={"syntax": "E003"})
    updated = dctx.dctx_with_code_alias(context, "TYPES", "E001")
    assert updated["code_map"] == {"syntax": "E003", "types": "E001"}
    assert context["code_map"] == {"syntax": "E003"}
    assert updated["revision"] == 1


def test_dctx_resolve_code_resolves_aliases_and_defaults():
    context = dctx.dctx_new(_SOURCE, code_map={"syntax": "E003"})
    assert dctx.dctx_resolve_code(context, "SYNTAX") == {
        "key": "SYNTAX",
        "code": "E003",
        "origin": "alias",
        "valid": True,
    }
    assert dctx.dctx_resolve_code(context, "missing", default="E004")["code"] == "E004"
    assert dctx.dctx_resolve_code(context, "E006")["origin"] == "direct"
    assert _error_code(lambda: dctx.dctx_resolve_code(context, "missing")) == "E003"


def test_dctx_line_count_handles_trailing_newlines():
    context = dctx.dctx_new(_SOURCE)
    assert dctx.dctx_line_count(context) == {
        "line_count": 2,
        "empty": False,
        "ends_with_newline": False,
    }
    assert dctx.dctx_line_count("a\n")["ends_with_newline"] is True
    assert dctx.dctx_line_count("")["empty"] is True


def test_dctx_line_text_reports_out_of_range_lines():
    context = dctx.dctx_new(_SOURCE)
    assert dctx.dctx_line_text(context, 2) == {
        "line": 2,
        "text": "let y = 2;",
        "length": 10,
        "in_range": True,
    }
    assert dctx.dctx_line_text(context, 9)["in_range"] is False


def test_dctx_offset_to_line_col_uses_visualizer_positions():
    context = dctx.dctx_new(_SOURCE)
    result = dctx.dctx_offset_to_line_col(context, 8)
    assert result["line"] == 1 and result["col"] == 9
    assert result["position"] == {"line": 0, "character": 8}
    assert dctx.dctx_offset_to_line_col(context, 999)["clamped"] is True


def test_dctx_line_col_to_offset_maps_second_line():
    context = dctx.dctx_new(_SOURCE)
    result = dctx.dctx_line_col_to_offset(context, 2, 1)
    assert result["offset"] == 11
    assert dctx.dctx_line_col_to_offset(context, 0, 1)["offset"] == 0


def test_dctx_span_to_lsp_returns_editor_range():
    context = dctx.dctx_new(_SOURCE)
    result = dctx.dctx_span_to_lsp(context, (4, 5))
    assert result["range"] == {
        "start": {"line": 0, "character": 4},
        "end": {"line": 0, "character": 5},
    }
    assert result["span"] == [4, 5]


def test_dctx_span_clamp_reports_changes():
    context = dctx.dctx_new(_SOURCE)
    result = dctx.dctx_span_clamp(context, (8, 999))
    assert result["span"] == [8, len(_SOURCE)]
    assert result["clamped"] is True and result["source_length"] == len(_SOURCE)


def test_dctx_span_overlap_classifies_relations():
    overlap = dctx.dctx_span_overlap((0, 5), (3, 8))
    assert overlap["overlap"] is True and overlap["intersection"] == [3, 5]
    assert overlap["relation"] == "overlap"
    adjacent = dctx.dctx_span_overlap((0, 3), (3, 6))
    assert adjacent["overlap"] is False and adjacent["relation"] == "adjacent"
    assert dctx.dctx_span_overlap((0, 9), (2, 4))["relation"] == "contains"


def test_dctx_span_union_reports_gaps():
    result = dctx.dctx_span_union([(0, 2), (5, 7)])
    assert result["span"] == [0, 7]
    assert result["gaps"] == [[2, 5]] and result["contiguous"] is False
    assert _error_code(lambda: dctx.dctx_span_union([])) == "E003"


def test_dctx_normalize_severity_exposes_rank():
    assert dctx.dctx_normalize_severity("WARN") == {
        "severity": "warning",
        "rank": 2,
        "canonical": False,
    }
    assert _error_code(lambda: dctx.dctx_normalize_severity("bogus")) == "E003"


def test_dctx_make_record_uses_context_defaults_and_fingerprint():
    context = dctx.dctx_new(_SOURCE, "parse", "py", "main.pg", {"syntax": "E003"})
    record = dctx.dctx_make_record(
        context,
        "syntax",
        "bad token",
        span=(8, 9),
        note="note",
        help_text="help",
        cause=0,
    )
    assert record["code"] == "E003"
    assert (record["line"], record["col"]) == (1, 9)
    assert (record["phase"], record["backend"], record["path"]) == (
        "parsing",
        "python",
        "main.pg",
    )
    assert record["help"] == "help" and len(record["fingerprint"]) == 64


def test_dctx_record_from_exception_reads_structured_attributes():
    context = dctx.dctx_new(_SOURCE)
    error = PolyglotError("E001", "type mismatch", line=2, col=3, span=(8, 9))
    record = dctx.dctx_record_from_exception(context, error)
    assert record["code"] == "E001" and record["message"] == "type mismatch"
    assert (record["line"], record["col"]) == (2, 3)
    assert _error_code(lambda: dctx.dctx_record_from_exception(context, 7)) == "E003"


def test_dctx_record_from_mapping_uses_alias_table():
    context = dctx.dctx_new(_SOURCE, code_map={"types": "E001"})
    record = dctx.dctx_record_from_mapping(
        context, {"code": "types", "message": "bad type", "span": [8, 9]}
    )
    assert record["code"] == "E001" and record["message"] == "bad type"
    assert (
        _error_code(lambda: dctx.dctx_record_from_mapping(context, {"code": "E001"}))
        == "E003"
    )


def test_dctx_record_position_prefers_span():
    context = dctx.dctx_new(_SOURCE)
    record = {**_RECORD, "line": 9, "col": 9, "span": [8, 9]}
    assert dctx.dctx_record_position(context, record) == {
        "line": 1,
        "col": 9,
        "span": [8, 9],
        "origin": "span",
    }
    no_span = {**_RECORD, "line": 4, "col": 2, "span": None}
    assert dctx.dctx_record_position(context, no_span)["origin"] == "fields"


def test_dctx_merge_records_deduplicates_and_sorts():
    second = {**_RECORD, "code": "E001", "message": "type mismatch"}
    result = dctx.dctx_merge_records([_RECORD, second, _RECORD])
    assert result["total"] == 3 and result["unique"] == 2
    assert result["duplicates"] == 1
    assert [item["code"] for item in result["records"]] == ["E001", "E003"]


def test_dctx_group_by_phase_sorted_groups():
    other = {**_RECORD, "phase": "codegen"}
    result = dctx.dctx_group_by_phase([_RECORD, other, _RECORD])
    assert result["phases"] == ["codegen", "parsing"]
    assert result["counts"] == {"codegen": 1, "parsing": 2}
    assert result["total"] == 3


def test_dctx_filter_severity_keeps_records_at_or_above_threshold():
    info = {**_RECORD, "severity": "info"}
    result = dctx.dctx_filter_severity([_RECORD, info], "warning")
    assert result["kept"] == 1 and result["removed"] == 1
    assert result["records"][0]["severity"] == "error"


def test_dctx_code_summary_covers_all_eight_codes():
    result = dctx.dctx_code_summary([_RECORD, {**_RECORD, "code": "E008"}])
    assert set(result["counts"]) == {f"E{number:03d}" for number in range(1, 9)}
    assert result["counts"]["E003"] == 1 and result["total"] == 2
    assert result["by_code"] == {"E003": 1, "E008": 1}


def test_dctx_context_digest_is_stable_and_hex():
    context = dctx.dctx_new(_SOURCE)
    first = dctx.dctx_context_digest(context)
    assert first == dctx.dctx_context_digest(context)
    assert len(first["digest"]) == 64 and first["algorithm"] == "sha256"
    assert first["bytes"] > 0


def test_dctx_context_to_json_round_trips():
    context = dctx.dctx_new(_SOURCE, "parse", "py", "main.pg")
    serialized = dctx.dctx_context_to_json(context, records=[_RECORD])
    payload = json.loads(serialized["payload"])
    assert serialized["record_count"] == 1
    assert payload["phase"] == "parsing" and payload["records"][0]["code"] == "E003"
    assert dctx.dctx_from_json(serialized["payload"])["source"] == _SOURCE
    assert _error_code(lambda: dctx.dctx_from_json("{")) == "E008"


# ---------------------------------------------------------------------------
# diagnostics.format
# ---------------------------------------------------------------------------


def test_dfmt_ansi_and_dfmt_strip_ansi():
    colored = dfmt.dfmt_ansi("boom", "error")
    assert colored["text"] == "\x1b[31mboom\x1b[0m"
    assert dfmt.dfmt_ansi("boom", "error", enabled=False)["text"] == "boom"
    stripped = dfmt.dfmt_strip_ansi(colored["text"])
    assert stripped["text"] == "boom" and stripped["removed"] == 2


def test_dfmt_caret_aligns_highlight_width():
    result = dfmt.dfmt_caret("ab", "cd")
    assert result == {
        "caret": "^^",
        "prefix_width": 2,
        "highlight_width": 2,
        "tab_width": 4,
    }


def test_dfmt_gutter_formats_line_numbers():
    assert dfmt.dfmt_gutter(7, 3) == {"text": "  7", "width": 3, "number": 7}
    assert dfmt.dfmt_gutter(12)["text"] == "12"


def test_dfmt_header_renders_severity_code_and_message():
    assert dfmt.dfmt_header(_RECORD)["text"] == "error[E003]: bad token"
    colored = dfmt.dfmt_header(_RECORD, color=True)
    assert colored["text"].startswith("\x1b[31merror[E003]")


def test_dfmt_inline_renders_compact_location():
    result = dfmt.dfmt_inline(_RECORD)
    assert result["text"] == "src/main.pg:1:5: error[E003]: bad token"
    assert result["path"] == "src/main.pg"


def test_dfmt_snippet_renders_source_caret_and_context():
    result = dfmt.dfmt_snippet(_RECORD, _SOURCE)
    assert "error[E003]: bad token" in result["text"]
    assert "--> src/main.pg:1:5" in result["text"]
    assert any("^" in line for line in result["lines"])
    assert "context: phase=parsing backend=python" in result["text"]
    assert _error_code(lambda: dfmt.dfmt_snippet(_RECORD, None)) == "E006"
    assert (
        _error_code(lambda: dfmt.dfmt_snippet(_RECORD, _SOURCE, max_lines=0)) == "E003"
    )


def test_dfmt_code_frame_marks_focus_lines():
    result = dfmt.dfmt_code_frame(_RECORD, _SOURCE)
    assert result["start"] == 1 and result["end"] == 2
    assert result["focus_lines"] == [1]
    assert result["lines"][0].startswith(">>")


def test_dfmt_highlight_splits_source_without_losing_text():
    result = dfmt.dfmt_highlight(_SOURCE, (4, 5))
    assert result["segments"][1] == {
        "text": "x",
        "start": 4,
        "end": 5,
        "kind": "*",
    }
    assert result["text"] == _SOURCE


def test_dfmt_wrap_message_respects_width():
    result = dfmt.dfmt_wrap_message("one two three", width=7)
    assert result["lines"] == ["one two", "three"]
    assert all(len(line) <= 7 for line in result["lines"])


def test_dfmt_truncate_marks_shortened_text():
    result = dfmt.dfmt_truncate("abcdefgh", 4)
    assert result["text"] == "abc…" and result["truncated"] is True
    assert dfmt.dfmt_truncate("ab", 4)["truncated"] is False


def test_dfmt_terminal_width_expands_tabs_and_caps():
    result = dfmt.dfmt_terminal_width(["ab", "a\tb"])
    assert result["raw_width"] == 5 and result["width"] == 5
    assert dfmt.dfmt_terminal_width(["abcdef"], cap=3)["capped"] is True


def test_dfmt_escape_control_makes_controls_visible():
    result = dfmt.dfmt_escape_control("a\nb\tc")
    assert result["text"] == "a\\nb\\tc" and result["escaped"] == 2


def test_dfmt_json_diagnostic_and_payload_round_trip():
    single = dfmt.dfmt_json_diagnostic(_RECORD)
    assert json.loads(single["payload"])["code"] == "E003"
    payload = dfmt.dfmt_json_payload([_RECORD])
    assert payload["count"] == 1
    parsed = dfmt.dfmt_from_json(payload["payload"])
    assert parsed["count"] == 1 and parsed["records"][0]["message"] == "bad token"
    assert _error_code(lambda: dfmt.dfmt_from_json("not json")) == "E008"


def test_dfmt_csv_has_stable_header_and_rows():
    result = dfmt.dfmt_csv([_RECORD])
    assert result["header"] == ["code", "severity", "line", "col", "path", "message"]
    assert result["text"].startswith("code,severity,line,col,path,message\n")
    assert result["rows"] == 1


def test_dfmt_markdown_escapes_pipes():
    record = {**_RECORD, "message": "a | b"}
    result = dfmt.dfmt_markdown([record])
    assert "Code" in result["text"] and "Severity" in result["text"]
    assert "| ---- | -------- |" in result["text"]
    assert "a \\| b" in result["text"]


def test_dfmt_table_aligns_columns():
    result = dfmt.dfmt_table(["A", "B"], [["1", "2"]])
    assert result["widths"] == [1, 1]
    assert result["text"].splitlines() == [
        "| A | B |",
        "| - | - |",
        "| 1 | 2 |",
    ]
    assert _error_code(lambda: dfmt.dfmt_table(["A"], [["1", "2"]])) == "E003"


def test_dfmt_lsp_diagnostics_uses_visualizer_ranges():
    result = dfmt.dfmt_lsp_diagnostics([_RECORD], _SOURCE)
    diagnostic = result["diagnostics"][0]
    assert diagnostic["severity"] == 1
    assert diagnostic["range"]["start"] == {"line": 0, "character": 4}
    assert diagnostic["source"] == "python"


def test_dfmt_sarif_like_maps_severity_and_location():
    result = dfmt.dfmt_sarif_like([_RECORD])
    item = result["runs"][0]["results"][0]
    assert item["ruleId"] == "E003" and item["level"] == "error"
    assert item["locations"][0]["physicalLocation"]["region"]["startLine"] == 1


def test_dfmt_github_annotation_escapes_message():
    record = {**_RECORD, "message": "100% bad\nnext"}
    result = dfmt.dfmt_github_annotation(record)
    assert result["text"].startswith("::error file=src/main.pg,line=1,col=5")
    assert "100%25 bad%0Anext" in result["text"]


def test_dfmt_stack_indents_causal_chain():
    causal = {**_RECORD, "code": "E001", "cause": 0}
    result = dfmt.dfmt_stack([_RECORD, causal])
    assert result["depth"] == 2 and result["roots"] == 1
    assert result["lines"][1].startswith("  -> [E001]")


def test_dfmt_summary_line_counts_severities_per_code():
    info = {**_RECORD, "severity": "info"}
    result = dfmt.dfmt_summary_line([_RECORD, info])
    assert result["total"] == 2
    assert result["severities"] == {"error": 1, "info": 1, "warning": 0}
    assert result["text"].startswith("diagnostics: total=2")


def test_dfmt_path_label_redacts_explicit_prefixes():
    result = dfmt.dfmt_path_label(
        "/Users/alice/project/main.pg", home="/Users/alice", workspace="/none"
    )
    assert result["text"] == "~/project/main.pg"
    assert result["matched"] == ["home"]
    windows = dfmt.dfmt_path_label(
        r"C:\Users\alice\src\main.pg", home=r"C:\Users\alice"
    )
    assert windows["text"] == r"~\src\main.pg"


# ---------------------------------------------------------------------------
# diagnostics.telemetry
# ---------------------------------------------------------------------------


def test_dtele_new_initializes_empty_accumulator():
    state = dtele.dtele_new("build", {"ci": "true"})
    assert state["schema"] == "polyglot-telemetry/v2"
    assert state["tags"] == {"ci": "true"}
    assert state["samples"] == {} and state["events"] == []


def test_dtele_sample_converts_units_and_appends():
    state = dtele.dtele_new()
    result = dtele.dtele_sample(state, "latency", 5, unit="ms")
    assert result["seconds"] == 0.005 and result["count"] == 1
    assert result["telemetry"]["samples"]["latency"] == [0.005]
    assert _error_code(lambda: dtele.dtele_sample(state, "latency", "slow")) == "E003"


def test_dtele_counter_and_gauge_update_copies():
    state = dtele.dtele_new()
    counted = dtele.dtele_counter(state, "hits", 3)
    assert counted["value"] == 3
    gauged = dtele.dtele_gauge(counted["telemetry"], "temp", 21.5)
    assert gauged["previous"] is None and gauged["value"] == 21.5
    assert state["counters"] == {}


def test_dtele_timing_supports_explicit_and_injected_clocks():
    state = dtele.dtele_new()
    result = dtele.dtele_timing(state, "parse", 0.25)
    assert result["phase"] == "parsing" and result["seconds"] == 0.25
    assert result["observed"] is False
    ticks = iter([1.0, 1.5])
    measured = dtele.dtele_timing(state, "lex", lambda: None, clock=lambda: next(ticks))
    assert measured["seconds"] == 0.5 and measured["observed"] is True
    assert _error_code(lambda: dtele.dtele_timing(state, "lex", lambda: None)) == "E003"
    assert (
        _error_code(
            lambda: dtele.dtele_timing(state, "codegen", {"failure": "cargo failed"})
        )
        == "E004"
    )


def test_dtele_merge_combines_series_and_counters():
    first = dtele.dtele_sample(dtele.dtele_new("a"), "x", 1)["telemetry"]
    second = dtele.dtele_sample(dtele.dtele_new("b"), "x", 2)["telemetry"]
    second = dtele.dtele_counter(second, "hits", 4)["telemetry"]
    merged = dtele.dtele_merge(first, second)
    assert merged["left_name"] == "a" and merged["right_name"] == "b"
    assert merged["telemetry"]["samples"]["x"] == [1.0, 2.0]
    assert merged["telemetry"]["counters"]["hits"] == 4


def test_dtele_percentile_uses_nearest_rank():
    result = dtele.dtele_percentile([1, 2, 3, 4], 50)
    assert result["value"] == 2 and result["method"] == "nearest-rank"
    assert result["count"] == 4
    assert _error_code(lambda: dtele.dtele_percentile([], 50)) == "E003"


def test_dtele_stats_computes_distribution():
    result = dtele.dtele_stats([1, 2, 3, 4])
    assert result["mean"] == 2.5 and result["median"] == 2.5
    assert result["min"] == 1 and result["max"] == 4
    assert result["stdev"] == pytest.approx(1.118033988749895)


def test_dtele_histogram_buckets_and_counts_edges():
    result = dtele.dtele_histogram([0.0005, 0.005, 5.0], boundaries=(0.001, 0.01, 1.0))
    assert result["underflow"] == 1 and result["overflow"] == 1
    assert result["buckets"][1]["count"] == 1
    assert result["count"] == 3


def test_dtele_rate_computes_ratio():
    assert dtele.dtele_rate(3, 1) == {
        "label": "cache",
        "hits": 3,
        "misses": 1,
        "total": 4,
        "rate": 0.75,
    }
    assert _error_code(lambda: dtele.dtele_rate(-1, 0)) == "E003"


def test_dtele_phase_timings_canonicalizes_phases():
    result = dtele.dtele_phase_timings({"lex": 0.1, "parse": {"seconds": 0.2}})
    assert result["phases"]["lexing"]["seconds"] == 0.1
    assert result["phases"]["parsing"]["ms"] == 200.0
    assert result["critical_phase"] == "parsing"
    assert result["phases"]["codegen"]["observed"] is False
    assert (
        _error_code(
            lambda: dtele.dtele_phase_timings({"emit": {"failure": "zig missing"}})
        )
        == "E004"
    )


def test_dtele_cache_metrics_groups_backends_and_reasons():
    result = dtele.dtele_cache_metrics(
        [True, "miss", "stale", {"backend": "rs", "hit": False}]
    )
    assert result["hits"] == 1 and result["misses"] == 3
    assert result["backends"]["rust"]["misses"] == 1
    assert result["rebuild_reasons"] == {"stale": 1}
    assert result["hit_rate"] == 0.25


def test_dtele_benchmark_model_computes_modeled_speedups():
    model = dtele.dtele_benchmark_model(
        [
            {"backend": "python", "opt_level": "O0", "duration": 10},
            {"backend": "rust", "opt_level": "O0", "duration": 5},
        ],
        unit="ms",
    )
    assert model["baseline_duration_ms"] == 10.0
    assert model["comparisons"][1]["speedup"] == 2.0
    assert (
        _error_code(
            lambda: dtele.dtele_benchmark_model([{"backend": "quantum", "duration": 1}])
        )
        == "E007"
    )


def test_dtele_benchmark_compare_uses_tolerance():
    result = dtele.dtele_benchmark_compare([1, 2, 3], [2, 3, 4])
    assert result["ratio"] == pytest.approx(1.5)
    assert result["faster"] is False and result["within_tolerance"] is False
    assert dtele.dtele_benchmark_compare([1, 2], [1, 2])["within_tolerance"] is True


def test_dtele_trace_span_and_span_duration_validate_ranges():
    span = dtele.dtele_trace_span("root", "main", start=0, duration=5)
    assert span["end"] == 5 and span["duration"] == 5
    measured = dtele.dtele_span_duration(
        {"span_id": "a", "name": "a", "start": 1, "end": 3, "duration": 2}
    )
    assert measured["duration"] == 2 and measured["consistent"] is True
    assert (
        _error_code(
            lambda: dtele.dtele_span_duration(
                {"span_id": "a", "name": "a", "start": 1, "end": 3, "duration": 9}
            )
        )
        == "E003"
    )


def test_dtele_trace_tree_builds_hierarchy_and_rejects_bad_graphs():
    spans = [
        dtele.dtele_trace_span("root", "main", start=0, duration=5),
        dtele.dtele_trace_span("child", "parse", start=1, duration=2, parent="root"),
    ]
    tree = dtele.dtele_trace_tree(spans)
    assert tree["roots"] == ["root"] and tree["depth"] == 1
    assert tree["nodes"]["root"]["children"] == ["child"]
    missing = [
        dtele.dtele_trace_span("orphan", "x", start=0, duration=1, parent="gone")
    ]
    assert _error_code(lambda: dtele.dtele_trace_tree(missing)) == "E006"
    cyclic = [
        dtele.dtele_trace_span("a", "a", start=0, duration=1, parent="b"),
        dtele.dtele_trace_span("b", "b", start=0, duration=1, parent="a"),
    ]
    assert _error_code(lambda: dtele.dtele_trace_tree(cyclic)) == "E005"


def test_dtele_critical_path_finds_longest_chain():
    spans = [
        dtele.dtele_trace_span("root", "main", start=0, duration=5),
        dtele.dtele_trace_span("child", "parse", start=1, duration=2, parent="root"),
        dtele.dtele_trace_span("other", "emit", start=2, duration=1, parent="root"),
    ]
    result = dtele.dtele_critical_path(spans)
    assert result["span_ids"] == ["root", "child"]
    assert result["duration"] == 7 and result["branches"] == 1


def test_dtele_threshold_check_reports_violations_with_tolerance():
    strict = dtele.dtele_threshold_check({"latency": 12}, {"latency": {"max": 10}})
    assert strict["passes"] is False
    assert strict["violations"][0]["metric"] == "latency"
    relaxed = dtele.dtele_threshold_check(
        {"latency": 12}, {"latency": {"max": 10}}, tolerance=2
    )
    assert relaxed["passes"] is True and relaxed["checked"] == 1


def test_dtele_regression_detects_slower_candidate():
    result = dtele.dtele_regression([10, 10], [12, 12])
    assert result["regressed"] is True
    assert result["delta_ratio"] == pytest.approx(0.2)
    assert dtele.dtele_regression([10, 10], [10, 10])["regressed"] is False


def test_dtele_redact_tags_replaces_explicit_prefixes():
    result = dtele.dtele_redact_tags(
        {"path": "/Users/alice/project", "ci": "true"}, prefixes=["/Users/alice"]
    )
    assert result["tags"]["path"] == "<redacted:0>/project"
    assert result["redacted"] == ["path"] and result["count"] == 1


def test_dtele_serialize_deserialize_digest_and_summary():
    state = dtele.dtele_new("build", {"ci": "true"})
    state = dtele.dtele_sample(state, "x", 2)["telemetry"]
    serialized = dtele.dtele_serialize(state)
    assert json.loads(serialized["payload"])["name"] == "build"
    assert dtele.dtele_deserialize(serialized["payload"])["name"] == "build"
    assert len(dtele.dtele_digest(state)["digest"]) == 64
    summary = dtele.dtele_summary(state)
    assert summary["sample_count"] == 1 and summary["metric_names"] == ["x"]
    assert _error_code(lambda: dtele.dtele_deserialize("{")) == "E008"


# ---------------------------------------------------------------------------
# diagnostics.visualizer
# ---------------------------------------------------------------------------


def test_dviz_new_view_and_with_source_update_revision():
    view = dviz.dviz_new_view(_SOURCE, title="demo")
    assert view["schema"] == "polyglot-visualizer-view/v2"
    assert view["spans"] == [] and view["revision"] == 0
    updated = dviz.dviz_with_source(view, "abc")
    assert updated["source"] == "abc" and updated["revision"] == 1
    assert _error_code(lambda: dviz.dviz_new_view(123)) == "E003"


def test_dviz_add_span_labels_and_clamps_span():
    view = dviz.dviz_new_view(_SOURCE)
    result = dviz.dviz_add_span(view, (4, 5), "x", category="error")
    assert result["span"]["id"] == "s1" and result["span"]["line"] == 1
    assert result["count"] == 1 and result["view"]["revision"] == 1
    clamped = dviz.dviz_add_span(view, (4, 999), "tail")
    assert clamped["span"]["span"] == [4, len(_SOURCE)]


def test_dviz_span_range_and_offset_position_use_lsp_shapes():
    view = dviz.dviz_new_view(_SOURCE)
    assert dviz.dviz_span_range(view, (4, 5))["range"]["start"] == {
        "line": 0,
        "character": 4,
    }
    position = dviz.dviz_offset_position(view, 11)
    assert position["position"] == {"line": 1, "character": 0}
    assert dviz.dviz_offset_position(view, 999)["clamped"] is True


def test_dviz_line_map_returns_offsets_per_line():
    result = dviz.dviz_line_map(dviz.dviz_new_view(_SOURCE))
    assert result["count"] == 2
    assert result["lines"][1]["start"] == 11
    assert result["lines"][0]["text"] == "let x = 1;"


def test_dviz_source_gutter_numbers_lines():
    result = dviz.dviz_source_gutter(dviz.dviz_new_view(_SOURCE))
    assert result["lines"][0] == "1 | let x = 1;"
    assert result["lines"][1] == "2 | let y = 2;"
    assert result["line_count"] == 2


def test_dviz_highlight_line_marks_span_columns():
    added = dviz.dviz_add_span(dviz.dviz_new_view(_SOURCE), (4, 5), "x")
    result = dviz.dviz_highlight_line(added["view"], 1)
    assert result["columns"] == [4]
    assert result["marker_line"][4] == "^"
    assert result["span_ids"] == ["s1"]
    assert _error_code(lambda: dviz.dviz_highlight_line(added["view"], 99)) == "E003"


def test_dviz_annotate_appends_span_labels():
    view = dviz.dviz_add_span(dviz.dviz_new_view(_SOURCE), (4, 5), "x")["view"]
    result = dviz.dviz_annotate(view)
    assert result["lines"][0].endswith("; x")
    assert result["annotations"][0]["labels"] == ["x"]
    assert result["span_count"] == 1


def test_dviz_node_builds_stable_tree_node():
    node = dviz.dviz_node("root", category="root", props={"kind": "test"})
    assert node == {
        "id": "n1",
        "label": "root",
        "category": "root",
        "props": {"kind": "test"},
        "children": [],
    }


def test_dviz_tree_from_records_groups_by_phase_and_severity():
    tree = dviz.dviz_tree_from_records([_RECORD, {**_RECORD, "message": "again"}])
    assert tree["count"] == 5
    assert tree["depth"] == 3
    assert tree["tree"]["children"][0]["label"] == "parsing"
    assert tree["tree"]["children"][0]["children"][0]["label"] == "error"


def test_dviz_tree_count_find_flatten_and_prune():
    tree = dviz.dviz_tree_from_records([_RECORD])["tree"]
    assert dviz.dviz_tree_count(tree) == {"nodes": 4, "leaves": 1, "depth": 3}
    found = dviz.dviz_tree_find(tree, "category", "error")
    assert found["count"] == 2
    flat = dviz.dviz_tree_flatten(tree)
    assert flat["count"] == 4 and flat["nodes"][0]["path"] == "/"
    pruned = dviz.dviz_tree_prune(tree, max_depth=1)
    assert pruned["dropped"] > 0
    assert pruned["tree"]["children"][0]["category"] == "truncated"


def test_dviz_tree_serialization_round_trip():
    tree = dviz.dviz_tree_from_records([_RECORD])["tree"]
    serialized = dviz.dviz_tree_to_json(tree)
    assert json.loads(serialized["payload"])["id"] == tree["id"]
    assert dviz.dviz_tree_from_json(serialized["payload"])["label"] == tree["label"]
    assert _error_code(lambda: dviz.dviz_tree_from_json("{")) == "E008"


def test_dviz_color_for_category_and_legend_are_deterministic():
    assert dviz.dviz_color_for_category("error") == {
        "category": "error",
        "color": "red",
        "ansi": "\x1b[31m",
        "known": True,
    }
    assert dviz.dviz_color_for_category("other")["color"] == "neutral"
    legend = dviz.dviz_legend(["warning", "error"])
    assert [item["category"] for item in legend["entries"]] == ["error", "warning"]
    assert legend["count"] == 2


def test_dviz_timeline_renders_explicit_bars():
    result = dviz.dviz_timeline(
        [
            {"label": "lex", "start": 0, "end": 2},
            {"label": "parse", "start": 2, "end": 5},
        ]
    )
    assert result["width"] == 40 and result["total"] == 5
    assert "#" in result["lines"][0] and len(result["lines"][0]) > 12


def test_dviz_diff_lines_and_diff_spans_are_textual():
    lines = dviz.dviz_diff_lines("a\nb\n", "a\nc\n")
    assert (lines["added"], lines["removed"], lines["hunks"]) == (1, 1, 1)
    spans = dviz.dviz_diff_spans("abc", "axc")
    assert spans["changed"] == 1 and spans["spans"][0]["tag"] == "replace"
    assert spans["spans"][0]["replacement"] == "x"


def test_dviz_minimap_marks_annotated_columns():
    view = dviz.dviz_add_span(dviz.dviz_new_view(_SOURCE), (4, 5), "x")["view"]
    result = dviz.dviz_minimap(view, columns=10)
    assert result["columns"] == 10 and result["line_count"] == 2
    assert "#" in result["rows"][0]["text"]


def test_dviz_view_digest_and_view_report():
    view = dviz.dviz_add_span(dviz.dviz_new_view(_SOURCE), (4, 5), "x")["view"]
    digest = dviz.dviz_view_digest(view)
    assert len(digest["digest"]) == 64
    report = dviz.dviz_view_report(view, [_RECORD], include_diff="changed\n")
    assert report["span_count"] == 1 and report["tree"]["count"] == 4
    assert report["diff"]["removed"] == 2
    assert report["digest"]["digest"] == digest["digest"]
