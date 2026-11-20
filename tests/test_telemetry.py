"""Session telemetry: snapshot shape, redaction, formatting, and CLI commands."""

from __future__ import annotations

import json
import re

import pytest

from api.cli import main
from api.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    collector,
    format_terminal,
    hotpatch_status,
    snapshot,
)
from runtime.hotpatch import CAPABILITIES, platform_support


@pytest.fixture(autouse=True)
def _reset_telemetry():
    collector().reset()
    yield
    collector().reset()


def test_empty_snapshot_shape_and_note():
    data = snapshot()

    assert data["schema"] == "polyglot-telemetry/v1"
    assert data["available"] is False
    assert data["note"] == "no telemetry recorded in this session"
    assert data["timings"] == {
        "phases": {},
        "phase_ms": {},
        "total_ms": None,
        "critical_phase": None,
        "samples": 0,
    }
    assert data["cache"] == {
        "backends": {},
        "total_hits": 0,
        "total_misses": 0,
        "hit_rate": 0.0,
        "rebuild_reasons": {},
        "scopes": [],
    }
    assert data["monomorph"] == {
        "families": {},
        "total_specializations": 0,
        "samples": 0,
    }
    assert data["recent_events"] == []

    hotpatch = data["hotpatch"]
    assert list(hotpatch["capabilities"]) == sorted(CAPABILITIES)
    for name, capability in hotpatch["capabilities"].items():
        assert set(capability) == {"mechanism", "boundary", "diagnostic", "active"}
        assert capability["active"] is False
        assert capability["boundary"] == CAPABILITIES[name]["boundary"]
    assert hotpatch["platform"] == platform_support()
    assert hotpatch["epoch"] is None
    assert hotpatch["rollbacks"] == 0
    assert hotpatch["migrations"] == 0
    assert hotpatch["latest_diagnostic"] is None
    assert hotpatch["events"] == []


def test_snapshot_redacts_paths_and_sanitizes_scope_names(tmp_path):
    home_path = "/Users/waika/private/project.poly"
    temp_path = str(tmp_path / "cache.poly")
    collector().record_compile(backend="python", samples={"lex": 0.01})
    collector().record_cache(backend="python", hits=2, misses=1, scope=home_path)
    collector().record_hotpatch(
        "wrote patch",
        detail=f"wrote {temp_path}",
        diagnostic=f"see {home_path}",
    )

    data = snapshot()
    text = json.dumps(data, sort_keys=True)
    assert "/Users/waika" not in text
    assert "waika" not in text
    assert temp_path not in text
    assert "/private/var/folders" not in text
    assert all(
        "/" not in scope["scope"] and "\\" not in scope["scope"]
        for scope in data["cache"]["scopes"]
    )
    assert data["cache"]["scopes"][0]["scope"].startswith("<redacted:")


def test_record_compile_uses_phase_timing_helper_and_is_stable():
    telemetry = TelemetryCollector()
    telemetry.record_compile(
        backend="python",
        samples={"lex": 0.01, "emit": 0.02},
        source_length=42,
    )

    first = telemetry.snapshot(redact=False)
    assert first["timings"]["phase_ms"]["lexing"] == 10.0
    assert first["timings"]["phase_ms"]["codegen"] == 20.0
    assert first["timings"]["total_ms"] == 30.0
    assert first["timings"]["critical_phase"] == "codegen"
    assert first["timings"]["samples"] == 1
    assert telemetry.snapshot(redact=False) == first
    assert first["recent_events"][0]["kind"] == "compile"
    assert first["recent_events"][0]["payload"]["source_length"] == 42


def test_record_cache_accumulates_hits_misses_reasons_and_hit_rate():
    telemetry = TelemetryCollector()
    telemetry.record_cache(
        backend="python",
        hits=3,
        misses=1,
        rebuild_reasons={"stale": 1},
        scope="build",
    )
    telemetry.record_cache(
        backend="native",
        hits=1,
        misses=1,
        rebuild_reasons={"cold": 1},
        scope="build",
    )
    telemetry.record_cache(
        samples={"python": {"hits": 2, "misses": 0}},
        scope="repl",
    )

    cache = telemetry.snapshot(redact=False)["cache"]
    assert cache["total_hits"] == 6
    assert cache["total_misses"] == 2
    assert cache["hit_rate"] == 0.75
    assert cache["rebuild_reasons"] == {"cold": 1, "stale": 1}
    assert set(cache["backends"]) == {"python", "rust"}
    assert cache["backends"]["rust"]["hits"] == 1

    scopes = {item["scope"]: item for item in cache["scopes"]}
    assert scopes["build"]["hits"] == 4
    assert scopes["build"]["misses"] == 2
    assert scopes["repl"]["hit_rate"] == 1.0


def test_record_monomorph_counts_families_and_compile_payloads():
    telemetry = TelemetryCollector()
    telemetry.record_monomorph(
        {
            "families": [
                {"family": "Vec", "instantiations": 2, "growth_nodes": 8},
                {"family": "Option", "instantiations": 1},
            ],
            "total_instantiations": 3,
        }
    )
    telemetry.record_compile(
        backend="python",
        samples={"lex": 0.01},
        monomorph={"Vec": {"instantiations": 1}},
    )

    monomorph = telemetry.snapshot(redact=False)["monomorph"]
    assert monomorph["samples"] == 2
    assert monomorph["families"]["Vec"]["instantiations"] == 3
    assert monomorph["families"]["Vec"]["specializations"] == 3
    assert monomorph["families"]["Vec"]["growth_nodes"] == 8
    assert monomorph["families"]["Option"]["instantiations"] == 1
    assert monomorph["total_specializations"] == 4


def test_record_monomorph_accepts_inflation_report_and_parsed_program():
    from compiler.types.framework import monomorph_emit_inflation_report
    from lexer import lex
    from parser import parse

    source = """
fn pick<T>(x: T) -> T { x }
fn main() -> i64 { pick(1) + pick::<String>("x".to_string()).len() as i64 }
"""
    program = parse(lex(source))
    report = monomorph_emit_inflation_report(program)

    telemetry = TelemetryCollector()
    telemetry.record_monomorph(report.as_dict())
    telemetry.record_monomorph({"program": program})

    monomorph = telemetry.snapshot(redact=False)["monomorph"]
    assert monomorph["samples"] == 2
    assert monomorph["families"]["pick"]["instantiations"] == 4
    assert monomorph["total_specializations"] == 4


def test_record_hotpatch_updates_epoch_rollbacks_migrations_and_diagnostic():
    telemetry = TelemetryCollector()
    telemetry.record_hotpatch(
        "epoch advanced", capability="hotpatch_rcu_ast_epoch", epoch=4
    )
    telemetry.record_hotpatch(
        "rollback committed",
        capability="hotpatch_rollback_journal_commit",
        epoch=5,
        diagnostic="E008 undo failed",
        detail="scope=build",
    )
    telemetry.record_hotpatch(
        "migration complete", capability="hotpatch_mio_reactor_state", epoch=6
    )
    telemetry.record_hotpatch("diagnostic", diagnostic="E007 refused foreign text")

    hotpatch = telemetry.snapshot(redact=False)["hotpatch"]
    assert hotpatch["epoch"] == 6
    assert hotpatch["rollbacks"] == 1
    assert hotpatch["migrations"] == 1
    assert hotpatch["latest_diagnostic"] == "E007 refused foreign text"
    assert hotpatch["capabilities"]["hotpatch_rcu_ast_epoch"]["active"] is True
    assert hotpatch["capabilities"]["hotpatch_live_dylib_trampoline"]["active"] is False
    assert len(hotpatch["events"]) == 4
    assert (
        hotpatch_status(
            [
                event.to_dict()
                for event in telemetry._events_of_kind(frozenset({"hotpatch"}))
            ]
        )["epoch"]
        == 6
    )


def test_events_are_newest_first_bounded_and_reset_restores_empty_state():
    telemetry = TelemetryCollector(capacity=3)
    for index in range(5):
        telemetry.record_hotpatch(f"event {index}")

    events = telemetry.events(limit=10)
    assert [event["sequence"] for event in events] == [5, 4, 3]
    assert [event["payload"]["event"] for event in events] == [
        "event 4",
        "event 3",
        "event 2",
    ]
    assert telemetry.events(limit=2) == events[:2]

    telemetry.reset()
    assert telemetry.snapshot()["available"] is False
    assert telemetry.events() == []
    assert telemetry.snapshot()["note"] == "no telemetry recorded in this session"


def test_telemetry_event_to_dict_is_stable():
    event = TelemetryEvent("compile", "python", {"samples": {"lex": 1.0}}, 7)
    payload = event.to_dict()
    assert payload == {
        "kind": "compile",
        "backend": "python",
        "payload": {"samples": {"lex": 1.0}},
        "sequence": 7,
    }
    payload["payload"]["samples"]["lex"] = 2.0
    assert event.payload["samples"]["lex"] == 1.0


@pytest.mark.parametrize(
    ("section", "labels"),
    [
        (
            "all",
            ["Telemetry / compile timings", "Cache", "Hotpatch", "RCU epoch"],
        ),
        (
            "telemetry",
            [
                "Telemetry / compile timings",
                "Phases (ms)",
                "total",
                "Monomorph specializations",
            ],
        ),
        (
            "timings",
            ["Telemetry / compile timings", "Monomorph specializations"],
        ),
        (
            "cache",
            ["Cache", "hit rate", "Hits/misses by scope", "Rebuild reasons"],
        ),
        (
            "hotpatch",
            [
                "Hotpatch",
                "RCU epoch",
                "active capabilities",
                "rollbacks",
                "latest diagnostic",
                CAPABILITIES["hotpatch_rcu_ast_epoch"]["boundary"],
            ],
        ),
    ],
)
def test_format_terminal_sections_include_expected_labels(section, labels):
    collector().record_compile(backend="python", samples={"lex": 0.01})
    collector().record_cache(backend="python", hits=2, misses=1, scope="build")
    collector().record_monomorph({"Vec": {"instantiations": 3}})
    collector().record_hotpatch(
        "rollback committed",
        capability="hotpatch_rollback_journal_commit",
        epoch=3,
        diagnostic="E008 undo failed",
    )

    text = format_terminal(snapshot(), section=section)
    for label in labels:
        assert label in text
    assert "\x1b[" not in text
    assert "no telemetry recorded" not in text

    if section == "timings":
        assert text == format_terminal(snapshot(), section="telemetry")


@pytest.mark.parametrize("section", ["all", "telemetry", "cache", "hotpatch"])
def test_format_terminal_has_graceful_empty_state_everywhere(section):
    text = format_terminal(snapshot(), section=section)
    assert "no telemetry recorded in this session" in text
    assert "\x1b[" not in text


@pytest.mark.parametrize("section", ["all", "telemetry", "cache", "hotpatch"])
def test_format_terminal_color_uses_cyan_green_amber_only(section):
    collector().record_cache(backend="python", hits=1, misses=0, scope="build")
    collector().record_hotpatch("migration complete", diagnostic="E008 mismatch")
    payload = snapshot()

    colored = format_terminal(payload, section=section, color=True)
    for code in ("36", "32", "33"):
        assert f"\x1b[{code}m" in colored
    assert set(re.findall(r"\x1b\[(\d+)m", colored)) <= {"36", "32", "33", "0"}
    assert "\x1b[" not in format_terminal(payload, section=section)


def test_cli_telemetry_commands_and_existing_help(capsys):
    assert main(["telemetry"]) == 0
    output = capsys.readouterr().out
    assert "Telemetry / compile timings" in output
    assert "Cache" in output
    assert "Hotpatch" in output

    assert main(["telemetry", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "polyglot-telemetry/v1"

    assert main(["timings"]) == 0
    assert "Monomorph specializations" in capsys.readouterr().out

    assert main(["cache"]) == 0
    assert "Rebuild reasons" in capsys.readouterr().out

    assert main(["hotpatch"]) == 0
    assert "RCU epoch" in capsys.readouterr().out

    assert main(["telemetry", "--color"]) == 0
    assert "\x1b[36m" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "telemetry" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["telemetry", "--help"])
    assert exc.value.code == 0
    assert "--section" in capsys.readouterr().out
