"""Telemetry integration tests for the visualizer server and frontend."""

from __future__ import annotations

import json
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

import api.telemetry as telemetry_module
from api import visualizer_server
from api.telemetry import collector, save_baseline, snapshot
from api.visualizer_server import INDEX_FILE, health_payload, make_server, run_request

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "fn main() -> i64 { 1 }"


@pytest.fixture(autouse=True)
def _clean_telemetry() -> Any:
    collector().reset()
    yield
    collector().reset()


@contextmanager
def _serve() -> Any:
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urlerror.HTTPError as error:
        return error.code, json.loads(error.read())


def test_telemetry_endpoint_returns_redacted_snapshot() -> None:
    with (
        _serve() as base,
        urlrequest.urlopen(base + "/api/telemetry", timeout=30) as response,
    ):
        assert response.status == 200
        assert response.headers.get("Cache-Control") == "no-store"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("Referrer-Policy") == "no-referrer"
        payload = json.loads(response.read())

    assert payload["schema"] == "polyglot-telemetry/v1"
    assert {
        "available",
        "note",
        "timings",
        "cache",
        "monomorph",
        "hotpatch",
        "recent_events",
    } <= payload.keys()
    text = json.dumps(payload)
    assert sys.executable not in text
    assert str(ROOT) not in text
    assert re.search(r"/(?:Users|home|private)/", text) is None

    with _serve() as base, pytest.raises(urlerror.HTTPError) as error:
        urlrequest.urlopen(base + "/api/telemetry/nope", timeout=30)
    assert error.value.code == 404


@pytest.mark.parametrize(
    ("reused", "compile_ms", "expected_hits", "expected_misses"),
    [
        (False, 12.5, 0, 1),
        (False, 0.0, 0, 1),
        (True, 0.0, 1, 0),
        (None, 0.0, 0, 0),
    ],
)
def test_run_request_records_timings_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    reused: bool | None,
    compile_ms: float,
    expected_hits: int,
    expected_misses: int,
) -> None:
    def fake_backend(source: str, backend: str, *, timeout: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": backend,
            "ok": True,
            "timed_out": False,
            "compile_ms": compile_ms,
            "run_ms": 2.0,
            "duration_ms": 345.678,
            "diagnostics": [],
        }
        if reused is not None:
            result["reused"] = reused
        if reused is False:
            result["rebuild_reason"] = "source-changed"
        return result

    monkeypatch.setattr(visualizer_server, "run_backend_source", fake_backend)
    response = run_request({"source": SOURCE, "backends": ["python"]})

    assert set(response) == {"ok", "source_length", "results", "duration_ms"}
    assert response["ok"] is True
    assert response["source_length"] == len(SOURCE)
    assert response["results"][0]["backend"] == "python"

    data = snapshot()
    timings = data["timings"]
    assert timings["samples"] == 1
    assert timings["phase_ms"]["compile"] == pytest.approx(compile_ms)
    assert timings["phase_ms"]["total"] == pytest.approx(345.678)
    assert timings["codegen_fragments"] == 0
    cache = data["cache"]
    assert cache["pruned_entries"] == 0
    assert cache["total_hits"] == expected_hits
    assert cache["total_misses"] == expected_misses
    assert cache["backends"]["python"]["misses"] == expected_misses
    if expected_misses:
        assert "source-changed" in cache["rebuild_reasons"]
    assert cache["scopes"][0]["scope"] == "python"


def test_analyze_endpoint_records_compile_and_monomorph_telemetry() -> None:
    with _serve() as base:
        status, payload = _post_json(base + "/api/analyze", {"source": SOURCE})

    assert status == 200
    assert payload["ok"] is True
    data = snapshot()
    assert data["timings"]["samples"] == 1
    assert data["timings"]["phase_ms"]["monomorphization"] >= 0
    assert data["monomorph"]["samples"] == 1
    assert data["monomorph"]["total_specializations"] == 0


def test_frontend_contains_telemetry_panel() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    for token in (
        '<details class="panel telemetry-panel" id="telemetry-panel">',
        ">Telemetry</summary>",
        'id="telemetry-timings"',
        'id="telemetry-cache"',
        'id="telemetry-hotpatch"',
        'id="telemetry-refresh"',
        'id="telemetry-status"',
        'id="telemetry-compare-select"',
        'id="telemetry-compare-btn"',
        'id="telemetry-compare"',
        'fetch("/api/telemetry"',
        'fetch("/api/telemetry/baselines"',
        '"/api/telemetry?compare="',
        "codegen fragments:",
        "Pruned entries",
        "No telemetry recorded yet in this session.",
        "prefers-reduced-motion",
    ):
        assert token in html


def test_health_payload_includes_telemetry_schema_marker() -> None:
    assert health_payload()["telemetry_schema"] == "polyglot-telemetry/v1"
    with (
        _serve() as base,
        urlrequest.urlopen(base + "/api/health", timeout=30) as response,
    ):
        assert json.loads(response.read())["telemetry_schema"] == (
            "polyglot-telemetry/v1"
        )


def test_baselines_endpoint_and_compare_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "baselines"
    monkeypatch.setattr(telemetry_module, "_baseline_dir", lambda: root)
    with (
        _serve() as base,
        urlrequest.urlopen(base + "/api/telemetry/baselines", timeout=30) as response,
    ):
        assert json.loads(response.read()) == {"ok": True, "baselines": []}

    save_baseline("ui")
    with (
        _serve() as base,
        urlrequest.urlopen(base + "/api/telemetry/baselines", timeout=30) as response,
    ):
        assert json.loads(response.read())["baselines"] == ["ui"]

    with (
        _serve() as base,
        urlrequest.urlopen(base + "/api/telemetry?compare=ui", timeout=30) as response,
    ):
        payload = json.loads(response.read())
    assert payload["schema"] == "polyglot-telemetry/v1"
    assert payload["comparison"]["found"] is True
    assert payload["comparison"]["baseline"] == "ui"
    assert all("/" not in line for line in payload["comparison"]["differences"])

    with _serve() as base, pytest.raises(urlerror.HTTPError) as error:
        urlrequest.urlopen(base + "/api/telemetry?compare=../escape", timeout=30)
    assert error.value.code == 400


def test_run_request_forwards_real_fragment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_backend(source: str, backend: str, *, timeout: float) -> dict[str, Any]:
        return {
            "backend": backend,
            "ok": True,
            "compile_ms": 4.0,
            "duration_ms": 9.0,
            "codegen_fragments": 7,
            "diagnostics": [],
        }

    monkeypatch.setattr(visualizer_server, "run_backend_source", fake_backend)
    response = run_request({"source": SOURCE, "backends": ["python"]})
    assert response["ok"] is True
    assert snapshot()["timings"]["codegen_fragments"] == 7
