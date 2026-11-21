"""Deterministic Python/Rust-PyO3/WASM differential parity suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backends.wasm import (
    WasmTrapReport,
    wasm_guard_trap_and_oom_paths,
    wasm_layout_linear_memory_and_stack,
)
from tools import three_way_parity as twp


def test_case_generation_is_seed_reproducible() -> None:
    first = list(twp.iter_three_way_cases(2026))
    second = list(twp.iter_three_way_cases(2026))
    assert first == second
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert {case.category for case in first} == {
        "scalar",
        "recursive_generics",
        "traits_bounds",
        "diagnostics",
    }
    assert {case.expected_code for case in first if case.expected_code} == {
        "E001",
        "E002",
        "E003",
        "E008",
    }
    assert next(twp.iter_three_way_cases(2027)).source != first[0].source


def test_python_and_wasm_parity_with_isolated_cache(tmp_path: Path) -> None:
    report = twp.run_three_way_parity(
        2026,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
        backends=("python", "wasm"),
    )
    assert report.passed, report.mismatches
    assert report.cases == 12
    assert (
        report.outcomes["scalar-add"]["python"]["result"]
        == report.outcomes["scalar-add"]["wasm"]["result"]
    )
    assert report.skipped == {case_id: {} for case_id in report.outcomes}
    for case_id in ("recursive-struct", "trait-bounded-impl", "diagnostic-e008"):
        assert report.outcomes[case_id]["python"] == report.outcomes[case_id]["wasm"]
    assert (
        report.outcomes["diagnostic-e003-syntax"]["python"]["error"]["code"] == "E003"
    )
    assert report.outcomes["diagnostic-e003-syntax"]["wasm"]["error"]["code"] == "E003"
    assert report.artifacts == []
    assert not (tmp_path / "artifacts").exists()


def test_syntax_diagnostic_parity_is_consistent(tmp_path: Path) -> None:
    cases = [
        case
        for case in twp.iter_three_way_cases(2026)
        if case.case_id == "diagnostic-e003-syntax"
    ]
    report = twp.run_three_way_parity(
        2026,
        cases=cases,
        cache_root=tmp_path / "cache",
        include_native=False,
    )
    assert report.passed
    assert set(report.outcomes[cases[0].case_id]) == {"python", "wasm"}


def test_divergence_writes_minimal_artifacts_only(
    monkeypatch: Any, tmp_path: Path
) -> None:
    def wrong_wasm(case: twp.ParityCase, root: Path | None) -> dict[str, Any]:
        return {"result": -999, "stdout": "", "error": None}

    monkeypatch.setitem(twp._RUNNERS, "wasm", wrong_wasm)
    cases = [
        case for case in twp.iter_three_way_cases(2026) if case.category == "scalar"
    ]
    artifacts = tmp_path / "artifacts"
    report = twp.run_three_way_parity(
        2026,
        cases=cases,
        cache_root=tmp_path / "cache",
        artifacts_dir=artifacts,
        backends=("python", "wasm"),
    )
    assert not report.passed
    assert report.mismatches
    assert len(report.artifacts) == len(cases)
    assert all(Path(path).exists() for path in report.artifacts)
    assert sorted(path.name for path in artifacts.iterdir()) == [
        "2026-scalar-add.json",
        "2026-scalar-add.poly",
        "2026-scalar-loop.json",
        "2026-scalar-loop.poly",
    ]


def test_cli_reports_pass_and_fail(
    monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    report = twp.ThreeWayReport(seed=2026, cases=12)
    monkeypatch.setattr(twp, "run_three_way_parity", lambda *args, **kwargs: report)
    assert twp.main(["--seed", "2026", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["cases"] == 12

    report.mismatches.append({"case": "scalar-add"})
    assert twp.main(["--seed", "2026"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_wasm_trap_path_is_structured_e008() -> None:
    layout = wasm_layout_linear_memory_and_stack([], stack_slots=[16])
    report: WasmTrapReport = wasm_guard_trap_and_oom_paths(
        layout,
        accesses=[(0, layout.stack_top + 1)],
        span=(1, 4),
    )
    assert not report.ok and report.kind == "out_of_bounds"
    with pytest.raises(Exception) as raised:
        wasm_guard_trap_and_oom_paths(
            layout, accesses=[(0, layout.stack_top + 1)], strict=True, span=(1, 4)
        )
    assert getattr(raised.value, "code", None) == "E008"
    assert getattr(raised.value, "span", None) == (1, 4)


@pytest.mark.slow
def test_all_three_backends_parity(tmp_path: Path) -> None:
    if shutil_missing_cargo():
        pytest.skip("cargo is not available on PATH")
    report = twp.run_three_way_parity(
        2026,
        cache_root=tmp_path / "isolated-cache",
        artifacts_dir=tmp_path / "artifacts",
        include_native=True,
    )
    assert report.passed, report.mismatches
    assert "native" not in report.skipped.get("scalar-add", {})
    assert report.outcomes["scalar-add"]["native"]["result"] == 14
    assert "wasm" not in report.skipped.get("recursive-struct", {})
    assert (
        report.outcomes["recursive-struct"]["native"]["result"]
        == report.outcomes["recursive-struct"]["wasm"]["result"]
    )
    assert report.artifacts == []


def test_unsupported_wasm_construct_is_an_explicit_boundary_skip(
    tmp_path: Path,
) -> None:
    case = twp.ParityCase(
        "dynamic-aggregate-abi",
        "boundary",
        "struct Point { x: i64 }\nfn get(p: Point) -> i64 { p.x }",
        function="get",
        args=(),
        wasm_supported=False,
    )
    report = twp.run_three_way_parity(
        2026,
        cases=[case],
        cache_root=tmp_path / "cache",
        backends=("python", "wasm"),
    )
    assert report.passed
    assert report.skipped[case.case_id]["wasm"].startswith("WASM backend")
    assert "wasm" not in report.outcomes[case.case_id]
    assert report.artifacts == []


def shutil_missing_cargo() -> bool:
    import shutil

    return shutil.which("cargo") is None
