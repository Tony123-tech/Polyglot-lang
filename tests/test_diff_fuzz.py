"""Tests for the deterministic Python/native differential fuzz harness."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import diff_fuzz as df

ROOT = Path(__file__).resolve().parent.parent


def _ok_compile(source: str, *, scope: str, cache_root: Path | None) -> dict[str, Any]:
    del source, scope, cache_root
    return {"result": 42, "stdout": "", "error": None}


def _different_compile(
    source: str, *, scope: str, cache_root: Path | None
) -> dict[str, Any]:
    del source, scope, cache_root
    return {"result": 43, "stdout": "", "error": None}


def test_generation_is_deterministic_and_covers_categories():
    first = df.generate_case(1234, 7)
    assert first == df.generate_case(1234, 7)
    assert first.source.encode("utf-8") == df.generate_case(1234, 7).source.encode()

    cases = list(df.iter_cases(1234, 16))
    assert len(cases) == 16
    assert [case.index for case in cases] == list(range(16))
    assert all(case.seed == 1234 for case in cases)
    assert {case.category for case in cases} == set(df.CATEGORIES)
    other_seed = list(df.iter_cases(9999, 16))
    assert {case.source for case in cases} != {case.source for case in other_seed}


def test_generated_sources_parse_and_are_natively_codegeneratable():
    from backends.python.translator import PythonBackend
    from backends.rust.codegen import RustCodegen

    for case in df.iter_cases(2026, 12):
        program = df._program_from_source(case.source)
        if case.category == df.CATEGORY_DIAGNOSTICS:
            try:
                PythonBackend(trusted=True).compile(program)
            except Exception as error:
                normalized = df.normalize_error(error)
                assert normalized is not None
                assert normalized["code"].startswith("E00")
            continue
        namespace = PythonBackend(trusted=True).compile(program)
        assert callable(namespace.get("main"))
        RustCodegen(module_name=f"dfuzz_probe_{case.index}").generate(program)


def test_mismatch_writes_exactly_two_artifacts_and_fails_report(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(df, "_python_compile", _ok_compile)
    monkeypatch.setattr(df, "_native_compile", _different_compile)
    artifacts = tmp_path / "artifacts"

    report = df.run_fuzz(
        77,
        1,
        cache_root=tmp_path / "cache",
        artifacts_dir=artifacts,
    )

    assert report.passed is False
    assert report.cases == 1
    assert len(report.mismatches) == 1
    files = sorted(path.name for path in artifacts.iterdir())
    assert files == ["77-0.json", "77-0.poly"]
    payload = json.loads((artifacts / "77-0.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 77
    assert payload["index"] == 0
    assert payload["category"] == df.CATEGORIES[0]
    assert payload["source"] == (artifacts / "77-0.poly").read_text(encoding="utf-8")
    assert "result" in payload["diff"]
    assert payload["artifacts"] == {"json": "77-0.json", "poly": "77-0.poly"}


def test_clean_run_creates_no_artifacts(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(df, "_python_compile", _ok_compile)
    monkeypatch.setattr(df, "_native_compile", _ok_compile)
    artifacts = tmp_path / "never-created"

    report = df.run_fuzz(
        88,
        2,
        cache_root=tmp_path / "cache",
        artifacts_dir=artifacts,
    )

    assert report.passed is True
    assert report.mismatches == []
    assert report.skipped == []
    assert not artifacts.exists()


@pytest.mark.parametrize("number", range(1, 9))
def test_error_normalization_extracts_e_codes(number: int) -> None:
    code = f"E00{number}"
    normalized = df.normalize_error(
        f"ValueError: error[{code}]: broken /Users/alice/secret"
    )
    assert normalized is not None
    assert normalized["code"] == code
    assert "broken" in normalized["message"]
    assert "/Users" not in normalized["message"]


def test_error_normalization_maps_messages_without_explicit_code() -> None:
    assert df.normalize_error("cargo failed") == df.normalize_error("E004 cargo failed")
    assert df.normalize_error("type mismatch at line 2")["code"] == "E001"
    assert df.normalize_error(None) is None


def test_side_effect_and_return_value_differences_are_detected(
    tmp_path: Path,
) -> None:
    case = df.generate_case(31, 1)

    def stdout_compile(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del source, scope, cache_root
        return {"result": 1, "stdout": "a trace\n", "error": None}

    def result_compile(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del source, scope, cache_root
        return {"result": 2, "stdout": "a trace\n", "error": None}

    base = df.run_case(
        case,
        python_compile=_ok_compile,
        native_compile=_ok_compile,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "clean",
    )
    assert base["mismatch"] is False
    assert not (tmp_path / "clean").exists()

    stdout_case = df.run_case(
        case,
        python_compile=stdout_compile,
        native_compile=_ok_compile,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "stdout-artifacts",
    )
    assert stdout_case["mismatch"] is True
    assert "stdout" in stdout_case["diff"]

    result_case = df.run_case(
        case,
        python_compile=_ok_compile,
        native_compile=result_compile,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "result-artifacts",
    )
    assert result_case["mismatch"] is True
    assert "result" in result_case["diff"]


def test_native_platform_failure_is_skipped_not_a_mismatch(tmp_path: Path) -> None:
    def toolchain_failure(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del source, scope, cache_root
        return {"result": None, "stdout": "", "error": "E004 cargo failed"}

    outcome = df.run_case(
        df.generate_case(99, 1),
        python_compile=_ok_compile,
        native_compile=toolchain_failure,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert outcome["mismatch"] is False
    assert outcome["skipped"] is True
    assert "E004" in outcome["skip_reason"]
    assert not (tmp_path / "artifacts").exists()


def test_non_finite_float_normalization_is_stable(
    monkeypatch: Any, tmp_path: Path
) -> None:
    value = {
        "nan": float("nan"),
        "inf": float("inf"),
        "negative_inf": -float("inf"),
    }
    normalized = df._jsonable(value)
    assert normalized == {"nan": "nan", "inf": "inf", "negative_inf": "-inf"}

    def float_compile(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del source, scope, cache_root
        return {"result": value, "stdout": "", "error": None}

    monkeypatch.setattr(df, "_python_compile", float_compile)
    monkeypatch.setattr(df, "_native_compile", float_compile)
    report = df.run_fuzz(
        3,
        2,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert report.passed is True
    assert json.dumps(report.as_dict(), allow_nan=False)


def test_python_only_backend_skips_native_without_mismatch(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(df, "_python_compile", _ok_compile)

    def no_native(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        raise AssertionError("native compile must not run for python-only runs")

    monkeypatch.setattr(df, "_native_compile", no_native)
    report = df.run_fuzz(
        41,
        4,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
        backends=("python",),
    )
    assert report.passed is True
    assert report.cases == 4
    assert len(report.skipped) == 4
    assert all(item["backend"] == "native" for item in report.skipped)
    assert not (tmp_path / "artifacts").exists()


def test_native_compile_uses_isolated_cache_dir_without_overriding_cargo_env(
    monkeypatch: Any, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class FakeNativeBackend:
        def __init__(self, cache_dir: Path, *, trusted: bool) -> None:
            seen["cache_dir"] = cache_dir
            seen["trusted"] = trusted

        def compile(
            self,
            program: Any,
            module_name: str,
            *,
            incremental: bool,
            cache_scope: str,
        ) -> dict[str, Any]:
            del program, module_name, incremental
            seen["target_dir"] = os.environ.get("CARGO_TARGET_DIR")
            seen["cache_scope"] = cache_scope
            return {"main": lambda: 5}

    monkeypatch.setattr(
        df.shutil, "which", lambda name: "/fake/cargo" if name == "cargo" else None
    )
    monkeypatch.setattr("backends.rust.compiler.NativeBackend", FakeNativeBackend)
    previous = os.environ.get("CARGO_TARGET_DIR")

    outcome = df._native_compile(
        "fn main() -> i64 { 5 }", scope="valid-scope", cache_root=tmp_path / "cache"
    )

    assert outcome["result"] == 5
    assert seen["cache_dir"] == tmp_path / "cache" / "native-cache"
    assert seen["trusted"] is True
    assert seen["cache_scope"] == "valid-scope"
    # NativeBackend chooses its own target directory below cache_dir so it can
    # find the extension afterwards; the fuzzer must not override it globally.
    assert seen["target_dir"] == previous
    assert os.environ.get("CARGO_TARGET_DIR") == previous


def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_fuzz.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--case-count" in completed.stdout


def test_cli_json_and_exit_codes(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(df, "_python_compile", _ok_compile)
    monkeypatch.setattr(df, "_native_compile", _ok_compile)
    artifacts = tmp_path / "artifacts"

    assert (
        df.main(
            [
                "--seed",
                "7",
                "--case-count",
                "2",
                "--artifacts",
                str(artifacts),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["cases"] == 2
    assert "/Users" not in json.dumps(payload)

    monkeypatch.setattr(df, "_native_compile", _different_compile)
    assert (
        df.main(
            [
                "--seed",
                "8",
                "--case-count",
                "1",
                "--artifacts",
                str(tmp_path / "mismatch"),
                "--json",
            ]
        )
        == 1
    )
    json.loads(capsys.readouterr().out)

    assert df.main(["--seed", "1", "--backends", "bogus", "--json"]) == 2
    assert "harness error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# E003 native parity
# ---------------------------------------------------------------------------


def _mismatch_comparison() -> dict[str, Any]:
    return {
        "python": {"result": 1, "stdout": "", "error": None},
        "native": {"result": 2, "stdout": "", "error": None},
        "mismatch": True,
    }


def _clean_comparison() -> dict[str, Any]:
    return {
        "python": {"result": 1, "stdout": "", "error": None},
        "native": {"result": 1, "stdout": "", "error": None},
        "mismatch": False,
    }


def test_e003_template_uses_shared_host_arity_error_without_skip() -> None:
    from backends.python.translator import PythonBackend
    from backends.rust.codegen import RustCodegen

    source = df._tpl_diagnostic_e003(random.Random("e003-test"))
    assert "__fs_read_text(p, p)" in source
    assert "skip-native" not in source

    program = df._program_from_source(source)
    for generate in (
        lambda item: PythonBackend(trusted=True).compile(item),
        lambda item: RustCodegen(module_name="dfuzz_e003_test").generate(item),
    ):
        with pytest.raises(Exception) as excinfo:
            generate(program)
        normalized = df.normalize_error(excinfo.value)
        assert normalized is not None
        assert normalized["code"] == "E003"
        assert normalized["message"] == "__fs_read_text expects 1 argument, got 2"


def test_real_run_fuzz_does_not_skip_e003_case(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available for native comparison")

    # Seed 1, index 3 deterministically selects the E003 template.
    case = df.generate_case(1, 3)
    assert "__fs_read_text(p, p)" in case.source

    report = df.run_fuzz(
        1,
        4,
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert report.passed is True
    assert report.skipped == []
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# Mismatch shrinking
# ---------------------------------------------------------------------------


def test_shrink_mismatch_reduces_lines_and_keeps_same_class() -> None:
    marker = "fn dfuzz_marker() -> i64 { 1 }"
    source = (
        f"{marker}\n"
        "let noise_a = 1;\n"
        "let noise_b = 2;\n"
        "let noise_c = 3;\n"
        "let noise_d = 4;"
    )
    seen: list[str] = []

    def compare(candidate: str) -> dict[str, Any]:
        seen.append(candidate)
        return _mismatch_comparison() if marker in candidate else _clean_comparison()

    shrunk = df._shrink_mismatch(source, compare, max_iterations=8)

    assert len(shrunk.splitlines()) < len(source.splitlines())
    assert marker in shrunk
    original_class = df._mismatch_signature(compare(source))
    shrunk_class = df._mismatch_signature(compare(shrunk))
    assert original_class is not None
    assert shrunk_class == original_class


def test_shrink_mismatch_respects_iteration_cap() -> None:
    seen: list[str] = []

    def compare(candidate: str) -> dict[str, Any]:
        seen.append(candidate)
        return _mismatch_comparison()

    source = "a\nb\nc\nd\ne\nf\n"
    shrunk = df._shrink_mismatch(source, compare, max_iterations=3)
    assert len(seen) <= 3
    assert len(shrunk.splitlines()) < len(source.splitlines())

    seen.clear()
    unchanged = df._shrink_mismatch(source, compare, max_iterations=1)
    assert unchanged == source
    assert len(seen) == 1


def test_shrink_mismatch_leaves_clean_matches_untouched() -> None:
    seen: list[str] = []

    def compare(candidate: str) -> dict[str, Any]:
        seen.append(candidate)
        return _clean_comparison()

    source = "fn main() -> i64 { 1 }\n"
    assert df._shrink_mismatch(source, compare, max_iterations=6) == source
    assert seen == [source]


def test_run_case_saves_shrunk_reproducer_artifacts(tmp_path: Path) -> None:
    marker = "fn dfuzz_marker() -> i64 { 1 }"
    source = f"{marker}\nlet a = 1;\nlet b = 2;\nlet c = 3;"

    def compile_marker(
        result: int, source_text: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del scope, cache_root
        return {
            "result": result if marker in source_text else 0,
            "stdout": "",
            "error": None,
        }

    case = df.FuzzCase(
        seed=5,
        index=0,
        category=df.CATEGORIES[0],
        source=source,
    )
    outcome = df.run_case(
        case,
        python_compile=lambda src, **kw: compile_marker(1, src, **kw),
        native_compile=lambda src, **kw: compile_marker(2, src, **kw),
        cache_root=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert outcome["mismatch"] is True
    assert len(outcome["source"].splitlines()) < len(source.splitlines())
    poly_path = tmp_path / "artifacts" / "5-0.poly"
    json_path = tmp_path / "artifacts" / "5-0.json"
    assert poly_path.read_text(encoding="utf-8") == outcome["source"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source"] == poly_path.read_text(encoding="utf-8")
    assert "result" in payload["diff"]


# ---------------------------------------------------------------------------
# Expanded corpus
# ---------------------------------------------------------------------------


def test_every_declared_category_appears_in_generated_batch() -> None:
    count = len(df.CATEGORIES) * 2
    cases = list(df.iter_cases(31415, count))
    assert {case.category for case in cases} == set(df.CATEGORIES)
    assert [case.source for case in cases] == [
        df.generate_case(31415, index).source for index in range(count)
    ]


def test_new_templates_parse_compile_and_codegenerate() -> None:
    from backends.python.translator import PythonBackend
    from backends.rust.codegen import RustCodegen

    new_categories = (
        df.CATEGORY_OPTION_RESULT,
        df.CATEGORY_STRINGS_FLOATS,
        df.CATEGORY_COLLECTIONS,
        df.CATEGORY_ASYNC_FUTURES,
        df.CATEGORY_STDLIB_INCLUDES,
        df.CATEGORY_IMMUTABLE_STATE,
    )
    checked = 0
    for category in new_categories:
        for template in df._TEMPLATES[category]:
            for sample in range(3):
                source = template(random.Random(f"template-check:{sample}"))
                program = df._program_from_source(source)
                namespace = PythonBackend(trusted=True).compile(program)
                assert callable(namespace.get("main"))
                namespace["main"]()
                RustCodegen(module_name=f"dfuzz_template_{checked}").generate(program)
                checked += 1
    assert checked > 0
