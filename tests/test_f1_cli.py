"""F1: CLI, packaging delivery, and examples."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-m", "api.cli", *args],
        cwd=str(cwd or ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _cli_main(argv: list[str]) -> int:
    from api.cli import main

    return main(argv)


def test_run_hello_exits_zero_and_prints_five():
    result = _run_cli("run", "examples/hello.poly")
    assert result.returncode == 0
    assert result.stdout.strip() == "5"
    assert result.stderr == ""


def test_run_type_error_exits_one_and_prints_five_line_error(tmp_path: Path):
    source = tmp_path / "bad.poly"
    source.write_text(
        "fn main() -> i64 {\n"
        "    let x: i64 = 1 as usize;\n"
        "    x\n"
        "}\n",
        encoding="utf-8",
    )
    result = _run_cli("run", str(source))
    assert result.returncode == 1
    lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
    assert 5 <= len(lines) <= 6
    assert lines[0].startswith("error")


def test_check_terminates_example_exits_zero():
    result = _run_cli("check", "examples/terminates.poly")
    assert result.returncode == 0
    assert result.stderr == ""


def test_check_non_terminating_reports_error(tmp_path: Path):
    source = tmp_path / "nonterm.poly"
    source.write_text(
        "#[terminates]\n"
        "fn f(n: i64) -> i64 { f(n) }\n",
        encoding="utf-8",
    )
    result = _run_cli("check", str(source))
    assert result.returncode == 1
    assert "cannot prove termination" in result.stderr
    lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
    assert 5 <= len(lines) <= 6


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is unavailable")
def test_build_hello_produces_extension(tmp_path: Path):
    out_dir = tmp_path / "out"
    result = _run_cli(
        "build",
        "examples/hello.poly",
        "--backend",
        "native",
        "--out-dir",
        str(out_dir),
    )
    assert result.returncode == 0
    artifact = Path(result.stdout.strip())
    assert artifact.exists()
    assert artifact.suffix in {".so", ".dylib", ".pyd"}
    assert artifact.parent == out_dir


def test_examples_directory_contains_seven_runnable_files():
    examples = sorted((ROOT / "examples").glob("*.poly"))
    assert len(examples) == 7
    for path in examples:
        result = _run_cli("check", str(path.relative_to(ROOT)))
        assert result.returncode == 0, f"check failed for {path.name}: {result.stderr}"
        result = _run_cli("run", str(path.relative_to(ROOT)))
        assert result.returncode == 0, f"run failed for {path.name}: {result.stderr}"


def test_cli_entry_point_via_main_import(capsys):
    code = _cli_main(["run", "examples/hello.poly"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "5"
    assert captured.err == ""

def test_console_script_entry_point():
    """Verify the console script is registered and callable."""
    polyglot = shutil.which("polyglot")
    if polyglot is None:
        pytest.skip("polyglot console script not installed")
    result = subprocess.run(
        [polyglot, "run", "examples/hello.poly"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "5"

@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is unavailable")
@pytest.mark.slow
def test_built_extension_loads_and_runs(tmp_path):
    result = _run_cli("build", "examples/hello.poly", "--out-dir", str(tmp_path))
    assert result.returncode == 0
    # Find the .so/.dylib
    libs = list(tmp_path.glob("*.so")) + list(tmp_path.glob("*.dylib"))
    assert libs, "no extension produced"
    # Load it and call add
    from importlib.machinery import ExtensionFileLoader
    from importlib.util import module_from_spec, spec_from_loader
    loader = ExtensionFileLoader("polyglot_native", str(libs[0]))
    spec = spec_from_loader("polyglot_native", loader)
    assert spec is not None
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    assert mod.add(2, 3) == 5
