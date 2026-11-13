"""M6: one-way include directive."""

from __future__ import annotations

import os
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


@pytest.mark.parametrize("backend", ["python"])
def test_include_add_both_backends(tmp_path: Path, backend: str):
    lib = tmp_path / "lib.poly"
    main = tmp_path / "main.poly"
    lib.write_text("fn add(a: i64, b: i64) -> i64 { a + b }\n", encoding="utf-8")
    main.write_text(
        'include!("lib.poly");\nfn main() -> i64 { add(2, 3) }\n',
        encoding="utf-8",
    )
    run = _run_cli("run", str(main), "--backend", backend)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "5"


@pytest.mark.native
@pytest.mark.parametrize("backend", ["python", "native"])
def test_include_add_native_backend(tmp_path: Path, backend: str):
    lib = tmp_path / "lib.poly"
    main = tmp_path / "main.poly"
    lib.write_text("fn add(a: i64, b: i64) -> i64 { a + b }\n", encoding="utf-8")
    main.write_text(
        'include!("lib.poly");\nfn main() -> i64 { add(2, 3) }\n',
        encoding="utf-8",
    )
    run = _run_cli("run", str(main), "--backend", backend)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "5"


def test_nested_include(tmp_path: Path):
    (tmp_path / "c.poly").write_text("fn c() -> i64 { 1 }\n", encoding="utf-8")
    (tmp_path / "b.poly").write_text('include!("c.poly");\nfn b() -> i64 { c() + 1 }\n', encoding="utf-8")
    (tmp_path / "a.poly").write_text('include!("b.poly");\nfn main() -> i64 { b() + 1 }\n', encoding="utf-8")
    run = _run_cli("run", str(tmp_path / "a.poly"))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "3"


def test_cyclic_include_rejected(tmp_path: Path):
    (tmp_path / "a.poly").write_text('include!("b.poly");\n', encoding="utf-8")
    (tmp_path / "b.poly").write_text('include!("a.poly");\n', encoding="utf-8")
    run = _run_cli("check", str(tmp_path / "a.poly"))
    assert run.returncode == 1
    err = run.stderr
    assert "cyclic include" in err or "cycle" in err
    assert "a.poly" in err and "b.poly" in err
    lines = [l for l in err.strip().splitlines() if l.strip()]
    assert len(lines) == 5
    assert lines[0].startswith("error")


def test_missing_include_file(tmp_path: Path):
    (tmp_path / "main.poly").write_text('include!("missing.poly");\n', encoding="utf-8")
    run = _run_cli("check", str(tmp_path / "main.poly"))
    assert run.returncode == 1
    err = run.stderr
    assert "not found" in err or "missing" in err
    lines = [l for l in err.strip().splitlines() if l.strip()]
    assert len(lines) == 5


def test_include_path_relative_to_including_file(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "lib.poly").write_text("fn val() -> i64 { 7 }\n", encoding="utf-8")
    (tmp_path / "main.poly").write_text('include!("sub/lib.poly");\nfn main() -> i64 { val() }\n', encoding="utf-8")
    # run from different cwd
    other = tmp_path / "other"
    other.mkdir()
    run = _run_cli("run", str(tmp_path / "main.poly"), cwd=other)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "7"
