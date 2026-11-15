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
    (tmp_path / "b.poly").write_text(
        'include!("c.poly");\nfn b() -> i64 { c() + 1 }\n', encoding="utf-8"
    )
    (tmp_path / "a.poly").write_text(
        'include!("b.poly");\nfn main() -> i64 { b() + 1 }\n', encoding="utf-8"
    )
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
    (tmp_path / "main.poly").write_text(
        'include!("sub/lib.poly");\nfn main() -> i64 { val() }\n', encoding="utf-8"
    )
    # run from different cwd
    other = tmp_path / "other"
    other.mkdir()
    run = _run_cli("run", str(tmp_path / "main.poly"), cwd=other)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "7"


# ---------------------------------------------------------------------------
# Include hardening (E33-E37)
# ---------------------------------------------------------------------------


def test_duplicate_include_is_allowed_and_splices_twice(tmp_path: Path):
    # Textual include: the same file may be included twice. Items are spliced
    # twice; there is no de-duplication.
    from ast_nodes import FnDecl
    from include_expander import load_program

    (tmp_path / "lib.poly").write_text(
        "fn add(a: i64, b: i64) -> i64 { a + b }\n", encoding="utf-8"
    )
    (tmp_path / "main.poly").write_text(
        'include!("lib.poly");\n'
        'include!("lib.poly");\n'
        "fn main() -> i64 { add(2, 3) }\n",
        encoding="utf-8",
    )
    program = load_program(tmp_path / "main.poly")
    fn_names = [item.name for item in program.items if isinstance(item, FnDecl)]
    assert fn_names.count("add") == 2

    run = _run_cli("run", str(tmp_path / "main.poly"))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "5"


def test_include_with_parent_directory_segments_is_allowed(tmp_path: Path):
    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    (outer / "lib.poly").write_text("fn val() -> i64 { 9 }\n", encoding="utf-8")
    (inner / "main.poly").write_text(
        'include!("../outer/lib.poly");\nfn main() -> i64 { val() }\n',
        encoding="utf-8",
    )
    run = _run_cli("run", str(inner / "main.poly"))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "9"


def test_include_with_absolute_path_outside_repo_is_allowed(tmp_path: Path):
    lib = tmp_path / "abs_lib.poly"
    lib.write_text("fn val() -> i64 { 11 }\n", encoding="utf-8")
    main = tmp_path / "abs_main.poly"
    main.write_text(
        f'include!("{lib}");\nfn main() -> i64 {{ val() }}\n', encoding="utf-8"
    )
    run = _run_cli("run", str(main))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "11"


def test_include_only_file_produces_empty_program(tmp_path: Path):
    (tmp_path / "lib.poly").write_text("fn val() -> i64 { 1 }\n", encoding="utf-8")
    main = tmp_path / "only.poly"
    main.write_text('include!("lib.poly");\n', encoding="utf-8")

    check = _run_cli("check", str(main))
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == ""

    run = _run_cli("run", str(main))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == ""


def test_include_inside_block_is_parse_error(tmp_path: Path):
    (tmp_path / "lib.poly").write_text("fn val() -> i64 { 1 }\n", encoding="utf-8")
    main = tmp_path / "block.poly"
    main.write_text('fn main() -> i64 { include!("lib.poly"); 1 }\n', encoding="utf-8")
    run = _run_cli("check", str(main))
    assert run.returncode == 1
    assert "include! is only allowed at item position" in run.stderr


def test_include_with_non_string_argument_is_parse_error(tmp_path: Path):
    main = tmp_path / "bad.poly"
    main.write_text("include!(123);\n", encoding="utf-8")
    run = _run_cli("check", str(main))
    assert run.returncode == 1
    assert "include path string" in run.stderr
    assert "NUMBER" in run.stderr
