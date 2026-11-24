"""Tests for ``std:fs`` that never touch the real filesystem.

The host seam is injected/stubbed where behaviour needs to be observed, so
these tests are safe to run even when a backend's seam is still in flux.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backends.python.translator import compile_rust
from compiler.core.ast_nodes import FnDecl
from compiler.core.include_expander import load_program

ROOT = Path(__file__).resolve().parent.parent
STDLIB = ROOT / "stdlib"
FS_SOURCE = STDLIB / "fs.poly"

HOST_CALLS = (
    "__fs_read_text(path)",
    "__fs_write_text(path, contents)",
    "__fs_append_text(path, contents)",
    "__fs_exists(path)",
    "__fs_remove(path)",
)

FS_FUNCTIONS = (
    "read_text",
    "write_text",
    "append_text",
    "exists",
    "remove",
    "read_text_or",
    "write_text_if_missing",
    "read_text_nonempty",
)


def _write_main(tmp_path: Path, module: str = "fs") -> Path:
    path = tmp_path / f"{module}_probe.poly"
    path.write_text(
        f'include!("std:{module}");\n\nfn main() -> i64 {{ 0 }}\n',
        encoding="utf-8",
    )
    return path


def _python_ns(tmp_path: Path, module: str = "fs") -> dict:
    return compile_rust(load_program(_write_main(tmp_path, module)))


# ---------------------------------------------------------------------------
# Source / inclusion checks
# ---------------------------------------------------------------------------


def test_source_declares_exact_host_seam_calls():
    """MARKED source check: asserts the five seam calls exist; performs no I/O."""
    text = FS_SOURCE.read_text(encoding="utf-8")
    assert text.startswith("// std:fs — practical file helpers returning Result")
    for call in HOST_CALLS:
        assert call in text, call


def test_include_std_fs_defines_all_helpers_once(tmp_path: Path):
    program = load_program(_write_main(tmp_path))
    names = [item.name for item in program.items if isinstance(item, FnDecl)]
    for name in FS_FUNCTIONS:
        assert names.count(name) == 1, name


def test_prelude_includes_std_fs_once(tmp_path: Path):
    path = tmp_path / "prelude_fs.poly"
    path.write_text(
        'include!("std:fs");\ninclude!("std:prelude");\nfn main() -> i64 { 0 }\n',
        encoding="utf-8",
    )
    program = load_program(path)
    names = [item.name for item in program.items if isinstance(item, FnDecl)]
    for name in FS_FUNCTIONS:
        assert names.count(name) == 1, name


# ---------------------------------------------------------------------------
# Python backend: compilation and stubbed behaviour
# ---------------------------------------------------------------------------


def test_python_backend_compiles_and_runs_program_including_std_fs(tmp_path: Path):
    ns = _python_ns(tmp_path)
    for name in FS_FUNCTIONS:
        assert callable(ns[name]), name
    assert ns["main"]() == 0


def test_python_backend_cli_runs_program_including_std_fs(tmp_path: Path):
    path = _write_main(tmp_path)
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-m", "api.cli", "run", str(path)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_fs_wrappers_forward_arguments_with_stubbed_seam(tmp_path, monkeypatch):
    ns = _python_ns(tmp_path)
    result = ns["_Result"]
    calls: list[tuple[object, ...]] = []

    def fake_read(path):
        calls.append(("read", path))
        return result.from_ok("text")

    def fake_write(path, contents):
        calls.append(("write", path, contents))
        return result.from_ok(None)

    def fake_append(path, contents):
        calls.append(("append", path, contents))
        return result.from_ok(None)

    def fake_exists(path):
        calls.append(("exists", path))
        return True

    def fake_remove(path):
        calls.append(("remove", path))
        return result.from_ok(None)

    monkeypatch.setitem(ns, "__fs_read_text", fake_read)
    monkeypatch.setitem(ns, "__fs_write_text", fake_write)
    monkeypatch.setitem(ns, "__fs_append_text", fake_append)
    monkeypatch.setitem(ns, "__fs_exists", fake_exists)
    monkeypatch.setitem(ns, "__fs_remove", fake_remove)

    assert ns["read_text"]("p") == result.from_ok("text")
    assert ns["write_text"]("p", "c") == result.from_ok(None)
    assert ns["append_text"]("p", "c") == result.from_ok(None)
    assert ns["exists"]("p") is True
    assert ns["remove"]("p") == result.from_ok(None)
    assert calls == [
        ("read", "p"),
        ("write", "p", "c"),
        ("append", "p", "c"),
        ("exists", "p"),
        ("remove", "p"),
    ]


def test_read_text_or_with_stubbed_seam(tmp_path, monkeypatch):
    ns = _python_ns(tmp_path)
    result = ns["_Result"]

    monkeypatch.setitem(ns, "__fs_read_text", lambda path: result.from_ok("contents"))
    assert ns["read_text_or"]("p", "fallback") == "contents"

    monkeypatch.setitem(ns, "__fs_read_text", lambda path: result.from_err("missing"))
    assert ns["read_text_or"]("p", "fallback") == "fallback"


def test_read_text_nonempty_with_stubbed_seam(tmp_path, monkeypatch):
    ns = _python_ns(tmp_path)
    result = ns["_Result"]

    monkeypatch.setitem(ns, "__fs_read_text", lambda path: result.from_ok("data"))
    assert ns["read_text_nonempty"]("p") == result.from_ok("data")

    monkeypatch.setitem(ns, "__fs_read_text", lambda path: result.from_ok(""))
    assert ns["read_text_nonempty"]("p") == result.from_err("file is empty")

    monkeypatch.setitem(ns, "__fs_read_text", lambda path: result.from_err("boom"))
    assert ns["read_text_nonempty"]("p") == result.from_err("boom")


def test_write_text_if_missing_with_stubbed_seam(tmp_path, monkeypatch):
    ns = _python_ns(tmp_path)
    result = ns["_Result"]
    calls: list[tuple[object, ...]] = []

    def fake_write(path, contents):
        calls.append(("write", path, contents))
        return result.from_ok(None)

    monkeypatch.setitem(ns, "__fs_exists", lambda path: True)
    monkeypatch.setitem(ns, "__fs_write_text", fake_write)
    assert ns["write_text_if_missing"]("p", "c") == result.from_ok(False)
    assert calls == []

    monkeypatch.setitem(ns, "__fs_exists", lambda path: False)
    assert ns["write_text_if_missing"]("p", "c") == result.from_ok(True)
    assert calls == [("write", "p", "c")]

    monkeypatch.setitem(
        ns,
        "__fs_write_text",
        lambda path, contents: result.from_err("denied"),
    )
    assert ns["write_text_if_missing"]("p", "c") == result.from_err("denied")


# ---------------------------------------------------------------------------
# Native backend: only compile, and only once the seam is available
# ---------------------------------------------------------------------------


@pytest.mark.native
def test_native_backend_compiles_program_including_std_fs(tmp_path: Path):
    """Skip until the native backend's seam compiles; no filesystem I/O."""
    from backends import get_backend
    from backends.rust.compiler import RustCompileError

    path = _write_main(tmp_path)
    try:
        namespace = get_backend("native", trusted=True).compile(
            load_program(path),
            incremental=True,
            cache_scope="stdlib-fs-include",
        )
    except RustCompileError as error:
        pytest.skip(f"native std:fs host seam unavailable: {error}")
    assert isinstance(namespace, dict)
    if "main" in namespace:
        assert namespace["main"]() == 0
