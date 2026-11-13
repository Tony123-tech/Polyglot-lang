"""M7: diagnostics upgrade (error codes and spans)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

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


def _run_source_cli(source: str, cmd: str = "check") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.poly"
        p.write_text(source, encoding="utf-8")
        return _run_cli(cmd, str(p))


def test_e001_boundary_error_has_code_and_span():
    source = "fn main() -> i64 {\n    let x: i64 = 1 as usize;\n    x\n}\n"
    r = _run_source_cli(source, "run")
    assert r.returncode == 1
    err = r.stderr
    assert "error[E001]:" in err
    assert "span" in err


def test_e002_termination_error_has_code_and_span():
    source = "#[terminates]\nfn f(n: i64) -> i64 { f(n) }\n"
    r = _run_source_cli(source, "check")
    assert r.returncode == 1
    err = r.stderr
    assert "error[E002]:" in err
    assert "span" in err


def test_e005_include_cycle_has_code_and_chain():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.poly").write_text('include!("b.poly");\n', encoding="utf-8")
        (d / "b.poly").write_text('include!("a.poly");\n', encoding="utf-8")
        r = _run_cli("check", str(d / "a.poly"))
    assert r.returncode == 1
    err = r.stderr
    assert "error[E005]:" in err
    assert "a.poly" in err and "b.poly" in err


def test_every_error_class_emits_code_exactly_once():
    # Sample one instance per key class via CLI/compile paths
    cases = [
        ("E001", "fn main()->i64{ let x:i64 = 1 as usize; x }\n", "run"),
        ("E002", "#[terminates]\nfn f(n:i64)->i64{ f(n) }\n", "check"),
        ("E003", "fn main(){ unknown_macro!(x); }\n", "run"),
    ]
    for code, src, cmd in cases:
        r = _run_source_cli(src, cmd)
        err = r.stderr
        assert err.count(f"error[{code}]:") == 1, (code, err)
    # E004, E005-E008 via direct construction/paths
    from backends.errors import (
        InvalidIntegerParseLiteralError,
        TranslateError,
        UnsupportedNativePatternError,
    )
    from backends.rust.compiler import RustCompileError
    from include_expander import IncludeCycleError, IncludeNotFoundError
    errs = [
        (RustCompileError("boom"), "E004"),
        (TranslateError("x", 1, 1), "E003"),
        (UnsupportedNativePatternError("x", 1, 1), "E007"),
        (InvalidIntegerParseLiteralError("x", 1, 1), "E008"),
        (IncludeCycleError("m","r","h","n",1,1), "E005"),
        (IncludeNotFoundError("m","r","h","n",1,1), "E006"),
    ]
    for e, code in errs:
        s = str(e)
        assert s.count(f"error[{code}]:") == 1, (code, s)


def test_span_present_for_multi_token_expression():
    # Parse a simple binary expression and ensure span is set
    from lexer import lex
    from parser import parse
    program = parse(lex("fn main() -> i64 { let x = a + b; 0 }\n"))
    fn = program.items[0]
    let_stmt = fn.body.stmts[0]
    expr = let_stmt.value
    assert getattr(expr, "span", None) is not None
    assert expr.span[1] > expr.span[0]
