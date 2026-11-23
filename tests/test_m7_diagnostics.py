"""M7: diagnostics upgrade (error codes and spans)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from compiler.core.lexer import lex
from compiler.core.parser import parse

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


def _run_source_cli(
    source: str, cmd: str = "check"
) -> subprocess.CompletedProcess[str]:
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
    from compiler.core.include_expander import IncludeCycleError, IncludeNotFoundError

    errs = [
        (RustCompileError("boom"), "E004"),
        (TranslateError("x", 1, 1), "E003"),
        (UnsupportedNativePatternError("x", 1, 1), "E007"),
        (InvalidIntegerParseLiteralError("x", 1, 1), "E008"),
        (IncludeCycleError("m", "r", "h", "n", 1, 1), "E005"),
        (IncludeNotFoundError("m", "r", "h", "n", 1, 1), "E006"),
    ]
    for e, code in errs:
        s = str(e)
        assert s.count(f"error[{code}]:") == 1, (code, s)


def test_span_present_for_multi_token_expression():
    # Parse a simple binary expression and ensure span is set
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    program = parse(lex("fn main() -> i64 { let x = a + b; 0 }\n"))
    fn = program.items[0]
    let_stmt = fn.body.stmts[0]
    expr = let_stmt.value
    assert getattr(expr, "span", None) is not None
    assert expr.span[1] > expr.span[0]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _e001(),
        lambda: _e002(),
        lambda: _e003(),
        lambda: _e004(),
        lambda: _e005(),
        lambda: _e006(),
        lambda: _e007(),
        lambda: _e008(),
    ],
    ids=["E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008"],
)
def test_every_error_class_prints_span_when_available(factory):
    error = factory()
    text = str(error)
    assert "error[E" in text
    assert "(span 10..20)" in text, text
    assert len([line for line in text.splitlines() if line.strip()]) >= 5


def _e001():
    from backends.rust.compiler import TypeErrorForBoundary

    return TypeErrorForBoundary("i64", "u8", 1, 5, span=(10, 20))


def _e002():
    from compiler.analysis.termination import TerminationError

    return TerminationError("f", "no proof", "add a bound", 1, 5, span=(10, 20))


def _e003():
    from backends.errors import TranslateError

    return TranslateError("unsupported node", 1, 5, span=(10, 20))


def _e004():
    from backends.rust.compiler import RustCompileError

    return RustCompileError("cargo failed", 1, 5, span=(10, 20))


def _e005():
    from compiler.core.include_expander import IncludeCycleError

    return IncludeCycleError("cycle", "cycle", "help", "note", 1, 5, span=(10, 20))


def _e006():
    from compiler.core.include_expander import IncludeNotFoundError

    return IncludeNotFoundError(
        "missing", "missing", "help", "note", 1, 5, span=(10, 20)
    )


def _e007():
    from backends.errors import UnsupportedNativePatternError

    return UnsupportedNativePatternError("bad pattern", 1, 5, span=(10, 20))


def _e008():
    from backends.errors import InvalidIntegerParseLiteralError

    return InvalidIntegerParseLiteralError("nope", 1, 5, span=(10, 20))


def test_native_unsupported_pattern_reports_e007_not_e003():
    # The native backend has no TranslateError (E003) path: it reports
    # unsupported patterns as E007. See SPEC 10.31.
    from backends.errors import UnsupportedNativePatternError
    from backends.rust.compiler import NativeBackend

    source = "fn f(x: Box<i64>) -> i64 { match x { box value => *value } }"
    with pytest.raises(UnsupportedNativePatternError) as exc_info:
        NativeBackend(cache_dir="/tmp/m7_e007").compile(parse(lex(source)))
    text = str(exc_info.value)
    assert "error[E007]:" in text
    assert "error[E003]:" not in text


def test_e008_includes_literal_and_span():
    from backends.rust.compiler import NativeBackend, RustCompileError

    source = 'fn bad() -> i64 { "not an integer".parse::<i64>() }'
    with pytest.raises(RustCompileError) as exc_info:
        NativeBackend(cache_dir="/tmp/m7_e008").compile(parse(lex(source)))
    text = str(exc_info.value)
    assert "error[E008]:" in text
    assert "not an integer" in text
    assert "(span " in text


def test_e006_includes_resolved_path_and_include_site():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "main.poly").write_text('include!("missing.poly");\n', encoding="utf-8")
        r = _run_cli("check", str(d / "main.poly"))
    assert r.returncode == 1
    err = r.stderr
    assert f"{(d / 'missing.poly').resolve()}" in err
    assert "main.poly" in err
    assert "line 1, column 1" in err
    assert "(span " in err


def test_e005_multi_file_cycle_prints_full_chain():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.poly").write_text('include!("b.poly");\n', encoding="utf-8")
        (d / "b.poly").write_text('include!("c.poly");\n', encoding="utf-8")
        (d / "c.poly").write_text('include!("a.poly");\n', encoding="utf-8")
        r = _run_cli("check", str(d / "a.poly"))
    assert r.returncode == 1
    err = r.stderr
    assert "error[E005]:" in err
    # The chain must appear in both the message and the note line.
    message_line = err.splitlines()[0]
    note_line = next(line for line in err.splitlines() if "note:" in line)
    for name in ("a.poly", "b.poly", "c.poly"):
        assert name in message_line
        assert name in note_line
    assert " -> " in message_line
    assert message_line.endswith("a.poly")


def _json_error(args: list[str]) -> dict:
    import json

    r = _run_cli(*args, "--error-format=json")
    assert r.returncode == 1
    payload = json.loads(r.stderr)
    for key in ("code", "message", "line", "col", "span", "source", "fix"):
        assert key in payload
        assert payload["code"] in {f"E{i:03d}" for i in range(1, 9)}
    return payload


def test_error_format_json_e001():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.poly"
        p.write_text("fn f(a: i64) -> u8 { a }\nfn main() -> i64 { 0 }\n", "utf-8")
        payload = _json_error(["check", str(p)])
    assert payload["code"] == "E001"
    assert payload["span"] is not None
    assert payload["line"] == 1


def test_error_format_json_e002():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.poly"
        p.write_text("#[terminates]\nfn f(n: i64) -> i64 { f(n) }\n", "utf-8")
        payload = _json_error(["check", str(p)])
    assert payload["code"] == "E002"
    assert payload["span"] is not None


def test_error_format_json_e005():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.poly").write_text('include!("b.poly");\n', encoding="utf-8")
        (d / "b.poly").write_text('include!("a.poly");\n', encoding="utf-8")
        payload = _json_error(["check", str(d / "a.poly")])
    assert payload["code"] == "E005"
    assert payload["span"] is not None
    assert payload["source"] is not None


def test_error_format_json_e001_fix_suggests_cast():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.poly"
        p.write_text("fn f(a: i64) -> u8 { a }\nfn main() -> i64 { 0 }\n", "utf-8")
        payload = _json_error(["check", str(p)])
    fix = payload["fix"]
    assert "as u8" in fix["message"]
    assert fix["edits"][0]["newText"] == " as u8"


def test_error_format_json_e002_fix_uses_checker_suggestion():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.poly"
        p.write_text("#[terminates]\nfn f(n: i64) -> i64 { f(n) }\n", "utf-8")
        payload = _json_error(["check", str(p)])
    assert payload["code"] == "E002"
    assert payload["fix"]["message"] == (
        "recurse with a smaller argument under a matching lower-bound guard"
    )


def test_public_codes_stop_at_e008():
    from runtime.diagnostics import ERROR_CODES

    assert set(ERROR_CODES) == {f"E{i:03d}" for i in range(1, 9)}
    assert "E009" not in ERROR_CODES
