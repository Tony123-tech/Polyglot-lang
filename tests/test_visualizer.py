"""Tests for the interactive AST / monomorphization visualizer."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from api.visualizer import (
    analyze,
    diagnostic,
    fix_hint,
    offset_to_lsp_position,
    program_to_source,
    span_to_lsp_range,
)
from api.visualizer_server import make_server, run_backend_source
from backends.errors import (
    InvalidIntegerParseLiteralError,
    TranslateError,
    UnsupportedNativePatternError,
)
from backends.rust.compiler import RustCompileError, TypeErrorForBoundary
from include_expander import IncludeCycleError, IncludeNotFoundError
from lexer import lex
from monomorph import monomorphize
from parser import parse
from termination import TerminationError

ROOT = Path(__file__).resolve().parent.parent

GENERIC_SOURCE = """
struct Box<T: Clone> { value: T }

impl<T: Clone> Box<T> {
    fn new(value: T) -> Box<T> { Box { value: value } }
    fn get(&self) -> T { self.value.clone() }
}

fn identity<T>(x: T) -> T { x }

fn main() -> i64 {
    let a: Box<i64> = Box::new(40);
    let b: Box<String> = Box::new("polyglot".to_string());
    a.get() + identity::<i64>(2)
}
"""


def _tree_categories(node: dict[str, Any]) -> set[str]:
    categories = {node.get("category", "")}
    for child in node.get("children", []):
        categories |= _tree_categories(child)
    return categories


def _tree_types(node: dict[str, Any]) -> set[str]:
    types = {node.get("type", "")}
    for child in node.get("children", []):
        types |= _tree_types(child)
    return types


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_analyze_returns_json_trees_and_concrete_source():
    result = analyze(GENERIC_SOURCE)

    assert result["ok"] is True
    assert result["diagnostics"] == []
    assert result["original"]["item_count"] == 4
    assert result["original"]["generic_count"] == 3
    assert result["monomorphized"]["item_count"] > result["original"]["item_count"]

    concrete = result["monomorphized"]["source"]
    assert "struct Box_i64" in concrete
    assert "struct Box_String" in concrete
    assert "impl Box_i64" in concrete
    assert "impl Box_String" in concrete
    assert "fn identity_i64(x: i64) -> i64" in concrete
    assert "Box<T>" not in concrete

    instantiations = {
        (item["base"], item["concrete"], item["kind"])
        for item in result["monomorphized"]["instantiations"]
    }
    assert ("Box", "Box_i64", "struct") in instantiations
    assert ("Box", "Box_String", "struct") in instantiations
    assert ("identity", "identity_i64", "function") in instantiations

    # Everything returned by analyze() must survive a JSON round-trip.
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)["ok"] is True


def test_analyze_tree_is_colour_categorised():
    result = analyze(GENERIC_SOURCE)
    tree = result["original"]["tree"]

    assert tree["category"] == "program"
    categories = _tree_categories(tree)
    assert {
        "program",
        "struct",
        "impl",
        "function",
        "generic",
        "type",
        "statement",
        "expression",
    } <= categories

    types = _tree_types(tree)
    assert {"FnDecl", "StructDecl", "ImplBlock", "GenericParam", "TypeName"} <= types

    # Struct fields are surfaced even though the parser stores them as tuples.
    struct = next(child for child in tree["children"] if child["type"] == "StructDecl")
    field_types = {child["type"] for child in struct["children"]}
    assert "StructField" in field_types


def test_program_to_source_round_trips_monomorphized_ast():
    program = monomorphize(parse(lex(GENERIC_SOURCE)))
    source = program_to_source(program)
    assert "Box_i64" in source
    assert "identity_i64" in source
    # The concrete source is valid Polyglot again.
    reparsed = monomorphize(parse(lex(source)))
    names = {item.name for item in reparsed.items if hasattr(item, "name")}
    assert {"Box_i64", "Box_String", "identity_i64", "main"} <= names


def test_analyze_reports_syntax_error_with_span():
    source = "fn main( -> i64 { 1 }\n"
    result = analyze(source)

    assert result["ok"] is False
    assert result["original"] is None
    diagnostic = result["diagnostics"][0]
    assert diagnostic["phase"] == "parse"
    assert diagnostic["code"] == "SYNTAX"
    assert diagnostic["line"] == 1
    assert diagnostic["span"] is not None
    start, end = diagnostic["span"]
    assert 0 <= start < end <= len(source)
    assert diagnostic["message"]


def test_analyze_reports_infinite_generic_expansion():
    source = """
fn grow<T>() -> i64 { grow::<Vec<T>>() }
fn main() -> i64 { grow::<i64>() }
"""
    result = analyze(source)

    assert result["ok"] is False
    diagnostic = result["diagnostics"][0]
    assert diagnostic["phase"] == "monomorphize"
    assert diagnostic["code"] == "E003"
    assert "recursion limit" in diagnostic["message"]


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            TypeErrorForBoundary(
                "usize",
                "i64",
                1,
                1,
                "let x: i64 = n as usize;",
                (0, 3),
            ),
            "E001",
        ),
        (TerminationError("f", "reason", "help", 1, 1, (0, 3)), "E002"),
        (TranslateError("unsupported", 1, 1, (0, 3)), "E003"),
        (
            RustCompileError(
                message="rust failed",
                source_line=1,
                source_col=1,
                span=(0, 3),
            ),
            "E004",
        ),
        (IncludeCycleError("cycle", "r", "h", "n", 1, 1, span=(0, 3)), "E005"),
        (IncludeNotFoundError("missing", "r", "h", "n", 1, 1, span=(0, 3)), "E006"),
        (UnsupportedNativePatternError("pattern", 1, 1, (0, 3)), "E007"),
        (InvalidIntegerParseLiteralError("5", 1, 1, (0, 3)), "E008"),
    ],
)
def test_diagnostic_supports_e001_through_e008(error, expected_code):
    source = "fn main() { 1 }\n"
    payload = diagnostic(error, source, phase="test")

    assert payload["code"] == expected_code
    assert payload["phase"] == "test"
    assert payload["line"] == 1
    assert payload["span"] == [0, 3]
    assert payload["detail"]
    assert payload["fix"] is None or (
        isinstance(payload["fix"], dict)
        and "message" in payload["fix"]
        and "edits" in payload["fix"]
    )


def test_diagnostic_e001_fix_suggests_explicit_cast() -> None:
    error = TypeErrorForBoundary(
        "usize", "i64", 1, 6, "let x: i64 = 1 as usize;", (16, 26)
    )
    payload = diagnostic(
        error, "fn main() { let x: i64 = 1 as usize; x }\n", phase="test"
    )

    fix = payload["fix"]
    assert "as i64" in fix["message"]
    assert fix["edits"] == [
        {
            "range": {
                "start": {"line": 0, "character": 26},
                "end": {"line": 0, "character": 26},
            },
            "newText": " as i64",
        }
    ]


def test_diagnostic_e002_fix_uses_checker_suggestion() -> None:
    error = TerminationError("f", "no proof", "add a decreasing bound", 1, 1, (0, 3))
    payload = diagnostic(error, "fn main() { 1 }\n", phase="test")

    assert payload["fix"]["message"] == "add a decreasing bound"
    assert payload["fix"]["edits"] == []


def test_diagnostic_e003_unknown_macro_suggests_supported_alternatives() -> None:
    error = TranslateError("unknown macro mystery!", 1, 1, (0, 3))
    payload = diagnostic(error, "fn main() { mystery!(); }\n", phase="test")

    assert payload["code"] == "E003"
    assert "println!" in payload["fix"]["message"]
    assert "vec!" in payload["fix"]["message"]
    assert payload["fix"]["edits"] == []


def test_diagnostic_e006_fix_uses_include_help_text() -> None:
    error = IncludeNotFoundError(
        "missing", "missing", "check the include path", "note", 1, 1, span=(0, 3)
    )
    payload = diagnostic(error, 'include!("missing.poly");\n', phase="test")

    assert payload["code"] == "E006"
    assert payload["fix"]["message"] == "check the include path"


def test_fix_hint_falls_back_to_line_col_without_span() -> None:
    class BoundaryError(Exception):
        code = "E001"
        line = 3
        col = 7
        expected_type = "u8"

        def __init__(self) -> None:
            super().__init__("error[E001]: type mismatch")

    hint = fix_hint(BoundaryError(), "")
    assert hint is not None
    assert "as u8" in hint.message
    assert hint.to_dict()["edits"][0]["range"]["start"] == {
        "line": 2,
        "character": 7,
    }


def test_span_helpers_use_exact_offsets() -> None:
    source = "abc\ndef\nghi\n"
    assert offset_to_lsp_position(source, 0) == {"line": 0, "character": 0}
    assert offset_to_lsp_position(source, 9) == {"line": 2, "character": 1}
    assert span_to_lsp_range(source, [2, 9]) == {
        "start": {"line": 0, "character": 2},
        "end": {"line": 2, "character": 1},
    }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


@contextmanager
def _serve():
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


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urlrequest.urlopen(url, timeout=30) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urlerror.HTTPError as error:
        return error.code, error.read(), error.headers.get_content_type()


def _get_json(url: str) -> tuple[int, dict[str, Any]]:
    status, body, _ = _get(url)
    return status, json.loads(body)


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urlerror.HTTPError as error:
        return error.code, json.loads(error.read())


def test_server_serves_frontend_and_health():
    with _serve() as base:
        status, body, content_type = _get(base + "/")
        assert status == 200
        assert content_type == "text/html"
        assert b"Polyglot Visualizer" in body

        status, health = _get_json(base + "/api/health")
        assert status == 200
        assert health["ok"] is True
        assert health["name"] == "polyglot-visualizer"
        assert "cargo" in health
        assert "python" in health

        status, missing = _get_json(base + "/api/nope")
        assert status == 404
        assert missing["ok"] is False


def test_server_analyze_endpoint():
    with _serve() as base:
        status, payload = _post_json(base + "/api/analyze", {"source": GENERIC_SOURCE})
        assert status == 200
        assert payload["ok"] is True
        assert payload["monomorphized"]["instantiations"]
        assert "Box_i64" in payload["monomorphized"]["source"]

        status, payload = _post_json(base + "/api/analyze", {"source": "fn main("})
        assert status == 200
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["phase"] == "parse"

        status, payload = _post_json(base + "/api/analyze", {})
        assert status == 400
        assert "source" in payload["error"]


def test_server_run_endpoint_executes_python_backend():
    source = 'fn main() -> i64 { println!("visualizer"); 7 }'
    with _serve() as base:
        status, payload = _post_json(
            base + "/api/run",
            {"source": source, "backends": ["python"], "timeout": 60},
        )
        assert status == 200
        assert payload["ok"] is True
        result = payload["results"][0]
        assert result["backend"] == "python"
        assert result["ok"] is True
        assert result["stdout"] == "visualizer\n"
        assert result["return_value"] == 7
        assert result["compile_ms"] >= 0
        assert result["peak_memory_bytes"] > 0
        assert result["diagnostics"] == []


def test_server_run_rejects_unknown_backend():
    with _serve() as base:
        status, payload = _post_json(
            base + "/api/run",
            {"source": "fn main() -> i64 { 1 }", "backends": ["brainfuck"]},
        )
        assert status == 400
        assert "unknown backend" in payload["error"]


def test_server_run_times_out_infinite_program():
    source = """
fn main() -> i64 {
    let mut n: i64 = 0;
    loop { n = n + 1; }
    n
}
"""
    result = run_backend_source(source, "python", timeout=1.0)
    assert result["timed_out"] is True
    assert result["ok"] is False
    assert "timed out" in result["stderr"]
    assert result["diagnostics"]


# ---------------------------------------------------------------------------
# CLI exec metadata
# ---------------------------------------------------------------------------


def _run_exec(tmp_path: Path, source: str, backend: str = "python"):
    result_file = tmp_path / "exec.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "api.cli",
            "exec",
            "--stdin",
            "--backend",
            backend,
            "--result-file",
            str(result_file),
        ],
        input=source,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    return process, payload


def test_cli_exec_writes_timing_and_memory_metadata(tmp_path: Path):
    process, payload = _run_exec(
        tmp_path,
        'fn main() -> i64 { println!("exec"); 41 + 1 }',
    )

    assert process.returncode == 0
    assert process.stdout == "exec\n"
    assert payload["ok"] is True
    assert payload["backend"] == "python"
    assert payload["compile_ms"] >= 0
    assert payload["run_ms"] >= 0
    assert payload["total_ms"] >= payload["compile_ms"]
    assert payload["peak_memory_bytes"] > 0
    assert payload["return_value"] == 42
    assert payload["diagnostics"] == []


def test_cli_exec_reports_coded_diagnostic_with_span(tmp_path: Path):
    source = "fn main() -> i64 {\n    let x: i64 = 1 as usize;\n    x\n}\n"
    process, payload = _run_exec(tmp_path, source)

    assert process.returncode == 1
    assert payload["ok"] is False
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "E001"
    assert diagnostic["phase"] == "python"
    assert diagnostic["line"] == 2
    start, end = diagnostic["span"]
    assert source[start:end] == "1 as usize"
    assert "error[E001]" in process.stderr


def test_cli_exec_reports_parse_error_phase(tmp_path: Path):
    process, payload = _run_exec(tmp_path, "fn main( { 1 }\n")

    assert process.returncode == 1
    assert payload["ok"] is False
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["phase"] == "parse"
    assert diagnostic["code"] == "SYNTAX"
    assert diagnostic["span"] is not None


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_visualizer_frontend_is_packaged():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "static/*.html" in package_data["api"]
    assert (ROOT / "api" / "static" / "visualizer.html").is_file()
