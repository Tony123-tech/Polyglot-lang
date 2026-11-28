"""Language server tests for navigation and JSON-RPC framing."""

from __future__ import annotations

import io
import json
from typing import Any

from api.lsp import (
    ServerState,
    _lsp_diagnostic,
    _position_to_offset,
    drain_notifications,
    handle_request,
    read_message,
    serve,
    write_message,
)

SOURCE = """\
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(1, 2) }
"""

E001_SOURCE = """\
fn main() -> i64 {
    let x: i64 = 1 as usize;
    x
}
"""


def _open(text: str = SOURCE) -> ServerState:
    state = ServerState()
    handle_request(
        state,
        "textDocument/didOpen",
        {"textDocument": {"uri": "file:///main.poly", "text": text}},
    )
    return state


def test_definition_finds_function() -> None:
    state = _open()
    result = handle_request(
        state,
        "textDocument/definition",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "position": {"line": 1, "character": 20},
        },
    )

    assert result == {
        "uri": "file:///main.poly",
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 3},
        },
    }


def test_definition_handles_broken_neighbour() -> None:
    text = """\
fn broken( { }
fn target() -> i64 { 1 }
fn main() -> i64 { target() }
"""
    state = _open(text)
    result = handle_request(
        state,
        "textDocument/definition",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "position": {"line": 2, "character": 20},
        },
    )

    assert result is not None
    assert result["range"]["start"] == {"line": 1, "character": 0}


def test_completion_lists_top_level_symbols() -> None:
    text = """\
struct Point { x: i64 }
enum Color { Red, Green }
const LIMIT: i64 = 4;
fn add(a: i64, b: i64) -> i64 { a + b }
"""
    state = _open(text)
    result = handle_request(
        state,
        "textDocument/completion",
        {"textDocument": {"uri": "file:///main.poly"}},
    )

    labels = {item["label"] for item in result}
    assert {"Point", "Color", "LIMIT", "add"} <= labels


def test_full_document_change_replaces_text() -> None:
    state = _open("fn old() -> i64 { 1 }\nfn use_old() -> i64 { old() }\n")
    handle_request(
        state,
        "textDocument/didChange",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "contentChanges": [{"text": "fn fresh() -> i64 { 2 }\n"}],
        },
    )
    result = handle_request(
        state,
        "textDocument/completion",
        {"textDocument": {"uri": "file:///main.poly"}},
    )

    assert [item["label"] for item in result] == ["fresh"]


def test_incremental_change_replaces_range() -> None:
    state = _open("fn old() -> i64 { 1 }\n")
    handle_request(
        state,
        "textDocument/didChange",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "contentChanges": [
                {
                    "range": {
                        "start": {"line": 0, "character": 3},
                        "end": {"line": 0, "character": 6},
                    },
                    "text": "fresh",
                }
            ],
        },
    )
    result = handle_request(
        state,
        "textDocument/completion",
        {"textDocument": {"uri": "file:///main.poly"}},
    )

    assert [item["label"] for item in result] == ["fresh"]


def test_json_rpc_round_trip() -> None:
    stdin = io.BytesIO()
    write_message(
        stdin, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    write_message(
        stdin, {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}}
    )
    write_message(stdin, {"jsonrpc": "2.0", "method": "exit", "params": {}})
    stdin.seek(0)
    stdout = io.BytesIO()

    serve(stdin, stdout)

    stdout.seek(0)
    first = read_message(stdout)
    second = read_message(stdout)
    assert first is not None and first["id"] == 1
    assert first["result"]["capabilities"]["definitionProvider"] is True
    assert second == {"jsonrpc": "2.0", "id": 2, "result": None}


def test_unknown_request_returns_error() -> None:
    stdin = io.BytesIO()
    write_message(
        stdin,
        {"jsonrpc": "2.0", "id": 9, "method": "workspace/unknown", "params": {}},
    )
    write_message(stdin, {"jsonrpc": "2.0", "method": "exit", "params": {}})
    stdin.seek(0)
    stdout = io.BytesIO()

    serve(stdin, stdout)

    stdout.seek(0)
    response = read_message(stdout)
    assert response is not None
    assert response["error"]["code"] == -32601


def test_cli_exposes_lsp_command() -> None:
    from api.cli import build_parser, cmd_lsp

    args = build_parser().parse_args(["lsp"])
    assert args.func is cmd_lsp


def test_write_message_sets_content_length() -> None:
    stream = io.BytesIO()
    write_message(stream, {"jsonrpc": "2.0", "id": 1, "result": "ok"})
    payload = stream.getvalue()
    header, _, body = payload.partition(b"\r\n\r\n")
    assert header == f"Content-Length: {len(body)}".encode()
    assert json.loads(body)["result"] == "ok"


# ---------------------------------------------------------------------------
# Diagnostics publication
# ---------------------------------------------------------------------------


def _one_notification(state: ServerState) -> dict[str, Any]:
    pending = drain_notifications(state)
    assert len(pending) == 1
    return pending[0]


def test_initialize_does_not_advertise_publish_diagnostics_provider() -> None:
    result = handle_request(ServerState(), "initialize", {})
    capabilities = result["capabilities"]

    # LSP 3.17 has no publishDiagnosticsProvider capability; diagnostics are
    # unsolicited notifications sent on didOpen/didChange/didSave.
    assert "publishDiagnosticsProvider" not in capabilities
    assert capabilities["textDocumentSync"] == 1
    assert capabilities["definitionProvider"] is True


def test_did_open_publishes_exact_span_range_and_fix_data() -> None:
    state = ServerState()
    handle_request(
        state,
        "textDocument/didOpen",
        {"textDocument": {"uri": "file:///main.poly", "text": E001_SOURCE}},
    )
    notification = _one_notification(state)

    assert notification["jsonrpc"] == "2.0"
    assert notification["method"] == "textDocument/publishDiagnostics"
    assert notification["params"]["uri"] == "file:///main.poly"
    items = notification["params"]["diagnostics"]
    assert len(items) == 1
    item = items[0]

    token = "1 as usize"
    offset = E001_SOURCE.index(token)
    assert item["range"] == {
        "start": {"line": 1, "character": offset - 19},
        "end": {"line": 1, "character": offset - 19 + len(token)},
    }
    assert item["range"]["start"] == {"line": 1, "character": 17}
    assert item["range"]["end"] == {"line": 1, "character": 27}
    assert item["severity"] == 1
    assert item["source"] == "polyglot"
    assert item["code"] == "E001"
    assert item["data"]["code"] == "E001"
    fix = item["data"]["fix"]
    assert "as i64" in fix["message"]
    assert fix["edits"] == [
        {
            "range": {
                "start": {"line": 1, "character": 27},
                "end": {"line": 1, "character": 27},
            },
            "newText": " as i64",
        }
    ]


def test_did_change_publishes_diagnostics_for_new_text() -> None:
    state = _open()
    assert _one_notification(state)["params"]["diagnostics"] == []

    handle_request(
        state,
        "textDocument/didChange",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "contentChanges": [{"text": E001_SOURCE}],
        },
    )
    notification = _one_notification(state)
    assert [item["code"] for item in notification["params"]["diagnostics"]] == ["E001"]


def test_did_save_publishes_diagnostics() -> None:
    state = _open()
    drain_notifications(state)

    handle_request(
        state,
        "textDocument/didSave",
        {"textDocument": {"uri": "file:///main.poly"}},
    )
    assert _one_notification(state)["params"]["diagnostics"] == []

    handle_request(
        state,
        "textDocument/didChange",
        {
            "textDocument": {"uri": "file:///main.poly"},
            "contentChanges": [{"text": E001_SOURCE}],
        },
    )
    drain_notifications(state)

    handle_request(
        state,
        "textDocument/didSave",
        {"textDocument": {"uri": "file:///main.poly"}},
    )
    items = _one_notification(state)["params"]["diagnostics"]
    assert [item["code"] for item in items] == ["E001"]


def test_clean_document_publishes_empty_diagnostics() -> None:
    state = ServerState()
    handle_request(
        state,
        "textDocument/didOpen",
        {"textDocument": {"uri": "file:///clean.poly", "text": SOURCE}},
    )
    assert _one_notification(state)["params"]["diagnostics"] == []


def test_unknown_macro_publishes_supported_alternative_hint() -> None:
    state = ServerState()
    text = "fn main() { unknown_macro!(x); }\n"
    handle_request(
        state,
        "textDocument/didOpen",
        {"textDocument": {"uri": "file:///macro.poly", "text": text}},
    )
    item = _one_notification(state)["params"]["diagnostics"][0]
    assert item["code"] == "E003"
    fix = item["data"]["fix"]
    assert "println!" in fix["message"]
    assert fix["edits"] == []


def test_lsp_diagnostic_converts_multiline_span_and_line_col_fallback() -> None:
    text = "abc\ndef\nghi\n"
    item = _lsp_diagnostic(
        text,
        {
            "code": "E003",
            "message": "boom",
            "span": [2, 9],
            "line": 1,
            "col": 3,
            "fix": None,
        },
    )
    assert item["range"] == {
        "start": {"line": 0, "character": 2},
        "end": {"line": 2, "character": 1},
    }

    fallback = _lsp_diagnostic(
        text,
        {"code": "E001", "message": "boom", "span": None, "line": 2, "col": 4},
    )
    assert fallback["range"] == {
        "start": {"line": 1, "character": 3},
        "end": {"line": 1, "character": 4},
    }


def test_lsp_positions_and_incoming_offsets_are_utf16() -> None:
    text = "🙂 abcdef"
    item = _lsp_diagnostic(
        text,
        {"code": "E003", "message": "boom", "span": [2, 5], "line": 1, "col": 3},
    )

    # The emoji occupies two UTF-16 code units but one Python character.
    assert item["range"] == {
        "start": {"line": 0, "character": 3},
        "end": {"line": 0, "character": 6},
    }
    assert _position_to_offset(text, {"line": 0, "character": 3}) == 2


def test_did_open_publishes_every_recovered_syntax_error() -> None:
    state = ServerState()
    text = "fn a() { let = 1; }\nfn b() { let = 2; }\n"
    handle_request(
        state,
        "textDocument/didOpen",
        {"textDocument": {"uri": "file:///many.poly", "text": text}},
    )

    items = _one_notification(state)["params"]["diagnostics"]
    assert len(items) == 2
    assert [item["code"] for item in items] == ["SYNTAX", "SYNTAX"]
    assert [item["range"]["start"]["line"] for item in items] == [0, 1]


def test_serve_flushes_publish_diagnostics_notification() -> None:
    stdin = io.BytesIO()
    write_message(
        stdin,
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": "file:///main.poly", "text": SOURCE}},
        },
    )
    write_message(stdin, {"jsonrpc": "2.0", "method": "exit", "params": {}})
    stdin.seek(0)
    stdout = io.BytesIO()

    serve(stdin, stdout)

    stdout.seek(0)
    first = read_message(stdout)
    assert first is not None
    assert first["method"] == "textDocument/publishDiagnostics"
    assert first["params"]["diagnostics"] == []
