"""Language server tests for navigation and JSON-RPC framing."""

from __future__ import annotations

import io
import json

from api.lsp import ServerState, handle_request, read_message, serve, write_message

SOURCE = """\
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(1, 2) }
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
    write_message(stdin, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    write_message(stdin, {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
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
