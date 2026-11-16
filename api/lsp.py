"""Minimal Language Server for Polyglot.

The server speaks JSON-RPC over stdio and implements the small surface needed
for editor navigation: document sync, go-to-definition, and completion.
It deliberately avoids a third-party LSP dependency so the parser and AST are
the only source of truth.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from ast_nodes import (
    ConstDecl,
    EnumDecl,
    FnDecl,
    ImplBlock,
    Program,
    StructDecl,
    TypeAlias,
)
from lexer import lex
from parser import parse_recovering

COMPLETION_FUNCTION = 3
COMPLETION_CLASS = 7
COMPLETION_ENUM = 13
COMPLETION_CONSTANT = 14
COMPLETION_STRUCT = 22
COMPLETION_TYPE_PARAMETER = 25


class LspError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ServerState:
    documents: dict[str, str] = field(default_factory=dict)
    shutting_down: bool = False
    exited: bool = False


def _parse_document(text: str) -> Program:
    program, _errors = parse_recovering(lex(text))
    return program


def _word_at(text: str, position: dict[str, Any]) -> str | None:
    line = int(position.get("line", 0))
    character = int(position.get("character", 0))
    lines = text.split("\n")
    if line < 0 or line >= len(lines):
        return None
    line_text = lines[line]
    offset = sum(len(part) + 1 for part in lines[:line]) + min(
        character, len(line_text)
    )

    def is_word_char(index: int) -> bool:
        return 0 <= index < len(text) and (text[index].isalnum() or text[index] == "_")

    if not is_word_char(offset) and is_word_char(offset - 1):
        offset -= 1
    if not is_word_char(offset):
        return None
    start = offset
    end = offset + 1
    while is_word_char(start - 1):
        start -= 1
    while is_word_char(end):
        end += 1
    return text[start:end]


def _item_symbols(program: Program) -> list[tuple[str, int, Any]]:
    symbols: list[tuple[str, int, Any]] = []
    for item in program.items:
        if isinstance(item, FnDecl):
            symbols.append((item.name, COMPLETION_FUNCTION, item))
        elif isinstance(item, StructDecl):
            symbols.append((item.name, COMPLETION_STRUCT, item))
        elif isinstance(item, EnumDecl):
            symbols.append((item.name, COMPLETION_ENUM, item))
        elif isinstance(item, ConstDecl):
            symbols.append((item.name, COMPLETION_CONSTANT, item))
        elif isinstance(item, TypeAlias):
            symbols.append((item.name, COMPLETION_TYPE_PARAMETER, item))
        elif isinstance(item, ImplBlock):
            for member in item.items:
                if isinstance(member, FnDecl):
                    symbols.append((member.name, COMPLETION_FUNCTION, member))
    return symbols


def _location(uri: str, name: str, node: Any) -> dict[str, Any]:
    line = max(0, int(getattr(node, "line", 1)) - 1)
    character = max(0, int(getattr(node, "col", 1)) - 1)
    return {
        "uri": uri,
        "range": {
            "start": {"line": line, "character": character},
            "end": {"line": line, "character": character + len(name)},
        },
    }


def _text_document_uri(params: dict[str, Any]) -> str:
    return str(params["textDocument"]["uri"])


def handle_request(
    state: ServerState,
    method: str,
    params: dict[str, Any] | None,
) -> Any:
    params = params or {}

    if method == "initialize":
        return {
            "capabilities": {
                "textDocumentSync": 1,
                "definitionProvider": True,
                "completionProvider": {"triggerCharacters": [".", ":"]},
            },
            "serverInfo": {"name": "polyglot-lsp", "version": "0.1.0"},
        }
    if method in {"initialized", "workspace/didChangeConfiguration"}:
        return None
    if method == "shutdown":
        state.shutting_down = True
        return None
    if method == "exit":
        state.exited = True
        return None

    if method == "textDocument/didOpen":
        document = params["textDocument"]
        state.documents[str(document["uri"])] = str(document.get("text", ""))
        return None
    if method == "textDocument/didChange":
        uri = _text_document_uri(params)
        text = state.documents.get(uri, "")
        for change in params.get("contentChanges", []):
            if "range" not in change:
                text = str(change.get("text", ""))
            else:
                text = _apply_incremental_change(text, change)
        state.documents[uri] = text
        return None
    if method == "textDocument/didClose":
        state.documents.pop(_text_document_uri(params), None)
        return None

    if method == "textDocument/definition":
        uri = _text_document_uri(params)
        text = state.documents.get(uri, "")
        word = _word_at(text, params.get("position", {}))
        if word is None:
            return None
        for name, _kind, node in _item_symbols(_parse_document(text)):
            if name == word:
                return _location(uri, name, node)
        return None

    if method == "textDocument/completion":
        uri = _text_document_uri(params)
        text = state.documents.get(uri, "")
        seen: set[str] = set()
        items = []
        for name, kind, _node in _item_symbols(_parse_document(text)):
            if name in seen:
                continue
            seen.add(name)
            items.append({"label": name, "kind": kind})
        return items

    raise LspError(-32601, f"method not found: {method}")


def _position_to_offset(text: str, position: dict[str, Any]) -> int:
    line = int(position.get("line", 0))
    character = int(position.get("character", 0))
    lines = text.split("\n")
    return sum(len(part) + 1 for part in lines[:line]) + min(
        character, len(lines[line]) if line < len(lines) else 0
    )


def _apply_incremental_change(text: str, change: dict[str, Any]) -> str:
    range_info = change["range"]
    start = _position_to_offset(text, range_info["start"])
    end = _position_to_offset(text, range_info["end"])
    return text[:start] + str(change.get("text", "")) + text[end:]


def read_message(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line in {"\r\n", "\n"}:
            break
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stream.read(length)
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    stream.write(data)
    stream.flush()


def serve(stdin: BinaryIO, stdout: BinaryIO) -> None:
    state = ServerState()
    while not state.exited:
        message = read_message(stdin)
        if message is None:
            break
        method = message.get("method")
        if not isinstance(method, str):
            continue
        request_id = message.get("id")
        try:
            result = handle_request(state, method, message.get("params"))
        except LspError as error:
            if request_id is not None:
                write_message(
                    stdout,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": error.code, "message": error.message},
                    },
                )
            continue
        if request_id is not None:
            write_message(
                stdout,
                {"jsonrpc": "2.0", "id": request_id, "result": result},
            )


def main() -> int:
    serve(sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
