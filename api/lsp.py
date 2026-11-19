"""Minimal Language Server for Polyglot.

The server speaks JSON-RPC over stdio and implements the small surface needed
for editor navigation: document sync, go-to-definition, and completion.  It
also publishes structured E001-E008 diagnostics on open/change/save, using the
same front-end validation as ``polyglot check``.  It deliberately avoids a
third-party LSP dependency so the parser and AST are the only source of truth.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from api.visualizer import diagnostic, span_to_lsp_range
from ast_nodes import (
    ConstDecl,
    EnumDecl,
    FnDecl,
    ImplBlock,
    Program,
    StructDecl,
    TypeAlias,
)
from include_expander import (
    IncludeCycleError,
    IncludeNotFoundError,
    expand_includes,
)
from lexer import LexError, lex
from parser import ParseError, parse, parse_recovering

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
    notifications: list[dict[str, Any]] = field(default_factory=list)
    shutting_down: bool = False
    exited: bool = False


def drain_notifications(state: ServerState) -> list[dict[str, Any]]:
    """Return and clear queued server-initiated notifications."""
    pending = list(state.notifications)
    state.notifications.clear()
    return pending


def _uri_to_path(uri: str) -> str:
    """Best-effort conversion of a ``file:`` document URI to a local path."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return url2pathname(unquote(parsed.path))
    return uri


def _validate_document(text: str, uri: str | None = None) -> list[dict[str, Any]]:
    """Run the front-end checks used by ``polyglot check`` on document text.

    Parsing is strict (as in ``load_program``), includes are resolved relative
    to the document URI when possible, and the Python translator performs the
    monomorphization/termination checks.  Only the public E001-E008 codes (plus
    lexer/parser ``SYNTAX``) are surfaced; unrelated internal exceptions are
    ignored rather than mislabelled as language diagnostics.
    """
    try:
        program = parse(lex(text))
    except (LexError, ParseError) as error:
        return [diagnostic(error, text, phase="parse")]

    if uri:
        try:
            program = expand_includes(program, _uri_to_path(uri))
        except (IncludeCycleError, IncludeNotFoundError) as error:
            return [diagnostic(error, text, phase="check")]

    try:
        from backends.python.translator import Translator

        Translator().translate_program(program)
    except Exception as error:
        code = getattr(error, "code", None)
        if code in {"E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008"}:
            return [diagnostic(error, text, phase="check")]
    return []


# All public codes are hard errors in `polyglot check`; the explicit mapping
# keeps LSP severity deterministic and identical to the CLI's interpretation.
_SEVERITY_BY_CODE: dict[str, int] = {
    "SYNTAX": 1,
    **{f"E{index:03d}": 1 for index in range(1, 9)},
}


def _severity(code: str | None) -> int:
    """Map a public diagnostic code to an LSP DiagnosticSeverity (1 = Error)."""
    return _SEVERITY_BY_CODE.get(code or "", 1)


def _lsp_diagnostic(text: str, item: dict[str, Any]) -> dict[str, Any]:
    """Convert one structured diagnostic into an LSP ``Diagnostic``."""
    code = item.get("code")
    return {
        "range": span_to_lsp_range(
            text,
            item.get("span"),
            line=item.get("line"),
            col=item.get("col"),
        ),
        "severity": _severity(code),
        "source": "polyglot",
        "message": str(item.get("message") or ""),
        "code": code,
        "data": {"code": code, "fix": item.get("fix")},
    }


def _publish_diagnostics(uri: str, text: str) -> dict[str, Any]:
    """Build a ``textDocument/publishDiagnostics`` notification."""
    diagnostics = [
        _lsp_diagnostic(text, item) for item in _validate_document(text, uri)
    ]
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": uri, "diagnostics": diagnostics},
    }


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
        # LSP 3.17 has no `publishDiagnosticsProvider` capability: publishing
        # diagnostics is an unsolicited server -> client notification.  Keep
        # the numeric (full-document) sync form and simply send notifications
        # on didOpen/didChange/didSave.
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
        uri = str(document["uri"])
        text = str(document.get("text", ""))
        state.documents[uri] = text
        state.notifications.append(_publish_diagnostics(uri, text))
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
        state.notifications.append(_publish_diagnostics(uri, text))
        return None
    if method == "textDocument/didSave":
        uri = _text_document_uri(params)
        text = params.get("text")
        if text is None:
            text = state.documents.get(uri, "")
        else:
            text = str(text)
            state.documents[uri] = text
        state.notifications.append(_publish_diagnostics(uri, text))
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
        for notification in drain_notifications(state):
            write_message(stdout, notification)
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
