"""FRAME 1: lexing, parsing and exact source spans.

The ten functions in this module form the first frame of the framework
registry (see :mod:`framework_functions`).  They are deliberately pure and
dependency-free beyond the compiler's own ``lexer``, ``parser``, ``ast_nodes``
and ``api.visualizer`` modules.

Span conventions
----------------
* ``char_span`` is a half-open ``[start, end)`` pair of Python string indices.
* ``byte_span`` is the corresponding pair of UTF-8 byte offsets, derived by
  encoding ``source[:offset]``.
* ``line_col_span`` is ``[[start_line, start_col], [end_line, end_col]]`` with
  one-based lines and one-based character columns.

Diagnostic conventions
----------------------
Every diagnostic is a plain dictionary with ``code``, ``message``, ``line``,
``col``, ``char_span``, ``byte_span``, ``line_col_span`` and ``kind`` keys.
Lexical failures use ``E001``; syntax failures use ``SYNTAX``.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any, Final, NoReturn

import compiler.core.ast_nodes as ast
from api.visualizer import SourcePrinter, program_to_source
from compiler.core.lexer import SIMPLE_ESCAPES, Lexer, Token, TokenType, lex
from compiler.core.parser import ParseError, Parser

__all__ = [
    "ast_node_span_normalize",
    "ast_pretty_print_canonical",
    "lex_raw_and_interpolated_strings",
    "lex_recover_unterminated_token",
    "lex_source_with_spans",
    "parse_error_sync_recovery",
    "parse_generic_type_params",
    "parse_macro_invocations",
    "parse_pattern_match_arms",
    "parse_pratt_expr_with_precedence",
]

_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_EXPR_STOP: Final[frozenset[TokenType]] = frozenset(
    {
        TokenType.EOF,
        TokenType.SEMI,
        TokenType.COMMA,
        TokenType.RPAREN,
        TokenType.RBRACKET,
        TokenType.RBRACE,
        TokenType.COLON,
        TokenType.FATARROW,
    }
)
_OPENERS: Final[dict[TokenType, TokenType]] = {
    TokenType.LPAREN: TokenType.RPAREN,
    TokenType.LBRACKET: TokenType.RBRACKET,
    TokenType.LBRACE: TokenType.RBRACE,
}
_CLOSERS: Final[frozenset[TokenType]] = frozenset(_OPENERS.values())
_DELIMITER_TEXT: Final[dict[TokenType, str]] = {
    TokenType.LPAREN: "()",
    TokenType.LBRACKET: "[]",
    TokenType.LBRACE: "{}",
}

# Explicit Pratt precedence table.  Higher numbers bind tighter.
PRECEDENCE_TABLE: Final[dict[str, int]] = {
    "=": 1,
    "+=": 1,
    "-=": 1,
    "*=": 1,
    "/=": 1,
    "+|=": 1,
    "-|=": 1,
    "*|=": 1,
    "..": 2,
    "..=": 2,
    "||": 3,
    "&&": 4,
    "==": 5,
    "!=": 5,
    "<": 6,
    ">": 6,
    "<=": 6,
    ">=": 6,
    "|": 7,
    "^": 8,
    "&": 9,
    "+": 10,
    "-": 10,
    "+|": 10,
    "-|": 10,
    "*": 11,
    "/": 11,
    "%": 11,
    "*|": 11,
    "as": 12,
    "unary": 13,
    "postfix": 14,
    "primary": 15,
}
_RIGHT_ASSOCIATIVE: Final[frozenset[str]] = frozenset(
    {"=", "+=", "-=", "*=", "/=", "+|=", "-|=", "*|=", "..", "..="}
)
_INFIX_OPS: Final[dict[TokenType, str]] = {
    TokenType.EQ: "=",
    TokenType.PLUSEQ: "+=",
    TokenType.MINUSEQ: "-=",
    TokenType.STAREQ: "*=",
    TokenType.SLASHEQ: "/=",
    TokenType.PLUS_PIPE_EQ: "+|=",
    TokenType.MINUS_PIPE_EQ: "-|=",
    TokenType.STAR_PIPE_EQ: "*|=",
    TokenType.DOTDOT: "..",
    TokenType.DOTDOTEQ: "..=",
    TokenType.OROR: "||",
    TokenType.ANDAND: "&&",
    TokenType.EQEQ: "==",
    TokenType.NEQ: "!=",
    TokenType.LT: "<",
    TokenType.GT: ">",
    TokenType.LE: "<=",
    TokenType.GE: ">=",
    TokenType.PIPE: "|",
    TokenType.CARET: "^",
    TokenType.AMP: "&",
    TokenType.PLUS: "+",
    TokenType.MINUS: "-",
    TokenType.PLUS_PIPE: "+|",
    TokenType.MINUS_PIPE: "-|",
    TokenType.STAR: "*",
    TokenType.SLASH: "/",
    TokenType.PERCENT: "%",
    TokenType.STAR_PIPE: "*|",
    TokenType.AS: "as",
}
_PREFIX_OPS: Final[dict[TokenType, str]] = {
    TokenType.PLUS: "+",
    TokenType.MINUS: "-",
    TokenType.BANG: "!",
    TokenType.STAR: "*",
    TokenType.AMP: "&",
}
_ITEM_START: Final[frozenset[TokenType]] = frozenset(
    {
        TokenType.FN,
        TokenType.STRUCT,
        TokenType.ENUM,
        TokenType.IMPL,
        TokenType.TRAIT,
        TokenType.USE,
        TokenType.MOD,
        TokenType.CONST,
        TokenType.STATIC,
        TokenType.TYPE,
        TokenType.PUB,
        TokenType.ASYNC,
        TokenType.UNSAFE,
        TokenType.HASH,
        TokenType.LET,
        TokenType.RETURN,
        TokenType.IF,
        TokenType.LOOP,
        TokenType.WHILE,
        TokenType.FOR,
        TokenType.MATCH,
        TokenType.BREAK,
        TokenType.CONTINUE,
    }
)


# ---------------------------------------------------------------------------
# Shared span / token helpers
# ---------------------------------------------------------------------------


def _offset_to_line_col(source: str, offset: int) -> tuple[int, int]:
    """Convert a character offset to one-based (line, column)."""
    offset = max(0, min(offset, len(source)))
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    return line, offset - last_newline


def _line_col_to_offset(source: str, line: int, col: int) -> int:
    """Convert one-based (line, column) to a character offset, clamped."""
    if line < 1 or col < 1:
        return 0
    lines = source.split("\n")
    if line > len(lines):
        return len(source)
    base = sum(len(part) + 1 for part in lines[: line - 1])
    return min(len(source), base + col - 1)


def _char_to_byte(source: str, offset: int) -> int:
    """Byte offset of a character offset under UTF-8 encoding."""
    return len(source[: max(0, min(offset, len(source)))].encode("utf-8"))


def _raw_string_prefix(source: str, start: int) -> tuple[int, int] | None:
    """Return ``(quote_index, hash_count)`` when a raw string starts here."""
    if source[start : start + 1] != "r":
        return None
    index = start + 1
    while index < len(source) and source[index] == "#":
        index += 1
    if source[index : index + 1] == '"':
        return index, index - start - 1
    return None


def _string_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Return ``(end_offset, terminated)`` for a string beginning at *start*."""
    raw = _raw_string_prefix(source, start)
    if raw is not None:
        quote, hashes = raw
        closing = '"' + "#" * hashes
        close = source.find(closing, quote + 1)
        if close < 0:
            return len(source), False
        return close + len(closing), True

    if source.startswith('"""', start):
        delimiter = '"""'
        index = start + 3
        while index < len(source):
            if source.startswith(delimiter, index):
                return index + 3, True
            if source[index] == "\\":
                index += 2
            else:
                index += 1
        return len(source), False

    if source[start : start + 1] != '"':
        return start, False
    index = start + 1
    while index < len(source):
        if source[index] == '"':
            return index + 1, True
        if source[index] == "\\":
            index += 2
        else:
            index += 1
    return len(source), False


def _char_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Return ``(end_offset, terminated)`` for a char literal."""
    if source[start : start + 1] != "'":
        return start, False
    index = start + 1
    while index < len(source):
        if source[index] == "'":
            return index + 1, True
        if source[index] == "\\":
            index += 2
        else:
            index += 1
    return len(source), False


def _block_comment_end(source: str, start: int) -> int | None:
    """Index just past a (possibly nested) block comment, or ``None``."""
    if not source.startswith("/*", start):
        return None
    index = start + 2
    depth = 1
    while index < len(source):
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    return None


def _token_end_char_offset(source: str, token: Token) -> int:
    """Exact exclusive character offset of a lexer token's raw lexeme."""
    if token.type == TokenType.EOF:
        return len(source)
    if token.type == TokenType.STRING:
        return _string_literal_end(source, token.pos)[0]
    if token.type == TokenType.CHAR:
        return _char_literal_end(source, token.pos)[0]
    end = token.pos + len(token.value)
    return max(token.pos, min(end, len(source)))


def _make_record(
    source: str,
    type_name: str,
    value: str,
    line: int,
    col: int,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Build the common structured token/diagnostic span record."""
    start = max(0, min(start, len(source)))
    end = max(start, min(end, len(source)))
    end_line, end_col = _offset_to_line_col(source, end)
    return {
        "type": type_name,
        "value": value,
        "text": source[start:end],
        "line": line,
        "col": col,
        "char_span": [start, end],
        "byte_span": [_char_to_byte(source, start), _char_to_byte(source, end)],
        "line_col_span": [[line, col], [end_line, end_col]],
    }


def _token_record(source: str, token: Token) -> dict[str, Any]:
    """Convert a :class:`lexer.Token` into a fully spanned record."""
    end = _token_end_char_offset(source, token)
    return _make_record(
        source, token.type.name, token.value, token.line, token.col, token.pos, end
    )


def _diagnostic(
    code: str,
    message: str,
    source: str,
    start: int,
    end: int | None = None,
    *,
    kind: str = "TOKEN",
) -> dict[str, Any]:
    """Build a structured diagnostic with character, byte and line/column spans."""
    start = max(0, min(start, len(source)))
    stop = len(source) if end is None else max(start, min(end, len(source)))
    record = _make_record(
        source, code, message, *_offset_to_line_col(source, start), start, stop
    )
    record.pop("text")
    record.pop("type")
    record["code"] = code
    record["message"] = message
    record["kind"] = kind
    return record


def _decode_escape(source: str, index: int) -> tuple[str, int, str | None]:
    """Decode one backslash escape; returns ``(value, consumed, problem)``."""
    if index + 1 >= len(source):
        return "", 1, "unterminated escape"
    ch = source[index + 1]
    if ch in SIMPLE_ESCAPES:
        return SIMPLE_ESCAPES[ch], 2, None
    if ch == "\n":
        return "", 2, None

    digits = ""
    consumed = 0
    if ch == "x":
        digits = source[index + 2 : index + 4]
        consumed = 4
    elif ch == "u" and source[index + 2 : index + 3] == "{":
        close = source.find("}", index + 3)
        if close >= 0:
            digits = source[index + 3 : close]
            consumed = close - index + 1
    elif ch == "u":
        digits = source[index + 2 : index + 6]
        consumed = 6
    if consumed:
        if digits and all(digit in _HEX_DIGITS for digit in digits):
            codepoint = int(digits, 16)
            if codepoint <= 0x10FFFF:
                return chr(codepoint), consumed, None
        return ch, 2, f"invalid {'unicode' if ch == 'u' else 'hex'} escape"
    return ch, 2, f"unknown escape \\{ch}"


def _wordish(text: str) -> bool:
    return bool(text) and (text[0].isalnum() or text[0] in "_'")


def _tokens_text(tokens: list[Token]) -> str:
    """Join token values with minimal, deterministic spacing."""
    parts: list[str] = []
    previous = ""
    for token in tokens:
        if token.type == TokenType.EOF:
            break
        if parts and _wordish(previous) and _wordish(token.value):
            parts.append(" ")
        parts.append(token.value)
        previous = token.value
    return "".join(parts)


def _as_tokens(source: str | list[Token]) -> list[Token]:
    """Accept source text or tokens and return an independent token list."""
    tokens = lex(source) if isinstance(source, str) else list(source)
    if not tokens or tokens[-1].type != TokenType.EOF:
        last = tokens[-1] if tokens else Token(TokenType.EOF, "", 1, 1, 0)
        end = last.pos + len(last.value)
        tokens = [*tokens, Token(TokenType.EOF, "", last.line, last.col, end)]
    return tokens


def _is_ast_node(value: object) -> bool:
    return is_dataclass(value) and hasattr(value, "line") and hasattr(value, "col")


def _ast_children(value: object):
    """Yield every AST node nested anywhere inside *value*."""
    if _is_ast_node(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _ast_children(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _ast_children(item)


def _direct_ast_children(node: object):
    """Yield the immediate AST children of *node* (never the node itself)."""
    for field in fields(node):
        if field.name in {"line", "col", "span", "source_lines"}:
            continue
        yield from _ast_children(getattr(node, field.name))


def _ast_to_dict(value: object) -> Any:
    """Convert an AST node (or nested container) into a JSON-like dictionary."""
    if _is_ast_node(value):
        result: dict[str, Any] = {"kind": type(value).__name__}
        for field in fields(value):
            if field.name in {"line", "col", "span", "source_lines"}:
                continue
            result[field.name] = _ast_to_dict(getattr(value, field.name))
        result["line"] = value.line
        result["col"] = value.col
        if getattr(value, "span", None) is not None:
            result["char_span"] = list(value.span)
        return result
    if isinstance(value, (list, tuple)):
        return [_ast_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _ast_to_dict(item) for key, item in value.items()}
    return value


def _matching_brace_end(source: str, start: int) -> int | None:
    """Find the offset just past the brace group at/after *start*."""
    index = source.find("{", start)
    if index < 0:
        return None
    depth = 0
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


# ---------------------------------------------------------------------------
# 1. lex_source_with_spans
# ---------------------------------------------------------------------------


def lex_source_with_spans(source: str) -> dict[str, Any]:
    """Lex *source* and report exact character, UTF-8 byte and line/column spans.

    ``Token.pos`` is a character offset in the existing lexer, so byte offsets
    are derived as ``len(source[:pos].encode("utf-8"))``.  String and char
    tokens get their raw lexeme length recovered from the source rather than
    from the decoded ``Token.value``.  The returned EOF token spans the empty
    range at the end of the input.
    """
    tokens = lex(source)
    records = [_token_record(source, token) for token in tokens]
    return {
        "source": source,
        "char_length": len(source),
        "byte_length": len(source.encode("utf-8")),
        "line_count": source.count("\n") + 1,
        "token_count": len(records),
        "tokens": records,
    }


# ---------------------------------------------------------------------------
# 2. lex_raw_and_interpolated_strings
# ---------------------------------------------------------------------------


def _flush_literal(
    segments: list[dict[str, Any]], literal: list[str], start: int, stop: int
) -> None:
    """Append the accumulated literal text as one string segment."""
    if literal:
        segments.append(
            {"kind": "literal", "value": "".join(literal), "char_span": [start, stop]}
        )
        literal.clear()


def _scan_interpolation(
    source: str, brace: int, content_end: int
) -> tuple[int, int, bool]:
    """Find the expression inside ``{...}`` honour nesting and nested strings."""
    depth = 0
    index = brace
    while index < content_end:
        ch = source[index]
        if ch == '"':
            index = max(index + 1, _string_literal_end(source, index)[0])
            continue
        if ch == "'":
            index = max(index + 1, _char_literal_end(source, index)[0])
            continue
        if ch == "\\":
            index += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return brace + 1, index, True
        index += 1
    return brace + 1, content_end, False


def lex_raw_and_interpolated_strings(source: str) -> dict[str, Any]:
    """Scan raw, escaped and brace-interpolated string literals in *source*.

    Raw strings (``r"..."`` and ``r##"..."##``) keep backslashes and braces
    verbatim.  Ordinary strings decode ``\\n``, ``\\xHH``, ``\\uHHHH`` and
    ``\\u{...}`` escapes, and every unescaped ``{expr}`` becomes an
    interpolation segment whose braces may be nested.  Problems are reported
    as structured diagnostics instead of raising.
    """
    tokens: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    index = 0
    while index < len(source):
        ch = source[index]
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            comment_end = _block_comment_end(source, index)
            if comment_end is None:
                diagnostics.append(
                    _diagnostic(
                        "E001",
                        "unterminated block comment",
                        source,
                        index,
                        len(source),
                        kind="COMMENT",
                    )
                )
                break
            index = comment_end
            continue

        raw = _raw_string_prefix(source, index)
        if raw is not None:
            quote, hashes = raw
            closing = '"' + "#" * hashes
            close = source.find(closing, quote + 1)
            terminated = close >= 0
            end = close + len(closing) if terminated else len(source)
            inner_end = close if terminated else len(source)
            value = source[quote + 1 : inner_end]
            record = _make_record(
                source, "STRING", value, *_offset_to_line_col(source, index), index, end
            )
            record.update(
                {
                    "kind": "raw",
                    "terminated": terminated,
                    "hashes": hashes,
                    "segments": [
                        {
                            "kind": "literal",
                            "value": value,
                            "char_span": [quote + 1, inner_end],
                        }
                    ],
                    "interpolations": [],
                    "escapes": [],
                }
            )
            tokens.append(record)
            if not terminated:
                diagnostics.append(
                    _diagnostic(
                        "E001",
                        "unterminated raw string",
                        source,
                        index,
                        end,
                        kind="STRING",
                    )
                )
            index = end
            continue

        if ch == "'":
            index = max(index + 1, _char_literal_end(source, index)[0])
            continue

        if ch != '"':
            index += 1
            continue

        end, terminated = _string_literal_end(source, index)
        delimiter = '"""' if source.startswith('"""', index) else '"'
        content_start = index + len(delimiter)
        inner_end = end - len(delimiter) if terminated else len(source)
        segments: list[dict[str, Any]] = []
        interpolations: list[dict[str, Any]] = []
        escapes: list[dict[str, Any]] = []
        literal: list[str] = []
        literal_start = content_start
        cursor = content_start

        while cursor < inner_end:
            current = source[cursor]
            if current == "\\":
                decoded, consumed, problem = _decode_escape(source, cursor)
                literal.append(decoded)
                escapes.append(
                    {
                        "raw": source[cursor : cursor + consumed],
                        "value": decoded,
                        "char_span": [cursor, cursor + consumed],
                    }
                )
                if problem is not None:
                    diagnostics.append(
                        _diagnostic(
                            "E001",
                            problem,
                            source,
                            cursor,
                            cursor + consumed,
                            kind="ESCAPE",
                        )
                    )
                cursor += consumed
                continue
            if current == "{":
                expr_start, expr_end, closed = _scan_interpolation(
                    source, cursor, inner_end
                )
                _flush_literal(segments, literal, literal_start, cursor)
                expression = source[expr_start:expr_end]
                segments.append(
                    {
                        "kind": "interpolation",
                        "value": expression,
                        "char_span": [cursor, expr_end + (1 if closed else 0)],
                    }
                )
                interpolations.append(
                    {
                        "expression": expression,
                        "char_span": [expr_start, expr_end],
                        "terminated": closed,
                    }
                )
                if not closed:
                    diagnostics.append(
                        _diagnostic(
                            "E001",
                            "unterminated interpolation",
                            source,
                            cursor,
                            inner_end,
                            kind="INTERPOLATION",
                        )
                    )
                    cursor = inner_end
                    literal_start = inner_end
                    continue
                cursor = expr_end + 1
                literal_start = cursor
                continue
            literal.append(current)
            cursor += 1
        _flush_literal(segments, literal, literal_start, inner_end)

        value = "".join(
            segment["value"] if segment["kind"] == "literal" else "{}"
            for segment in segments
        )
        record = _make_record(
            source, "STRING", value, *_offset_to_line_col(source, index), index, end
        )
        record.update(
            {
                "kind": "interpolated" if interpolations else "string",
                "terminated": terminated,
                "segments": segments,
                "interpolations": interpolations,
                "escapes": escapes,
            }
        )
        tokens.append(record)
        if not terminated:
            diagnostics.append(
                _diagnostic(
                    "E001", "unterminated string", source, index, end, kind="STRING"
                )
            )
        index = end

    return {
        "tokens": tokens,
        "strings": tokens,
        "diagnostics": diagnostics,
        "token_count": len(tokens),
    }


# ---------------------------------------------------------------------------
# 3. lex_recover_unterminated_token
# ---------------------------------------------------------------------------


class _RecoveryLexError(Exception):
    """Internal, catchable form of a lexer error."""

    def __init__(self, message: str, line: int, col: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


class _RecoveringLexer(Lexer):
    """Lexer whose errors are recoverable values rather than raised LexError."""

    def error(  # type: ignore[override]
        self, msg: str, line: int | None = None, col: int | None = None
    ) -> NoReturn:
        raise _RecoveryLexError(
            msg, self.line if line is None else line, self.col if col is None else col
        )


def _lex_one_token(lexer: Lexer) -> None:
    """Run exactly the dispatch portion of ``Lexer.lex`` for one token."""
    line, col, pos = lexer.line, lexer.col, lexer.i
    ch = lexer.peek()
    if ch == "r" and _raw_string_prefix(lexer.src, pos) is not None:
        lexer.lex_raw_string(line, col, pos)
    elif lexer.src.startswith('"""', pos):
        lexer.lex_multiline_string(line, col, pos)
    elif ch == '"':
        lexer.lex_string(line, col, pos)
    elif ch == "'":
        lexer.lex_quote(line, col, pos)
    elif ch.isalpha() or ch == "_":
        lexer.lex_ident(line, col, pos)
    elif ch.isdigit():
        lexer.lex_number(line, col, pos)
    elif lexer.lex_operator(line, col, pos):
        return
    else:
        raise _RecoveryLexError(f"unexpected character {ch!r}", line, col)


def _recovery_stop(source: str, start: int) -> int:
    """First statement-safe boundary at or after *start*."""
    candidates = [
        boundary
        for boundary in (source.find(mark, start + 1) for mark in ("\n", ";", "}"))
        if boundary >= 0
    ]
    return min(candidates) if candidates else len(source)


def _unterminated_kind(source: str, start: int) -> str:
    if source.startswith("/*", start):
        return "COMMENT"
    if source.startswith("'", start):
        return "CHAR"
    if source[start : start + 1] == '"' or _raw_string_prefix(source, start):
        return "STRING"
    return "UNKNOWN"


def lex_recover_unterminated_token(source: str) -> dict[str, Any]:
    """Lex a whole file, recovering from unterminated tokens with E001 diagnostics.

    When the shared lexer fails (unterminated string, char, escape or block
    comment), a structured ``E001`` diagnostic is recorded, a synthetic token
    covering the malformed text is emitted, and lexing resumes at the next
    newline, semicolon or closing brace so later statements still tokenize.
    """
    lexer = _RecoveringLexer(source)
    tokens: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    while True:
        try:
            lexer.skip_whitespace_and_comments()
        except _RecoveryLexError as error:
            start = _line_col_to_offset(source, error.line, error.col)
            lexer.i, lexer.line, lexer.col = start, error.line, error.col
            diagnostics.append(
                _diagnostic(
                    "E001",
                    error.message,
                    source,
                    start,
                    len(source),
                    kind=_unterminated_kind(source, start),
                )
            )
            stop = _recovery_stop(source, start)
            if stop <= start:
                stop = min(len(source), start + 1)
            while lexer.i < stop:
                lexer.advance()
            continue
        if lexer.at_end():
            break

        start = lexer.i
        line, col = lexer.line, lexer.col
        try:
            _lex_one_token(lexer)
        except _RecoveryLexError as error:
            stop = _recovery_stop(source, start)
            if stop <= start:
                stop = min(len(source), start + 1)
            kind = _unterminated_kind(source, start)
            tokens.append(
                _make_record(source, kind, source[start:stop], line, col, start, stop)
            )
            diagnostics.append(
                _diagnostic("E001", error.message, source, start, stop, kind=kind)
            )
            lexer.i, lexer.line, lexer.col = start, line, col
            while lexer.i < stop:
                lexer.advance()
            continue
        tokens.append(_token_record(source, lexer.tokens[-1]))

    return {
        "tokens": tokens,
        "diagnostics": diagnostics,
        "token_count": len(tokens),
        "diagnostic_count": len(diagnostics),
    }


# ---------------------------------------------------------------------------
# 4. parse_pratt_expr_with_precedence
# ---------------------------------------------------------------------------


class _ParseFailure(Exception):
    """Raised internally when the miniature Pratt parser cannot continue."""


class _PrattParser:
    """Small explicit-table Pratt parser producing structured expression nodes."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

    def peek(self, offset: int = 0) -> Token:
        position = min(self.index + offset, len(self.tokens) - 1)
        return self.tokens[position]

    def parse(self, min_precedence: int = 1) -> dict[str, Any]:
        left = self._prefix()
        while True:
            token = self.peek()
            if token.type in _EXPR_STOP:
                break
            if token.type == TokenType.LPAREN:
                left = self._call(left)
                continue
            if token.type == TokenType.DOT:
                left = self._dot(left)
                continue
            if token.type == TokenType.LBRACKET:
                left = self._index(left)
                continue
            if token.type == TokenType.QUESTION:
                self.index += 1
                left = {
                    "kind": "try",
                    "operand": left,
                    "precedence": PRECEDENCE_TABLE["postfix"],
                    "char_span": [left["char_span"][0], token.pos + 1],
                }
                continue
            op = _INFIX_OPS.get(token.type)
            if op is None:
                break
            precedence = PRECEDENCE_TABLE[op]
            if precedence < min_precedence:
                break
            self.index += 1
            next_min = precedence if op in _RIGHT_ASSOCIATIVE else precedence + 1
            right = self.parse(next_min)
            left = {
                "kind": "binary",
                "op": op,
                "precedence": precedence,
                "left": left,
                "right": right,
                "char_span": [left["char_span"][0], right["char_span"][1]],
            }
        return left

    def _prefix(self) -> dict[str, Any]:
        token = self.peek()
        if token.type in _PREFIX_OPS:
            op = _PREFIX_OPS[token.type]
            self.index += 1
            if token.type == TokenType.AMP and self.peek().type == TokenType.MUT:
                self.index += 1
                op = "&mut"
            operand = self.parse(PRECEDENCE_TABLE["unary"])
            return {
                "kind": "unary",
                "op": op,
                "precedence": PRECEDENCE_TABLE["unary"],
                "operand": operand,
                "char_span": [token.pos, operand["char_span"][1]],
            }
        return self._primary()

    @staticmethod
    def _literal(token: Token, source_text: str, end: int) -> dict[str, Any]:
        return {
            "kind": "literal",
            "value": source_text,
            "precedence": PRECEDENCE_TABLE["primary"],
            "char_span": [token.pos, end],
        }

    def _primary(self) -> dict[str, Any]:
        token = self.peek()
        end = (
            _token_end_char_offset(self._source_hint, token)
            if self._source_hint
            else token.pos + len(token.value)
        )
        if token.type == TokenType.NUMBER:
            self.index += 1
            return self._literal(token, token.value, end)
        if token.type in (TokenType.STRING, TokenType.CHAR):
            self.index += 1
            return self._literal(token, token.value, end)
        if token.type in (TokenType.TRUE, TokenType.FALSE):
            self.index += 1
            return self._literal(token, token.value, end)
        if token.type == TokenType.IDENT:
            self.index += 1
            text = token.value
            segments = [text]
            while (
                self.peek().type == TokenType.COLONCOLON
                and self.peek(1).type == TokenType.IDENT
            ):
                self.index += 1
                segment = self.peek()
                self.index += 1
                segments.append(segment.value)
                end = segment.pos + len(segment.value)
            if self.peek().type == TokenType.BANG:
                return self._macro(token, segments, end)
            return {
                "kind": "name" if len(segments) == 1 else "path",
                "value": "::".join(segments),
                "segments": segments,
                "precedence": PRECEDENCE_TABLE["primary"],
                "char_span": [token.pos, end],
            }
        if token.type == TokenType.LPAREN:
            self.index += 1
            first = self.parse()
            elems = [first]
            while self.peek().type == TokenType.COMMA:
                self.index += 1
                if self.peek().type == TokenType.RPAREN:
                    break
                elems.append(self.parse())
            close = self.peek()
            if close.type != TokenType.RPAREN:
                raise _ParseFailure("expected ')' in expression")
            self.index += 1
            if len(elems) == 1:
                return {
                    "kind": "group",
                    "expr": first,
                    "precedence": PRECEDENCE_TABLE["primary"],
                    "char_span": [token.pos, close.pos + 1],
                }
            return {
                "kind": "tuple",
                "elems": elems,
                "precedence": PRECEDENCE_TABLE["primary"],
                "char_span": [token.pos, close.pos + 1],
            }
        if token.type == TokenType.LBRACKET:
            self.index += 1
            elems: list[dict[str, Any]] = []
            while self.peek().type not in (TokenType.RBRACKET, TokenType.EOF):
                elems.append(self.parse())
                if self.peek().type != TokenType.COMMA:
                    break
                self.index += 1
            close = self.peek()
            if close.type != TokenType.RBRACKET:
                raise _ParseFailure("expected ']' in array expression")
            self.index += 1
            return {
                "kind": "array",
                "elems": elems,
                "precedence": PRECEDENCE_TABLE["primary"],
                "char_span": [token.pos, close.pos + 1],
            }
        raise _ParseFailure(f"unexpected token {token.type.name} in expression")

    def _call(self, callee: dict[str, Any]) -> dict[str, Any]:
        self.index += 1
        args: list[dict[str, Any]] = []
        while self.peek().type not in (TokenType.RPAREN, TokenType.EOF):
            args.append(self.parse())
            if self.peek().type != TokenType.COMMA:
                break
            self.index += 1
        close = self.peek()
        if close.type != TokenType.RPAREN:
            raise _ParseFailure("expected ')' in call")
        self.index += 1
        return {
            "kind": "call",
            "callee": callee,
            "args": args,
            "precedence": PRECEDENCE_TABLE["postfix"],
            "char_span": [callee["char_span"][0], close.pos + 1],
        }

    def _dot(self, receiver: dict[str, Any]) -> dict[str, Any]:
        self.index += 1
        name = self.peek()
        if name.type != TokenType.IDENT:
            raise _ParseFailure("expected field or method name after '.'")
        self.index += 1
        if self.peek().type != TokenType.LPAREN:
            return {
                "kind": "field",
                "receiver": receiver,
                "name": name.value,
                "precedence": PRECEDENCE_TABLE["postfix"],
                "char_span": [receiver["char_span"][0], name.pos + len(name.value)],
            }
        call = self._call(
            {
                "kind": "name",
                "value": name.value,
                "precedence": PRECEDENCE_TABLE["primary"],
                "char_span": [name.pos, name.pos + len(name.value)],
            }
        )
        return {
            "kind": "method",
            "receiver": receiver,
            "name": name.value,
            "args": call["args"],
            "precedence": PRECEDENCE_TABLE["postfix"],
            "char_span": [receiver["char_span"][0], call["char_span"][1]],
        }

    def _index(self, receiver: dict[str, Any]) -> dict[str, Any]:
        self.index += 1
        index = self.parse()
        close = self.peek()
        if close.type != TokenType.RBRACKET:
            raise _ParseFailure("expected ']' in index expression")
        self.index += 1
        return {
            "kind": "index",
            "receiver": receiver,
            "index": index,
            "precedence": PRECEDENCE_TABLE["postfix"],
            "char_span": [receiver["char_span"][0], close.pos + 1],
        }

    def _macro(
        self, name_token: Token, segments: list[str], end: int
    ) -> dict[str, Any]:
        self.index += 1  # '!'
        opener = self.peek()
        closer = _OPENERS.get(opener.type)
        if closer is None:
            raise _ParseFailure("expected delimiter after macro name")
        self.index += 1
        depth = 1
        while self.index < len(self.tokens) and depth:
            token = self.peek()
            self.index += 1
            if token.type in _OPENERS:
                depth += 1
            elif token.type == closer:
                depth -= 1
        if depth:
            raise _ParseFailure("unterminated macro invocation")
        close = self.tokens[self.index - 1]
        return {
            "kind": "macro",
            "path": "::".join(segments),
            "delimiter": _DELIMITER_TEXT[opener.type],
            "char_span": [name_token.pos, close.pos + len(close.value)],
        }

    _source_hint: str = ""

    @classmethod
    def with_source(cls, tokens: list[Token], source: str) -> _PrattParser:
        parser = cls(tokens)
        parser._source_hint = source
        return parser


def _expr_dict_text(node: dict[str, Any]) -> str:
    """Canonical text for a structured expression produced by ``_PrattParser``."""
    kind = node["kind"]
    if kind == "literal":
        return str(node["value"])
    if kind in ("name", "path"):
        return str(node["value"])
    if kind in ("group",):
        return f"({_expr_dict_text(node['expr'])})"
    if kind == "tuple":
        return "(" + ", ".join(_expr_dict_text(item) for item in node["elems"]) + ")"
    if kind == "array":
        return "[" + ", ".join(_expr_dict_text(item) for item in node["elems"]) + "]"
    if kind == "unary":
        return f"{node['op']}{_expr_dict_text(node['operand'])}"
    if kind == "binary":
        return (
            f"{_expr_dict_text(node['left'])} {node['op']} "
            f"{_expr_dict_text(node['right'])}"
        )
    if kind == "call":
        args = ", ".join(_expr_dict_text(arg) for arg in node["args"])
        return f"{_expr_dict_text(node['callee'])}({args})"
    if kind == "method":
        args = ", ".join(_expr_dict_text(arg) for arg in node["args"])
        return f"{_expr_dict_text(node['receiver'])}.{node['name']}({args})"
    if kind == "field":
        return f"{_expr_dict_text(node['receiver'])}.{node['name']}"
    if kind == "index":
        return f"{_expr_dict_text(node['receiver'])}[{_expr_dict_text(node['index'])}]"
    if kind == "try":
        return f"{_expr_dict_text(node['operand'])}?"
    if kind == "macro":
        start, stop = node["delimiter"]
        return f"{node['path']}!{start}{stop}"
    raise TypeError(f"unsupported expression kind {kind!r}")


def parse_pratt_expr_with_precedence(source: str | list[Token]) -> dict[str, Any]:
    """Parse one expression using the explicit :data:`PRECEDENCE_TABLE`.

    *source* may be source text or a list of :class:`lexer.Token` objects.
    Binary, unary, call, index, field, method, try and macro postfix forms are
    represented as nested dictionaries with ``kind``, ``op``, ``precedence``
    and exact ``char_span`` keys.  Assignment and range operators are
    right-associative; all other infix operators are left-associative.
    """
    text = source if isinstance(source, str) else ""
    tokens = _as_tokens(source)
    parser = _PrattParser.with_source(tokens, text)
    try:
        expression = parser.parse()
    except _ParseFailure as error:
        return {
            "ast": None,
            "error": str(error),
            "precedence_table": dict(PRECEDENCE_TABLE),
            "tokens": [_token_record(text, token) for token in tokens],
        }
    return {
        "ast": expression,
        "canonical": _expr_dict_text(expression),
        "precedence_table": dict(PRECEDENCE_TABLE),
        "tokens_consumed": parser.index,
        "remaining": [
            token.type.name
            for token in tokens[parser.index :]
            if token.type != TokenType.EOF
        ],
    }


# ---------------------------------------------------------------------------
# 5. parse_generic_type_params
# ---------------------------------------------------------------------------


def _parse_type_tokens(
    tokens: list[Token], index: int, stops: frozenset[TokenType]
) -> tuple[str, int, Token, Token]:
    """Consume one type expression, returning its text and token boundaries."""
    start = tokens[index]
    nesting = 0
    cursor = index
    while cursor < len(tokens) and tokens[cursor].type != TokenType.EOF:
        token = tokens[cursor]
        if nesting == 0 and token.type in stops:
            break
        if token.type in _OPENERS or token.type == TokenType.LT:
            nesting += 1
        elif token.type in _CLOSERS or token.type == TokenType.GT:
            if nesting == 0:
                break
            nesting -= 1
        cursor += 1
    end = tokens[max(cursor - 1, index)]
    return _tokens_text(tokens[index:cursor]), cursor, start, end


def parse_generic_type_params(source: str | list[Token]) -> list[dict[str, Any]]:
    """Parse a ``<T, U: Clone + Debug, V = i64>`` generic parameter list.

    Every returned entry records its ``name``, ``kind`` (``type``,
    ``lifetime`` or ``const``), inline trait ``bounds``, optional ``default``,
    the parameter's original ``text`` and an exact ``char_span``.  Bounds and
    defaults keep their own source text and nested generic arguments.
    """
    tokens = _as_tokens(source)
    opening = next(
        (index for index, token in enumerate(tokens) if token.type == TokenType.LT),
        None,
    )
    if opening is None:
        return []
    index = opening + 1
    params: list[dict[str, Any]] = []
    stops = frozenset({TokenType.COMMA, TokenType.GT, TokenType.EQ, TokenType.PLUS})

    while index < len(tokens) and tokens[index].type not in (
        TokenType.GT,
        TokenType.EOF,
    ):
        start = tokens[index]
        param_start = index
        if start.type == TokenType.LIFETIME:
            index += 1
            bounds: list[dict[str, Any]] = []
            if index < len(tokens) and tokens[index].type == TokenType.COLON:
                index += 1
                while index < len(tokens):
                    bound, index, bound_start, bound_end = _parse_type_tokens(
                        tokens, index, stops
                    )
                    bounds.append(
                        {
                            "name": bound,
                            "text": bound,
                            "char_span": [
                                bound_start.pos,
                                bound_end.pos + len(bound_end.value),
                            ],
                        }
                    )
                    if index < len(tokens) and tokens[index].type == TokenType.PLUS:
                        index += 1
                        continue
                    break
            params.append(
                {
                    "name": start.value,
                    "kind": "lifetime",
                    "bounds": bounds,
                    "default": None,
                    "text": _tokens_text(tokens[param_start:index]),
                    "line": start.line,
                    "col": start.col,
                    "char_span": [start.pos, start.pos + len(start.value)],
                }
            )
        elif start.type == TokenType.CONST:
            index += 1
            name = tokens[index] if index < len(tokens) else start
            index += 1
            ty: dict[str, Any] | None = None
            default: dict[str, Any] | None = None
            if index < len(tokens) and tokens[index].type == TokenType.COLON:
                index += 1
                ty_text, index, ty_start, ty_end = _parse_type_tokens(
                    tokens,
                    index,
                    frozenset({TokenType.COMMA, TokenType.GT, TokenType.EQ}),
                )
                ty = {
                    "text": ty_text,
                    "char_span": [ty_start.pos, ty_end.pos + len(ty_end.value)],
                }
            if index < len(tokens) and tokens[index].type == TokenType.EQ:
                index += 1
                default_text, index, default_start, default_end = _parse_type_tokens(
                    tokens, index, frozenset({TokenType.COMMA, TokenType.GT})
                )
                default = {
                    "text": default_text,
                    "char_span": [
                        default_start.pos,
                        default_end.pos + len(default_end.value),
                    ],
                }
            params.append(
                {
                    "name": name.value,
                    "kind": "const",
                    "type": ty,
                    "bounds": [],
                    "default": default,
                    "text": _tokens_text(tokens[param_start:index]),
                    "line": start.line,
                    "col": start.col,
                    "char_span": [start.pos, default_end.pos + len(default_end.value)]
                    if default is not None
                    else [start.pos, ty_end.pos + len(ty_end.value)]
                    if ty is not None
                    else [start.pos, name.pos + len(name.value)],
                }
            )
        elif start.type == TokenType.IDENT:
            index += 1
            bounds = []
            if index < len(tokens) and tokens[index].type == TokenType.COLON:
                index += 1
                while index < len(tokens):
                    bound, index, bound_start, bound_end = _parse_type_tokens(
                        tokens, index, stops
                    )
                    bounds.append(
                        {
                            "name": bound.split("<", 1)[0],
                            "text": bound,
                            "char_span": [
                                bound_start.pos,
                                bound_end.pos + len(bound_end.value),
                            ],
                        }
                    )
                    if index < len(tokens) and tokens[index].type == TokenType.PLUS:
                        index += 1
                        continue
                    break
            default = None
            end_token = bound_end if bounds else start
            if index < len(tokens) and tokens[index].type == TokenType.EQ:
                index += 1
                default_text, index, default_start, end_token = _parse_type_tokens(
                    tokens, index, frozenset({TokenType.COMMA, TokenType.GT})
                )
                default = {
                    "text": default_text,
                    "char_span": [
                        default_start.pos,
                        end_token.pos + len(end_token.value),
                    ],
                }
            params.append(
                {
                    "name": start.value,
                    "kind": "type",
                    "bounds": bounds,
                    "default": default,
                    "text": _tokens_text(tokens[param_start:index]),
                    "line": start.line,
                    "col": start.col,
                    "char_span": [start.pos, end_token.pos + len(end_token.value)],
                }
            )
        else:
            index += 1
            continue

        if index < len(tokens) and tokens[index].type == TokenType.COMMA:
            index += 1
            continue
        break

    return params


# ---------------------------------------------------------------------------
# 6. parse_pattern_match_arms
# ---------------------------------------------------------------------------


def parse_pattern_match_arms(source: str | list[Token]) -> dict[str, Any]:
    """Parse match arms with nested enum, struct, tuple and guarded patterns.

    The input may be a complete ``match expr { ... }`` or just a comma-separated
    arm list.  Arms are converted into JSON-like dictionaries via the project's
    AST nodes, including ``pattern``, optional ``guard``, ``body``, canonical
    texts and source spans adjusted back to the caller's coordinate space.
    """
    text = source if isinstance(source, str) else _tokens_text(source)
    text = text.strip()
    probe = lex(text)
    if probe and probe[0].type == TokenType.MATCH:
        wrapper = text
        prefix_length = 0
    else:
        prefix = "match __f1_subject__ { "
        wrapper = prefix + text + " }"
        prefix_length = len(prefix)

    match_node = Parser(lex(wrapper)).parse_match()
    normalized = ast_node_span_normalize(match_node, wrapper)
    printer = SourcePrinter()

    def shift(span: tuple[int, int] | list[int] | None) -> list[int] | None:
        if span is None:
            return None
        return [span[0] - prefix_length, span[1] - prefix_length]

    arms: list[dict[str, Any]] = []
    for position, arm in enumerate(normalized.arms):
        entry: dict[str, Any] = {
            "index": position,
            "pattern": _ast_to_dict(arm.pattern),
            "pattern_text": printer.pattern(arm.pattern),
            "guard": _ast_to_dict(arm.guard) if arm.guard is not None else None,
            "guard_text": printer.expr(arm.guard) if arm.guard is not None else None,
            "body": _ast_to_dict(arm.body),
            "body_text": printer.expr(arm.body),
            "line": arm.line,
            "col": arm.col,
            "char_span": shift(arm.span),
        }
        arms.append(entry)

    return {
        "scrutinee": _ast_to_dict(normalized.scrutinee),
        "arms": arms,
        "arm_count": len(arms),
        "source": wrapper[prefix_length : prefix_length + len(text)],
        "char_span": shift(normalized.span),
    }


# ---------------------------------------------------------------------------
# 7. parse_macro_invocations
# ---------------------------------------------------------------------------


def _split_macro_args(tokens: list[Token]) -> list[list[Token]]:
    """Split macro arguments on top-level commas, keeping nested groups intact."""
    args: list[list[Token]] = []
    current: list[Token] = []
    depth = 0
    angle = 0
    for token in tokens:
        if token.type == TokenType.EOF:
            break
        if token.type in _OPENERS:
            depth += 1
        elif token.type in _CLOSERS:
            depth = max(0, depth - 1)
        elif token.type == TokenType.LT:
            angle += 1
        elif token.type == TokenType.GT and angle:
            angle -= 1
        elif token.type == TokenType.COMMA and depth == 0 and angle == 0:
            args.append(current)
            current = []
            continue
        current.append(token)
    if current:
        args.append(current)
    return args


def _macro_context(tokens: list[Token], index: int, close_index: int) -> str:
    """Heuristic statement/expression classification for a macro invocation."""
    previous = tokens[index - 1] if index > 0 else None
    expression_context = {
        TokenType.LET,
        TokenType.EQ,
        TokenType.RETURN,
        TokenType.COMMA,
        TokenType.LPAREN,
        TokenType.LBRACKET,
        TokenType.PLUS,
        TokenType.MINUS,
        TokenType.STAR,
        TokenType.SLASH,
        TokenType.PERCENT,
    }
    if previous is not None and previous.type in expression_context:
        return "expression"
    following = tokens[close_index + 1] if close_index + 1 < len(tokens) else None
    if following is not None and following.type == TokenType.SEMI:
        return "statement"
    return "expression"


def parse_macro_invocations(source: str | list[Token]) -> dict[str, Any]:
    """Parse ``name!(...)``, ``name![...]`` and ``name!{...}`` invocations.

    Each invocation records its path, delimiter, statement/expression context,
    raw token text, top-level arguments (parsed with the frame's Pratt parser),
    exact spans and whether the delimiter was balanced.  Nested invocations are
    discovered as well; unbalanced delimiters yield diagnostics rather than
    terminating the scan.
    """
    tokens = _as_tokens(source)
    text = source if isinstance(source, str) else _tokens_text(source)
    macros: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens) and tokens[index].type != TokenType.EOF:
        if not (
            tokens[index].type == TokenType.IDENT
            and index + 2 < len(tokens)
            and tokens[index + 1].type == TokenType.BANG
        ):
            index += 1
            continue

        name_token = tokens[index]
        opener = tokens[index + 2]
        closer = _OPENERS.get(opener.type)
        if closer is None:
            diagnostics.append(
                _diagnostic(
                    "SYNTAX",
                    "expected ( [ or { after macro name",
                    text,
                    name_token.pos,
                    name_token.pos + len(name_token.value),
                    kind="MACRO",
                )
            )
            index += 2
            continue

        inner_start = index + 3
        cursor = inner_start
        stack = [closer]
        while cursor < len(tokens) and tokens[cursor].type != TokenType.EOF and stack:
            token = tokens[cursor]
            if token.type in _OPENERS:
                stack.append(_OPENERS[token.type])
            elif token.type == stack[-1]:
                stack.pop()
            cursor += 1
        terminated = not stack
        close_index = cursor - 1 if terminated else len(tokens) - 1
        inner_end = close_index if terminated else len(tokens)
        close_token = tokens[close_index]
        arg_tokens = tokens[inner_start:inner_end]
        args: list[dict[str, Any]] = []
        for raw_args in _split_macro_args(arg_tokens):
            try:
                parsed = parse_pratt_expr_with_precedence(raw_args)
            except _ParseFailure:
                parsed = None
            if parsed is not None and parsed["ast"] is not None:
                args.append(parsed["ast"])
            else:
                args.append(
                    {
                        "kind": "tokens",
                        "value": _tokens_text(raw_args),
                        "char_span": [
                            raw_args[0].pos if raw_args else close_token.pos,
                            raw_args[-1].pos + len(raw_args[-1].value)
                            if raw_args
                            else close_token.pos,
                        ],
                    }
                )
        end = close_token.pos + len(close_token.value)
        macro_record = _make_record(
            text,
            "MACRO",
            _tokens_text(tokens[index : close_index + 1]),
            name_token.line,
            name_token.col,
            name_token.pos,
            end,
        )
        macro_record.update(
            {
                "path": name_token.value,
                "delimiter": _DELIMITER_TEXT[opener.type],
                "kind": _macro_context(tokens, index, close_index),
                "args": args,
                "arg_text": _tokens_text(arg_tokens),
                "terminated": terminated,
                "inner_start": inner_start,
            }
        )
        macros.append(macro_record)
        if not terminated:
            diagnostics.append(
                _diagnostic(
                    "SYNTAX",
                    "unterminated macro invocation",
                    text,
                    name_token.pos,
                    end,
                    kind="MACRO",
                )
            )
        index = inner_start

    return {
        "macros": macros,
        "invocations": macros,
        "diagnostics": diagnostics,
        "count": len(macros),
    }


# ---------------------------------------------------------------------------
# 8. parse_error_sync_recovery
# ---------------------------------------------------------------------------


def _sync_index(tokens: list[Token], start: int) -> int:
    """Find the token index just past the next statement/item boundary."""
    depth = 0
    index = max(start, 0)
    while index < len(tokens):
        token = tokens[index]
        if token.type in _OPENERS:
            depth += 1
        elif token.type in _CLOSERS:
            if depth == 0:
                return index + 1
            depth -= 1
            if depth == 0 and token.type == TokenType.RBRACE:
                return index + 1
        elif depth == 0 and token.type == TokenType.SEMI:
            return index + 1
        elif (
            depth == 0
            and index > start
            and token.type in _ITEM_START
            and tokens[index - 1].type == TokenType.SEMI
        ):
            return index
        index += 1
    return len(tokens)


def parse_error_sync_recovery(source: str | list[Token]) -> dict[str, Any]:
    """Parse many items/statements, synchronizing after each syntax error.

    A shared :class:`parser.Parser` is driven over item and statement starts.
    On :class:`parser.ParseError` a structured ``SYNTAX`` diagnostic is
    recorded and the cursor jumps to the next semicolon, closing brace or
    item boundary, allowing later diagnostics from the same file to be
    collected instead of aborting at the first failure.
    """
    tokens = _as_tokens(source)
    text = source if isinstance(source, str) else _tokens_text(source)
    parser = Parser(tokens)
    recovered: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    sync_points: list[dict[str, Any]] = []

    while not parser.at_end():
        start = parser.i
        try:
            if (
                parser.peek().type in _ITEM_START
                and parser.peek().type != TokenType.LET
            ):
                node = parser.parse_item()
            elif parser.check(TokenType.LET):
                node = parser.parse_let()
            elif parser.check(TokenType.ERRDEFER):
                node = parser.parse_errdefer()
            else:
                node = parser.parse_expr()
                parser.eat(TokenType.SEMI)
        except ParseError as error:
            token = next(
                (
                    item
                    for item in tokens
                    if item.line == error.line and item.col == error.col
                ),
                tokens[min(parser.i, len(tokens) - 1)],
            )
            diagnostics.append(
                _diagnostic(
                    "SYNTAX",
                    error.msg,
                    text,
                    token.pos,
                    token.pos + len(token.value),
                    kind="STATEMENT",
                )
            )
            stop = _sync_index(tokens, max(parser.i, start + 1))
            sync_points.append(
                {
                    "from": start,
                    "to": stop,
                    "token": tokens[min(stop, len(tokens) - 1)].type.name,
                }
            )
            parser.i = stop
            continue

        recovered.append(_ast_to_dict(node))
        if parser.i <= start:
            parser.i += 1

    return {
        "diagnostics": diagnostics,
        "sync_points": sync_points,
        "recovered": recovered,
        "recovered_count": len(recovered),
        "diagnostic_count": len(diagnostics),
    }


# ---------------------------------------------------------------------------
# 9. ast_node_span_normalize
# ---------------------------------------------------------------------------


def ast_node_span_normalize(node: Any, source: str | None = None) -> Any:
    """Return a normalized deep copy of an AST with exact nested source spans.

    Leaf nodes derive their span from ``line``/``col`` plus the raw token at
    that position in *source*; composite nodes take the union of their child
    spans, extending to the opening keyword and to the matching brace for
    blocks and match expressions.  Existing spans are preserved when they are
    already valid, and the input tree is never mutated.
    """
    result = copy.deepcopy(node)
    tokens = lex(source) if source else []
    token_at = {(token.line, token.col): token for token in tokens}

    def normalize(value: object) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                normalize(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                normalize(item)
            return
        if not _is_ast_node(value):
            return

        children = list(_direct_ast_children(value))
        for child in children:
            normalize(child)

        current = getattr(value, "span", None)
        if (
            isinstance(current, (tuple, list))
            and len(current) == 2
            and all(isinstance(part, int) for part in current)
            and 0 <= current[0] <= current[1] <= (len(source) if source else current[1])
        ):
            value.span = (current[0], current[1])
            if not children:
                return

        spans = [
            tuple(child.span) for child in children if getattr(child, "span", None)
        ]
        own_offset = (
            _line_col_to_offset(source, value.line, value.col) if source else None
        )
        if not spans:
            if own_offset is None:
                return
            token = token_at.get((value.line, value.col))
            end = _token_end_char_offset(source, token) if token else own_offset + 1
            value.span = (own_offset, max(own_offset, end))
            return

        start = min(
            min(span[0] for span in spans),
            own_offset if own_offset is not None else len(source),
        )
        end = max(span[1] for span in spans)
        if source and isinstance(value, (ast.Block, ast.MatchExpr)):
            closing = _matching_brace_end(source, start)
            if closing is not None:
                end = max(end, closing)
        value.span = (start, end)

    normalize(result)
    return result


# ---------------------------------------------------------------------------
# 10. ast_pretty_print_canonical
# ---------------------------------------------------------------------------


def ast_pretty_print_canonical(node: Any) -> str:
    """Serialize an AST node to canonical Polyglot source text.

    Programs, items, statements, expressions, types and patterns are rendered
    with :class:`api.visualizer.SourcePrinter`; the output is deterministic so
    parse/print/parse round trips are stable.  Structured expression
    dictionaries from ``parse_pratt_expr_with_precedence`` are also accepted.
    """
    printer = SourcePrinter()
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if "kind" in node:
            return _expr_dict_text(node)
        raise TypeError("cannot canonicalize a dictionary without a 'kind' key")
    if isinstance(node, (list, tuple)):
        return "\n".join(ast_pretty_print_canonical(item) for item in node)
    if isinstance(node, ast.Program):
        return program_to_source(node)
    if isinstance(node, ast.Item):
        return printer.item(node, 0)
    if isinstance(node, ast.Stmt):
        return printer.stmt(node, 0)
    if isinstance(node, ast.Expr):
        return printer.expr(node)
    if isinstance(node, ast.Type):
        return ast.type_display_name(node)
    if isinstance(node, ast.Pattern):
        return printer.pattern(node)
    if isinstance(node, ast.GenericParam):
        bounds = "".join(f" + {bound.name}" for bound in node.bounds)
        return f"{node.name}{bounds}"
    if isinstance(node, ast.FnParam):
        ty = f": {ast.type_display_name(node.ty)}" if node.ty is not None else ""
        return f"{printer.pattern(node.pattern)}{ty}"
    if isinstance(node, ast.MatchArm):
        guard = f" if {printer.expr(node.guard)}" if node.guard is not None else ""
        return f"{printer.pattern(node.pattern)}{guard} => {printer.expr(node.body)}"
    if isinstance(node, ast.EnumVariant):
        fields = (
            "(" + ", ".join(ast.type_display_name(field) for field in node.fields) + ")"
            if node.fields is not None
            else ""
        )
        return f"{node.name}{fields}"
    if isinstance(node, ast.Attr):
        args = f"({node.args})" if node.args is not None else ""
        return f"#[{node.name}{args}]"
    if isinstance(node, ast.TraitItem):
        if node.fn is not None:
            return printer.fn(node.fn, 0)
        if node.ty is not None:
            return f"type Alias = {ast.type_display_name(node.ty)};"
    raise TypeError(f"cannot canonicalize node of type {type(node).__name__}")
