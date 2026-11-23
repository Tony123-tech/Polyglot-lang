"""Pure lexical helpers built on the project's real :mod:`lexer`.

Every token record in this module is produced by :func:`lexer.lex`; raw
lexemes are recovered from the source text because a lexer token's ``value``
is decoded for strings and characters.  Spans are half-open character offset
lists ``[start, end]`` with a matching UTF-8 ``byte_span`` where required.
Diagnostic codes follow the frontend E001-E008 surface.
"""

from __future__ import annotations

import hashlib
import re
import string
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError
from compiler.core.lexer import (
    KEYWORDS,
    SIMPLE_ESCAPES,
    Lexer,
    LexError,
    Token,
    TokenType,
    lex,
)

__all__ = [
    "lex_balance_delimiters",
    "lex_char_class",
    "lex_count_by_type",
    "lex_decode_numeric",
    "lex_decode_string_escapes",
    "lex_detect_bom",
    "lex_escape_table",
    "lex_find_token",
    "lex_identifiers",
    "lex_indent_levels",
    "lex_keyword_boundary",
    "lex_keywords",
    "lex_literals",
    "lex_measure_token_widths",
    "lex_normalize_newlines",
    "lex_operators",
    "lex_scan_comments",
    "lex_source_fingerprint",
    "lex_split_lines",
    "lex_token_distance",
    "lex_token_window",
    "lex_tokenize",
    "lex_tokenize_with_offsets",
    "lex_validate_identifier",
    "lex_validate_source",
]

_KEYWORD_TYPES: Final[frozenset[TokenType]] = frozenset(KEYWORDS.values())
_NON_OPERATOR_TYPES: Final[frozenset[TokenType]] = frozenset(
    {
        TokenType.IDENT,
        TokenType.NUMBER,
        TokenType.STRING,
        TokenType.CHAR,
        TokenType.LIFETIME,
        TokenType.EOF,
    }
    | _KEYWORD_TYPES
)
_LITERAL_KINDS: Final[dict[TokenType, str]] = {
    TokenType.NUMBER: "number",
    TokenType.STRING: "string",
    TokenType.CHAR: "char",
    TokenType.TRUE: "bool",
    TokenType.FALSE: "bool",
}
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_OPENERS: Final[dict[str, str]] = {"(": ")", "[": "]", "{": "}"}
_CLOSERS: Final[frozenset[str]] = frozenset(_OPENERS.values())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_source(source: object) -> str:
    """Return *source* or raise E003 when the caller passed the wrong type."""
    if not isinstance(source, str):
        raise PolyglotError("E003", "source must be a str", line=1, col=1)
    return source


def _line_col(source: str, offset: int) -> tuple[int, int]:
    """One-based (line, column) for a clamped character offset."""
    offset = max(0, min(offset, len(source)))
    last_newline = source.rfind("\n", 0, offset)
    return source.count("\n", 0, offset) + 1, offset - last_newline


def _byte_offset(source: str, offset: int) -> int:
    """UTF-8 byte offset corresponding to a character offset."""
    return len(source[: max(0, min(offset, len(source)))].encode("utf-8"))


def _offset_from_line_col(source: str, line: int, col: int) -> int:
    """Character offset for a one-based line/column pair, clamped."""
    lines = source.split("\n")
    if line < 1 or col < 1 or line > len(lines):
        return len(source) if line > len(lines) else 0
    return min(len(source), sum(len(part) + 1 for part in lines[: line - 1]) + col - 1)


def _diagnostic(
    code: str,
    message: str,
    source: str,
    start: int,
    end: int,
    *,
    kind: str,
) -> dict[str, Any]:
    """Build a structured lexical diagnostic (never raises for source text)."""
    start = max(0, min(start, len(source)))
    end = max(start, min(end, len(source)))
    line, col = _line_col(source, start)
    return {
        "code": code,
        "message": message,
        "line": line,
        "col": col,
        "char_span": [start, end],
        "kind": kind,
    }


def _fail(
    code: str,
    message: str,
    source: str,
    start: int,
    end: int | None = None,
) -> NoReturn:
    """Raise a located :class:`PolyglotError` for malformed input."""
    start = max(0, min(start, len(source)))
    stop = len(source) if end is None else max(start, min(end, len(source)))
    line, col = _line_col(source, start)
    raise PolyglotError(code, message, line=line, col=col, span=(start, stop))


def _real_lex(source: str) -> list[Token]:
    """Call the real lexer, translating its ``LexError`` into E001."""
    try:
        return lex(source)
    except LexError as error:
        start = _offset_from_line_col(source, error.line, error.col)
        _fail("E001", error.msg, source, start, min(len(source), start + 1))


def _raw_string_quote(source: str, start: int) -> int | None:
    """Index of the opening quote when a raw string begins at *start*."""
    if source[start : start + 1] != "r":
        return None
    index = start + 1
    while index < len(source) and source[index] == "#":
        index += 1
    return index if source[index : index + 1] == '"' else None


def _string_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Exclusive end offset of a string lexeme and whether it is terminated."""
    quote = _raw_string_quote(source, start)
    if quote is not None:
        closing = '"' + "#" * (quote - start - 1)
        close = source.find(closing, quote + 1)
        if close < 0:
            return len(source), False
        return close + len(closing), True
    delimiter = '"""' if source.startswith('"""', start) else '"'
    index = start + len(delimiter)
    while index < len(source):
        if source.startswith(delimiter, index):
            return index + len(delimiter), True
        index += 2 if source[index] == "\\" else 1
    return len(source), False


def _char_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Exclusive end offset of a char lexeme and whether it is terminated."""
    if source[start : start + 1] != "'":
        return start, False
    index = start + 1
    while index < len(source):
        if source[index] == "'":
            return index + 1, True
        index += 2 if source[index] == "\\" else 1
    return len(source), False


def _token_end_char_offset(source: str, token: Token) -> int:
    """Exclusive source offset of a real token's raw lexeme."""
    if token.type == TokenType.EOF:
        return len(source)
    if token.type == TokenType.STRING:
        return _string_literal_end(source, token.pos)[0]
    if token.type == TokenType.CHAR:
        return _char_literal_end(source, token.pos)[0]
    return max(token.pos, min(len(source), token.pos + len(token.value)))


def _token_record(source: str, token: Token) -> dict[str, Any]:
    """Convert a real lexer token into the common token record shape."""
    start = token.pos
    end = _token_end_char_offset(source, token)
    return {
        "type": token.type.name,
        "value": token.value,
        "line": token.line,
        "col": token.col,
        "char_span": [start, end],
        "byte_span": [_byte_offset(source, start), _byte_offset(source, end)],
    }


def _records(source: str) -> list[dict[str, Any]]:
    """Tokenize validated source into common records (EOF included)."""
    return [
        _token_record(source, token) for token in _real_lex(_require_source(source))
    ]


def _normalize(source: str) -> str:
    """Collapse CRLF and lone CR line endings to LF."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


class _LexFailure(Exception):
    """Catchable, located form of a lexer error used by validation."""

    def __init__(self, message: str, line: int, col: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


class _RecoveringLexer(Lexer):
    """Real lexer whose ``error`` reports failures as values."""

    def error(  # type: ignore[override]
        self, msg: str, line: int | None = None, col: int | None = None
    ) -> NoReturn:
        raise _LexFailure(
            msg, self.line if line is None else line, self.col if col is None else col
        )


def _dispatch_one(lexer: Lexer) -> None:
    """Run exactly one token dispatch step of ``Lexer.lex``."""
    line, col, pos = lexer.line, lexer.col, lexer.i
    ch = lexer.peek()
    if ch == "r" and lexer._is_raw_string_start():
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
        raise _LexFailure(f"unexpected character {ch!r}", line, col)


def _recovery_stop(source: str, start: int) -> int:
    """First newline, semicolon or brace at or after *start*."""
    candidates = [
        found
        for found in (source.find(mark, start + 1) for mark in ("\n", ";", "}"))
        if found >= 0
    ]
    return min(candidates) if candidates else len(source)


def _problem_kind(source: str, start: int) -> str:
    """Coarse diagnostic kind for a failing source position."""
    if source.startswith("/*", start):
        return "COMMENT"
    if source[start : start + 1] == "'":
        return "CHAR"
    if source[start : start + 1] == '"' or _raw_string_quote(source, start) is not None:
        return "STRING"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# 1. lex_tokenize
# ---------------------------------------------------------------------------


def lex_tokenize(source: str) -> list[dict]:
    """Tokenize *source* into fully spanned records including the final EOF.

    String and char records use their raw source lexeme for ``char_span``;
    ``value`` stays the decoded token value from the real lexer.
    """
    return _records(source)


# ---------------------------------------------------------------------------
# 2. lex_count_by_type
# ---------------------------------------------------------------------------


def lex_count_by_type(source: str) -> dict[str, int]:
    """Count tokens by ``TokenType`` name, excluding EOF, ordered by name."""
    counts: dict[str, int] = {}
    for record in _records(source):
        if record["type"] == TokenType.EOF.name:
            continue
        name = record["type"]
        counts[name] = counts.get(name, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


# ---------------------------------------------------------------------------
# 3. lex_keywords
# ---------------------------------------------------------------------------


def lex_keywords(source: str) -> list[dict]:
    """Return only reserved-word tokens using the lexer's keyword map."""
    keywords: list[dict] = []
    for record in _records(source):
        if TokenType[record["type"]] in _KEYWORD_TYPES:
            keywords.append(
                {
                    "keyword": record["value"],
                    "line": record["line"],
                    "col": record["col"],
                    "char_span": record["char_span"],
                }
            )
    return keywords


# ---------------------------------------------------------------------------
# 4. lex_identifiers
# ---------------------------------------------------------------------------


def lex_identifiers(source: str) -> list[str]:
    """List distinct IDENT values in order of first appearance."""
    names: dict[str, None] = {}
    for record in _records(source):
        if record["type"] == TokenType.IDENT.name:
            names.setdefault(record["value"], None)
    return list(names)


# ---------------------------------------------------------------------------
# 5. lex_literals
# ---------------------------------------------------------------------------


def lex_literals(source: str) -> list[dict]:
    """Collect number, string, char and boolean literal records.

    ``raw`` is the exact source slice of the literal; an unterminated string
    or char lexeme is reported as E001.
    """
    text = _require_source(source)
    literals: list[dict] = []
    for token in _real_lex(text):
        kind = _LITERAL_KINDS.get(token.type)
        if kind is None:
            continue
        end, terminated = (
            _string_literal_end(text, token.pos)
            if token.type == TokenType.STRING
            else _char_literal_end(text, token.pos)
            if token.type == TokenType.CHAR
            else (token.pos + len(token.value), True)
        )
        if not terminated:
            _fail("E001", f"unterminated {kind} literal", text, token.pos, end)
        literals.append(
            {
                "kind": kind,
                "raw": text[token.pos : end],
                "value": token.value,
                "line": token.line,
                "col": token.col,
                "char_span": [token.pos, end],
            }
        )
    return literals


# ---------------------------------------------------------------------------
# 6. lex_operators
# ---------------------------------------------------------------------------


def lex_operators(source: str) -> list[dict]:
    """List operator and punctuation tokens, excluding literals and keywords."""
    operators: list[dict] = []
    for record in _records(source):
        if TokenType[record["type"]] in _NON_OPERATOR_TYPES:
            continue
        operators.append(
            {
                "op": record["value"],
                "type": record["type"],
                "line": record["line"],
                "col": record["col"],
                "char_span": record["char_span"],
            }
        )
    return operators


# ---------------------------------------------------------------------------
# 7. lex_validate_identifier
# ---------------------------------------------------------------------------


def lex_validate_identifier(name: str) -> dict:
    """Validate one identifier against the lexer's identifier and keyword rules."""
    if not isinstance(name, str):
        raise PolyglotError("E003", "identifier must be a str", line=1, col=1)
    if not name:
        _fail("E001", "identifier must not be empty", name, 0, 0)
    if name[0].isdigit():
        _fail("E001", "identifier must not start with a digit", name, 0, 1)
    for index, char in enumerate(name):
        if not (char.isalnum() or char == "_"):
            _fail(
                "E001",
                f"invalid identifier character {char!r}",
                name,
                index,
                index + 1,
            )
    if name in KEYWORDS:
        _fail("E001", f"{name!r} is a reserved keyword", name, 0, len(name))
    return {"name": name, "valid": True, "reason": None, "code": None}


# ---------------------------------------------------------------------------
# 8. lex_decode_numeric
# ---------------------------------------------------------------------------


def lex_decode_numeric(raw: str) -> dict:
    """Decode an integer or float literal with radix prefixes and underscores."""
    if not isinstance(raw, str):
        raise PolyglotError("E003", "numeric literal must be a str", line=1, col=1)
    if not raw or any(char.isspace() for char in raw):
        _fail("E008", "malformed numeric literal", raw, 0, len(raw))

    radix = 10
    digits = raw
    prefix_length = 0
    if raw[:2].lower() in ("0x", "0o", "0b"):
        radix = {"x": 16, "o": 8, "b": 2}[raw[1].lower()]
        prefix_length = 2
        digits = raw[2:]
    if not digits or digits[0] == "_" or digits[-1] == "_" or "__" in digits:
        _fail("E008", "malformed numeric literal", raw, prefix_length, len(raw))

    cleaned = digits.replace("_", "")
    if radix != 10:
        valid = {16: _HEX_DIGITS, 8: frozenset("01234567"), 2: frozenset("01")}[radix]
        if any(char not in valid for char in cleaned):
            _fail("E008", f"invalid base-{radix} digits", raw, prefix_length, len(raw))
        return {
            "raw": raw,
            "value": int(cleaned, radix),
            "kind": "int",
            "radix": radix,
        }

    if cleaned.isdigit():
        return {"raw": raw, "value": int(cleaned), "kind": "int", "radix": 10}
    if not _FLOAT_PATTERN.fullmatch(cleaned):
        _fail("E008", "malformed numeric literal", raw, 0, len(raw))
    try:
        value = float(cleaned)
    except ValueError:
        _fail("E008", "malformed float literal", raw, 0, len(raw))
    return {"raw": raw, "value": value, "kind": "float", "radix": 10}


# ---------------------------------------------------------------------------
# 9. lex_decode_string_escapes
# ---------------------------------------------------------------------------


def lex_decode_string_escapes(raw: str) -> dict:
    """Decode escapes in the *inner* text of an ordinary string literal."""
    if not isinstance(raw, str):
        raise PolyglotError("E003", "string content must be a str", line=1, col=1)
    decoded: list[str] = []
    escapes: list[dict] = []
    index = 0
    while index < len(raw):
        if raw[index] != "\\":
            decoded.append(raw[index])
            index += 1
            continue
        escape_start = index
        if index + 1 >= len(raw):
            _fail("E008", "unterminated escape", raw, index, index + 1)
        marker = raw[index + 1]
        consumed = 2
        value: str
        if marker in SIMPLE_ESCAPES:
            value = SIMPLE_ESCAPES[marker]
        elif marker == "x":
            digits = raw[index + 2 : index + 4]
            if len(digits) != 2 or any(digit not in _HEX_DIGITS for digit in digits):
                _fail("E008", "invalid hex escape", raw, index, index + 4)
            value = chr(int(digits, 16))
            consumed = 4
        elif marker == "u":
            braced = raw[index + 2 : index + 3] == "{"
            if braced:
                close = raw.find("}", index + 3)
                if close < 0:
                    _fail("E008", "unterminated unicode escape", raw, index, len(raw))
                digits = raw[index + 3 : close]
                consumed = close - index + 1
            else:
                digits = raw[index + 2 : index + 6]
                consumed = 6
            if not digits or any(digit not in _HEX_DIGITS for digit in digits):
                _fail("E008", "invalid unicode escape", raw, index, index + consumed)
            if not braced and len(digits) != 4:
                _fail("E008", "invalid unicode escape", raw, index, index + consumed)
            codepoint = int(digits, 16)
            if codepoint > 0x10FFFF:
                _fail(
                    "E008", "unicode escape out of range", raw, index, index + consumed
                )
            value = chr(codepoint)
        else:
            _fail("E008", f"unknown escape \\{marker}", raw, index, index + 2)
        decoded.append(value)
        escapes.append(
            {
                "raw": raw[escape_start : escape_start + consumed],
                "value": value,
                "char_span": [escape_start, escape_start + consumed],
            }
        )
        index += consumed
    return {"value": "".join(decoded), "escapes": escapes, "char_length": len(decoded)}


# ---------------------------------------------------------------------------
# 10. lex_detect_bom
# ---------------------------------------------------------------------------


def lex_detect_bom(source: str) -> dict:
    """Detect a leading UTF-8 BOM and reject a BOM found later in the text."""
    text = _require_source(source)
    misplaced = text.find("\ufeff", 1)
    if misplaced >= 0:
        _fail("E001", "misplaced byte order mark", text, misplaced, misplaced + 1)
    if text.startswith("\ufeff"):
        return {"encoding": "utf-8", "bom_length": 1, "content_start": 1}
    return {"encoding": None, "bom_length": 0, "content_start": 0}


# ---------------------------------------------------------------------------
# 11. lex_normalize_newlines
# ---------------------------------------------------------------------------


def lex_normalize_newlines(source: str) -> dict:
    """Replace CRLF and CR with LF, counting each replacement once."""
    text = _require_source(source)
    normalized = _normalize(text)
    return {
        "source": normalized,
        "replacements": text.count("\r"),
        "line_count": normalized.count("\n") + 1,
    }


# ---------------------------------------------------------------------------
# 12. lex_split_lines
# ---------------------------------------------------------------------------


def lex_split_lines(source: str) -> list[dict]:
    """Split *source* on LF into records with exact character and byte spans."""
    text = _require_source(source)
    lines: list[dict] = []
    offset = 0
    for index, part in enumerate(text.split("\n")):
        start = offset
        end = start + len(part)
        lines.append(
            {
                "index": index,
                "text": part,
                "char_span": [start, end],
                "byte_span": [_byte_offset(text, start), _byte_offset(text, end)],
                "line_no": index + 1,
            }
        )
        offset = end + 1
    return lines


# ---------------------------------------------------------------------------
# 13. lex_scan_comments
# ---------------------------------------------------------------------------


def lex_scan_comments(source: str) -> dict:
    """Collect line and nested block comments, ignoring comment-like text in literals.

    Unterminated block comments become E001 diagnostics rather than raised
    exceptions.
    """
    text = _require_source(source)
    comments: list[dict] = []
    diagnostics: list[dict] = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            end = len(text) if newline < 0 else newline
            comments.append(
                {
                    "kind": "line",
                    "text": text[index:end],
                    "char_span": [index, end],
                    "nesting": 0,
                }
            )
            index = end
            continue
        if text.startswith("/*", index):
            cursor = index + 2
            depth = 1
            maximum = 1
            while cursor < len(text):
                if text.startswith("/*", cursor):
                    depth += 1
                    maximum = max(maximum, depth)
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                    if depth == 0:
                        comments.append(
                            {
                                "kind": "block",
                                "text": text[index:cursor],
                                "char_span": [index, cursor],
                                "nesting": maximum,
                            }
                        )
                        break
                else:
                    cursor += 1
            else:
                diagnostics.append(
                    _diagnostic(
                        "E001",
                        "unterminated block comment",
                        text,
                        index,
                        len(text),
                        kind="COMMENT",
                    )
                )
                break
            index = cursor
            continue
        ch = text[index]
        if ch == '"' or _raw_string_quote(text, index) is not None:
            end, _terminated = _string_literal_end(text, index)
        elif ch == "'":
            end, _terminated = _char_literal_end(text, index)
        else:
            index += 1
            continue
        index = max(index + 1, end)
    return {"comments": comments, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# 14. lex_balance_delimiters
# ---------------------------------------------------------------------------


def lex_balance_delimiters(source: str) -> dict:
    """Balance parentheses, brackets and braces outside strings and comments."""
    text = _require_source(source)
    stack: list[tuple[str, int]] = []
    pairs: list[dict] = []
    unclosed: list[dict] = []
    unexpected: list[dict] = []
    diagnostics: list[dict] = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline
            continue
        if text.startswith("/*", index):
            cursor = index + 2
            depth = 1
            while cursor < len(text) and depth:
                if text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            index = cursor
            continue
        ch = text[index]
        if ch == '"' or _raw_string_quote(text, index) is not None:
            end, _terminated = _string_literal_end(text, index)
            index = max(index + 1, end)
            continue
        if ch == "'":
            end, _terminated = _char_literal_end(text, index)
            index = max(index + 1, end)
            continue
        if ch in _OPENERS:
            stack.append((ch, index))
        elif ch in _CLOSERS:
            if not stack:
                unexpected.append({"close": ch, "char_span": [index, index + 1]})
                diagnostics.append(
                    _diagnostic(
                        "E003",
                        f"unexpected delimiter {ch!r}",
                        text,
                        index,
                        index + 1,
                        kind="DELIMITER",
                    )
                )
            else:
                opener, start = stack.pop()
                if _OPENERS[opener] == ch:
                    pairs.append(
                        {"open": opener, "close": ch, "char_span": [start, index + 1]}
                    )
                else:
                    unexpected.append({"close": ch, "char_span": [index, index + 1]})
                    diagnostics.append(
                        _diagnostic(
                            "E003",
                            f"mismatched delimiter {ch!r}, expected {_OPENERS[opener]!r}",
                            text,
                            index,
                            index + 1,
                            kind="DELIMITER",
                        )
                    )
        index += 1
    for opener, start in stack:
        unclosed.append({"open": opener, "char_span": [start, start + 1]})
        diagnostics.append(
            _diagnostic(
                "E003",
                f"unclosed delimiter {opener!r}",
                text,
                start,
                start + 1,
                kind="DELIMITER",
            )
        )
    return {
        "balanced": not diagnostics,
        "pairs": pairs,
        "unclosed": unclosed,
        "unexpected": unexpected,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# 15. lex_indent_levels
# ---------------------------------------------------------------------------


def lex_indent_levels(source: str) -> list[dict]:
    """Measure indentation of each non-blank line in characters and width."""
    text = _require_source(source)
    levels: list[dict] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        prefix = line[: len(line) - len(line.lstrip(" \t"))]
        spaces = prefix.count(" ")
        tabs = prefix.count("\t")
        levels.append(
            {
                "line": number,
                "spaces": spaces,
                "tabs": tabs,
                "width": spaces + tabs * 4,
                "text_start": len(prefix),
            }
        )
    return levels


# ---------------------------------------------------------------------------
# 16. lex_source_fingerprint
# ---------------------------------------------------------------------------


def lex_source_fingerprint(source: str) -> str:
    """SHA-256 digest of the canonical token stream after newline normalization."""
    text = _normalize(_require_source(source))
    canonical = "\n".join(
        f"{record['type']}:{record['value']}" for record in _records(text)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 17. lex_token_window
# ---------------------------------------------------------------------------


def lex_token_window(source: str, center: int, radius: int = 2) -> list[dict]:
    """Return tokens in an inclusive radius around the token nearest *center*."""
    text = _require_source(source)
    if not isinstance(center, int):
        raise PolyglotError("E003", "center must be an int", line=1, col=1)
    if not 0 <= center <= len(text):
        _fail("E003", "center outside source", text, center, center + 1)
    if not isinstance(radius, int):
        raise PolyglotError("E003", "radius must be an int", line=1, col=1)
    if radius < 0:
        _fail("E003", "radius must be non-negative", text, 0, 1)
    records = _records(text)

    def distance(record: dict) -> int:
        start, end = record["char_span"]
        if start <= center < end:
            return 0
        if start == end:
            return abs(center - start)
        return min(abs(center - start), abs(center - (end - 1)))

    index = min(
        range(len(records)),
        key=lambda position: (distance(records[position]), position),
    )
    return records[max(0, index - radius) : min(len(records) - 1, index + radius) + 1]


# ---------------------------------------------------------------------------
# 18. lex_find_token
# ---------------------------------------------------------------------------


def lex_find_token(source: str, value: str, *, nth: int = 0) -> dict | None:
    """Return the ``nth`` token record whose decoded value equals *value*."""
    text = _require_source(source)
    if not isinstance(value, str):
        raise PolyglotError("E003", "searched token value must be a str", line=1, col=1)
    if not isinstance(nth, int) or nth < 0:
        _fail("E003", "nth must be non-negative", text, 0, 1)
    found = 0
    for record in _records(text):
        if record["value"] == value:
            if found == nth:
                return record
            found += 1
    return None


# ---------------------------------------------------------------------------
# 19. lex_token_distance
# ---------------------------------------------------------------------------


def lex_token_distance(source: str, first: str, second: str) -> int | None:
    """Token-index gap from the first *first* token to the next *second* token."""
    text = _require_source(source)
    if not isinstance(first, str) or not isinstance(second, str):
        raise PolyglotError("E003", "token values must be str", line=1, col=1)
    records = _records(text)
    start = next(
        (index for index, record in enumerate(records) if record["value"] == first),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index
            for index in range(start + 1, len(records))
            if records[index]["value"] == second
        ),
        None,
    )
    return None if end is None else end - start


# ---------------------------------------------------------------------------
# 20. lex_validate_source
# ---------------------------------------------------------------------------


def lex_validate_source(source: str) -> list[dict]:
    """Collect lexical diagnostics with recovery, without raising for source errors."""
    text = _require_source(source)
    lexer = _RecoveringLexer(text)
    diagnostics: list[dict] = []
    while True:
        try:
            lexer.skip_whitespace_and_comments()
        except _LexFailure as error:
            start = _offset_from_line_col(text, error.line, error.col)
            diagnostics.append(
                _diagnostic(
                    "E001",
                    error.message,
                    text,
                    start,
                    len(text),
                    kind=_problem_kind(text, start),
                )
            )
            break
        if lexer.at_end():
            break
        start = lexer.i
        line, col = lexer.line, lexer.col
        try:
            _dispatch_one(lexer)
        except _LexFailure as error:
            stop = _recovery_stop(text, start)
            if stop <= start:
                stop = min(len(text), start + 1)
            diagnostics.append(
                _diagnostic(
                    "E001",
                    error.message,
                    text,
                    start,
                    stop,
                    kind=_problem_kind(text, start),
                )
            )
            lexer.i, lexer.line, lexer.col = start, line, col
            while lexer.i < stop:
                lexer.advance()
    return diagnostics


# ---------------------------------------------------------------------------
# 21. lex_keyword_boundary
# ---------------------------------------------------------------------------


def lex_keyword_boundary(source: str, offset: int) -> dict:
    """Report whether *offset* is inside or at the edge of a keyword token."""
    text = _require_source(source)
    if not isinstance(offset, int):
        raise PolyglotError("E003", "offset must be an int", line=1, col=1)
    if not 0 <= offset <= len(text):
        _fail("E003", "offset outside source", text, offset, offset + 1)
    for keyword in lex_keywords(text):
        start, end = keyword["char_span"]
        if start < offset < end:
            return {
                "offset": offset,
                "at_boundary": False,
                "keyword": keyword["keyword"],
                "char_span": [start, end],
            }
        if offset in (start, end):
            return {
                "offset": offset,
                "at_boundary": True,
                "keyword": keyword["keyword"],
                "char_span": [start, end],
            }
    return {
        "offset": offset,
        "at_boundary": False,
        "keyword": None,
        "char_span": [offset, offset],
    }


# ---------------------------------------------------------------------------
# 22. lex_char_class
# ---------------------------------------------------------------------------


def lex_char_class(ch: str) -> str:
    """Classify one character using standard ``str`` predicates."""
    if not isinstance(ch, str) or len(ch) != 1:
        raise PolyglotError("E001", "expected exactly one character", line=1, col=1)
    if ch == "\n":
        return "newline"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "alpha"
    if ch.isalnum():
        return "alnum"
    if ch.isspace():
        return "space"
    if ch in string.punctuation:
        return "punct"
    return "other"


# ---------------------------------------------------------------------------
# 23. lex_escape_table
# ---------------------------------------------------------------------------


def lex_escape_table() -> dict[str, str]:
    """Return a fresh copy of the lexer's standard single-character escapes."""
    return dict(SIMPLE_ESCAPES)


# ---------------------------------------------------------------------------
# 24. lex_measure_token_widths
# ---------------------------------------------------------------------------


def lex_measure_token_widths(source: str) -> list[dict]:
    """Measure character and UTF-8 byte width of every token value, including EOF."""
    widths: list[dict] = []
    for record in _records(source):
        value = record["value"]
        widths.append(
            {
                "type": record["type"],
                "value": value,
                "chars": len(value),
                "bytes": len(value.encode("utf-8")),
                "char_span": record["char_span"],
            }
        )
    return widths


# ---------------------------------------------------------------------------
# 25. lex_tokenize_with_offsets
# ---------------------------------------------------------------------------


def lex_tokenize_with_offsets(
    source: str, start_line: int, start_col: int
) -> list[dict]:
    """Tokenize *source* and shift every line/column by caller-supplied origins."""
    text = _require_source(source)
    if not isinstance(start_line, int) or start_line < 1:
        _fail("E003", "start_line must be positive", text, 0, 1)
    if not isinstance(start_col, int) or start_col < 1:
        _fail("E003", "start_col must be positive", text, 0, 1)
    records: list[dict] = []
    for record in _records(text):
        original_line, original_col = record["line"], record["col"]
        shifted = dict(record)
        shifted["line"] = original_line + start_line
        shifted["col"] = original_col + start_col
        shifted["original_line"] = original_line
        shifted["original_col"] = original_col
        records.append(shifted)
    return records
