"""Where a literal ends: one scanner, shared by the lexer and the span helpers.

Both :mod:`compiler.frontend.lexing` and :mod:`compiler.frontend.framework`
need to know the exclusive end offset of a string, char or raw-string lexeme
before they can report a span.  Each used to carry its own copy of the scanning
loop, and the two copies had already drifted apart in how they treat a call that
does not actually start at a quote.  The canonical answer lives here.
"""

from __future__ import annotations

from compiler.core.lexer import Token, TokenType


def raw_string_prefix(source: str, start: int) -> tuple[int, int] | None:
    """Return ``(quote_index, hash_count)`` when a raw string starts at *start*."""
    if source[start : start + 1] != "r":
        return None
    index = start + 1
    while index < len(source) and source[index] == "#":
        index += 1
    if source[index : index + 1] == '"':
        return index, index - start - 1
    return None


def string_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Return ``(end_offset, terminated)`` for a string beginning at *start*."""
    raw = raw_string_prefix(source, start)
    if raw is not None:
        quote, hashes = raw
        closing = '"' + "#" * hashes
        close = source.find(closing, quote + 1)
        if close < 0:
            return len(source), False
        return close + len(closing), True

    if source.startswith('"""', start):
        index = start + 3
        while index < len(source):
            if source.startswith('"""', index):
                return index + 3, True
            index += 2 if source[index] == "\\" else 1
        return len(source), False

    if source[start : start + 1] != '"':
        return start, False
    index = start + 1
    while index < len(source):
        if source[index] == '"':
            return index + 1, True
        index += 2 if source[index] == "\\" else 1
    return len(source), False


def char_literal_end(source: str, start: int) -> tuple[int, bool]:
    """Return ``(end_offset, terminated)`` for a char literal."""
    if source[start : start + 1] != "'":
        return start, False
    index = start + 1
    while index < len(source):
        if source[index] == "'":
            return index + 1, True
        index += 2 if source[index] == "\\" else 1
    return len(source), False


def token_end(source: str, token: Token) -> int:
    """Exclusive source offset of a real token's raw lexeme."""
    if token.type == TokenType.EOF:
        return len(source)
    if token.type == TokenType.STRING:
        return string_literal_end(source, token.pos)[0]
    if token.type == TokenType.CHAR:
        return char_literal_end(source, token.pos)[0]
    return max(token.pos, min(len(source), token.pos + len(token.value)))


__all__ = [
    "char_literal_end",
    "raw_string_prefix",
    "string_literal_end",
    "token_end",
]
