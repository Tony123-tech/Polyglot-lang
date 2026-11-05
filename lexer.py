"""
Polyglot lexer v2

Standalone module. Import from parser.py with:
    from lexer import Token, TokenType, LexError, Lexer, lex

Covers roadmap layer 1:
- keyword tokens
- comparison / logical / assignment operators
- punctuators
- line comments + nested block comments
- string escapes
- 0x / 0b / 0o number literals
- char literals
- identifiers starting with _
- raw strings r"..."
- multi-line strings \"\"\"...\"\"\"
- Unicode identifiers
- float scientific notation (1e10, 1.5e-3)
- number _ separators (1_000_000)
- lifetimes 'a
- macro !
- attributes #
- path separator ::
- ranges .. ..=
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(Enum):
    # literals
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    CHAR = auto()

    # keywords
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    LET = auto()
    MUT = auto()
    FN = auto()
    MATCH = auto()
    LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRUE = auto()
    FALSE = auto()
    STRUCT = auto()
    ENUM = auto()
    IMPL = auto()
    TRAIT = auto()
    MOD = auto()
    USE = auto()
    PUB = auto()
    CONST = auto()
    STATIC = auto()
    TYPE = auto()
    AS = auto()
    WHERE = auto()
    SELF = auto()
    UNSAFE = auto()
    ASYNC = auto()
    AWAIT = auto()
    DYN = auto()

    # arithmetic
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()

    # comparison
    EQEQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    # logical
    ANDAND = auto()
    OROR = auto()
    BANG = auto()

    # assignment
    EQ = auto()
    PLUSEQ = auto()
    MINUSEQ = auto()
    STAREQ = auto()
    SLASHEQ = auto()

    # punctuators
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMI = auto()
    COLON = auto()
    COLONCOLON = auto()
    ARROW = auto()
    FATARROW = auto()
    DOT = auto()
    DOTDOT = auto()
    DOTDOTEQ = auto()
    DOTDOTDOT = auto()
    PIPE = auto()
    AMP = auto()
    HASH = auto()
    AT = auto()
    QUESTION = auto()
    UNDERSCORE = auto()

    # special
    LIFETIME = auto()  # 'a
    EOF = auto()


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    col: int
    pos: int  # char offset, used for cache / debug

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"{msg} at {line}:{col}")
        self.msg = msg
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Keyword table
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, TokenType] = {
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "return": TokenType.RETURN,
    "let": TokenType.LET,
    "mut": TokenType.MUT,
    "fn": TokenType.FN,
    "match": TokenType.MATCH,
    "loop": TokenType.LOOP,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "impl": TokenType.IMPL,
    "trait": TokenType.TRAIT,
    "mod": TokenType.MOD,
    "use": TokenType.USE,
    "pub": TokenType.PUB,
    "const": TokenType.CONST,
    "static": TokenType.STATIC,
    "type": TokenType.TYPE,
    "as": TokenType.AS,
    "where": TokenType.WHERE,
    "self": TokenType.SELF,
    "unsafe": TokenType.UNSAFE,
    "async": TokenType.ASYNC,
    "await": TokenType.AWAIT,
    "dyn": TokenType.DYN,
}


# ---------------------------------------------------------------------------
# Operator table (longest match first)
# ---------------------------------------------------------------------------

OPERATORS: list[tuple[str, TokenType]] = [
    # 3 chars
    ("..=", TokenType.DOTDOTEQ),
    ("...", TokenType.DOTDOTDOT),
    # 2 chars
    ("::", TokenType.COLONCOLON),
    ("->", TokenType.ARROW),
    ("=>", TokenType.FATARROW),
    ("==", TokenType.EQEQ),
    ("!=", TokenType.NEQ),
    ("<=", TokenType.LE),
    (">=", TokenType.GE),
    ("&&", TokenType.ANDAND),
    ("||", TokenType.OROR),
    ("+=", TokenType.PLUSEQ),
    ("-=", TokenType.MINUSEQ),
    ("*=", TokenType.STAREQ),
    ("/=", TokenType.SLASHEQ),
    ("..", TokenType.DOTDOT),
    # 1 char
    ("+", TokenType.PLUS),
    ("-", TokenType.MINUS),
    ("*", TokenType.STAR),
    ("/", TokenType.SLASH),
    ("%", TokenType.PERCENT),
    ("<", TokenType.LT),
    (">", TokenType.GT),
    ("!", TokenType.BANG),
    ("=", TokenType.EQ),
    ("(", TokenType.LPAREN),
    (")", TokenType.RPAREN),
    ("{", TokenType.LBRACE),
    ("}", TokenType.RBRACE),
    ("[", TokenType.LBRACKET),
    ("]", TokenType.RBRACKET),
    (",", TokenType.COMMA),
    (";", TokenType.SEMI),
    (":", TokenType.COLON),
    (".", TokenType.DOT),
    ("|", TokenType.PIPE),
    ("&", TokenType.AMP),
    ("#", TokenType.HASH),
    ("@", TokenType.AT),
    ("?", TokenType.QUESTION),
]

# Sort so longest operators are tried first
OPERATORS.sort(key=lambda pair: -len(pair[0]))


# ---------------------------------------------------------------------------
# String escapes
# ---------------------------------------------------------------------------

SIMPLE_ESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "\\": "\\",
    '"': '"',
    "'": "'",
}


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    def __init__(self, source: str):
        self.src = source
        self.i = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    # ---- low-level helpers -------------------------------------------------

    def at_end(self) -> bool:
        return self.i >= len(self.src)

    def peek(self, offset: int = 0) -> str:
        j = self.i + offset
        return self.src[j] if j < len(self.src) else ""

    def advance(self) -> str:
        ch = self.src[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def advance_n(self, n: int) -> None:
        for _ in range(n):
            self.advance()

    def emit(self, ttype: TokenType, value: str, line: int, col: int, pos: int) -> None:
        self.tokens.append(Token(ttype, value, line, col, pos))

    def error(self, msg: str, line: int | None = None, col: int | None = None):
        raise LexError(msg, line or self.line, col or self.col)

    # ---- whitespace and comments ------------------------------------------

    def skip_whitespace_and_comments(self) -> None:
        while not self.at_end():
            ch = self.peek()
            if ch in " \t\r\n":
                self.advance()
            elif ch == "/" and self.peek(1) == "/":
                # Line comment: consume until newline
                while not self.at_end() and self.peek() != "\n":
                    self.advance()
            elif ch == "/" and self.peek(1) == "*":
                # Block comment (supports nesting)
                self.skip_block_comment()
            else:
                return

    def skip_block_comment(self) -> None:
        start_line, start_col = self.line, self.col
        self.advance_n(2)  # consume "/*"
        depth = 1
        while depth > 0:
            if self.at_end():
                self.error("unterminated block comment", start_line, start_col)
            if self.peek() == "/" and self.peek(1) == "*":
                self.advance_n(2)
                depth += 1
            elif self.peek() == "*" and self.peek(1) == "/":
                self.advance_n(2)
                depth -= 1
            else:
                self.advance()

    # ---- main loop ---------------------------------------------------------

    def lex(self) -> list[Token]:
        while not self.at_end():
            self.skip_whitespace_and_comments()
            if self.at_end():
                break

            line, col, pos = self.line, self.col, self.i
            ch = self.peek()

            # Raw string: r"..." or r#"..."#
            if ch == "r" and self._is_raw_string_start():
                self.lex_raw_string(line, col, pos)
                continue

            # Multi-line string: """..."""
            if ch == '"' and self.peek(1) == '"' and self.peek(2) == '"':
                self.lex_multiline_string(line, col, pos)
                continue

            # Normal string
            if ch == '"':
                self.lex_string(line, col, pos)
                continue

            # Char literal or lifetime
            if ch == "'":
                self.lex_quote(line, col, pos)
                continue

            # Identifier / keyword
            if ch.isalpha() or ch == "_":
                self.lex_ident(line, col, pos)
                continue

            # Number
            if ch.isdigit():
                self.lex_number(line, col, pos)
                continue

            # Operator / punctuator
            if self.lex_operator(line, col, pos):
                continue

            self.error(f"unexpected character {ch!r}")

        self.emit(TokenType.EOF, "", self.line, self.col, self.i)
        return self.tokens

    # ---- identifiers -------------------------------------------------------

    def lex_ident(self, line: int, col: int, pos: int) -> None:
        start = self.i
        # A lone "_" is UNDERSCORE, not an IDENT
        if self.peek() == "_" and not (self.peek(1).isalnum() or self.peek(1) == "_"):
            self.advance()
            self.emit(TokenType.UNDERSCORE, "_", line, col, pos)
            return
        while not self.at_end() and (self.peek().isalnum() or self.peek() == "_"):
            self.advance()
        text = self.src[start:self.i]
        ttype = KEYWORDS.get(text, TokenType.IDENT)
        self.emit(ttype, text, line, col, pos)

    # ---- numbers -----------------------------------------------------------

    def lex_number(self, line: int, col: int, pos: int) -> None:
        start = self.i

        # 0x / 0b / 0o
        if self.peek() == "0" and self.peek(1) in ("x", "X", "b", "B", "o", "O"):
            base_ch = self.peek(1).lower()
            self.advance_n(2)
            digits = self._consume_digits(base_ch)
            if not digits:
                self.error(f"expected digits after 0{base_ch}", line, col)
            self.emit(TokenType.NUMBER, self.src[start:self.i], line, col, pos)
            return

        # Integer part (with _ separators)
        self._consume_digits("dec")

        # Fractional part: only when "." is not followed by another "."
        # so `1..5` stays as a range, `1.` is allowed as a float
        if self.peek() == "." and self.peek(1) != ".":
            self.advance()
            self._consume_digits("dec")

        # Exponent
        if self.peek() in ("e", "E"):
            save_i, save_line, save_col = self.i, self.line, self.col
            self.advance()
            if self.peek() in ("+", "-"):
                self.advance()
            if self.peek().isdigit():
                self._consume_digits("dec")
            else:
                # Not actually an exponent, roll back
                self.i, self.line, self.col = save_i, save_line, save_col

        self.emit(TokenType.NUMBER, self.src[start:self.i], line, col, pos)

    def _consume_digits(self, base: str) -> str:
        """Consume digits of the given base, allowing '_' separators.

        '_' is rejected at the start and at the end of the digit run.
        """
        start = self.i
        allowed = {
            "x": "0123456789abcdefABCDEF",
            "b": "01",
            "o": "01234567",
            "dec": "0123456789",
        }[base]
        last_was_digit = False
        while not self.at_end():
            ch = self.peek()
            if ch in allowed:
                last_was_digit = True
                self.advance()
            elif ch == "_" and last_was_digit and self.peek(1) in allowed:
                self.advance()
            else:
                break
        return self.src[start:self.i]

    # ---- strings -----------------------------------------------------------

    def lex_string(self, line: int, col: int, pos: int) -> None:
        self.advance()  # opening quote
        buf: list[str] = []
        while not self.at_end():
            ch = self.peek()
            if ch == '"':
                self.advance()
                self.emit(TokenType.STRING, "".join(buf), line, col, pos)
                return
            if ch == "\\":
                buf.append(self.read_escape(line, col))
            else:
                buf.append(self.advance())
        self.error("unterminated string", line, col)

    def lex_multiline_string(self, line: int, col: int, pos: int) -> None:
        self.advance_n(3)  # consume opening """
        buf: list[str] = []
        while not self.at_end():
            if self.peek() == '"' and self.peek(1) == '"' and self.peek(2) == '"':
                self.advance_n(3)
                self.emit(TokenType.STRING, "".join(buf), line, col, pos)
                return
            ch = self.peek()
            if ch == "\\":
                buf.append(self.read_escape(line, col))
            else:
                buf.append(self.advance())
        self.error("unterminated multi-line string", line, col)

    def _is_raw_string_start(self) -> bool:
        # Matches r"..." or r#"..."#
        if self.peek() != "r":
            return False
        j = 1
        while self.peek(j) == "#":
            j += 1
        return self.peek(j) == '"'

    def lex_raw_string(self, line: int, col: int, pos: int) -> None:
        self.advance()  # consume 'r'
        hashes = 0
        while self.peek() == "#":
            self.advance()
            hashes += 1
        if self.peek() != '"':
            self.error("malformed raw string", line, col)
        self.advance()  # opening quote
        closing = '"' + "#" * hashes
        start = self.i
        while not self.at_end():
            if self.src.startswith(closing, self.i):
                value = self.src[start:self.i]
                self.advance_n(len(closing))
                self.emit(TokenType.STRING, value, line, col, pos)
                return
            self.advance()
        self.error("unterminated raw string", line, col)

    def read_escape(self, str_line: int, str_col: int) -> str:
        self.advance()  # consume backslash
        if self.at_end():
            self.error("unterminated escape", str_line, str_col)
        ch = self.advance()
        if ch in SIMPLE_ESCAPES:
            return SIMPLE_ESCAPES[ch]
        if ch == "x":
            return self._read_hex_escape(2)
        if ch == "u":
            if self.peek() == "{":
                self.advance()
                start = self.i
                while not self.at_end() and self.peek() != "}":
                    self.advance()
                if self.at_end():
                    self.error("unterminated unicode escape", str_line, str_col)
                hex_digits = self.src[start:self.i]
                self.advance()  # consume '}'
                return chr(int(hex_digits, 16))
            return self._read_hex_escape(4)
        # Backslash followed by newline = line continuation
        if ch == "\n":
            return ""
        self.error(f"unknown escape \\{ch}", str_line, str_col)
        return ""  # unreachable

    def _read_hex_escape(self, n: int) -> str:
        digits = []
        for _ in range(n):
            if self.at_end() or self.peek() not in "0123456789abcdefABCDEF":
                self.error("invalid hex escape")
            digits.append(self.advance())
        return chr(int("".join(digits), 16))

    # ---- char / lifetime ---------------------------------------------------

    def lex_quote(self, line: int, col: int, pos: int) -> None:
        """Disambiguate 'a' (char literal) from 'a (lifetime)."""
        # Escape right after quote -> char literal
        if self.peek(1) == "\\":
            self.lex_char(line, col, pos)
            return
        # Single char followed by closing quote -> char literal
        if self.peek(2) == "'" and self.peek(1) != "":
            self.lex_char(line, col, pos)
            return
        # Otherwise: lifetime
        self.advance()  # consume opening '
        if not (self.peek().isalpha() or self.peek() == "_"):
            self.error("expected identifier or char after '", line, col)
        start = self.i
        while not self.at_end() and (self.peek().isalnum() or self.peek() == "_"):
            self.advance()
        self.emit(TokenType.LIFETIME, "'" + self.src[start:self.i], line, col, pos)

    def lex_char(self, line: int, col: int, pos: int) -> None:
        self.advance()  # consume opening '
        if self.at_end():
            self.error("unterminated char literal", line, col)
        if self.peek() == "\\":
            value = self.read_escape(line, col)
        else:
            value = self.advance()
        if self.peek() != "'":
            self.error("unterminated char literal", line, col)
        self.advance()  # consume closing '
        self.emit(TokenType.CHAR, value, line, col, pos)

    # ---- operators ---------------------------------------------------------

    def lex_operator(self, line: int, col: int, pos: int) -> bool:
        for text, ttype in OPERATORS:
            if self.src.startswith(text, self.i):
                self.advance_n(len(text))
                self.emit(ttype, text, line, col, pos)
                return True
        return False


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def lex(source: str) -> list[Token]:
    return Lexer(source).lex()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = r'''
        // line comment
        /* block /* nested */ comment */
        fn main() -> i64 {
            let x: i64 = 0xFF_FF;
            let y = 1_000_000 + 1.5e-3;
            let s = "hi\n\t\"there\"";
            let r = r"raw \n stays";
            let m = """multi
line""";
            let c = 'a';
            let e = '\n';
            fn borrow<'a>(x: &'a i64) -> &'a i64 { x }
            println!("{}", x);
            if x >= 10 && y != 0 { return x; } else { return 0; }
            for i in 0..=10 { }
        }
    '''
    for tok in lex(src):
        print(tok)
