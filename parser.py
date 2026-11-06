"""
Recursive-descent + Pratt parser for Polyglot.

Imports from lexer.py:
    from lexer import Token, TokenType

Covers roadmap layer 2:

Statement layer:
- let x = expr; / let mut x = expr;
- x = expr;
- return expr;
- if cond { ... } else { ... }
- while cond { ... }
- for x in iter { ... }
- loop { ... }
- break; / continue;
- block expression { ... }
- match expr { pattern => expr, ... }

Expression layer:
- unary -x / !x / *x / &x / &mut x
- comparison / logical / arithmetic
- function call / method call / field access / index
- tuple / array / struct literal
- closure |x| x + 1
- cast x as i64
- range a..b / a..=b

Type layer:
- type annotation let x: i64 = 1;
- function signature fn f(a: i64) -> i64
- generic Vec<T> / HashMap<K, V>
"""

from __future__ import annotations

from ast_nodes import (
    ArrayExpr,
    Assign,
    Attr,
    Await,
    Binary,
    Block,
    Break,
    Call,
    Cast,
    Closure,
    CompoundAssign,
    ConstDecl,
    Continue,
    EnumDecl,
    EnumVariant,
    Expr,
    ExprStmt,
    Field,
    FnDecl,
    FnParam,
    For,
    GenericParam,
    Ident,
    IfExpr,
    ImplBlock,
    Index,
    Item,
    ItemStmt,
    LetStmt,
    LitBool,
    LitChar,
    LitFloat,
    LitInt,
    LitStr,
    Loop,
    MacroCall,
    MatchArm,
    MatchExpr,
    MethodCall,
    ModDecl,
    Paren,
    PatBinding,
    PatBox,
    PatConst,
    PatEnum,
    Path,
    PatIdent,
    PatLiteral,
    PatMacro,
    PatOr,
    PatPath,
    PatRange,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    Pattern,
    PatTuple,
    PatWildcard,
    Program,
    Range,
    RepeatArray,
    Return,
    Stmt,
    StructDecl,
    StructLiteral,
    TraitDecl,
    TraitItem,
    Try,
    TupleExpr,
    Type,
    TypeAlias,
    TypeArray,
    TypeDynTrait,
    TypeFn,
    TypeImplTrait,
    TypeName,
    TypeRef,
    TypeSlice,
    TypeTuple,
    TypeUnit,
    Unary,
    UseDecl,
    While,
)
from lexer import Token, TokenType

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"{msg} at {line}:{col}")
        self.msg = msg
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Precedence tables (Pratt)
# ---------------------------------------------------------------------------

INFIX_PREC: dict[TokenType, int] = {
    TokenType.OROR: 1,
    TokenType.ANDAND: 2,
    TokenType.EQEQ: 3,
    TokenType.NEQ: 3,
    TokenType.LT: 4,
    TokenType.GT: 4,
    TokenType.LE: 4,
    TokenType.GE: 4,
    TokenType.PIPE: 3,
    TokenType.AMP: 4,
    TokenType.PLUS: 5,
    TokenType.MINUS: 5,
    TokenType.STAR: 6,
    TokenType.SLASH: 6,
    TokenType.PERCENT: 6,
}

PREFIX_PREC = 7
POSTFIX_PREC = 8
ASSIGN_PREC = 0
CAST_PREC = 9


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    # ---- token helpers -----------------------------------------------------

    def at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self, offset: int = 0) -> Token:
        j = self.i + offset
        if j >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[j]

    def check(self, ttype: TokenType) -> bool:
        return self.peek().type == ttype

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.type != TokenType.EOF:
            self.i += 1
        return tok

    def eat(self, ttype: TokenType) -> Token | None:
        if self.check(ttype):
            return self.advance()
        return None

    def expect(self, ttype: TokenType, msg: str | None = None) -> Token:
        if self.check(ttype):
            return self.advance()
        tok = self.peek()
        want = msg or ttype.name
        self.error(f"expected {want}, got {tok.type.name} {tok.value!r}", tok)
        raise AssertionError("unreachable")  # for type checkers

    def error(self, msg: str, tok: Token | None = None):
        tok = tok or self.peek()
        raise ParseError(msg, tok.line, tok.col)

    # ---- entry -------------------------------------------------------------

    def parse_program(self) -> Program:
        items: list[Item] = []
        while not self.at_end():
            items.append(self.parse_item())
        first = self.tokens[0] if self.tokens else Token(TokenType.EOF, "", 1, 1, 0)
        source_lines = {
            token.line: token.source_line_text
            for token in self.tokens
            if token.source_line_text is not None
        }
        return Program(first.line, first.col, items, source_lines)

    # ---- attributes --------------------------------------------------------

    def parse_attrs(self) -> list[Attr]:
        attrs: list[Attr] = []
        while self.check(TokenType.HASH):
            tok = self.advance()  # '#'
            self.expect(TokenType.LBRACKET)
            # Capture name and optional (args) as raw text
            name_tok = self.expect(TokenType.IDENT)
            args = None
            if self.check(TokenType.LPAREN):
                depth = 0
                start = self.i
                while not self.at_end():
                    if self.check(TokenType.LPAREN):
                        depth += 1
                    elif self.check(TokenType.RPAREN):
                        depth -= 1
                        if depth == 0:
                            self.advance()
                            break
                    self.advance()
                args = "".join(t.value for t in self.tokens[start:self.i - 1])
            self.expect(TokenType.RBRACKET)
            attrs.append(Attr(tok.line, tok.col, name_tok.value, args))
        return attrs

    # ---- items -------------------------------------------------------------

    def parse_item(self) -> Item:
        # Attributes are consumed but not yet attached to items (deferred to v11).
        self.parse_attrs()

        is_pub = False
        if self.check(TokenType.PUB):
            self.advance()
            is_pub = True

        is_async = False
        is_unsafe = False
        while True:
            if self.check(TokenType.ASYNC):
                self.advance()
                is_async = True
            elif self.check(TokenType.UNSAFE):
                self.advance()
                is_unsafe = True
            else:
                break

        tok = self.peek()

        if tok.type == TokenType.FN:
            return self.parse_fn(is_pub, is_async, is_unsafe)
        if tok.type == TokenType.STRUCT:
            return self.parse_struct(is_pub)
        if tok.type == TokenType.ENUM:
            return self.parse_enum(is_pub)
        if tok.type == TokenType.IMPL:
            return self.parse_impl()
        if tok.type == TokenType.TRAIT:
            return self.parse_trait(is_pub)
        if tok.type == TokenType.USE:
            return self.parse_use()
        if tok.type == TokenType.MOD:
            return self.parse_mod()
        if tok.type == TokenType.CONST:
            return self.parse_const(is_static=False, is_pub=is_pub)
        if tok.type == TokenType.STATIC:
            return self.parse_const(is_static=True, is_pub=is_pub)
        if tok.type == TokenType.TYPE:
            return self.parse_type_alias(is_pub)

        self.error(f"expected item, got {tok.type.name} {tok.value!r}")
        raise AssertionError("unreachable")

    # ---- fn ----------------------------------------------------------------

    def parse_fn(self, is_pub: bool, is_async: bool, is_unsafe: bool) -> FnDecl:
        tok = self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENT).value
        generics = self.parse_generics()
        self.expect(TokenType.LPAREN)
        params: list[FnParam] = []
        while not self.check(TokenType.RPAREN):
            params.append(self.parse_fn_param())
            if not self.eat(TokenType.COMMA):
                break
        self.expect(TokenType.RPAREN)
        ret = None
        if self.eat(TokenType.ARROW):
            ret = self.parse_type()
        body = None
        if self.check(TokenType.LBRACE):
            body = self.parse_block()
        else:
            self.expect(TokenType.SEMI)
        return FnDecl(
            tok.line, tok.col, name, params, ret, body,
            generics, is_pub, is_async, is_unsafe,
        )

    def parse_fn_param(self) -> FnParam:
        tok = self.peek()
        pattern = self.parse_pattern()

        # Method receiver: &self, &mut, self, self - no `: type` follows.
        if self._is_self_pattern(pattern):
            return FnParam(tok.line, tok.col, pattern, None)

        self.expect(TokenType.COLON)
        ty = self.parse_type()
        return FnParam(tok.line, tok.col, pattern, ty)

    def _is_self_pattern(self, pattern: Pattern) -> bool:
        """True if the pattern is `self`, `&self, or `&mut self."""
        if isinstance(pattern, PatIdent) and pattern.name == "self":
            return True
        if isinstance(pattern, PatRef):
            return self._is_self_pattern(pattern.inner)
        return False
    def parse_generics(self) -> list[GenericParam]:
        params: list[GenericParam] = []
        if not self.eat(TokenType.LT):
            return params
        while not self.check(TokenType.GT) and not self.at_end():
            tok = self.peek()
            if self.check(TokenType.LIFETIME):
                self.advance()  # lifetime bounds are skipped
            else:
                name = self.expect(TokenType.IDENT).value
                bounds: list[TypeName] = []
                if self.eat(TokenType.COLON):
                    while True:
                        bounds.append(self.parse_type_name())
                        if not self.eat(TokenType.PLUS):
                            break
                params.append(GenericParam(tok.line, tok.col, name, bounds))
            if not self.eat(TokenType.COMMA):
                break
        self.expect(TokenType.GT)
        return params

    # ---- struct / enum -----------------------------------------------------

    def parse_struct(self, is_pub: bool) -> StructDecl:
        tok = self.expect(TokenType.STRUCT)
        name = self.expect(TokenType.IDENT).value
        generics = self.parse_generics()
        fields: list[tuple[str, Type, bool]] = []
        if self.eat(TokenType.LBRACE):
            while not self.check(TokenType.RBRACE):
                field_pub = bool(self.eat(TokenType.PUB))
                fname = self.expect(TokenType.IDENT).value
                self.expect(TokenType.COLON)
                fty = self.parse_type()
                fields.append((fname, fty, field_pub))
                if not self.eat(TokenType.COMMA):
                    break
            self.expect(TokenType.RBRACE)
        else:
            self.expect(TokenType.SEMI)
        return StructDecl(tok.line, tok.col, name, fields, generics, is_pub)

    def parse_enum(self, is_pub: bool) -> EnumDecl:
        tok = self.expect(TokenType.ENUM)
        name = self.expect(TokenType.IDENT).value
        generics = self.parse_generics()
        self.expect(TokenType.LBRACE)
        variants: list[EnumVariant] = []
        while not self.check(TokenType.RBRACE):
            vtok = self.expect(TokenType.IDENT)
            fields: list[Type] | None = None
            if self.check(TokenType.LPAREN):
                self.advance()
                fields = []
                while not self.check(TokenType.RPAREN):
                    fields.append(self.parse_type())
                    if not self.eat(TokenType.COMMA):
                        break
                self.expect(TokenType.RPAREN)
            elif self.check(TokenType.LBRACE):
                self.advance()
                fields = []
                while not self.check(TokenType.RBRACE):
                    self.expect(TokenType.IDENT)
                    self.expect(TokenType.COLON)
                    fields.append(self.parse_type())
                    if not self.eat(TokenType.COMMA):
                        break
                self.expect(TokenType.RBRACE)
            variants.append(EnumVariant(vtok.line, vtok.col, vtok.value, fields))
            if not self.eat(TokenType.COMMA):
                break
        self.expect(TokenType.RBRACE)
        return EnumDecl(tok.line, tok.col, name, variants, generics, is_pub)

    # ---- impl / trait ------------------------------------------------------

    def parse_impl(self) -> ImplBlock:
        tok = self.expect(TokenType.IMPL)
        generics = self.parse_generics()
        first_ty = self.parse_type()
        trait: Type | None = None
        if self.eat(TokenType.FOR):
            trait = first_ty
            target_ty = self.parse_type()
        else:
            target_ty = first_ty
        self.expect(TokenType.LBRACE)
        items: list[Item] = []
        while not self.check(TokenType.RBRACE):
            items.append(self.parse_item())
        self.expect(TokenType.RBRACE)
        return ImplBlock(tok.line, tok.col, target_ty, trait, items, generics)

    def parse_trait(self, is_pub: bool) -> TraitDecl:
        tok = self.expect(TokenType.TRAIT)
        name = self.expect(TokenType.IDENT).value
        generics = self.parse_generics()
        self.expect(TokenType.LBRACE)
        items: list[TraitItem] = []
        while not self.check(TokenType.RBRACE):
            if self.check(TokenType.FN):
                fn = self.parse_fn(False, False, False)
                items.append(TraitItem(fn.line, fn.col, fn, None))
            elif self.check(TokenType.TYPE):
                t = self.expect(TokenType.TYPE)
                # Associated type name is consumed but not stored yet.
                self.expect(TokenType.IDENT)
                ty = None
                if self.eat(TokenType.EQ):
                    ty = self.parse_type()
                    self.expect(TokenType.SEMI)
                items.append(TraitItem(t.line, t.col, None, ty))
            else:
                self.error("expected trait item")
        self.expect(TokenType.RBRACE)
        return TraitDecl(tok.line, tok.col, name, items, generics, is_pub)

    # ---- use / mod / const / type alias ------------------------------------

    def parse_use(self) -> UseDecl:
        tok = self.expect(TokenType.USE)
        parts: list[str] = []
        while not self.check(TokenType.SEMI) and not self.at_end():
            t = self.advance()
            parts.append(t.value)
        self.expect(TokenType.SEMI)
        return UseDecl(tok.line, tok.col, "".join(parts))

    def parse_mod(self) -> ModDecl:
        tok = self.expect(TokenType.MOD)
        name = self.expect(TokenType.IDENT).value
        if self.eat(TokenType.SEMI):
            return ModDecl(tok.line, tok.col, name, None)
        self.expect(TokenType.LBRACE)
        items: list[Item] = []
        while not self.check(TokenType.RBRACE):
            items.append(self.parse_item())
        self.expect(TokenType.RBRACE)
        return ModDecl(tok.line, tok.col, name, items)

    def parse_const(self, is_static: bool, is_pub: bool) -> ConstDecl:
        tok = self.advance()  # CONST or STATIC
        self.eat(TokenType.MUT)
        name = self.expect(TokenType.IDENT).value
        ty = None
        if self.eat(TokenType.COLON):
            ty = self.parse_type()
        self.expect(TokenType.EQ)
        value = self.parse_expr()
        self.expect(TokenType.SEMI)
        return ConstDecl(tok.line, tok.col, name, ty, value, is_static, is_pub)

    def parse_type_alias(self, is_pub: bool) -> TypeAlias:
        tok = self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENT).value
        generics = self.parse_generics()
        self.expect(TokenType.EQ)
        ty = self.parse_type()
        self.expect(TokenType.SEMI)
        return TypeAlias(tok.line, tok.col, name, ty, generics, is_pub)

    # ---- types -------------------------------------------------------------

    def parse_type(self) -> Type:
        tok = self.peek()

        # &
        if self.check(TokenType.AMP):
            self.advance()
            mutable = bool(self.eat(TokenType.MUT))
            if self.check(TokenType.LIFETIME):
                self.advance()  # lifetime is ignored by the parser
            inner = self.parse_type()
            return TypeRef(tok.line, tok.col, inner, mutable)

        # impl Trait
        if self.check(TokenType.IMPL):
            self.advance()
            bounds = [self.parse_type_name()]
            while self.eat(TokenType.PLUS):
                bounds.append(self.parse_type_name())
            return TypeImplTrait(tok.line, tok.col, bounds)

        # dyn Trait
        if self.check(TokenType.DYN):
            self.advance()
            bounds = [self.parse_type_name()]
            while self.eat(TokenType.PLUS):
                bounds.append(self.parse_type_name())
            return TypeDynTrait(tok.line, tok.col, bounds)

        # ( ) or tuple
        if self.check(TokenType.LPAREN):
            self.advance()
            if self.check(TokenType.RPAREN):
                self.advance()
                return TypeUnit(tok.line, tok.col)
            elems = [self.parse_type()]
            while self.eat(TokenType.COMMA) and not self.check(TokenType.RPAREN):
                elems.append(self.parse_type())
            self.expect(TokenType.RPAREN)
            return TypeTuple(tok.line, tok.col, elems)

        # [ ] slice or array
        if self.check(TokenType.LBRACKET):
            self.advance()
            inner = self.parse_type()
            if self.eat(TokenType.SEMI):
                size = self.parse_expr()
                self.expect(TokenType.RBRACKET)
                return TypeArray(tok.line, tok.col, inner, size)
            self.expect(TokenType.RBRACKET)
            return TypeSlice(tok.line, tok.col, inner)

        # fn pointer
        if self.check(TokenType.FN):
            self.advance()
            self.expect(TokenType.LPAREN)
            params: list[Type] = []
            while not self.check(TokenType.RPAREN):
                params.append(self.parse_type())
                if not self.eat(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)
            ret = None
            if self.eat(TokenType.ARROW):
                ret = self.parse_type()
            return TypeFn(tok.line, tok.col, params, ret)

        return self.parse_type_name()

    def parse_type_name(self) -> TypeName:
        tok = self.expect(TokenType.IDENT)
        args: list[Type] = []
        if self.check(TokenType.LT):
            self.advance()
            while not self.check(TokenType.GT):
                args.append(self.parse_type())
                if not self.eat(TokenType.COMMA):
                    break
            self.expect(TokenType.GT)
        return TypeName(tok.line, tok.col, tok.value, args)

    # ---- patterns ----------------------------------------------------------

    def parse_pattern(self) -> Pattern:
        tok = self.peek()
        alternatives = [self.parse_pattern_atom()]
        while self.eat(TokenType.PIPE):
            alternatives.append(self.parse_pattern_atom())
        if len(alternatives) == 1:
            return alternatives[0]
        return PatOr(tok.line, tok.col, alternatives)

    def parse_pattern_atom(self) -> Pattern:
        tok = self.peek()

        if self.check(TokenType.DOTDOT):
            self.advance()
            return PatRest(tok.line, tok.col)

        if self.check(TokenType.DOTDOTEQ):
            inclusive = self.advance().type == TokenType.DOTDOTEQ
            end = None if self._pattern_ends_here() else self.parse_prefix()
            return PatRange(tok.line, tok.col, None, end, inclusive)

        if self.check(TokenType.UNDERSCORE):
            self.advance()
            return PatWildcard(tok.line, tok.col)

        if self.check(TokenType.CONST):
            self.advance()
            path = self._parse_pattern_path()
            return PatConst(tok.line, tok.col, path)

        # `self` is a keyword token but valid as a pattern (method receiver).
        if self.check(TokenType.SELF):
            self.advance()
            return PatIdent(tok.line, tok.col, "self", False)

        if self.check(TokenType.MUT) or self.check(TokenType.IDENT):
            mutable = bool(self.eat(TokenType.MUT))
            if self.check(TokenType.IDENT):
                name_tok = self.advance()
                name = name_tok.value
                while self.eat(TokenType.COLONCOLON):
                    name += "::" + self.expect(TokenType.IDENT).value
                if name_tok.value == "box" and "::" not in name:
                    return PatBox(
                        tok.line, tok.col, self.parse_pattern_atom(),
                    )
                if self.check(TokenType.AT):
                    self.advance()
                    inner = self.parse_pattern()
                    return PatBinding(tok.line, tok.col, name, inner)
                if self.eat(TokenType.BANG):
                    open_type = self.peek().type
                    delimiters = {
                        TokenType.LPAREN: TokenType.RPAREN,
                        TokenType.LBRACE: TokenType.RBRACE,
                        TokenType.LBRACKET: TokenType.RBRACKET,
                    }
                    closing = delimiters.get(open_type)
                    if closing is None:
                        self.error("expected delimiter after pattern macro '!'", tok)
                    delimiter = {
                        TokenType.LPAREN: "()",
                        TokenType.LBRACE: "{}",
                        TokenType.LBRACKET: "[]",
                    }[open_type]
                    self.advance()
                    token_parts: list[str] = []
                    opening_types = {
                        TokenType.LPAREN: TokenType.RPAREN,
                        TokenType.LBRACE: TokenType.RBRACE,
                        TokenType.LBRACKET: TokenType.RBRACKET,
                    }
                    closing_stack = [closing]
                    while closing_stack and not self.at_end():
                        current = self.advance()
                        if current.type in opening_types:
                            closing_stack.append(opening_types[current.type])
                        elif current.type == closing_stack[-1]:
                            closing_stack.pop()
                            if not closing_stack:
                                break
                        token_parts.append(current.value)
                    if closing_stack:
                        self.error("unterminated pattern macro invocation", tok)
                    return PatMacro(
                        tok.line, tok.col, name, "".join(token_parts), delimiter,
                    )
                if self.check(TokenType.LBRACE):
                    self.advance()
                    fields: list[tuple[str, Pattern]] = []
                    rest = False
                    while not self.check(TokenType.RBRACE):
                        if self.eat(TokenType.DOTDOT):
                            if rest:
                                self.error(
                                    "struct pattern may contain only one rest",
                                )
                            rest = True
                            self.eat(TokenType.COMMA)
                            if self.check(TokenType.DOTDOT):
                                self.error(
                                    "struct pattern may contain only one rest",
                                )
                            break
                        field_tok = self.expect(TokenType.IDENT)
                        if self.eat(TokenType.COLON):
                            field_pattern = self.parse_pattern()
                        else:
                            field_pattern = PatIdent(
                                field_tok.line, field_tok.col, field_tok.value,
                            )
                        fields.append((field_tok.value, field_pattern))
                        if not self.eat(TokenType.COMMA):
                            break
                    self.expect(TokenType.RBRACE)
                    return PatStruct(tok.line, tok.col, name, fields, rest)
                if self.check(TokenType.LPAREN):
                    self.advance()
                    args: list[Pattern] = []
                    while not self.check(TokenType.RPAREN):
                        args.append(self.parse_pattern())
                        if not self.eat(TokenType.COMMA):
                            break
                    self.expect(TokenType.RPAREN)
                    return PatEnum(tok.line, tok.col, name, args)
                if name == "box":
                    return PatBox(
                        tok.line, tok.col, self.parse_pattern_atom(),
                    )
                if "::" in name or name[0].isupper():
                    return PatPath(tok.line, tok.col, name)
                return PatIdent(tok.line, tok.col, name, mutable)

        if self.check(TokenType.NUMBER) or self.check(TokenType.STRING) \
                or self.check(TokenType.CHAR) or self.check(TokenType.TRUE) \
                or self.check(TokenType.FALSE) or self.check(TokenType.MINUS):
            start = self.parse_prefix()
            if self.check(TokenType.DOTDOT) or self.check(TokenType.DOTDOTEQ):
                inclusive = self.advance().type == TokenType.DOTDOTEQ
                end = None if self._pattern_ends_here() else self.parse_prefix()
                return PatRange(tok.line, tok.col, start, end, inclusive)
            return PatLiteral(tok.line, tok.col, start)

        if self.check(TokenType.LPAREN):
            self.advance()
            if self.check(TokenType.RPAREN):
                self.advance()
                return PatTuple(tok.line, tok.col, [])
            elems: list[Pattern] = []
            elems.append(self.parse_pattern())
            has_comma = bool(self.eat(TokenType.COMMA))
            while has_comma and not self.check(TokenType.RPAREN):
                elems.append(self.parse_pattern())
                has_comma = bool(self.eat(TokenType.COMMA))
            self.expect(TokenType.RPAREN)
            if len(elems) == 1 and not has_comma:
                return elems[0]
            return PatTuple(tok.line, tok.col, elems)

        if self.check(TokenType.AMP):
            self.advance()
            mutable = bool(self.eat(TokenType.MUT))
            inner = self.parse_pattern()
            return PatRef(tok.line, tok.col, inner, mutable)

        if self.check(TokenType.LBRACKET):
            self.advance()
            elems: list[Pattern] = []
            rest_count = 0
            while not self.check(TokenType.RBRACKET):
                elem = self.parse_pattern()
                if isinstance(elem, PatRest) or (
                    isinstance(elem, PatBinding)
                    and isinstance(elem.inner, PatRest)
                ):
                    rest_count += 1
                    if rest_count > 1:
                        raise ParseError(
                            "slice pattern may contain only one rest pattern",
                            elem.line,
                            elem.col,
                        )
                elems.append(elem)
                if not self.eat(TokenType.COMMA):
                    break
            self.expect(TokenType.RBRACKET)
            return PatSlice(tok.line, tok.col, elems)

        self.error(f"expected pattern, got {tok.type.name}")
        raise AssertionError("unreachable")

    def _parse_pattern_path(self) -> str:
        path = self.expect(TokenType.IDENT, "constant path").value
        while self.eat(TokenType.COLONCOLON):
            path += "::" + self.expect(TokenType.IDENT, "path segment").value
        return path

    def _pattern_ends_here(self) -> bool:
        return self.peek().type in {
            TokenType.COMMA, TokenType.FATARROW, TokenType.RPAREN,
            TokenType.RBRACKET, TokenType.RBRACE, TokenType.PIPE,
            TokenType.IF, TokenType.EOF,
        }

    # ---- statements --------------------------------------------------------

    def parse_block(self) -> Block:
        tok = self.expect(TokenType.LBRACE)
        stmts: list[Stmt] = []
        tail: Expr | None = None
        while not self.check(TokenType.RBRACE) and not self.at_end():
            if self.peek().type in {
                TokenType.FN, TokenType.STRUCT, TokenType.ENUM,
                TokenType.IMPL, TokenType.TRAIT, TokenType.USE,
                TokenType.MOD, TokenType.CONST, TokenType.STATIC,
                TokenType.TYPE, TokenType.PUB, TokenType.HASH,
            }:
                item = self.parse_item()
                stmts.append(ItemStmt(item.line, item.col, item))
                continue

            if self.check(TokenType.LET):
                stmts.append(self.parse_let())
                continue

            e = self.parse_expr()
            if self.eat(TokenType.SEMI):
                stmts.append(ExprStmt(e.line, e.col, e))
            elif self.check(TokenType.RBRACE):
                tail = e
                break
            else:
                stmts.append(ExprStmt(e.line, e.col, e))
        self.expect(TokenType.RBRACE)
        return Block(tok.line, tok.col, stmts, tail)

    def parse_let(self) -> LetStmt:
        tok = self.expect(TokenType.LET)
        mutable = bool(self.eat(TokenType.MUT))
        pattern = self.parse_pattern()
        ty = None
        if self.eat(TokenType.COLON):
            ty = self.parse_type()
        value = None
        if self.eat(TokenType.EQ):
            value = self.parse_expr()
        self.expect(TokenType.SEMI)
        return LetStmt(tok.line, tok.col, pattern, ty, value, mutable)

    # ---- expressions (Pratt) ----------------------------------------------

    def parse_expr(self, min_prec: int = 0) -> Expr:
        left = self.parse_prefix()

        while True:
            tok = self.peek()

            # Assignment (right-associative, lowest precedence)
            if tok.type in (
                TokenType.EQ, TokenType.PLUSEQ,
                TokenType.MINUSEQ, TokenType.STAREQ, TokenType.SLASHEQ,
            ) and min_prec <= ASSIGN_PREC:
                op = tok.value
                self.advance()
                right = self.parse_expr(ASSIGN_PREC)
                if tok.type == TokenType.EQ:
                    left = Assign(tok.line, tok.col, left, right)
                else:
                    left = CompoundAssign(tok.line, tok.col, op, left, right)
                continue

            # Cast
            if tok.type == TokenType.AS and min_prec <= CAST_PREC:
                self.advance()
                ty = self.parse_type()
                left = Cast(tok.line, tok.col, left, ty)
                continue

            # Range
            if tok.type in (TokenType.DOTDOT, TokenType.DOTDOTEQ) and min_prec <= 1:
                self.advance()
                inclusive = tok.type == TokenType.DOTDOTEQ
                end = None
                if not self._at_expr_end():
                    end = self.parse_expr(2)
                left = Range(tok.line, tok.col, left, end, inclusive)
                continue

            # Postfix: call
            if tok.type == TokenType.LPAREN:
                self.advance()
                args: list[Expr] = []
                while not self.check(TokenType.RPAREN):
                    args.append(self.parse_expr())
                    if not self.eat(TokenType.COMMA):
                        break
                self.expect(TokenType.RPAREN)
                left = Call(tok.line, tok.col, left, args)
                continue

            # Postfix: field / method
            if tok.type == TokenType.DOT:
                self.advance()
                if not self.check(TokenType.IDENT):
                    self.error("expected field or method name after '.'")
                m = self.advance()
                generic_args: list[Type] = []
                if self.eat(TokenType.COLONCOLON):
                    self.expect(TokenType.LT)
                    while not self.check(TokenType.GT):
                        generic_args.append(self.parse_type())
                        if not self.eat(TokenType.COMMA):
                            break
                    self.expect(TokenType.GT)
                if self.check(TokenType.LPAREN):
                    self.advance()
                    args: list[Expr] = []
                    while not self.check(TokenType.RPAREN):
                        args.append(self.parse_expr())
                        if not self.eat(TokenType.COMMA):
                            break
                    self.expect(TokenType.RPAREN)
                    left = MethodCall(
                        tok.line, tok.col, left, m.value, args, generic_args,
                    )
                else:
                    left = Field(tok.line, tok.col, left, m.value)
                continue

            # Postfix: index
            if tok.type == TokenType.LBRACKET:
                self.advance()
                idx = self.parse_expr()
                self.expect(TokenType.RBRACKET)
                left = Index(tok.line, tok.col, left, idx)
                continue

            # Postfix: try (?)
            if tok.type == TokenType.QUESTION:
                self.advance()
                left = Try(tok.line, tok.col, left)
                continue

            # Infix
            prec = INFIX_PREC.get(tok.type)
            if prec is None or prec < min_prec:
                break
            self.advance()
            right = self.parse_expr(prec + 1)
            op = self._infix_op(tok.type)
            left = Binary(tok.line, tok.col, op, left, right)

        return left

    def _infix_op(self, ttype: TokenType) -> str:
        return {
            TokenType.PLUS: "+", TokenType.MINUS: "-",
            TokenType.STAR: "*", TokenType.SLASH: "/",
            TokenType.PERCENT: "%",
            TokenType.EQEQ: "==", TokenType.NEQ: "!=",
            TokenType.LT: "<", TokenType.GT: ">",
            TokenType.LE: "<=", TokenType.GE: ">=",
            TokenType.ANDAND: "&&", TokenType.OROR: "||",
            TokenType.AMP: "&", TokenType.PIPE: "|",
        }[ttype]

    def _at_expr_end(self) -> bool:
        return self.peek().type in {
            TokenType.SEMI, TokenType.RPAREN, TokenType.RBRACKET,
            TokenType.RBRACE, TokenType.COMMA, TokenType.EOF,
        }

    # ---- prefix ------------------------------------------------------------

    def parse_prefix(self) -> Expr:
        tok = self.peek()

        # Unary - and !
        if tok.type in (TokenType.MINUS, TokenType.BANG):
            self.advance()
            operand = self.parse_expr(PREFIX_PREC)
            return Unary(tok.line, tok.col, tok.value, operand)

        # Deref
        if tok.type == TokenType.STAR:
            self.advance()
            operand = self.parse_expr(PREFIX_PREC)
            return Unary(tok.line, tok.col, "*", operand)

        # Reference
        if tok.type == TokenType.AMP:
            self.advance()
            mutable = bool(self.eat(TokenType.MUT))
            operand = self.parse_expr(PREFIX_PREC)
            return Unary(tok.line, tok.col, "&mut" if mutable else "&", operand, mutable)

        # return / break / continue
        if tok.type == TokenType.RETURN:
            self.advance()
            value = None if self._at_expr_end() else self.parse_expr()
            return Return(tok.line, tok.col, value)

        if tok.type == TokenType.BREAK:
            self.advance()
            value = None if self._at_expr_end() else self.parse_expr()
            return Break(tok.line, tok.col, value)

        if tok.type == TokenType.CONTINUE:
            self.advance()
            return Continue(tok.line, tok.col)

        # if
        if tok.type == TokenType.IF:
            return self.parse_if()

        # match
        if tok.type == TokenType.MATCH:
            return self.parse_match()

        # loop
        if tok.type == TokenType.LOOP:
            self.advance()
            body = self.parse_block()
            return Loop(tok.line, tok.col, body)

        # while
        if tok.type == TokenType.WHILE:
            self.advance()
            cond = self.parse_expr()
            body = self.parse_block()
            return While(tok.line, tok.col, cond, body)

        # for
        if tok.type == TokenType.FOR:
            self.advance()
            pattern = self.parse_pattern()
            self.expect(TokenType.IN)
            iter_expr = self.parse_expr()
            body = self.parse_block()
            return For(tok.line, tok.col, pattern, iter_expr, body)

        # block expression
        if tok.type == TokenType.LBRACE:
            return self.parse_block()

        # closure
        if tok.type in (TokenType.PIPE, TokenType.OROR):
            return self.parse_closure()

        # async block
        if tok.type == TokenType.ASYNC:
            self.advance()
            if self.check(TokenType.LBRACE):
                return self.parse_block()
            self.error("expected block after async")

        # prefix await
        if tok.type == TokenType.AWAIT:
            self.advance()
            operand = self.parse_expr(PREFIX_PREC)
            return Await(tok.line, tok.col, operand)

        # macro: name ! ( ... ) / [ ... ] / { ... }
        if tok.type == TokenType.IDENT and self.peek(1).type == TokenType.BANG:
            name = tok.value
            self.advance()
            self.advance()  # '!'
            if self.check(TokenType.LPAREN):
                self.advance()
                args: list[Expr] = []
                while not self.check(TokenType.RPAREN):
                    args.append(self.parse_expr())
                    if not self.eat(TokenType.COMMA):
                        break
                self.expect(TokenType.RPAREN)
                return MacroCall(tok.line, tok.col, name, args)
            if self.check(TokenType.LBRACKET):
                self.advance()
                start = self.i
                while not self.check(TokenType.RBRACKET) and not self.at_end():
                    self.advance()
                raw = "".join(t.value for t in self.tokens[start:self.i])
                self.expect(TokenType.RBRACKET)
                return MacroCall(tok.line, tok.col, name, raw)
            if self.check(TokenType.LBRACE):
                self.advance()
                start = self.i
                while not self.check(TokenType.RBRACE) and not self.at_end():
                    self.advance()
                raw = "".join(t.value for t in self.tokens[start:self.i])
                self.expect(TokenType.RBRACE)
                return MacroCall(tok.line, tok.col, name, raw)
            self.error("expected ( [ or { after macro name")

        return self.parse_primary()

    # ---- primary -----------------------------------------------------------

    def parse_primary(self) -> Expr:
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            return self._make_number(tok)

        if tok.type == TokenType.STRING:
            self.advance()
            return LitStr(tok.line, tok.col, tok.value)

        if tok.type == TokenType.CHAR:
            self.advance()
            return LitChar(tok.line, tok.col, tok.value)

        if tok.type == TokenType.TRUE:
            self.advance()
            return LitBool(tok.line, tok.col, True)

        if tok.type == TokenType.FALSE:
            self.advance()
            return LitBool(tok.line, tok.col, False)

        # tuple / paren
        if tok.type == TokenType.LPAREN:
            self.advance()
            if self.check(TokenType.RPAREN):
                self.advance()
                return TupleExpr(tok.line, tok.col, [])
            first = self.parse_expr()
            if self.check(TokenType.COMMA):
                elems = [first]
                while self.eat(TokenType.COMMA):
                    if self.check(TokenType.RPAREN):
                        break
                    elems.append(self.parse_expr())
                self.expect(TokenType.RPAREN)
                return TupleExpr(tok.line, tok.col, elems)
            self.expect(TokenType.RPAREN)
            return Paren(tok.line, tok.col, first)

        # array
        if tok.type == TokenType.LBRACKET:
            self.advance()
            if self.check(TokenType.RBRACKET):
                self.advance()
                return ArrayExpr(tok.line, tok.col, [])
            first = self.parse_expr()
            if self.eat(TokenType.SEMI):
                count = self.parse_expr()
                self.expect(TokenType.RBRACKET)
                return RepeatArray(tok.line, tok.col, first, count)
            elems = [first]
            while self.eat(TokenType.COMMA):
                if self.check(TokenType.RBRACKET):
                    break
                elems.append(self.parse_expr())
            self.expect(TokenType.RBRACKET)
            return ArrayExpr(tok.line, tok.col, elems)

        # ident / path / struct literal
        if tok.type == TokenType.IDENT:
            self.advance()
            segments = [tok.value]
            while self.check(TokenType.COLONCOLON) and self.peek(1).type == TokenType.IDENT:
                self.advance()
                segments.append(self.advance().value)
            if self.check(TokenType.LBRACE) and self._looks_like_struct_literal():
                return self.parse_struct_literal(tok, segments)
            if len(segments) > 1:
                return Path(tok.line, tok.col, segments)
            return Ident(tok.line, tok.col, segments[0])

        if tok.type == TokenType.SELF:
            self.advance()
            return Ident(tok.line, tok.col, "self")

        self.error(f"expected expression, got {tok.type.name} {tok.value!r}")
        raise AssertionError("unreachable")

    def _looks_like_struct_literal(self) -> bool:
        if not self.check(TokenType.LBRACE):
            return False
        nxt = self.peek(1)
        if nxt.type == TokenType.DOTDOT or nxt.type == TokenType.RBRACE:
            return True
        if nxt.type == TokenType.IDENT:
            after = self.peek(2)
            return after.type in (TokenType.COLON, TokenType.COMMA, TokenType.RBRACE)
        return False

    def parse_struct_literal(self, tok: Token, segments: list[str]) -> StructLiteral:
        self.expect(TokenType.LBRACE)
        fields: list[tuple[str, Expr]] = []
        rest: Expr | None = None
        while not self.check(TokenType.RBRACE):
            if self.check(TokenType.DOTDOT):
                self.advance()
                rest = self.parse_expr()
                break
            fname = self.expect(TokenType.IDENT).value
            if self.eat(TokenType.COLON):
                value = self.parse_expr()
            else:
                value = Ident(tok.line, tok.col, fname)
            fields.append((fname, value))
            if not self.eat(TokenType.COMMA):
                break
        self.expect(TokenType.RBRACE)
        return StructLiteral(tok.line, tok.col, "::".join(segments), fields, rest)

    def _make_number(self, tok: Token) -> Expr:
        raw = tok.value
        clean = raw.replace("_", "")
        low = clean.lower()
        if low.startswith("0x"):
            return LitInt(tok.line, tok.col, int(clean, 16), raw)
        if low.startswith("0b"):
            return LitInt(tok.line, tok.col, int(clean, 2), raw)
        if low.startswith("0o"):
            return LitInt(tok.line, tok.col, int(clean, 8), raw)
        if "." in clean or "e" in low:
            return LitFloat(tok.line, tok.col, float(clean), raw)
        return LitInt(tok.line, tok.col, int(clean), raw)

    # ---- control flow ------------------------------------------------------

    def parse_if(self) -> IfExpr:
        tok = self.expect(TokenType.IF)
        cond = self.parse_expr()
        then_block = self.parse_block()
        else_branch: Expr | Block | IfExpr | None = None
        if self.eat(TokenType.ELSE):
            if self.check(TokenType.IF):
                else_branch = self.parse_if()
            else:
                else_branch = self.parse_block()
        return IfExpr(tok.line, tok.col, cond, then_block, else_branch)

    def parse_match(self) -> MatchExpr:
        tok = self.expect(TokenType.MATCH)
        scrutinee = self.parse_expr()
        self.expect(TokenType.LBRACE)
        arms: list[MatchArm] = []
        while not self.check(TokenType.RBRACE):
            arm_tok = self.peek()
            pattern = self.parse_pattern()
            self._validate_rest_pattern(pattern)
            guard = None
            if self.eat(TokenType.IF):
                guard = self.parse_expr()
            self.expect(TokenType.FATARROW)
            body = self.parse_expr()
            arms.append(MatchArm(arm_tok.line, arm_tok.col, pattern, guard, body))
            if self.eat(TokenType.COMMA):
                continue
            if self.check(TokenType.RBRACE):
                break
            if isinstance(body, Block):
                continue
            self.error("expected ',' or '}' after match arm")
        self.expect(TokenType.RBRACE)
        return MatchExpr(tok.line, tok.col, scrutinee, arms)

    def _validate_rest_pattern(
        self, pattern: Pattern, *, allowed: bool = False,
    ) -> None:
        if isinstance(pattern, PatRest):
            if not allowed:
                raise ParseError(
                    "rest pattern is only valid in a slice or struct",
                    pattern.line,
                    pattern.col,
                )
            return
        if isinstance(pattern, PatSlice):
            for elem in pattern.elems:
                self._validate_rest_pattern(elem, allowed=True)
            return
        if isinstance(pattern, PatBinding):
            self._validate_rest_pattern(pattern.inner, allowed=allowed)
            return
        if isinstance(pattern, (PatTuple, PatOr, PatEnum)):
            children = (
                pattern.elems if isinstance(pattern, PatTuple)
                else pattern.alts if isinstance(pattern, PatOr)
                else pattern.args
            )
            for child in children:
                self._validate_rest_pattern(child)
            return
        if isinstance(pattern, PatStruct):
            for _name, child in pattern.fields:
                self._validate_rest_pattern(child)



    def parse_closure(self) -> Closure:
        tok = self.peek()
        params: list[tuple[Pattern, Type | None]] = []
        if self.check(TokenType.OROR):
            self.advance()
        else:
            self.expect(TokenType.PIPE)
            while not self.check(TokenType.PIPE):
                p = self.parse_pattern_atom()
                ty = None
                if self.eat(TokenType.COLON):
                    ty = self.parse_type()
                params.append((p, ty))
                if not self.eat(TokenType.COMMA):
                    break
            self.expect(TokenType.PIPE)
        ret = None
        if self.eat(TokenType.ARROW):
            ret = self.parse_type()
        body: Expr = self.parse_block() if self.check(TokenType.LBRACE) else self.parse_expr()
        return Closure(tok.line, tok.col, params, body, ret)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def parse(tokens: list[Token]) -> Program:
    return Parser(tokens).parse_program()
