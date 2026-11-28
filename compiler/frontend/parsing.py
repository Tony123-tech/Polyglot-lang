"""Source-level parsing helpers for the frontend parsing frame.

Every function in this module drives the project's real :mod:`lexer` and
:mod:`parser` (or the framework's explicit-table Pratt parser) and returns
plain dictionaries/lists with exact character spans.  No AST node is
synthesised: when the grammar has no dedicated node (for example ``where``
clauses or enum discriminants) the limitation is documented and reflected in
the result rather than hidden behind a fabricated node.

Failures are reported as :class:`compiler.frontend.errors.PolyglotError`:

* ``E003`` for malformed source and API misuse,
* ``E001`` when a successfully parsed node has the wrong shape,
* ``E006`` when a required construct or token is absent.
"""

from __future__ import annotations

from typing import Any

import compiler.core.ast_nodes as ast
from api.visualizer import SourcePrinter
from compiler.core.lexer import LexError, Token, TokenType, lex
from compiler.core.parser import ParseError, Parser, parse
from compiler.frontend.errors import PolyglotError
from compiler.frontend.framework import (
    _ast_to_dict,
    _expr_dict_text,
    _matching_brace_end,
    _token_end_char_offset,
    _tokens_text,
    ast_node_span_normalize,
    parse_pattern_match_arms,
    parse_pratt_expr_with_precedence,
)

__all__ = [
    "parse_arm_guard_simple",
    "parse_array_expr",
    "parse_attributes",
    "parse_binary_expr",
    "parse_block_expr",
    "parse_call_expr",
    "parse_cast_expr",
    "parse_enum_variants",
    "parse_expression_depth",
    "parse_field_expr",
    "parse_function_signature",
    "parse_generic_args",
    "parse_if_chain",
    "parse_impl_methods",
    "parse_index_expr",
    "parse_items",
    "parse_lambda_expr",
    "parse_let_binding",
    "parse_literal_expr",
    "parse_loop_body",
    "parse_match_arms_simple",
    "parse_path_expr",
    "parse_pattern_simple",
    "parse_range_expr",
    "parse_return_expr",
    "parse_source_program",
    "parse_statement_kind",
    "parse_struct_fields",
    "parse_trait_methods",
    "parse_try_expr",
    "parse_tuple_expr",
    "parse_type_expr",
    "parse_unary_expr",
    "parse_visibility",
    "parse_where_predicates",
]

_PRINTER = SourcePrinter()

_DEPTH_OPEN: dict[TokenType, TokenType] = {
    TokenType.LPAREN: TokenType.RPAREN,
    TokenType.LBRACKET: TokenType.RBRACKET,
    TokenType.LBRACE: TokenType.RBRACE,
    TokenType.LT: TokenType.GT,
}
_DEPTH_CLOSE = frozenset(_DEPTH_OPEN.values())

_ITEM_KEYWORDS: dict[TokenType, str] = {
    TokenType.FN: "item",
    TokenType.STRUCT: "item",
    TokenType.ENUM: "item",
    TokenType.IMPL: "item",
    TokenType.TRAIT: "item",
    TokenType.USE: "item",
    TokenType.MOD: "item",
    TokenType.CONST: "item",
    TokenType.STATIC: "item",
    TokenType.TYPE: "item",
    TokenType.PUB: "item",
    TokenType.ASYNC: "item",
    TokenType.UNSAFE: "item",
    TokenType.HASH: "item",
    TokenType.LET: "let",
    TokenType.RETURN: "return",
    TokenType.BREAK: "break",
    TokenType.CONTINUE: "continue",
    TokenType.ERRDEFER: "errdefer",
    TokenType.DEFER: "defer",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _line_col_offset(source: str, line: int, col: int) -> int:
    """Translate a one-based ``(line, col)`` pair into a character offset."""
    lines = source.split("\n")
    if line < 1 or line > len(lines):
        return len(source)
    base = sum(len(part) + 1 for part in lines[: line - 1])
    return min(len(source), base + max(0, col - 1))


def _parse_failure(source: str, error: BaseException) -> PolyglotError:
    """Wrap a lexer/parser exception in an ``E003`` diagnostic."""
    line = int(getattr(error, "line", 0) or 0)
    col = int(getattr(error, "col", 0) or 0)
    start = _line_col_offset(source, line, col) if line or col else 0
    end = min(len(source), start + 1)
    return PolyglotError(
        "E003",
        f"parse failed: {error}",
        reason=f"{type(error).__name__}: {error}",
        help_text="check the Polyglot syntax at the reported location",
        line=line,
        col=col,
        span=(start, end),
    )


def _lex(source: str) -> list[Token]:
    """Lex *source*, converting lexical failures to ``PolyglotError``."""
    if not isinstance(source, str):
        raise PolyglotError(
            "E003", f"expected source string, got {type(source).__name__}"
        )
    try:
        return lex(source)
    except LexError as error:
        raise _parse_failure(source, error) from error


def _parse_program(source: str) -> ast.Program:
    """Parse a whole program with the project's real parser."""
    tokens = _lex(source)
    try:
        return parse(tokens)
    except ParseError as error:
        raise _parse_failure(source, error) from error


def _parse_with(source: str, method: str) -> tuple[Any, Parser, list[Token]]:
    """Parse one construct with *method* and return it with parser state."""
    tokens = _lex(source)
    parser = Parser(tokens)
    try:
        node = getattr(parser, method)()
    except ParseError as error:
        raise _parse_failure(source, error) from error
    return node, parser, tokens


def _full_span(source: str, tokens: list[Token]) -> list[int]:
    """Return the exact half-open character span covered by real tokens."""
    real = [token for token in tokens if token.type != TokenType.EOF]
    if not real:
        raise PolyglotError("E006", "no token found in source")
    start = real[0].pos
    end = _token_end_char_offset(source, real[-1])
    return [start, end]


def _require_end(parser: Parser, tokens: list[Token], source: str) -> None:
    """Reject trailing tokens after the construct that was requested."""
    if parser.at_end():
        return
    token = parser.peek()
    raise PolyglotError(
        "E003",
        f"unexpected trailing token {token.value!r} after parsed construct",
        line=token.line,
        col=token.col,
        span=[token.pos, _token_end_char_offset(source, token)],
    )


def _node_span(node: Any) -> list[int]:
    """Return a normalized AST node's span as a plain list."""
    span = getattr(node, "span", None)
    if not isinstance(span, (tuple, list)) or len(span) != 2:
        raise PolyglotError("E001", f"missing span on {type(node).__name__} node")
    return [int(span[0]), int(span[1])]


def _expr(source: str) -> tuple[Any, list[int]]:
    """Parse exactly one expression; return normalized node and full span."""
    node, parser, tokens = _parse_with(source, "parse_expr")
    _require_end(parser, tokens, source)
    return ast_node_span_normalize(node, source), _full_span(source, tokens)


def _statement(source: str, method: str) -> tuple[Any, list[int]]:
    """Parse exactly one statement using the named ``Parser`` method."""
    node, parser, tokens = _parse_with(source, method)
    _require_end(parser, tokens, source)
    return ast_node_span_normalize(node, source), _full_span(source, tokens)


def _single_item(source: str, expected: type, label: str) -> tuple[Any, list[int]]:
    """Parse a program containing exactly one item of the expected type."""
    program = _parse_program(source)
    if len(program.items) != 1:
        raise PolyglotError(
            "E003",
            f"expected exactly one {label}, got {len(program.items)} items",
        )
    item = program.items[0]
    if not isinstance(item, expected):
        raise PolyglotError(
            "E001",
            f"expected {expected.__name__}, got {type(item).__name__}",
        )
    return item, _full_span(source, _lex(source))


def _require(node: Any, expected: type, label: str) -> None:
    """Raise ``E001`` when a parsed node has an unexpected shape."""
    if not isinstance(node, expected):
        raise PolyglotError(
            "E001",
            f"expected {expected.__name__} ({label}), got {type(node).__name__}",
        )


def _token_span(token: Token) -> list[int]:
    """Half-open span of a single token."""
    return [token.pos, token.pos + len(token.value)]


def _group_span(group: list[Token]) -> list[int]:
    """Span covering every token in a non-empty token group."""
    if not group:
        raise PolyglotError("E006", "empty token group")
    return [group[0].pos, _group_end(group)]


def _group_end(group: list[Token]) -> int:
    """Exclusive end offset of the last token in *group*."""
    last = group[-1]
    return last.pos + len(last.value)


def _split_groups(
    source: str, tokens: list[Token], opener: TokenType, closer: TokenType
) -> list[list[Token]]:
    """Split the contents of the first ``opener ... closer`` pair on commas."""
    start = next(
        (index for index, token in enumerate(tokens) if token.type == opener), None
    )
    if start is None:
        raise PolyglotError("E006", f"missing {opener.name} delimiter")
    stack: list[TokenType] = []
    groups: list[list[Token]] = []
    current: list[Token] = []
    for token in tokens[start:]:
        if token.type in _DEPTH_OPEN:
            stack.append(_DEPTH_OPEN[token.type])
            if len(stack) > 1:
                current.append(token)
            continue
        if token.type in _DEPTH_CLOSE:
            if not stack:
                break
            stack.pop()
            if not stack:
                break
            current.append(token)
            continue
        if not stack:
            continue
        if token.type == TokenType.COMMA and len(stack) == 1:
            if current:
                groups.append(current)
            current = []
            continue
        current.append(token)
    if stack:
        raise PolyglotError(
            "E003",
            f"unterminated {opener.name} ... {closer.name} group",
            span=_token_span(tokens[start]),
        )
    if current:
        groups.append(current)
    return groups


def _split_top_level(tokens: list[Token], separator: TokenType) -> list[list[Token]]:
    """Split *tokens* on a separator that is not nested inside delimiters."""
    groups: list[list[Token]] = []
    current: list[Token] = []
    stack: list[TokenType] = []
    for token in tokens:
        if token.type in _DEPTH_OPEN:
            stack.append(_DEPTH_OPEN[token.type])
        elif token.type in _DEPTH_CLOSE and stack:
            stack.pop()
        if token.type == separator and not stack:
            if current:
                groups.append(current)
            current = []
            continue
        current.append(token)
    if current:
        groups.append(current)
    return groups


def _fn_signature(fn: ast.FnDecl) -> str:
    """Render a function's signature without its body."""
    prefix = "async " if fn.is_async else ""
    prefix += "unsafe " if fn.is_unsafe else ""
    generics = _PRINTER.generic_params(fn)
    params = ", ".join(_PRINTER.param(param) for param in fn.params)
    ret = f" -> {ast.type_display_name(fn.ret)}" if fn.ret is not None else ""
    return f"{prefix}fn {fn.name}{generics}({params}){ret}"


def _branch_span(source: str, keyword: Any, block: ast.Block) -> list[int]:
    """Span from an if/else keyword through its then/else block's ``}``."""
    start = _line_col_offset(source, keyword.line, keyword.col)
    body_start = _line_col_offset(source, block.line, block.col)
    closing = _matching_brace_end(source, body_start)
    if closing is None:
        raise PolyglotError("E003", "unterminated block in if chain")
    return [start, closing]


def _type_record(ty: ast.Type, source: str) -> dict[str, Any]:
    """Convert one real type node into the structured type record."""
    span = _node_span(ty)
    text = ast.type_display_name(ty)
    args: list[dict[str, Any]] = []
    if isinstance(ty, ast.TypeName):
        if ty.args:
            return {
                "text": text,
                "kind": "generic",
                "name": ty.name,
                "args": [_type_record(arg, source) for arg in ty.args],
                "char_span": span,
            }
        return {
            "text": text,
            "kind": "named",
            "name": ty.name,
            "args": [],
            "char_span": span,
        }
    if isinstance(ty, ast.TypeRef):
        kind, name = "ref", "&mut" if ty.mutable else "&"
        args = [_type_record(ty.inner, source)]
    elif isinstance(ty, ast.TypeSlice):
        kind, name = "slice", "slice"
        args = [_type_record(ty.inner, source)]
    elif isinstance(ty, ast.TypeArray):
        kind, name = "slice", "array"
        args = [_type_record(ty.inner, source)]
    elif isinstance(ty, ast.TypeTuple):
        kind, name = "tuple", "tuple"
        args = [_type_record(item, source) for item in ty.elems]
    elif isinstance(ty, ast.TypeFn):
        kind, name = "fn", "fn"
        args = [_type_record(param, source) for param in ty.params]
        if ty.ret is not None:
            args.append(_type_record(ty.ret, source))
    elif isinstance(ty, ast.TypeUnit):
        kind, name = "tuple", "()"
    elif isinstance(ty, (ast.TypeImplTrait, ast.TypeDynTrait)):
        kind, name = "named", text
        args = [_type_record(bound, source) for bound in ty.bounds]
    else:
        raise PolyglotError("E001", f"unsupported type node {type(ty).__name__}")
    return {
        "text": text,
        "kind": kind,
        "name": name,
        "args": args,
        "char_span": span,
    }


def _pratt(source: str) -> dict[str, Any]:
    """Run the framework Pratt parser and reject incomplete expressions."""
    data = parse_pratt_expr_with_precedence(source)
    if data.get("ast") is None:
        raise PolyglotError("E003", f"expression parse failed: {data.get('error')}")
    if data.get("remaining"):
        raise PolyglotError("E003", f"unexpected trailing tokens: {data['remaining']}")
    return data


def _expr_depth(node: dict[str, Any]) -> int:
    """Return the nesting depth of a structured Pratt expression."""
    kind = node["kind"]
    children: list[dict[str, Any]] = []
    if kind == "binary":
        children = [node["left"], node["right"]]
    elif kind == "unary":
        children = [node["operand"]]
    elif kind == "group":
        children = [node["expr"]]
    elif kind in ("call", "method"):
        children = [node.get("callee", node.get("receiver")), *node["args"]]
    elif kind in ("array", "tuple"):
        children = list(node["elems"])
    elif kind == "field":
        children = [node["receiver"]]
    elif kind == "index":
        children = [node["receiver"], node["index"]]
    elif kind == "try":
        children = [node["operand"]]
    nested = kind in {
        "group",
        "call",
        "method",
        "array",
        "tuple",
        "binary",
        "unary",
        "field",
        "index",
        "try",
    }
    child_depth = max((_expr_depth(child) for child in children), default=0)
    return (1 if nested else 0) + child_depth


# ---------------------------------------------------------------------------
# 1-12: items, types and leading syntax
# ---------------------------------------------------------------------------


def parse_source_program(source: str) -> ast.Program:
    """Parse *source* into a real :class:`ast_nodes.Program` via ``parse(lex())``."""
    return _parse_program(source)


def parse_items(source: str) -> list[dict]:
    """Return every top-level item as a structured dictionary."""
    return [_ast_to_dict(item) for item in _parse_program(source).items]


def parse_function_signature(source: str) -> dict:
    """Describe a real ``FnDecl`` signature, with or without a body."""
    fn, span = _single_item(source, ast.FnDecl, "function declaration")
    params = [
        {
            "name": _PRINTER.pattern(param.pattern),
            "type": ast.type_display_name(param.ty) if param.ty is not None else None,
        }
        for param in fn.params
    ]
    return {
        "name": fn.name,
        "params": params,
        "return_type": ast.type_display_name(fn.ret) if fn.ret is not None else None,
        "generic_params": [param.name for param in fn.generics],
        "is_async": fn.is_async,
        "is_pub": fn.is_pub,
        "char_span": span,
    }


def parse_struct_fields(source: str) -> list[dict]:
    """Return each field of a real ``StructDecl`` with its exact source span."""
    struct, _ = _single_item(source, ast.StructDecl, "struct declaration")
    groups = _split_groups(source, _lex(source), TokenType.LBRACE, TokenType.RBRACE)
    if len(groups) != len(struct.fields):
        raise PolyglotError(
            "E001",
            f"field/AST count mismatch: {len(groups)} groups, "
            f"{len(struct.fields)} fields",
        )
    fields: list[dict[str, Any]] = []
    for (name, ty, is_pub), group in zip(struct.fields, groups, strict=True):
        if name not in [token.value for token in group]:
            raise PolyglotError("E001", f"field {name!r} not found in source group")
        fields.append(
            {
                "name": name,
                "type": ast.type_display_name(ty),
                "is_pub": is_pub,
                "char_span": _group_span(group),
            }
        )
    return fields


def parse_enum_variants(source: str) -> list[dict]:
    """Return the variants of a real ``EnumDecl``.

    The current grammar has no discriminant syntax, so ``discriminant`` is
    always ``None``; named and tuple fields are reported as their real types.
    """
    enum, _ = _single_item(source, ast.EnumDecl, "enum declaration")
    return [
        {
            "name": variant.name,
            "fields": [ast.type_display_name(field) for field in variant.fields]
            if variant.fields is not None
            else None,
            "discriminant": None,
        }
        for variant in enum.variants
    ]


def parse_trait_methods(source: str) -> list[dict]:
    """Return the methods declared in a real ``TraitDecl``.

    Associated-type items are ignored because they are not methods; the
    parser does not retain their names.
    """
    trait, _ = _single_item(source, ast.TraitDecl, "trait declaration")
    methods: list[dict[str, Any]] = []
    for item in trait.items:
        if item.fn is None:
            continue
        methods.append(
            {
                "name": item.fn.name,
                "signature": _fn_signature(item.fn),
                "has_default": item.fn.body is not None,
            }
        )
    return methods


def parse_impl_methods(source: str) -> list[dict]:
    """Return every method of a real ``ImplBlock`` with its signature."""
    impl, _ = _single_item(source, ast.ImplBlock, "impl block")
    methods: list[dict[str, Any]] = []
    for item in impl.items:
        _require(item, ast.FnDecl, "impl member")
        methods.append(
            {
                "name": item.name,
                "signature": _fn_signature(item),
                "has_body": item.body is not None,
            }
        )
    return methods


def parse_type_expr(source: str) -> dict:
    """Parse one standalone type and describe its real ``Type`` node tree."""
    ty, parser, tokens = _parse_with(source, "parse_type")
    _require_end(parser, tokens, source)
    normalized = ast_node_span_normalize(ty, source)
    record = _type_record(normalized, source)
    record["char_span"] = _full_span(source, tokens)
    return record


def parse_generic_args(source: str) -> list[dict]:
    """Return the arguments inside ``<...>``; ``[]`` when there are none."""
    tokens = _lex(source)
    if not any(token.type == TokenType.LT for token in tokens):
        return []
    groups = _split_groups(source, tokens, TokenType.LT, TokenType.GT)
    return [
        {"text": _tokens_text(group), "char_span": _group_span(group)}
        for group in groups
    ]


def parse_where_predicates(source: str) -> list[dict]:
    """Extract predicates from a ``where`` clause using real lexer tokens.

    The shared parser has no ``where`` AST node, so predicates are read from
    the token stream with nested generic arguments kept intact.
    """
    tokens = _lex(source)
    start = next(
        (index for index, token in enumerate(tokens) if token.type == TokenType.WHERE),
        None,
    )
    if start is None:
        return []
    clause: list[Token] = []
    for token in tokens[start + 1 :]:
        if token.type in (TokenType.LBRACE, TokenType.SEMI, TokenType.EOF):
            break
        clause.append(token)
    predicates: list[dict[str, Any]] = []
    for group in _split_top_level(clause, TokenType.COMMA):
        if not group:
            continue
        name = group[0]
        if name.type not in (TokenType.IDENT, TokenType.LIFETIME):
            raise PolyglotError(
                "E003",
                f"expected predicate name, got {name.type.name}",
                line=name.line,
                col=name.col,
                span=_token_span(name),
            )
        colon = next(
            (
                index
                for index, token in enumerate(group)
                if token.type == TokenType.COLON
            ),
            None,
        )
        if colon is None:
            raise PolyglotError(
                "E006",
                f"missing ':' in where predicate for {name.value!r}",
                line=name.line,
                col=name.col,
                span=_token_span(name),
            )
        bounds: list[str] = []
        for bound in _split_top_level(group[colon + 1 :], TokenType.PLUS):
            if bound:
                bounds.append(_tokens_text(bound))
        if not bounds:
            raise PolyglotError(
                "E006", f"missing bounds for where predicate {name.value!r}"
            )
        predicates.append(
            {
                "name": name.value,
                "bounds": bounds,
                "char_span": _group_span(group),
            }
        )
    return predicates


def parse_attributes(source: str) -> list[dict]:
    """Return leading ``#[...]`` attributes from the real parser's ``Attr`` nodes."""
    tokens = _lex(source)
    parser = Parser(tokens)
    try:
        attrs = parser.parse_attrs()
    except ParseError as error:
        raise _parse_failure(source, error) from error
    attributes: list[dict[str, Any]] = []
    for attr in attrs:
        start, end = _node_span(attr)
        raw = source[start:end]
        opening = raw.find("(")
        args: str | None = None
        if opening >= 0:
            depth = 0
            closing: int | None = None
            for index in range(opening, len(raw)):
                if raw[index] == "(":
                    depth += 1
                elif raw[index] == ")":
                    depth -= 1
                    if depth == 0:
                        closing = index
                        break
            stop = closing if closing is not None else len(raw)
            args = raw[opening + 1 : stop].strip() or None
        attributes.append({"name": attr.name, "args": args, "char_span": [start, end]})
    return attributes


def parse_visibility(source: str) -> str:
    """Return ``'pub'`` when the first token is ``PUB``, else ``'private'``."""
    token = next((item for item in _lex(source) if item.type != TokenType.EOF), None)
    if token is None:
        raise PolyglotError("E006", "no leading token to classify")
    return "pub" if token.type == TokenType.PUB else "private"


# ---------------------------------------------------------------------------
# 13-25: statements and expressions
# ---------------------------------------------------------------------------


def parse_let_binding(source: str) -> dict:
    """Describe a real ``let`` statement including its initializer."""
    node, span = _statement(source, "parse_let")
    _require(node, ast.LetStmt, "let binding")
    name = (
        node.pattern.name
        if isinstance(node.pattern, ast.PatIdent)
        else _PRINTER.pattern(node.pattern)
    )
    return {
        "name": name,
        "mutable": node.mutable,
        "type": ast.type_display_name(node.ty) if node.ty is not None else None,
        "init": _PRINTER.expr(node.value) if node.value is not None else None,
        "char_span": span,
    }


def parse_match_arms_simple(source: str) -> list[dict]:
    """Return pattern/guard/body text for each arm of a real ``MatchExpr``."""
    try:
        data = parse_pattern_match_arms(source)
    except (LexError, ParseError) as error:
        source_text = source if isinstance(source, str) else ""
        raise _parse_failure(source_text, error) from error
    return [
        {
            "pattern": arm["pattern_text"],
            "guard": arm["guard_text"],
            "body": arm["body_text"],
            "char_span": list(arm["char_span"]),
        }
        for arm in data["arms"]
    ]


def parse_if_chain(source: str) -> list[dict]:
    """Return each branch of a real ``IfExpr`` in source order."""
    node, _ = _expr(source)
    _require(node, ast.IfExpr, "if expression")
    branches: list[dict[str, Any]] = []
    current: Any = node
    while True:
        branches.append(
            {
                "condition": _PRINTER.expr(current.cond),
                "body": _PRINTER.block(current.then_block, 0),
                "char_span": _branch_span(source, current, current.then_block),
            }
        )
        following = current.else_branch
        if isinstance(following, ast.IfExpr):
            current = following
            continue
        if isinstance(following, ast.Block):
            branches.append(
                {
                    "condition": None,
                    "body": _PRINTER.block(following, 0),
                    "char_span": _branch_span(source, following, following),
                }
            )
        elif following is not None:
            branches.append(
                {
                    "condition": None,
                    "body": _PRINTER.expr(following),
                    "char_span": _node_span(following),
                }
            )
        break
    return branches


def parse_loop_body(source: str) -> dict:
    """Describe the header and body of a real ``for``, ``while`` or ``loop``."""
    node, span = _expr(source)
    if isinstance(node, ast.For):
        kind = "for"
        header = f"for {_PRINTER.pattern(node.pattern)} in {_PRINTER.expr(node.iter)}"
    elif isinstance(node, ast.While):
        kind = "while"
        header = f"while {_PRINTER.expr(node.cond)}"
    elif isinstance(node, ast.WhileLet):
        kind = "while let"
        header = (
            f"while let {_PRINTER.pattern(node.pattern)} = {_PRINTER.expr(node.value)}"
        )
    elif isinstance(node, ast.Loop):
        kind, header = "loop", "loop"
    else:
        raise PolyglotError(
            "E001", f"expected loop expression, got {type(node).__name__}"
        )
    return {
        "kind": kind,
        "header": header,
        "body": _PRINTER.block(node.body, 0),
        "char_span": span,
    }


def parse_call_expr(source: str) -> dict:
    """Describe a real ``Call`` node, using framework Pratt text for operands."""
    node, _ = _expr(source)
    _require(node, ast.Call, "call expression")
    data = _pratt(source)
    call = data["ast"]
    if call["kind"] != "call":
        raise PolyglotError("E001", f"expected Pratt call node, got {call['kind']!r}")
    return {
        "callee": _expr_dict_text(call["callee"]),
        "args": [_expr_dict_text(arg) for arg in call["args"]],
        "char_span": list(call["char_span"]),
    }


def parse_path_expr(source: str) -> dict:
    """Return the segments and text of a real ``Path`` (or single ``Ident``)."""
    node, span = _expr(source)
    if isinstance(node, ast.Path):
        segments = list(node.segments)
    elif isinstance(node, ast.Ident):
        segments = [node.name]
    else:
        raise PolyglotError(
            "E001", f"expected path expression, got {type(node).__name__}"
        )
    return {"segments": segments, "text": "::".join(segments), "char_span": span}


def parse_range_expr(source: str) -> dict:
    """Return the bounds of a real ``Range`` expression."""
    node, span = _expr(source)
    _require(node, ast.Range, "range expression")
    return {
        "start": _PRINTER.expr(node.start) if node.start is not None else None,
        "end": _PRINTER.expr(node.end) if node.end is not None else None,
        "inclusive": node.inclusive,
        "char_span": span,
    }


def parse_literal_expr(source: str) -> dict:
    """Describe a single real literal node and its raw lexeme."""
    node, span = _expr(source)
    kinds = {
        ast.LitInt: "int",
        ast.LitFloat: "float",
        ast.LitBool: "bool",
        ast.LitStr: "string",
        ast.LitChar: "char",
    }
    kind = kinds.get(type(node))
    if kind is None:
        raise PolyglotError("E001", f"expected a literal, got {type(node).__name__}")
    raw = getattr(node, "raw", None)
    if raw is None:
        raw = source[span[0] : span[1]]
    value = str(node.value)
    if isinstance(node, ast.LitBool):
        value = "true" if node.value else "false"
    return {"kind": kind, "raw": raw, "value": value, "char_span": span}


def parse_unary_expr(source: str) -> dict:
    """Describe a real ``Unary`` expression such as ``-x`` or ``&mut x``."""
    node, span = _expr(source)
    _require(node, ast.Unary, "unary expression")
    return {
        "op": node.op,
        "operand": _PRINTER.expr(node.operand),
        "char_span": span,
    }


def parse_binary_expr(source: str) -> dict:
    """Describe a binary root using the framework Pratt precedence table."""
    node, _ = _expr(source)
    _require(node, ast.Binary, "binary expression")
    data = _pratt(source)
    binary = data["ast"]
    if binary["kind"] != "binary":
        raise PolyglotError(
            "E001", f"expected Pratt binary node, got {binary['kind']!r}"
        )
    return {
        "op": binary["op"],
        "left": _expr_dict_text(binary["left"]),
        "right": _expr_dict_text(binary["right"]),
        "precedence": int(binary["precedence"]),
        "char_span": list(binary["char_span"]),
    }


def parse_block_expr(source: str) -> dict:
    """Return the statements and optional tail of a real ``Block``."""
    node, span = _statement(source, "parse_block")
    _require(node, ast.Block, "block expression")
    return {
        "statements": [_PRINTER.stmt(stmt, 0) for stmt in node.stmts],
        "tail": _PRINTER.expr(node.tail) if node.tail is not None else None,
        "char_span": span,
    }


def parse_lambda_expr(source: str) -> dict:
    """Describe a real Polyglot closure (the grammar's lambda form)."""
    node, span = _expr(source)
    _require(node, ast.Closure, "closure expression")
    params: list[str] = []
    for pattern, ty in node.params:
        text = _PRINTER.pattern(pattern)
        if ty is not None:
            text += f": {ast.type_display_name(ty)}"
        params.append(text)
    return {"params": params, "body": _PRINTER.expr(node.body), "char_span": span}


def parse_return_expr(source: str) -> dict:
    """Describe a real ``return`` expression, with or without a value."""
    node, parser, tokens = _parse_with(source, "parse_expr")
    parser.eat(TokenType.SEMI)
    _require_end(parser, tokens, source)
    _require(node, ast.Return, "return expression")
    normalized = ast_node_span_normalize(node, source)
    return {
        "value": _PRINTER.expr(normalized.value)
        if normalized.value is not None
        else None,
        "char_span": _full_span(source, tokens),
    }


# ---------------------------------------------------------------------------
# 26-35: compound expressions and classification
# ---------------------------------------------------------------------------


def parse_tuple_expr(source: str) -> dict:
    """Return the elements of a real tuple with at least two elements."""
    node, span = _expr(source)
    _require(node, ast.TupleExpr, "tuple expression")
    if len(node.elems) < 2:
        raise PolyglotError("E003", "tuple expression requires at least two elements")
    return {
        "elements": [_PRINTER.expr(item) for item in node.elems],
        "char_span": span,
    }


def parse_array_expr(source: str) -> dict:
    """Return the elements of a real array literal."""
    node, span = _expr(source)
    _require(node, ast.ArrayExpr, "array expression")
    return {
        "elements": [_PRINTER.expr(item) for item in node.elems],
        "char_span": span,
    }


def parse_index_expr(source: str) -> dict:
    """Describe a real index expression ``receiver[index]``."""
    node, span = _expr(source)
    _require(node, ast.Index, "index expression")
    return {
        "receiver": _PRINTER.expr(node.receiver),
        "index": _PRINTER.expr(node.index),
        "char_span": span,
    }


def parse_field_expr(source: str) -> dict:
    """Describe a real field access ``receiver.field``."""
    node, span = _expr(source)
    _require(node, ast.Field, "field expression")
    return {
        "receiver": _PRINTER.expr(node.receiver),
        "field": node.name,
        "char_span": span,
    }


def parse_try_expr(source: str) -> dict:
    """Describe the operand of a real ``?`` (try) expression."""
    node, span = _expr(source)
    _require(node, ast.Try, "try expression")
    return {"operand": _PRINTER.expr(node.expr), "char_span": span}


def parse_cast_expr(source: str) -> dict:
    """Describe a real ``as`` cast and its canonical target type."""
    node, span = _expr(source)
    _require(node, ast.Cast, "cast expression")
    return {
        "expr": _PRINTER.expr(node.expr),
        "type": ast.type_display_name(node.ty),
        "char_span": span,
    }


def parse_pattern_simple(source: str) -> dict:
    """Parse one pattern and report its real node kind and canonical text."""
    node, parser, tokens = _parse_with(source, "parse_pattern")
    _require_end(parser, tokens, source)
    normalized = ast_node_span_normalize(node, source)
    return {
        "kind": type(node).__name__,
        "text": _PRINTER.pattern(normalized),
        "char_span": _full_span(source, tokens),
    }


def parse_arm_guard_simple(source: str) -> dict | None:
    """Parse a leading ``if expr`` guard; return ``None`` when absent."""
    tokens = _lex(source)
    real = [token for token in tokens if token.type != TokenType.EOF]
    if not real or real[0].type != TokenType.IF:
        return None
    remaining = tokens[1:]
    parser = Parser(remaining)
    try:
        expr = parser.parse_expr()
    except ParseError as error:
        raise _parse_failure(source, error) from error
    _require_end(parser, remaining, source)
    normalized = ast_node_span_normalize(expr, source)
    return {
        "condition": _PRINTER.expr(normalized),
        "char_span": _full_span(source, remaining),
    }


def parse_statement_kind(source: str) -> str:
    """Classify one statement from its leading token and real parse shape."""
    tokens = _lex(source)
    first = next((item for item in tokens if item.type != TokenType.EOF), None)
    if first is None:
        return "unknown"
    kind = _ITEM_KEYWORDS.get(first.type, "expr")
    method = {
        "let": "parse_let",
        "errdefer": "parse_errdefer",
        "item": "parse_item",
    }.get(kind, "parse_expr")
    parser = Parser(tokens)
    try:
        getattr(parser, method)()
    except ParseError:
        return "unknown"
    parser.eat(TokenType.SEMI)
    if not parser.at_end():
        return "unknown"
    return kind


def parse_expression_depth(source: str) -> int:
    """Return the maximum nesting depth of a Pratt-parsed expression."""
    tokens = _lex(source)
    if not any(token.type != TokenType.EOF for token in tokens):
        raise PolyglotError("E003", "empty expression")
    data = _pratt(source)
    return _expr_depth(data["ast"])
