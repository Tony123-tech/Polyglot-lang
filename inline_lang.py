"""Decorator entry point for inline Rust-like programs.

Reads the decorated function's source, converts the Python ``def`` header into
a Polyglot ``fn`` item, and compiles the result through the configured backend.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from ast_nodes import FnDecl
from backends import get_backend
from backends.python.translator import TranslateError
from backends.rust.compiler import RustCompileError
from lexer import lex
from parser import ParseError, parse


class InlineLangError(RuntimeError):
    """Raised when inline source cannot be extracted or converted."""


def _strip_decorators(lines: list[str]) -> list[str]:
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        if stripped.startswith("@") or not stripped:
            index += 1
            continue
        break
    return lines[index:]


def _format_params(args: ast.arguments) -> str:
    parts: list[str] = []
    for arg in args.posonlyargs + args.args:
        if arg.annotation is None:
            raise InlineLangError(
                f"inline_lang requires an annotation for parameter {arg.arg!r}"
            )
        parts.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
    return ", ".join(parts)


def _format_return(returns: ast.expr | None) -> str:
    if returns is None:
        return ""
    return f" -> {ast.unparse(returns)}"


def _ensure_return_semicolons(body: list[str]) -> list[str]:
    """Append a semicolon to ``return`` statements that lack one."""
    result: list[str] = []
    for line in body:
        stripped = line.strip()
        if (
            stripped.startswith("return")
            and not stripped.endswith(";")
            and not stripped.endswith("{")
        ):
            line = line.rstrip() + ";"
        result.append(line)
    return result


def _python_def_to_rust(source: str) -> tuple[str, int]:
    """Convert a decorated Python ``def`` into a Polyglot ``fn`` item.

    Returns the converted source and the one-based line number of the
    original ``def`` header within the provided source, so callers can map
    Polyglot diagnostics back to the Python file.
    """
    dedented = textwrap.dedent(source)
    lines = dedented.splitlines()
    lines = _strip_decorators(lines)
    if not lines:
        raise InlineLangError("inline_lang requires a function definition")

    tree = ast.parse("\n".join(lines))
    fn_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(fn_nodes) != 1:
        raise InlineLangError(
            "inline_lang requires exactly one function definition"
        )
    fn = fn_nodes[0]

    keyword = "async fn" if isinstance(fn, ast.AsyncFunctionDef) else "fn"
    params = _format_params(fn.args)
    returns = _format_return(fn.returns)
    header = f"{keyword} {fn.name}({params}){returns} {{"

    body_lines = lines[fn.body[0].lineno - 1 :]
    body_lines = textwrap.dedent("\n".join(body_lines)).splitlines()
    body_lines = _ensure_return_semicolons(body_lines)

    def_index = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith(("def ", "async def "))
    )

    converted = "\n".join([header, *body_lines, "}"])
    return converted, def_index + 1


def _free_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    bound = {arg.arg for arg in fn.args.posonlyargs + fn.args.args}
    used: set[str] = set()
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return used - bound


def _inline_line(exc: Exception, line_offset: int) -> int | None:
    if isinstance(exc, (ParseError, TranslateError)):
        return exc.line + line_offset if exc.line else None
    if isinstance(exc, RustCompileError):
        return (
            exc.source_line + line_offset if exc.source_line else None
        )
    return None


def _bind(program: Any, namespace: dict[str, Any]) -> Any:
    fn_items = [item for item in program.items if isinstance(item, FnDecl)]
    if len(program.items) == 1 and len(fn_items) == 1:
        return namespace[fn_items[0].name]
    return namespace


def compile_inline_source(
    source: str,
    *,
    backend: str = "python",
    trusted: bool = False,
    line_offset: int = 0,
    extra_globals: dict[str, Any] | None = None,
) -> Any:
    """Compile a Polyglot source string as an inline program.

    Errors from parsing or backend compilation are wrapped in
    ``InlineLangError`` with the original exception as ``__cause__``.
    """
    try:
        program = parse(lex(source))
        namespace = get_backend(backend, trusted=trusted).compile(program)
        if extra_globals:
            namespace.update(extra_globals)
    except Exception as exc:
        error = InlineLangError("inline_lang failed to compile source")
        error.inline_line = _inline_line(exc, line_offset)
        raise error from exc
    return _bind(program, namespace)


def inline_lang(
    lang: str,
    *,
    backend: str = "python",
    trusted: bool = False,
    extra_globals: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Compile the decorated function body as an inline Polyglot program."""

    def decorator(fn: Callable[..., Any]) -> Any:
        if lang != "rust":
            raise InlineLangError(f"unsupported inline language: {lang!r}")

        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError) as exc:
            raise InlineLangError(
                "inline_lang requires inspectable source"
            ) from exc

        rust_source, def_line = _python_def_to_rust(source)

        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError as exc:
            raise InlineLangError("inline_lang requires valid Python") from exc
        fn_node = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        free = _free_names(fn_node)

        closure = inspect.getclosurevars(fn)
        captured: dict[str, Any] = {}
        captured.update(closure.globals)
        captured.update(closure.nonlocals)
        if extra_globals:
            captured.update(extra_globals)

        if free and backend == "native":
            raise InlineLangError(
                "inline_lang: native backend does not support closures yet; "
                'use backend="python" or pass values as parameters'
            )

        try:
            return compile_inline_source(
                rust_source,
                backend=backend,
                trusted=trusted,
                line_offset=def_line - 1,
                extra_globals=captured if backend == "python" else None,
            )
        except InlineLangError as exc:
            wrapped = InlineLangError(
                f"inline_lang failed to compile {fn.__name__!r}"
            )
            wrapped.inline_line = getattr(exc, "inline_line", None)
            raise wrapped from exc

    return decorator
