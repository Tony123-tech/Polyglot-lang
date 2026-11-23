"""One-way include expansion for polyglot-lang (M6)."""

from __future__ import annotations

from pathlib import Path

from compiler.core.ast_nodes import IncludeDecl, Item, Program
from compiler.core.lexer import lex
from compiler.core.parser import parse


class IncludeError(Exception):
    """Base class for include-related errors with five-line formatting."""

    code = ""

    def __init__(
        self,
        message: str,
        reason: str,
        help_text: str,
        note: str,
        line: int,
        col: int,
        source_path: Path | None = None,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.message = message
        self.reason = reason
        self.help_text = help_text
        self.note = note
        self.line = line
        self.col = col
        self.source_path = source_path
        self.span = span
        super().__init__(self.format())

    def format(self) -> str:
        prefix = f"error[{self.code}]: " if self.code else "error: "
        loc = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            loc += f" (span {self.span[0]}..{self.span[1]})"
        if self.source_path is not None:
            loc += f" in {self.source_path}"
        return (
            f"{prefix}{self.message}\n"
            f"  reason: {self.reason}\n"
            f"  help: {self.help_text}\n"
            f"  note: {self.note}\n"
            f"  {loc}"
        )

    def __str__(self) -> str:
        return self.format()


class IncludeCycleError(IncludeError):
    code = "E005"


class IncludeNotFoundError(IncludeError):
    code = "E006"


STDLIB_DIR = Path(__file__).resolve().parents[2] / "stdlib"
_STD_PREFIX = "std:"


def is_std_include(raw: str) -> bool:
    """Return True when an include path names a built-in standard library module."""
    return raw.startswith(_STD_PREFIX)


def resolve_std_include(raw: str) -> Path:
    """Resolve ``std:<name>`` to a file inside the bundled ``stdlib/`` directory.

    The name is always interpreted relative to :data:`STDLIB_DIR` and must stay
    inside it. Traversal, absolute paths, and missing modules raise
    :class:`IncludeNotFoundError` (E006).
    """
    name = raw[len(_STD_PREFIX) :]
    root = STDLIB_DIR.resolve()

    def reject(reason: str, help_text: str) -> IncludeNotFoundError:
        return IncludeNotFoundError(
            f"std include not found: {raw}",
            reason,
            help_text,
            f"stdlib root: {root}",
            1,
            1,
        )

    if not name:
        raise reject(
            "empty std module name",
            'write an include such as include!("std:math");',
        )
    if "\x00" in name:
        raise reject(
            "std module name contains a NUL byte",
            "use a module name such as math or numpy",
        )

    relative = Path(name)
    if relative.is_absolute():
        raise reject(
            "std include paths are relative to the bundled stdlib directory",
            "remove the leading path separator",
        )

    target = relative if name.endswith(".poly") else Path(f"{name}.poly")
    candidate = (root / target).resolve()
    if not candidate.is_relative_to(root):
        raise reject(
            "std include escapes the bundled stdlib directory",
            "use a plain module name such as math or numpy",
        )
    if not candidate.is_file():
        raise reject(
            f"missing std module: {name}",
            "check the module name; stdlib modules live under stdlib/",
        )
    return candidate


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _wrap_include_error(
    error: IncludeError,
    item: IncludeDecl,
    source: Path,
) -> IncludeError:
    cls = type(error)
    return cls(
        error.message,
        error.reason,
        error.help_text,
        error.note,
        item.line,
        item.col,
        source,
        span=getattr(item, "span", None),
    )


def _resolve_include_path(item: IncludeDecl, abs_path: Path) -> tuple[Path, bool]:
    """Return the resolved include target and whether it is a std module."""
    if is_std_include(item.path):
        return resolve_std_include(item.path), True
    include_path = Path(item.path)
    if not include_path.is_absolute():
        include_path = abs_path.parent / include_path
    return include_path, False


def load_program(
    path: str | Path,
    *,
    _stack: list[Path] | None = None,
    _std_included: set[Path] | None = None,
) -> Program:
    """Parse and expand includes starting from a file path."""
    abs_path = Path(path).expanduser().resolve()
    stack = list(_stack or [])
    std_included = set() if _std_included is None else _std_included
    for i, seen in enumerate(stack):
        if seen == abs_path:
            chain = " -> ".join(str(p) for p in stack[i:] + [abs_path])
            source = stack[-1] if stack else abs_path
            raise IncludeCycleError(
                f"cyclic include detected: {chain}",
                f"include cycle: {chain}",
                "break the include cycle so each file is included at most once",
                f"include chain: {chain}",
                1,
                1,
                source,
            )

    if not abs_path.exists() or not abs_path.is_file():
        parent = stack[-1] if stack else None
        reason = (
            f"missing include target: {abs_path}"
            if not abs_path.exists()
            else f"include target is not a regular file: {abs_path}"
        )
        raise IncludeNotFoundError(
            f"include file not found: {abs_path}"
            if not abs_path.exists()
            else f"include target is not a file: {abs_path}",
            reason,
            "check that the include path exists, is a regular file, and is readable",
            f"resolved path: {abs_path}",
            1,
            1,
            parent,
        )

    try:
        source = _read_source(abs_path)
    except OSError as error:
        parent = stack[-1] if stack else None
        raise IncludeNotFoundError(
            f"include file cannot be read: {abs_path}",
            f"unreadable include target: {error}",
            "check that the include path exists, is a regular file, and is readable",
            f"resolved path: {abs_path}",
            1,
            1,
            parent,
        ) from error
    program = parse(lex(source))
    expanded_items: list[Item] = []
    merged_lines = dict(program.source_lines)
    next_stack = stack + [abs_path]

    for item in program.items:
        if isinstance(item, IncludeDecl):
            try:
                include_path, is_std = _resolve_include_path(item, abs_path)
            except IncludeNotFoundError as e:
                raise _wrap_include_error(e, item, abs_path) from e
            if is_std and include_path in std_included:
                continue
            try:
                included = load_program(
                    include_path,
                    _stack=next_stack,
                    _std_included=std_included,
                )
            except IncludeCycleError as e:
                # Ensure location/chain includes the include site if needed.
                if (
                    e.line == 1
                    and e.col == 1
                    and e.source_path in (None, abs_path, Path(include_path).resolve())
                ):
                    raise _wrap_include_error(e, item, abs_path) from e
                raise
            except IncludeNotFoundError as e:
                raise _wrap_include_error(e, item, abs_path) from e
            if is_std:
                std_included.add(include_path)
            expanded_items.extend(included.items)
            merged_lines.update(included.source_lines)
        else:
            expanded_items.append(item)

    return Program(
        program.line, program.col, expanded_items, merged_lines, span=program.span
    )


def expand_includes(
    program: Program,
    source_path: str | Path,
    *,
    _std_included: set[Path] | None = None,
) -> Program:
    """Expand IncludeDecl items in an already-parsed program."""
    abs_path = Path(source_path).expanduser().resolve()
    expanded_items: list[Item] = []
    merged_lines = dict(program.source_lines)
    stack = [abs_path]
    std_included = set() if _std_included is None else _std_included
    for item in program.items:
        if isinstance(item, IncludeDecl):
            try:
                include_path, is_std = _resolve_include_path(item, abs_path)
            except IncludeNotFoundError as e:
                raise _wrap_include_error(e, item, abs_path) from e
            if is_std and include_path in std_included:
                continue
            try:
                included = load_program(
                    include_path,
                    _stack=stack,
                    _std_included=std_included,
                )
            except IncludeCycleError as e:
                raise _wrap_include_error(e, item, abs_path) from e
            except IncludeNotFoundError as e:
                raise _wrap_include_error(e, item, abs_path) from e
            if is_std:
                std_included.add(include_path)
            expanded_items.extend(included.items)
            merged_lines.update(included.source_lines)
        else:
            expanded_items.append(item)
    return Program(
        program.line, program.col, expanded_items, merged_lines, span=program.span
    )
