"""One-way include expansion for polyglot-lang (M6)."""

from __future__ import annotations

from pathlib import Path

from ast_nodes import IncludeDecl, Item, Program
from lexer import lex
from parser import parse


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
    ) -> None:
        self.message = message
        self.reason = reason
        self.help_text = help_text
        self.note = note
        self.line = line
        self.col = col
        self.source_path = source_path
        super().__init__(self.format())

    def format(self) -> str:
        prefix = f"error[{self.code}]: " if self.code else "error: "
        loc = f"at: line {self.line}, column {self.col}"
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


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_program(path: str | Path, *, _stack: list[Path] | None = None) -> Program:
    """Parse and expand includes starting from a file path."""
    abs_path = Path(path).expanduser().resolve()
    stack = list(_stack or [])
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

    if not abs_path.exists():
        parent = stack[-1] if stack else None
        raise IncludeNotFoundError(
            f"include file not found: {abs_path}",
            f"missing include target: {abs_path}",
            "check that the include path exists and is readable",
            f"resolved path: {abs_path}",
            1,
            1,
            parent,
        )

    source = _read_source(abs_path)
    program = parse(lex(source))
    expanded_items: list[Item] = []
    merged_lines = dict(program.source_lines)
    next_stack = stack + [abs_path]

    for item in program.items:
        if isinstance(item, IncludeDecl):
            include_path = Path(item.path)
            if not include_path.is_absolute():
                include_path = abs_path.parent / include_path
            try:
                included = load_program(include_path, _stack=next_stack)
            except IncludeCycleError as e:
                # Ensure location/chain includes the include site if needed.
                if e.line == 1 and e.col == 1 and e.source_path in (None, abs_path, Path(include_path).resolve()):
                    raise IncludeCycleError(
                        e.message,
                        e.reason,
                        e.help_text,
                        e.note,
                        item.line,
                        item.col,
                        abs_path,
                    ) from e
                raise
            except IncludeNotFoundError as e:
                raise IncludeNotFoundError(
                    e.message,
                    e.reason,
                    e.help_text,
                    e.note,
                    item.line,
                    item.col,
                    abs_path,
                ) from e
            expanded_items.extend(included.items)
            merged_lines.update(included.source_lines)
        else:
            expanded_items.append(item)

    return Program(program.line, program.col, expanded_items, merged_lines, span=program.span)


def expand_includes(program: Program, source_path: str | Path) -> Program:
    """Expand IncludeDecl items in an already-parsed program."""
    abs_path = Path(source_path).expanduser().resolve()
    expanded_items: list[Item] = []
    merged_lines = dict(program.source_lines)
    stack = [abs_path]
    for item in program.items:
        if isinstance(item, IncludeDecl):
            include_path = Path(item.path)
            if not include_path.is_absolute():
                include_path = abs_path.parent / include_path
            try:
                included = load_program(include_path, _stack=stack)
            except IncludeCycleError as e:
                raise IncludeCycleError(
                    e.message,
                    e.reason,
                    e.help_text,
                    e.note,
                    item.line,
                    item.col,
                    abs_path,
                ) from e
            except IncludeNotFoundError as e:
                raise IncludeNotFoundError(
                    e.message,
                    e.reason,
                    e.help_text,
                    e.note,
                    item.line,
                    item.col,
                    abs_path,
                ) from e
            expanded_items.extend(included.items)
            merged_lines.update(included.source_lines)
        else:
            expanded_items.append(item)
    return Program(program.line, program.col, expanded_items, merged_lines, span=program.span)
