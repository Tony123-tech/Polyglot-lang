"""Stage 1: compile the subset from source to canonical IR."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .subset import check_types, digest, emit_ir, emit_zig, parse_subset


@dataclass(frozen=True)
class Artifact:
    """Everything stage 1 produced for one source file."""

    ir: str
    digest: str
    functions: tuple[str, ...]
    source_chars: int
    zig: str = ""

    @property
    def ir_lines(self) -> int:
        return len(self.ir.splitlines())


def bootstrap_stage1_compile_subset(source: str, *, emit_zig_text: bool = False) -> Artifact:
    """Parse, type-check, and lower ``source`` to canonical IR."""
    if not source.strip():
        raise PillarError("E003", "no source to compile", help_text="pass a subset program")
    program = parse_subset(source)
    check_types(program)
    ir = emit_ir(program)
    zig = emit_zig(program) if emit_zig_text else ""
    return Artifact(
        ir=ir,
        digest=digest(ir),
        functions=tuple(function.name for function in program.functions),
        source_chars=len(source),
        zig=zig,
    )
