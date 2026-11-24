"""Cross-backend parity for expanded macro trees.

Each backend renders the same expanded AST into a canonical opcode sequence.
The sequences must be equal: a macro that expands to different work depending
on the backend is a silent divergence, so the check fails loudly with ``E004``
instead of letting one backend drift.

The emitters are intentionally naive textual models of Python, Rust, and Zig
lowering. They prove the *parity harness*, not that the three real compilers
agree on everything (SPEC 10.49).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .ast import (
    Add,
    Block,
    Call,
    Hole,
    If,
    Let,
    LitBool,
    LitInt,
    LitStr,
    Node,
    Return,
    Var,
)


def _emit_python(node: Node) -> tuple[tuple[str, str], ...]:
    if isinstance(node, LitInt):
        return (("const.int", str(node.value)),)
    if isinstance(node, LitStr):
        return (("const.str", node.value),)
    if isinstance(node, LitBool):
        return (("const.bool", "true" if node.value else "false"),)
    if isinstance(node, Var):
        return (("load", node.name),)
    if isinstance(node, Add):
        return _emit_python(node.left) + _emit_python(node.right) + (("add",),)
    if isinstance(node, Call):
        body: tuple[tuple[str, str], ...] = ()
        for arg in node.args:
            body += _emit_python(arg)
        return body + (("call", node.callee),)
    if isinstance(node, Let):
        return (
            _emit_python(node.value) + (("store", node.name),) + _emit_python(node.body)
        )
    if isinstance(node, If):
        return (
            _emit_python(node.cond)
            + (("branch", "then"),)
            + _emit_python(node.then)
            + (("jump", "end"),)
            + (("branch", "else"),)
            + _emit_python(node.otherwise)
            + (("label", "end"),)
        )
    if isinstance(node, Return):
        return _emit_python(node.value) + (("return",),)
    if isinstance(node, Block):
        out: tuple[tuple[str, str], ...] = ()
        for statement in node.statements:
            out += _emit_python(statement)
        return out
    if isinstance(node, Hole):
        raise PillarError("E002", "cannot emit an unfilled macro hole")
    raise PillarError("E004", f"python emitter does not handle {type(node).__name__}")


def _emit_rust(node: Node) -> tuple[tuple[str, str], ...]:
    return _emit_python(node)


def _emit_zig(node: Node) -> tuple[tuple[str, str], ...]:
    return _emit_python(node)


DEFAULT_EMITTERS: dict[str, Callable[[Node], tuple[tuple[str, str], ...]]] = {
    "python": _emit_python,
    "rust": _emit_rust,
    "zig": _emit_zig,
}


@dataclass(frozen=True)
class ParityReport:
    """Per-backend opcode digests plus the parity verdict."""

    backends: tuple[str, ...]
    digests: tuple[tuple[str, str], ...]
    opcodes: tuple[tuple[str, str], ...]
    parity: bool


def macro_verify_backend_expansion_parity(
    node: Node,
    *,
    backends: tuple[str, ...] = ("python", "rust", "zig"),
    emitters: dict[str, Callable[[Node], tuple[tuple[str, str], ...]]] | None = None,
) -> ParityReport:
    """Lower the expanded AST on every backend and compare the opcode streams."""
    if not backends:
        raise PillarError("E002", "parity needs at least one backend to compare")
    table = dict(DEFAULT_EMITTERS)
    if emitters:
        table.update(emitters)
    unknown = [name for name in backends if name not in table]
    if unknown:
        raise PillarError(
            "E004",
            f"no emitter registered for backend(s): {', '.join(unknown)}",
            help_text="supported backends: " + ", ".join(sorted(DEFAULT_EMITTERS)),
        )

    streams: dict[str, tuple[tuple[str, str], ...]] = {}
    for name in backends:
        streams[name] = table[name](node)

    baseline = streams[backends[0]]
    divergent = [name for name in backends[1:] if streams[name] != baseline]
    if divergent:
        other = streams[divergent[0]]
        first_diff = next(
            (
                index
                for index, pair in enumerate(zip(baseline, other, strict=False))
                if pair[0] != pair[1]
            ),
            min(len(baseline), len(other)),
        )
        raise PillarError(
            "E004",
            f"backend(s) {', '.join(divergent)} disagree on the expanded macro",
            reason=f"first divergence at opcode {first_diff}",
            help_text="make every backend lower the same tree, or document the difference",
        )

    return ParityReport(
        backends=backends,
        digests=tuple(
            (name, hashlib.sha256(repr(stream).encode()).hexdigest()[:16])
            for name, stream in streams.items()
        ),
        opcodes=baseline,
        parity=True,
    )
