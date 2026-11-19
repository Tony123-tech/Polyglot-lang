"""Zig extension package for Polyglot-lang."""

from __future__ import annotations

from typing import Any

from ast_nodes import Program
from zig.backend import ZigBackend, ZigCodegen, ZigCompileError
from zig.features import int_bounds, saturating_op, wrap_arbitrary_int
from zig.lowering import lower_zig_ast


def inject_zig_runtime(ns: dict[str, Any]) -> None:
    """Inject Zig runtime helpers into a Python backend namespace."""
    ns["_zig_sat_add"] = lambda a, b: saturating_op(a, b, "+|")
    ns["_zig_sat_sub"] = lambda a, b: saturating_op(a, b, "-|")
    ns["_zig_sat_mul"] = lambda a, b: saturating_op(a, b, "*|")


def compile_zig(program: Program, *, trusted: bool = True) -> dict[str, Any]:
    return ZigBackend(trusted=trusted).compile(program)


__all__ = [
    "ZigBackend",
    "ZigCodegen",
    "ZigCompileError",
    "compile_zig",
    "inject_zig_runtime",
    "int_bounds",
    "lower_zig_ast",
    "saturating_op",
    "wrap_arbitrary_int",
]
