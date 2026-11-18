"""Shared validation for explicitly requested but unavailable language features."""

from __future__ import annotations

from ast_nodes import FnDecl, Node, Program
from runtime.diagnostics import PillarError


def _walk(value: object):
    if isinstance(value, Node):
        yield value
        for field_name in getattr(value, "__dataclass_fields__", {}):
            if field_name in {"line", "col", "span", "source_lines"}:
                continue
            yield from _walk(getattr(value, field_name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def validate_program_features(program: Program) -> None:
    """Reject feature annotations that have no implementation in any backend."""
    for node in _walk(program):
        if not isinstance(node, FnDecl):
            continue
        for attr in node.attributes:
            if attr.name not in {"gpu", "npu"}:
                continue
            raise PillarError(
                "E007",
                f"the `#[{attr.name}]` accelerator feature is not supported",
                reason="GPU/NPU kernel offloading is not implemented",
                help_text=(
                    "remove the accelerator attribute or use the supported "
                    "Python/native CPU backend"
                ),
                note="UnsupportedAcceleratorFeature",
                line=attr.line,
                col=attr.col,
                span=attr.span,
            )


__all__ = ["validate_program_features"]
