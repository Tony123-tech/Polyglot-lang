"""Shared validation for explicitly requested but unavailable language features."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from compiler.analysis.flow_diagnostics import warn_unreachable_after_return
from compiler.core.ast_nodes import FnDecl, Node, Program
from runtime.diagnostics import PillarError

# Pure source metadata: it never contains child nodes, and `source_lines` can
# be large enough that walking it would be wasteful.
_SKIP_FIELDS = frozenset({"line", "col", "span", "source_lines"})


def _walk(value: object, seen: set[int] | None = None) -> Iterator[Node]:
    """Yield every :class:`Node` reachable from ``value`` exactly once.

    ``Node`` dataclasses are unhashable and callers may hand us a DAG (shared
    subtrees) or even a cyclic graph, so identity tracking is explicit.  Only
    dataclass fields and container values are traversed; scalar metadata and
    ``None`` are ignored.
    """
    if seen is None:
        seen = set()
    if isinstance(value, Node):
        if id(value) in seen:
            return
        seen.add(id(value))
        yield value
        for field_name in getattr(value, "__dataclass_fields__", {}):
            if field_name in _SKIP_FIELDS:
                continue
            yield from _walk(getattr(value, field_name), seen)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, seen)


def validate_program_features(program: Program) -> None:
    """Reject unavailable feature annotations and flag dead control flow."""
    warn_unreachable_after_return(program)
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
