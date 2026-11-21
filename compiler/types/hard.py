"""Hard generic/type-system edge cases (Frame C, type-system half).

These helpers add bounded expansion and coherence analysis on top of the
existing :mod:`compiler.types.framework` passes.  They are deterministic and
do not compile native code, so they can run on every platform.

Diagnostic map: ``E001`` associated-type/coherence mismatch, ``E002``
specialization budget or higher-rank bound violation, ``E005`` illegal
recursive dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from runtime.diagnostics import PillarError

Span = tuple[int, int] | None


def _fail(
    code: str, message: str, *, span: Span = None, reason: str | None = None
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text="bound the recursion or align the trait implementation",
        note="compiler.types/hard",
        span=span,
    )


@dataclass
class RecursiveExpansionReport:
    """Bounded unrolling of a (possibly mutually) recursive generic graph."""

    root: str
    canonical_root: str
    expansions: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    placeholders: tuple[str, ...]
    aliases: Mapping[str, str]
    budget: int
    exhausted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "canonical_root": self.canonical_root,
            "expansions": list(self.expansions),
            "cycles": [list(cycle) for cycle in self.cycles],
            "placeholders": list(self.placeholders),
            "aliases": dict(self.aliases),
            "budget": self.budget,
            "exhausted": self.exhausted,
        }


def types_expand_mutually_recursive_generics(
    specs: Mapping[str, Sequence[str]],
    *,
    root: str,
    budget: int = 64,
    aliases: Mapping[str, str] | None = None,
    span: Span = None,
) -> RecursiveExpansionReport:
    """Unroll recursive generic definitions with cycle detection and a budget.

    ``specs`` maps a type name to the generic types it references.  Alias
    unification is applied before traversal so ``Tree<T>`` and its alias
    ``Node<T>`` share one placeholder cache.  Cycles are recorded but unrolled
    at most ``budget`` times; exceeding the budget is an ``E002`` diagnostic,
    never an unbounded recursion or ``RecursionError``.
    """
    if budget <= 0:
        raise _fail("E002", "specialization budget must be positive", span=span)
    alias_map = dict(aliases or {})

    def canonical(name: str) -> str:
        seen: set[str] = set()
        current = name
        while current in alias_map and current not in seen:
            seen.add(current)
            current = alias_map[current]
        return current

    merged: dict[str, list[str]] = {}
    for name, children in specs.items():
        target = canonical(name)
        bucket = merged.setdefault(target, [])
        for child in children:
            resolved = canonical(child)
            if resolved not in bucket:
                bucket.append(resolved)
    canonical_specs: dict[str, tuple[str, ...]] = {
        name: tuple(children) for name, children in merged.items()
    }
    canonical_root = canonical(root)
    if canonical_root not in canonical_specs:
        raise _fail(
            "E005",
            f"unknown generic root {root!r}",
            span=span,
            reason="the root must appear in the generic specification",
        )

    expansions: list[str] = []
    cycles: list[tuple[str, ...]] = []
    placeholders: set[str] = set()
    stack: list[str] = []
    exhausted = False

    def visit(name: str) -> None:
        nonlocal exhausted
        if len(expansions) >= budget:
            exhausted = True
            raise _fail(
                "E002",
                f"specialization budget of {budget} exceeded at {name!r}",
                span=span,
                reason="recursive generic expansion is too large",
            )
        expansions.append(name)
        placeholders.add(name)
        if name in stack:
            start = stack.index(name)
            cycle = tuple(stack[start:] + [name])
            if cycle not in cycles:
                cycles.append(cycle)
            if len(expansions) >= budget:
                exhausted = True
                raise _fail(
                    "E002",
                    f"specialization budget of {budget} exceeded at {name!r}",
                    span=span,
                    reason="recursive generic expansion is too large",
                )
            return
        stack.append(name)
        for child in canonical_specs.get(name, ()):
            if child not in canonical_specs:
                raise _fail(
                    "E005",
                    f"generic {name!r} references undefined type {child!r}",
                    span=span,
                )
            visit(child)
        stack.pop()

    visit(canonical_root)
    return RecursiveExpansionReport(
        root=root,
        canonical_root=canonical_root,
        expansions=tuple(expansions),
        cycles=tuple(cycles),
        placeholders=tuple(sorted(placeholders)),
        aliases=alias_map,
        budget=budget,
        exhausted=exhausted,
    )


@dataclass
class CoherenceReport:
    """Trait overlap, associated-type, and dispatch analysis result."""

    trait: str
    implementations: tuple[str, ...]
    dispatch: dict[str, str] = field(default_factory=dict)
    projections: dict[str, dict[str, str]] = field(default_factory=dict)
    higher_rank: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "trait": self.trait,
            "implementations": list(self.implementations),
            "dispatch": dict(self.dispatch),
            "projections": {
                key: dict(value) for key, value in self.projections.items()
            },
            "higher_rank": list(self.higher_rank),
        }


def types_check_coherence_and_projection(
    implementations: Sequence[Mapping[str, Any]],
    *,
    trait: str,
    associated: Mapping[str, str] | None = None,
    span: Span = None,
) -> CoherenceReport:
    """Check overlap, associated types, and higher-rank bounds for one trait."""
    associated = dict(associated or {})
    seen: dict[tuple[str, str], str] = {}
    dispatch: dict[str, str] = {}
    projections: dict[str, dict[str, str]] = {}
    higher_rank: list[str] = []
    for impl in implementations:
        name = str(impl.get("name", ""))
        impl_trait = str(impl.get("trait", trait))
        target = str(impl.get("for", ""))
        if not name or not target:
            raise _fail(
                "E001", "trait implementation needs a name and target", span=span
            )
        key = (impl_trait, target)
        if key in seen:
            raise _fail(
                "E001",
                f"coherence violation: {name!r} overlaps {seen[key]!r} for "
                f"{impl_trait} {target}",
                span=span,
            )
        if impl_trait != trait:
            continue
        seen[key] = name
        dispatch[target] = name
        values = dict(impl.get("assoc", {}))
        missing = sorted(set(associated) - set(values))
        if missing:
            raise _fail(
                "E002",
                f"implementation {name!r} is missing associated type(s): "
                f"{', '.join(missing)}",
                span=span,
            )
        projections[name] = {
            key_name: str(values[key_name]) for key_name in sorted(associated)
        }
        for bound in impl.get("bounds", ()):
            text = str(bound)
            if "for<" in text:
                lifetime = text[text.index("for<") + 4 :].split(">", 1)[0].lstrip("'")
                declared = {
                    str(token).lstrip("'") for token in impl.get("lifetimes", ())
                }
                if lifetime not in declared:
                    raise _fail(
                        "E002",
                        f"implementation {name!r} does not satisfy higher-rank "
                        f"bound {text!r}",
                        span=span,
                    )
                higher_rank.append(f"{name}:{lifetime}")
    return CoherenceReport(
        trait=trait,
        implementations=tuple(seen.values()),
        dispatch=dispatch,
        projections=projections,
        higher_rank=tuple(higher_rank),
    )


def types_enforce_higher_rank_bounds(
    bounds: Sequence[str],
    *,
    supplied: Sequence[str],
    span: Span = None,
) -> dict[str, Any]:
    """Validate ``for<'a>``-style bounds against supplied lifetime tokens."""
    available = {str(token).lstrip("'") for token in supplied}
    required: list[str] = []
    for bound in bounds:
        text = str(bound)
        if "for<" not in text:
            continue
        lifetime = text[text.index("for<") + 4 :].split(">", 1)[0].strip("'")
        if not lifetime:
            raise _fail("E002", f"malformed higher-rank bound {text!r}", span=span)
        if lifetime not in available:
            raise _fail(
                "E002",
                f"higher-rank bound {text!r} needs lifetime '{lifetime}",
                span=span,
            )
        required.append(lifetime)
    return {
        "ok": True,
        "required": sorted(set(required)),
        "supplied": sorted(available),
    }


__all__ = [
    "CoherenceReport",
    "RecursiveExpansionReport",
    "types_check_coherence_and_projection",
    "types_enforce_higher_rank_bounds",
    "types_expand_mutually_recursive_generics",
]
