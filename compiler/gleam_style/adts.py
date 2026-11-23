"""Gleam-inspired ADT definitions and exhaustive pattern matching.

Gleam's type system is Hindley-Milner with algebraic data types (ADTs) and
*mandatory* exhaustiveness checking: a `case` expression that does not
cover every variant is a compile error, not a runtime crash.  This module
models that discipline for `polyglot-lang`:

* :class:`Variant` / :class:`AdtDef` describe an ADT (a set of variants,
  each with optional payload types).
* :func:`check_exhaustive` validates that a list of patterns covers every
  variant of an ADT.  On failure it returns a :class:`MissingPatterns`
  witness listing the *concrete* uncovered variants (e.g.
  ``Some(Int)``, ``None``) rather than a vague "match not exhaustive"
  message.
* :func:`check_total_adt` runs HM-style unification over an ADT's variants
  to verify that variant payload types are well-formed and that recursive
  references (e.g. ``Tree<T>`` referencing ``Tree<T>``) resolve.

Diagnostics follow the shared ``E001``-``E008`` surface: ``E002`` for
non-exhaustive matches, ``E001`` for malformed ADT definitions, ``E005``
for recursive cycles that cannot be resolved, and ``E003`` for API misuse.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "AdtDef",
    "AdtRegistry",
    "ExhaustivenessResult",
    "MissingPatterns",
    "Variant",
    "check_exhaustive",
    "check_total_adt",
]


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty string")
    return value


def _identifier(name: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
        _fail("E001", f"{label} is not a legal identifier: {name!r}")
    return name


@dataclass(frozen=True)
class Variant:
    """One variant of an ADT, optionally carrying payload types."""

    name: str
    fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "fields": list(self.fields)}

    def witness(self) -> str:
        """Return a concrete missing-pattern witness for this variant.

        For example ``Some(Int)`` or ``None`` — the string a user would
        write in a ``case`` arm to cover this variant.
        """
        if not self.fields:
            return self.name
        return f"{self.name}({', '.join(self.fields)})"


@dataclass(frozen=True)
class AdtDef:
    """An algebraic data type: a named set of variants."""

    name: str
    variants: tuple[Variant, ...]
    generics: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variants": [variant.as_dict() for variant in self.variants],
            "generics": list(self.generics),
        }

    def variant_names(self) -> frozenset[str]:
        return frozenset(variant.name for variant in self.variants)

    def variant(self, name: str) -> Variant | None:
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None


@dataclass(frozen=True)
class MissingPatterns:
    """The concrete variants not covered by a match expression."""

    witnesses: tuple[str, ...]
    adt_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "adt": self.adt_name,
            "missing": list(self.witnesses),
            "exhaustive": False,
        }

    def format(self) -> str:
        return (
            f"non-exhaustive match on {self.adt_name!r}; "
            f"missing patterns: {', '.join(self.witnesses)}"
        )


@dataclass(frozen=True)
class ExhaustivenessResult:
    """Outcome of an exhaustiveness check."""

    adt_name: str
    exhaustive: bool
    covered: tuple[str, ...]
    missing: MissingPatterns | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "adt": self.adt_name,
            "exhaustive": self.exhaustive,
            "covered": list(self.covered),
            "missing": self.missing.as_dict() if self.missing else None,
        }


# A pattern is described as a tuple: (variant_name, payload_pattern_strings).
# The payload strings are opaque to the exhaustiveness check — they are used
# only to build the witness string.  A bare string is a wildcard/ident
# pattern that does not cover any specific variant.
Pattern = str | tuple[str, tuple[str, ...]]


def _variant_from_pattern(pattern: Pattern) -> str | None:
    """Return the variant name covered by *pattern*, or ``None`` for wildcards."""
    if isinstance(pattern, str):
        return None
    return pattern[0]


def _witness_for(pattern: Pattern, adt: AdtDef) -> str:
    if isinstance(pattern, str):
        return pattern
    name, payloads = pattern
    return f"{name}({', '.join(payloads)})" if payloads else name


def check_exhaustive(
    adt: AdtDef,
    patterns: Sequence[Pattern],
    *,
    has_wildcard: bool = False,
) -> ExhaustivenessResult:
    """Check that *patterns* cover every variant of *adt*.

    A wildcard pattern (a bare string, or ``has_wildcard=True``) makes the
    match trivially exhaustive.  Otherwise every variant name in *adt* must
    appear as the head of at least one pattern.  Missing variants are
    returned as a :class:`MissingPatterns` witness with concrete
    ``Variant(...)`` strings.

    Raises ``E002`` when the match is non-exhaustive.
    """
    _validate_adt(adt)
    covered: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        name = _variant_from_pattern(pattern)
        if name is None:
            has_wildcard = True
            continue
        if name not in adt.variant_names():
            _fail(
                "E001",
                f"unknown variant {name!r} in match on {adt.name!r}",
                reason="the pattern references a variant not declared on the ADT",
            )
        if name not in seen:
            seen.add(name)
            covered.append(name)

    all_names = [variant.name for variant in adt.variants]
    missing_names = [name for name in all_names if name not in seen]
    if has_wildcard or not missing_names:
        return ExhaustivenessResult(
            adt_name=adt.name,
            exhaustive=True,
            covered=tuple(covered),
            missing=None,
        )
    missing_witnesses = tuple(
        _witness_for(_pattern_for_variant(name, adt), adt) for name in missing_names
    )
    missing = MissingPatterns(witnesses=missing_witnesses, adt_name=adt.name)
    _fail(
        "E002",
        missing.format(),
        reason="every ADT variant must be covered by a pattern or a wildcard",
    )


def _pattern_for_variant(name: str, adt: AdtDef) -> Pattern:
    variant = adt.variant(name)
    if variant is None:
        return name
    if not variant.fields:
        return (name, ())
    return (name, variant.fields)


def _validate_adt(adt: AdtDef) -> None:
    if not isinstance(adt, AdtDef):
        _fail("E003", "expected an AdtDef")
    _identifier(adt.name, "ADT name")
    if not adt.variants:
        _fail("E001", f"ADT {adt.name!r} has no variants")
    seen: set[str] = set()
    for variant in adt.variants:
        if variant.name in seen:
            _fail(
                "E001",
                f"duplicate variant {variant.name!r} in {adt.name!r}",
            )
        seen.add(variant.name)
        _identifier(variant.name, "variant name")
        for field_name in variant.fields:
            if not field_name:
                _fail("E001", f"empty payload type in {variant.name!r}")
    for generic in adt.generics:
        _identifier(generic, "generic parameter")


def check_total_adt(
    adt: AdtDef,
    *,
    bindings: Mapping[str, str] | None = None,
    budget: int = 64,
) -> dict[str, Any]:
    """Validate an ADT's structure and resolve recursive generic references.

    Performs a Hindley-Milner-style walk over the ADT's variants and payload
    types, expanding type aliases via *bindings* and detecting recursive
    cycles (e.g. ``Tree<T>`` whose variant references ``Tree<T>``).  Cycles
    are recorded but bounded by *budget* — exceeding it is an ``E002``
    diagnostic, never unbounded recursion.

    Returns a structured report with the resolved variant signatures, any
    cycles found, and a ``total`` flag indicating whether every variant's
    payloads are well-formed.
    """
    if budget <= 0:
        _fail("E002", "specialization budget must be positive")
    _validate_adt(adt)
    alias_map: dict[str, str] = dict(bindings or {})
    # Seed the alias map with the ADT's own generic parameters so recursive
    # references like ``Tree<T>`` resolve to themselves rather than erroring.
    for generic in adt.generics:
        alias_map.setdefault(generic, generic)
    alias_map.setdefault(adt.name, adt.name)

    def canonical(name: str) -> str:
        seen: set[str] = set()
        current = name
        while current in alias_map and current not in seen:
            seen.add(current)
            nxt = alias_map[current]
            if nxt == current:
                break
            current = nxt
        return current

    expansions: list[str] = []
    cycles: list[tuple[str, ...]] = []
    stack: list[str] = []
    variant_signatures: dict[str, tuple[str, ...]] = {}

    def visit(name: str) -> None:
        if len(expansions) >= budget:
            _fail(
                "E002",
                f"ADT resolution budget of {budget} exceeded at {name!r}",
                reason="recursive generic expansion is too large",
            )
        expansions.append(name)
        if name in stack:
            start = stack.index(name)
            cycle = tuple(stack[start:] + [name])
            if cycle not in cycles:
                cycles.append(cycle)
            return
        stack.append(name)
        # When *name* is the ADT itself (e.g. a recursive ``Tree<T>``
        # reference), visit all of its variants to detect cycles through the
        # ADT boundary.  When *name* is a variant, visit only that variant.
        variant = adt.variant(name)
        if variant is not None:
            resolved: list[str] = []
            for field_name in variant.fields:
                resolved.append(canonical(field_name))
                visit(canonical(field_name))
            variant_signatures[name] = tuple(resolved)
        elif name == adt.name:
            # The ADT name was referenced (recursive type); visit each variant.
            for child in adt.variants:
                visit(child.name)
        stack.pop()

    for variant in adt.variants:
        visit(variant.name)

    total = True
    for payloads in variant_signatures.values():
        for payload in payloads:
            if not payload:
                total = False
                break

    return {
        "adt": adt.name,
        "generics": list(adt.generics),
        "variants": {
            name: list(payloads) for name, payloads in variant_signatures.items()
        },
        "cycles": [list(cycle) for cycle in cycles],
        "expansions": expansions,
        "budget": budget,
        "total": total,
        "model_only": True,
    }


class AdtRegistry:
    """A registry of ADTs keyed by name, with exhaustiveness helpers.

    The registry is the entry point for callers that want to declare several
    ADTs up front and then check match expressions against them by name.
    """

    def __init__(self) -> None:
        self._adts: dict[str, AdtDef] = {}

    def register(self, adt: AdtDef) -> None:
        _validate_adt(adt)
        if adt.name in self._adts:
            _fail("E001", f"ADT {adt.name!r} is already registered")
        self._adts[adt.name] = adt

    def get(self, name: str) -> AdtDef:
        adt = self._adts.get(name)
        if adt is None:
            _fail("E006", f"unknown ADT {name!r}", reason="no such ADT is registered")
        return adt

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adts))

    def check_match(
        self, adt_name: str, patterns: Sequence[Pattern], *, has_wildcard: bool = False
    ) -> ExhaustivenessResult:
        return check_exhaustive(self.get(adt_name), patterns, has_wildcard=has_wildcard)
