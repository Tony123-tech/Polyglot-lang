"""Gleam-inspired cross-target contracts with capability checks.

Gleam compiles to Erlang/BEAM, JavaScript, and (experimentally) native
targets from a single source language.  The discipline that makes this work
is that the *language* is pure and the *runtime* is explicit about what each
target supports.

This module models that discipline for ``polyglot-lang``:

* :class:`Target` enumerates the supported backends (Python, Rust/PyO3, Zig,
  and optional BEAM/JS descriptors).
* :class:`Capability` enumerates the features a target may or may not support
  (ADTs, bit-array matching, SIMD, OTP actors, etc.).
* :class:`TargetDescriptor` declares which capabilities a target has.
* :func:`emit_contract` validates that a pure functional data transformation
  (described as a set of required capabilities) can be emitted for a target,
  raising ``E007`` (unsupported feature) with a concrete message when the
  target lacks a required capability.

Everything here is pure Python and never touches native toolchains.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "Capability",
    "Target",
    "TargetContract",
    "TargetDescriptor",
    "emit_contract",
]


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


class Target(Enum):
    """A backend target for cross-target contract emission."""

    PYTHON = "python"
    RUST = "rust"
    ZIG = "zig"
    BEAM = "beam"
    JS = "js"

    @classmethod
    def from_name(cls, name: str) -> Target:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E007", f"unknown target: {name!r}")


class Capability(Enum):
    """A feature a target may or may not support."""

    ADT = "adt"
    EXHAUSTIVE_MATCH = "exhaustive_match"
    BIT_ARRAY = "bit_array"
    SIMD = "simd"
    OTP_ACTOR = "otp_actor"
    SUPERVISION = "supervision"
    ERROR_UNIONS = "error_unions"
    C_INTEROP = "c_interop"
    PYTHON_INTEROP = "python_interop"

    @classmethod
    def from_name(cls, name: str) -> Capability:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E007", f"unknown capability: {name!r}")


# Capability matrix for the known targets.  This is the single source of
# truth; callers should not hardcode capability assumptions.
_CAPABILITY_MATRIX: Final[dict[Target, frozenset[Capability]]] = {
    Target.PYTHON: frozenset(
        {
            Capability.ADT,
            Capability.EXHAUSTIVE_MATCH,
            Capability.BIT_ARRAY,
            Capability.ERROR_UNIONS,
            Capability.PYTHON_INTEROP,
        }
    ),
    Target.RUST: frozenset(
        {
            Capability.ADT,
            Capability.EXHAUSTIVE_MATCH,
            Capability.ERROR_UNIONS,
            Capability.C_INTEROP,
            Capability.PYTHON_INTEROP,
        }
    ),
    Target.ZIG: frozenset(
        {
            Capability.ADT,
            Capability.EXHAUSTIVE_MATCH,
            Capability.ERROR_UNIONS,
            Capability.C_INTEROP,
            Capability.SIMD,
        }
    ),
    Target.BEAM: frozenset(
        {
            Capability.ADT,
            Capability.EXHAUSTIVE_MATCH,
            Capability.BIT_ARRAY,
            Capability.OTP_ACTOR,
            Capability.SUPERVISION,
            Capability.ERROR_UNIONS,
        }
    ),
    Target.JS: frozenset(
        {
            Capability.ADT,
            Capability.EXHAUSTIVE_MATCH,
            Capability.BIT_ARRAY,
            Capability.ERROR_UNIONS,
        }
    ),
}


@dataclass(frozen=True)
class TargetDescriptor:
    """Describes one target's capabilities, optionally overriding the matrix."""

    target: Target
    overrides: frozenset[Capability] = field(default_factory=frozenset)
    disabled: frozenset[Capability] = field(default_factory=frozenset)

    def capabilities(self) -> frozenset[Capability]:
        base = _CAPABILITY_MATRIX.get(self.target, frozenset())
        return (base | self.overrides) - self.disabled

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "capabilities": sorted(c.value for c in self.capabilities()),
            "overrides": sorted(c.value for c in self.overrides),
            "disabled": sorted(c.value for c in self.disabled),
        }


@dataclass(frozen=True)
class TargetContract:
    """A cross-target emission contract for one pure transformation."""

    name: str
    required: frozenset[Capability]
    target: TargetDescriptor
    supported: bool
    missing: frozenset[Capability]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.target.value,
            "supported": self.supported,
            "required": sorted(c.value for c in self.required),
            "missing": sorted(c.value for c in self.missing),
        }

    def format(self) -> str:
        if self.supported:
            return f"contract {self.name!r} is supported on target {self.target.target.value}"
        missing = ", ".join(sorted(c.value for c in self.missing))
        return (
            f"contract {self.name!r} is not supported on target "
            f"{self.target.target.value}; missing capabilities: {missing}"
        )


def emit_contract(
    name: str,
    required: AbstractSet[Capability] | frozenset[Capability],
    target: TargetDescriptor | Target,
) -> TargetContract:
    """Validate that *target* supports every capability in *required*.

    Returns a :class:`TargetContract` describing the outcome.  Raises
    ``E007`` when the target lacks a required capability, with a message
    naming the missing capabilities so the caller can either fall back to a
    different target or declare the transformation out of scope.

    ``E003`` is raised for API misuse (unknown target, empty contract name).
    """
    if not name or not isinstance(name, str):
        _fail("E003", "contract name must be a non-empty string")
    if not isinstance(required, (set, frozenset)):
        _fail("E003", "required capabilities must be a set of Capability values")
    if not required:
        _fail("E003", "a contract must require at least one capability")
    for capability in required:
        if not isinstance(capability, Capability):
            _fail("E003", f"not a Capability value: {capability!r}")
    if isinstance(target, TargetDescriptor):
        descriptor = target
    elif isinstance(target, Target):
        descriptor = TargetDescriptor(target=target)
    else:
        _fail("E007", f"unknown target: {target!r}")
    available = descriptor.capabilities()
    missing = frozenset(c for c in required if c not in available)
    contract = TargetContract(
        name=name,
        required=frozenset(required),
        target=descriptor,
        supported=not missing,
        missing=missing,
    )
    if missing:
        _fail(
            "E007",
            contract.format(),
            reason="the target lacks a capability required by this transformation",
        )
    return contract
