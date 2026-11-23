"""Mojo-inspired value/origin semantics with explicit lifetime tracking.

Mojo's ownership system distinguishes *values* (owned, copied or moved) from
*references* (borrowed, never owned), and tracks the *origin* lifetime of each
reference so the compiler can prevent use-after-free and origin escapes.
Mojo also has an ``UnsafeAnyOrigin`` escape hatch that widens a reference's
origin to "anything" — this module models that as an explicit, auditable
operation rather than a silent widening.

This module is a pure-Python *analysis* pass: it does not generate code or
touch native memory.  It tracks, for each binding in a scope:

* :class:`ValueKind` — whether the binding is ``Copyable``, ``Movable``, or
  ``BorrowOnly``.
* :class:`Origin` — whether the binding is ``Owned`` or ``Borrowed`` (with
  an optional ``lifetime_id`` for borrow checking).

Operations that would violate ownership (using a moved value, returning a
borrowed reference as owned, widening to ``UnsafeAnyOrigin`` without an
explicit annotation) raise structured ``E001``–``E008`` diagnostics.

Diagnostics follow the shared surface: ``E002`` for use-after-move, ``E001``
for origin escapes, ``E007`` for ``UnsafeAnyOrigin`` widening without the
``#[unsafe_origin]`` annotation, and ``E003`` for API misuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "Origin",
    "OriginEnv",
    "ValueKind",
    "check_borrow",
    "check_origin_escape",
]

_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


class ValueKind(Enum):
    """Whether a binding owns its value and how it may be used."""

    COPYABLE = "copyable"
    MOVABLE = "movable"
    BORROW_ONLY = "borrow_only"

    @classmethod
    def from_name(cls, name: str) -> ValueKind:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E003", f"unknown value kind: {name!r}")


class Origin(Enum):
    """The origin lifetime of a binding.

    ``Owned`` bindings own their value and may be moved or borrowed.
    ``Borrowed`` bindings reference a value owned elsewhere and may not be
    moved or returned as owned.  ``UnsafeAnyOrigin`` is an explicit escape
    hatch that widens a borrowed reference's origin — it requires the
    ``#[unsafe_origin]`` annotation and is always auditable.
    """

    OWNED = "owned"
    BORROWED = "borrowed"
    UNSAFE_ANY_ORIGIN = "unsafe_any_origin"

    @classmethod
    def from_name(cls, name: str) -> Origin:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E003", f"unknown origin: {name!r}")


@dataclass(frozen=True)
class _Binding:
    """A single binding's type, value kind, origin, and lifetime."""

    name: str
    value_kind: ValueKind
    origin: Origin
    lifetime_id: int = 0
    moved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_kind": self.value_kind.value,
            "origin": self.origin.value,
            "lifetime_id": self.lifetime_id,
            "moved": self.moved,
        }


@dataclass
class OriginEnv:
    """A scoped environment tracking bindings and their ownership.

    The environment is a stack of scopes.  Each scope maps binding names to
    :class:`_Binding` records.  Lookups search from the innermost scope
    outward.  When a binding is moved, its ``moved`` flag is set; any
    subsequent use raises ``E002``.
    """

    _scopes: list[dict[str, _Binding]] = field(default_factory=lambda: [{}])
    _lifetime_counter: int = 0

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        if len(self._scopes) <= 1:
            _fail("E003", "cannot pop the root scope")
        self._scopes.pop()

    def declare(
        self,
        name: str,
        value_kind: ValueKind,
        origin: Origin,
        *,
        lifetime_id: int | None = None,
    ) -> None:
        if not _IDENTIFIER_RE.match(name):
            _fail("E001", f"invalid binding name: {name!r}")
        if lifetime_id is None:
            self._lifetime_counter += 1
            lifetime_id = self._lifetime_counter
        if origin == Origin.UNSAFE_ANY_ORIGIN:
            # UnsafeAnyOrigin is only legal with an explicit annotation;
            # declaring it without one is an E007 diagnostic.
            _fail(
                "E007",
                f"binding {name!r} declares UnsafeAnyOrigin without the "
                "#[unsafe_origin] annotation",
                reason="UnsafeAnyOrigin widening requires explicit opt-in",
            )
        self._scopes[-1][name] = _Binding(
            name=name,
            value_kind=value_kind,
            origin=origin,
            lifetime_id=lifetime_id,
        )

    def lookup(self, name: str) -> _Binding | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def move(self, name: str) -> None:
        """Mark *name* as moved.  Raises ``E002`` if already moved."""
        binding = self.lookup(name)
        if binding is None:
            _fail("E001", f"use of undeclared binding {name!r}")
        if binding.moved:
            _fail(
                "E002",
                f"use of moved value {name!r}",
                reason="a moved value cannot be used again",
            )
        if binding.value_kind == ValueKind.BORROW_ONLY:
            _fail(
                "E002",
                f"cannot move borrowed-only binding {name!r}",
                reason="borrowed values cannot be moved",
            )
        if binding.origin == Origin.BORROWED:
            _fail(
                "E002",
                f"cannot move borrowed binding {name!r}",
                reason="a borrowed reference does not own its value",
            )
        # Replace the binding in its scope with a moved copy.
        for scope in reversed(self._scopes):
            if name in scope:
                scope[name] = _Binding(
                    name=binding.name,
                    value_kind=binding.value_kind,
                    origin=binding.origin,
                    lifetime_id=binding.lifetime_id,
                    moved=True,
                )
                return

    def as_dict(self) -> dict[str, Any]:
        return {
            "scopes": [
                {name: binding.as_dict() for name, binding in scope.items()}
                for scope in self._scopes
            ],
            "lifetime_counter": self._lifetime_counter,
        }


def check_borrow(
    env: OriginEnv,
    binding_name: str,
    *,
    expected_lifetime: int | None = None,
) -> _Binding:
    """Validate that *binding_name* can be borrowed in the current scope.

    Returns the binding's :class:`_Binding` record on success.  Raises:

    * ``E002`` if the binding has been moved (use-after-move).
    * ``E001`` if the binding is undeclared or the borrow would outlive the
      binding's origin (``expected_lifetime`` does not match).
    """
    binding = env.lookup(binding_name)
    if binding is None:
        _fail("E001", f"borrow of undeclared binding {binding_name!r}")
    if binding.moved:
        _fail(
            "E002",
            f"borrow of moved value {binding_name!r}",
            reason="a moved value cannot be borrowed",
        )
    if expected_lifetime is not None and binding.lifetime_id != expected_lifetime:
        _fail(
            "E001",
            f"borrow of {binding_name!r} with lifetime {binding.lifetime_id} "
            f"does not match expected lifetime {expected_lifetime}",
            reason="the borrow would outlive the binding's origin",
        )
    return binding


def check_origin_escape(
    env: OriginEnv,
    binding_name: str,
    *,
    return_as_owned: bool,
    allow_unsafe: bool = False,
) -> _Binding:
    """Validate that returning *binding_name* does not escape its origin.

    A ``Borrowed`` binding cannot be returned as ``Owned`` — that would let
    the caller outlive the borrow.  A ``UNSAFE_ANY_ORIGIN`` binding requires
    ``allow_unsafe=True`` (the ``#[unsafe_origin]`` annotation).

    Raises:

    * ``E001`` if a borrowed binding is returned as owned.
    * ``E007`` if an unsafe-origin binding is returned without the opt-in.
    """
    binding = env.lookup(binding_name)
    if binding is None:
        _fail("E001", f"escape of undeclared binding {binding_name!r}")
    if binding.moved:
        _fail(
            "E002",
            f"return of moved value {binding_name!r}",
            reason="a moved value cannot be returned",
        )
    if binding.origin == Origin.BORROWED and return_as_owned:
        _fail(
            "E001",
            f"borrowed binding {binding_name!r} cannot be returned as owned",
            reason="the caller would outlive the borrow's origin",
        )
    if binding.origin == Origin.UNSAFE_ANY_ORIGIN and not allow_unsafe:
        _fail(
            "E007",
            f"unsafe-origin binding {binding_name!r} returned without "
            "#[unsafe_origin] annotation",
            reason="UnsafeAnyOrigin returns require explicit opt-in",
        )
    return binding
