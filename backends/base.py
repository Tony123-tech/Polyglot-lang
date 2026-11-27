"""Backend interface for Polyglot.

This module deliberately imports no concrete backend.  Every backend module
inherits from :class:`BackendBase`, so a top-level import in the other direction
closes an import cycle.  The ``get_backend()`` factory therefore lives in
:mod:`backends.registry`; import it from :mod:`backends` (the public surface),
not from here.
"""

from __future__ import annotations

from typing import Any, Protocol

from compiler.core.ast_nodes import Program


class UntrustedCompileError(RuntimeError):
    """Raised when a backend refuses to compile untrusted input."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(
            f"Backend {backend_name!r} refuses to compile untrusted input. "
            "Pass trusted=True to get_backend() if the source is trusted, "
            "or run compilation in an isolated process. See SECURITY.md."
        )
        self.backend_name = backend_name


class Backend(Protocol):
    """A backend that can compile a Polyglot Program into a usable namespace."""

    name: str
    """Backend name ("python", "native", "wasm", or "jit")."""

    trusted: bool
    """Whether compilation of untrusted input is allowed. See SECURITY.md."""

    def compile(self, program: Program) -> dict[str, Any]:
        """
        Compile the program and return a namespace (dict) containing
        the resulting functions, classes, and constants.
        """
        ...


class BackendBase:
    """Concrete base for backends. Provides the trust gate."""

    name: str = ""

    def __init__(self, *, trusted: bool = True) -> None:
        self.trusted = trusted

    def _check_trust(self) -> None:
        """Raise if this backend was created with trusted=False."""
        if not self.trusted:
            raise UntrustedCompileError(self.name)


__all__ = [
    "Backend",
    "BackendBase",
    "UntrustedCompileError",
]
