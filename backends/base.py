"""Backend interface for Polyglot."""

from __future__ import annotations

from typing import Any, Protocol

from ast_nodes import Program


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
    trusted: bool

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


def get_backend(name: str = "python", *, trusted: bool = True) -> Backend:
    if name == "python":
        from backends.python.translator import PythonBackend
        return PythonBackend(trusted=trusted)
    if name == "native":
        from backends.rust.compiler import NativeBackend
        return NativeBackend(trusted=trusted)
    raise ValueError(f"Unknown backend: {name!r}")


__all__ = [
    "Backend",
    "BackendBase",
    "UntrustedCompileError",
    "get_backend",
]
