"""
Backend abstraction with explicit trust gate.

Trust model:
- `get_backend(name)` returns a backend whose `trusted` flag is taken from
  the `POLYGLOT_TRUSTED` environment variable (default: "0" = untrusted).
- `get_backend(name, trusted=True)` explicitly allows compilation.
- `get_backend(name, trusted=False)` forces the backend to refuse.

The trust gate is a defence-in-depth measure, NOT a sandbox. If you expose
compile() to untrusted input, you MUST isolate at the OS level (container,
seccomp, no network, read-only FS).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from ast_nodes import Program

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UntrustedCompileError(RuntimeError):
    """Raised when a backend is asked to compile untrusted input."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(
            f"Backend {backend_name!r} refuses to compile untrusted input. "
            f"Pass trusted=True to get_backend() if the source is trusted, "
            f"or run compilation in an isolated process. See SECURITY.md."
        )
        self.backend_name = backend_name


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Backend(ABC):
    """Common interface for all backends."""

    name: str = "base"

    def __init__(self, *, trusted: bool = True) -> None:
        self.trusted = trusted

    @abstractmethod
    def compile(self, program: Program) -> dict[str, Any]:
        """Compile a Program and return its namespace."""
        raise NotImplementedError

    def _check_trust(self) -> None:
        """Raise if this backend was created with trusted=False."""
        if not self.trusted:
            raise UntrustedCompileError(self.name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backend(
    name: str = "python",
    *,
    trusted: bool | None = None,
) -> Backend:
    """Return a backend instance.

    Args:
        name: "python" or "native".
        trusted: whether the source is trusted. Defaults to the value of
            the `POLYGLOT_TRUSTED` env var (default "0" = untrusted).

    Example:
        # Trusted (local dev, CI): explicit opt-in
        backend = get_backend("native", trusted=True)

        # Untrusted (HTTP, plugin marketplace): default, refuses compile
        backend = get_backend("native")
        backend.compile(program)  # raises UntrustedCompileError

    Security note:
        trusted=False is a *speed bump*, not a sandbox. It prevents
        accidental use in API handlers. If you must compile untrusted
        input, run it in a container / VM with no network and read-only FS.
    """
    if trusted is None:
        trusted = os.environ.get("POLYGLOT_TRUSTED", "0") == "1"

    if name == "python":
        from backends.python.translator import PythonBackend
        return PythonBackend(trusted=trusted)
    if name == "native":
        from backends.rust.compiler import NativeBackend
        return NativeBackend(trusted=trusted)
    raise ValueError(f"Unknown backend: {name!r}")
