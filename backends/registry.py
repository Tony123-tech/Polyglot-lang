"""Backend factory, kept separate from :mod:`backends.base`.

``backends.base`` defines the ``Backend`` protocol and ``BackendBase`` that every
concrete backend inherits from.  The factory has to *reach* those concrete
backends, so living in ``base.py`` it forced an import cycle:

    backends.base -> backends.python.translator -> backends.base

The imports below are already lazy (they run inside the function), so nothing
actually breaks at runtime, but the cycle is real in the module graph and it
makes import order load-bearing.  Keeping the factory in its own module
restores a single direction: concrete backend -> ``backends.base``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from backends.base import Backend


def _trusted_from_env() -> bool:
    """Read POLYGLOT_TRUSTED; anything other than "1" is untrusted."""
    return os.environ.get("POLYGLOT_TRUSTED", "0") == "1"


def get_backend(
    name: str = "python",
    *,
    trusted: bool | None = None,
    target_width: int | None = None,
) -> Backend:
    """Return a backend by name.

    Args:
        name: "python" (transpile to Python AST), "native" (compile
            to a Rust/PyO3 extension), "wasm" (emit WAT and run it
            through wasmtime), or "jit" (encode AArch64 machine code
            directly; macOS on Apple Silicon only). Default: "python".
        trusted: Allow compilation of untrusted input. If None, read
            from POLYGLOT_TRUSTED (default: "0"). See SECURITY.md.
        target_width: Simulate a 32- or 64-bit usize/isize target.
            Python backend only; must be None for native and wasm.

    Returns:
        A Backend with a .compile(program) method that returns a
        namespace dict.

    Raises:
        ValueError: Unknown backend name, or target_width passed to
            the native backend.
        UntrustedCompileError: On .compile() if trusted=False.
    """
    if trusted is None:
        trusted = _trusted_from_env()
    if name == "python":
        from backends.python.translator import PythonBackend

        return PythonBackend(trusted=trusted, target_width=target_width)
    if name == "native":
        if target_width is not None:
            raise ValueError("target_width is supported only by the Python backend")
        from backends.rust.compiler import NativeBackend

        return NativeBackend(trusted=trusted)
    if name == "wasm":
        if target_width is not None:
            raise ValueError("target_width is supported only by the Python backend")
        from backends.wasm.codegen import WasmBackend

        return WasmBackend(trusted=trusted)
    if name == "jit":
        if target_width is not None:
            raise ValueError("target_width is supported only by the Python backend")
        from backends.jit.backend import JITBackend

        return JITBackend(trusted=trusted)

    if name == "auto":
        if target_width is not None:
            raise ValueError("target_width is supported only by the Python backend")
        from backends.auto import AutoBackend

        return AutoBackend(trusted=trusted)

    if name == "zig":
        if target_width is not None:
            raise ValueError("target_width is supported only by the Python backend")
        from zig.backend import ZigBackend

        return ZigBackend(trusted=trusted)
    raise ValueError(f"Unknown backend: {name!r}")


__all__ = ["get_backend"]
