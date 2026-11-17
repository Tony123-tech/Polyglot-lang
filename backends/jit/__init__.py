"""Zero-dependency Apple Silicon JIT backend for Polyglot.

Public entry point: :class:`backends.jit.backend.JITBackend`, also available
through ``backends.get_backend("jit")``.
"""

from backends.jit.backend import JITBackend, JITModule
from backends.jit.errors import (
    JITCompileError,
    JITError,
    JITRuntimeError,
    JITUnsupportedError,
)

__all__ = [
    "JITBackend",
    "JITCompileError",
    "JITError",
    "JITModule",
    "JITRuntimeError",
    "JITUnsupportedError",
]
