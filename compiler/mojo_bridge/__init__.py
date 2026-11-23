"""Mojo-inspired value/origin semantics, vectorized layout, and CPython interop.

Three submodules bring Mojo's signature strengths to ``polyglot-lang`` while
respecting platform/runtime boundaries:

* :mod:`compiler.mojo_bridge.origins` — explicit value vs. reference/origin
  lifetimes (``Copyable``/``Movable``/borrowed vs. owned origins without
  silent ``UnsafeAnyOrigin`` widening) with structured ``E001``–``E008``
  diagnostics on invalid borrows or origin escapes.
* :mod:`compiler.mojo_bridge.layout` — contiguous tensor/buffer layouts,
  SIMD/vector loop lowering, and MLIR-style pass pipelines with clear
  capability reporting when native MLIR/GPU toolchains are absent.
* :mod:`compiler.mojo_bridge.cpython_bridge` — clean bridging between Python
  objects/modules and typed native values, respecting PEP 667 and avoiding
  hot-loop boxing overhead.

All code is pure Python and never touches native MLIR/GPU toolchains, raw
addresses, ``MAP_JIT``, or PEP 667 frame locals.
"""

from __future__ import annotations

from compiler.mojo_bridge.cpython_bridge import (
    BridgeResult,
    bridge_python_value,
    check_pep_667_safe,
)
from compiler.mojo_bridge.layout import (
    BufferLayout,
    KernelLowering,
    LayoutOrder,
    lower_vectorized_kernel,
    probe_mlir_available,
)
from compiler.mojo_bridge.origins import (
    Origin,
    OriginEnv,
    ValueKind,
    check_borrow,
    check_origin_escape,
)

__all__ = [
    "BridgeResult",
    "BufferLayout",
    "KernelLowering",
    "LayoutOrder",
    "Origin",
    "OriginEnv",
    "ValueKind",
    "bridge_python_value",
    "check_borrow",
    "check_origin_escape",
    "check_pep_667_safe",
    "lower_vectorized_kernel",
    "probe_mlir_available",
]
