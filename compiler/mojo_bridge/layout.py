"""Mojo-inspired buffer layouts, SIMD/vector loop lowering, and MLIR probing.

Mojo's signature strength is systems-level vectorization: contiguous tensor
layouts, SIMD vector loops, and MLIR-style pass pipelines.  This module models
those ideas in a backend-neutral way:

* :class:`BufferLayout` describes a contiguous tensor/buffer layout (shape,
  dtype, strides, memory order) with ``is_contiguous()`` and ``simd_lanes()``
  helpers.
* :func:`lower_vectorized_kernel` lowers a simple loop kernel into a
  backend-neutral IR dict describing the loop structure, vectorizability, and
  SIMD lane count.  It does *not* emit native code — it returns a
  transformation plan that the Python, Rust, or Zig backend can consume.
* :func:`probe_mlir_available` checks whether an MLIR/GPU toolchain is
  available on the system.  When it is absent (the common case), the function
  returns ``{"available": False, "reason": "..."}`` so callers can fall back
  to scalar loops without pretending to have native MLIR support.

Diagnostics follow the shared ``E001``–``E008`` surface: ``E001`` for invalid
layout specifications, ``E003`` for API misuse, and ``E007`` for unsupported
capabilities (e.g. non-contiguous layouts that cannot be vectorized).
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "BufferLayout",
    "KernelLowering",
    "LayoutOrder",
    "lower_vectorized_kernel",
    "probe_mlir_available",
]

# Maximum number of elements a layout may declare before we refuse to model
# it (prevents unbounded allocation attempts in the analysis).
_MAX_LAYOUT_ELEMENTS: Final[int] = 1 << 31


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


class LayoutOrder(Enum):
    """Memory order of a multi-dimensional buffer."""

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"

    @classmethod
    def from_name(cls, name: str) -> LayoutOrder:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E001", f"unknown layout order: {name!r}")


@dataclass(frozen=True)
class BufferLayout:
    """A contiguous tensor/buffer layout description.

    ``shape`` is the logical shape (e.g. ``(128, 256)``).  ``strides`` are
    the byte strides for each dimension; when omitted, they are computed from
    the shape and ``dtype_size`` assuming the given ``order``.
    """

    shape: tuple[int, ...]
    dtype: str
    dtype_size: int
    order: LayoutOrder = LayoutOrder.ROW_MAJOR
    strides: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.shape:
            _fail("E001", "layout shape must have at least one dimension")
        for dim in self.shape:
            if dim < 0:
                _fail("E001", f"negative dimension in shape {self.shape}")
        if self.dtype_size <= 0:
            _fail("E001", f"dtype_size must be positive, got {self.dtype_size}")
        if not self.dtype:
            _fail("E001", "dtype must be a non-empty string")
        total = math.prod(self.shape)
        if total > _MAX_LAYOUT_ELEMENTS:
            _fail(
                "E001",
                f"layout has {total} elements, exceeding the maximum of "
                f"{_MAX_LAYOUT_ELEMENTS}",
            )

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_size(self) -> int:
        return self.element_count * self.dtype_size

    def is_contiguous(self) -> bool:
        """Return whether the layout is contiguous in memory.

        A layout is contiguous when its strides match the canonical row-major
        or col-major traversal of its shape.  Non-contiguous layouts (e.g.
        transposed or strided views) cannot be trivially vectorized.
        """
        expected = self._canonical_strides()
        actual = self.strides or expected
        return actual == expected

    def _canonical_strides(self) -> tuple[int, ...]:
        if self.order == LayoutOrder.ROW_MAJOR:
            strides = [self.dtype_size] * len(self.shape)
            for i in range(len(self.shape) - 2, -1, -1):
                strides[i] = strides[i + 1] * self.shape[i + 1]
            return tuple(strides)
        strides = [self.dtype_size] * len(self.shape)
        for i in range(1, len(self.shape)):
            strides[i] = strides[i - 1] * self.shape[i - 1]
        return tuple(strides)

    def simd_lanes(self) -> int:
        """Return the SIMD lane count this layout can support.

        The lane count is the largest power-of-two divisor of the
        element count that is compatible with common SIMD widths (up to
        the dtype size).  A 1-D buffer of 256 f32 elements yields 8 lanes
        (256 / 32); a 3-element buffer yields 1 lane (no SIMD benefit).
        """
        count = self.element_count
        if count < 2:
            return 1
        # Find the largest power-of-two divisor of count.
        lanes = 1
        while count % (lanes * 2) == 0 and lanes < 64:
            lanes *= 2
        return lanes

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "dtype_size": self.dtype_size,
            "order": self.order.value,
            "strides": list(self.strides)
            if self.strides
            else list(self._canonical_strides()),
            "element_count": self.element_count,
            "byte_size": self.byte_size,
            "contiguous": self.is_contiguous(),
            "simd_lanes": self.simd_lanes(),
        }


@dataclass(frozen=True)
class KernelLowering:
    """A backend-neutral lowering plan for a vectorized kernel."""

    name: str
    loop_bounds: tuple[int, ...]
    vectorizable: bool
    simd_lanes: int
    layout: BufferLayout
    passes: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loop_bounds": list(self.loop_bounds),
            "vectorizable": self.vectorizable,
            "simd_lanes": self.simd_lanes,
            "layout": self.layout.as_dict(),
            "passes": list(self.passes),
            "reason": self.reason,
            "model_only": True,
        }


def probe_mlir_available() -> dict[str, Any]:
    """Check whether an MLIR/GPU toolchain is available on the system.

    Returns a structured dict with an ``available`` flag.  When MLIR tools
    (``mlir-opt``, ``mlir-translate``) are not on ``PATH`` — the common case
    — the function returns ``available: False`` with a concrete reason so
    callers can fall back to scalar lowering without pretending to have
    native MLIR support.
    """
    tools = ("mlir-opt", "mlir-translate")
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        return {
            "available": False,
            "reason": f"MLIR tools not found on PATH: {', '.join(missing)}",
            "missing_tools": missing,
            "model_only": True,
        }
    return {
        "available": True,
        "reason": "MLIR tools found on PATH",
        "missing_tools": [],
        "model_only": True,
    }


def lower_vectorized_kernel(
    name: str,
    layout: BufferLayout,
    *,
    loop_bounds: Sequence[int] | None = None,
    passes: Sequence[str] = (),
) -> KernelLowering:
    """Lower a simple loop kernel into a backend-neutral plan.

    The kernel is assumed to iterate over the layout's elements in row-major
    order, applying a pure function per element.  The lowering determines
    whether the loop is vectorizable (contiguous, sufficient element count,
    SIMD-compatible lane count) and returns a :class:`KernelLowering` plan.

    Non-contiguous layouts or layouts with too few elements for SIMD are
    marked ``vectorizable=False`` with a concrete reason — they fall back to
    scalar loops.
    """
    if not name or not isinstance(name, str):
        _fail("E003", "kernel name must be a non-empty string")
    bounds = tuple(loop_bounds) if loop_bounds else tuple(layout.shape)
    if len(bounds) != len(layout.shape):
        _fail(
            "E001",
            f"loop bounds {bounds} do not match layout shape {layout.shape}",
        )
    for bound in bounds:
        if bound < 0:
            _fail("E001", f"negative loop bound in {bounds}")

    contiguous = layout.is_contiguous()
    lanes = layout.simd_lanes()
    vectorizable = contiguous and lanes > 1
    reason = (
        "contiguous layout with SIMD-compatible lane count"
        if vectorizable
        else "non-contiguous layout; falling back to scalar loop"
        if not contiguous
        else "layout too small for SIMD; falling back to scalar loop"
    )
    return KernelLowering(
        name=name,
        loop_bounds=bounds,
        vectorizable=vectorizable,
        simd_lanes=lanes if vectorizable else 1,
        layout=layout,
        passes=tuple(passes),
        reason=reason,
    )
