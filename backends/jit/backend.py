"""The Apple Silicon JIT backend.

Source code is lowered to typed SSA (``backends.jit.ir``), register
allocated (``backends.jit.regalloc``), encoded directly to AArch64 machine
code (``backends.jit.codegen_arm64``) and executed out of an ``mmap``ed
``MAP_JIT`` region.

Only the scalar subset of Polyglot is supported: integer/float arithmetic,
casts, ``if``/``while``/``loop``/``for``, integer ``match``, and direct calls
between functions.  Strings, collections, structs, enums, traits, closures
and async functions are rejected with :class:`JITUnsupportedError`.
"""

from __future__ import annotations

import contextlib
import ctypes
from collections.abc import Callable
from typing import Any

from ast_nodes import Program
from backends.base import BackendBase
from backends.jit.codegen_arm64 import CompiledProgram, compile_program
from backends.jit.errors import (
    JITError,
    JITRuntimeError,
    JITUnsupportedError,
)
from backends.jit.ir import (
    BOOL,
    IRType,
    LoweredProgram,
    lower_program,
    normalize_int,
)
from backends.jit.memory import ExecutableMemory, is_supported_host
from monomorph import monomorphize

CONTEXT_BYTES = 64
TRAP_DIVIDE_BY_ZERO = 1
TRAP_SHIFT_OUT_OF_RANGE = 2


def _ctypes_signature(ty: IRType) -> type:
    if ty.is_float:
        return ctypes.c_double
    if ty.is_unsigned:
        return ctypes.c_uint64
    return ctypes.c_int64


def _convert_argument(value: Any, ty: IRType) -> Any:
    if ty.is_float:
        return float(value)
    if ty is BOOL:
        return 1 if value else 0
    if isinstance(value, bool):
        value = int(value)
    return normalize_int(int(value), ty)


def _convert_result(value: Any, ty: IRType) -> Any:
    if ty.is_unit:
        return None
    if ty.is_float:
        return float(value)
    if ty is BOOL:
        return bool(value)
    return int(value)


class JITModule:
    """Executable machine code plus typed Python wrappers for each function."""

    def __init__(self, lowered: LoweredProgram) -> None:
        self.lowered = lowered
        self.context = ctypes.create_string_buffer(CONTEXT_BYTES)
        self.context_address = ctypes.addressof(self.context)
        self.compiled: CompiledProgram = compile_program(
            lowered,
            context_address=self.context_address,
        )
        self.memory: ExecutableMemory = self.compiled.memory
        self.functions: dict[str, Callable[..., Any]] = {}
        for name in lowered.source_order:
            compiled_function = self.compiled.functions[name]
            signature = lowered.signatures[name]
            prototype = ctypes.CFUNCTYPE(
                _ctypes_signature(signature.ret_ty),
                *[_ctypes_signature(ty) for ty in signature.param_types],
            )
            address = self.memory.address + compiled_function.entry_offset
            raw = prototype(address)
            self.functions[name] = self._wrap(name, signature, raw)

    # -- trap handling -----------------------------------------------------

    def _reset_trap(self) -> None:
        ctypes.memset(self.context_address, 0, 8)

    def _read_trap(self) -> int:
        return int(ctypes.c_uint64.from_address(self.context_address).value)

    def _raise_trap(self, code: int) -> None:
        if code == TRAP_DIVIDE_BY_ZERO:
            raise ZeroDivisionError("division by zero")
        if code == TRAP_SHIFT_OUT_OF_RANGE:
            raise JITRuntimeError("shift count out of range (expected 0..63)")
        raise JITRuntimeError(f"JIT trap {code}")

    # -- wrappers ----------------------------------------------------------

    def _wrap(self, name, signature, raw):
        module = self

        def invoke(*args: Any) -> Any:
            if len(args) != len(signature.param_types):
                raise TypeError(
                    f"{name}() takes {len(signature.param_types)} arguments "
                    f"but {len(args)} were given"
                )
            converted = [
                _convert_argument(value, ty)
                for value, ty in zip(args, signature.param_types)
            ]
            module._reset_trap()
            result = raw(*converted)
            trap = module._read_trap()
            if trap:
                module._raise_trap(trap)
            return _convert_result(result, signature.ret_ty)

        invoke.__name__ = name
        invoke.__qualname__ = name
        invoke.__doc__ = f"JIT-compiled Polyglot function `{name}`."
        return invoke

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        self.memory.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        with contextlib.suppress(Exception):
            self.close()


class JITBackend(BackendBase):
    """Compile Polyglot to AArch64 machine code with no external toolchain."""

    name = "jit"

    def __init__(self, *, trusted: bool = True) -> None:
        super().__init__(trusted=trusted)
        self.module: JITModule | None = None

    def compile(self, program: Program) -> dict[str, Any]:
        self._check_trust()
        if not is_supported_host():
            raise JITUnsupportedError(
                "the JIT backend requires macOS on Apple Silicon (arm64)"
            )
        lowered = lower_program(monomorphize(program))
        if not lowered.source_order:
            raise JITUnsupportedError(
                "the JIT backend needs at least one scalar function"
            )
        module = JITModule(lowered)
        self.module = module
        namespace: dict[str, Any] = dict(module.functions)
        namespace.update(lowered.constants)
        return namespace


__all__ = [
    "JITBackend",
    "JITError",
    "JITModule",
    "JITRuntimeError",
    "JITUnsupportedError",
]
