"""Apple Silicon executable-memory management for the JIT backend.

macOS on Apple Silicon enforces W^X at the hardware level.  A page that is
mapped with ``MAP_JIT`` starts writable (in the thread-local JIT write mode)
and must be toggled to executable with
``pthread_jit_write_protect_np(1)`` before it is called.  The instruction
cache is not coherent with the data cache, so every write must be followed by
``sys_icache_invalidate``.

This module is deliberately dependency-free: everything goes through
``ctypes`` and ``mmap`` from libSystem.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import struct
from dataclasses import dataclass, field
from typing import Self

# ---------------------------------------------------------------------------
# Constants (macOS <sys/mman.h>)
# ---------------------------------------------------------------------------

PROT_NONE = 0x00
PROT_READ = 0x01
PROT_WRITE = 0x02
PROT_EXEC = 0x04

MAP_SHARED = 0x0001
MAP_PRIVATE = 0x0002
MAP_FIXED = 0x0010
MAP_ANON = 0x1000  # MAP_ANONYMOUS on macOS
MAP_JIT = 0x0800

PAGE_SIZE = int(os.sysconf("SC_PAGESIZE")) if hasattr(os, "sysconf") else 4096

_MAP_FAILED = ctypes.c_void_p(-1).value

# ---------------------------------------------------------------------------
# libSystem bindings
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_mprotect = _libc.mprotect
_mprotect.restype = ctypes.c_int
_mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def _load_libsystem() -> ctypes.CDLL:
    candidates = (
        "/usr/lib/libSystem.B.dylib",
        "/usr/lib/system/libsystem_kernel.dylib",
    )
    for path in candidates:
        try:
            return ctypes.CDLL(path, use_errno=True)
        except OSError:  # pragma: no cover - platform dependent
            continue
    return _libc


_libsystem = _load_libsystem()

_pthread_jit_write_protect_np = getattr(
    _libsystem,
    "pthread_jit_write_protect_np",
    None,
)
if _pthread_jit_write_protect_np is not None:
    _pthread_jit_write_protect_np.restype = None
    _pthread_jit_write_protect_np.argtypes = [ctypes.c_int]

_sys_icache_invalidate = getattr(_libsystem, "sys_icache_invalidate", None)
if _sys_icache_invalidate is not None:
    _sys_icache_invalidate.restype = None
    _sys_icache_invalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


class JITMemoryError(RuntimeError):
    """Raised when executable memory cannot be mapped or committed."""


def is_supported_host() -> bool:
    """Return True on macOS/arm64, the only target supported by this JIT."""
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (value + alignment - 1) & ~(alignment - 1)


def jit_write_protect(enabled: bool) -> None:
    """Toggle the calling thread's JIT write permission (Apple Silicon)."""
    if _pthread_jit_write_protect_np is not None:
        _pthread_jit_write_protect_np(1 if enabled else 0)


def invalidate_icache(address: int, size: int) -> None:
    """Invalidate the instruction cache for ``address .. address+size``."""
    if size <= 0:
        return
    if _sys_icache_invalidate is not None:
        _sys_icache_invalidate(ctypes.c_void_p(address), ctypes.c_size_t(size))
    else:  # pragma: no cover - non-Apple fallback
        # Best effort: clearing the cache line is not available portably.
        ctypes.memset(ctypes.c_void_p(address), 0, 0)


@dataclass
class ExecutableMemory:
    """A single executable region that owns its mapping.

    The object keeps the ``mmap`` base address and size alive.  Callers must
    keep a reference to the instance for as long as any ``ctypes`` function
    pointer created from :meth:`address` can be invoked.
    """

    size: int
    address: int = field(init=False)
    _closed: bool = field(default=False, init=False)
    _write_mode: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        self.size = align_up(self.size, PAGE_SIZE)
        flags = MAP_PRIVATE | MAP_ANON | MAP_JIT
        # MAP_JIT must be requested with PROT_READ|PROT_WRITE|PROT_EXEC on
        # macOS.  The per-thread write protect call is what enforces W^X.
        addr = _libc.mmap(
            None,
            self.size,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            flags,
            -1,
            0,
        )
        if addr is None or addr == _MAP_FAILED:
            errno = ctypes.get_errno()
            raise JITMemoryError(
                f"mmap(MAP_JIT) failed for {self.size} bytes: errno {errno}"
            )
        self.address = int(addr)

    # -- ownership ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        if self._write_mode:
            jit_write_protect(True)
            self._write_mode = False
        _libc.munmap(ctypes.c_void_p(self.address), self.size)
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    # -- writing -----------------------------------------------------------

    def begin_write(self) -> None:
        if self._closed:
            raise JITMemoryError("executable memory is closed")
        jit_write_protect(False)
        self._write_mode = True

    def end_write(self, *, size: int | None = None) -> None:
        if self._closed:
            raise JITMemoryError("executable memory is closed")
        invalidate_icache(self.address, self.size if size is None else size)
        jit_write_protect(True)
        self._write_mode = False

    def write(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        if self._closed:
            raise JITMemoryError("executable memory is closed")
        payload = bytes(data)
        if offset < 0 or offset + len(payload) > self.size:
            raise JITMemoryError(
                f"write of {len(payload)} bytes at {offset} exceeds region "
                f"of {self.size} bytes"
            )
        if not self._write_mode:
            self.begin_write()
        ctypes.memmove(self.address + offset, payload, len(payload))

    def write_words(self, offset: int, words: list[int] | tuple[int, ...]) -> None:
        self.write(offset, b"".join(struct.pack("<I", word & 0xFFFFFFFF) for word in words))

    def commit(self, *, size: int | None = None) -> None:
        """Finish a write transaction and invalidate the icache."""
        self.end_write(size=size)

    # -- reading / calling -------------------------------------------------

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise JITMemoryError("read outside executable region")
        return ctypes.string_at(self.address + offset, size)

    def address_at(self, offset: int = 0) -> int:
        if offset < 0 or offset > self.size:
            raise JITMemoryError("offset outside executable region")
        return self.address + offset

    def function_pointer(self, offset: int, prototype: type[ctypes._CFuncPtr]):  # type: ignore[attr-defined]
        """Return a ctypes callable for ``prototype`` at ``offset``.

        The returned object does not keep ``self`` alive; callers must hold a
        reference to this ``ExecutableMemory`` (the JIT namespace does).
        """
        return prototype(self.address_at(offset))


def allocate_executable(data: bytes) -> ExecutableMemory:
    """Convenience helper used by unit tests and small stubs."""
    memory = ExecutableMemory(max(PAGE_SIZE, len(data)))
    memory.write(0, data)
    memory.commit(size=max(PAGE_SIZE, len(data)))
    return memory
