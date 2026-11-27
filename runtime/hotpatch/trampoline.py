"""Atomic trampolines for live shared libraries and owned JIT pages.

Two relocation paths are implemented:

* **GOT/PLT slots** - the 8-byte pointer cell for an already-bound symbol is
  located by value-scanning the Mach-O ``__got``/``__la_symbol_ptr``/
  ``__auth_got`` sections, or writable mappings on Linux, for the value
  returned by ``dlsym``.  This is not a full ELF ``DT_JMPREL`` relocation walk
  and does not cover chained fixups.  The page is made writable, the aligned
  pointer is stored with a single 8-byte write, and the original protection is
  restored.  A thread that is executing the old target continues safely; the
  next indirect call sees the new pointer.
* **Inline JMP sleds** - ``ldr x16, #8; br x16; .quad target`` on AArch64 and
  ``mov rax, imm64; jmp rax`` on x86_64.  Sleds are only written into
  :class:`ExecutableRegion` pages owned by this process (``MAP_JIT`` on Apple
  Silicon), never into foreign ``__TEXT``.

Foreign executable text is refused with ``E007``: rewriting a page that the
dynamic loader mapped read-execute (and that may be code-signed/shared) cannot
be done safely or portably from a user-space module.
"""

from __future__ import annotations

import ctypes
import dataclasses
import mmap
import platform
import struct
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, Self

from runtime.hotpatch.errors import HotpatchError, abi_error, relocation_error

_PAGE = mmap.PAGESIZE
PROT_READ = 1
PROT_WRITE = 2
PROT_EXEC = 4
PROT_COPY = 0x10
MAP_PRIVATE = 0x0002
MAP_ANON_MACOS = 0x1000
MAP_ANON_LINUX = 0x20
MAP_JIT = 0x0800
VM_PROT_READ = 0x1
VM_PROT_WRITE = 0x2
VM_PROT_COPY = 0x10
_DARWIN = sys.platform == "darwin"
# ``RTLD_DEFAULT`` is not a portable constant: dyld spells it -2, while glibc
# spells it NULL and reads every other handle as a ``link_map`` pointer. Handing
# glibc the macOS value dereferences 0xfffffffffffffffe, so ``dlsym`` does not
# return NULL for an unknown symbol there - it kills the process with SIGSEGV.
RTLD_DEFAULT = ctypes.c_void_p(-2 if _DARWIN else 0)
_MACHINE = platform.machine().lower()
_AARCH64 = _MACHINE in {"arm64", "aarch64"}
_X86_64 = _MACHINE in {"x86_64", "amd64"}

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.dlsym.restype = ctypes.c_void_p
libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


def _bind_optional(name: str, restype: Any, argtypes: list[Any]) -> Any:
    try:
        function = getattr(libc, name)
    except AttributeError:
        return None
    function.restype = restype
    function.argtypes = argtypes
    return function


_mach_task_self = None
_mach_vm_protect = None
_jit_write_protect = None
_jit_write_supported = None
_icache_invalidate = None
if _DARWIN:
    try:
        _mach_task_self = ctypes.c_int.in_dll(libc, "mach_task_self_")
    except ValueError:  # pragma: no cover - non-standard libc
        _mach_task_self = None
    _mach_vm_protect = _bind_optional(
        "mach_vm_protect",
        ctypes.c_int,
        [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
        ],
    )
    _jit_write_protect = _bind_optional(
        "pthread_jit_write_protect_np", None, [ctypes.c_int]
    )
    _jit_write_supported = _bind_optional(
        "pthread_jit_write_protect_supported_np", ctypes.c_int, []
    )
    _icache_invalidate = _bind_optional(
        "sys_icache_invalidate", None, [ctypes.c_void_p, ctypes.c_size_t]
    )
else:
    _icache_invalidate = _bind_optional(
        "__clear_cache", None, [ctypes.c_void_p, ctypes.c_void_p]
    )


# ---------------------------------------------------------------------------
# Signature handling
# ---------------------------------------------------------------------------

_SIGNATURE_TYPES: dict[str, Any] = {
    "void": None,
    "bool": ctypes.c_bool,
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u8": ctypes.c_uint8,
    "u16": ctypes.c_uint16,
    "u32": ctypes.c_uint32,
    "u64": ctypes.c_uint64,
    "isize": ctypes.c_ssize_t,
    "usize": ctypes.c_size_t,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "ptr": ctypes.c_void_p,
}

# The host's fixed-width and pointer-sized ctypes can be the same object (for
# example ``c_int64`` and ``c_ssize_t`` are both ``c_long`` on macOS).  Prefer
# the explicit fixed-width spelling so signatures round-trip as written.
_CTYPE_NAMES: dict[Any, str] = {None: "void"}
for _name in (
    "bool",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f32",
    "f64",
    "ptr",
    "isize",
    "usize",
):
    _CTYPE_NAMES.setdefault(_SIGNATURE_TYPES[_name], _name)


def parse_signature(signature: str) -> tuple[Any, list[Any]]:
    """Parse ``"i64(i64, ptr)"`` into ``(restype, argtypes)``."""
    text = signature.replace(" ", "")
    opening = text.find("(")
    if opening == -1 or not text.endswith(")"):
        raise abi_error(
            f"invalid ABI signature {signature!r}",
            detail="expected 'return(args...)' such as 'i64(i64, ptr)'",
        )
    restype_name = text[:opening]
    args_text = text[opening + 1 : -1]
    if restype_name not in _SIGNATURE_TYPES:
        raise abi_error(
            f"unknown return type {restype_name!r} in {signature!r}",
        )
    argtypes: list[Any] = []
    for argument in filter(None, args_text.split(",")):
        if argument not in _SIGNATURE_TYPES:
            raise abi_error(f"unknown argument type {argument!r} in {signature!r}")
        ctype = _SIGNATURE_TYPES[argument]
        argtypes.append(ctype if ctype is not None else ctypes.c_void_p)
    return _SIGNATURE_TYPES[restype_name], argtypes


def signature_text(procedure: Any) -> str:
    """Return a normalised signature for a ``CFUNCTYPE`` instance."""
    restype = _CTYPE_NAMES.get(getattr(procedure, "restype", None), "?")
    args = ", ".join(_CTYPE_NAMES.get(arg, "?") for arg in procedure.argtypes or ())
    return f"{restype}({args})"


def wrap_callable(function: Callable[..., Any], signature: str) -> tuple[Any, Any]:
    """Wrap a Python callable in the C signature and keep it alive."""
    restype, argtypes = parse_signature(signature)
    prototype = ctypes.CFUNCTYPE(restype, *argtypes)
    procedure = prototype(function)
    return procedure, procedure


# ---------------------------------------------------------------------------
# Protection and atomic writes
# ---------------------------------------------------------------------------


def _page_protection(address: int) -> int | None:
    """Best-effort query of the current page protection."""
    start = address & ~(_PAGE - 1)
    if not _DARWIN:
        for mapping in _read_proc_maps():
            mapped_start, mapped_end, _path, writable = mapping
            if mapped_start <= start < mapped_end:
                return PROT_READ | (PROT_WRITE if writable else 0)
        return None
    if _mach_vm_protect is None or _mach_task_self is None:
        return None

    class VmRegionBasicInfo64(ctypes.Structure):
        _fields_ = [
            ("protection", ctypes.c_int),
            ("max_protection", ctypes.c_int),
            ("inheritance", ctypes.c_int),
            ("shared", ctypes.c_int),
            ("reserved", ctypes.c_int),
            ("offset", ctypes.c_uint64),
            ("behavior", ctypes.c_int),
            ("user_wired_count", ctypes.c_uint16),
        ]

    query = _bind_optional(
        "mach_vm_region",
        ctypes.c_int,
        [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ],
    )
    if query is None:
        return None
    region_address = ctypes.c_uint64(start)
    region_size = ctypes.c_uint64(0)
    info = VmRegionBasicInfo64()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
    object_name = ctypes.c_uint32(0)
    result = query(
        _mach_task_self.value,
        ctypes.byref(region_address),
        ctypes.byref(region_size),
        9,  # VM_REGION_BASIC_INFO_64
        ctypes.byref(info),
        ctypes.byref(count),
        ctypes.byref(object_name),
    )
    if result != 0:
        return None
    return info.protection


def _make_writable(address: int, size: int) -> Callable[[], None]:
    """Make a page writable and return a restore callback."""
    start = address & ~(_PAGE - 1)
    span = max(size + (address - start), 1)
    original = _page_protection(start)
    if libc.mprotect(ctypes.c_void_p(start), span, PROT_READ | PROT_WRITE) == 0:
        restore_protection = (
            original if original is not None else PROT_READ | PROT_WRITE
        )
        return lambda: libc.mprotect(ctypes.c_void_p(start), span, restore_protection)
    if _DARWIN and _mach_vm_protect is not None and _mach_task_self is not None:
        result = _mach_vm_protect(
            _mach_task_self.value,
            start,
            span,
            0,
            VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY,
        )
        if result == 0:  # KERN_SUCCESS
            restore_protection = original if original is not None else VM_PROT_READ
            return lambda: _mach_vm_protect(
                _mach_task_self.value, start, span, 0, restore_protection
            )
    raise HotpatchError(
        f"cannot make page {start:#x} writable",
        code="E007",
        reason="mprotect and mach_vm_protect both refused the relocation page",
        help_text="the image may be code-signed, read-only shared cache text, or SIP-protected",
    )


def _check_pointer_cell(address: int) -> None:
    """Refuse to dereference an address that is not a mapped, aligned cell.

    The trampoline API accepts raw addresses from embedders.  Passing a bogus
    integer to ``ctypes.c_void_p.from_address`` would segfault the interpreter,
    so validate against the process mappings first and report a structured
    ``E007`` instead.  There is an inherent TOCTOU window, but it closes the
    common case of a stale or mistyped address.
    """
    if address <= 0:
        raise relocation_error("pointer cell address is NULL or negative")
    if address % struct.calcsize("P") != 0:
        raise relocation_error(f"pointer cell {address:#x} is not word-aligned")
    if _page_protection(address) is None:
        raise HotpatchError(
            f"pointer cell {address:#x} is not in a mapped page",
            code="E007",
            reason="the address does not belong to a live mapping in this process",
            help_text="pass an address from dlsym, an owned buffer, or an mmap",
        )


def atomic_load_ptr(address: int) -> int:
    _check_pointer_cell(address)
    return ctypes.c_void_p.from_address(address).value or 0


def atomic_store_ptr(address: int, value: int) -> None:
    """Store an aligned pointer with one machine-word write."""
    _check_pointer_cell(address)
    ctypes.c_void_p.from_address(address).value = value


def flush_instruction_cache(address: int, size: int) -> None:
    if _icache_invalidate is None:
        return
    if _DARWIN:
        _icache_invalidate(ctypes.c_void_p(address), size)
    else:
        _icache_invalidate(ctypes.c_void_p(address), ctypes.c_void_p(address + size))


def encode_jump(target: int) -> bytes:
    """Encode an absolute jump sled for the host architecture."""
    if _AARCH64:
        # ldr x16, #8 ; br x16 ; .quad target
        return struct.pack("<IIQ", 0x58000050, 0xD61F0200, target)
    if _X86_64:
        # mov rax, imm64 ; jmp rax
        return b"\x48\xb8" + struct.pack("<Q", target) + b"\xff\xe0"
    raise HotpatchError(
        f"inline trampolines are not implemented for {_MACHINE}",
        code="E007",
        reason="only x86_64 and AArch64 jump encodings are provided",
        help_text="use the GOT/PLT slot path on this architecture",
    )


# ---------------------------------------------------------------------------
# Owned executable pages
# ---------------------------------------------------------------------------


class ExecutableRegion:
    """A page range that this process owns and may rewrite."""

    def __init__(self, size: int = _PAGE, *, name: str = "hotpatch") -> None:
        self.name = name
        self.size = max(_PAGE, ((size + _PAGE - 1) // _PAGE) * _PAGE)
        self._jit = False
        self._address = 0
        self._closed = False
        if _DARWIN and _AARCH64 and _jit_write_protect is not None:
            self._address = self._map(PROT_READ | PROT_WRITE | PROT_EXEC, jit=True)
            if self._address:
                self._jit = True
        if not self._address:
            self._address = self._map(PROT_READ | PROT_WRITE, jit=False)
            if not self._address:
                raise HotpatchError(
                    "cannot allocate an executable page",
                    code="E007",
                    reason=f"mmap failed with errno {ctypes.get_errno()}",
                    help_text="the process may forbid anonymous executable memory",
                )
        if not self._jit and (
            libc.mprotect(
                ctypes.c_void_p(self._address), self.size, PROT_READ | PROT_EXEC
            )
            != 0
        ):
            self.close()
            raise HotpatchError(
                "cannot mark the page executable",
                code="E007",
                reason=f"mprotect failed with errno {ctypes.get_errno()}",
            )
        self._ctypes_buffer = (ctypes.c_char * self.size).from_address(self._address)

    def _map(self, prot: int, *, jit: bool) -> int:
        flags = MAP_PRIVATE | (_DARWIN and MAP_ANON_MACOS or MAP_ANON_LINUX)
        if jit:
            flags |= MAP_JIT
        address = libc.mmap(None, self.size, prot, flags, -1, 0)
        if address in (0, ctypes.c_void_p(-1).value):
            return 0
        return int(address)

    @property
    def address(self) -> int:
        return self._address

    def __int__(self) -> int:
        return self._address

    def write(self, offset: int, data: bytes) -> None:
        if self._closed:
            raise HotpatchError("executable region is closed")
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise relocation_error(
                f"write of {len(data)} bytes at offset {offset} exceeds {self.size}",
            )
        if self._jit and _jit_write_protect is not None:
            _jit_write_protect(0)
            try:
                ctypes.memmove(self._address + offset, data, len(data))
            finally:
                flush_instruction_cache(self._address + offset, len(data))
                _jit_write_protect(1)
            return
        restore = _make_executable_writable(self._address, self.size)
        try:
            ctypes.memmove(self._address + offset, data, len(data))
        finally:
            restore()
            flush_instruction_cache(self._address + offset, len(data))

    def read(self, offset: int, size: int) -> bytes:
        return ctypes.string_at(self._address + offset, size)

    def clear(self) -> None:
        self.write(0, b"\x00" * self.size)

    def function(self, signature: str = "i64()") -> Any:
        restype, argtypes = parse_signature(signature)
        prototype = ctypes.CFUNCTYPE(restype, *argtypes)
        return prototype(self._address)

    def close(self) -> None:
        if not self._closed and self._address:
            libc.munmap(ctypes.c_void_p(self._address), self.size)
            self._address = 0
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def available(cls) -> bool:
        try:
            region = cls(_PAGE, name="probe")
        except HotpatchError, OSError:
            return False
        region.close()
        return True


def _make_executable_writable(address: int, size: int) -> Callable[[], None]:
    if libc.mprotect(ctypes.c_void_p(address), size, PROT_READ | PROT_WRITE) != 0:
        raise HotpatchError(
            "cannot reopen the executable page for writing",
            code="E007",
            reason=f"mprotect failed with errno {ctypes.get_errno()}",
        )
    return lambda: libc.mprotect(ctypes.c_void_p(address), size, PROT_READ | PROT_EXEC)


# ---------------------------------------------------------------------------
# Mach-O / ELF GOT discovery
# ---------------------------------------------------------------------------

_LC_SEGMENT_64 = 0x19
_GOT_SECTIONS = {"__got", "__la_symbol_ptr", "__auth_got", "__symbol_ptr"}


@dataclasses.dataclass(frozen=True)
class MachOImage:
    index: int
    path: str
    header: int
    slide: int
    sections: tuple[tuple[str, str, int, int], ...]

    def got_sections(self) -> Iterator[tuple[str, str, int, int]]:
        for segname, sectname, addr, size in self.sections:
            if sectname in _GOT_SECTIONS:
                yield segname, sectname, addr, size


def _dyld_images() -> list[MachOImage]:
    if not _DARWIN:
        return []
    count = libc._dyld_image_count
    count.restype = ctypes.c_uint32
    header_of = libc._dyld_get_image_header
    header_of.restype = ctypes.c_void_p
    header_of.argtypes = [ctypes.c_uint32]
    slide_of = libc._dyld_get_image_vmaddr_slide
    slide_of.restype = ctypes.c_int64
    slide_of.argtypes = [ctypes.c_uint32]
    name_of = libc._dyld_get_image_name
    name_of.restype = ctypes.c_char_p
    name_of.argtypes = [ctypes.c_uint32]
    images: list[MachOImage] = []
    for index in range(count()):
        header = header_of(index)
        if not header:
            continue
        raw = ctypes.string_at(header, 32)
        if struct.unpack_from("<I", raw, 0)[0] != 0xFEEDFACF:
            continue
        ncmds = struct.unpack_from("<I", raw, 16)[0]
        offset = 32
        sections: list[tuple[str, str, int, int]] = []
        # Cap the walk: a corrupt load-command chain must not walk off the
        # mapped header image.  Real dylibs have tens of commands at most.
        for _ in range(min(ncmds, 1024)):
            if offset + 8 > 4096:
                break
            command, command_size = struct.unpack_from(
                "<II", ctypes.string_at(header + offset, 8), 0
            )
            if command == _LC_SEGMENT_64:
                if command_size < 72 or offset + command_size > 4096:
                    break
                segment = ctypes.string_at(header + offset, 72)
                nsects = struct.unpack_from("<I", segment, 64)[0]
                section_offset = offset + 72
                # Each section header is 80 bytes within this load command.
                for _section in range(min(nsects, 512)):
                    if section_offset + 80 > offset + command_size:
                        break
                    section = ctypes.string_at(header + section_offset, 80)
                    sectname = section[:16].rstrip(b"\x00").decode(errors="replace")
                    segname = section[16:32].rstrip(b"\x00").decode(errors="replace")
                    address, size = struct.unpack_from("<QQ", section, 32)
                    sections.append((segname, sectname, address, size))
                    section_offset += 80
            if command_size <= 0:
                break
            offset += command_size
        path = (name_of(index) or b"").decode(errors="replace")
        images.append(
            MachOImage(
                index=index,
                path=path,
                header=int(header),
                slide=int(slide_of(index)),
                sections=tuple(sections),
            )
        )
    return images


@dataclasses.dataclass(frozen=True)
class GotSlot:
    address: int
    symbol: str
    value: int
    image: str
    segment: str
    section: str

    def patch(self, replacement: int, *, span: tuple[int, int] | None = None) -> Any:
        return _patch_slot(self, replacement, span=span)


def resolve_symbol(symbol: str) -> int | None:
    value = libc.dlsym(RTLD_DEFAULT, symbol.encode())
    return int(value) if value else None


def find_got_slots(
    symbol: str,
    *,
    library: str | None = None,
    span: tuple[int, int] | None = None,
) -> list[GotSlot]:
    """Find already-bound GOT/PLT cells whose value is ``symbol``."""
    target = resolve_symbol(symbol)
    if target is None:
        raise relocation_error(
            f"symbol {symbol!r} is not resolvable in this process",
            span=span,
            detail="dlsym(RTLD_DEFAULT) returned NULL",
        )
    slots: list[GotSlot] = []
    seen: set[int] = set()
    if _DARWIN:
        for image in _dyld_images():
            if library is not None and library not in image.path:
                continue
            for segname, sectname, address, size in image.got_sections():
                base = image.slide + address
                count = min(size // 8, 1_000_000)
                for index in range(count):
                    slot = base + index * 8
                    if (
                        ctypes.c_void_p.from_address(slot).value == target
                        and slot not in seen
                    ):
                        seen.add(slot)
                        slots.append(
                            GotSlot(
                                slot,
                                symbol,
                                target,
                                image.path,
                                segname,
                                sectname,
                            )
                        )
    else:
        slots.extend(
            _elf_got_slots(symbol, target, library=library, span=span, seen=seen)
        )
    return slots


def _elf_got_slots(
    symbol: str,
    target: int,
    *,
    library: str | None,
    span: tuple[int, int] | None,
    seen: set[int],
) -> list[GotSlot]:
    """Locate Linux GOT-like cells by scanning writable mappings.

    ``symbol`` is retained for diagnostics and API symmetry with the Mach-O
    path; this fallback matches the bound pointer value from ``dlsym`` rather
    than parsing ELF relocations.  The macOS path above is the one exercised
    by the default test suite.
    """
    slots: list[GotSlot] = []
    # A full DT_JMPREL walk (including chained fixups) is intentionally out of
    # scope; only writable maps belonging to the requested library are scanned.
    for mapping in _read_proc_maps():
        start, end, path, writable = mapping
        if not writable:
            continue
        if library is not None and library not in path:
            continue
        for slot in range(start, end - 7, 8):
            if ctypes.c_void_p.from_address(slot).value == target and slot not in seen:
                seen.add(slot)
                slots.append(GotSlot(slot, symbol, target, path, "", ""))
    if not slots:
        raise relocation_error(
            f"no bound GOT/PLT slot found for {symbol!r}",
            span=span,
            detail="the symbol may still be lazily bound or resolved via a chained fixup",
        )
    return slots


def _read_proc_maps() -> list[tuple[int, int, str, bool]]:
    entries: list[tuple[int, int, str, bool]] = []
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) < 2:
                    continue
                start_text, end_text = fields[0].split("-")
                permissions = fields[1]
                path = fields[-1] if len(fields) >= 6 else ""
                entries.append(
                    (
                        int(start_text, 16),
                        int(end_text, 16),
                        path,
                        "w" in permissions,
                    )
                )
    except OSError as error:  # pragma: no cover - non-Linux fallback
        raise HotpatchError(
            "cannot inspect process mappings",
            code="E007",
            reason=str(error),
        ) from None
    return entries


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrampolinePatch:
    slot: int
    old_value: int
    new_value: int
    symbol: str
    image: str = ""
    section: str = ""
    span: tuple[int, int] | None = None
    keepalive: Any = None
    _active: bool = True

    @property
    def kind(self) -> str:
        """An indirect GOT/PLT or dispatch-slot swap."""
        return "pointer"

    @property
    def active(self) -> bool:
        return self._active

    def rollback(self) -> bool:
        if not self._active:
            return False
        restore = _make_writable(self.slot, 8)
        try:
            atomic_store_ptr(self.slot, self.old_value)
        finally:
            restore()
        self._active = False
        return True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()


@dataclasses.dataclass
class TextPatch:
    region: ExecutableRegion
    offset: int
    original: bytes
    target: int
    span: tuple[int, int] | None = None
    _active: bool = True

    @property
    def kind(self) -> str:
        """An inline jump sled inside an owned executable region."""
        return "sled"

    @property
    def active(self) -> bool:
        return self._active

    def rollback(self) -> bool:
        if not self._active:
            return False
        self.region.write(self.offset, self.original)
        self._active = False
        return True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()


def _patch_slot(
    slot: GotSlot,
    replacement: int,
    *,
    span: tuple[int, int] | None = None,
    keepalive: Any = None,
) -> TrampolinePatch:
    if replacement == 0:
        raise relocation_error(
            "replacement pointer is NULL", span=span, detail="refusing to call NULL"
        )
    old_value = atomic_load_ptr(slot.address)
    restore = _make_writable(slot.address, 8)
    try:
        atomic_store_ptr(slot.address, replacement)
    finally:
        restore()
    return TrampolinePatch(
        slot=slot.address,
        old_value=old_value,
        new_value=replacement,
        symbol=slot.symbol,
        image=slot.image,
        section=slot.section,
        span=span,
        keepalive=keepalive,
    )


class LiveTrampoline:
    """Discover GOT cells and rewrite them, or patch owned JIT pages."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._patches: list[TrampolinePatch | TextPatch] = []
        self.atomic_swaps = 0

    def discover(self, symbol: str, *, library: str | None = None) -> list[GotSlot]:
        return find_got_slots(symbol, library=library)

    def slot(self, symbol: str, *, library: str | None = None) -> GotSlot:
        slots = self.discover(symbol, library=library)
        if not slots:
            raise relocation_error(
                f"no GOT/PLT slot found for {symbol!r}",
                detail="the symbol is not bound through a writable pointer cell",
            )
        if len({slot.address for slot in slots}) > 1:
            raise relocation_error(
                f"symbol {symbol!r} has multiple GOT slots",
                detail=f"candidates: {[hex(slot.address) for slot in slots]}",
            )
        return slots[0]

    def patch_symbol(
        self,
        symbol: str,
        replacement: Callable[..., Any] | int,
        *,
        signature: str | None = None,
        library: str | None = None,
        span: tuple[int, int] | None = None,
        source: str | None = None,
    ) -> TrampolinePatch:
        slot = self.slot(symbol, library=library)
        pointer, keepalive = _replacement_pointer(
            replacement, signature, span=span, source=source
        )
        with self._lock:
            patch = _patch_slot(slot, pointer, span=span, keepalive=keepalive)
            self._patches.append(patch)
            self.atomic_swaps += 1
            return patch

    def patch_slot(
        self,
        address: int,
        replacement: Callable[..., Any] | int,
        *,
        signature: str | None = None,
        symbol: str = "slot",
        span: tuple[int, int] | None = None,
        source: str | None = None,
    ) -> TrampolinePatch:
        pointer, keepalive = _replacement_pointer(
            replacement, signature, span=span, source=source
        )
        slot = GotSlot(int(address), symbol, atomic_load_ptr(address), "", "", "")
        with self._lock:
            patch = _patch_slot(slot, pointer, span=span, keepalive=keepalive)
            self._patches.append(patch)
            self.atomic_swaps += 1
            return patch

    def patch_text(
        self,
        region: ExecutableRegion,
        target: Callable[..., Any] | int,
        *,
        offset: int = 0,
        span: tuple[int, int] | None = None,
    ) -> TextPatch:
        if not isinstance(region, ExecutableRegion):
            raise HotpatchError(
                "inline sleds may only be written into an owned ExecutableRegion",
                code="E007",
                span=span,
                reason="foreign __TEXT pages are code-signed/shared and not safely writable",
                help_text="patch a GOT/PLT slot instead, or allocate a JIT region",
            )
        target_address = (
            target
            if isinstance(target, int)
            else ctypes.cast(
                _replacement_pointer(target, None, span=span)[0], ctypes.c_void_p
            ).value
        )
        if target_address is None:
            raise relocation_error("replacement has no address", span=span)
        sled = encode_jump(int(target_address))
        original = region.read(offset, len(sled))
        with self._lock:
            region.write(offset, sled)
            patch = TextPatch(region, offset, original, int(target_address), span)
            self._patches.append(patch)
            return patch

    def rollback_all(self) -> int:
        restored = 0
        with self._lock:
            for patch in reversed(self._patches):
                if patch.rollback():
                    restored += 1
            self._patches = [patch for patch in self._patches if patch._active]
        return restored

    @property
    def patches(self) -> tuple[TrampolinePatch | TextPatch, ...]:
        with self._lock:
            return tuple(self._patches)


def _signature_matches(procedure: Any, expected: str) -> bool:
    restype, argtypes = parse_signature(expected)
    return (procedure.restype, list(procedure.argtypes or ())) == (restype, argtypes)


def _replacement_pointer(
    replacement: Callable[..., Any] | int,
    signature: str | None,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> tuple[int, Any]:
    if isinstance(replacement, int):
        return replacement, None
    if isinstance(replacement, ctypes._CFuncPtr):  # type: ignore[attr-defined]
        if signature is not None and not _signature_matches(replacement, signature):
            raise abi_error(
                "replacement ABI signature does not match the expected signature",
                span=span,
                source=source,
                detail=(
                    f"expected {signature.replace(' ', '')}, "
                    f"got {signature_text(replacement)}"
                ),
            )
        return ctypes.cast(replacement, ctypes.c_void_p).value or 0, replacement
    if signature is None:
        raise abi_error(
            "a Python callable replacement requires an explicit ABI signature",
            span=span,
            source=source,
            detail="pass signature='i64(i64)' or a CFUNCTYPE instance",
        )
    procedure, keepalive = wrap_callable(replacement, signature)
    return ctypes.cast(procedure, ctypes.c_void_p).value or 0, keepalive


def hotpatch_live_dylib_trampoline(
    symbol: str | None = None,
    replacement: Callable[..., Any] | int | None = None,
    *,
    slot: int | None = None,
    region: ExecutableRegion | None = None,
    offset: int = 0,
    signature: str | None = None,
    expected_signature: str | None = None,
    library: str | None = None,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> TrampolinePatch | TextPatch:
    """Patch a bound GOT/PLT cell or write an inline sled into an owned page.

    * ``symbol`` + ``replacement``: atomically rewrite the symbol's GOT cell.
    * ``slot`` + ``replacement``: rewrite an explicit pointer cell (tests and
      embedders that own the dispatch table).
    * ``region`` + ``replacement``: write an absolute jump sled at ``offset``.

    ``expected_signature`` is compared against a ``CFUNCTYPE`` replacement or
    against ``signature`` when a Python callable is wrapped.
    """
    trampoline = LiveTrampoline()
    if expected_signature is not None:
        signature = signature or expected_signature
        if isinstance(
            replacement,
            ctypes._CFuncPtr,  # type: ignore[attr-defined]
        ) and not _signature_matches(replacement, expected_signature):
            raise abi_error(
                "replacement ABI signature does not match expected_signature",
                span=span,
                source=source,
                detail=(
                    f"expected {expected_signature.replace(' ', '')}, "
                    f"got {signature_text(replacement)}"
                ),
            )
    if region is not None:
        if replacement is None:
            raise relocation_error("region patching requires a replacement", span=span)
        return trampoline.patch_text(region, replacement, offset=offset, span=span)
    if replacement is None:
        raise relocation_error(
            "no replacement was supplied",
            span=span,
            detail="pass a callable, CFUNCTYPE instance, or absolute address",
        )
    if slot is not None:
        return trampoline.patch_slot(
            slot,
            replacement,
            signature=signature,
            span=span,
            source=source,
        )
    if symbol is None:
        raise relocation_error(
            "no symbol, slot, or region was supplied",
            span=span,
            detail="specify where the trampoline should live",
        )
    return trampoline.patch_symbol(
        symbol,
        replacement,
        signature=signature,
        library=library,
        span=span,
        source=source,
    )


__all__ = [
    "ExecutableRegion",
    "GotSlot",
    "LiveTrampoline",
    "TextPatch",
    "TrampolinePatch",
    "atomic_load_ptr",
    "atomic_store_ptr",
    "encode_jump",
    "find_got_slots",
    "flush_instruction_cache",
    "hotpatch_live_dylib_trampoline",
    "parse_signature",
    "resolve_symbol",
    "signature_text",
    "wrap_callable",
]
