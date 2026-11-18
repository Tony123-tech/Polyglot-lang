"""Versioned, CRC-checked binary framing primitives.

The wire layer is deliberately small and boring: unsigned/signed varints,
fixed-width integers, length-prefixed byte strings, and a frame header.  The
value codec in :mod:`distributed.transmute` builds on these primitives.

Frame layout (network byte order)::

    magic      4 bytes   b"PLDY"
    version    1 byte    format version (currently 1)
    kind       1 byte    payload kind (state, checkpoint, rpc, hello)
    flags      2 bytes   reserved; unknown bits are ignored
    length     4 bytes   payload length in bytes
    payload    N bytes
    crc32      4 bytes   CRC-32 of the payload

Every read is bounds checked.  Truncated or oversized frames raise
:class:`WireError`; the caller converts that into an ``E008`` diagnostic.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Final

MAGIC: Final = b"PLDY"
FORMAT_VERSION: Final = 1
HEADER: Final = struct.Struct(">4sBBHI")
TRAILER: Final = struct.Struct(">I")
HEADER_SIZE: Final = HEADER.size
TRAILER_SIZE: Final = TRAILER.size

KIND_STATE: Final = 1
KIND_CHECKPOINT: Final = 2
KIND_RPC: Final = 3
KIND_HELLO: Final = 4

MAX_FRAME_BYTES: Final = 64 * 1024 * 1024
MAX_FIELD_BYTES: Final = 16 * 1024 * 1024
MAX_VARINT_BYTES: Final = 16
MAX_DEPTH: Final = 256


class WireError(ValueError):
    """A malformed or unsupported wire payload."""

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        self.offset = offset
        self.reason = message
        if offset is not None:
            message = f"{message} (offset {offset})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class Writer:
    """A growable big-endian byte writer with varint helpers."""

    __slots__ = ("buf",)

    def __init__(self) -> None:
        self.buf = bytearray()

    # -- fixed width ------------------------------------------------------
    def u8(self, value: int) -> None:
        if not 0 <= value <= 0xFF:
            raise WireError(f"u8 out of range: {value}")
        self.buf.append(value)

    def u16(self, value: int) -> None:
        if not 0 <= value <= 0xFFFF:
            raise WireError(f"u16 out of range: {value}")
        self.buf += struct.pack(">H", value)

    def u32(self, value: int) -> None:
        if not 0 <= value <= 0xFFFFFFFF:
            raise WireError(f"u32 out of range: {value}")
        self.buf += struct.pack(">I", value)

    def u64(self, value: int) -> None:
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise WireError(f"u64 out of range: {value}")
        self.buf += struct.pack(">Q", value)

    def f64(self, value: float) -> None:
        self.buf += struct.pack(">d", value)

    # -- varints ----------------------------------------------------------
    def varint(self, value: int) -> None:
        """Write a non-negative integer as an unsigned LEB128 varint."""
        if value < 0:
            raise WireError(f"varint requires a non-negative value: {value}")
        while value > 0x7F:
            self.buf.append((value & 0x7F) | 0x80)
            value >>= 7
        self.buf.append(value)

    def svarint(self, value: int) -> None:
        """Write a signed integer using zig-zag + unsigned varint."""
        zigzag = (value << 1) if value >= 0 else ((-value << 1) - 1)
        self.varint(zigzag)

    # -- length-prefixed --------------------------------------------------
    def raw(self, value: bytes | bytearray | memoryview) -> None:
        if len(value) > MAX_FIELD_BYTES:
            raise WireError(
                f"field is too large: {len(value)} bytes (limit {MAX_FIELD_BYTES})"
            )
        self.varint(len(value))
        self.buf += value

    def string(self, value: str) -> None:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise WireError(f"string is not valid UTF-8: {error}") from error
        self.raw(encoded)

    # -- patching ---------------------------------------------------------
    def patch_u32(self, position: int, value: int) -> None:
        if not 0 <= position <= len(self.buf) - 4:
            raise WireError(f"cannot patch at offset {position}")
        struct.pack_into(">I", self.buf, position, value)

    def build(self) -> bytes:
        return bytes(self.buf)

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class Reader:
    """A bounds-checked big-endian reader."""

    __slots__ = ("data", "max_bytes", "pos")

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        max_bytes: int = MAX_FRAME_BYTES,
    ) -> None:
        length = len(data)
        if length > max_bytes:
            raise WireError(f"payload is too large: {length} bytes (limit {max_bytes})")
        self.data = memoryview(data)
        self.pos = 0
        self.max_bytes = max_bytes

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def _take(self, count: int) -> memoryview:
        if count < 0:
            raise WireError(f"negative read length: {count}", offset=self.pos)
        end = self.pos + count
        if end > len(self.data):
            raise WireError(
                f"truncated payload: need {count} byte(s), have {self.remaining}",
                offset=self.pos,
            )
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self._take(8))[0]

    def f64(self) -> float:
        return struct.unpack(">d", self._take(8))[0]

    def varint(self) -> int:
        result = 0
        shift = 0
        for index in range(MAX_VARINT_BYTES):
            byte = self.u8()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
        raise WireError(f"varint exceeds {MAX_VARINT_BYTES} bytes", offset=self.pos)

    def svarint(self) -> int:
        value = self.varint()
        if value & 1:
            return -((value >> 1) + 1)
        return value >> 1

    def raw(self) -> bytes:
        length = self.varint()
        if length > MAX_FIELD_BYTES:
            raise WireError(
                f"field is too large: {length} bytes (limit {MAX_FIELD_BYTES})",
                offset=self.pos,
            )
        return bytes(self._take(length))

    def string(self) -> str:
        encoded = self.raw()
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WireError(
                f"invalid UTF-8 string: {error}", offset=self.pos
            ) from error

    def expect_end(self) -> None:
        if self.remaining:
            raise WireError(f"{self.remaining} trailing byte(s)", offset=self.pos)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    kind: int
    payload: bytes
    version: int = FORMAT_VERSION
    flags: int = 0

    @property
    def digest(self) -> int:
        return zlib.crc32(self.payload) & 0xFFFFFFFF


def encode_frame(
    kind: int,
    payload: bytes,
    *,
    flags: int = 0,
    version: int = FORMAT_VERSION,
) -> bytes:
    if not 0 <= kind <= 0xFF:
        raise WireError(f"frame kind out of range: {kind}")
    if not 0 <= flags <= 0xFFFF:
        raise WireError(f"frame flags out of range: {flags}")
    if len(payload) > MAX_FRAME_BYTES:
        raise WireError(
            f"frame payload is too large: {len(payload)} bytes "
            f"(limit {MAX_FRAME_BYTES})"
        )
    header = HEADER.pack(MAGIC, version, kind, flags, len(payload))
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    return b"".join((header, payload, crc))


def decode_frame(data: bytes | bytearray | memoryview) -> Frame:
    view = memoryview(data)
    if len(view) < HEADER_SIZE + TRAILER_SIZE:
        raise WireError(f"frame is too short: {len(view)} byte(s)", offset=0)
    magic, version, kind, flags, length = HEADER.unpack(view[:HEADER_SIZE])
    if magic != MAGIC:
        raise WireError(
            f"bad frame magic {bytes(magic)!r}; expected {MAGIC!r}", offset=0
        )
    if version > FORMAT_VERSION:
        raise WireError(
            f"frame format v{version} is newer than supported v{FORMAT_VERSION}",
            offset=4,
        )
    if length > MAX_FRAME_BYTES:
        raise WireError(
            f"frame payload is too large: {length} bytes (limit {MAX_FRAME_BYTES})",
            offset=6,
        )
    expected = HEADER_SIZE + length + TRAILER_SIZE
    if len(view) != expected:
        raise WireError(
            f"frame length mismatch: header says {expected} byte(s), got {len(view)}",
            offset=6,
        )
    payload = bytes(view[HEADER_SIZE : HEADER_SIZE + length])
    (crc,) = TRAILER.unpack(view[HEADER_SIZE + length :])
    actual = zlib.crc32(payload) & 0xFFFFFFFF
    if crc != actual:
        raise WireError(
            f"frame CRC mismatch: expected {actual:#010x}, got {crc:#010x}",
            offset=HEADER_SIZE + length,
        )
    return Frame(kind=kind, payload=payload, version=version, flags=flags)


def peek_kind(data: bytes | bytearray | memoryview) -> int | None:
    """Return the frame kind without validating the payload."""
    view = memoryview(data)
    if len(view) < HEADER_SIZE:
        return None
    return view[5]


__all__ = [
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "KIND_CHECKPOINT",
    "KIND_HELLO",
    "KIND_RPC",
    "KIND_STATE",
    "MAGIC",
    "MAX_DEPTH",
    "MAX_FIELD_BYTES",
    "MAX_FRAME_BYTES",
    "TRAILER_SIZE",
    "Frame",
    "Reader",
    "WireError",
    "Writer",
    "decode_frame",
    "encode_frame",
    "peek_kind",
]
