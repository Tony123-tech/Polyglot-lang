"""Direct AArch64 (ARM64) machine-code generation for the JIT backend.

No external assembler is invoked: every instruction is bit-packed into a
32-bit little-endian word with ``struct.pack("<I", word)`` and emitted into
executable memory allocated by :mod:`backends.jit.memory`.

The generated code follows AAPCS64:

* integer arguments in ``X0``-``X7``, floating-point arguments in ``D0``-``D7``
* results in ``X0`` / ``D0``
* ``X19``-``X27`` and ``D8``-``D15`` are callee-saved and used for temporaries
* ``X29`` is the frame pointer, ``X30`` the link register
* ``X16``/``X17`` are intra-procedure scratch registers
* ``X28`` is reserved to point at the trap context (division by zero, ...)

The frame is allocated once in the prologue, so ``SP`` stays 16-byte aligned
at every call site.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from backends.jit.errors import JITCompileError, JITUnsupportedError
from backends.jit.ir import (
    BOOL,
    BasicBlock,
    BinOp,
    BrCond,
    Call,
    Cast,
    Cmp,
    Const,
    IRFunction,
    IRType,
    Jmp,
    LocalGet,
    LocalSet,
    LoweredProgram,
    Nop,
    Phi,
    Ret,
    UnOp,
    Unreachable,
    Value,
)
from backends.jit.memory import ExecutableMemory, align_up
from backends.jit.regalloc import (
    Allocation,
    Location,
    allocate,
    parallel_copy,
)

MASK64 = (1 << 64) - 1

# ---------------------------------------------------------------------------
# Condition codes
# ---------------------------------------------------------------------------

EQ = 0
NE = 1
HS = 2
LO = 3
MI = 4
PL = 5
HI = 8
LS = 9
GE = 10
LT = 11
GT = 12
LE = 13

SIGNED_COND = {"==": EQ, "!=": NE, "<": LT, "<=": LE, ">": GT, ">=": GE}
UNSIGNED_COND = {"==": EQ, "!=": NE, "<": LO, "<=": LS, ">": HI, ">=": HS}
FLOAT_COND = {"==": EQ, "!=": NE, "<": MI, "<=": LS, ">": GT, ">=": GE}
INVERT_COND = {
    EQ: NE,
    NE: EQ,
    HS: LO,
    LO: HS,
    MI: PL,
    PL: MI,
    HI: LS,
    LS: HI,
    GE: LT,
    LT: GE,
    GT: LE,
    LE: GT,
}


def _r(name: str) -> int:
    """Register number for ``X0``-``X30`` / ``D0``-``D31`` / ``SP``."""
    if name in {"SP", "XZR", "WZR", "X31"}:
        return 31
    if name in {"FP", "X29"}:
        return 29
    if name in {"LR", "X30"}:
        return 30
    if name[:1] in {"X", "W", "D", "S", "B", "H", "Q"}:
        return int(name[1:])
    raise JITCompileError(f"unknown register {name!r}")


# ---------------------------------------------------------------------------
# Instruction encoders
# ---------------------------------------------------------------------------


def movz(rd: int, imm16: int, shift: int = 0) -> int:
    """MOVZ Xd, #imm16, LSL #shift (64-bit)."""
    assert shift % 16 == 0 and 0 <= shift <= 48
    return 0xD2800000 | ((shift // 16) << 21) | ((imm16 & 0xFFFF) << 5) | rd


def movk(rd: int, imm16: int, shift: int = 0) -> int:
    """MOVK Xd, #imm16, LSL #shift (64-bit)."""
    assert shift % 16 == 0 and 0 <= shift <= 48
    return 0xF2800000 | ((shift // 16) << 21) | ((imm16 & 0xFFFF) << 5) | rd


def mov_imm(rd: int, value: int) -> list[int]:
    """Materialize an arbitrary 64-bit value into ``Xd``."""
    value &= MASK64
    words = [movz(rd, value & 0xFFFF)]
    for shift in (16, 32, 48):
        chunk = (value >> shift) & 0xFFFF
        if chunk:
            words.append(movk(rd, chunk, shift))
    return words


def mov_reg(rd: int, rm: int) -> int:
    """MOV Xd, Xm (ORR Xd, XZR, Xm)."""
    return 0xAA0003E0 | (rm << 16) | rd


def add_reg(rd: int, rn: int, rm: int) -> int:
    return 0x8B000000 | (rm << 16) | (rn << 5) | rd


def sub_reg(rd: int, rn: int, rm: int) -> int:
    return 0xCB000000 | (rm << 16) | (rn << 5) | rd


def add_ext(rd: int, rn: int, rm: int) -> int:
    """ADD Xd, Xn, Xm (extended register form; Rn may be SP)."""
    return 0x8B200000 | (rm << 16) | (3 << 13) | (rn << 5) | rd


def sub_ext(rd: int, rn: int, rm: int) -> int:
    """SUB Xd, Xn, Xm (extended register form; Rn may be SP)."""
    return 0xCB200000 | (rm << 16) | (3 << 13) | (rn << 5) | rd


def add_imm(rd: int, rn: int, imm12: int) -> int:
    assert 0 <= imm12 <= 0xFFF
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd


def sub_imm(rd: int, rn: int, imm12: int) -> int:
    assert 0 <= imm12 <= 0xFFF
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd


def and_reg(rd: int, rn: int, rm: int) -> int:
    return 0x8A000000 | (rm << 16) | (rn << 5) | rd


def orr_reg(rd: int, rn: int, rm: int) -> int:
    return 0xAA000000 | (rm << 16) | (rn << 5) | rd


def eor_reg(rd: int, rn: int, rm: int) -> int:
    return 0xCA000000 | (rm << 16) | (rn << 5) | rd


def mul(rd: int, rn: int, rm: int) -> int:
    return 0x9B007C00 | (rm << 16) | (rn << 5) | rd


def sdiv(rd: int, rn: int, rm: int) -> int:
    return 0x9AC00C00 | (rm << 16) | (rn << 5) | rd


def udiv(rd: int, rn: int, rm: int) -> int:
    return 0x9AC00800 | (rm << 16) | (rn << 5) | rd


def msub(rd: int, rn: int, rm: int, ra: int) -> int:
    """MSUB Xd, Xn, Xm, Xa  =>  Xd = Xa - Xn * Xm."""
    return 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd


def lslv(rd: int, rn: int, rm: int) -> int:
    return 0x9AC02000 | (rm << 16) | (rn << 5) | rd


def lsrv(rd: int, rn: int, rm: int) -> int:
    return 0x9AC02400 | (rm << 16) | (rn << 5) | rd


def asrv(rd: int, rn: int, rm: int) -> int:
    return 0x9AC02800 | (rm << 16) | (rn << 5) | rd


def cmp_reg(rn: int, rm: int) -> int:
    """CMP Xn, Xm  =>  SUBS XZR, Xn, Xm."""
    return 0xEB00001F | (rm << 16) | (rn << 5)


def cmp_imm(rn: int, imm12: int) -> int:
    """CMP Xn, #imm12  =>  SUBS XZR, Xn, #imm12."""
    assert 0 <= imm12 <= 0xFFF
    return 0xF100001F | (imm12 << 10) | (rn << 5)


def cset(rd: int, cond: int) -> int:
    """CSET Xd, cond  =>  CSINC Xd, XZR, XZR, invert(cond)."""
    return 0x9A9F07E0 | (INVERT_COND[cond] << 12) | rd


def sxtb(rd: int, rn: int) -> int:
    return 0x93401C00 | (rn << 5) | rd


def sxth(rd: int, rn: int) -> int:
    return 0x93403C00 | (rn << 5) | rd


def sxtw(rd: int, rn: int) -> int:
    return 0x93407C00 | (rn << 5) | rd


def uxtb(rd: int, rn: int) -> int:
    return 0x53001C00 | (rn << 5) | rd


def uxth(rd: int, rn: int) -> int:
    return 0x53003C00 | (rn << 5) | rd


def uxtw(rd: int, rn: int) -> int:
    return 0xD3407C00 | (rn << 5) | rd


def fadd(rd: int, rn: int, rm: int) -> int:
    return 0x1E602800 | (rm << 16) | (rn << 5) | rd


def fsub(rd: int, rn: int, rm: int) -> int:
    return 0x1E603800 | (rm << 16) | (rn << 5) | rd


def fmul(rd: int, rn: int, rm: int) -> int:
    return 0x1E600800 | (rm << 16) | (rn << 5) | rd


def fdiv(rd: int, rn: int, rm: int) -> int:
    return 0x1E601800 | (rm << 16) | (rn << 5) | rd


def fneg(rd: int, rn: int) -> int:
    return 0x1E614000 | (rn << 5) | rd


def fmov_dd(rd: int, rn: int) -> int:
    return 0x1E604000 | (rn << 5) | rd


def fmov_dx(rd: int, rn: int) -> int:
    """FMOV Dd, Xn (bit pattern move)."""
    return 0x9E670000 | (rn << 5) | rd


def fcmp_reg(rn: int, rm: int) -> int:
    return 0x1E602000 | (rm << 16) | (rn << 5)


def fcmp_zero(rn: int) -> int:
    return 0x1E602008 | (rn << 5)


def scvtf(rd: int, rn: int) -> int:
    """SCVTF Dd, Xn."""
    return 0x9E620000 | (rn << 5) | rd


def ucvtf(rd: int, rn: int) -> int:
    return 0x9E630000 | (rn << 5) | rd


def fcvtzs(rd: int, rn: int) -> int:
    """FCVTZS Xd, Dn."""
    return 0x9E780000 | (rn << 5) | rd


def fcvtzu(rd: int, rn: int) -> int:
    return 0x9E790000 | (rn << 5) | rd


_LOAD_KINDS = {
    "x": (0xF9400000, 3),
    "w": (0xB9400000, 2),
    "b": (0x39400000, 0),
    "h": (0x79400000, 1),
    "sb": (0x39800000, 0),
    "sh": (0x79800000, 1),
    "sw": (0xB9800000, 2),
    "d": (0xFD400000, 3),
}

_STORE_KINDS = {
    "x": (0xF9000000, 3),
    "w": (0xB9000000, 2),
    "b": (0x39000000, 0),
    "h": (0x79000000, 1),
    "d": (0xFD000000, 3),
}


def ldst(base_word: int, scale: int, rt: int, rn: int, offset: int) -> int:
    assert offset % (1 << scale) == 0
    imm12 = offset >> scale
    assert 0 <= imm12 <= 0xFFF
    return base_word | (imm12 << 10) | (rn << 5) | rt


def stp(rt: int, rt2: int, rn: int, offset: int, *, fp: bool = False) -> int:
    """STP Rt, Rt2, [Rn, #offset] with a signed 7-bit scaled offset."""
    assert offset % 8 == 0 and -512 <= offset <= 504
    base = 0x6D000000 if fp else 0xA9000000
    imm7 = (offset // 8) & 0x7F
    return base | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt


def ldp(rt: int, rt2: int, rn: int, offset: int, *, fp: bool = False) -> int:
    assert offset % 8 == 0 and -512 <= offset <= 504
    base = 0x6D400000 if fp else 0xA9400000
    imm7 = (offset // 8) & 0x7F
    return base | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt


def stp_pre(rt: int, rt2: int, rn: int, offset: int) -> int:
    assert offset % 8 == 0 and -512 <= offset <= 504
    imm7 = (offset // 8) & 0x7F
    return 0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt


def ldp_post(rt: int, rt2: int, rn: int, offset: int) -> int:
    assert offset % 8 == 0 and -512 <= offset <= 504
    imm7 = (offset // 8) & 0x7F
    return 0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt


# ---------------------------------------------------------------------------
# Assembler with local label fixups
# ---------------------------------------------------------------------------


class Assembler:
    """Collect 32-bit words and patch PC-relative branch fixups."""

    def __init__(self) -> None:
        self.words: list[int] = []
        self._labels: dict[object, int] = {}
        self._fixups: list[tuple[int, object, str, int, int]] = []

    # -- emission ----------------------------------------------------------

    def emit(self, word: int) -> int:
        self.words.append(word & 0xFFFFFFFF)
        return len(self.words) - 1

    def emit_all(self, words: list[int]) -> None:
        for word in words:
            self.emit(word)

    def here(self) -> int:
        return len(self.words)

    def label(self) -> object:
        return object()

    def place(self, label: object) -> None:
        self._labels[label] = len(self.words)

    # -- branches ----------------------------------------------------------

    def b(self, label: object) -> int:
        return self._fixup(0x14000000, label, "b")

    def bcond(self, cond: int, label: object) -> int:
        return self._fixup(0x54000000 | cond, label, "bcond")

    def cbz(self, rt: int, label: object) -> int:
        return self._fixup(0xB4000000 | rt, label, "cbz")

    def cbnz(self, rt: int, label: object) -> int:
        return self._fixup(0xB5000000 | rt, label, "cbnz")

    def _fixup(self, base: int, label: object, kind: str) -> int:
        index = self.emit(base)
        self._fixups.append((index, label, kind, 0, 0))
        return index

    def resolve(self) -> None:
        for index, label, kind, _rt, _cond in self._fixups:
            if label not in self._labels:
                raise JITCompileError("internal error: unresolved machine-code label")
            delta = self._labels[label] - index
            if kind == "b":
                self.words[index] = 0x14000000 | (delta & 0x3FFFFFF)
            elif kind == "bcond":
                base = self.words[index] & 0xF
                self.words[index] = 0x54000000 | ((delta & 0x7FFFF) << 5) | base
            else:
                base = self.words[index] & 0x1F
                opcode = 0xB4000000 if kind == "cbz" else 0xB5000000
                self.words[index] = opcode | ((delta & 0x7FFFF) << 5) | base

    def bytes(self) -> bytes:
        self.resolve()
        return b"".join(struct.pack("<I", word) for word in self.words)


# ---------------------------------------------------------------------------
# Per-function code generation
# ---------------------------------------------------------------------------


def _slot_kind(ty: IRType) -> str:
    if ty.is_float:
        return "d"
    if ty is BOOL:
        return "x"
    if ty.bits == 8:
        return "sb" if ty.signed else "b"
    if ty.bits == 16:
        return "sh" if ty.signed else "h"
    if ty.bits == 32:
        return "sw" if ty.signed else "w"
    return "x"


def _store_kind(load_kind: str) -> str:
    return {"sb": "b", "sh": "h", "sw": "w"}.get(load_kind, load_kind)


class FunctionCodegen:
    def __init__(self, function: IRFunction, context_address: int) -> None:
        self.function = function
        self.context_address = context_address
        self.asm = Assembler()
        self.alloc: Allocation = allocate(function)

        self.has_float = any(value.ty.is_float for value in self.alloc.intervals)
        self.save_area = 80 + (64 if self.has_float else 0)
        self.slot_offsets = {
            slot: self.save_area + index * 8
            for index, slot in enumerate(function.slots)
        }
        self.spill_base = self.save_area + len(function.slots) * 8
        self.frame_size = align_up(
            self.spill_base + self.alloc.spill_count * 8,
            16,
        )

        self.block_labels = {block: self.asm.label() for block in function.blocks}
        self.body_label = self.asm.label()
        self.trap_label = self.asm.label()
        self.epilogue_label = self.asm.label()

        self.call_fixups: list[tuple[int, str]] = []

    # -- location helpers --------------------------------------------------

    def _location(self, value: Value) -> Location:
        interval = self.alloc.intervals.get(value)
        if interval is not None and interval.reg is not None:
            return Location(reg=interval.reg)
        if interval is not None and interval.spill is not None:
            return Location(spill=self.spill_base + interval.spill * 8)
        if value.fixed_reg is not None:
            return Location(reg=value.fixed_reg)
        raise JITCompileError(
            f"internal error: value {value!r} has no register or spill slot"
        )

    def _int_reg(self, value: Value, scratch: str = "X16") -> int:
        location = self._location(value)
        if location.reg is not None:
            return _r(location.reg)
        assert location.spill is not None
        self._load(location.spill, _r(scratch), "x")
        return _r(scratch)

    def _float_reg(self, value: Value, scratch: str = "D16") -> int:
        location = self._location(value)
        if location.reg is not None:
            return _r(location.reg)
        assert location.spill is not None
        self._load(location.spill, _r(scratch), "d")
        return _r(scratch)

    def _dest(self, value: Value, scratch: int = 0) -> tuple[int, bool]:
        location = self._location(value)
        if location.reg is not None:
            return _r(location.reg), False
        return scratch, True

    def _write_int(self, value: Value, reg: int) -> None:
        location = self._location(value)
        if location.spill is not None:
            self._store(reg, location.spill, "x")
        elif _r(location.reg) != reg:  # type: ignore[arg-type]
            self.asm.emit(mov_reg(_r(location.reg), reg))  # type: ignore[arg-type]

    def _write_float(self, value: Value, reg: int) -> None:
        location = self._location(value)
        if location.spill is not None:
            self._store(reg, location.spill, "d")
        elif _r(location.reg) != reg:  # type: ignore[arg-type]
            self.asm.emit(fmov_dd(_r(location.reg), reg))  # type: ignore[arg-type]

    # -- memory ------------------------------------------------------------

    def _address(self, scratch: int, base: int, offset: int) -> None:
        """Materialize ``SP + offset`` (or ``X28 + offset``) in ``scratch``."""
        if base == 31:
            if 0 <= offset <= 0xFFF:
                self.asm.emit(add_imm(scratch, 31, offset))
            else:
                self.asm.emit_all(mov_imm(scratch, offset & MASK64))
                self.asm.emit(add_ext(scratch, 31, scratch))
        else:
            if 0 <= offset <= 0xFFF:
                self.asm.emit(add_imm(scratch, base, offset))
            else:
                self.asm.emit_all(mov_imm(scratch, offset & MASK64))
                self.asm.emit(add_ext(scratch, base, scratch))

    def _mem(
        self,
        *,
        load: bool,
        kind: str,
        rt: int,
        base: int,
        offset: int,
    ) -> None:
        table = _LOAD_KINDS if load else _STORE_KINDS
        base_word, scale = table[kind]
        if 0 <= offset and offset % (1 << scale) == 0 and (offset >> scale) <= 0xFFF:
            self.asm.emit(ldst(base_word, scale, rt, base, offset))
            return
        scratch = 16 if rt != 16 else 17
        self._address(scratch, base, offset)
        self.asm.emit(ldst(base_word, scale, rt, scratch, 0))

    def _load(self, offset: int, rt: int, kind: str) -> None:
        self._mem(load=True, kind=kind, rt=rt, base=31, offset=offset)

    def _store(self, rt: int, offset: int, kind: str) -> None:
        self._mem(load=False, kind=kind, rt=rt, base=31, offset=offset)

    # -- prologue / epilogue ----------------------------------------------

    def _emit_stub(self) -> None:
        self.asm.emit_all(mov_imm(28, self.context_address))
        self.asm.b(self.body_label)

    def _emit_prologue(self) -> None:
        self.asm.place(self.body_label)
        self.asm.emit(stp_pre(29, 30, 31, -16))
        self.asm.emit(add_imm(29, 31, 0))
        self._allocate_frame(self.frame_size)
        for index in range(0, 10, 2):
            self.asm.emit(stp(19 + index, 20 + index, 31, index * 8))
        if self.has_float:
            for index in range(0, 8, 2):
                self.asm.emit(stp(8 + index, 9 + index, 31, 80 + index * 8, fp=True))

    def _allocate_frame(self, size: int) -> None:
        if size == 0:
            return
        if size <= 0xFFF:
            self.asm.emit(sub_imm(31, 31, size))
        else:
            self.asm.emit_all(mov_imm(16, size))
            self.asm.emit(sub_ext(31, 31, 16))

    def _emit_epilogue(self) -> None:
        self.asm.place(self.trap_label)
        self.asm.emit(movz(0, 0))
        self.asm.place(self.epilogue_label)
        for index in range(0, 10, 2):
            self.asm.emit(ldp(19 + index, 20 + index, 31, index * 8))
        if self.has_float:
            for index in range(0, 8, 2):
                self.asm.emit(ldp(8 + index, 9 + index, 31, 80 + index * 8, fp=True))
        self.asm.emit(add_imm(31, 29, 0))
        self.asm.emit(ldp_post(29, 30, 31, 16))
        self.asm.emit(0xD65F03C0)  # RET

    def _emit_trap(self, code: int) -> None:
        self.asm.emit_all(mov_imm(16, code))
        self.asm.emit(0xF9000000 | (28 << 5) | 16)  # STR X16, [X28]
        self.asm.b(self.trap_label)

    def _emit_call_trap_check(self) -> None:
        self.asm.emit(0xF9400000 | (28 << 5) | 16)  # LDR X16, [X28]
        self.asm.cbnz(16, self.trap_label)

    # -- instructions ------------------------------------------------------

    def _emit_const(self, inst: Const) -> None:
        value = inst.value
        if value.ty.is_unit:
            return
        if value.ty.is_float:
            bits = struct.unpack("<Q", struct.pack("<d", float(inst.const)))[0]
            destination, spilled = self._dest(value)
            self.asm.emit_all(mov_imm(16, bits))
            self.asm.emit(fmov_dx(destination, 16))
            if spilled:
                self._write_float(value, destination)
            return
        destination, spilled = self._dest(value)
        self.asm.emit_all(mov_imm(destination, int(inst.const) & MASK64))
        if spilled:
            self._write_int(value, destination)

    def _emit_binop(self, inst: BinOp) -> None:
        result = inst.result
        if result.ty.is_float:
            left = self._float_reg(inst.left, "D16")
            right = self._float_reg(inst.right, "D17")
            destination, spilled = self._dest(result)
            if inst.op == "+":
                self.asm.emit(fadd(destination, left, right))
            elif inst.op == "-":
                self.asm.emit(fsub(destination, left, right))
            elif inst.op == "*":
                self.asm.emit(fmul(destination, left, right))
            elif inst.op == "/":
                self.asm.emit(fdiv(destination, left, right))
            else:
                raise JITUnsupportedError(
                    f"JIT does not support floating-point operator `{inst.op}`"
                )
            if spilled:
                self._write_float(result, destination)
            return

        left = self._int_reg(inst.left, "X16")
        right = self._int_reg(inst.right, "X17")
        destination, spilled = self._dest(result)
        op = inst.op

        if op in {"/", "%"}:
            ok = self.asm.label()
            self.asm.cbnz(right, ok)
            self._emit_trap(1)
            self.asm.place(ok)

        if op == "+":
            self.asm.emit(add_reg(destination, left, right))
        elif op == "-":
            self.asm.emit(sub_reg(destination, left, right))
        elif op == "*":
            self.asm.emit(mul(destination, left, right))
        elif op == "/":
            divider = udiv if result.ty.is_unsigned else sdiv
            self.asm.emit(divider(destination, left, right))
        elif op == "%":
            divider = udiv if result.ty.is_unsigned else sdiv
            self.asm.emit(divider(0, left, right))
            self.asm.emit(msub(destination, 0, right, left))
        elif op == "&":
            self.asm.emit(and_reg(destination, left, right))
        elif op == "|":
            self.asm.emit(orr_reg(destination, left, right))
        elif op == "^":
            self.asm.emit(eor_reg(destination, left, right))
        elif op in {"<<", ">>"}:
            ok = self.asm.label()
            self.asm.emit(cmp_imm(right, 64))
            self.asm.bcond(LO, ok)
            self._emit_trap(2)
            self.asm.place(ok)
            if op == "<<":
                self.asm.emit(lslv(destination, left, right))
            elif result.ty.is_unsigned:
                self.asm.emit(lsrv(destination, left, right))
            else:
                self.asm.emit(asrv(destination, left, right))
        else:
            raise JITUnsupportedError(f"JIT does not support operator `{op}`")

        if spilled:
            self._write_int(result, destination)

    def _emit_unop(self, inst: UnOp) -> None:
        if inst.op == "neg" and inst.result.ty.is_float:
            operand = self._float_reg(inst.operand, "D16")
            destination, spilled = self._dest(inst.result)
            self.asm.emit(fneg(destination, operand))
            if spilled:
                self._write_float(inst.result, destination)
            return
        raise JITUnsupportedError(f"JIT does not support unary `{inst.op}`")

    def _emit_cmp(self, inst: Cmp) -> None:
        destination, spilled = self._dest(inst.result)
        if inst.operand_ty.is_float:
            left = self._float_reg(inst.left, "D16")
            right = self._float_reg(inst.right, "D17")
            self.asm.emit(fcmp_reg(left, right))
            self.asm.emit(cset(destination, FLOAT_COND[inst.op]))
        else:
            left = self._int_reg(inst.left, "X16")
            right = self._int_reg(inst.right, "X17")
            self.asm.emit(cmp_reg(left, right))
            table = UNSIGNED_COND if inst.operand_ty.is_unsigned else SIGNED_COND
            self.asm.emit(cset(destination, table[inst.op]))
        if spilled:
            self._write_int(inst.result, destination)

    def _emit_cast(self, inst: Cast) -> None:
        target = inst.target
        source = inst.source
        destination, spilled = self._dest(inst.result)

        if source.is_float and target.is_float:
            operand = self._float_reg(inst.value, "D16")
            self.asm.emit(fmov_dd(destination, operand))
        elif source.is_float:
            operand = self._float_reg(inst.value, "D16")
            convert = fcvtzu if target.is_unsigned else fcvtzs
            self.asm.emit(convert(destination, operand))
        elif target.is_float:
            operand = self._int_reg(inst.value, "X16")
            convert = ucvtf if source.is_unsigned else scvtf
            self.asm.emit(convert(destination, operand))
        elif target.is_bool:
            operand = self._int_reg(inst.value, "X16")
            self.asm.emit(cmp_imm(operand, 0))
            self.asm.emit(cset(destination, NE))
        else:
            operand = self._int_reg(inst.value, "X16")
            if target.bits == 8:
                self.asm.emit(
                    sxtb(destination, operand)
                    if target.signed
                    else uxtb(destination, operand)
                )
            elif target.bits == 16:
                self.asm.emit(
                    sxth(destination, operand)
                    if target.signed
                    else uxth(destination, operand)
                )
            elif target.bits == 32:
                self.asm.emit(
                    sxtw(destination, operand)
                    if target.signed
                    else uxtw(destination, operand)
                )
            elif destination != operand:
                self.asm.emit(mov_reg(destination, operand))

        if spilled:
            if target.is_float:
                self._write_float(inst.result, destination)
            else:
                self._write_int(inst.result, destination)

    def _emit_local_get(self, inst: LocalGet) -> None:
        slot = inst.slot
        offset = self.slot_offsets[slot]
        kind = _slot_kind(slot.ty)
        if slot.ty.is_float:
            destination, spilled = self._dest(inst.result)
            self._load(offset, destination, kind)
            if spilled:
                self._write_float(inst.result, destination)
        else:
            destination, spilled = self._dest(inst.result)
            self._load(offset, destination, kind)
            if spilled:
                self._write_int(inst.result, destination)

    def _emit_local_set(self, inst: LocalSet) -> None:
        slot = inst.slot
        offset = self.slot_offsets[slot]
        kind = _store_kind(_slot_kind(slot.ty))
        if slot.ty.is_unit:
            return
        if slot.ty.is_float:
            source = self._float_reg(inst.value, "D16")
            self._store(source, offset, kind)
        else:
            source = self._int_reg(inst.value, "X16")
            self._store(source, offset, kind)

    def _emit_call(self, inst: Call) -> None:
        int_index = 0
        float_index = 0
        for arg, arg_type in zip(inst.args, inst.arg_types):
            if arg_type.is_float:
                source = self._float_reg(arg, "D17")
                if source != float_index:
                    self.asm.emit(fmov_dd(float_index, source))
                float_index += 1
            else:
                source = self._int_reg(arg, "X17")
                if source != int_index:
                    self.asm.emit(mov_reg(int_index, source))
                int_index += 1
        index = self.asm.emit(0x94000000)  # BL, patched after layout
        self.call_fixups.append((index, inst.name))
        self._emit_call_trap_check()
        if inst.result is not None:
            if inst.result.ty.is_float:
                destination, spilled = self._dest(inst.result)
                if destination != 0:
                    self.asm.emit(fmov_dd(destination, 0))
                if spilled:
                    self._write_float(inst.result, destination)
            else:
                self._write_int(inst.result, 0)

    def _emit_inst(self, inst) -> None:
        if isinstance(inst, Const):
            self._emit_const(inst)
        elif isinstance(inst, BinOp):
            self._emit_binop(inst)
        elif isinstance(inst, UnOp):
            self._emit_unop(inst)
        elif isinstance(inst, Cmp):
            self._emit_cmp(inst)
        elif isinstance(inst, Cast):
            self._emit_cast(inst)
        elif isinstance(inst, LocalGet):
            self._emit_local_get(inst)
        elif isinstance(inst, LocalSet):
            self._emit_local_set(inst)
        elif isinstance(inst, Call):
            self._emit_call(inst)
        elif isinstance(inst, (Phi, Nop)):
            return
        else:  # pragma: no cover - defensive
            raise JITCompileError(
                f"JIT codegen does not handle instruction {type(inst).__name__}"
            )

    # -- phi copies --------------------------------------------------------

    def _emit_phi_copies(self, block: BasicBlock) -> None:
        successors = IRFunction.successors(block)
        if len(successors) != 1:
            return
        successor = successors[0]
        phis = [inst for inst in successor.insts if isinstance(inst, Phi)]
        if not phis:
            return

        int_moves: list[tuple[Location, Location]] = []
        float_moves: list[tuple[Location, Location]] = []
        for phi in phis:
            for incoming_block, value in phi.incoming:
                if incoming_block is not block:
                    continue
                move = (self._location(value), self._location(phi.result))
                (float_moves if phi.result.ty.is_float else int_moves).append(move)

        self._emit_moves(int_moves, scratch_reg=16, load_reg=17, float_moves=False)
        self._emit_moves(float_moves, scratch_reg=16, load_reg=17, float_moves=True)

    def _emit_moves(
        self,
        moves: list[tuple[Location, Location]],
        *,
        scratch_reg: int,
        load_reg: int,
        float_moves: bool,
    ) -> None:
        if not moves:
            return
        scheduled = parallel_copy(
            moves,
            Location(reg=f"D{scratch_reg}" if float_moves else f"X{scratch_reg}"),
        )
        for source, destination in scheduled:
            if source.reg is not None:
                value_reg = _r(source.reg)
            else:
                assert source.spill is not None
                kind = "d" if float_moves else "x"
                self._load(source.spill, load_reg, kind)
                value_reg = load_reg
            if destination.spill is not None:
                kind = "d" if float_moves else "x"
                self._store(value_reg, destination.spill, kind)
            else:
                assert destination.reg is not None
                target_reg = _r(destination.reg)
                if target_reg != value_reg:
                    if float_moves:
                        self.asm.emit(fmov_dd(target_reg, value_reg))
                    else:
                        self.asm.emit(mov_reg(target_reg, value_reg))

    # -- terminators -------------------------------------------------------

    def _emit_terminator(self, block: BasicBlock) -> None:
        terminator = block.terminator
        if isinstance(terminator, Jmp):
            self.asm.b(self.block_labels[terminator.target])
        elif isinstance(terminator, BrCond):
            cond = self._int_reg(terminator.cond, "X16")
            self.asm.cbnz(cond, self.block_labels[terminator.true_block])
            self.asm.b(self.block_labels[terminator.false_block])
        elif isinstance(terminator, Ret):
            if terminator.value is not None and not self.function.ret_ty.is_unit:
                if self.function.ret_ty.is_float:
                    source = self._float_reg(terminator.value, "D16")
                    if source != 0:
                        self.asm.emit(fmov_dd(0, source))
                else:
                    source = self._int_reg(terminator.value, "X16")
                    if source != 0:
                        self.asm.emit(mov_reg(0, source))
            self.asm.b(self.epilogue_label)
        elif terminator is None or isinstance(terminator, Unreachable):
            self.asm.b(self.epilogue_label)
        else:  # pragma: no cover - defensive
            raise JITCompileError(
                f"JIT codegen does not handle terminator {type(terminator).__name__}"
            )

    # -- driver ------------------------------------------------------------

    def compile(self) -> tuple[list[int], list[tuple[int, str]]]:
        self._emit_stub()
        self._emit_prologue()
        for block in self.function.blocks:
            self.asm.place(self.block_labels[block])
            for inst in block.insts:
                if isinstance(inst, Phi):
                    continue
                self._emit_inst(inst)
            self._emit_phi_copies(block)
            self._emit_terminator(block)
        self._emit_epilogue()
        self.asm.resolve()
        return self.asm.words, self.call_fixups


# ---------------------------------------------------------------------------
# CFG edge splitting
# ---------------------------------------------------------------------------


def split_critical_edges(function: IRFunction) -> None:
    """Split edges out of multi-way branches that feed Phi nodes.

    Phi copies are emitted at the end of the predecessor block, so the copy
    must not be able to execute for a sibling edge.  Inserting an edge block
    with a single successor gives the copies an unambiguous home.
    """
    for block in list(function.blocks):
        successors = IRFunction.successors(block)
        if len(successors) <= 1:
            continue
        for successor in successors:
            phis = [inst for inst in successor.insts if isinstance(inst, Phi)]
            if not phis:
                continue
            edge = function.new_block(f"{block.label}_to_{successor.label}")
            edge.terminator = Jmp(successor)

            terminator = block.terminator
            if isinstance(terminator, Jmp):
                terminator.target = edge
            elif isinstance(terminator, BrCond):
                if terminator.true_block is successor:
                    terminator.true_block = edge
                if terminator.false_block is successor:
                    terminator.false_block = edge

            for phi in phis:
                phi.incoming = [
                    (edge if source is block else source, value)
                    for source, value in phi.incoming
                ]


# ---------------------------------------------------------------------------
# Program compilation
# ---------------------------------------------------------------------------


@dataclass
class CompiledFunction:
    name: str
    entry_offset: int
    word_count: int


@dataclass
class CompiledProgram:
    memory: ExecutableMemory
    functions: dict[str, CompiledFunction] = field(default_factory=dict)
    word_count: int = 0


def compile_program(
    lowered: LoweredProgram,
    *,
    context_address: int,
) -> CompiledProgram:
    """Encode every IR function into one executable region."""
    parts: list[tuple[str, list[int], list[tuple[int, str]]]] = []
    for name in lowered.source_order:
        function = lowered.functions[name]
        split_critical_edges(function)
        words, calls = FunctionCodegen(function, context_address).compile()
        parts.append((name, words, calls))

    total_words = sum(len(words) for _, words, _ in parts)
    if total_words == 0:
        raise JITCompileError("program contains no jittable functions")

    memory = ExecutableMemory(total_words * 4)
    offsets: dict[str, int] = {}
    cursor = 0
    for name, words, _calls in parts:
        offsets[name] = cursor
        cursor += len(words)

    for name, words, calls in parts:
        base = offsets[name]
        for index, target in calls:
            if target not in offsets:
                raise JITCompileError(
                    f"JIT cannot call `{target}`: no machine code generated"
                )
            delta = offsets[target] - (base + index)
            words[index] = 0x94000000 | (delta & 0x3FFFFFF)

    payload = b"".join(
        struct.pack("<I", word & 0xFFFFFFFF)
        for _, words, _calls in parts
        for word in words
    )
    memory.write(0, payload)
    memory.commit(size=len(payload))

    compiled = CompiledProgram(memory=memory, word_count=total_words)
    for name, words, _calls in parts:
        compiled.functions[name] = CompiledFunction(name, offsets[name] * 4, len(words))
    return compiled


__all__ = [
    "Assembler",
    "CompiledFunction",
    "CompiledProgram",
    "FunctionCodegen",
    "compile_program",
    "split_critical_edges",
]
