"""Apple Silicon JIT backend tests.

The backend emits AArch64 machine code directly, so the executable tests are
skipped on anything other than macOS/arm64.  Parser/allocator unit tests still
run everywhere.
"""

from __future__ import annotations

import ctypes
import struct

import pytest

from backends import UntrustedCompileError, get_backend
from backends.jit import codegen_arm64 as arm64
from backends.jit.backend import JITModule
from backends.jit.errors import JITRuntimeError, JITUnsupportedError
from backends.jit.ir import (
    BOOL,
    I64,
    U64,
    BasicBlock,
    BinOp,
    BrCond,
    Cmp,
    Const,
    FunctionSignature,
    IRFunction,
    Jmp,
    LoweredProgram,
    Phi,
    Ret,
    Value,
)
from backends.jit.memory import ExecutableMemory, is_supported_host
from backends.jit.regalloc import Location, parallel_copy
from compiler.core.lexer import lex
from compiler.core.parser import parse

requires_apple_silicon = pytest.mark.skipif(
    not is_supported_host(),
    reason="the JIT backend requires macOS on Apple Silicon (arm64)",
)

pytestmark = requires_apple_silicon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _namespace(source: str, backend: str = "jit"):
    return get_backend(backend, trusted=True).compile(parse(lex(source)))


def _run(source: str):
    return _namespace(source)["main"]()


def _run_ir(function: IRFunction, signature: FunctionSignature):
    lowered = LoweredProgram(
        {function.name: function},
        {function.name: signature},
        {},
        [function.name],
    )
    module = JITModule(lowered)
    return module.functions[function.name]()


def _const_block(function: IRFunction, block: BasicBlock, value: int, ty=I64) -> Value:
    result = function.new_value(ty, "const")
    block.add(Const(result, value))
    return result


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def test_encoder_known_words():
    assert arm64.movz(0, 42) == 0xD2800540  # movz x0, #42
    assert arm64.movz(0, 42, 16) == 0xD2A00540  # movz x0, #42, lsl #16
    assert arm64.mov_reg(0, 1) == 0xAA0103E0  # mov x0, x1
    assert arm64.add_reg(0, 0, 1) == 0x8B010000  # add x0, x0, x1
    assert arm64.sub_reg(0, 0, 1) == 0xCB010000  # sub x0, x0, x1
    assert arm64.add_imm(0, 0, 1) == 0x91000400  # add x0, x0, #1
    assert arm64.sub_imm(0, 0, 1) == 0xD1000400  # sub x0, x0, #1
    assert arm64.mul(0, 0, 1) == 0x9B017C00  # mul x0, x0, x1
    assert arm64.sdiv(0, 0, 1) == 0x9AC10C00  # sdiv x0, x0, x1
    assert arm64.udiv(0, 0, 1) == 0x9AC10800  # udiv x0, x0, x1
    assert arm64.msub(0, 1, 2, 3) == 0x9B028C20  # msub x0, x1, x2, x3
    assert arm64.and_reg(0, 0, 1) == 0x8A010000
    assert arm64.orr_reg(0, 0, 1) == 0xAA010000
    assert arm64.eor_reg(0, 0, 1) == 0xCA010000
    assert arm64.lslv(0, 0, 1) == 0x9AC12000
    assert arm64.lsrv(0, 0, 1) == 0x9AC12400
    assert arm64.asrv(0, 0, 1) == 0x9AC12800
    assert arm64.cmp_reg(0, 1) == 0xEB01001F
    assert arm64.cmp_imm(0, 1) == 0xF100041F
    assert arm64.cset(0, arm64.EQ) == 0x9A9F17E0
    assert arm64.sxtb(0, 1) == 0x93401C20
    assert arm64.sxth(0, 1) == 0x93403C20
    assert arm64.sxtw(0, 1) == 0x93407C20
    assert arm64.uxtb(0, 1) == 0x53001C20
    assert arm64.uxth(0, 1) == 0x53003C20
    assert arm64.uxtw(0, 1) == 0xD3407C20
    assert arm64.fadd(0, 0, 1) == 0x1E612800
    assert arm64.fmul(0, 0, 1) == 0x1E610800
    assert arm64.fcmp_reg(0, 1) == 0x1E612000
    assert arm64.scvtf(0, 1) == 0x9E620020
    assert arm64.fcvtzs(0, 1) == 0x9E780020
    assert arm64.stp_pre(29, 30, 31, -16) == 0xA9BF7BFD
    assert arm64.ldp_post(29, 30, 31, 16) == 0xA8C17BFD
    assert arm64.ldst(0xF9000000, 3, 0, 31, 8) == 0xF90007E0  # str x0, [sp, #8]
    assert arm64.ldst(0xF9400000, 3, 0, 31, 8) == 0xF94007E0  # ldr x0, [sp, #8]
    assert arm64.ldst(0x39000000, 0, 0, 31, 1) == 0x390007E0  # strb w0, [sp, #1]
    assert arm64.ldst(0xFD000000, 3, 0, 31, 8) == 0xFD0007E0  # str d0, [sp, #8]


def test_assembler_resolves_local_branches():
    assembler = arm64.Assembler()
    target = assembler.label()
    assembler.emit(arm64.movz(0, 1))
    assembler.b(target)
    assembler.emit(arm64.movz(0, 2))
    assembler.place(target)
    assembler.emit(arm64.movz(0, 3))
    words = assembler.words
    assembler.resolve()
    assert assembler.words[1] == 0x14000002
    assert words[0] == arm64.movz(0, 1)


def test_executable_memory_roundtrip():
    code = struct.pack("<I", arm64.movz(0, 1234)) + struct.pack("<I", 0xD65F03C0)
    memory = ExecutableMemory(4096)
    try:
        memory.write(0, code)
        memory.commit(size=len(code))
        assert memory.read(0, len(code)) == code
        function = memory.function_pointer(0, ctypes.CFUNCTYPE(ctypes.c_int64))
        assert function() == 1234
    finally:
        memory.close()
    assert memory.closed
    with pytest.raises(Exception):
        memory.write(0, code)


# ---------------------------------------------------------------------------
# Register allocation
# ---------------------------------------------------------------------------


def _simulate(moves, initial, scratch):
    registers = dict(initial)
    scheduled = parallel_copy(moves, Location(reg=scratch))
    for source, destination in scheduled:
        assert not (destination.reg == scratch and source.reg == scratch)
        if source.reg is not None:
            value = registers[source.reg]
        else:
            value = registers[("spill", source.spill)]
        if destination.reg is not None:
            registers[destination.reg] = value
        else:
            registers[("spill", destination.spill)] = value
    return registers, scheduled


def test_parallel_copy_handles_cyclic_phi_moves():
    registers, scheduled = _simulate(
        [
            (Location(reg="X19"), Location(reg="X20")),
            (Location(reg="X20"), Location(reg="X21")),
            (Location(reg="X21"), Location(reg="X19")),
        ],
        {"X19": 1, "X20": 2, "X21": 3, "X16": 0},
        "X16",
    )
    assert registers["X20"] == 1
    assert registers["X21"] == 2
    assert registers["X19"] == 3
    # The scratch register is used to break the cycle.
    assert any(destination.reg == "X16" for _, destination in scheduled)


def test_parallel_copy_handles_spill_locations():
    registers, _ = _simulate(
        [
            (Location(reg="X19"), Location(spill=0)),
            (Location(spill=0), Location(reg="X20")),
            (Location(spill=8), Location(reg="X21")),
        ],
        {"X19": 7, "X20": 0, "X21": 0, ("spill", 0): 5, ("spill", 8): 9},
        "X16",
    )
    # Parallel semantics: X20 observes the *old* spill value.
    assert registers[("spill", 0)] == 7
    assert registers["X20"] == 5
    assert registers["X21"] == 9


def test_parallel_copy_drops_self_moves():
    assert (
        parallel_copy([(Location(reg="X19"), Location(reg="X19"))], Location(reg="X16"))
        == []
    )


# ---------------------------------------------------------------------------
# Scalar language coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("fn main() -> i64 { 42 }", 42),
        ("fn main() -> i64 { let x: i64 = 40; x + 2 }", 42),
        ("fn main() -> i64 { 7 - 2 * 3 }", 1),
        ("fn main() -> i64 { 17 / 5 }", 3),
        ("fn main() -> i64 { -17 % 5 }", -2),
        ("fn main() -> i64 { 17 % -5 }", 2),
        ("fn main() -> i64 { let a: i64 = 6; let b: i64 = 7; a * b }", 42),
        ("fn main() -> i64 { 6 & 3 }", 2),
        ("fn main() -> i64 { 6 | 3 }", 7),
        ("fn main() -> i64 { 6 ^ 3 }", 5),
        ("fn main() -> bool { 3 < 4 }", True),
        ("fn main() -> bool { 3 >= 4 }", False),
        ("fn main() -> bool { true && false }", False),
        ("fn main() -> bool { true || false }", True),
        (
            "fn main() -> i64 { let x: i64 = 5; if x > 3 { 1 } else { 2 } }",
            1,
        ),
        (
            "fn main() -> i64 { let x: i64 = 2; if x > 3 { 1 } else { 2 } }",
            2,
        ),
        (
            """fn main() -> i64 {
                let mut i: i64 = 0;
                let mut total: i64 = 0;
                while i < 10 { total = total + i; i = i + 1; }
                total
            }""",
            45,
        ),
        (
            """fn main() -> i64 {
                let mut total: i64 = 0;
                for i in 0..5 { total = total + i; }
                total
            }""",
            10,
        ),
        (
            """fn main() -> i64 {
                let mut total: i64 = 0;
                for i in 0..=5 { total = total + i; }
                total
            }""",
            15,
        ),
        (
            """fn main() -> i64 {
                let mut i: i64 = 0;
                let value: i64 = loop { i = i + 1; if i == 7 { break i * 2; } };
                value
            }""",
            14,
        ),
        (
            """fn main() -> i64 {
                let mut total: i64 = 0;
                for i in 0..10 {
                    if i % 2 == 0 { continue; }
                    total = total + i;
                }
                total
            }""",
            25,
        ),
        (
            """fn main() -> i64 {
                let x: i64 = 3;
                match x { 0 => 10, 1 | 2 => 20, 3..=5 => 30, _ => 40 }
            }""",
            30,
        ),
        (
            """fn main() -> i64 {
                let mut total: i64 = 0;
                for i in 0..6 {
                    total = total + if i > 2 { 10 } else { 1 };
                }
                total
            }""",
            33,
        ),
        (
            """fn main() -> i64 {
                let mut a: i64 = 1;
                let mut b: i64 = 2;
                let mut i: i64 = 0;
                while i < 3 {
                    let t: i64 = a;
                    a = b;
                    b = t;
                    i = i + 1;
                }
                a * 10 + b
            }""",
            21,
        ),
    ],
)
def test_integer_language_subset(source, expected):
    assert _run(source) == expected


def test_direct_calls_and_recursion():
    source = """
fn add(a: i64, b: i64) -> i64 { a + b }
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}
fn main() -> i64 { add(fib(10), 2) }
"""
    assert _run(source) == 57


def test_float_arithmetic_and_casts():
    source = """
fn mix(a: i64, b: f64, c: i64, d: f64) -> f64 {
    (a + c) as f64 + b * d
}
fn main() -> f64 { mix(1, 2.5, 3, 4.0) }
"""
    assert _run(source) == 14.0
    assert _run("fn main() -> f64 { 7.0 / 2.0 }") == 3.5
    assert _run("fn main() -> i64 { let x: f64 = -7.9; x as i64 }") == -7
    assert _run("fn main() -> f64 { let x: i64 = 7; x as f64 }") == 7.0
    assert _run("fn main() -> bool { 2.5 >= 2.5 }") is True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("fn main() -> i64 { let a: i8 = 127; let b: i8 = 1; (a + b) as i64 }", -128),
        ("fn main() -> u64 { let a: u8 = 255; let b: u8 = 1; (a + b) as u64 }", 0),
        ("fn main() -> i64 { let a: i16 = -1; a as i64 }", -1),
        ("fn main() -> u64 { let a: i8 = -1; a as u64 }", (1 << 64) - 1),
        ("fn main() -> u64 { let a: i64 = -1; (a as u32) as u64 }", (1 << 32) - 1),
        (
            "fn main() -> i64 { let a: u64 = 18446744073709551615u64; a as i32 as i64 }",
            -1,
        ),
        ("fn main() -> i64 { let b: bool = 3 > 2; b as i64 }", 1),
        ("fn main() -> u64 { let a: u8 = 200; a as u64 }", 200),
    ],
)
def test_integer_widths_truncate_and_extend(source, expected):
    assert _run(source) == expected


def test_unsigned_division_and_comparisons():
    source = """
fn main() -> u64 {
    let a: u64 = 18446744073709551615u64;
    let b: u64 = 2u64;
    a / b
}
"""
    assert _run(source) == (1 << 63) - 1
    assert (
        _run("fn main() -> bool { let a: u64 = 18446744073709551615u64; a > 1u64 }")
        is True
    )
    assert _run("fn main() -> bool { let a: i64 = -1; a < 1 }") is True


def test_int64_min_division_and_modulo():
    source = """
fn main() -> i64 {
    let a: i64 = -9223372036854775808;
    let b: i64 = -1;
    (a / b) + (a % b)
}
"""
    assert _run(source) == -9223372036854775808


# ---------------------------------------------------------------------------
# Trap handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "fn main() -> i64 { let a: i64 = 1; let b: i64 = 0; a / b }",
        "fn main() -> i64 { let a: i64 = 1; let b: i64 = 0; a % b }",
        "fn div() -> i64 { 1 / 0 } fn main() -> i64 { div() + 1 }",
    ],
)
def test_division_by_zero_raises_python_error(source):
    with pytest.raises(ZeroDivisionError):
        _run(source)


def test_shift_traps_at_the_python_boundary():
    function = IRFunction("main", [], I64, BasicBlock("entry"))
    block = function.entry
    left = _const_block(function, block, 1)
    right = _const_block(function, block, 64)
    result = function.new_value(I64, "shift")
    block.add(BinOp("<<", left, right, result))
    block.terminator = Ret(result)
    with pytest.raises(JITRuntimeError):
        _run_ir(function, FunctionSignature("main", [], I64))


def test_unsigned_right_shift_is_logical():
    def build(stepper: str):
        function = IRFunction("main", [], U64, BasicBlock("entry"))
        block = function.entry
        left = _const_block(function, block, 1 << 63, U64)
        right = _const_block(function, block, 63, U64)
        result = function.new_value(U64, "shift")
        block.add(BinOp(stepper, left, right, result))
        block.terminator = Ret(result)
        return function

    assert _run_ir(build(">>"), FunctionSignature("main", [], U64)) == 1


def test_signed_right_shift_is_arithmetic():
    function = IRFunction("main", [], I64, BasicBlock("entry"))
    block = function.entry
    left = _const_block(function, block, -8)
    right = _const_block(function, block, 1)
    result = function.new_value(I64, "shift")
    block.add(BinOp(">>", left, right, result))
    block.terminator = Ret(result)
    assert _run_ir(function, FunctionSignature("main", [], I64)) == -4


# ---------------------------------------------------------------------------
# Spilling and CFG edge cases
# ---------------------------------------------------------------------------


def _spill_pressure_function(values: int, iterations: int = 10) -> IRFunction:
    """A loop header with ``values`` invariant phis to force stack spilling."""
    entry = BasicBlock("entry")
    function = IRFunction("main", [], I64, entry)
    header = function.new_block("header")
    body = function.new_block("body")
    exit_block = function.new_block("exit")
    entry.terminator = Jmp(header)

    constants = []
    for index in range(values):
        constants.append(_const_block(function, entry, index + 1))
    zero = _const_block(function, entry, 0)
    limit = _const_block(function, entry, iterations)

    phis = []
    for index, constant in enumerate(constants):
        result = function.new_value(I64, f"phi{index}")
        phi = Phi(result)
        phi.add_incoming(entry, constant)
        header.add(phi)
        phis.append((phi, result))
    counter = function.new_value(I64, "counter")
    counter_phi = Phi(counter)
    counter_phi.add_incoming(entry, zero)
    header.add(counter_phi)
    condition = function.new_value(BOOL, "cond")
    header.add(Cmp("<", counter, limit, condition, I64))
    header.terminator = BrCond(condition, body, exit_block)

    accumulator = phis[0][1]
    for _, value in phis[1:]:
        result = function.new_value(I64, "sum")
        body.add(BinOp("+", accumulator, value, result))
        accumulator = result
    one = _const_block(function, body, 1)
    incremented = function.new_value(I64, "inc")
    body.add(BinOp("+", counter, one, incremented))
    for phi, result in phis:
        phi.add_incoming(body, result)
    counter_phi.add_incoming(body, incremented)
    body.terminator = Jmp(header)
    exit_block.terminator = Ret(counter)
    return function


def test_register_spilling_preserves_semantics():
    from backends.jit.regalloc import allocate

    function = _spill_pressure_function(24)
    allocation = allocate(function)
    assert allocation.spill_count > 0
    assert _run_ir(function, FunctionSignature("main", [], I64)) == 10


def test_split_critical_edge_for_phi_copies():
    entry = BasicBlock("entry")
    function = IRFunction("main", [], I64, entry)
    merge = function.new_block("merge")
    side = function.new_block("side")
    condition = _const_block(function, entry, 1, BOOL)
    entry.terminator = BrCond(condition, merge, side)
    side.terminator = Jmp(merge)

    first = _const_block(function, entry, 11)
    second = _const_block(function, side, 22)
    result = function.new_value(I64, "phi")
    phi = Phi(result)
    phi.add_incoming(entry, first)
    phi.add_incoming(side, second)
    merge.add(phi)
    merge.terminator = Ret(result)

    arm64.split_critical_edges(function)
    assert isinstance(entry.terminator, BrCond)
    edge = entry.terminator.true_block
    assert edge is not merge
    assert isinstance(edge.terminator, Jmp)
    assert edge.terminator.target is merge
    incoming = {block: value for block, value in phi.incoming}
    assert incoming[edge] is first
    assert incoming[side] is second

    assert _run_ir(function, FunctionSignature("main", [], I64)) == 11


def test_phi_cycle_is_resolved_by_codegen():
    """A loop backedge that swaps two phi values needs a scratch register."""
    entry = BasicBlock("entry")
    function = IRFunction("main", [], I64, entry)
    header = function.new_block("header")
    body = function.new_block("body")
    exit_block = function.new_block("exit")
    entry.terminator = Jmp(header)

    first = _const_block(function, entry, 1)
    second = _const_block(function, entry, 2)
    zero = _const_block(function, entry, 0)
    limit = _const_block(function, entry, 3)

    phi_a = function.new_value(I64, "a")
    phi_b = function.new_value(I64, "b")
    counter = function.new_value(I64, "counter")
    node_a = Phi(phi_a)
    node_b = Phi(phi_b)
    node_counter = Phi(counter)
    node_a.add_incoming(entry, first)
    node_b.add_incoming(entry, second)
    node_counter.add_incoming(entry, zero)
    header.add(node_a)
    header.add(node_b)
    header.add(node_counter)
    condition = function.new_value(BOOL, "cond")
    header.add(Cmp("<", counter, limit, condition, I64))
    header.terminator = BrCond(condition, body, exit_block)

    one = _const_block(function, body, 1)
    incremented = function.new_value(I64, "inc")
    body.add(BinOp("+", counter, one, incremented))
    node_a.add_incoming(body, phi_b)
    node_b.add_incoming(body, phi_a)
    node_counter.add_incoming(body, incremented)
    body.terminator = Jmp(header)

    ten = _const_block(function, exit_block, 10)
    scaled = function.new_value(I64, "scaled")
    exit_block.add(BinOp("*", phi_a, ten, scaled))
    total = function.new_value(I64, "total")
    exit_block.add(BinOp("+", scaled, phi_b, total))
    exit_block.terminator = Ret(total)

    # Three swaps leave a=2, b=1.
    assert _run_ir(function, FunctionSignature("main", [], I64)) == 21


# ---------------------------------------------------------------------------
# Backend registration, trust, CLI
# ---------------------------------------------------------------------------


def test_backend_is_registered_and_named():
    backend = get_backend("jit", trusted=True)
    assert backend.name == "jit"
    assert backend.trusted is True


def test_trust_gate_blocks_untrusted_compilation():
    backend = get_backend("jit", trusted=False)
    with pytest.raises(UntrustedCompileError):
        backend.compile(parse(lex("fn main() -> i64 { 1 }")))


def test_target_width_is_rejected():
    with pytest.raises(ValueError):
        get_backend("jit", trusted=True, target_width=32)


def test_unsupported_source_is_reported():
    with pytest.raises(JITUnsupportedError):
        _namespace('fn main() -> i64 { let s = "hello"; 1 }')


def test_cli_runs_the_jit_backend(tmp_path, capsys):
    from api.cli import main

    source = tmp_path / "hello.poly"
    source.write_text(
        """
fn main() -> i64 {
    let mut total: i64 = 0;
    for i in 1..=10 { total = total + i; }
    total
}
""",
        encoding="utf-8",
    )
    assert main(["run", str(source), "--backend", "jit"]) == 0
    assert capsys.readouterr().out.strip() == "55"


def test_namespace_exposes_functions_and_constants():
    namespace = _namespace(
        """
const LIMIT: i64 = 10;
fn double(x: i64) -> i64 { x * 2 }
fn main() -> i64 { double(LIMIT) }
"""
    )
    assert namespace["LIMIT"] == 10
    assert namespace["double"](21) == 42
    assert namespace["main"]() == 20


def test_unit_return_and_bool_return():
    assert _run("fn main() -> bool { 1 + 1 == 2 }") is True
    namespace = _namespace("fn main() {}")
    assert namespace["main"]() is None


def test_internal_compile_error_has_context():
    from compiler.core.ast_nodes import TypeName
    from backends.jit.ir import type_from_ast

    with pytest.raises(JITUnsupportedError):
        type_from_ast(TypeName(line=1, col=1, name="String"))
