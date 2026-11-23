"""Zig-exclusive special functions: integer intrinsics, bit casts, packed structs.

Every test here compiles real Zig and calls the generated library, except the
rejection tests, which assert the backend refuses the program before Zig runs.
"""

from __future__ import annotations

import shutil

import pytest

from compiler.frontend.errors import PolyglotError
from lexer import lex
from parser import parse
from zig import compile_zig
from zig.backend import ZigCodegen
from zig.intrinsics import IntrinsicError, bit_width, mirror

pytestmark = pytest.mark.skipif(
    shutil.which("zig") is None, reason="zig toolchain not installed"
)


def _namespace(source: str) -> dict:
    return compile_zig(parse(lex(source)))


# -- counting intrinsics -------------------------------------------------


def test_popcount_matches_the_python_mirror() -> None:
    source = """
    fn pc8(x: u8) -> i64 { popcount!(x) }
    fn pc32(x: u32) -> i64 { popcount!(x) }
    fn pc64(x: u64) -> i64 { popcount!(x) }
    """
    ns = _namespace(source)
    for value in (0, 1, 0b1011, 0xFF, 0x80000000):
        assert ns["pc32"](value) == mirror("popcount", value, "u32")
    assert ns["pc8"](0xFF) == 8
    assert ns["pc64"](0xFFFFFFFFFFFFFFFF) == 64
    assert ns["pc64"](0) == 0


def test_clz_and_ctz_use_the_operand_width() -> None:
    source = """
    fn lz8(x: u8) -> i64 { clz!(x) }
    fn lz16(x: u16) -> i64 { clz!(x) }
    fn tz8(x: u8) -> i64 { ctz!(x) }
    fn tz32(x: u32) -> i64 { ctz!(x) }
    """
    ns = _namespace(source)
    assert ns["lz8"](0) == 8
    assert ns["lz8"](1) == 7
    assert ns["lz16"](1) == 15
    assert ns["tz8"](0) == 8
    assert ns["tz8"](8) == 3
    assert ns["tz32"](1 << 31) == 31


# -- byte and bit rearrangement ------------------------------------------


def test_bswap_keeps_the_operand_width() -> None:
    source = """
    fn swap16(x: u16) -> u16 { bswap!(x) }
    fn swap32(x: u32) -> u32 { bswap!(x) }
    fn swap64(x: u64) -> u64 { bswap!(x) }
    """
    ns = _namespace(source)
    assert ns["swap16"](0x1234) == 0x3412
    assert ns["swap32"](0x12345678) == 0x78563412
    assert ns["swap64"](0x0102030405060708) == 0x0807060504030201
    assert ns["swap32"](0x12345678) == mirror("bswap", 0x12345678, "u32")


def test_bitreverse_reverses_within_the_width() -> None:
    source = """
    fn rev8(x: u8) -> u8 { bitreverse!(x) }
    fn rev16(x: u16) -> u16 { bitreverse!(x) }
    """
    ns = _namespace(source)
    assert ns["rev8"](1) == 128
    assert ns["rev8"](0b10110000) == 0b00001101
    assert ns["rev16"](0x00FF) == 0xFF00
    assert ns["rev8"](7) == mirror("bitreverse", 7, "u8")


# -- bit casts ------------------------------------------------------------


def test_bitcast_reads_the_float_bit_pattern() -> None:
    source = """
    fn f32_bits(x: f32) -> u32 { bitcast!(x, u32) }
    fn bits_f32(x: u32) -> f32 { bitcast!(x, f32) }
    fn f64_bits(x: f64) -> u64 { bitcast!(x, u64) }
    """
    ns = _namespace(source)
    assert ns["f32_bits"](1.0) == 0x3F800000
    assert ns["f32_bits"](0.5) == 0x3F000000
    assert ns["bits_f32"](0x3F800000) == 1.0
    assert ns["f64_bits"](1.0) == 0x3FF0000000000000


def test_bitcast_rejects_a_width_mismatch() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("fn f(x: u8) -> u32 { bitcast!(x, u32) }")))
    assert error.value.code == "E001"
    assert "8 bits" in error.value.diagnostic.message
    assert "32 bits" in error.value.diagnostic.message


def test_bitcast_rejects_a_pointer_source() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(
            parse(lex("fn f(x: &[i64], n: usize) -> u32 { bitcast!(x, u32) }"))
        )
    assert error.value.code == "E001"


# -- intrinsic validation -------------------------------------------------


def test_intrinsic_rejects_a_float_operand() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("fn f(x: f64) -> i64 { popcount!(x) }")))
    assert error.value.code == "E001"
    assert "integer operand" in error.value.diagnostic.message


def test_bswap_rejects_a_non_byte_width() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("fn f(x: u7) -> i64 { bswap!(x) }")))
    assert error.value.code == "E001"
    assert "7 bits" in error.value.diagnostic.message


def test_value_intrinsic_arity_is_checked() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(
            parse(lex("fn f(x: u8) -> i64 { popcount!(x, x) }"))
        )
    assert error.value.code == "E003"


def test_python_mirror_agrees_on_bounds() -> None:
    assert mirror("popcount", 0xFF, "u8") == 8
    assert mirror("clz", 0, "u32") == 32
    assert mirror("ctz", 0, "u32") == 32
    assert bit_width("u3") == 3
    assert bit_width("f32") == 32
    assert bit_width("String") is None
    with pytest.raises(IntrinsicError):
        mirror("bswap", 1, "u7")


# -- packed structs -------------------------------------------------------


def test_packed_struct_size_is_the_sum_of_its_fields() -> None:
    source = """
    #[packed]
    struct Flags { a: u3, b: u5, c: bool }

    #[packed]
    struct Two { hi: u12, lo: u12 }

    fn flags_bits() -> i64 { bitsizeof!(Flags) }
    fn flags_bytes() -> i64 { sizeof!(Flags) }
    fn two_bits() -> i64 { bitsizeof!(Two) }
    fn two_bytes() -> i64 { sizeof!(Two) }
    """
    ns = _namespace(source)
    assert ns["flags_bits"]() == 9
    assert ns["flags_bytes"]() == 2
    assert ns["two_bits"]() == 24
    # @sizeOf rounds the backing integer up to its ABI alignment (u24 -> 4
    # bytes here), so the exact answer comes from bitsizeof! and the rounded
    # one from sizeof!.
    assert ns["two_bytes"]() == 4


def test_packed_struct_emits_a_zig_packed_struct() -> None:
    code = ZigCodegen("t").generate(
        parse(lex("#[packed] struct Flags { a: u3, b: u5, c: bool }"))
    )
    assert "const Flags = packed struct {" in code
    assert "a: u3," in code
    # The generated comment states the bit total so a layout change is visible
    # in the source rather than only in a failing test.
    assert "Flags: 9 bits, packed" in code


def test_packed_struct_rejects_a_float_field() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("#[packed] struct P { a: f64 }")))
    assert error.value.code == "E001"
    assert "f64" in error.value.diagnostic.message


def test_packed_struct_rejects_a_pointer_field() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(
            parse(lex("#[packed] struct P { a: &[i64], n: usize }"))
        )
    assert error.value.code == "E001"


def test_packed_struct_cannot_be_generic() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("#[packed] struct P<T> { a: u8 }")))
    assert error.value.code == "E003"


def test_type_intrinsic_rejects_an_unknown_type() -> None:
    with pytest.raises(PolyglotError) as error:
        ZigCodegen("t").generate(parse(lex("fn f() -> i64 { bitsizeof!(Nope) }")))
    assert error.value.code == "E001"
    assert "Nope" in error.value.diagnostic.message


# -- runtime safety opt-out ----------------------------------------------


def test_unchecked_function_disables_runtime_safety() -> None:
    source = """
    #[unchecked]
    fn unchecked_add(a: i64, b: i64) -> i64 { a + b }
    """
    code = ZigCodegen("t").generate(parse(lex(source)))
    assert "@setRuntimeSafety(false);" in code
    ns = _namespace(source)
    assert ns["unchecked_add"](2, 3) == 5
    assert ns["unchecked_add"](-7, 4) == -3


def test_checked_function_keeps_runtime_safety() -> None:
    code = ZigCodegen("t").generate(
        parse(lex("fn checked_add(a: i64, b: i64) -> i64 { a + b }"))
    )
    assert "@setRuntimeSafety(false);" not in code
