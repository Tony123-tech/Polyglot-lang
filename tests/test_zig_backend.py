import shutil

import pytest

from lexer import lex
from parser import parse
from zig import compile_zig
from zig.lowering import lower_zig_ast

pytestmark = pytest.mark.skipif(
    shutil.which("zig") is None, reason="zig toolchain not installed"
)


def test_zig_saturating_and_arbitrary_bit_width():
    src = """
    fn sat_add_u8(a: u8, b: u8) -> u8 { a +| b }
    fn sat_sub_u8(a: u8, b: u8) -> u8 { a -| b }
    fn sat_mul_u8(a: u8, b: u8) -> u8 { a *| b }
    fn sat_u24(x: u24, y: u24) -> u24 { x +| y }
    """
    ns = compile_zig(parse(lex(src)))
    assert ns["sat_add_u8"](250, 20) == 255
    assert ns["sat_sub_u8"](10, 30) == 0
    assert ns["sat_mul_u8"](20, 20) == 255
    assert ns["sat_u24"](16_777_210, 100) == 16_777_215


def test_zig_lowering_pass_for_python_fallback():
    src = "fn f(a: i64, b: i64) -> i64 { a +| b }"
    prog = lower_zig_ast(parse(lex(src)))
    call_node = prog.items[0].body.tail
    assert call_node.func.name == "_zig_sat_add"


def test_zig_parallel_thread_pool():
    import ctypes

    src = """
    #[parallel]
    fn parallel_scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
        for i in 0..n {
            out[i] = (inp[i] * factor) + 1.0;
        }
    }
    """
    ns = compile_zig(parse(lex(src)))
    n = 10_000
    inp_arr = (ctypes.c_double * n)(*[float(i) for i in range(n)])
    out_arr = (ctypes.c_double * n)()

    ns["parallel_scale"](out_arr, inp_arr, 2.5, n)

    assert out_arr[0] == 1.0
    assert out_arr[10] == 26.0
    assert out_arr[9999] == 9999.0 * 2.5 + 1.0


def test_zig_cli_run_and_build(tmp_path):
    from api.cli import main

    poly_file = tmp_path / "demo_zig.poly"
    poly_file.write_text(
        "fn main() -> i64 { (250 as u8 +| 20 as u8) as i64 }\n", encoding="utf-8"
    )
    assert main(["run", str(poly_file), "--backend", "zig"]) == 0
    assert (
        main(["build", str(poly_file), "--backend", "zig", "--out-dir", str(tmp_path)])
        == 0
    )
    assert (tmp_path / "demo_zig.so").exists()
