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


def test_zig_simd_vector_parallel():
    import ctypes

    src = """
    #[parallel]
    fn simd_fma(out: &mut [f64], inp: &[f64], scale: f64, bias: f64, chunks: usize) {
        let v_scale: Simd4<f64> = Simd::splat(scale);
        let v_bias: Simd4<f64> = Simd::splat(bias);
        for c in 0..chunks {
            let idx: usize = c * 4;
            let v: Simd4<f64> = Simd::load(inp, idx, 4);
            let res: Simd4<f64> = (v * v_scale) + v_bias;
            Simd::store(out, idx, 4, res);
        }
    }
    """
    ns = compile_zig(parse(lex(src)))
    n = 1024
    inp_arr = (ctypes.c_double * n)(*[float(i) for i in range(n)])
    out_arr = (ctypes.c_double * n)()
    ns["simd_fma"](out_arr, inp_arr, 2.0, 5.0, n // 4)
    assert (
        out_arr[0] == 5.0 and out_arr[3] == 11.0 and out_arr[1023] == 1023.0 * 2.0 + 5.0
    )


def test_zig_simd_horizontal_reduce_and_example():
    import ctypes
    from pathlib import Path

    src = Path("examples/zig_parallel_simd.poly").read_text(encoding="utf-8")
    ns = compile_zig(parse(lex(src)))
    a = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    b = (ctypes.c_double * 4)(10.0, 20.0, 30.0, 40.0)
    assert ns["simd_dot4"](a, b) == 300.0
    assert ns["main"]() == 255
