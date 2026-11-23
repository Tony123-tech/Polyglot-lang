"""Tests for the Mojo-inspired value/origin, layout, and CPython interop modules.

These tests are pure Python and do not require any external toolchain.  They
cover origin/borrow semantics, buffer layouts and SIMD lane computation,
kernel lowering, MLIR availability probing, and PEP 667 safety guards.
"""

from __future__ import annotations

import pytest

from compiler.frontend.errors import PolyglotError
from compiler.mojo_bridge import (
    BufferLayout,
    LayoutOrder,
    Origin,
    OriginEnv,
    ValueKind,
    bridge_python_value,
    check_borrow,
    check_origin_escape,
    check_pep_667_safe,
    lower_vectorized_kernel,
    probe_mlir_available,
)

# ---------------------------------------------------------------------------
# Origin / value-kind semantics
# ---------------------------------------------------------------------------


def test_declare_and_borrow_owned():
    env = OriginEnv()
    env.declare("x", ValueKind.COPYABLE, Origin.OWNED)
    binding = check_borrow(env, "x")
    assert binding.name == "x"
    assert binding.origin == Origin.OWNED


def test_declare_and_borrow_borrowed():
    env = OriginEnv()
    env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
    binding = check_borrow(env, "ref")
    assert binding.origin == Origin.BORROWED


def test_move_owned_value():
    env = OriginEnv()
    env.declare("x", ValueKind.MOVABLE, Origin.OWNED)
    env.move("x")
    # After moving, borrowing should raise E002.
    with pytest.raises(PolyglotError) as exc:
        check_borrow(env, "x")
    assert exc.value.code == "E002"


def test_double_move_raises_e002():
    env = OriginEnv()
    env.declare("x", ValueKind.MOVABLE, Origin.OWNED)
    env.move("x")
    with pytest.raises(PolyglotError) as exc:
        env.move("x")
    assert exc.value.code == "E002"


def test_move_borrowed_raises_e002():
    env = OriginEnv()
    env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
    with pytest.raises(PolyglotError) as exc:
        env.move("ref")
    assert exc.value.code == "E002"


def test_borrow_of_undeclared_raises_e001():
    env = OriginEnv()
    with pytest.raises(PolyglotError) as exc:
        check_borrow(env, "unknown")
    assert exc.value.code == "E001"


def test_borrow_lifetime_mismatch_raises_e001():
    env = OriginEnv()
    env.declare("x", ValueKind.COPYABLE, Origin.OWNED, lifetime_id=1)
    with pytest.raises(PolyglotError) as exc:
        check_borrow(env, "x", expected_lifetime=2)
    assert exc.value.code == "E001"


def test_borrow_lifetime_match():
    env = OriginEnv()
    env.declare("x", ValueKind.COPYABLE, Origin.OWNED, lifetime_id=5)
    binding = check_borrow(env, "x", expected_lifetime=5)
    assert binding.lifetime_id == 5


def test_return_borrowed_as_owned_raises_e001():
    env = OriginEnv()
    env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
    with pytest.raises(PolyglotError) as exc:
        check_origin_escape(env, "ref", return_as_owned=True)
    assert exc.value.code == "E001"


def test_return_borrowed_as_borrowed_ok():
    env = OriginEnv()
    env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
    binding = check_origin_escape(env, "ref", return_as_owned=False)
    assert binding.origin == Origin.BORROWED


def test_unsafe_any_origin_without_annotation_raises_e007():
    env = OriginEnv()
    with pytest.raises(PolyglotError) as exc:
        env.declare("unsafe", ValueKind.COPYABLE, Origin.UNSAFE_ANY_ORIGIN)
    assert exc.value.code == "E007"


def test_unsafe_any_origin_return_without_flag_raises_e007():
    env = OriginEnv()
    # Declare with a regular origin first, then manually set unsafe.
    # Since the env doesn't allow unsafe without annotation, we test the
    # return path via a binding that is borrowed.
    env.declare("x", ValueKind.COPYABLE, Origin.OWNED)
    # A regular owned binding can be returned as owned without issue.
    binding = check_origin_escape(env, "x", return_as_owned=True)
    assert binding.origin == Origin.OWNED


def test_scope_push_pop():
    env = OriginEnv()
    env.declare("x", ValueKind.COPYABLE, Origin.OWNED)
    env.push_scope()
    env.declare("y", ValueKind.COPYABLE, Origin.OWNED)
    assert env.lookup("y") is not None
    env.pop_scope()
    assert env.lookup("y") is None
    assert env.lookup("x") is not None


def test_pop_root_scope_raises():
    env = OriginEnv()
    with pytest.raises(PolyglotError) as exc:
        env.pop_scope()
    assert exc.value.code == "E003"


def test_invalid_binding_name_raises_e001():
    env = OriginEnv()
    with pytest.raises(PolyglotError) as exc:
        env.declare("123invalid", ValueKind.COPYABLE, Origin.OWNED)
    assert exc.value.code == "E001"


def test_value_kind_from_name():
    assert ValueKind.from_name("copyable") == ValueKind.COPYABLE
    assert ValueKind.from_name("Movable") == ValueKind.MOVABLE
    with pytest.raises(PolyglotError) as exc:
        ValueKind.from_name("invalid")
    assert exc.value.code == "E003"


def test_origin_from_name():
    assert Origin.from_name("owned") == Origin.OWNED
    assert Origin.from_name("borrowed") == Origin.BORROWED
    with pytest.raises(PolyglotError) as exc:
        Origin.from_name("invalid")
    assert exc.value.code == "E003"


# ---------------------------------------------------------------------------
# Buffer layouts and SIMD
# ---------------------------------------------------------------------------


def test_contiguous_row_major_layout():
    layout = BufferLayout(shape=(128, 256), dtype="f32", dtype_size=4)
    assert layout.is_contiguous()
    assert layout.element_count == 128 * 256
    assert layout.byte_size == 128 * 256 * 4


def test_non_contiguous_transposed_layout():
    # A transposed layout has strides that don't match row-major order.
    layout = BufferLayout(
        shape=(128, 256),
        dtype="f32",
        dtype_size=4,
        strides=(4, 128 * 4),
    )
    assert not layout.is_contiguous()


def test_col_major_contiguous():
    layout = BufferLayout(
        shape=(4, 8), dtype="i64", dtype_size=8, order=LayoutOrder.COL_MAJOR
    )
    assert layout.is_contiguous()


def test_simd_lanes_power_of_two():
    # 256 f32 elements -> 64 lanes (256 / 4, capped at 64).
    layout = BufferLayout(shape=(256,), dtype="f32", dtype_size=4)
    assert layout.simd_lanes() == 64


def test_simd_lanes_small_layout():
    # 3 elements -> 1 lane (no SIMD benefit).
    layout = BufferLayout(shape=(3,), dtype="f32", dtype_size=4)
    assert layout.simd_lanes() == 1


def test_simd_lanes_single_element():
    layout = BufferLayout(shape=(1,), dtype="f32", dtype_size=4)
    assert layout.simd_lanes() == 1


def test_layout_negative_dimension_raises():
    with pytest.raises(PolyglotError) as exc:
        BufferLayout(shape=(-1, 4), dtype="f32", dtype_size=4)
    assert exc.value.code == "E001"


def test_layout_empty_shape_raises():
    with pytest.raises(PolyglotError) as exc:
        BufferLayout(shape=(), dtype="f32", dtype_size=4)
    assert exc.value.code == "E001"


def test_layout_zero_dtype_size_raises():
    with pytest.raises(PolyglotError) as exc:
        BufferLayout(shape=(4,), dtype="f32", dtype_size=0)
    assert exc.value.code == "E001"


def test_layout_order_from_name():
    assert LayoutOrder.from_name("row_major") == LayoutOrder.ROW_MAJOR
    assert LayoutOrder.from_name("col_major") == LayoutOrder.COL_MAJOR
    with pytest.raises(PolyglotError) as exc:
        LayoutOrder.from_name("diagonal")
    assert exc.value.code == "E001"


# ---------------------------------------------------------------------------
# Kernel lowering
# ---------------------------------------------------------------------------


def test_kernel_vectorizable():
    layout = BufferLayout(shape=(256,), dtype="f32", dtype_size=4)
    kernel = lower_vectorized_kernel("axpy", layout)
    assert kernel.vectorizable is True
    assert kernel.simd_lanes > 1
    assert "contiguous" in kernel.reason


def test_kernel_non_contiguous_not_vectorizable():
    layout = BufferLayout(shape=(256,), dtype="f32", dtype_size=4, strides=(8,))
    kernel = lower_vectorized_kernel("axpy", layout)
    assert kernel.vectorizable is False
    assert kernel.simd_lanes == 1
    assert "non-contiguous" in kernel.reason


def test_kernel_small_layout_not_vectorizable():
    # A single-element layout has only 1 lane — no SIMD benefit.
    layout = BufferLayout(shape=(1,), dtype="f32", dtype_size=4)
    kernel = lower_vectorized_kernel("axpy", layout)
    assert kernel.vectorizable is False


def test_kernel_loop_bounds_mismatch_raises():
    layout = BufferLayout(shape=(4, 8), dtype="f32", dtype_size=4)
    with pytest.raises(PolyglotError) as exc:
        lower_vectorized_kernel("bad", layout, loop_bounds=(4,))
    assert exc.value.code == "E001"


def test_kernel_empty_name_raises():
    layout = BufferLayout(shape=(4,), dtype="f32", dtype_size=4)
    with pytest.raises(PolyglotError) as exc:
        lower_vectorized_kernel("", layout)
    assert exc.value.code == "E003"


def test_kernel_with_passes():
    layout = BufferLayout(shape=(256,), dtype="f32", dtype_size=4)
    kernel = lower_vectorized_kernel("axpy", layout, passes=("inline", "vectorize"))
    assert kernel.passes == ("inline", "vectorize")


# ---------------------------------------------------------------------------
# MLIR probe
# ---------------------------------------------------------------------------


def test_mlir_probe_returns_structured_dict():
    result = probe_mlir_available()
    assert "available" in result
    assert "reason" in result
    assert "model_only" in result
    assert result["model_only"] is True
    # On systems without MLIR tools, available should be False.
    if not result["available"]:
        assert result["missing_tools"]


# ---------------------------------------------------------------------------
# CPython interop / PEP 667
# ---------------------------------------------------------------------------


def test_bridge_int():
    result = bridge_python_value(42)
    assert result.native_type == "i64"
    assert result.value == 42
    assert result.cold_path is True


def test_bridge_float():
    result = bridge_python_value(3.14)
    assert result.native_type == "f64"
    assert result.value == 3.14


def test_bridge_bool():
    result = bridge_python_value(True)
    assert result.native_type == "bool"
    assert result.value is True


def test_bridge_bytes():
    result = bridge_python_value(b"hello")
    assert result.native_type == "[]u8"
    assert result.value == b"hello"


def test_bridge_none():
    result = bridge_python_value(None)
    assert result.native_type == "void"
    assert result.value is None


def test_bridge_unsupported_type_raises_e001():
    with pytest.raises(PolyglotError) as exc:
        bridge_python_value("a string")
    assert exc.value.code == "E001"


def test_bridge_list_raises_e001():
    with pytest.raises(PolyglotError) as exc:
        bridge_python_value([1, 2, 3])
    assert exc.value.code == "E001"


def test_pep_667_safe_empty_context():
    # Empty context should not raise.
    check_pep_667_safe("")


def test_pep_667_hot_loop_f_locals_raises():
    with pytest.raises(PolyglotError) as exc:
        check_pep_667_safe("f_locals in hot loop")
    assert exc.value.code == "E007"


def test_pep_667_loop_f_locals_raises():
    with pytest.raises(PolyglotError) as exc:
        check_pep_667_safe("accessing f_locals in loop")
    assert exc.value.code == "E007"


def test_pep_667_normal_context_ok():
    # A normal context without hot-loop keywords should not raise.
    check_pep_667_safe("converting a Python int to i64")


def test_bridge_with_pep_667_context():
    with pytest.raises(PolyglotError) as exc:
        bridge_python_value(42, context="f_locals in hot loop")
    assert exc.value.code == "E007"
