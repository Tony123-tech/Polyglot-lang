"""Deterministic tests for the offline heterogeneous-offloading models."""

from __future__ import annotations

import pytest

import compiler.heterogeneous as hetero
from compiler.core.ast_nodes import (
    Assign,
    Binary,
    Block,
    Call,
    CompoundAssign,
    ExprStmt,
    For,
    Ident,
    Index,
    LitInt,
    PatIdent,
    Range,
)
from compiler.heterogeneous import (
    AffineExpr,
    ANENode,
    BufferRequest,
    FusionPlan,
    LoopBound,
    PolyhedralDescriptor,
    TensorAccess,
    accel_async_stream_pipelining,
    accel_auto_tune_tile_sizes,
    accel_detect_tensor_polyhedral,
    accel_emit_metal_cuda_spirv,
    accel_kernel_fusion_epilogue,
    accel_lower_ast_to_mlir,
    accel_mixed_precision_quantize,
    accel_nanosecond_cpu_fallback,
    accel_npu_ane_graph_compile,
    accel_zero_copy_unified_mem,
)
from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import PillarError

_PUBLIC_FUNCTIONS = (
    "accel_detect_tensor_polyhedral",
    "accel_lower_ast_to_mlir",
    "accel_emit_metal_cuda_spirv",
    "accel_zero_copy_unified_mem",
    "accel_npu_ane_graph_compile",
    "accel_kernel_fusion_epilogue",
    "accel_async_stream_pipelining",
    "accel_auto_tune_tile_sizes",
    "accel_mixed_precision_quantize",
    "accel_nanosecond_cpu_fallback",
)


# ---------------------------------------------------------------------------
# AST builders
# ---------------------------------------------------------------------------


def _ident(name: str, *, span: tuple[int, int] | None = None) -> Ident:
    return Ident(0, 0, span=span, name=name)


def _int(value: int, *, span: tuple[int, int] | None = None) -> LitInt:
    return LitInt(0, 0, span=span, value=value, raw=str(value))


def _index(base: str, *indices: object, span=None) -> Index:
    node: object = _ident(base)
    for index in indices:
        node = Index(0, 0, span=span, receiver=node, index=index)
    assert isinstance(node, Index)
    return node


def _nest(
    inner_stmt: object,
    *,
    outer_end: object = None,
    inner_end: object = None,
    outer_span: tuple[int, int] | None = (47, 102),
) -> For:
    outer_end = outer_end if outer_end is not None else _ident("n")
    inner_end = inner_end if inner_end is not None else _ident("m")
    inner = For(
        4,
        3,
        span=(67, 102),
        pattern=PatIdent(4, 9, name="j"),
        iter=Range(4, 13, start=_int(0), end=inner_end, inclusive=False),
        body=Block(4, 13, stmts=[ExprStmt(5, 7, expr=inner_stmt)]),
    )
    return For(
        3,
        3,
        span=outer_span,
        pattern=PatIdent(3, 8, name="i"),
        iter=Range(3, 13, start=_int(0), end=outer_end, inclusive=False),
        body=Block(3, 13, stmts=[ExprStmt(4, 5, expr=inner)]),
    )


def matmul_nest() -> For:
    statement = CompoundAssign(
        5,
        7,
        span=(80, 102),
        op="+=",
        target=_index("out", _ident("i"), _ident("j")),
        value=Binary(
            5,
            20,
            span=(93, 102),
            op="*",
            left=_index("a", _ident("i"), _ident("j")),
            right=_index("b", _ident("j"), _ident("i")),
        ),
    )
    return _nest(statement)


def add_nest() -> For:
    statement = Assign(
        5,
        7,
        target=_index("out", _ident("i"), _ident("j")),
        value=Binary(
            5,
            20,
            op="+",
            left=_index("a", _ident("i"), _ident("j")),
            right=_index("b", _ident("i"), _ident("j")),
        ),
    )
    return _nest(statement)


def transpose_nest() -> For:
    statement = Assign(
        5,
        7,
        target=_index("out", _ident("i"), _ident("j")),
        value=_index("in", _ident("j"), _ident("i")),
    )
    return _nest(statement)


def descriptor(
    op: str,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    shape: tuple[int, ...] = (8, 8),
) -> PolyhedralDescriptor:
    loops = tuple(
        LoopBound(
            name=f"i{axis}",
            lower=AffineExpr(),
            upper=AffineExpr((), dim),
        )
        for axis, dim in enumerate(shape)
    )
    accesses = (
        *(
            TensorAccess(name, (AffineExpr(((f"i{idx % len(shape)}", 1),), 0),), "read")
            for idx, name in enumerate(reads)
        ),
        *(
            TensorAccess(
                name, (AffineExpr(((f"i{idx % len(shape)}", 1),), 0),), "write"
            )
            for idx, name in enumerate(writes)
        ),
    )
    return PolyhedralDescriptor(loops=loops, accesses=accesses, op=op)


# ---------------------------------------------------------------------------
# 81 -- affine detection
# ---------------------------------------------------------------------------


def test_all_public_function_names_are_exported():
    for name in _PUBLIC_FUNCTIONS:
        assert callable(getattr(hetero, name))
    for name in _PUBLIC_FUNCTIONS:
        assert name in hetero.__all__


def test_detect_nested_affine_matmul_descriptor():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    assert desc.depth == 2
    assert desc.loop_names == ("i", "j")
    assert desc.op == "matmul"
    assert desc.tensors == ("a", "b", "out")
    assert desc.writes == ("out",)
    assert desc.reads == ("a", "b", "out")
    assert desc.rank == 2
    assert desc.shape() == (256, 256)
    assert desc.to_dict()["loops"][0]["trip_count"] == "n"
    assert desc.to_dict()["accesses"][0]["indices"] == ["i", "j"]


def test_detect_canonicalizes_affine_terms_and_inclusive_bounds():
    inner_body = Assign(
        5,
        7,
        target=_index("out", _ident("i")),
        value=Binary(
            5,
            9,
            op="+",
            left=_index("a", Binary(5, 9, op="*", left=_int(2), right=_ident("i"))),
            right=_int(3),
        ),
    )
    loop = For(
        3,
        3,
        pattern=PatIdent(3, 8, name="i"),
        iter=Range(
            3,
            13,
            start=Binary(3, 14, op="-", left=_ident("k"), right=_int(1)),
            end=Binary(3, 20, op="+", left=_ident("i"), right=_int(0)),
            inclusive=True,
        ),
        body=Block(3, 13, stmts=[ExprStmt(5, 7, expr=inner_body)]),
    )
    # The loop bound above references its own induction variable only to
    # exercise canonicalization; the detector keeps it as a symbolic term.
    desc = accel_detect_tensor_polyhedral(loop)
    assert desc.loops[0].lower.text == "k - 1"
    assert desc.loops[0].upper.text == "i"
    assert desc.loops[0].inclusive is True
    assert desc.loops[0].trip_count.text == "i - k + 2"
    index = desc.accesses[1].indices[0]
    assert index.text == "2*i"


def test_detect_rejects_nonlinear_bound_with_exact_span():
    nonlinear = Binary(
        3,
        20,
        span=(14, 21),
        op="*",
        left=_ident("i"),
        right=_ident("j"),
    )
    loop = _nest(
        Assign(5, 7, target=_index("out", _ident("i")), value=_index("a", _ident("i"))),
        outer_end=nonlinear,
    )
    with pytest.raises(PillarError) as excinfo:
        accel_detect_tensor_polyhedral(loop)
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (14, 21)
    assert "non-affine operator" in str(excinfo.value)


def test_detect_rejects_call_in_tensor_index():
    call = Call(5, 16, span=(90, 97), func=_ident("f"), args=[_ident("i")])
    loop = _nest(
        Assign(5, 7, target=_index("out", _ident("i")), value=_index("a", call))
    )
    with pytest.raises(PillarError) as excinfo:
        accel_detect_tensor_polyhedral(loop)
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (90, 97)
    assert "non-affine expression" in str(excinfo.value)


def test_detect_requires_loops_and_tensor_accesses():
    block = Block(0, 0, stmts=[])
    with pytest.raises(PillarError) as excinfo:
        accel_detect_tensor_polyhedral(block, span=(1, 2), line=3, col=4)
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (1, 2)

    no_access = _nest(Assign(5, 7, target=_ident("c"), value=_int(1)))
    with pytest.raises(PillarError) as excinfo:
        accel_detect_tensor_polyhedral(no_access)
    assert excinfo.value.code == "E003"
    assert "no tensor accesses" in str(excinfo.value)


def test_detect_transpose_access_pattern():
    desc = accel_detect_tensor_polyhedral(transpose_nest())
    assert desc.op == "transpose"
    write, read = desc.accesses[0], desc.accesses[1]
    assert write.mode == "write" and read.mode == "read"
    assert [expr.text for expr in write.indices] == ["i", "j"]
    assert [expr.text for expr in read.indices] == ["j", "i"]


def test_detect_uses_real_parser_spans_and_explicit_override():
    source = "fn main() -> i64 {\n  for i in 0..n {\n    out[i] = a[i];\n  }\n  0\n}\n"
    program = parse(lex(source))
    from_ast = accel_detect_tensor_polyhedral(program)
    outer = program.items[0].body.stmts[0].expr
    assert from_ast.span == outer.span
    assert (from_ast.line, from_ast.col) == (outer.line, outer.col)

    overridden = accel_detect_tensor_polyhedral(program, span=(200, 210), line=9, col=2)
    assert overridden.span == (200, 210)
    assert (overridden.line, overridden.col) == (9, 2)


# ---------------------------------------------------------------------------
# 82 -- textual IR
# ---------------------------------------------------------------------------


def test_lower_emits_scf_linalg_and_tensor_ops_deterministically():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    ir = accel_lower_ast_to_mlir(desc)
    assert ir == accel_lower_ast_to_mlir(desc)
    assert "textual only, no device code is compiled" in ir
    assert "module {" in ir and "func.func @accel_scf" in ir
    assert "scf.for %i" in ir and "scf.for %j" in ir
    assert "tensor.extract" in ir and "tensor.insert" in ir
    assert "linalg.generic" in ir
    assert "arith.mulf" in ir
    assert 'iterator_types = ["parallel", "parallel"]' in ir
    assert "%sym_n" in ir  # symbolic bounds become function parameters


def test_lower_rejects_bad_inputs_with_pillar_errors():
    with pytest.raises(PillarError) as excinfo:
        accel_lower_ast_to_mlir("not a descriptor", span=(3, 8), line=2, col=1)
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == (3, 8)

    with pytest.raises(PillarError) as excinfo:
        accel_lower_ast_to_mlir(
            PolyhedralDescriptor(loops=(), accesses=(), op="add"),
            span=(9, 11),
        )
    assert excinfo.value.code == "E003"


# ---------------------------------------------------------------------------
# 83 -- kernel source text and uniform E007 rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,marker",
    [
        ("metal", "kernel void accel_kernel"),
        ("cuda", "__global__ void accel_kernel"),
        ("spirv", "OpEntryPoint GLCompute"),
    ],
)
def test_emit_kernel_source_text_for_each_target(target, marker):
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    source = accel_emit_metal_cuda_spirv(desc, target=target)
    assert source.target == target
    assert source.model_only is True
    assert source.entry_point == "accel_kernel"
    assert marker in source.source
    assert "MODEL-ONLY" in source.source
    assert (
        "no compiler, driver, or device API" in source.source
        or "not a valid SPIR-V" in source.source
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dispatch": True},
        {"compile_kernel": True},
        {"dispatch": True, "compile_kernel": True},
    ],
)
def test_emit_rejects_real_device_work_uniformly(kwargs):
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    with pytest.raises(PillarError) as excinfo:
        accel_emit_metal_cuda_spirv(desc, span=(11, 22), line=4, col=9, **kwargs)
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (11, 22)
    assert (excinfo.value.line, excinfo.value.col) == (4, 9)
    assert "not implemented" in str(excinfo.value)
    assert excinfo.value.diagnostic.note == "DeviceDispatchUnavailable"


def test_emit_dispatch_error_inherits_descriptor_span():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    assert desc.span == (47, 102)
    with pytest.raises(PillarError) as excinfo:
        accel_emit_metal_cuda_spirv(desc, dispatch=True)
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (47, 102)


def test_emit_rejects_non_string_target_with_pillar_error():
    desc = accel_detect_tensor_polyhedral(add_nest())
    with pytest.raises(PillarError) as excinfo:
        accel_emit_metal_cuda_spirv(desc, target=object(), span=(2, 5))
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (2, 5)


def test_emit_rejects_unavailable_target():
    desc = accel_detect_tensor_polyhedral(add_nest())
    with pytest.raises(PillarError) as excinfo:
        accel_emit_metal_cuda_spirv(desc, target="vulkan", span=(1, 7))
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (1, 7)


# ---------------------------------------------------------------------------
# 84 -- zero-copy buffer views
# ---------------------------------------------------------------------------


def test_zero_copy_plan_is_pure_offset_arithmetic():
    plan = accel_zero_copy_unified_mem(
        [
            BufferRequest("A", 100, alignment=64),
            BufferRequest("B", 40, dtype="f16", alignment=32, read_only=True),
        ]
    )
    assert [view.offset for view in plan.views] == [0, 128]
    assert [view.padded_length for view in plan.views] == [128, 64]
    assert plan.total_nbytes == 192
    assert plan.device_mapped is False
    assert "offset=128" in plan.text
    assert "no device pointer is created" in plan.text


def test_zero_copy_rejects_device_mapping_with_span():
    with pytest.raises(PillarError) as excinfo:
        accel_zero_copy_unified_mem(
            BufferRequest("A", 16), map_device=True, span=(30, 44), line=7, col=3
        )
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (30, 44)
    assert (excinfo.value.line, excinfo.value.col) == (7, 3)
    assert "unified-memory device mapping" in str(excinfo.value)


def test_zero_copy_validates_requests_with_pillar_errors():
    with pytest.raises(PillarError) as excinfo:
        accel_zero_copy_unified_mem(BufferRequest("A", 0))
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_zero_copy_unified_mem(BufferRequest("A", 8, dtype="f64"))
    assert excinfo.value.code == "E001"

    with pytest.raises(PillarError) as excinfo:
        accel_zero_copy_unified_mem(
            [BufferRequest("A", 8), BufferRequest("A", 8)], span=(2, 5)
        )
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (2, 5)

    with pytest.raises(PillarError) as excinfo:
        accel_zero_copy_unified_mem([])
    assert excinfo.value.code == "E003"


# ---------------------------------------------------------------------------
# 85 -- ANE graph partition
# ---------------------------------------------------------------------------


def test_ane_partitions_matmul_descriptor():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    plan = accel_npu_ane_graph_compile(desc, tile=(32, 16))
    assert plan.kind == "matmul"
    assert plan.dispatchable is False
    assert len(plan.partitions) == 1
    part = plan.partitions[0]
    assert part.tile == (32, 16)
    assert part.output == "out"
    assert [node.op for node in part.nodes] == ["load", "load", "matmul", "store"]
    assert all(isinstance(node, ANENode) for node in plan.nodes)
    assert "no ANE program is compiled or dispatched" in plan.graph_text
    assert "tile=32x16" in plan.graph_text


def test_ane_handles_elementwise_and_missing_tensor_output():
    elementwise = accel_npu_ane_graph_compile(
        accel_detect_tensor_polyhedral(add_nest())
    )
    assert elementwise.kind == "elementwise"

    scalar_acc = PolyhedralDescriptor(
        loops=(LoopBound("i", AffineExpr(), AffineExpr((), 16)),),
        accesses=(TensorAccess("a", (AffineExpr((("i", 1),), 0),), "read"),),
        op="reduce_sum",
    )
    plan = accel_npu_ane_graph_compile(scalar_acc)
    assert plan.partitions[0].output == "acc"
    assert "scalar accumulator" in plan.graph_text


def test_ane_rejects_dispatch_and_invalid_matrix_shape():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    with pytest.raises(PillarError) as excinfo:
        accel_npu_ane_graph_compile(desc, dispatch=True, span=(60, 70), line=8, col=2)
    assert excinfo.value.code == "E007"
    assert excinfo.value.span == (60, 70)

    shallow = PolyhedralDescriptor(
        loops=(LoopBound("i", AffineExpr(), AffineExpr((), 16)),),
        accesses=(
            TensorAccess("a", (AffineExpr((("i", 1),), 0),), "read"),
            TensorAccess("b", (AffineExpr((("i", 1),), 0),), "read"),
            TensorAccess("o", (AffineExpr((("i", 1),), 0),), "write"),
        ),
        op="mul",
    )
    with pytest.raises(PillarError) as excinfo:
        accel_npu_ane_graph_compile(shallow)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_npu_ane_graph_compile(desc, tile=(0, 16))
    assert excinfo.value.code == "E003"


# ---------------------------------------------------------------------------
# 86 -- fusion
# ---------------------------------------------------------------------------


def test_fusion_orders_elementwise_reduction_and_transpose():
    plan = accel_kernel_fusion_epilogue(
        [
            descriptor("add", ("a", "b"), ("t1",)),
            descriptor("reduce_sum", ("t1",), ("t2",)),
            descriptor("transpose", ("t2",), ("out",)),
        ]
    )
    assert isinstance(plan, FusionPlan)
    assert [stage.kind for stage in plan.stages] == [
        "elementwise",
        "reduction",
        "transpose",
    ]
    assert [stage.index for stage in plan.stages] == [0, 1, 2]
    assert plan.writes == ("out", "t1", "t2")
    assert "fused_kernel @fused_epilogue" in plan.text
    assert "elementwise.add" in plan.text
    assert "reduction.reduce_sum" in plan.text
    assert "transpose.transpose" in plan.text


def test_fusion_topologically_sorts_out_of_order_inputs():
    plan = accel_kernel_fusion_epilogue(
        [
            descriptor("transpose", ("t1",), ("out",)),
            descriptor("add", ("a", "b"), ("t1",)),
        ]
    )
    assert [stage.index for stage in plan.stages] == [1, 0]


def test_fusion_rejects_mixed_rank_descriptors():
    with pytest.raises(PillarError) as excinfo:
        accel_kernel_fusion_epilogue(
            [
                descriptor("add", ("a",), ("t",), shape=(8, 8)),
                descriptor("copy", ("t",), ("out",), shape=(8,)),
            ],
            span=(6, 12),
        )
    assert excinfo.value.code == "E003"
    assert excinfo.value.span == (6, 12)


def test_fusion_rejects_cycles_conflicts_and_bad_types():
    with pytest.raises(PillarError) as excinfo:
        accel_kernel_fusion_epilogue(
            [
                descriptor("add", ("t2",), ("t1",)),
                descriptor("transpose", ("t1",), ("t2",)),
            ],
            span=(5, 9),
        )
    assert excinfo.value.code == "E005"
    assert excinfo.value.span == (5, 9)
    assert excinfo.value.diagnostic.note == "FusionDependencyCycle"

    with pytest.raises(PillarError) as excinfo:
        accel_kernel_fusion_epilogue(
            [descriptor("add", ("a",), ("x",)), descriptor("sub", ("b",), ("x",))]
        )
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_kernel_fusion_epilogue([object()], span=(0, 3))
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == (0, 3)

    with pytest.raises(PillarError) as excinfo:
        accel_kernel_fusion_epilogue([])
    assert excinfo.value.code == "E003"


# ---------------------------------------------------------------------------
# 87 -- stream / event model
# ---------------------------------------------------------------------------


def test_pipeline_builds_dma_kernel_dependencies():
    plan = accel_kernel_fusion_epilogue(
        [
            descriptor("add", ("a", "b"), ("t1",)),
            descriptor("reduce_sum", ("t1",), ("out",)),
        ]
    )
    schedule = accel_async_stream_pipelining(plan, depth=3, streams=4)
    by_name = {event.name: event for event in schedule.events}
    assert len(schedule.events) == 9
    assert by_name["kernel_0"].depends_on == ("dma_in_0",)
    assert by_name["dma_out_0"].depends_on == ("kernel_0",)
    assert by_name["dma_in_1"].depends_on == ("dma_in_0",)
    assert by_name["dma_in_2"].depends_on == ("dma_in_1", "dma_out_0")
    assert by_name["kernel_2"].tick > by_name["kernel_0"].tick
    assert schedule.serial_ticks == 9
    assert schedule.overlaps is True
    assert "no device streams or events are created" in schedule.text


def test_pipeline_accepts_descriptor_and_validates_arguments():
    schedule = accel_async_stream_pipelining(
        accel_detect_tensor_polyhedral(add_nest()), depth=1, streams=2
    )
    assert len(schedule.events) == 3

    with pytest.raises(PillarError) as excinfo:
        accel_async_stream_pipelining(descriptor("add"), depth=0)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_async_stream_pipelining(descriptor("add"), streams=1)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_async_stream_pipelining(object(), span=(1, 4))
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == (1, 4)


# ---------------------------------------------------------------------------
# 88 -- deterministic tiling sweep
# ---------------------------------------------------------------------------


def test_auto_tune_is_deterministic_and_chooses_minimum_cost():
    desc = descriptor("mul", ("a", "b", "out"), ("out",), shape=(128, 128))
    first = accel_auto_tune_tile_sizes(desc)
    second = accel_auto_tune_tile_sizes(desc)
    assert first == second
    assert first.shape == (128, 128)
    assert first.best.cost_cycles == min(trial.cost_cycles for trial in first.trials)
    assert [trial.cost_cycles for trial in first.trials] == sorted(
        trial.cost_cycles for trial in first.trials
    )
    assert first.best.estimated_us > 0
    assert "no measurement" in first.model


def test_auto_tune_estimated_us_matches_the_cycle_model():
    desc = descriptor("mul", ("a", "b", "out"), ("out",), shape=(128, 128))
    result = accel_auto_tune_tile_sizes(desc)
    assert all(
        trial.estimated_us == round(trial.cost_cycles / 2400.0, 6)
        for trial in result.trials
    )


def test_auto_tune_explicit_candidates_and_shape_from_constants():
    desc = descriptor("add", ("a",), ("out",), shape=(64, 64))
    result = accel_auto_tune_tile_sizes(desc, candidates=((8, 8), (16, 16), (32, 32)))
    assert result.shape == (64, 64)
    assert [trial.tile for trial in result.trials] == [(8, 8), (16, 16), (32, 32)]
    assert result.best.tile == (8, 8)  # deterministic tie-break

    symbolic = accel_detect_tensor_polyhedral(add_nest())
    tuned = accel_auto_tune_tile_sizes(symbolic, default_dim=64)
    assert tuned.shape == (64, 64)


def test_auto_tune_validates_candidate_rank_and_sizes():
    desc = descriptor("add", ("a",), ("out",), shape=(16, 16))
    with pytest.raises(PillarError) as excinfo:
        accel_auto_tune_tile_sizes(desc, candidates=((8,),), span=(2, 6))
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == (2, 6)

    with pytest.raises(PillarError) as excinfo:
        accel_auto_tune_tile_sizes(desc, candidates=((8, 0),))
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_auto_tune_tile_sizes(desc, candidates=())
    assert excinfo.value.code == "E003"


# ---------------------------------------------------------------------------
# 89 -- quantization with explicit verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "int8"])
def test_quantization_satisfies_declared_error_bounds(dtype):
    values = [0.0, 0.1, -0.3, 1.0, 3.14159, -12.5]
    result = accel_mixed_precision_quantize(values, dtype=dtype)
    assert result.dtype == dtype
    assert result.verified is True
    assert len(result.quantized) == len(values)
    assert len(result.bits) == len(values)
    assert all(
        error <= bound
        for error, bound in zip(result.errors, result.error_bounds, strict=True)
    )
    assert result.max_error <= result.error_bound
    if dtype == "int8":
        assert all(isinstance(value, int) for value in result.quantized)
        assert result.scale == pytest.approx(12.5 / 127)
        assert result.max_error <= result.scale / 2 + 1e-8
    else:
        assert all(isinstance(value, float) for value in result.quantized)


def test_bf16_bound_is_looser_than_fp16_bound():
    values = [0.1, -3.75, 12.5]
    fp16 = accel_mixed_precision_quantize(values, dtype="fp16")
    bf16 = accel_mixed_precision_quantize(values, dtype="bf16")
    assert bf16.error_bound > fp16.error_bound
    assert bf16.max_error >= fp16.max_error


def test_quantization_rejects_bad_inputs_and_overflow():
    for values, dtype, code in (
        ([float("nan")], "fp16", "E003"),
        ([float("inf")], "fp16", "E003"),
        ([], "fp16", "E003"),
        ([1.0], "fp32", "E003"),
        ([1e6], "fp16", "E001"),
        (["not a number"], "int8", "E003"),  # type: ignore[list-item]
    ):
        with pytest.raises(PillarError) as excinfo:
            accel_mixed_precision_quantize(values, dtype=dtype, span=(4, 8))
        assert excinfo.value.code == code
        assert excinfo.value.span == (4, 8)


def test_quantization_verification_failure_is_e002(monkeypatch):
    original = hetero._quantize_scalar

    def broken(value, dtype, scale):
        quantized, _, bits, bound = original(value, dtype, scale)
        return quantized, float(quantized) + 10.0, bits, bound

    monkeypatch.setattr(hetero, "_quantize_scalar", broken)
    with pytest.raises(PillarError) as excinfo:
        accel_mixed_precision_quantize([1.0], dtype="fp16")
    assert excinfo.value.code == "E002"
    assert excinfo.value.diagnostic.note == "QuantizationBoundViolation"

    unchecked = accel_mixed_precision_quantize([1.0], dtype="fp16", strict=False)
    assert unchecked.verified is False


# ---------------------------------------------------------------------------
# 90 -- CPU fallback
# ---------------------------------------------------------------------------


def test_cpu_fallback_is_honest_about_timing():
    desc = accel_detect_tensor_polyhedral(matmul_nest())
    plan = accel_nanosecond_cpu_fallback(desc, lanes=8, cycles_per_ns=3.0)
    assert plan.iterations == 256 * 256
    assert plan.vectorized is True
    assert plan.estimated_ns > 0
    assert plan.guaranteed_ns is None
    assert plan.speedup_estimate >= 1.0
    assert "no timing guarantee" in plan.text
    assert "guaranteed_ns = none" in plan.text
    assert "estimated" in plan.text and "not measured" in plan.text


def test_cpu_fallback_does_not_double_count_vector_tail():
    desc = descriptor("add", ("a",), ("out",), shape=(10, 10))
    plan = accel_nanosecond_cpu_fallback(desc, lanes=8)
    iterations = 100
    work = 1 + len(desc.accesses)
    expected = 32 + 4 * 8 + (iterations // 8) * work * 2 + (iterations % 8) * work
    assert plan.iterations == iterations
    assert plan.estimated_cycles == expected


def test_cpu_fallback_scalar_mode_and_fusion_plan():
    desc = descriptor("add", ("a", "b"), ("out",), shape=(64, 16))
    vector = accel_nanosecond_cpu_fallback(desc, lanes=8)
    scalar = accel_nanosecond_cpu_fallback(desc, lanes=1)
    assert scalar.vectorized is False
    assert scalar.estimated_cycles == scalar.scalar_cycles
    assert vector.estimated_ns < scalar.estimated_ns

    plan = accel_kernel_fusion_epilogue(
        [
            descriptor("add", ("a", "b"), ("t",)),
            descriptor("transpose", ("t",), ("out",)),
        ]
    )
    fallback = accel_nanosecond_cpu_fallback(plan)
    assert fallback.iterations == 64
    assert fallback.work_per_iteration == 3


def test_cpu_fallback_validates_arguments():
    desc = descriptor("add", ("a",), ("out",))
    with pytest.raises(PillarError) as excinfo:
        accel_nanosecond_cpu_fallback(desc, lanes=0)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_nanosecond_cpu_fallback(desc, cycles_per_ns=0)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        accel_nanosecond_cpu_fallback(object(), span=(7, 12))
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == (7, 12)
