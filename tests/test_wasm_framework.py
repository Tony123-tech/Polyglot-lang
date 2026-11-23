"""Focused tests for the WASM lowering/ABI/interop frame (Frame B)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from backends.wasm import (
    AbiField,
    AbiSpec,
    HostSignature,
    wasm_bind_host_imports_and_exports,
    wasm_guard_trap_and_oom_paths,
    wasm_isolate_instance_state,
    wasm_layout_linear_memory_and_stack,
    wasm_lower_ir_to_module,
    wasm_marshal_adt_and_result_abi,
    wasm_unmarshal_adt_and_result_abi,
    wasm_verify_dual_backend_parity,
)
from backends.wasm import framework as wasm_frame
from runtime.diagnostics import PillarError

SOURCE = "fn add(a: i64, b: i64) -> i64 { a + b }"
SPAN = (4, 12)


@pytest.fixture
def artifact() -> Any:
    pytest.importorskip("wasmtime")
    return wasm_lower_ir_to_module(SOURCE, span=SPAN)


def test_lower_ir_to_module_emits_wat_and_wasm(artifact: Any) -> None:
    assert artifact.exports == ("add",)
    assert "(module" in artifact.wat
    assert artifact.wasm is not None and len(artifact.wasm) > 100
    assert artifact.signatures["add"].params == ("i64", "i64")
    assert artifact.signatures["add"].ret == "i64"
    assert artifact.span == SPAN
    assert artifact.as_dict()["lowered_from"] == "source"


def test_lower_ir_to_module_accepts_hir_and_mir() -> None:
    pytest.importorskip("wasmtime")
    from compiler.ir.framework import ir_lower_ast_to_hir, ir_lower_hir_to_mir
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    program = parse(lex(SOURCE))
    hir = ir_lower_ast_to_hir(program)
    mir = ir_lower_hir_to_mir(hir)
    assert wasm_lower_ir_to_module(hir).lowered_from == "hir"
    assert wasm_lower_ir_to_module(mir).lowered_from == "mir"


def test_lower_unsupported_construct_is_structured() -> None:
    with pytest.raises(PillarError) as raised:
        wasm_lower_ir_to_module(
            "struct Point { x: i64 }\nfn get(p: Point) -> i64 { p.x }", span=SPAN
        )
    assert raised.value.code == "E003"


def test_linear_memory_layout_aligns_and_checks_limits() -> None:
    layout = wasm_layout_linear_memory_and_stack(
        [AbiField("x", "i64"), AbiField("flag", "bool"), AbiField("xs", "slice")],
        stack_slots=[24],
    )
    assert layout.field_offsets["x"] == 0
    assert layout.field_offsets["flag"] == 8
    assert layout.field_offsets["xs"] == 16
    assert layout.stack_base % 16 == 0
    assert layout.stack_slots == (32,)
    assert layout.stack_top == layout.stack_base + 32

    with pytest.raises(PillarError) as raised:
        wasm_layout_linear_memory_and_stack(
            [AbiField("x", "i64")], max_bytes=4, span=SPAN
        )
    assert raised.value.code == "E008"
    assert raised.value.span == SPAN


def test_marshal_round_trip_for_struct_result_str_and_slice() -> None:
    layout = wasm_layout_linear_memory_and_stack(
        [AbiField("x", "i64"), AbiField("flag", "bool")], stack_slots=[16]
    )
    struct_spec = AbiSpec(
        "struct", (AbiField("x", "i64"), AbiField("flag", "bool")), "point"
    )
    value = {"x": -17, "flag": True}
    blob = wasm_marshal_adt_and_result_abi(value, spec=struct_spec, layout=layout)
    assert (
        wasm_unmarshal_adt_and_result_abi(blob, spec=struct_spec, layout=layout)
        == value
    )

    result_spec = AbiSpec(
        "result", (AbiField("tag", "bool"), AbiField("value", "i64")), "answer"
    )
    assert wasm_unmarshal_adt_and_result_abi(
        wasm_marshal_adt_and_result_abi(("Ok", 99), spec=result_spec, layout=layout),
        spec=result_spec,
        layout=layout,
    ) == ("Ok", 99)
    assert wasm_unmarshal_adt_and_result_abi(
        wasm_marshal_adt_and_result_abi(("Err", None), spec=result_spec, layout=layout),
        spec=result_spec,
        layout=layout,
    ) == ("Err", None)

    str_spec = AbiSpec("str", (), "text")
    assert (
        wasm_unmarshal_adt_and_result_abi(
            wasm_marshal_adt_and_result_abi("héllo", spec=str_spec, layout=layout),
            spec=str_spec,
            layout=layout,
        )
        == "héllo"
    )
    slice_spec = AbiSpec("slice", (), "items")
    assert wasm_unmarshal_adt_and_result_abi(
        wasm_marshal_adt_and_result_abi([1, -2, 3], spec=slice_spec, layout=layout),
        spec=slice_spec,
        layout=layout,
    ) == [1, -2, 3]


def test_marshal_rejects_bad_shape_and_out_of_range() -> None:
    layout = wasm_layout_linear_memory_and_stack([AbiField("x", "i64")])
    spec = AbiSpec("struct", (AbiField("x", "i64"),), "one")
    with pytest.raises(PillarError) as raised:
        wasm_marshal_adt_and_result_abi({}, spec=spec, layout=layout, span=SPAN)
    assert raised.value.code == "E001"

    with pytest.raises(PillarError) as raised:
        wasm_marshal_adt_and_result_abi(
            {"x": 2**70}, spec=spec, layout=layout, span=SPAN
        )
    assert raised.value.code == "E008"


def test_trap_paths_are_structured_and_strict_mode_raises() -> None:
    layout = wasm_layout_linear_memory_and_stack(
        [AbiField("x", "i64")], stack_slots=[16]
    )
    assert wasm_guard_trap_and_oom_paths(layout, accesses=[(0, 8)]).kind == "none"

    oob = wasm_guard_trap_and_oom_paths(layout, accesses=[(layout.stack_top, 1)])
    assert oob.kind == "out_of_bounds" and not oob.ok
    overflow = wasm_guard_trap_and_oom_paths(layout, stack_depth=2)
    assert overflow.kind == "stack_overflow"
    oom = wasm_guard_trap_and_oom_paths(layout, requested_bytes=layout.max_bytes + 1)
    assert oom.kind == "oom"
    with pytest.raises(PillarError) as raised:
        wasm_guard_trap_and_oom_paths(
            layout, accesses=[(-1, 4)], strict=True, span=SPAN
        )
    assert raised.value.code == "E008"
    assert raised.value.span == SPAN


def test_host_import_bridge_calls_real_wasmtime(artifact: Any) -> None:
    binding = wasm_bind_host_imports_and_exports(
        artifact,
        imports=[HostSignature("host_add", ("i64", "i64"), "i64")],
        exports=[HostSignature("add", ("i64", "i64"), "i64", source="host_add")],
        host_functions={"host_add": lambda x, y: x + y},
    )
    assert binding.namespace["add"](20, 22) == 42
    assert binding.as_dict()["callable_exports"] == ["add"]

    with pytest.raises(PillarError) as raised:
        binding.namespace["add"](1)
    assert raised.value.code == "E001"


def test_host_bridge_validates_signatures_and_surfaces_failures(artifact: Any) -> None:
    with pytest.raises(PillarError) as raised:
        wasm_bind_host_imports_and_exports(
            artifact,
            imports=[HostSignature("missing", ("i64",), "i64")],
            exports=[],
            host_functions={},
        )
    assert raised.value.code == "E001"

    with pytest.raises(PillarError) as raised:
        wasm_bind_host_imports_and_exports(
            artifact,
            imports=[HostSignature("host_add", ("i64", "i64"), "i64")],
            exports=[
                HostSignature("add", ("i64", "i64"), "i64", source="not_imported")
            ],
            host_functions={"host_add": lambda x, y: x + y},
        )
    assert raised.value.code == "E001"

    def failing_host(x: int, y: int) -> int:
        raise RuntimeError("host exploded")

    binding = wasm_bind_host_imports_and_exports(
        artifact,
        imports=[HostSignature("host_add", ("i64", "i64"), "i64")],
        exports=[HostSignature("add", ("i64", "i64"), "i64", source="host_add")],
        host_functions={"host_add": failing_host},
    )
    with pytest.raises(PillarError) as raised:
        binding.namespace["add"](1, 2)
    assert raised.value.code == "E008"
    assert "host exploded" in str(raised.value)


def test_instance_state_is_isolated(artifact: Any, tmp_path: Path) -> None:
    with wasm_isolate_instance_state(
        artifact, run_id="run-1", cache_root=tmp_path / "cache"
    ) as first:
        first_id = first.instance_id
        first_store = first.store
        assert first.namespace["add"](2, 3) == 5
        if first.memory is not None:
            first.memory.data_ptr(first_store)[0] = 41  # type: ignore[index]
    with wasm_isolate_instance_state(
        artifact, run_id="run-2", cache_root=tmp_path / "cache"
    ) as second:
        assert second.instance_id != first_id
        assert second.store is not first_store
        assert second.namespace["add"](2, 3) == 5
        if second.memory is not None:
            assert second.memory.data_ptr(second.store)[0] == 0  # type: ignore[index]
    assert first.store is None
    assert first.namespace == {}


def test_dual_backend_parity_python_wasm(artifact: Any) -> None:
    report = wasm_verify_dual_backend_parity(
        SOURCE, calls={"add": (2, 3)}, include_native=False
    )
    assert report.passed
    assert report.outcomes["add"]["results"] == {
        "python": 5,
        "wasm": 5,
    }


def test_dual_backend_parity_detects_divergence(monkeypatch: Any) -> None:
    @contextmanager
    def fake_isolate(artifact: Any, **kwargs: Any) -> Any:
        class Fake:
            def __init__(self) -> None:
                self.namespace = {"add": lambda a, b: 999}

            def as_dict(self) -> dict[str, Any]:
                return {}

        yield Fake()

    monkeypatch.setattr(wasm_frame, "wasm_isolate_instance_state", fake_isolate)
    report = wasm_verify_dual_backend_parity(
        SOURCE, calls={"add": (2, 3)}, include_native=False
    )
    assert not report.passed
    assert report.mismatches
    assert report.mismatches[0]["backends"] == ["python", "wasm"]


@pytest.mark.slow
def test_dual_backend_parity_three_backends(tmp_path: Path) -> None:
    report = wasm_verify_dual_backend_parity(
        SOURCE,
        calls={"add": (2, 3)},
        cache_root=tmp_path / "isolated-cache",
        include_native=True,
    )
    assert "native" not in report.skipped, report.skipped
    assert report.passed, report.mismatches
    assert report.outcomes["add"]["results"]["native"] == 5


def test_parity_skips_unsupported_wasm_source() -> None:
    report = wasm_verify_dual_backend_parity(
        "struct Point { x: i64 }\nfn get(p: Point) -> i64 { p.x }",
        calls={},
        include_native=False,
    )
    assert "wasm" in report.skipped
