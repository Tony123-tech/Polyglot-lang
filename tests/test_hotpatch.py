"""Hot-patching foundations: AST diffing, live state migration, native slots."""

from __future__ import annotations

import ctypes
import threading

import pytest

from ast_nodes import FnDecl, StructDecl
from backends.hotpatch import (
    ChangeKind,
    HotpatchManager,
    NativeSymbolTable,
    diff_programs,
)
from backends.python.translator import PythonBackend
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError


def _program(source: str):
    return parse(lex(source))


BASE = """
struct Counter { value: i64 }

impl Counter {
    fn bump(&self) -> i64 { self.value + 1 }
}

fn main() -> i64 { 0 }
"""

BUMP_2 = BASE.replace("self.value + 1", "self.value + 2")

ADD_FIELD = """
struct Counter { value: i64, label: i64 }

impl Counter {
    fn bump(&self) -> i64 { self.value + 2 }
}

fn main() -> i64 { 0 }
"""

# `Option<i64>` lowers to `object`, which has no type-derived zero value.
NO_ZERO_FIELD = """
struct Counter { value: i64, extra: Option<i64> }

impl Counter {
    fn bump(&self) -> i64 { self.value + 1 }
}

fn main() -> i64 { 0 }
"""

WITH_ITEMS = """
const MODE: i64 = 1;

struct Counter { value: i64 }

impl Counter {
    fn bump(&self) -> i64 { self.value + 1 }
}

fn main() -> i64 { 0 }
"""

EXTRA_ITEMS = """
const MODE: i64 = 2;
const EXTRA: i64 = 5;

struct Counter { value: i64 }

impl Counter {
    fn bump(&self) -> i64 { self.value + 1 }
}

fn helper() -> i64 { 9 }

fn main() -> i64 { 0 }
"""

SIGNATURE = BASE.replace(
    "fn bump(&self) -> i64 { self.value + 1 }",
    "fn bump(&self, step: i64) -> i64 { self.value + step }",
)


def test_diff_classifies_body_and_layout_changes():
    base = _program(BASE)
    assert diff_programs(base, _program(BASE)).changes == []

    body = diff_programs(base, _program(BUMP_2))
    assert [(change.name, change.kind) for change in body.changes] == [
        ("bump", ChangeKind.BODY)
    ]

    layout = diff_programs(base, _program(ADD_FIELD))
    assert {(change.name, change.kind) for change in layout.changes} >= {
        ("Counter", ChangeKind.STATE_LAYOUT),
        ("bump", ChangeKind.BODY),
    }
    assert layout.requires_migration


def test_diff_rejects_signature_changes():
    plan = diff_programs(_program(BASE), _program(SIGNATURE))
    assert [change.kind for change in plan.incompatible] == [ChangeKind.SIGNATURE]


def test_hotpatch_swaps_method_and_preserves_identity():
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    assert manager.call("main") == 0
    assert counter.bump() == 11

    result = manager.patch(_program(BUMP_2))
    assert result.generation == 2
    assert result.swapped == 1
    assert counter.bump() == 12
    assert manager.namespace["Counter"](7).bump() == 9


def test_hotpatch_migrates_new_field_with_explicit_default():
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    manager.patch(_program(BUMP_2))
    result = manager.patch(_program(ADD_FIELD), field_defaults={"label": 7})
    assert result.migrated_objects == 1
    assert counter.label == 7
    assert counter.bump() == 12
    assert manager.namespace["Counter"](1, 2).label == 2


def test_hotpatch_rejects_new_required_field():
    manager = HotpatchManager(_program(BASE))
    manager.register(manager.namespace["Counter"](10))
    with pytest.raises(PillarError) as excinfo:
        manager.patch(_program(NO_ZERO_FIELD))
    assert excinfo.value.code == "E001"


def test_hotpatch_rejects_signature_change_without_touching_state():
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    with pytest.raises(PillarError) as excinfo:
        manager.patch(_program(SIGNATURE))
    assert excinfo.value.code == "E001"
    assert counter.bump() == 11


def test_hotpatch_requires_quiescent_point():
    manager = HotpatchManager(_program(BASE))
    with manager.execution(), pytest.raises(PillarError) as excinfo:
        manager.patch(_program(BUMP_2))
    assert excinfo.value.code == "E003"
    manager.patch(_program(BUMP_2))


def test_rollback_restores_code_and_object_state():
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    manager.patch(_program(BUMP_2))
    manager.patch(_program(ADD_FIELD), field_defaults={"label": 7})
    assert counter.bump() == 12
    assert counter.label == 7

    assert manager.rollback() == 2
    assert not hasattr(counter, "label")
    assert counter.bump() == 12
    assert manager.rollback() == 1
    assert counter.bump() == 11


def test_rollback_removes_added_items_and_restores_constants():
    manager = HotpatchManager(_program(WITH_ITEMS))
    assert manager.namespace["MODE"] == 1
    assert "EXTRA" not in manager.namespace
    assert "helper" not in manager.namespace

    manager.patch(_program(EXTRA_ITEMS))
    assert manager.namespace["MODE"] == 2
    assert manager.namespace["EXTRA"] == 5
    assert manager.namespace["helper"]() == 9

    manager.rollback()
    assert manager.namespace["MODE"] == 1
    assert "EXTRA" not in manager.namespace
    assert "helper" not in manager.namespace


def test_program_contains_expected_items():
    program = _program(BASE)
    assert [item.name for item in program.items if isinstance(item, FnDecl)] == ["main"]
    assert [item.name for item in program.items if isinstance(item, StructDecl)] == [
        "Counter"
    ]


def test_native_symbol_table_swaps_atomically():
    signature = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)
    table = NativeSymbolTable()
    table.register("inc", signature, lambda value: value + 1)
    assert table.invoke("inc", 41) == 42

    stop = threading.Event()
    seen: list[int] = []

    def hammer():
        while not stop.is_set():
            seen.append(table.invoke("inc", 0))

    workers = [threading.Thread(target=hammer) for _ in range(4)]
    for worker in workers:
        worker.start()
    try:
        for value in range(2, 20):
            table.swap("inc", lambda n, value=value: n + value)
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    assert seen
    assert table.resolve("inc")(1) == 20
    assert table.swap_count == 18


def test_native_text_patching_is_rejected():
    table = NativeSymbolTable()
    with pytest.raises(PillarError) as excinfo:
        table.patch_text("symbol", b"\x00")
    assert excinfo.value.code == "E007"


def test_compile_helper_matches_backend_namespace():
    program = _program(BASE)
    namespace = PythonBackend(trusted=True).compile(program)
    assert namespace["Counter"](3).bump() == 4


def test_new_primitive_field_gets_an_automatic_zero():
    # The Python backend lowers `i64` to the runtime type `int`, so a migration
    # must recognise runtime types, not just Polyglot TypeName nodes.
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    result = manager.patch(_program(ADD_FIELD))
    assert result.migrated_objects == 1
    assert counter.label == 0
    assert counter.bump() == 12


def test_new_fields_get_type_derived_zeros_for_all_primitives():
    base = "struct S { v: i64 }\nfn main() -> i64 { 0 }"
    added = "struct S { v: i64, a: f64, b: bool, c: String, d: Vec<i64> }\nfn main() -> i64 { 0 }"
    manager = HotpatchManager(_program(base))
    obj = manager.register(manager.namespace["S"](1))
    manager.patch(_program(added))
    assert (obj.a, obj.b, obj.c, obj.d) == (0.0, False, "", [])


def test_registering_the_same_object_twice_does_not_double_migrate():
    manager = HotpatchManager(_program(BASE))
    counter = manager.namespace["Counter"](10)
    manager.register(counter)
    manager.register(counter)
    result = manager.patch(_program(ADD_FIELD), field_defaults={"label": 3})
    assert result.migrated_objects == 1
    manager.rollback()
    assert not hasattr(counter, "label")


def test_body_patch_tolerates_a_non_deepcopyable_live_object():
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))
    counter.value = threading.Lock()
    result = manager.patch(_program(BUMP_2))
    assert result.swapped == 1
    assert manager.generation == 2


def test_call_missing_function_is_e006():
    manager = HotpatchManager(_program(BASE))
    with pytest.raises(PillarError) as excinfo:
        manager.call("does_not_exist")
    assert excinfo.value.code == "E006"


def test_unsupported_item_addition_is_reported_as_incompatible():
    with_module = "mod helper { fn f() -> i64 { 1 } }\n" + BASE
    plan = diff_programs(_program(BASE), _program(with_module))
    assert plan.incompatible
    assert plan.incompatible[0].name == "helper"


def test_native_register_is_atomic_under_concurrency():
    signature = ctypes.CFUNCTYPE(ctypes.c_int64)
    table = NativeSymbolTable()
    barrier = threading.Barrier(8)
    results: list[str] = []

    def attempt():
        barrier.wait()
        try:
            table.register("once", signature, lambda: 1)
            results.append("ok")
        except PillarError as error:
            results.append(error.code)

    workers = [threading.Thread(target=attempt) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert results.count("ok") == 1
    assert results.count("E001") == 7
    assert table.generation == 1
    assert table.swap_count == 0


def test_concurrent_body_patches_keep_generation_consistent():
    manager = HotpatchManager(_program(BASE))
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def hammer():
        barrier.wait()
        try:
            for _ in range(5):
                manager.patch(_program(BUMP_2))
                manager.patch(_program(BASE))
        except BaseException as error:  # pragma: no cover - failure path
            errors.append(error)

    workers = [threading.Thread(target=hammer) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors
    assert manager.generation == 41
    assert manager.patch_count == 40
