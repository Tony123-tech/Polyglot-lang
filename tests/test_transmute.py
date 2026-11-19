"""Speculative dynamic-to-native transmutation foundations."""

from __future__ import annotations

import threading

import pytest

from runtime.diagnostics import PillarError
from runtime.transmute import (
    DEOPT_COST_MODEL,
    PolyglotSpecializer,
    Specialization,
    SpeculativeTransmuter,
    TypeSignature,
    jit_inline_cache_polymorphic,
    polyglot_type,
    signature_of,
    transmute,
)


def _add(left, right):
    return left + right


class _DoublingSpecializer:
    """Pretend compiler: results are tagged so tests can tell paths apart."""

    def __init__(self) -> None:
        self.requests: list[TypeSignature] = []

    def compile(self, name, signature, fallback):
        del name, fallback
        self.requests.append(signature)
        if all(ty is int for ty in signature.types):
            return Specialization(
                fn=lambda *args: ("native", sum(args)),
                backend="native",
            )
        if all(ty is str for ty in signature.types):
            return Specialization(
                fn=lambda *args: ("native", "".join(args)),
                backend="native",
            )
        return Specialization(fn=lambda *args: ("native", args), backend="native")


def _warm(transmuter: SpeculativeTransmuter, *args, calls: int = 1) -> None:
    for _ in range(calls):
        transmuter(*args)
    assert transmuter.wait()


def test_signature_of_uses_exact_runtime_types():
    assert signature_of((1, 2.0, True)).render() == "(int, float, bool)"


def test_background_specialization_installs_after_the_threshold():
    spec = _DoublingSpecializer()
    fn = transmute(_add, spec, threshold=3)
    try:
        assert fn(1, 2) == 3
        assert fn(1, 2) == 3
        assert fn.path() is None
        assert fn(1, 2) == 3
        assert fn.wait()
        path = fn.path()
        assert path is not None
        assert path.backend == "native"
        assert path.signature == signature_of((1, 2))
        assert fn(4, 5) == ("native", 9)
        stats = fn.stats()
        assert stats.calls == 4
        assert stats.specialized_calls == 1
        assert stats.compilations == 1
        assert spec.requests == [signature_of((1, 2))]
    finally:
        fn.close()


def test_calls_keep_using_the_fallback_while_compilation_runs():
    started = threading.Event()
    release = threading.Event()

    class Blocking:
        def compile(self, name, signature, fallback):
            del name, signature, fallback
            started.set()
            assert release.wait(5)
            return Specialization(lambda *args: ("compiled", sum(args)), "test")

    fn = transmute(_add, Blocking(), threshold=1)
    try:
        assert fn(1, 2) == 3
        assert started.wait(5)
        assert fn(2, 3) == 5  # still on the fallback
        release.set()
        assert fn.wait()
        assert fn(2, 3) == ("compiled", 5)
    finally:
        release.set()
        fn.close()


def test_type_mismatch_deopts_to_the_fallback():
    fn = transmute(_add, _DoublingSpecializer(), threshold=2)
    try:
        _warm(fn, 1, 2, calls=2)
        assert fn(1, 2) == ("native", 3)
        assert fn("a", "b") == "ab"
        stats = fn.stats()
        assert stats.deopts == 1
        assert fn.path() is None
        assert fn(1, 2) == 3
    finally:
        fn.close()


def test_deopt_budget_disables_specialization_permanently():
    fn = transmute(
        _add,
        _DoublingSpecializer(),
        threshold=1,
        max_deopts=1,
    )
    try:
        _warm(fn, 1, 2)
        assert fn(1, 2) == ("native", 3)

        _warm(fn, "a", "b")
        assert fn("a", "b") == ("native", "ab")

        assert fn(1, 2) == 3  # second deopt exhausts the budget
        stats = fn.stats()
        assert stats.deopts == 2
        assert stats.disabled
        assert fn.path() is None
        assert fn(1, 2) == 3
        assert fn.stats().compilations == stats.compilations
    finally:
        fn.close()


def test_keyword_calls_stay_on_the_fallback_without_profiling():
    fn = transmute(_add, _DoublingSpecializer(), threshold=1)
    try:
        assert fn(left=1, right=2) == 3
        assert fn.profile() == ()
        assert fn.stats().fallbacks == 1
    finally:
        fn.close()


class _FailingSpecializer:
    def compile(self, name, signature, fallback):
        del name, signature, fallback
        raise PillarError("E004", "pretend compiler exploded")


def test_specializer_failure_is_contained():
    fn = transmute(_add, _FailingSpecializer(), threshold=1)
    try:
        assert fn(1, 2) == 3
        assert not fn.wait()
        stats = fn.stats()
        assert stats.failures == 1
        assert stats.compilations == 0
        assert fn.path() is None
        assert fn(4, 5) == 9
    finally:
        fn.close()


def test_strict_mode_surfaces_the_compiler_error():
    fn = transmute(
        _add,
        _FailingSpecializer(),
        threshold=1,
        raise_on_compile_error=True,
    )
    try:
        assert fn(1, 2) == 3
        with pytest.raises(PillarError) as excinfo:
            fn.wait()
        assert excinfo.value.code == "E004"
    finally:
        fn.close()


def test_force_specialize_compiles_synchronously():
    fn = transmute(_add, _DoublingSpecializer(), threshold=100)
    try:
        assert fn(1, 2) == 3
        path = fn.force_specialize()
        assert path is not None
        assert path.signature == signature_of((1, 2))
        assert fn(1, 2) == ("native", 3)
    finally:
        fn.close()


def test_force_specialize_without_observations_is_a_noop():
    fn = transmute(_add, _DoublingSpecializer())
    try:
        assert fn.force_specialize() is None
    finally:
        fn.close()


def test_signature_profiles_are_bounded():
    fn = transmute(_add, _DoublingSpecializer(), threshold=100, max_signatures=2)
    try:
        fn(1, 2)
        fn("a", "b")
        fn(1.0, 2.0)
        assert len(fn.profile()) == 2
    finally:
        fn.close()


def test_deoptimize_is_explicit_and_idempotent():
    fn = transmute(_add, _DoublingSpecializer(), threshold=1)
    try:
        _warm(fn, 1, 2)
        assert fn.path() is not None
        fn.deoptimize()
        assert fn.path() is None
        fn.deoptimize()
        assert fn.stats().deopts == 1
    finally:
        fn.close()


def test_context_manager_closes_the_executor():
    with transmute(_add, _DoublingSpecializer(), threshold=1) as fn:
        assert fn(1, 2) == 3
        fn.wait()


def test_deopt_cost_model_is_documented_honestly():
    assert "tuple" in DEOPT_COST_MODEL
    assert "nanosecond" not in DEOPT_COST_MODEL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"max_deopts": -1},
        {"max_signatures": 0},
    ],
)
def test_invalid_construction_parameters_are_rejected(kwargs):
    with pytest.raises(PillarError) as excinfo:
        transmute(_add, _DoublingSpecializer(), **kwargs)
    assert excinfo.value.code == "E003"


def test_calls_after_close_stay_on_the_fallback():
    fn = transmute(_add, _DoublingSpecializer(), threshold=1)
    assert fn(1, 2) == 3
    fn.wait()
    fn.close()
    assert fn(1, 2) == 3
    assert fn(1.0, 2.0) == 3.0
    assert fn(left=1, right=2) == 3
    assert fn.force_specialize(signature_of((1, 2))) is None
    stats = fn.stats()
    assert stats.calls == 4
    assert stats.fallbacks == 4
    assert stats.specialized_calls == 0
    assert stats.active_signature is None
    fn.close()  # idempotent


def test_wait_prunes_finished_futures():
    fn = transmute(_add, _DoublingSpecializer(), threshold=1)
    try:
        _warm(fn, 1, 2)
        assert fn._futures == []
        fn("a", "b")
        fn.wait()
        assert fn._futures == []
    finally:
        fn.close()


def test_deopt_budget_zero_disables_on_the_first_deopt():
    fn = transmute(_add, _DoublingSpecializer(), threshold=1, max_deopts=0)
    try:
        _warm(fn, 1, 2)
        assert fn(1, 2) == ("native", 3)
        # The first type mismatch exhausts the zero budget and disables.
        assert fn("a", "b") == "ab"
        stats = fn.stats()
        assert stats.deopts == 1
        assert stats.disabled
        assert fn.force_specialize() is None
        assert fn("a", "b") == "ab"
    finally:
        fn.close()


def test_a_later_success_does_not_hide_an_earlier_failure():
    class _FailsForInts:
        def compile(self, name, signature, fallback):
            del name, fallback
            if signature.types[0] is int:
                raise PillarError("E004", "int specialization failed")
            return Specialization(lambda *args: ("native", args), "test")

    fn = transmute(_add, _FailsForInts(), threshold=1, raise_on_compile_error=True)
    try:
        assert fn(1, 2) == 3
        with pytest.raises(PillarError, match="int specialization failed"):
            fn.wait()
        assert fn("a", "b") == "ab"
        with pytest.raises(PillarError, match="int specialization failed"):
            fn.wait()
        assert fn.stats().failures == 1
        assert fn.stats().compilations == 1
    finally:
        fn.close()


def test_polyglot_specializer_wraps_source_generation_failures():
    def broken_source(_name, _signature):
        raise ValueError("generator bug")

    specializer = PolyglotSpecializer(broken_source)
    with pytest.raises(PillarError) as excinfo:
        specializer.compile("test", signature_of((1,)), _add)
    assert excinfo.value.code == "E004"
    assert "generator bug" in excinfo.value.reason


def test_inline_cache_prefers_the_hottest_signatures():
    cold = signature_of(("x",))
    warm = signature_of((1,))
    hottest = signature_of((1.0,))
    assert jit_inline_cache_polymorphic(
        [(cold, 1), (hottest, 50), (warm, 10)], ways=2
    ) == ((hottest, 50), (warm, 10))
    with pytest.raises(PillarError) as excinfo:
        jit_inline_cache_polymorphic([], ways=0)
    assert excinfo.value.code == "E003"


def test_concurrent_calls_keep_stats_consistent():
    fn = transmute(_add, _DoublingSpecializer(), threshold=10_000)
    start = threading.Barrier(8)

    def worker() -> None:
        start.wait(timeout=5)
        for index in range(50):
            assert fn(index, 1) == index + 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        stats = fn.stats()
        assert stats.calls == 400
        assert stats.fallbacks == 400
        assert stats.calls == stats.fallbacks + stats.specialized_calls
    finally:
        fn.close()


def test_polyglot_type_maps_the_scalar_subset():
    assert polyglot_type(int) == "i64"
    assert polyglot_type(float) == "f64"
    assert polyglot_type(bool) == "bool"
    with pytest.raises(PillarError) as excinfo:
        polyglot_type(str)
    assert excinfo.value.code == "E007"


def test_polyglot_specializer_drives_a_backend(monkeypatch):
    requested: list[tuple[str, TypeSignature]] = []

    def source_for(name, signature):
        requested.append((name, signature))
        ty = polyglot_type(signature.types[0])
        return f"fn {name}(x: {ty}) -> {ty} {{\n    x + 1\n}}\n"

    class FakeBackend:
        def compile(self, program):
            assert program.items
            return {"special": lambda value: value + 1}

    import backends

    monkeypatch.setattr(backends, "get_backend", lambda *_a, **_k: FakeBackend())

    specializer = PolyglotSpecializer(source_for, backend="native")
    fn = transmute(lambda value: value, specializer, name="special", threshold=1)
    try:
        assert fn(41) == 41
        assert fn.wait()
        assert fn(41) == 42
        stats = fn.stats()
        assert stats.active_backend == "native"
        assert requested == [("special", signature_of((41,)))]
    finally:
        fn.close()


def test_polyglot_specializer_reports_a_missing_export(monkeypatch):
    class EmptyBackend:
        def compile(self, program):
            del program
            return {}

    import backends

    monkeypatch.setattr(backends, "get_backend", lambda *_a, **_k: EmptyBackend())

    specializer = PolyglotSpecializer(
        lambda name, signature: f"fn {name}(x: i64) -> i64 {{ x }}"
    )
    fn = transmute(lambda value: value, specializer, name="special", threshold=1)
    try:
        fn(1)
        assert not fn.wait()
        assert fn.stats().failures == 1
    finally:
        fn.close()


@pytest.mark.slow
def test_specializes_through_the_native_backend(native_backend):
    del native_backend  # ensure cargo is available before compiling
    specializer = PolyglotSpecializer(
        lambda name, signature: (
            f"fn {name}(x: {polyglot_type(signature.types[0])}) -> i64 {{\n"
            "    x + 1\n"
            "}\n"
        ),
        backend="native",
    )
    fn = transmute(
        lambda value: value + 1,
        specializer,
        name="special",
        threshold=1,
        raise_on_compile_error=True,
    )
    try:
        assert fn(41) == 42
        assert fn.wait(timeout=180.0)
        assert fn(41) == 42
        stats = fn.stats()
        assert stats.active_backend == "native"
        assert stats.specialized_calls == 1
    finally:
        fn.close()
