"""Bounded formal-synthesis foundations."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from backends import get_backend
from compiler.core.ast_nodes import FnDecl
from proofs.synthesis import (
    I64_MAX,
    I64_MIN,
    Cond,
    ExprNode,
    Interval,
    LockOrder,
    ParamSpec,
    SynthesisSpec,
    _expr_from_text,
    check_certificate,
    compile_both,
    emit_function_ast,
    emit_program,
    emit_source,
    environment,
    extract_counterexample_trace,
    interpret_spec,
    parse_condition,
    synthesize,
    verify_bounds,
    verify_lock_order,
)
from runtime.diagnostics import PillarError

ADD_SPEC = """
function add(a: i64, b: i64)
assume a is between 0 and 100
assume b is between 0 and 100
ensure result == a + b
"""

INC_SPEC = """
function inc(a: i64)
assume a is between 0 and 100
ensure result == a + 1
"""


def test_interpret_spec_extracts_params_ranges_and_postcondition():
    spec = interpret_spec(ADD_SPEC)
    assert spec.name == "add"
    assert [(param.name, param.lo, param.hi) for param in spec.params] == [
        ("a", 0, 100),
        ("b", 0, 100),
    ]
    assert spec.postcondition == "result == a + b"


def test_interpret_spec_requires_a_postcondition():
    with pytest.raises(PillarError) as excinfo:
        interpret_spec("function f(a: i64)")
    assert excinfo.value.code == "E003"


def test_interpret_spec_supports_for_all_sugar():
    spec = interpret_spec("function inc(a: i64)\nfor all a in 0..50, result == a + 1")
    assert spec.params[0].lo == 0
    assert spec.params[0].hi == 50
    assert spec.postcondition == "result == a + 1"


def test_synthesize_proves_a_candidate_and_emits_a_certificate():
    spec = interpret_spec(ADD_SPEC)
    candidate, certificate = synthesize(spec)
    assert candidate.render() in {"(a + b)", "(b + a)"}
    assert certificate.interval == (0, 200)
    assert check_certificate(certificate)
    assert certificate.digest
    assert any(step.rule == "obligation" for step in certificate.steps)


def test_certificate_detects_tampering():
    spec = interpret_spec(INC_SPEC)
    _candidate, certificate = synthesize(spec)
    tampered = dataclasses.replace(certificate, candidate="(a + 2)")
    with pytest.raises(PillarError) as excinfo:
        check_certificate(tampered)
    assert excinfo.value.code == "E008"


def test_certificate_rejects_a_mismatched_interval():
    spec = interpret_spec(INC_SPEC)
    _candidate, certificate = synthesize(spec)
    tampered = dataclasses.replace(certificate, interval=(0, 0))
    with pytest.raises(PillarError) as excinfo:
        check_certificate(tampered)
    assert excinfo.value.code == "E008"


def test_interval_equality_alone_is_not_a_proof():
    spec = interpret_spec(ADD_SPEC)
    env = environment(spec)
    # `2 * a` and `a + b` share the interval [0, 200]; that must not prove
    # equality, or the synthesizer would emit the wrong function.
    conjunction = Cond(
        "==",
        left=ExprNode("mul", args=(ExprNode.const(2), ExprNode.var("a"))),
        right=ExprNode("add", args=(ExprNode.var("a"), ExprNode.var("b"))),
    )
    assert conjunction.check(env) == "unknown"


def test_synthesis_refuses_unbounded_arithmetic():
    spec = SynthesisSpec(
        name="add",
        params=(ParamSpec("a"), ParamSpec("b")),
        postcondition="result == a + b",
    )
    with pytest.raises(PillarError) as excinfo:
        synthesize(spec)
    assert excinfo.value.code == "E002"


def test_unsatisfiable_preconditions_are_e002():
    spec = SynthesisSpec(
        name="bad",
        params=(ParamSpec("a", lo=0, hi=10),),
        postcondition="result == a",
        preconditions=("a >= 8", "a <= 3"),
    )
    with pytest.raises(PillarError) as excinfo:
        synthesize(spec)
    assert excinfo.value.code == "E002"


def test_impossible_postcondition_is_e002():
    spec = SynthesisSpec(
        name="bad",
        params=(ParamSpec("a", lo=0, hi=10),),
        postcondition="result == a * a * a * a * a + 99",
        max_depth=1,
    )
    with pytest.raises(PillarError) as excinfo:
        synthesize(spec)
    assert excinfo.value.code == "E002"


def test_unsupported_condition_grammar_is_e003():
    with pytest.raises(PillarError) as excinfo:
        parse_condition("len([1, 2]) > 0")
    assert excinfo.value.code == "E003"


def test_emit_source_embeds_the_proof_guard():
    spec = interpret_spec(INC_SPEC)
    candidate, certificate = synthesize(spec)
    source = emit_source(spec, candidate)
    assert source.startswith("fn inc(a: i64) -> i64 {")
    assert "assert!(" in source
    assert "let result =" in source
    assert check_certificate(certificate)


def test_emitted_program_runs_on_the_python_backend():
    spec = interpret_spec(INC_SPEC)
    candidate, certificate = synthesize(spec)
    program = emit_program(spec, candidate, certificate)
    namespaces = compile_both(program)
    assert namespaces["python"]["inc"](41) == 42


def test_direct_ast_emission_matches_source_emission():
    spec = interpret_spec(ADD_SPEC)
    candidate, certificate = synthesize(spec)
    fn = emit_function_ast(spec, candidate, certificate)
    assert isinstance(fn, FnDecl)
    program = emit_program(spec, candidate, certificate)
    namespace = get_backend("python", trusted=True).compile(program)
    assert namespace["add"](7, 8) == 15


def test_emission_rejects_an_unrelated_candidate():
    spec = interpret_spec(INC_SPEC)
    _candidate, certificate = synthesize(spec)
    with pytest.raises(PillarError) as excinfo:
        emit_program(spec, ExprNode.var("a"), certificate)
    assert excinfo.value.code == "E003"


def test_bounds_checker_proves_literal_indices():
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    program = parse(lex("fn main() -> i64 { let xs = [1, 2, 3]; xs[0] }"))
    obligations = verify_bounds(program)
    assert [item.status for item in obligations] == ["proved"]


def test_bounds_checker_flags_out_of_range_indices():
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    program = parse(lex("fn main() -> i64 { let xs = [1, 2, 3]; xs[5] }"))
    obligations = verify_bounds(program)
    assert [item.status for item in obligations] == ["failed"]
    assert obligations[0].identifier == "bounds:xs[5]"


def test_bounds_checker_marks_dynamic_indices_unknown():
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    program = parse(lex("fn main(i: i64) -> i64 { let xs = [1, 2, 3]; xs[i] }"))
    obligations = verify_bounds(program)
    assert [item.status for item in obligations] == ["unknown"]


def test_lock_order_accepts_a_total_order():
    order = LockOrder((("a", "b"), ("b", "c")))
    obligation = verify_lock_order(order)
    assert obligation.status == "proved"


def test_lock_order_rejects_a_cycle():
    with pytest.raises(PillarError) as excinfo:
        verify_lock_order(LockOrder((("a", "b"), ("b", "a")), span=(12, 32)))
    assert excinfo.value.code == "E002"
    assert excinfo.value.span == (12, 32)
    assert "a -> b -> a" in str(excinfo.value)


def test_modulo_intervals_follow_python_semantics():
    env = {"a": Interval(I64_MIN, I64_MAX)}
    modulo = ExprNode("mod", args=(ExprNode.var("a"), ExprNode.const(3)))
    assert modulo.interval(env) == Interval(0, 2)
    assert parse_condition("a % 3 == 3").check(env) == "failed"

    negative = ExprNode("mod", args=(ExprNode.var("a"), ExprNode.const(-3)))
    assert negative.interval(env) == Interval(-2, 0)
    assert parse_condition("a % -3 == -5").check(env) == "failed"

    constant = ExprNode("mod", args=(ExprNode.const(-10), ExprNode.const(3)))
    assert constant.interval({}) == Interval(2, 2)

    for divisor in (ExprNode.const(0), ExprNode.var("b")):
        undefined = ExprNode("mod", args=(ExprNode.var("a"), divisor))
        assert undefined.interval(env) is None


def test_polynomial_identity_is_not_used_across_overflow():
    a = ExprNode.var("a")
    b = ExprNode.var("b")
    wrapped = ExprNode("sub", args=(ExprNode("add", args=(a, b)), b))
    env = {"a": Interval(I64_MIN, I64_MAX), "b": Interval(I64_MIN, I64_MAX)}
    assert Cond("==", left=a, right=wrapped).check(env) == "unknown"


def test_bang_notation_round_trips_without_breaking_not_equal():
    condition = parse_condition("!(a == b)")
    assert condition == parse_condition("not (a == b)")
    env = {"a": Interval(0, 0), "b": Interval(1, 1)}
    assert condition.check(env) == "proved"
    assert parse_condition("a != b").check(env) == "proved"
    assert parse_condition("!(a != b)").check(env) == "failed"


def test_condition_and_expression_depth_guards_are_e003():
    with pytest.raises(PillarError) as excinfo:
        parse_condition("not (" * 80 + "a == b" + ")" * 80)
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        _expr_from_text("1+" * 80 + "a")
    assert excinfo.value.code == "E003"


def test_long_lock_order_does_not_exhaust_python_recursion():
    edges = tuple((f"lock{index}", f"lock{index + 1}") for index in range(1200))
    obligation = verify_lock_order(LockOrder(edges))
    assert obligation.status == "proved"


def test_certificate_with_recomputed_digest_must_match_fresh_steps():
    spec = interpret_spec(INC_SPEC)
    _candidate, certificate = synthesize(spec)
    steps = list(certificate.steps)
    steps[0] = dataclasses.replace(
        steps[0], detail={**steps[0].detail, "lo": steps[0].detail["lo"] + 1}
    )
    forged = dataclasses.replace(certificate, steps=tuple(steps))
    body = forged.body()
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    forged = dataclasses.replace(forged, digest=digest)

    with pytest.raises(PillarError) as excinfo:
        check_certificate(forged)
    assert excinfo.value.code == "E008"


def test_emission_rejects_injected_names_and_types():
    unsafe = dataclasses.replace(interpret_spec(INC_SPEC), name="inc); import os")
    candidate, certificate = synthesize(unsafe)
    with pytest.raises(PillarError) as excinfo:
        emit_source(unsafe, candidate)
    assert excinfo.value.code == "E003"
    with pytest.raises(PillarError) as excinfo:
        emit_function_ast(unsafe, candidate, certificate)
    assert excinfo.value.code == "E003"

    bad_type = dataclasses.replace(
        interpret_spec(INC_SPEC),
        params=(ParamSpec("a", "i64); import os", 0, 100),),
    )
    candidate, _certificate = synthesize(bad_type)
    with pytest.raises(PillarError) as excinfo:
        emit_source(bad_type, candidate)
    assert excinfo.value.code == "E003"


def test_counterexample_trace_reports_interval_evidence_without_inventing_a_model():
    failed = extract_counterexample_trace(
        parse_condition("a < b"),
        {"a": Interval(10, 20), "b": Interval(0, 5)},
    )
    assert failed.status == "failed"
    assert "a in [10, 20]" in failed.detail
    assert "b in [0, 5]" in failed.detail

    unknown = extract_counterexample_trace(
        parse_condition("a == b"),
        {"a": Interval(0, 10), "b": Interval(0, 10)},
    )
    assert unknown.status == "unknown"
    assert "not a concrete model" in unknown.detail

    with pytest.raises(PillarError) as excinfo:
        extract_counterexample_trace(
            parse_condition("a < b"),
            {"a": Interval(0, 1), "b": Interval(2, 3)},
        )
    assert excinfo.value.code == "E002"


@pytest.mark.slow
def test_emitted_program_runs_on_the_native_backend(native_backend):
    spec = interpret_spec(INC_SPEC)
    candidate, certificate = synthesize(spec)
    program = emit_program(spec, candidate, certificate)
    namespace = native_backend.compile(program)
    assert namespace["inc"](41) == 42
