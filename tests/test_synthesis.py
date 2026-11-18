"""Bounded formal-synthesis foundations."""

from __future__ import annotations

import dataclasses

import pytest

from ast_nodes import FnDecl
from backends import get_backend
from proofs.synthesis import (
    Cond,
    ExprNode,
    LockOrder,
    ParamSpec,
    SynthesisSpec,
    check_certificate,
    compile_both,
    emit_function_ast,
    emit_program,
    emit_source,
    environment,
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
    from lexer import lex
    from parser import parse

    program = parse(lex("fn main() -> i64 { let xs = [1, 2, 3]; xs[0] }"))
    obligations = verify_bounds(program)
    assert [item.status for item in obligations] == ["proved"]


def test_bounds_checker_flags_out_of_range_indices():
    from lexer import lex
    from parser import parse

    program = parse(lex("fn main() -> i64 { let xs = [1, 2, 3]; xs[5] }"))
    obligations = verify_bounds(program)
    assert [item.status for item in obligations] == ["failed"]
    assert obligations[0].identifier == "bounds:xs[5]"


def test_bounds_checker_marks_dynamic_indices_unknown():
    from lexer import lex
    from parser import parse

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


@pytest.mark.slow
def test_emitted_program_runs_on_the_native_backend(native_backend):
    spec = interpret_spec(INC_SPEC)
    candidate, certificate = synthesize(spec)
    program = emit_program(spec, candidate, certificate)
    namespace = native_backend.compile(program)
    assert namespace["inc"](41) == 42
