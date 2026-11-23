"""Bootstrap subset front end, canonical IR, and fixed-point verification."""

from __future__ import annotations

import pytest

from bootstrap import (
    Artifact,
    bootstrap_diff_stage_ir_hashes,
    bootstrap_stage1_compile_subset,
    bootstrap_verify_stage2_fixed_point,
    digest,
    emit_ir,
    parse_ir,
    parse_subset,
)
from runtime.diagnostics import PillarError

ADD_SOURCE = """
fn add(x, y) -> i64 {
    let sum = x + y;
    return sum;
}
"""

BRANCH_SOURCE = """
fn clamp(n) -> i64 {
    if n < 0 {
        return 0;
    } else {
        return n;
    }
}
"""

TWO_FN_SOURCE = ADD_SOURCE + """
fn twice(n) -> i64 {
    return add(n, n);
}
"""


def test_source_lowers_to_canonical_ir() -> None:
    artifact = bootstrap_stage1_compile_subset(ADD_SOURCE)
    assert artifact.functions == ("add",)
    assert artifact.ir == "(fn add (params x y) (ret i64) (block (let sum (add (var x) (var y))) (return (var sum))))"
    assert artifact.ir_lines == 1
    assert artifact.digest == digest(artifact.ir)


def test_identical_sources_have_identical_digests() -> None:
    left = bootstrap_stage1_compile_subset(ADD_SOURCE)
    right = bootstrap_stage1_compile_subset("  " + ADD_SOURCE.replace("add", "add") + "\n\n")
    assert left.digest == right.digest


def test_syntax_error_is_a_translation_error() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset("fn broken( -> i64 { return 1; }")
    assert error.value.code == "E003"


def test_wrong_return_type_is_type_error() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset("fn f(x) -> bool { return x + 1; }")
    assert error.value.code == "E001"
    assert "return type" in str(error.value)


def test_unknown_variable_is_type_error() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset("fn f() -> i64 { return missing; }")
    assert error.value.code == "E001"
    assert "missing" in str(error.value)


def test_unknown_function_is_missing_artifact() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset("fn f() -> i64 { return ghost(1); }")
    assert error.value.code == "E006"


def test_duplicate_parameter_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset("fn f(x, x) -> i64 { return x; }")
    assert error.value.code == "E003"
    assert "duplicate" in str(error.value)


def test_zig_emitter_renders_a_return_only_function() -> None:
    artifact = bootstrap_stage1_compile_subset(
        "fn one() -> i64 { return 1 + 2; }", emit_zig_text=True
    )
    assert artifact.zig.splitlines()[0] == "fn one() i64 {"
    assert "return (1 + 2);" in artifact.zig


def test_zig_emitter_refuses_unmodelled_bodies() -> None:
    with pytest.raises(PillarError) as error:
        bootstrap_stage1_compile_subset(ADD_SOURCE, emit_zig_text=True)
    assert error.value.code == "E007"


def test_ir_round_trips_through_the_reader() -> None:
    artifact = bootstrap_stage1_compile_subset(BRANCH_SOURCE)
    program = parse_ir(artifact.ir)
    assert emit_ir(program) == artifact.ir


def test_stage2_reaches_a_fixed_point() -> None:
    artifact = bootstrap_stage1_compile_subset(TWO_FN_SOURCE)
    report = bootstrap_verify_stage2_fixed_point(artifact)
    assert report.stable
    assert report.rounds == 1
    assert report.stage2_digest == artifact.digest
    assert report.ir_lines == 2


def test_stage2_rejects_malformed_ir() -> None:
    bad = Artifact(ir="(fn f (params) (ret i64) (block (return (mystery 1))))", digest=digest("x"), functions=("f",), source_chars=1)
    with pytest.raises(PillarError) as error:
        bootstrap_verify_stage2_fixed_point(bad)
    assert error.value.code == "E008"


def test_diff_reports_equality_and_divergence() -> None:
    left = bootstrap_stage1_compile_subset(ADD_SOURCE)
    same = bootstrap_stage1_compile_subset(ADD_SOURCE)
    other = bootstrap_stage1_compile_subset(ADD_SOURCE.replace("sum", "total"))
    equal = bootstrap_diff_stage_ir_hashes(left, same)
    assert equal.equal
    assert "=" in equal.short
    different = bootstrap_diff_stage_ir_hashes(left, other)
    assert not different.equal
    assert different.first_diff_line == 1
    assert "!=" in different.short


def test_parse_subset_is_stable_across_whitespace() -> None:
    tight = parse_subset("fn f(x)->i64{return x+1;}")
    loose = parse_subset("fn f(x) -> i64 {\n  return x + 1;\n}\n")
    assert emit_ir(tight) == emit_ir(loose)
