"""Typed, hygienic macro engine on a miniature AST."""

from __future__ import annotations

import pytest

from compiler.macro_engine import (
    Add,
    Block,
    Call,
    Hole,
    If,
    Let,
    LitBool,
    LitInt,
    LitStr,
    Return,
    Var,
    infer_type,
    macro_expand_typed_ast_splice,
    macro_resolve_hygienic_scopes,
    macro_verify_backend_expansion_parity,
)
from runtime.diagnostics import PillarError


def _swap_template() -> Let:
    # let tmp = $0; $1
    return Let("tmp", Hole(0), Hole(1))


def test_template_binders_are_renamed() -> None:
    expansion = macro_resolve_hygienic_scopes(_swap_template(), scope_id=4)
    assert expansion.binders == (("tmp", "tmp#4"),)
    body = expansion.template
    assert isinstance(body, Let)
    assert body.name == "tmp#4"


def test_definition_site_free_names_are_qualified() -> None:
    template = Call("println", (Var("counter"),))
    expansion = macro_resolve_hygienic_scopes(
        template, scope_id=2, definition_site_names=("counter",)
    )
    assert expansion.free == (("counter", "counter@2"),)
    assert expansion.template == Call("println", (Var("counter@2"),))


def test_call_site_capture_is_prevented_and_reported() -> None:
    template = Add(Hole(0), Var("limit"))
    expansion = macro_resolve_hygienic_scopes(
        template, scope_id=7, call_site_names=("limit",)
    )
    assert expansion.captured == ("limit",)
    assert expansion.template == Add(Hole(0), Var("limit@7"))


def test_plain_global_names_stay_bare() -> None:
    template = Call("helper", (Hole(0),))
    expansion = macro_resolve_hygienic_scopes(template, scope_id=1)
    assert expansion.free == ()
    assert expansion.template == template


def test_hygiene_is_deterministic() -> None:
    first = macro_resolve_hygienic_scopes(_swap_template(), scope_id=9)
    second = macro_resolve_hygienic_scopes(_swap_template(), scope_id=9)
    assert first == second


def test_negative_scope_id_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        macro_resolve_hygienic_scopes(_swap_template(), scope_id=-3)
    assert error.value.code == "E003"


def test_block_binders_are_scoped_sequentially() -> None:
    template = Block((Let("x", LitInt(1), LitInt(0)), Var("x")))
    expansion = macro_resolve_hygienic_scopes(template, scope_id=5)
    assert isinstance(expansion.template, Block)
    assert expansion.template.statements[1] == Var("x#5")
    assert expansion.binders == (("x", "x#5"),)


def test_splice_fills_every_hole() -> None:
    result = macro_expand_typed_ast_splice(
        _swap_template(),
        (LitInt(3), Var("counter")),
        param_types=("i64", "unknown"),
    )
    assert result.holes_filled == 2
    assert result.ast == Let("tmp", LitInt(3), Var("counter"))
    assert result.depth == 0


def test_splice_rejects_wrong_arity() -> None:
    with pytest.raises(PillarError) as error:
        macro_expand_typed_ast_splice(
            _swap_template(), (LitInt(1),), param_types=("i64",)
        )
    assert error.value.code == "E002"
    assert "hole" in str(error.value)


def test_splice_rejects_wrong_type() -> None:
    with pytest.raises(PillarError) as error:
        macro_expand_typed_ast_splice(
            Let("n", Hole(0), Hole(0)),
            (LitStr("oops"), LitInt(2)),
            param_types=("i64", "i64"),
        )
    assert error.value.code == "E001"
    assert "str" in str(error.value)


def test_splice_accepts_unknown_names() -> None:
    result = macro_expand_typed_ast_splice(
        Let("v", Hole(0), Return(Hole(0))),
        (Var("payload"), Var("payload")),
        param_types=("i64", "i64"),
    )
    assert result.holes_filled == 2


def test_splice_enforces_the_depth_limit() -> None:
    with pytest.raises(PillarError) as error:
        macro_expand_typed_ast_splice(
            Hole(0), (LitInt(1),), param_types=("i64",), depth=70, max_depth=64
        )
    assert error.value.code == "E002"
    assert "depth" in str(error.value)


def test_infer_type_is_small_and_honest() -> None:
    assert infer_type(LitInt(1)) == "i64"
    assert infer_type(LitStr("")) == "str"
    assert infer_type(LitBool(True)) == "bool"
    assert infer_type(Var("x")) == "unknown"


def test_add_requires_i64_operands() -> None:
    with pytest.raises(PillarError) as error:
        infer_type(Add(LitInt(1), LitStr("x")))
    assert error.value.code == "E001"


def test_backends_agree_on_a_simple_expansion() -> None:
    tree = Let("tmp#4", LitInt(3), Call("println", (Var("tmp#4"),)))
    report = macro_verify_backend_expansion_parity(tree)
    assert report.parity
    assert report.backends == ("python", "rust", "zig")
    assert len({digest for _name, digest in report.digests}) == 1
    assert report.opcodes[-1] == ("call", "println")


def test_divergent_backend_is_reported() -> None:
    tree = Return(LitInt(1))

    def shifted(node):
        return (("const.int", "2"),)

    with pytest.raises(PillarError) as error:
        macro_verify_backend_expansion_parity(tree, emitters={"zig": shifted})
    assert error.value.code == "E004"
    assert "zig" in str(error.value)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        macro_verify_backend_expansion_parity(LitInt(1), backends=("python", "wasm"))
    assert error.value.code == "E004"
    assert "wasm" in str(error.value)


def test_unfilled_hole_cannot_be_emitted() -> None:
    with pytest.raises(PillarError) as error:
        macro_verify_backend_expansion_parity(Return(Hole(0)))
    assert error.value.code == "E002"


def test_hyphen_and_pipeline_expand_then_verify() -> None:
    template = If(Add(Hole(0), LitInt(1)), Return(LitInt(1)), Return(LitInt(0)))
    hygienic = macro_resolve_hygienic_scopes(template, scope_id=3)
    expanded = macro_expand_typed_ast_splice(
        hygienic.template, (Var("n"),), param_types=("i64",)
    )
    report = macro_verify_backend_expansion_parity(expanded.ast)
    assert report.parity
    assert ("add",) in report.opcodes
    assert report.opcodes[0] == ("load", "n")
