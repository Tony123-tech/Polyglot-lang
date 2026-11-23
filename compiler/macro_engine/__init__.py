"""Hygienic, typed macro expansion with cross-backend verification."""

from __future__ import annotations

from .ast import (
    Add,
    Block,
    Call,
    Hole,
    If,
    Let,
    LitBool,
    LitInt,
    LitStr,
    Node,
    Return,
    Var,
)
from .hygiene import HygienicExpansion, macro_resolve_hygienic_scopes
from .parity import ParityReport, macro_verify_backend_expansion_parity
from .typed_splice import ExpandResult, infer_type, macro_expand_typed_ast_splice

__all__ = [
    "Add",
    "Block",
    "Call",
    "ExpandResult",
    "Hole",
    "HygienicExpansion",
    "If",
    "Let",
    "LitBool",
    "LitInt",
    "LitStr",
    "Node",
    "ParityReport",
    "Return",
    "Var",
    "infer_type",
    "macro_expand_typed_ast_splice",
    "macro_resolve_hygienic_scopes",
    "macro_verify_backend_expansion_parity",
]
