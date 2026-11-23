"""Hygienic scope resolution for macro templates.

Two rules, both borrowed from Scheme's ``syntax-rules`` literature:

* A name introduced by the template that *binds* something (``let x = ...``)
  is renamed to ``x#<scope>`` so it can never collide with a call-site name.
* A name the template *references* but does not bind resolves at the macro's
  definition site (``n@<scope>``), so a call-site binding with the same
  spelling cannot capture it.

The pass is total and deterministic: same template and scope always produce the
same rewritten AST and the same binder/free tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

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


def binder_name(name: str, scope_id: int) -> str:
    return f"{name}#{scope_id}"


def definition_name(name: str, scope_id: int) -> str:
    return f"{name}@{scope_id}"


@dataclass(frozen=True)
class HygienicExpansion:
    """The rewritten template plus what the pass decided about each name."""

    template: Node
    scope_id: int
    binders: tuple[tuple[str, str], ...]
    free: tuple[tuple[str, str], ...]
    captured: tuple[str, ...]


def _walk(node: Node, env: dict[str, str], scope_id: int, state: dict[str, list]) -> Node:
    if isinstance(node, Var):
        if node.name in env:
            return Var(env[node.name])
        if node.name in state["definition"]:
            qualified = definition_name(node.name, scope_id)
            state["free"].append((node.name, qualified))
            return Var(qualified)
        if node.name in state["call_site"]:
            # Leaving this bare is exactly the capture the pass exists to stop.
            state["captured"].append(node.name)
            qualified = definition_name(node.name, scope_id)
            state["free"].append((node.name, qualified))
            return Var(qualified)
        return node
    if isinstance(node, Let):
        renamed = binder_name(node.name, scope_id)
        state["binders"].append((node.name, renamed))
        inner_env = dict(env)
        inner_env[node.name] = renamed
        value = _walk(node.value, env, scope_id, state)
        body = _walk(node.body, inner_env, scope_id, state)
        return Let(renamed, value, body)
    if isinstance(node, Add):
        return Add(_walk(node.left, env, scope_id, state), _walk(node.right, env, scope_id, state))
    if isinstance(node, Call):
        return Call(node.callee, tuple(_walk(arg, env, scope_id, state) for arg in node.args))
    if isinstance(node, If):
        return If(
            _walk(node.cond, env, scope_id, state),
            _walk(node.then, env, scope_id, state),
            _walk(node.otherwise, env, scope_id, state),
        )
    if isinstance(node, Return):
        return Return(_walk(node.value, env, scope_id, state))
    if isinstance(node, Block):
        current = dict(env)
        statements: list[Node] = []
        for statement in node.statements:
            if isinstance(statement, Let):
                renamed = binder_name(statement.name, scope_id)
                state["binders"].append((statement.name, renamed))
                statements.append(
                    Let(
                        renamed,
                        _walk(statement.value, current, scope_id, state),
                        _walk(statement.body, current, scope_id, state),
                    )
                )
                current[statement.name] = renamed
            else:
                statements.append(_walk(statement, current, scope_id, state))
        return Block(tuple(statements))
    if isinstance(node, (LitInt, LitStr, LitBool, Hole)):
        return node
    raise PillarError(
        "E003",
        f"macro resolver does not understand node {type(node).__name__}",
        help_text="extend compiler/macro_engine/ast.py before expanding this template",
    )


def macro_resolve_hygienic_scopes(
    template: Node,
    *,
    scope_id: int,
    call_site_names: tuple[str, ...] = (),
    definition_site_names: tuple[str, ...] = (),
) -> HygienicExpansion:
    """Rename template binders and qualify template free references."""
    if scope_id < 0:
        raise PillarError("E003", f"macro scope id must be >= 0, got {scope_id}")
    state: dict[str, list] = {
        "binders": [],
        "free": [],
        "captured": [],
        "call_site": set(call_site_names),
        "definition": set(definition_site_names),
    }
    rewritten = _walk(template, {}, scope_id, state)
    return HygienicExpansion(
        template=rewritten,
        scope_id=scope_id,
        binders=tuple(state["binders"]),
        free=tuple(state["free"]),
        captured=tuple(state["captured"]),
    )
