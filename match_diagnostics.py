"""Shared, lightweight diagnostics for match patterns."""

from __future__ import annotations

import warnings

from ast_nodes import (
    Ident,
    MatchExpr,
    PatBinding,
    PatEnum,
    PatIdent,
    PatLiteral,
    PatOr,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    Pattern,
    PatTuple,
    PatWildcard,
)


def warn_match_issues(match: MatchExpr) -> None:
    seen_literals: set[tuple[str, object]] = set()
    unreachable_warned = False
    duplicate_warned = False
    catch_all_seen = False
    for arm in match.arms:
        if catch_all_seen and not unreachable_warned:
            warnings.warn(
                f"unreachable match arm at {arm.line}:{arm.col}",
                UserWarning,
                stacklevel=3,
            )
            unreachable_warned = True
        if isinstance(arm.pattern, PatLiteral):
            literal_value = arm.pattern.value
            literal = (
                type(literal_value).__name__,
                getattr(literal_value, "value", repr(literal_value)),
            )
            if literal in seen_literals and not duplicate_warned:
                warnings.warn(
                    f"duplicate literal match arm {literal[1]!r} "
                    f"at {arm.line}:{arm.col}",
                    UserWarning,
                    stacklevel=3,
                )
                duplicate_warned = True
            seen_literals.add(literal)
        if arm.guard is None and _is_catch_all(arm.pattern):
            catch_all_seen = True

    if isinstance(match.scrutinee, Ident):
        shadow_warned = False
        for arm in match.arms:
            if (
                arm.guard is not None
                and match.scrutinee.name in _bound_names(arm.pattern)
                and not shadow_warned
            ):
                shadow_warned = True
                warnings.warn(
                    f"match binding shadows scrutinee "
                    f"{match.scrutinee.name!r} at {arm.line}:{arm.col}",
                    UserWarning,
                    stacklevel=3,
                )
                break


def _is_catch_all(pattern: Pattern) -> bool:
    if isinstance(pattern, (PatWildcard, PatIdent)):
        return True
    return isinstance(pattern, PatOr) and any(
        _is_catch_all(alt) for alt in pattern.alts
    )


def _bound_names(pattern: Pattern) -> set[str]:
    if isinstance(pattern, PatIdent):
        return {pattern.name}
    if isinstance(pattern, PatBinding):
        return {pattern.name} | _bound_names(pattern.inner)
    if isinstance(pattern, PatRef):
        return _bound_names(pattern.inner)
    if isinstance(pattern, PatOr):
        return set().union(*(_bound_names(alt) for alt in pattern.alts))
    if isinstance(pattern, PatTuple):
        return set().union(*(_bound_names(elem) for elem in pattern.elems))
    if isinstance(pattern, PatSlice):
        return set().union(*(_bound_names(elem) for elem in pattern.elems))
    if isinstance(pattern, PatStruct):
        return set().union(
            *(_bound_names(child) for _name, child in pattern.fields),
        )
    if isinstance(pattern, PatEnum):
        return set().union(*(_bound_names(child) for child in pattern.args))
    if isinstance(pattern, PatRest):
        return set()
    return set()
