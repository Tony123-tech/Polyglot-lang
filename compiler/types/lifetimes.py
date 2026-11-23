"""Lifetime and borrow-region helpers for the FRAME 2 rollout.

Lifetimes are canonical strings such as ``'a`` or ``'static`` and occur inside
the same canonical type strings as :mod:`compiler.types.inference` (for example
``&'a mut Vec<i64>``).  This module only reasons about regions: it parses and
rewrites them, solves ``outlives`` constraints, applies elision rules and runs a
small deterministic borrow check over event lists.

Diagnostics use E001 for invalid borrows or unsatisfied outlives obligations,
E003 for API misuse and E005 for cycles in the outlives graph.  No native
backend or platform-specific facility is involved.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import compiler.core.ast_nodes as ast
from compiler.types.inference import (
    _canonical,
    _fail,
    _iter_items,
    _require_mapping,
    _require_sequence,
    _signature_from_value,
    infer_expr_type,
    infer_function_signature,
)

__all__ = [
    "life_borrow_check",
    "life_check_constraints",
    "life_check_program",
    "life_constraint",
    "life_drop_order",
    "life_elide",
    "life_environment",
    "life_erase",
    "life_extend",
    "life_extract",
    "life_freeze",
    "life_generativity",
    "life_infer_borrow",
    "life_liveness",
    "life_parse",
    "life_region_fingerprint",
    "life_shorten",
    "life_solve",
    "life_substitute",
    "life_unify_regions",
]

_LIFETIME_RE: Final[re.Pattern[str]] = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*")
_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"&(?:(?P<life>'[A-Za-z_][A-Za-z0-9_]*)\s+)?(?P<mutable>mut\s+)?(?P<inner>.+)$"
)
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "let",
        "mut",
        "use",
        "read",
        "write",
        "drop",
        "fn",
        "return",
        "borrow",
        "release",
        "move",
    }
)
_RELATIONS: Final[frozenset[str]] = frozenset({"outlives", "equal"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lifetime(value: object, label: str = "lifetime") -> str:
    """Validate and normalize a lifetime name, reporting E003 when invalid."""
    if not isinstance(value, str):
        _fail("E003", f"{label} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not _LIFETIME_RE.fullmatch(text):
        _fail("E003", f"invalid lifetime {value!r}: expected 'name")
    return text


def _lifetimes(text: str) -> list[str]:
    """Return every lifetime mentioned in *text*, sorted and unique."""
    return sorted(set(_LIFETIME_RE.findall(text)))


def _region_map(value: object, label: str) -> dict[str, Any]:
    """Validate a mapping of lifetime names to lifetime names."""
    raw = _require_mapping(value, label)
    return {
        _lifetime(key, f"{label} key"): _lifetime(item, f"{label} value")
        for key, item in raw.items()
    }


def _constraint(value: object) -> dict[str, Any]:
    """Normalize one outlives constraint record."""
    if isinstance(value, Mapping):
        if "left" not in value or "right" not in value:
            _fail("E003", "lifetime constraint requires left and right keys")
        left, right = value["left"], value["right"]
        relation = str(value.get("relation", "outlives"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        left, right = value
        relation = "outlives"
    else:
        _fail("E003", "lifetime constraint must be a pair or mapping")
    if relation not in _RELATIONS:
        _fail("E003", f"unsupported lifetime relation {relation!r}")
    return {
        "left": _lifetime(left, "left lifetime"),
        "right": _lifetime(right, "right lifetime"),
        "relation": relation,
    }


def _reachable(graph: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """Compute the transitive closure of an outlives graph."""
    result: dict[str, set[str]] = {}
    for source, targets in graph.items():
        seen: set[str] = set()
        pending = list(targets)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph.get(current, []))
        if source not in targets:
            seen.discard(source)
        result[source] = seen
    return result


def _cycles(graph: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Report deterministic cycles in an outlives graph with E005 diagnostics."""
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def visit(name: str) -> None:
        state[name] = 1
        stack.append(name)
        for target in sorted(graph.get(name, [])):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1:
                path = tuple(stack[stack.index(target) :])
                normalized = min(
                    path[index:] + path[:index] for index in range(len(path))
                )
                if normalized not in seen:
                    seen.add(normalized)
                    cycles.append(
                        {
                            "code": "E005",
                            "region": normalized[0],
                            "path": list(normalized),
                            "message": "outlives cycle",
                        }
                    )
        stack.pop()
        state[name] = 2

    for name in sorted(graph):
        if state.get(name, 0) == 0:
            visit(name)
    return cycles


def _signature(value: object) -> dict[str, Any]:
    """Normalize a function signature for elision analysis."""
    return _signature_from_value(value)


def _named(value: object) -> str:
    """Return a stable name for a borrow-check event target."""
    if isinstance(value, ast.Ident):
        return value.name
    if isinstance(value, ast.Expr):
        return type(value).__name__
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("name", "<anonymous>"))
    _fail("E003", f"unsupported borrow target {type(value).__name__}")


# ---------------------------------------------------------------------------
# Public lifetime functions
# ---------------------------------------------------------------------------


def life_parse(lifetime: object) -> dict[str, Any]:
    """Parse a lifetime token into its canonical and bare forms."""
    text = _lifetime(lifetime)
    return {
        "name": text,
        "bare": text[1:],
        "is_static": text == "'static",
        "source": lifetime,
    }


def life_extract(ty: object) -> dict[str, Any]:
    """Collect the lifetime names appearing in a canonical type expression."""
    canonical = _canonical(ty)
    names = _lifetimes(canonical)
    return {"lifetimes": names, "count": len(names), "type": canonical}


def life_infer_borrow(
    expr: ast.Expr, env: Mapping[str, object] | None = None
) -> dict[str, Any]:
    """Infer the region and mutability of a borrow expression.

    Literals and constants are ``'static``; borrowing a tracked identifier
    uses its environment lifetime and everything else is ``'local``.
    """
    if not isinstance(expr, ast.Expr):
        _fail("E003", f"expected an AST expression, got {type(expr).__name__}")
    bindings: dict[str, str] = {}
    if env is not None:
        for name, value in _require_mapping(env, "env").items():
            bindings[str(name)] = (
                _lifetime(value)
                if isinstance(value, str) and value.startswith("'")
                else "'local"
            )
    mutable = False
    target_expr = expr
    if isinstance(expr, ast.Unary) and expr.op in {"&", "&mut"}:
        mutable = expr.op == "&mut" or expr.mutable
        target_expr = expr.operand
    if isinstance(
        target_expr, (ast.LitInt, ast.LitFloat, ast.LitStr, ast.LitChar, ast.LitBool)
    ):
        return {
            "lifetime": "'static",
            "mutable": mutable,
            "target": type(target_expr).__name__,
            "kind": "literal",
            "env": dict(sorted(bindings.items())),
        }
    if isinstance(target_expr, ast.Ident):
        lifetime = bindings.get(target_expr.name, "'local")
        return {
            "lifetime": lifetime,
            "mutable": mutable,
            "target": target_expr.name,
            "kind": "tracked" if target_expr.name in bindings else "local",
            "env": dict(sorted(bindings.items())),
        }
    inferred = infer_expr_type(target_expr)["type"]
    return {
        "lifetime": "'local",
        "mutable": mutable,
        "target": inferred or type(target_expr).__name__,
        "kind": "expression",
        "env": dict(sorted(bindings.items())),
    }


def life_elide(signature: object) -> dict[str, Any]:
    """Apply lifetime elision rules to a function signature."""
    record = _signature(signature)
    inputs: list[str] = []
    for param in record["params"]:
        inputs.extend(_lifetimes(_canonical(param["type"])))
    unique_inputs = sorted(set(inputs))
    return_text = _canonical(record["return"])
    explicit = _lifetimes(return_text)
    if explicit:
        outputs, rule = explicit, "explicit"
    elif len(unique_inputs) == 1:
        outputs, rule = [unique_inputs[0]], "sole-input"
    elif any(str(param.get("name")) == "self" for param in record["params"]):
        self_lifetimes: list[str] = []
        for param in record["params"]:
            if str(param.get("name")) == "self":
                self_lifetimes.extend(_lifetimes(_canonical(param["type"])))
        outputs, rule = sorted(set(self_lifetimes)) or ["'local"], "self-receiver"
    else:
        outputs, rule = ["'e0"], "fresh"
    return {
        "inputs": unique_inputs,
        "outputs": outputs,
        "rule": rule,
        "return_type": return_text,
        "explicit": bool(explicit),
        "function": str(record.get("name", "<anonymous>")),
    }


def life_constraint(
    left: object, right: object, *, relation: str = "outlives"
) -> dict[str, Any]:
    """Build a normalized outlives or equality constraint between two regions."""
    return {
        "left": _lifetime(left, "left lifetime"),
        "right": _lifetime(right, "right lifetime"),
        "relation": relation
        if relation in _RELATIONS
        else _fail("E003", f"unsupported lifetime relation {relation!r}"),
        "code": "",
    }


def life_solve(constraints: Sequence[object]) -> dict[str, Any]:
    """Solve outlives constraints into a transitively closed relation graph."""
    normalized = [
        _constraint(item) for item in _require_sequence(constraints, "constraints")
    ]
    graph: dict[str, set[str]] = {}
    for item in normalized:
        left, right = item["left"], item["right"]
        graph.setdefault(left, set()).add(right)
        if item["relation"] == "equal":
            graph.setdefault(right, set()).add(left)
    closure = _reachable(graph)
    cycles = _cycles(graph)
    return {
        "outlives": {key: sorted(value) for key, value in sorted(closure.items())},
        "cycles": cycles,
        "closed": True,
        "nodes": sorted(
            set(graph) | {name for targets in closure.values() for name in targets}
        ),
    }


def life_check_constraints(
    constraints: Sequence[object], solution: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Check constraints against a solved outlives relation."""
    normalized = [
        _constraint(item) for item in _require_sequence(constraints, "constraints")
    ]
    raw = _require_mapping(solution, "solution")
    graph = {
        _lifetime(key): set(_require_sequence(value, "solution value"))
        for key, value in raw.items()
    }
    closure = _reachable(graph)
    satisfied: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for item in normalized:
        left, right, relation = item["left"], item["right"], item["relation"]
        holds = (
            right in closure.get(left, set())
            if relation == "outlives"
            else left in closure.get(right, set()) and right in closure.get(left, set())
        )
        record = {
            "left": left,
            "right": right,
            "relation": relation,
            "code": "" if holds else "E001",
        }
        (satisfied if holds else violations).append(record)
    return {"satisfied": satisfied, "violations": violations, "ok": not violations}


def life_environment(bindings: Mapping[str, object]) -> dict[str, Any]:
    """Build a variable-to-lifetime environment from annotated types."""
    raw = _require_mapping(bindings, "bindings")
    lifetimes: dict[str, str] = {}
    types: dict[str, str] = {}
    for name, value in raw.items():
        canonical = _canonical(value)
        types[str(name)] = canonical
        found = _lifetimes(canonical)
        lifetimes[str(name)] = found[0] if found else "'local"
    return {
        "lifetimes": dict(sorted(lifetimes.items())),
        "types": dict(sorted(types.items())),
        "names": sorted(lifetimes),
    }


def life_extend(
    env: Mapping[str, object], name: str, lifetime: object
) -> dict[str, Any]:
    """Return an environment extended with one additional lifetime binding."""
    raw = _require_mapping(env, "env")
    if not isinstance(name, str) or not name:
        _fail("E003", "name must be a non-empty string")
    region = _lifetime(lifetime)
    updated = {str(key): _lifetime(value) for key, value in raw.items()}
    added = name not in updated
    updated[name] = region
    return {
        "env": dict(sorted(updated.items())),
        "added": added,
        "name": name,
        "lifetime": region,
    }


def life_shorten(
    region: object, target: object, solution: Mapping[str, Sequence[str]] | None = None
) -> dict[str, Any]:
    """Check whether *region* may be shortened to *target*."""
    source, destination = _lifetime(region), _lifetime(target)
    path: list[str] = []
    if solution is not None:
        raw = _require_mapping(solution, "solution")
        graph = {
            _lifetime(key): list(_require_sequence(value, "solution value"))
            for key, value in raw.items()
        }
        reachable = _reachable(graph)
        path = sorted(reachable.get(source, set()))
        valid = destination in path or source == destination
    else:
        valid = source == destination or source == "'static"
    return {
        "from": source,
        "to": destination,
        "valid": valid,
        "code": "" if valid else "E001",
        "path": path,
    }


def life_generativity(
    scope: int, *, prefix: str = "'g", used: Sequence[str] = ()
) -> dict[str, Any]:
    """Allocate a fresh generative lifetime scoped to *scope*."""
    if isinstance(scope, bool) or not isinstance(scope, int) or scope < 0:
        _fail("E003", "scope must be a non-negative integer")
    if not isinstance(prefix, str) or not prefix:
        _fail("E003", "prefix must be a non-empty string")
    occupied = set(_require_sequence(used, "used"))
    index = 0
    while f"{prefix}{scope}_{index}" in occupied:
        index += 1
    return {
        "name": f"{prefix}{scope}_{index}",
        "scope": scope,
        "index": index,
        "prefix": prefix,
    }


def life_freeze(ty: object) -> dict[str, Any]:
    """Replace every lifetime in a type with ``'static``."""
    canonical = _canonical(ty)
    found = _lifetimes(canonical)
    frozen = canonical
    for name in found:
        frozen = frozen.replace(name, "'static")
    return {"type": frozen, "replaced": found, "count": len(found)}


def life_erase(ty: object) -> dict[str, Any]:
    """Remove explicit lifetimes while preserving reference mutability."""
    canonical = _canonical(ty)
    found = _lifetimes(canonical)

    def rewrite(match: re.Match[str]) -> str:
        mutability = "mut " if match.group("mutable") else ""
        return f"&{mutability}{match.group('inner')}"

    erased = _REF_RE.sub(rewrite, canonical)
    return {"type": erased, "removed": found, "count": len(found)}


def life_substitute(ty: object, mapping: Mapping[str, object]) -> dict[str, Any]:
    """Substitute lifetime names in a canonical type expression."""
    canonical = _canonical(ty)
    replacements = _region_map(mapping, "mapping")
    result = canonical
    for source, destination in sorted(replacements.items()):
        result = re.sub(rf"{re.escape(source)}\b", destination, result)
    return {
        "type": result,
        "replacements": [
            f"{source} -> {destination}"
            for source, destination in sorted(replacements.items())
        ],
        "mapping": dict(sorted(replacements.items())),
    }


def life_unify_regions(
    left: object,
    right: object,
    solution: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Choose the shorter of two regions under an optional outlives solution."""
    source, destination = _lifetime(left), _lifetime(right)
    if source == destination:
        region, relation = source, "equal"
    else:
        raw = _require_mapping(solution, "solution") if solution is not None else {}
        graph = {
            _lifetime(key): list(_require_sequence(value, "solution value"))
            for key, value in raw.items()
        }
        closure = _reachable(graph)
        if destination in closure.get(source, set()):
            region, relation = destination, "outlives"
        elif source in closure.get(destination, set()):
            region, relation = source, "outlived-by"
        else:
            region, relation = min((source, destination)), "unrelated"
    return {
        "region": region,
        "equal": source == destination,
        "relation": relation,
        "left": source,
        "right": destination,
    }


def life_borrow_check(uses: Sequence[object]) -> dict[str, Any]:
    """Run a deterministic borrow check over a sequence of borrow events."""
    events = _require_sequence(uses, "uses")
    active: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, str]] = []
    for index, raw in enumerate(events):
        if isinstance(raw, Mapping):
            event = str(raw.get("action", "read"))
            name = _named(raw.get("name"))
            mutable = bool(raw.get("mutable", False))
        elif isinstance(raw, str):
            parts = raw.split()
            event = parts[0] if parts else "read"
            name = parts[1] if len(parts) > 1 else "<anonymous>"
            mutable = "mut" in parts
        else:
            _fail("E003", f"unsupported borrow event {type(raw).__name__}")
        current = active.get(name)
        if event in {"move", "write"}:
            if current is not None:
                violations.append(
                    {
                        "code": "E001",
                        "index": str(index),
                        "name": name,
                        "message": f"{event} while an active borrow exists",
                    }
                )
            active.pop(name, None)
        elif event == "borrow":
            if current is not None and (mutable or current["mutable"]):
                violations.append(
                    {
                        "code": "E001",
                        "index": str(index),
                        "name": name,
                        "message": "overlapping mutable borrow",
                    }
                )
            else:
                active[name] = {"mutable": mutable, "index": index}
        elif event == "release":
            active.pop(name, None)
        elif event == "read" and current is not None and current["mutable"]:
            violations.append(
                {
                    "code": "E001",
                    "index": str(index),
                    "name": name,
                    "message": "read while mutably borrowed",
                }
            )
        elif event not in {"read", "borrow", "release", "move", "write"}:
            _fail("E003", f"unsupported borrow action {event!r}")
    return {
        "ok": not violations,
        "violations": violations,
        "events": len(events),
        "active": sorted(active),
    }


def life_liveness(statements: Sequence[str]) -> dict[str, Any]:
    """Compute first and last statement indices at which each name is live."""
    items = _require_sequence(statements, "statements")
    ranges: dict[str, list[int]] = {}
    for index, raw in enumerate(items):
        if not isinstance(raw, str):
            _fail("E003", "statement must be a string")
        for word in _WORD_RE.findall(raw):
            if word in _KEYWORDS:
                continue
            if word in ranges:
                ranges[word][1] = index
            else:
                ranges[word] = [index, index]
    return {
        "live": {name: list(span) for name, span in sorted(ranges.items())},
        "last_use": {name: span[1] for name, span in sorted(ranges.items())},
        "names": sorted(ranges),
    }


def life_drop_order(values: Sequence[str]) -> dict[str, Any]:
    """Return reverse creation order for deterministic destructor emission."""
    items = [_named(item) for item in _require_sequence(values, "values")]
    return {"order": list(reversed(items)), "source": items, "count": len(items)}


def life_region_fingerprint(ty: object) -> dict[str, Any]:
    """Hash the canonical lifetime structure of a type."""
    canonical = _canonical(ty)
    names = _lifetimes(canonical)
    payload = "|".join([canonical, *names])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "sha256": digest,
        "lifetimes": names,
        "count": len(names),
        "algorithm": "sha256",
    }


def life_check_program(program: ast.Program) -> dict[str, Any]:
    """Check function signatures for escaping lifetimes and elision mistakes."""
    if not isinstance(program, ast.Program):
        _fail("E003", f"expected an ast.Program, got {type(program).__name__}")
    functions: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    for item in _iter_items(program.items):
        if not isinstance(item, ast.FnDecl):
            continue
        signature = infer_function_signature(item)
        elision = life_elide(signature)
        inputs = elision["inputs"]
        outputs = elision["outputs"]
        escapes = [
            name
            for name in outputs
            if name not in inputs and not name.startswith("'static")
        ]
        for name in escapes:
            issues.append(
                {
                    "code": "E001",
                    "function": item.name,
                    "lifetime": name,
                    "message": f"lifetime {name} escapes its function",
                }
            )
        functions[item.name] = {
            "inputs": inputs,
            "outputs": outputs,
            "rule": elision["rule"],
            "escapes": escapes,
        }
    return {
        "functions": dict(sorted(functions.items())),
        "issues": issues,
        "checked": len(functions),
        "names": sorted(functions),
    }
