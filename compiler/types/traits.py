"""Trait-bound inspection helpers for the FRAME 2 rollout.

Trait obligations are represented as small dictionaries: a bound has a
``param`` and a list of ``traits``, an impl record has a ``trait``, a ``target``
and its ``methods``.  Type expressions use the canonical strings accepted by
:mod:`compiler.types.inference`, so matching a blanket impl such as
``impl<T> Trait for Vec<T>`` is deterministic and independent of source
whitespace.

The diagnostic surface follows E001-E008: E001 marks an unsatisfied or
overlapping impl, E003 marks API misuse, E005 marks a supertrait cycle, and
E006 marks a missing associated type.  All logic is pure Python.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import ast_nodes as ast
from compiler.frontend.errors import PolyglotError
from compiler.types.inference import (
    _canonical,
    _fail,
    _format,
    _is_var,
    _iter_items,
    _match_pattern,
    _node,
    _require_mapping,
    _require_sequence,
    _substitute,
    infer_function_signature,
)

__all__ = [
    "trait_bound_fingerprint",
    "trait_check_bound",
    "trait_check_impl_completeness",
    "trait_check_program",
    "trait_coherence_check",
    "trait_collect_bounds",
    "trait_collect_impls",
    "trait_default_body_map",
    "trait_detect_cycle",
    "trait_dispatch_table",
    "trait_impl_methods",
    "trait_linearize",
    "trait_object_safety",
    "trait_orphan_check",
    "trait_parse_bound",
    "trait_required_methods",
    "trait_resolve_associated_type",
    "trait_resolve_impl",
    "trait_substitute_bound",
    "trait_supertraits",
]

_METADATA_FIELDS: frozenset[str] = frozenset({"line", "col", "span", "source_lines"})
_BUILTIN_IMPLS: dict[str, frozenset[str]] = {
    "Clone": frozenset({"bool", "char", "String", "i64", "u64", "f64"}),
    "Copy": frozenset({"bool", "char", "i64", "u64", "i32", "usize"}),
    "Debug": frozenset({"bool", "char", "String", "i64", "f64"}),
    "Display": frozenset({"bool", "char", "String", "i64", "f64"}),
    "PartialEq": frozenset({"bool", "char", "String", "i64", "f64"}),
    "Eq": frozenset({"bool", "char", "String", "i64"}),
    "Hash": frozenset({"bool", "char", "String", "i64"}),
    "Default": frozenset({"bool", "char", "String", "i64", "f64"}),
    "Send": frozenset({"bool", "char", "String", "i64", "f64"}),
    "Sync": frozenset({"bool", "char", "String", "i64", "f64"}),
}
_CONTAINER_TRAITS: frozenset[str] = frozenset(
    {"Clone", "Copy", "Debug", "PartialEq", "Eq", "Hash", "Send", "Sync"}
)
_CONTAINER_NAMES: frozenset[str] = frozenset({"Vec", "Option", "Result", "Box"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _crate_prefix(name: str) -> str:
    """Return the crate part of a ``crate::Item`` path."""
    return name.split("::", 1)[0] if "::" in name else ""


def _path_name(value: object) -> str:
    """Normalize a possibly qualified path without parsing its crate prefix."""
    if isinstance(value, str) and "::" in value:
        return value.strip()
    return _canonical(value)


def _split_top(text: str, delimiter: str) -> list[str]:
    """Split on a delimiter that is not nested inside angle brackets."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _trait_part(value: object) -> dict[str, Any]:
    """Parse one ``Trait`` or ``Trait<Args>`` fragment."""
    if isinstance(value, ast.TypeName):
        text = _format(_node(_canonical(value)))
    elif isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        if "name" not in value:
            _fail("E003", "trait fragment mapping requires a 'name' key")
        return {
            "name": str(value["name"]),
            "args": [
                _canonical(item)
                for item in _require_sequence(value.get("args", []), "args")
            ],
            "text": str(value["name"]),
        }
    else:
        _fail("E003", f"unsupported trait fragment {type(value).__name__}")
    canonical = _canonical(text)
    node = _node(canonical)
    if node[0] != "name":
        _fail("E003", f"expected a named trait, got {canonical}")
    args = [_format(arg) for arg in node[2]]
    return {
        "name": node[1],
        "args": args,
        "text": f"{node[1]}<{', '.join(args)}>" if args else node[1],
    }


def _bound_text(bound: Mapping[str, Any]) -> str:
    """Render a parsed bound dictionary back to source-like spelling."""
    traits = " + ".join(part["text"] for part in bound["traits"])
    return f"{bound['param']}: {traits}" if traits else str(bound["param"])


def _walk_ast(value: object) -> list[ast.Node]:
    """Collect AST nodes beneath *value* in deterministic pre-order."""
    found: list[ast.Node] = []
    if isinstance(value, ast.Node):
        found.append(value)
    if is_dataclass(value) and not isinstance(value, type):
        children = [
            getattr(value, field.name)
            for field in fields(value)
            if field.name not in _METADATA_FIELDS
        ]
    elif isinstance(value, (list, tuple)):
        children = list(value)
    elif isinstance(value, Mapping):
        children = list(value.values())
    else:
        children = []
    for child in children:
        found.extend(_walk_ast(child))
    return found


def _impl_record(impl: ast.ImplBlock) -> dict[str, Any]:
    """Convert an impl block into the dictionary shape used by this module."""
    trait = _trait_part(impl.trait)["name"] if impl.trait is not None else ""
    methods: list[dict[str, str]] = []
    for item in impl.items:
        if isinstance(item, ast.FnDecl):
            signature = infer_function_signature(item)
            methods.append(
                {
                    "name": item.name,
                    "signature": f"fn({', '.join(p['type'] for p in signature['params'])}) -> {signature['return']}",
                    "is_async": str(bool(item.is_async)),
                }
            )
    associated: dict[str, str] = {}
    for item in impl.items:
        if isinstance(item, ast.TypeAlias):
            associated[item.name] = _canonical(item.ty)
    return {
        "trait": trait,
        "target": _canonical(impl.ty),
        "methods": methods,
        "generics": [param.name for param in impl.generics],
        "associated": associated,
    }


def _records(value: object, label: str) -> list[Mapping[str, Any]]:
    """Validate a sequence of impl records."""
    items = _require_sequence(value, label)
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or "trait" not in item or "target" not in item:
            _fail("E003", f"{label}[{index}] needs 'trait' and 'target' keys")
        result.append(item)
    return result


def _declarations(value: object) -> dict[str, list[str]]:
    """Normalize a trait-to-supertraits mapping."""
    raw = _require_mapping(value, "declarations")
    return {
        str(trait): [
            str(parent) for parent in _require_sequence(parents, "supertraits")
        ]
        for trait, parents in raw.items()
    }


# ---------------------------------------------------------------------------
# Public trait functions
# ---------------------------------------------------------------------------


def trait_parse_bound(
    bound: object, *, default_param: str | None = None
) -> dict[str, Any]:
    """Parse a bound such as ``T: Clone + Into<String>`` into structured parts."""
    parameter = default_param or "T"
    parts: Sequence[object]
    if isinstance(bound, Mapping):
        parameter = str(bound.get("param", parameter))
        parts = _require_sequence(bound.get("traits", []), "traits")
    elif isinstance(bound, ast.TypeName):
        parts = [bound]
    elif isinstance(bound, str):
        text = bound.strip()
        if ":" in text:
            parameter, _, text = text.partition(":")
            parameter = parameter.strip()
        parts = _split_top(text, "+")
    else:
        _fail("E003", f"unsupported bound value {type(bound).__name__}")
    if not parameter:
        _fail("E003", "bound parameter must be a non-empty name")
    traits = [_trait_part(part) for part in parts]
    parsed = {"param": parameter, "traits": traits, "count": len(traits)}
    return {**parsed, "text": _bound_text(parsed)}


def trait_collect_bounds(item: ast.Item) -> dict[str, Any]:
    """Collect generic-parameter bounds declared by one AST item."""
    if not isinstance(
        item, (ast.FnDecl, ast.StructDecl, ast.EnumDecl, ast.TraitDecl, ast.ImplBlock)
    ):
        _fail("E003", f"unsupported item {type(item).__name__}")
    bounds: list[dict[str, Any]] = []
    parameters: list[str] = []
    for param in getattr(item, "generics", []):
        parameters.append(param.name)
        for bound in param.bounds:
            bounds.append(trait_parse_bound(bound, default_param=param.name))
    return {
        "item": getattr(
            item, "name", _canonical(item.ty) if isinstance(item, ast.ImplBlock) else ""
        ),
        "bounds": bounds,
        "count": len(bounds),
        "parameters": sorted(parameters),
    }


def trait_collect_impls(program: ast.Program) -> dict[str, Any]:
    """Index every trait impl in a program by trait name."""
    if not isinstance(program, ast.Program):
        _fail("E003", f"expected an ast.Program, got {type(program).__name__}")
    impls: dict[str, list[dict[str, Any]]] = {}
    for item in _iter_items(program.items):
        if isinstance(item, ast.ImplBlock) and item.trait is not None:
            record = _impl_record(item)
            impls.setdefault(record["trait"], []).append(record)
    return {
        "impls": {trait: list(records) for trait, records in sorted(impls.items())},
        "traits": sorted(impls),
        "count": sum(len(records) for records in impls.values()),
    }


def trait_resolve_impl(
    trait_name: str, target: object, impls: Mapping[str, Sequence[object]]
) -> dict[str, Any]:
    """Find the first impl whose target pattern matches *target*."""
    if not isinstance(trait_name, str) or not trait_name:
        _fail("E003", "trait_name must be a non-empty string")
    canonical = _canonical(target)
    node = _node(canonical)
    _require_mapping(impls, "impls")
    for raw in impls.get(trait_name, []):
        record = _records([raw], "impls")[0]
        variables = {str(name) for name in record.get("generics", [])}
        subst: dict[str, tuple] = {}
        try:
            _match_pattern(_node(_canonical(record["target"])), node, subst, variables)
        except PolyglotError as error:
            if error.code == "E001":
                continue
            raise
        return {
            "found": True,
            "impl": dict(record),
            "substitution": dict(
                sorted((key, _format(value)) for key, value in subst.items())
            ),
            "code": "",
        }
    return {"found": False, "impl": None, "substitution": {}, "code": "E001"}


def trait_check_bound(
    bound: object,
    target: object,
    impls: Mapping[str, Sequence[object]],
    *,
    assume_builtins: bool = True,
) -> dict[str, Any]:
    """Check one bound against known impls and the primitive builtin table."""
    parsed = trait_parse_bound(bound)
    canonical = _canonical(target)
    missing: list[str] = []
    path: list[str] = []
    for part in parsed["traits"]:
        name, args = part["name"], part["args"]
        if assume_builtins and canonical in _BUILTIN_IMPLS.get(name, frozenset()):
            path.append(f"{canonical}: {name} (builtin)")
            continue
        if (
            assume_builtins
            and canonical.startswith(tuple(_CONTAINER_NAMES))
            and name in _CONTAINER_TRAITS
        ):
            path.append(f"{canonical}: {name} (container)")
            continue
        if args and not canonical.startswith(name):
            missing.append(part["text"])
            continue
        resolved = trait_resolve_impl(name, canonical, impls)
        if resolved["found"]:
            path.append(f"{canonical}: {name} (impl)")
        else:
            missing.append(part["text"])
    return {
        "bound": parsed["text"],
        "param": parsed["param"],
        "target": canonical,
        "satisfied": not missing,
        "missing": missing,
        "path": path,
        "code": "" if not missing else "E001",
    }


def trait_check_program(
    program: ast.Program, *, assume_builtins: bool = True
) -> dict[str, Any]:
    """Check every declared generic bound against its concrete substitutions."""
    collected = trait_collect_impls(program)
    checks: list[dict[str, Any]] = []
    for item in _iter_items(program.items):
        declared = trait_collect_bounds(item)
        if not declared["bounds"]:
            continue
        name = declared["item"]
        type_nodes = [
            node
            for node in _walk_ast(item)
            if isinstance(node, ast.TypeName) and node.name == name and node.args
        ]
        for type_node in type_nodes:
            arguments = [_canonical(arg) for arg in type_node.args]
            if len(arguments) != len(declared["parameters"]):
                continue
            for param, argument in zip(declared["parameters"], arguments):
                for bound in declared["bounds"]:
                    if bound["param"] != param:
                        continue
                    checks.append(
                        trait_check_bound(
                            bound,
                            argument,
                            collected["impls"],
                            assume_builtins=assume_builtins,
                        )
                    )
    violations = [check for check in checks if not check["satisfied"]]
    return {
        "checks": checks,
        "violations": violations,
        "satisfied": len(checks) - len(violations),
        "checked": len(checks),
        "code": "E001" if violations else "",
    }


def trait_supertraits(
    trait_name: str, declarations: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Return the transitive supertrait closure of *trait_name*."""
    graph = _declarations(declarations)
    ancestors: set[str] = set()
    edges: list[str] = []
    active: set[str] = set()

    def visit(name: str, depth: int) -> int:
        if name in active:
            _fail("E005", f"supertrait cycle through {name!r}")
        if name in ancestors:
            return depth
        active.add(name)
        best = depth
        for parent in sorted(graph.get(name, [])):
            edges.append(f"{name} -> {parent}")
            best = max(best, visit(parent, depth + 1))
            ancestors.add(parent)
        active.discard(name)
        ancestors.add(name)
        return best

    depth = visit(trait_name, 0)
    return {
        "trait": trait_name,
        "ancestors": sorted(ancestors - {trait_name}),
        "edges": sorted(set(edges)),
        "depth": depth,
    }


def trait_detect_cycle(declarations: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Report every supertrait cycle with a stable diagnostic code."""
    graph = _declarations(declarations)
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def visit(name: str) -> None:
        state[name] = 1
        stack.append(name)
        for parent in sorted(graph.get(name, [])):
            if state.get(parent, 0) == 0:
                visit(parent)
            elif state.get(parent) == 1:
                path = tuple(stack[stack.index(parent) :])
                normalized = min(
                    path[index:] + path[:index] for index in range(len(path))
                )
                if normalized not in seen:
                    seen.add(normalized)
                    cycles.append(
                        {
                            "code": "E005",
                            "path": list(normalized),
                            "message": "supertrait cycle",
                        }
                    )
        stack.pop()
        state[name] = 2

    for name in sorted(graph):
        if state.get(name, 0) == 0:
            visit(name)
    return {"cycles": cycles, "acyclic": not cycles, "traits": sorted(graph)}


def trait_impl_methods(impl: ast.ImplBlock) -> dict[str, Any]:
    """List the methods implemented by one impl block."""
    if not isinstance(impl, ast.ImplBlock):
        _fail("E003", f"expected an ast.ImplBlock, got {type(impl).__name__}")
    record = _impl_record(impl)
    methods = [
        {**method, "is_async": method["is_async"] == "True"}
        for method in record["methods"]
    ]
    return {
        "trait": record["trait"],
        "target": record["target"],
        "methods": methods,
        "count": len(methods),
    }


def trait_required_methods(decl: ast.TraitDecl) -> dict[str, Any]:
    """Split a trait's methods into required signatures and default bodies."""
    if not isinstance(decl, ast.TraitDecl):
        _fail("E003", f"expected an ast.TraitDecl, got {type(decl).__name__}")
    required: list[str] = []
    defaults: list[str] = []
    for item in decl.items:
        if item.fn is None:
            continue
        (defaults if item.fn.body is not None else required).append(item.fn.name)
    return {
        "trait": decl.name,
        "required": sorted(required),
        "defaults": sorted(defaults),
        "count": len(required) + len(defaults),
    }


def trait_check_impl_completeness(
    decl: ast.TraitDecl, impls: Sequence[object]
) -> dict[str, Any]:
    """Check that each impl of *decl* supplies all required methods."""
    required = trait_required_methods(decl)["required"]
    normalized = [
        _impl_record(item) if isinstance(item, ast.ImplBlock) else item
        for item in _require_sequence(impls, "impls")
    ]
    records = _records(normalized, "impls")
    results: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("trait")) != decl.name:
            continue
        provided = {
            str(method["name"])
            for method in _require_sequence(record.get("methods", []), "methods")
            if isinstance(method, Mapping)
        }
        defaults = set(trait_required_methods(decl)["defaults"])
        missing = sorted(set(required) - provided - defaults)
        extra = sorted(provided - set(required) - defaults)
        results.append(
            {
                "trait": decl.name,
                "target": _canonical(record["target"]),
                "missing": missing,
                "extra": extra,
                "complete": not missing,
                "code": "E001" if missing else "",
            }
        )
    return {
        "trait": decl.name,
        "impls": results,
        "complete": all(result["complete"] for result in results),
        "code": "E001" if any(result["missing"] for result in results) else "",
    }


def trait_linearize(
    trait_name: str, declarations: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Return a stable dependency order for a trait and its supertraits."""
    graph = _declarations(declarations)
    detected = trait_detect_cycle(graph)
    if not detected["acyclic"]:
        _fail("E005", "cannot linearize cyclic supertraits")
    dependencies: dict[str, set[str]] = {}
    for name in {trait_name, *graph}:
        pending = [name]
        visited: set[str] = set()
        collected: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            direct = sorted(graph.get(current, []))
            pending.extend(direct)
            collected.update(direct)
        dependencies[name] = collected
    relevant = dependencies.get(trait_name, set()) | {trait_name}
    remaining: list[tuple[int, str]] = [
        (len(dependencies.get(name, set()) & relevant), name) for name in relevant
    ]
    heapq.heapify(remaining)
    order: list[str] = []
    levels: dict[str, int] = {}
    while remaining:
        degree, name = heapq.heappop(remaining)
        levels[name] = degree
        order.append(name)
    return {"order": order, "levels": levels, "count": len(order)}


def trait_dispatch_table(program: ast.Program) -> dict[str, Any]:
    """Build a static dispatch table from trait impls in a program."""
    collected = trait_collect_impls(program)
    entries: list[dict[str, Any]] = []
    for trait, records in collected["impls"].items():
        for record in records:
            methods = {
                str(method["name"]): f"{record['target']}::{method['name']}"
                for method in record.get("methods", [])
                if isinstance(method, Mapping)
            }
            entries.append(
                {
                    "trait": trait,
                    "target": record["target"],
                    "methods": dict(sorted(methods.items())),
                }
            )
    return {"entries": entries, "count": len(entries)}


def trait_coherence_check(impls: Sequence[object]) -> dict[str, Any]:
    """Detect overlapping impl targets for the same trait (E001)."""
    records = _records(impls, "impls")
    conflicts: list[dict[str, str]] = []
    checked = 0
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["trait"]), []).append(record)
    for trait, group in sorted(grouped.items()):
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                checked += 1
                left_node = _node(_canonical(left["target"]))
                right_node = _node(_canonical(right["target"]))
                if left_node == right_node:
                    overlap = True
                else:
                    overlap = _is_var(left_node) or _is_var(right_node)
                if overlap:
                    conflicts.append(
                        {
                            "code": "E001",
                            "trait": trait,
                            "left": _format(left_node),
                            "right": _format(right_node),
                        }
                    )
    return {"conflicts": conflicts, "coherent": not conflicts, "checked": checked}


def trait_substitute_bound(
    bound: object, mapping: Mapping[str, object]
) -> dict[str, Any]:
    """Apply a type substitution to every trait argument in a bound."""
    parsed = trait_parse_bound(bound)
    _require_mapping(mapping, "mapping")
    replacements: dict[str, str] = {}
    traits: list[dict[str, Any]] = []
    for part in parsed["traits"]:
        args = [
            _format(_substitute(_node(arg), {k: v for k, v in mapping.items()}))
            for arg in part["args"]
        ]
        text = f"{part['name']}<{', '.join(args)}>" if args else part["name"]
        traits.append({"name": part["name"], "args": args, "text": text})
        replacements[part["name"]] = text
    result = {**parsed, "traits": traits}
    return {
        "bound": _bound_text(result),
        "traits": [part["text"] for part in traits],
        "replacements": dict(sorted(replacements.items())),
        "count": len(traits),
    }


def trait_orphan_check(
    impl: object, *, local_crates: Sequence[str] = ()
) -> dict[str, Any]:
    """Apply the orphan rule to an impl given the caller's local crates."""
    record = _records([impl], "impl")[0]
    trait = _path_name(record["trait"])
    target = _path_name(record["target"])
    local = {str(crate) for crate in local_crates}
    trait_local = not _crate_prefix(trait) or _crate_prefix(trait) in local
    target_local = not _crate_prefix(target) or _crate_prefix(target) in local
    allowed = trait_local or target_local
    reason = (
        "trait or target is local"
        if allowed
        else "foreign trait cannot be implemented for a foreign target"
    )
    return {
        "allowed": allowed,
        "reason": reason,
        "code": "" if allowed else "E001",
        "trait": trait,
        "target": target,
    }


def trait_object_safety(decl: ast.TraitDecl) -> dict[str, Any]:
    """Report methods that prevent a trait from being used as a trait object."""
    if not isinstance(decl, ast.TraitDecl):
        _fail("E003", f"expected an ast.TraitDecl, got {type(decl).__name__}")
    unsafe_methods: list[str] = []
    reasons: dict[str, str] = {}
    for item in decl.items:
        if item.fn is None:
            continue
        fn = item.fn
        if fn.generics:
            unsafe_methods.append(fn.name)
            reasons[fn.name] = "generic method"
            continue
        if fn.ret is not None and "Self" in _canonical(fn.ret):
            unsafe_methods.append(fn.name)
            reasons[fn.name] = "returns Self"
    return {
        "trait": decl.name,
        "safe": not unsafe_methods,
        "unsafe_methods": sorted(unsafe_methods),
        "reasons": dict(sorted(reasons.items())),
        "code": "E001" if unsafe_methods else "",
    }


def trait_default_body_map(decl: ast.TraitDecl) -> dict[str, Any]:
    """Map each method of a trait to whether it provides a default body."""
    if not isinstance(decl, ast.TraitDecl):
        _fail("E003", f"expected an ast.TraitDecl, got {type(decl).__name__}")
    defaults: dict[str, bool] = {}
    for item in decl.items:
        if item.fn is not None:
            defaults[item.fn.name] = item.fn.body is not None
    return {
        "trait": decl.name,
        "defaults": dict(sorted(defaults.items())),
        "required": sorted(name for name, has_body in defaults.items() if not has_body),
        "count": len(defaults),
    }


def trait_bound_fingerprint(bounds: Sequence[object]) -> dict[str, Any]:
    """Hash a set of bounds in a sorted, source-independent form."""
    canonical = sorted(
        f"{parsed['param']}: "
        + " + ".join(sorted(part["text"] for part in parsed["traits"]))
        for parsed in (
            trait_parse_bound(bound) for bound in _require_sequence(bounds, "bounds")
        )
    )
    digest = hashlib.sha256("|".join(canonical).encode("utf-8")).hexdigest()
    return {
        "sha256": digest,
        "canonical": canonical,
        "count": len(canonical),
        "algorithm": "sha256",
    }


def trait_resolve_associated_type(
    trait_name: str, assoc_name: str, impls: Sequence[object]
) -> dict[str, Any]:
    """Look up an associated type on the first matching impl (E006 if absent)."""
    if not isinstance(trait_name, str) or not isinstance(assoc_name, str):
        _fail("E003", "trait_name and assoc_name must be strings")
    for record in _records(impls, "impls"):
        if str(record["trait"]) != trait_name:
            continue
        associated = _require_mapping(record.get("associated", {}), "associated")
        if assoc_name in associated:
            return {
                "found": True,
                "type": _canonical(associated[assoc_name]),
                "trait": trait_name,
                "associated": assoc_name,
                "code": "",
            }
    return {
        "found": False,
        "type": None,
        "trait": trait_name,
        "associated": assoc_name,
        "code": "E006",
    }
