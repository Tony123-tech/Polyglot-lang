"""Monomorphization bookkeeping helpers for the FRAME 2 rollout.

A specialization is identified by a canonical ``Name<Arg, ...>`` key.  Caches,
alias maps and dependency graphs use those keys so placeholder registration,
recursive expansion, pruning and cache validation all share one vocabulary.
Where an AST program is available, :func:`mono_program_report` consults the
project's real :class:`monomorph.Monomorphizer` for generated-item counts.

Diagnostics use E002 for template inflation, E003 for API misuse, E005 for
dependency cycles, E006 for missing cache artifacts and E008 for malformed
cache keys.  The implementation is pure Python and platform independent.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Final

import compiler.core.ast_nodes as ast
from compiler.analysis import monomorph
from compiler.frontend.errors import PolyglotError
from compiler.types.inference import (
    _canonical,
    _fail,
    _format,
    _iter_items,
    _node,
    _rank,
    _require_mapping,
    _require_sequence,
    _substitute,
)

__all__ = [
    "mono_cache_stats",
    "mono_collect_params",
    "mono_default_arguments",
    "mono_dependency_graph",
    "mono_detect_cycle",
    "mono_expand_recursive",
    "mono_fingerprint",
    "mono_fresh_name",
    "mono_inflation_report",
    "mono_key",
    "mono_parse_key",
    "mono_program_report",
    "mono_propagate_vec",
    "mono_prune",
    "mono_register",
    "mono_resolve",
    "mono_substitute",
    "mono_topological_order",
    "mono_unify_keys",
    "mono_validate_cache",
]

_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_:]*")
_MANGLE_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _name(value: object, label: str = "name") -> str:
    """Validate a specialization family name."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        _fail("E003", f"{label} must be a qualified identifier, got {value!r}")
    return value


def _dependency_list(value: object, label: str = "dependencies") -> list[str]:
    """Validate and canonicalize a list of dependency keys."""
    return [_canonical(item) for item in _require_sequence(value, label)]


def _graph(value: object) -> dict[str, list[str]]:
    """Normalize a dependency graph mapping node keys to dependency keys."""
    raw = _require_mapping(value, "graph")
    return {_canonical(key): _dependency_list(items) for key, items in raw.items()}


def _walk_type_nodes(value: object) -> list[ast.Node]:
    """Collect AST nodes under *value* in pre-order, skipping source metadata."""
    found: list[ast.Node] = []
    if isinstance(value, ast.Node):
        found.append(value)
    if is_dataclass(value) and not isinstance(value, type):
        children = [
            getattr(value, field.name)
            for field in fields(value)
            if field.name not in {"line", "col", "span", "source_lines"}
        ]
    elif isinstance(value, (list, tuple)):
        children = list(value)
    elif isinstance(value, Mapping):
        children = list(value.values())
    else:
        children = []
    for child in children:
        found.extend(_walk_type_nodes(child))
    return found


def _groups(pairs: Sequence[tuple[Any, Any]]) -> tuple[dict[Any, Any], list[list[Any]]]:
    """Build union-find groups for alias pairs, preserving first appearance."""
    parent: dict[Any, Any] = {}
    order: list[Any] = []

    def find(key: Any) -> Any:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    for canonical, alias in pairs:
        for key in (canonical, alias):
            if key not in parent:
                parent[key] = key
                order.append(key)
        first, second = find(canonical), find(alias)
        if first != second:
            parent[second] = first
    grouped: dict[Any, list[Any]] = {}
    for key in order:
        grouped.setdefault(find(key), []).append(key)
    return parent, [grouped[root] for root in sorted(grouped, key=str)]


# ---------------------------------------------------------------------------
# Public monomorphization functions
# ---------------------------------------------------------------------------


def mono_key(name: object, args: Sequence[object]) -> dict[str, Any]:
    """Build a canonical specialization key from a family name and type args."""
    family = _name(name)
    arguments = [_canonical(arg) for arg in _require_sequence(args, "args")]
    key = f"{family}<{', '.join(arguments)}>" if arguments else family
    return {
        "key": key,
        "name": family,
        "args": arguments,
        "arity": len(arguments),
        "display": key,
    }


def mono_parse_key(key: object) -> dict[str, Any]:
    """Decode a canonical specialization key into its family and arguments."""
    if not isinstance(key, str) or not key.strip():
        _fail("E008", "specialization key must be a non-empty string")
    text = _canonical(key)
    node = _node(text)
    if node[0] != "name":
        _fail("E008", f"malformed specialization key {key!r}")
    return {
        "key": text,
        "name": node[1],
        "args": [_format(arg) for arg in node[2]],
        "arity": len(node[2]),
    }


def mono_fresh_name(
    name: object, args: Sequence[object], *, separator: str = "__"
) -> dict[str, Any]:
    """Mangle a specialization key into a deterministic concrete symbol name."""
    record = mono_key(name, args)
    if not isinstance(separator, str) or not separator:
        _fail("E003", "separator must be a non-empty string")
    parts = [record["name"], *(_MANGLE_RE.sub("_", arg) for arg in record["args"])]
    concrete = separator.join(parts)
    return {
        "name": concrete,
        "key": record["key"],
        "separator": separator,
        "args": record["args"],
    }


def mono_register(cache: dict[Any, Any], key: object, value: object) -> dict[str, Any]:
    """Register a placeholder before expansion; the first value for a key wins."""
    if not isinstance(cache, dict):
        _fail("E003", f"cache must be a dict, got {type(cache).__name__}")
    canonical = _canonical(key)
    created = canonical not in cache
    if created:
        cache[canonical] = value
    return {
        "key": canonical,
        "value": cache[canonical],
        "created": created,
        "size": len(cache),
    }


def mono_resolve(cache: Mapping[Any, Any], key: object) -> dict[str, Any]:
    """Resolve a specialization key, raising E006 when it is not cached."""
    raw = _require_mapping(cache, "cache")
    canonical = _canonical(key)
    if canonical not in raw:
        raise PolyglotError(
            "E006", f"missing specialization artifact for {canonical!r}", phase="types"
        )
    return {"found": True, "key": canonical, "value": raw[canonical], "code": ""}


def mono_substitute(ty: object, mapping: Mapping[str, object]) -> dict[str, Any]:
    """Apply generic-argument substitution to a type expression."""
    canonical = _canonical(ty)
    _require_mapping(mapping, "mapping")
    replacements = dict(
        sorted((str(key), _canonical(value)) for key, value in mapping.items())
    )
    result = _format(_substitute(_node(canonical), replacements))
    return {
        "type": result,
        "mapping": replacements,
        "replacements": [f"{key} -> {value}" for key, value in replacements.items()],
        "unchanged": result == canonical,
    }


def mono_collect_params(item: object) -> dict[str, Any]:
    """Collect generic parameters and their bound names from a declaration."""
    if not isinstance(
        item, (ast.FnDecl, ast.StructDecl, ast.EnumDecl, ast.TraitDecl, ast.ImplBlock)
    ):
        _fail("E003", f"unsupported declaration {type(item).__name__}")
    params = list(getattr(item, "generics", []))
    bounds = {
        param.name: [str(bound.name) for bound in param.bounds] for param in params
    }
    name = getattr(
        item, "name", _canonical(item.ty) if isinstance(item, ast.ImplBlock) else ""
    )
    return {
        "name": str(name),
        "params": [param.name for param in params],
        "bounds": dict(sorted(bounds.items())),
        "count": len(params),
    }


def mono_default_arguments(params: Sequence[object]) -> dict[str, Any]:
    """Generate deterministic ``i64`` defaults for generic parameters."""
    names: list[str] = []
    for param in _require_sequence(params, "params"):
        if isinstance(param, ast.GenericParam):
            names.append(param.name)
        elif isinstance(param, str) and param:
            names.append(param)
        else:
            _fail("E003", f"unsupported generic parameter {type(param).__name__}")
    arguments = ["i64" for _ in names]
    return {"args": arguments, "params": names, "count": len(arguments)}


def mono_dependency_graph(
    specializations: Mapping[str, Sequence[object]],
) -> dict[str, Any]:
    """Build a deterministic dependency graph over specialization keys."""
    raw = _graph(specializations)
    nodes = sorted(set(raw) | {dep for deps in raw.values() for dep in deps})
    outgoing = {key: sorted(set(deps)) for key, deps in sorted(raw.items())}
    edges = [[key, dep] for key, deps in outgoing.items() for dep in deps]
    return {"nodes": nodes, "outgoing": outgoing, "edges": edges, "count": len(edges)}


def mono_topological_order(graph: Mapping[str, Sequence[object]]) -> dict[str, Any]:
    """Order specializations so dependencies precede their users (E005 on cycles)."""
    normalized = _graph(graph)
    nodes = set(normalized)
    for deps in normalized.values():
        nodes.update(deps)
    indegree = {node: 0 for node in nodes}
    dependents: dict[str, list[str]] = {node: [] for node in nodes}
    for node, deps in normalized.items():
        for dep in set(deps):
            indegree[node] += 1
            dependents[dep].append(node)
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    levels: dict[str, int] = {}
    while ready:
        node = heapq.heappop(ready)
        levels[node] = max(
            (levels[dep] + 1 for dep in normalized.get(node, [])), default=0
        )
        order.append(node)
        for dependent in sorted(dependents[node]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(order) != len(nodes):
        pending = sorted(node for node, degree in indegree.items() if degree > 0)
        _fail("E005", f"cyclic specialization dependencies among {', '.join(pending)}")
    return {"order": order, "levels": dict(sorted(levels.items())), "count": len(order)}


def mono_detect_cycle(graph: Mapping[str, Sequence[object]]) -> dict[str, Any]:
    """Report dependency cycles as structured E005 diagnostics."""
    normalized = _graph(graph)
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dep in sorted(set(normalized.get(node, []))):
            if state.get(dep, 0) == 0:
                visit(dep)
            elif state.get(dep) == 1:
                path = tuple(stack[stack.index(dep) :])
                normalized_path = min(
                    path[index:] + path[:index] for index in range(len(path))
                )
                if normalized_path not in seen:
                    seen.add(normalized_path)
                    cycles.append(
                        {
                            "code": "E005",
                            "path": list(normalized_path),
                            "message": "specialization cycle",
                        }
                    )
        stack.pop()
        state[node] = 2

    for node in sorted(
        set(normalized) | {dep for deps in normalized.values() for dep in deps}
    ):
        if state.get(node, 0) == 0:
            visit(node)
    return {"cycles": cycles, "acyclic": not cycles, "nodes": sorted(state)}


def mono_expand_recursive(
    spec: Mapping[str, object],
    cache: dict[Any, Any],
    *,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Expand a recursive generic item with placeholder-before-expansion semantics."""
    if not isinstance(spec, Mapping):
        _fail("E003", "spec must be a mapping")
    if not isinstance(cache, dict):
        _fail("E003", "cache must be a dict")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        _fail("E003", "max_depth must be a positive integer")
    family = _name(spec.get("name"), "spec name")
    arguments = [
        _canonical(arg) for arg in _require_sequence(spec.get("args", []), "spec args")
    ]
    record = mono_key(family, arguments)
    key = record["key"]
    concrete = mono_fresh_name(family, arguments)["name"]
    if key in cache:
        return {
            "key": key,
            "concrete": concrete,
            "items": [cache[key]],
            "created": False,
            "self_references": 0,
            "depth": 0,
            "cache_size": len(cache),
        }
    placeholder = {"kind": "placeholder", "key": key, "concrete": concrete}
    mono_register(cache, key, placeholder)
    items: list[Any] = [placeholder]
    self_references = 0
    depth = 0
    dependencies = spec.get("dependencies", spec.get("fields", []))
    for raw in _require_sequence(dependencies, "spec dependencies"):
        if isinstance(raw, Mapping):
            dep_key = mono_key(raw.get("name", family), raw.get("args", []))["key"]
        else:
            dep_key = _canonical(raw)
        if dep_key == key:
            items.append(cache[key])
            self_references += 1
            continue
        depth += 1
        if depth > max_depth:
            _fail("E005", f"recursive expansion exceeded max_depth={max_depth}")
        if dep_key not in cache:
            mono_register(
                cache,
                dep_key,
                {
                    "kind": "placeholder",
                    "key": dep_key,
                    "concrete": mono_parse_key(dep_key)["name"],
                },
            )
        items.append(cache[dep_key])
    return {
        "key": key,
        "concrete": concrete,
        "items": items,
        "created": True,
        "self_references": self_references,
        "depth": depth,
        "cache_size": len(cache),
    }


def mono_unify_keys(
    cache: dict[Any, Any], aliases: Sequence[tuple[Any, Any]]
) -> dict[str, Any]:
    """Map alias cache keys onto their canonical specialization instance."""
    if not isinstance(cache, dict):
        _fail("E003", "cache must be a dict")
    pairs: list[tuple[Any, Any]] = []
    for raw in _require_sequence(aliases, "aliases"):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            _fail("E003", "alias must be a (canonical, alias) pair")
        pairs.append((raw[0], raw[1]))
    _, grouped = _groups(pairs)
    changed = 0
    groups: list[dict[str, Any]] = []
    for group in grouped:
        value: Any = None
        for key in group:
            if key in cache:
                value = cache[key]
                break
        if value is None:
            continue
        for key in group:
            if key not in cache or cache[key] is not value:
                cache[key] = value
                changed += 1
        groups.append({"keys": sorted(map(str, group)), "value": value})
    return {
        "changed": changed,
        "cache": dict(sorted(cache.items(), key=lambda item: str(item[0]))),
        "groups": groups,
        "aliases": len(pairs),
    }


def mono_propagate_vec(
    elements: Sequence[object], *, expected: object | None = None
) -> dict[str, Any]:
    """Unify vector element types, honoring an expected ``Vec<T>`` annotation."""
    inputs = [_canonical(item) for item in _require_sequence(elements, "elements")]
    expected_text = _canonical(expected) if expected is not None else None
    conflicts: list[dict[str, str]] = []
    element: tuple | None = None
    for index, text in enumerate(inputs):
        node = _node(text)
        if element is None:
            element = node
            continue
        if _format(element) == text:
            continue
        if _rank(element) is not None and _rank(node) is not None:
            element = node if (_rank(node) or 0) > (_rank(element) or 0) else element
        else:
            conflicts.append(
                {
                    "code": "E001",
                    "index": str(index),
                    "found": text,
                    "expected": _format(element),
                }
            )
    expected_element: str | None = None
    if expected_text is not None:
        node = _node(expected_text)
        if node[0] == "name" and node[1] == "Vec" and node[2]:
            expected_element = _format(node[2][0])
            if element is None:
                element = node[2][0]
            elif _format(element) != expected_element and not conflicts:
                conflicts.append(
                    {
                        "code": "E001",
                        "found": _format(element),
                        "expected": expected_element,
                    }
                )
        else:
            conflicts.append(
                {"code": "E001", "found": expected_text, "expected": "Vec<T>"}
            )
    chosen = _format(element) if element is not None else expected_element or "?T0"
    return {
        "element": chosen,
        "vec": f"Vec<{chosen}>",
        "inputs": inputs,
        "conflicts": conflicts,
    }


def mono_prune(
    cache: Mapping[str, Sequence[object]], roots: Sequence[str]
) -> dict[str, Any]:
    """Keep only specializations reachable from the supplied roots."""
    graph = _graph(cache)
    reachable: set[str] = set()
    pending = [_canonical(root) for root in _require_sequence(roots, "roots")]
    while pending:
        key = pending.pop()
        if key in reachable:
            continue
        reachable.add(key)
        pending.extend(graph.get(key, []))
    nodes = sorted(set(graph) | {dep for deps in graph.values() for dep in deps})
    return {
        "kept": sorted(reachable & set(nodes)),
        "removed": sorted(set(nodes) - reachable),
        "visited": len(reachable),
        "count": len(nodes),
    }


def mono_inflation_report(
    specializations: Mapping[str, int], *, max_instantiations: int = 8
) -> dict[str, Any]:
    """Summarize per-family instantiation counts and emit E002 warnings."""
    raw = _require_mapping(specializations, "specializations")
    if (
        isinstance(max_instantiations, bool)
        or not isinstance(max_instantiations, int)
        or max_instantiations < 1
    ):
        _fail("E003", "max_instantiations must be a positive integer")
    families: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    total = 0
    for family, count in sorted(raw.items()):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("E003", f"instantiation count for {family!r} must be non-negative")
        total += count
        families.append({"family": str(family), "instantiations": count})
        if count > max_instantiations:
            warnings.append(
                {
                    "code": "E002-INF1",
                    "family": str(family),
                    "message": f"{family} produced {count} specializations (limit {max_instantiations})",
                }
            )
    return {
        "families": families,
        "warnings": warnings,
        "total": total,
        "over_limit": [warning["family"] for warning in warnings],
    }


def mono_fingerprint(key: object) -> dict[str, Any]:
    """Hash a canonical specialization key for stable cache comparison."""
    decoded = mono_parse_key(key)
    digest = hashlib.sha256(decoded["key"].encode("utf-8")).hexdigest()
    return {"sha256": digest, "key": decoded["key"], "algorithm": "sha256"}


def mono_cache_stats(cache: Mapping[Any, Any]) -> dict[str, Any]:
    """Summarize a specialization cache by key and value kind."""
    raw = _require_mapping(cache, "cache")
    keys = sorted(str(key) for key in raw)
    kinds: dict[str, int] = {}
    for value in raw.values():
        kind = (
            str(value.get("kind", "value"))
            if isinstance(value, Mapping)
            else type(value).__name__
        )
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "size": len(raw),
        "keys": keys,
        "kinds": dict(sorted(kinds.items())),
        "values": len(raw),
    }


def mono_validate_cache(cache: Mapping[Any, Any]) -> dict[str, Any]:
    """Validate cache keys and dependencies, reporting E008/E006 issues."""
    raw = _require_mapping(cache, "cache")
    issues: list[dict[str, str]] = []
    for key, value in sorted(raw.items(), key=lambda item: str(item[0])):
        try:
            mono_parse_key(key)
        except PolyglotError as error:
            issues.append(
                {"code": "E008", "key": str(key), "message": error.diagnostic.message}
            )
            continue
        if isinstance(value, Mapping):
            for dep in value.get("dependencies", []):
                if dep not in raw:
                    issues.append(
                        {
                            "code": "E006",
                            "key": str(key),
                            "message": f"missing dependency {dep!r}",
                        }
                    )
    return {
        "valid": not issues,
        "issues": issues,
        "size": len(raw),
    }


def mono_program_report(program: ast.Program) -> dict[str, Any]:
    """Report generic families and generated counts from the real monomorphizer."""
    if not isinstance(program, ast.Program):
        _fail("E003", f"expected an ast.Program, got {type(program).__name__}")
    worker = monomorph.Monomorphizer(copy.deepcopy(program))
    worker.run()
    generics = sorted(
        set(worker.generic_fns)
        | set(worker.generic_structs)
        | set(worker.generic_enums)
    )
    keys: list[str] = []
    for item in _iter_items(program.items):
        for node in _walk_type_nodes(item):
            if isinstance(node, ast.TypeName) and node.args:
                keys.append(
                    mono_key(
                        node.name,
                        [_format(_node(_canonical(arg))) for arg in node.args],
                    )["key"]
                )
    return {
        "generics": generics,
        "keys": sorted(set(keys)),
        "generated": {
            "functions": len(worker.generated_fns),
            "structs": len(worker.generated_structs),
        },
        "checked": len(generics),
    }
