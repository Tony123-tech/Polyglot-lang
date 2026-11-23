"""Ergonomic trait and type-system helpers for the Rust backend.

Three things Rust developers hit daily are modelled here, without generating
Rust and without invoking Cargo:

* **coherent extension** -- implement a trait for a type you do not own, and
  find out at plan time whether the orphan rule or overlapping impls forbid it
  (``rste_impl_plan``);
* **blanket composition** -- chain adapters whose ``provides`` satisfies the
  next one's ``requires``, so a caller never writes the bound soup by hand
  (``rste_compose``);
* **fast-path dispatch** -- order specialised handlers ahead of a fallback and
  prove they are disjoint, while stating honestly whether the plan needs
  nightly ``specialization`` or is just monomorphisation
  (``rste_fast_path``), plus the associated-type shapes for iterator/GAT
  adapters (``rste_iter_adapter``).

Everything here is a plan: a deterministic description of what Rust code would
look like and what rustc would reject. The emitted strings are stable for a
given input, and an invalid plan raises
:class:`compiler.frontend.errors.PolyglotError` with the framework codes:

* ``E001`` -- an invalid type or trait name;
* ``E002`` -- a coherence obligation that cannot be satisfied (orphan rule,
  non-exhaustive dispatch, missing composition path);
* ``E003`` -- API misuse (wrong argument type, duplicate handler, bad key);
* ``E005`` -- a cycle in the adapter graph.

The limits are recorded in SPEC 10.56: this models the rules rustc enforces on
stable Rust, not trait solving. It does not know about associated-type
equality, negative impls beyond an explicit exclusion list, or lifetime
polymorphism, so it can approve a plan that rustc later rejects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "rste_compose",
    "rste_fast_path",
    "rste_impl_plan",
    "rste_iter_adapter",
    "rste_try_adapter",
]

#: Scalar types that are `Copy` in both Rust and this language.
_SCALARS = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
    }
)

_CATEGORIES = ("copy", "primitive", "slice", "concrete")


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured trait-planning diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-traits-ext")


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("E003", f"{what} must be a non-empty str, got {type(value).__name__}")
    return value


def _name(value: object, what: str) -> str:
    """A Rust type or trait name: identifier, optional path, no generics."""
    name = _text(value, what)
    for part in name.split("::"):
        if not part.isidentifier():
            _fail("E001", f"{what} is not a valid Rust path segment: {name!r}")
    return name


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        _fail("E003", f"{what} must be a bool, got {type(value).__name__}")
    return value


def _str_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("E003", f"{what} must be a sequence of str")
    return tuple(_name(item, what) for item in value)


# ---------------------------------------------------------------------------
# 1. rste_impl_plan -- coherence: orphan rule, overlap, disjoint specialisation
# ---------------------------------------------------------------------------


def rste_impl_plan(impls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate an impl set against the orphan rule and overlap rules.

    ``impls`` is a sequence of mappings with these keys:

    ``trait``
        Trait name (required).
    ``for_type``
        Self type name (required).
    ``trait_local`` / ``type_local``
        Whether this crate owns the trait / the type (required, bool).
    ``kind``
        ``"concrete"`` (default) or ``"blanket"``.
    ``bounds``
        Bounds that the blanket impl requires (optional, sequence of names).
    ``excludes``
        For a blanket impl, the concrete types it does *not* cover. That is the
        only way stable Rust can keep a blanket impl and a concrete impl
        disjoint, so this is the field that decides whether the plan needs
        nightly ``specialization`` (optional).

    Returns a dict with the normalised impls, the split between concrete and
    blanket impls, whether the plan is stable-reachable, and the emitted
    ``impl`` headers in declaration order.
    """
    if not isinstance(impls, Sequence) or isinstance(impls, (str, bytes)):
        _fail("E003", "impls must be a sequence of mappings")
    if not impls:
        _fail("E003", "impls must not be empty")

    normalised: list[dict[str, Any]] = []
    for raw in impls:
        if not isinstance(raw, Mapping):
            _fail("E003", "each impl must be a mapping")
        trait = _name(raw.get("trait"), "trait")
        for_type = _name(raw.get("for_type"), "for_type")
        trait_local = _flag(raw.get("trait_local"), "trait_local")
        type_local = _flag(raw.get("type_local"), "type_local")
        kind = raw.get("kind", "concrete")
        if kind not in {"concrete", "blanket"}:
            _fail("E003", f"kind must be 'concrete' or 'blanket', got {kind!r}")
        bounds = _str_list(raw.get("bounds", ()), "bounds")
        excludes = _str_list(raw.get("excludes", ()), "excludes")
        if kind == "concrete" and (bounds or excludes):
            _fail(
                "E003",
                "a concrete impl takes no bounds and no excludes; "
                "that only applies to a blanket impl",
            )
        if kind == "concrete" and not type_local and not trait_local:
            _fail(
                "E002",
                f"orphan rule: `impl {trait} for {for_type}` is allowed only when "
                f"this crate owns `{trait}` or `{for_type}`; both are foreign",
            )
        normalised.append(
            {
                "trait": trait,
                "for_type": for_type,
                "trait_local": trait_local,
                "type_local": type_local,
                "kind": kind,
                "bounds": bounds,
                "excludes": excludes,
            }
        )

    concrete = [i for i in normalised if i["kind"] == "concrete"]
    blanket = [i for i in normalised if i["kind"] == "blanket"]

    seen: dict[tuple[str, str], int] = {}
    for index, item in enumerate(concrete):
        key = (item["trait"], item["for_type"])
        if key in seen:
            _fail(
                "E002",
                f"conflicting implementations: `impl {item['trait']} for "
                f"{item['for_type']}` is declared twice (positions "
                f"{seen[key]} and {index})",
            )
        seen[key] = index

    overlapping: list[dict[str, str]] = []
    for b in blanket:
        for c in concrete:
            if b["trait"] != c["trait"]:
                continue
            if c["for_type"] in b["excludes"]:
                continue
            if b["for_type"] == c["for_type"]:
                continue
            overlapping.append(
                {
                    "trait": c["trait"],
                    "blanket_for": b["for_type"],
                    "concrete_for": c["for_type"],
                }
            )

    stable = not overlapping
    if overlapping:
        first = overlapping[0]
        _fail(
            "E002",
            f"`impl {first['trait']} for {first['blanket_for']}` overlaps "
            f"`impl {first['trait']} for {first['concrete_for']}`: a blanket "
            "impl already covers that type, so stable Rust has no disjoint "
            "specialisation. Add the concrete type to the blanket impl's "
            "exclusion list, split the trait, or accept autoderef-based "
            "dispatch instead",
        )

    emitted = []
    for item in normalised:
        if item["kind"] == "blanket":
            bounds = ", ".join(item["bounds"]) or "T"
            emitted.append(f"impl<T: {bounds}> {item['trait']} for {item['for_type']}")
        else:
            emitted.append(f"impl {item['trait']} for {item['for_type']}")

    return {
        "impls": tuple(normalised),
        "concrete": tuple(concrete),
        "blanket": tuple(blanket),
        "overlapping": (),
        "stable": stable,
        "requires_specialization": not stable,
        "emitted": tuple(emitted),
    }


# ---------------------------------------------------------------------------
# 2. rste_compose -- blanket adapter composition
# ---------------------------------------------------------------------------


def rste_compose(
    adapters: Sequence[Mapping[str, Any]],
    *,
    available: Sequence[str] = (),
    target: str,
) -> dict[str, Any]:
    """Find a chain of adapters whose ``requires`` are satisfied in order.

    Each adapter is a mapping with ``name``, ``requires`` and ``provides``
    (sequences of bound names). Composition walks the adapters repeatedly:
    an adapter can be applied once every bound in its ``requires`` is either
    already ``available`` or provided by an adapter applied earlier. The
    returned ``bounds`` string is the ``where`` clause the caller would write
    instead of repeating every intermediate bound.

    A cycle in the adapter graph raises ``E005``; an unreachable ``target``
    raises ``E002``. The chain is deterministic: adapters are considered in
    declaration order, and each adapter is applied at most once.
    """
    if not isinstance(adapters, Sequence) or isinstance(adapters, (str, bytes)):
        _fail("E003", "adapters must be a sequence of mappings")
    known = _str_list(available, "available") if available else ()
    goal = _name(target, "target")

    parsed: list[dict[str, Any]] = []
    for raw in adapters:
        if not isinstance(raw, Mapping):
            _fail("E003", "each adapter must be a mapping")
        name = _name(raw.get("name"), "adapter name")
        req = _str_list(raw.get("requires", ()), f"requires of {name}")
        prov = _str_list(raw.get("provides"), f"provides of {name}")
        if not prov:
            _fail("E002", f"adapter `{name}` provides nothing, so it cannot help")
        parsed.append({"name": name, "requires": req, "provides": prov})

    names = [a["name"] for a in parsed]
    if len(set(names)) != len(names):
        _fail("E003", f"adapter names must be unique, got {names}")

    # Cycle detection over the provides -> requires relation.
    graph: dict[str, set[str]] = {a["name"]: set() for a in parsed}
    for a in parsed:
        for b in parsed:
            if a["name"] != b["name"] and set(a["provides"]) & set(b["requires"]):
                graph[a["name"]].add(b["name"])
    state: dict[str, int] = {}

    def _walk(node: str, trail: tuple[str, ...]) -> None:
        if state.get(node) == 1:
            cycle = " -> ".join((*trail, node))
            _fail("E005", f"adapter composition cycle: {cycle}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for nxt in sorted(graph[node]):
            _walk(nxt, (*trail, node))
        state[node] = 2

    for a in parsed:
        _walk(a["name"], ())

    satisfied = set(known)
    applied: list[str] = []
    progress = True
    while progress:
        progress = False
        for a in parsed:
            if a["name"] in applied:
                continue
            if set(a["requires"]) <= satisfied:
                satisfied |= set(a["provides"])
                applied.append(a["name"])
                progress = True

    if goal not in satisfied:
        missing = [
            m
            for m in sorted({r for a in parsed for r in a["requires"]})
            if m not in satisfied
        ]
        hint = f"missing bound(s): {', '.join(missing)}" if missing else "no chain"
        _fail(
            "E002",
            f"no composition path provides `{goal}` from "
            f"{', '.join(sorted(known)) or 'no bounds'}: {hint}",
        )

    if not applied:
        _fail("E002", f"`{goal}` is already available; nothing to compose")

    bounds = ", ".join(sorted(satisfied - set(known)))
    return {
        "path": tuple(applied),
        "requires": tuple(
            sorted({r for a in parsed if a["name"] in applied for r in a["requires"]})
        ),
        "provides": tuple(sorted(satisfied)),
        "bounds": bounds,
        "emitted": tuple(
            f"fn run<T>(value: T) -> _ where T: {bounds}"
            if bounds
            else "fn run<T>(value: T) -> _"
        ),
    }


# ---------------------------------------------------------------------------
# 3. rste_fast_path -- disjoint dispatch with an honest specialisation flag
# ---------------------------------------------------------------------------


def rste_fast_path(
    cases: Sequence[Mapping[str, Any]],
    *,
    fallback: str | None = None,
    exhaustive: bool = False,
) -> dict[str, Any]:
    """Order specialised handlers before a fallback and prove disjointness.

    A case is ``{"category": <one of copy|primitive|slice|concrete>,
    "handler": <name>, "covers": [<type names>]}``. ``covers`` is explicit
    rather than inferred, because guessing coverage is exactly how a
    dispatcher ends up with two handlers that both match:

    * two fast cases whose ``covers`` intersect raise ``E002``;
    * a non-scalar ``copy`` case (a generic such as ``T: Copy``) makes the
      plan require nightly ``specialization`` and is reported through
      ``requires_specialization``;
    * a ``primitive`` or ``slice`` case covering more than one type is the
      same situation: one arm, many types, so a concrete arm below it cannot
      be proven disjoint;
    * ``exhaustive=True`` with no ``fallback`` raises ``E002``.

    The returned ``arms`` are the dispatch arms in evaluation order
    (fast paths first, fallback last), and ``stable`` says whether the plan
    is reachable on stable Rust. Nothing here claims the generated code is
    faster; it claims only that the arms do not overlap and that the order is
    deterministic.
    """
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        _fail("E003", "cases must be a sequence of mappings")
    if not cases:
        _fail("E003", "cases must not be empty")
    if fallback is not None:
        fallback = _name(fallback, "fallback")

    parsed: list[dict[str, Any]] = []
    for raw in cases:
        if not isinstance(raw, Mapping):
            _fail("E003", "each case must be a mapping")
        category = raw.get("category", "concrete")
        if category not in _CATEGORIES:
            _fail(
                "E003",
                f"category must be one of {', '.join(_CATEGORIES)}, got {category!r}",
            )
        handler = _name(raw.get("handler"), "handler")
        covers = _str_list(raw.get("covers"), f"covers of {handler}")
        if not covers:
            _fail("E002", f"case `{handler}` covers nothing, so it can never fire")
        parsed.append({"category": category, "handler": handler, "covers": covers})

    handlers = [c["handler"] for c in parsed]
    if len(set(handlers)) != len(handlers):
        _fail("E003", f"handler names must be unique, got {handlers}")

    overlap: list[tuple[str, str, str]] = []
    for i, a in enumerate(parsed):
        for b in parsed[i + 1 :]:
            shared = tuple(sorted(set(a["covers"]) & set(b["covers"])))
            if shared:
                overlap.append((a["handler"], b["handler"], ", ".join(shared)))
    if overlap:
        first = overlap[0]
        _fail(
            "E002",
            f"dispatch arms `{first[0]}` and `{first[1]}` both match "
            f"{first[2]}; narrow `covers` or drop one arm",
        )

    if exhaustive and fallback is None:
        _fail(
            "E002",
            "an exhaustive dispatch needs a fallback arm; `covers` cannot prove "
            "coverage of every type",
        )

    def _needs_specialization(case: dict[str, Any]) -> bool:
        # One arm matching many types cannot be proven disjoint from a concrete
        # arm below it, which is exactly what stable Rust cannot express.
        if len(case["covers"]) > 1:
            return True
        return case["category"] == "copy" and case["covers"][0] not in _SCALARS

    requires_specialization = any(_needs_specialization(c) for c in parsed)

    arms = [f"{' | '.join(c['covers'])} => {c['handler']}" for c in parsed]
    if fallback is not None:
        arms.append(f"_ => {fallback}")

    return {
        "cases": tuple(parsed),
        "arms": tuple(arms),
        "order": tuple(c["handler"] for c in parsed)
        + (() if fallback is None else (fallback,)),
        "overlap": (),
        "requires_specialization": requires_specialization,
        "stable": not requires_specialization,
        "fallback": fallback,
    }


# ---------------------------------------------------------------------------
# 4. rste_iter_adapter / rste_try_adapter -- iterator and GAT shapes
# ---------------------------------------------------------------------------


def _assoc_shape(
    *, item: str, borrowed: bool, fallible: bool, error: str | None
) -> dict[str, Any]:
    if borrowed:
        item_ty = f"&'a {item}"
    else:
        item_ty = item
    if fallible:
        if error is None:
            _fail("E003", "a fallible adapter needs an error type")
        item_ty = f"Result<{item_ty}, {error}>"
    return {
        "item": item_ty,
        "lifetime": "'a" if borrowed else None,
        "gat": borrowed,
    }


def rste_iter_adapter(
    item_type: str,
    *,
    borrowed: bool = False,
    fallible: bool = False,
    error_type: str | None = None,
    name: str = "Adapter",
) -> dict[str, Any]:
    """Emit the associated-type shape of an ``Iterator`` adapter.

    The point is the shape a caller would otherwise hand-write: which
    associated types exist, whether a lifetime binder is needed, and where the
    GAT form applies. ``borrowed=True`` produces ``type Item = &'a T`` (an
    implementation with ``impl<'a> Iterator for Adapter<'a>``),
    ``fallible=True`` wraps the item in ``Result``, and a borrowed *fallible*
    adapter produces ``&'a T`` inside the ``Result``, which is the case that
    usually forces a GAT in a trait definition.
    """
    item = _name(item_type, "item_type")
    adapter = _name(name, "name")
    shape = _assoc_shape(
        item=item,
        borrowed=_flag(borrowed, "borrowed"),
        fallible=_flag(fallible, "fallible"),
        error=_name(error_type, "error_type") if error_type is not None else None,
    )
    lifetime = "<'a>" if shape["lifetime"] else ""
    emitted = [
        f"impl{lifetime} Iterator for {adapter}{lifetime} {{",
        f"    type Item = {shape['item']};",
        "}",
    ]
    return {
        "name": adapter,
        "item": shape["item"],
        "lifetime": shape["lifetime"],
        "gat": shape["gat"],
        "bounds": "Iterator" if shape["lifetime"] is None else "Iterator + 'a",
        "emitted": tuple(emitted),
    }


def rste_try_adapter(
    ok_type: str,
    error_type: str,
    *,
    borrowed: bool = False,
    name: str = "TryAdapter",
) -> dict[str, Any]:
    """Emit the ``Try``/``try_fold`` shape: item, residual, and short-circuit.

    Rust's ``Try`` adapters carry a residual type, so the emitted shape names
    both ``type Item = Result<T, E>`` and the ``try_fold`` contract: a
    short-circuit returns ``ControlFlow::Break`` with the first error, and the
    adapter must not swallow later errors. The returned ``short_circuit``
    field says which branch wins, so a caller does not have to guess.
    """
    ok = _name(ok_type, "ok_type")
    err = _name(error_type, "error_type")
    adapter = _name(name, "name")
    shape = _assoc_shape(
        item=ok,
        borrowed=_flag(borrowed, "borrowed"),
        fallible=True,
        error=err,
    )
    lifetime = "<'a>" if shape["lifetime"] else ""
    emitted = [
        f"impl{lifetime} Iterator for {adapter}{lifetime} {{",
        f"    type Item = {shape['item']};",
        "}",
        "",
        "// try_fold: first Err wins; later items are not evaluated",
    ]
    return {
        "name": adapter,
        "item": shape["item"],
        "residual": err,
        "lifetime": shape["lifetime"],
        "short_circuit": "first-error",
        "emitted": tuple(emitted),
    }
