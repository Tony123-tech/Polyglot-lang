"""Borrow, split, and arena ergonomics for the Rust backend.

Three pain points Rust programmers fight with borrowck are modelled here:

* **disjoint splitting** -- two mutable views of different fields of one
  struct, or of disjoint windows of one slice, without cloning the whole value
  and without holding a whole-struct borrow (``rsbe_split_fields``,
  ``rsbe_split_windows``);
* **generational handles** -- an arena whose keys carry an epoch, so a stale
  handle is rejected instead of aliasing a recycled slot, which is what
  ``Rc<RefCell<...>>`` graphs and parent pointers usually paper over
  (``rsbe_arena_plan``, ``rsbe_arena_simulate``);
* **scoped drop** -- cleanup that runs on every exit path, including the error
  path, in reverse acquisition order (``rsbe_scope_exit``).

``rsbe_arena_simulate`` is the part that is actually exercised rather than
described: it runs the arena protocol against the same slot/epoch bookkeeping
the emitted Rust would use, so a test can prove that a handle from before a
removal is refused after the slot is reused.

Codes follow the framework surface:

* ``E001`` -- invalid capacity, range, or field name;
* ``E002`` -- an obligation that cannot be satisfied (overlapping split, a
  whole-struct borrow that forbids the split, generation exhaustion);
* ``E003`` -- API misuse.

The model is not a borrow checker: it plans splits and validates the
disjointness argument a caller would have to make to ``unsafe`` code, and it
does not claim the generated code is faster. See SPEC 10.56.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "rsbe_arena_plan",
    "rsbe_arena_simulate",
    "rsbe_scope_exit",
    "rsbe_split_fields",
    "rsbe_split_windows",
]

_MAX_GENERATIONS = 32


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured borrow-ergonomics diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-borrow-ergo")


def _identifier(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("E003", f"{what} must be a non-empty str, got {type(value).__name__}")
    if not value.isidentifier():
        _fail("E001", f"{what} is not a valid Rust identifier: {value!r}")
    return value


def _count(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{what} must be an int, got {type(value).__name__}")
    if value < minimum:
        _fail("E001", f"{what} must be >= {minimum}, got {value}")
    return value


# ---------------------------------------------------------------------------
# 1. rsbe_split_fields -- disjoint mutable views of one struct
# ---------------------------------------------------------------------------


def rsbe_split_fields(
    fields: Sequence[str],
    *,
    borrowed_whole: Sequence[str] = (),
    struct_name: str = "self",
    mutable: bool = True,
) -> dict[str, Any]:
    """Plan a field-wise split of one struct into disjoint views.

    ``borrowed_whole`` lists names that already hold a whole-struct borrow
    (for example ``let all: &mut S = &mut s;``). A split is refused while one
    of them is live, because the whole-struct borrow covers every field:
    that is the exact case where a caller reaches for a clone or for
    ``Rc<RefCell<..>>`` and pays for it. ``mutable=False`` plans shared views
    instead, which are allowed to coexist.

    Returns one view per field, the emitted ``let`` bindings, and the
    disjointness proof in the form a comment would carry. Duplicate field
    names raise ``E003`` (two views of one field cannot both be mutable);
    naming ``self`` as a field raises ``E001``.
    """
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        _fail("E003", "fields must be a sequence of str")
    if not fields:
        _fail("E003", "fields must not be empty")

    names = [_identifier(f, "field name") for f in fields]
    if "self" in names:
        _fail("E001", "`self` is not a field; split a field instead of the whole value")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _fail(
            "E003",
            f"duplicate field(s) {duplicates}: two views of one field would alias",
        )

    if not isinstance(borrowed_whole, Sequence) or isinstance(
        borrowed_whole, (str, bytes)
    ):
        _fail("E003", "borrowed_whole must be a sequence of str")
    holders = [_identifier(h, "borrow holder") for h in borrowed_whole]
    if holders:
        _fail(
            "E002",
            f"cannot split `{struct_name}` while "
            f"{', '.join(holders)} holds a whole-struct borrow; "
            "release it first, or split before taking the whole borrow",
        )

    if not isinstance(mutable, bool):
        _fail("E003", "mutable must be a bool")

    reference = "&mut" if mutable else "&"
    views = tuple(
        {
            "field": name,
            "binding": f"{reference} {struct_name}.{name}",
            "mutable": mutable,
        }
        for name in names
    )
    emitted = tuple(f"let {name} = {reference} {struct_name}.{name};" for name in names)
    return {
        "views": views,
        "fields": tuple(names),
        "disjoint": True,
        "mutable": mutable,
        "proof": (
            "distinct fields of one struct do not overlap, so N mutable views "
            "of N distinct fields satisfy the aliasing rule"
            if mutable
            else "shared views never conflict with each other"
        ),
        "emitted": emitted,
    }


# ---------------------------------------------------------------------------
# 2. rsbe_split_windows -- disjoint slice windows
# ---------------------------------------------------------------------------


def rsbe_split_windows(
    length: int,
    windows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Plan disjoint ``&mut xs[a..b]`` windows over one sequence.

    ``windows`` is a sequence of ``(start, end)`` half-open ranges. They must
    be in bounds, non-empty, and non-overlapping; the returned plan lists the
    emitted bindings, how many elements are covered, and the gaps, which is
    how a caller proves the split to a reviewer. Overlap raises ``E002``
    naming both windows, an out-of-bounds range raises ``E001``, and a
    malformed range raises ``E003``.
    """
    total = _count(length, "length", minimum=0)
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        _fail("E003", "windows must be a sequence of (start, end)")
    if not windows:
        _fail("E003", "windows must not be empty")

    parsed: list[tuple[int, int]] = []
    for index, raw in enumerate(windows):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            _fail("E003", f"window {index} must be a (start, end) pair")
        start = _count(raw[0], f"window {index} start")
        end = _count(raw[1], f"window {index} end")
        if end <= start:
            _fail("E003", f"window {index} is empty: start {start} >= end {end}")
        if end > total:
            _fail(
                "E001",
                f"window {index} ends at {end}, past the length {total}",
            )
        parsed.append((start, end))

    ordered = sorted(parsed)
    for i in range(len(ordered) - 1):
        a_start, a_end = ordered[i]
        b_start, b_end = ordered[i + 1]
        if a_end > b_start:
            _fail(
                "E002",
                f"windows [{a_start}..{a_end}] and [{b_start}..{b_end}] overlap "
                f"on {min(a_end, b_end) - b_start} element(s); two mutable "
                "windows over one element would alias",
            )

    covered = sum(end - start for start, end in ordered)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ordered:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < total:
        gaps.append((cursor, total))

    names = [f"w{i}" for i in range(len(ordered))]
    emitted = tuple(
        f"let &mut {n} = &mut xs[{s}..{e}];" for n, (s, e) in zip(names, ordered)
    )
    return {
        "windows": tuple(
            {"name": n, "range": (s, e), "len": e - s}
            for n, (s, e) in zip(names, ordered)
        ),
        "covered": covered,
        "gaps": tuple(gaps),
        "exhaustive": covered == total,
        "emitted": emitted,
    }


# ---------------------------------------------------------------------------
# 3. rsbe_arena_plan -- generational index arena
# ---------------------------------------------------------------------------


def rsbe_arena_plan(
    capacity: int, *, generations: int = _MAX_GENERATIONS, item_type: str = "T"
) -> dict[str, Any]:
    """Describe the slot/key shapes of a generational arena.

    A key is ``ArenaKey { index: u32, epoch: u32 }``; a slot stores the value
    plus a generation counter that increments on every removal, and a slot
    whose generation reaches ``generations`` is burned. ``remove`` and ``get``
    compare the key's epoch against the slot's generation, so a handle from
    before a removal is refused even when the slot has been reused. That is
    the property ``rsbe_arena_simulate`` exercises.

    ``capacity`` must be positive and ``generations`` must be in
    ``1..=32``, because an epoch that wraps silently is worse than no epoch at
    all: the model refuses to plan a wrapping arena.
    """
    slots = _count(capacity, "capacity", minimum=1)
    gens = _count(generations, "generations", minimum=1)
    if gens > _MAX_GENERATIONS:
        _fail(
            "E001",
            f"generations must be <= {_MAX_GENERATIONS}, got {gens}: a wider "
            "epoch needs an explicit wrap policy, which this arena does not "
            "provide",
        )
    item = _identifier(item_type, "item_type")

    emitted = (
        "pub struct ArenaKey { pub index: u32, pub epoch: u32 }",
        f"struct Slot<{item}> {{ generation: u32, alive: bool, value: Option<{item}> }}",
        "// get/remove: key.epoch == slot.generation && slot.alive, else Stale",
    )
    return {
        "capacity": slots,
        "generations": gens,
        "key_bytes": 8,
        "slot_overhead_bytes": 8,
        "memory_note": (
            f"slots hold {item} plus an 8-byte generation/alive header; "
            "key is 8 bytes; nothing here is a measurement"
        ),
        "burned_after": gens - 1,
        "emitted": emitted,
        "operations": ("insert", "get", "remove", "contains", "clear"),
    }


def rsbe_arena_simulate(
    ops: Sequence[Mapping[str, Any]],
    *,
    capacity: int = 4,
    generations: int = _MAX_GENERATIONS,
) -> dict[str, Any]:
    """Run the arena protocol and report every stale-handle decision.

    Ops are ``{"op": "insert", "value": <anything>}``,
    ``{"op": "get", "key": {"index": i, "epoch": e}}``,
    ``{"op": "remove", "key": ...}``, and ``{"op": "clear"}``. Insert reuses
    the lowest free slot, increments nothing, and returns a key whose epoch is
    the slot's current generation. Remove marks the slot free and increments
    its generation, so an old key no longer matches.

    The trace records one entry per op with the outcome (``ok``, ``stale``,
    ``full``, ``burned``) so a test can assert exactly which handle was
    refused and why.
    """
    slots = _count(capacity, "capacity", minimum=1)
    gens = _count(generations, "generations", minimum=1)
    if gens > _MAX_GENERATIONS:
        _fail("E001", f"generations must be <= {_MAX_GENERATIONS}, got {gens}")
    if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes)):
        _fail("E003", "ops must be a sequence of mappings")

    state = [{"generation": 0, "alive": False, "value": None} for _ in range(slots)]
    trace: list[dict[str, Any]] = []
    live = 0
    stale_detected = False

    def _key(raw: object, what: str) -> tuple[int, int]:
        if not isinstance(raw, Mapping):
            _fail("E003", f"{what} must be a mapping with index and epoch")
        index = _count(raw.get("index"), f"{what} index")
        epoch = _count(raw.get("epoch"), f"{what} epoch")
        if index >= slots:
            _fail(
                "E001",
                f"{what} index {index} is outside capacity {slots}",
            )
        return index, epoch

    for position, raw in enumerate(ops):
        if not isinstance(raw, Mapping):
            _fail("E003", f"op {position} must be a mapping")
        kind = raw.get("op")
        if kind == "insert":
            target = None
            for index, slot in enumerate(state):
                if not slot["alive"] and slot["generation"] < gens:
                    target = index
                    break
            if target is None:
                burned = all(s["generation"] >= gens for s in state)
                outcome = "burned" if burned else "full"
                trace.append({"op": "insert", "outcome": outcome, "position": position})
                continue
            slot = state[target]
            slot["alive"] = True
            slot["value"] = raw.get("value")
            live += 1
            key = {"index": target, "epoch": slot["generation"]}
            trace.append(
                {"op": "insert", "outcome": "ok", "position": position, "key": key}
            )
        elif kind in {"get", "remove"}:
            key = _key(raw.get("key"), f"op {position} key")
            index, epoch = key
            slot = state[index]
            if not slot["alive"] or slot["generation"] != epoch:
                stale_detected = True
                trace.append(
                    {
                        "op": kind,
                        "outcome": "stale",
                        "position": position,
                        "key": {"index": index, "epoch": epoch},
                        "slot_generation": slot["generation"],
                        "alive": slot["alive"],
                    }
                )
                continue
            if kind == "get":
                trace.append(
                    {
                        "op": "get",
                        "outcome": "ok",
                        "position": position,
                        "value": slot["value"],
                    }
                )
            else:
                slot["alive"] = False
                slot["value"] = None
                slot["generation"] += 1
                live -= 1
                trace.append({"op": "remove", "outcome": "ok", "position": position})
        elif kind == "clear":
            for slot in state:
                if slot["alive"]:
                    slot["alive"] = False
                    slot["value"] = None
                    slot["generation"] += 1
            live = 0
            trace.append({"op": "clear", "outcome": "ok", "position": position})
        else:
            _fail("E003", f"op {position} has an unknown op {kind!r}")

    return {
        "trace": tuple(trace),
        "live": live,
        "capacity": slots,
        "generations": gens,
        "stale_detected": stale_detected,
    }


# ---------------------------------------------------------------------------
# 4. rsbe_scope_exit -- deterministic cleanup on every exit path
# ---------------------------------------------------------------------------


def rsbe_scope_exit(
    actions: Sequence[str],
    *,
    guard: str = "ScopeGuard",
    include_unwind: bool = True,
) -> dict[str, Any]:
    """Plan cleanup guards that run on the happy and error paths.

    Every action becomes a closure stored by a guard whose ``Drop`` impl runs
    them in reverse acquisition order, which is what makes the cleanup correct
    on an early ``return`` and on an unwind. ``include_unwind=False`` plans an
    explicit ``run()`` call instead, for code compiled with ``panic = "abort"``
    where ``Drop`` still runs but unwinding does not.

    Duplicate action names raise ``E003``; an empty list raises ``E003`` because
    a guard with nothing to do is dead weight.
    """
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        _fail("E003", "actions must be a sequence of str")
    if not actions:
        _fail("E003", "actions must not be empty")
    names = [_identifier(a, "action name") for a in actions]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _fail("E003", f"duplicate action(s) {duplicates}: cleanup names must be unique")
    guard_name = _identifier(guard, "guard")
    if not isinstance(include_unwind, bool):
        _fail("E003", "include_unwind must be a bool")

    reversed_names = tuple(reversed(names))
    runs_on = ("early-return", "end-of-scope", "error-path")
    if include_unwind:
        runs_on += ("unwind",)

    emitted = [
        f"struct {guard_name} {{ actions: Vec<Box<dyn FnOnce()>> }}",
        f"impl Drop for {guard_name} {{",
        "    fn drop(&mut self) {",
        "        while let Some(action) = self.actions.pop() { action(); }",
        "    }",
        "}",
        "",
        "// acquisition order: " + ", ".join(names),
        "// drop order:        " + ", ".join(reversed_names),
    ]
    return {
        "actions": tuple(names),
        "drop_order": reversed_names,
        "runs_on": runs_on,
        "emitted": tuple(emitted),
    }
