"""Async, error, and PyO3 boundary helpers for the Rust backend.

Three boundary problems are modelled here rather than left to ad-hoc
``map_err`` chains and hand-written ``unsafe`` blocks:

* **structured error context** -- a chain that keeps every ``caused by:``
  layer, plus a non-short-circuiting aggregator that reports every failed
  validation instead of only the first (``rsai_error_chain``,
  ``rsai_aggregate``). Both map onto the framework's ``E001``-``E008`` codes,
  so the text a caller sees matches the rest of the toolchain;
* **cancellation-safe async** -- a retry schedule with checkpoints, a deadline
  allocator that never over-commits a budget, and a commit/rollback plan for a
  multi-step task, so a drop mid-way cannot silently lose work
  (``rsai_retry_plan``, ``rsai_deadline_plan``, ``rsai_cancel_scope``);
* **PyO3 conversion and GIL discipline** -- typed ``Result``/enum/slice
  mappings, and a plan that refuses a Python-object operation inside a
  ``allow_threads`` region (``rsai_convert_result``, ``rsai_convert_enum``,
  ``rsai_gil_plan``), which is the boundary PEP 667 makes visible for dict
  views.

Codes: ``E001`` for an invalid name or budget, ``E002`` for an unsatisfiable
obligation (infeasible deadline, conflicting GIL region), ``E003`` for API
misuse, and the aggregated code is the lowest one among the collected errors.
This module does not run futures and does not touch a real GIL: the schedules
are deterministic plans, and the limits are in SPEC 10.56.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError
from runtime.diagnostics import ERROR_CODES

__all__ = [
    "rsai_aggregate",
    "rsai_cancel_scope",
    "rsai_convert_enum",
    "rsai_convert_result",
    "rsai_deadline_plan",
    "rsai_error_chain",
    "rsai_gil_plan",
    "rsai_retry_plan",
]


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured async/PyO3 boundary diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-async-interop")


def _code(value: object, what: str) -> str:
    if not isinstance(value, str):
        _fail("E003", f"{what} must be a str, got {type(value).__name__}")
    if value not in ERROR_CODES:
        _fail(
            "E003",
            f"{what} must be one of {', '.join(sorted(ERROR_CODES))}, got {value!r}",
        )
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("E003", f"{what} must be a non-empty str, got {type(value).__name__}")
    return value


def _int(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{what} must be an int, got {type(value).__name__}")
    if value < minimum:
        _fail("E001", f"{what} must be >= {minimum}, got {value}")
    return value


def _ident(value: object, what: str) -> str:
    name = _text(value, what)
    if not name.isidentifier():
        _fail("E001", f"{what} is not a valid Rust identifier: {name!r}")
    return name


# ---------------------------------------------------------------------------
# 1. rsai_error_chain -- context that is not thrown away
# ---------------------------------------------------------------------------


def rsai_error_chain(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a ``caused by:`` chain from outermost step to root cause.

    Each step is ``{"operation": <name>, "code": <E00n>, "message": <str>}``.
    The first step is the outermost failure, the last is the root cause, and
    every layer is kept in the report: that is what a bare ``map_err`` chain
    throws away. The headline code is the outermost step's code, because the
    caller's own operation is what failed first in their frame; ``root_code``
    records the origin.

    A single step is allowed. An unknown code raises ``E003`` rather than
    inventing one, and an empty sequence raises ``E003``.
    """
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        _fail("E003", "steps must be a sequence of mappings")
    if not steps:
        _fail("E003", "steps must not be empty")

    parsed: list[dict[str, str]] = []
    for index, raw in enumerate(steps):
        if not isinstance(raw, Mapping):
            _fail("E003", f"step {index} must be a mapping")
        parsed.append(
            {
                "operation": _ident(raw.get("operation"), f"step {index} operation"),
                "code": _code(raw.get("code"), f"step {index} code"),
                "message": _text(raw.get("message"), f"step {index} message"),
            }
        )

    head = parsed[0]
    lines = [
        f"error[{head['code']}]: {head['message']}",
        f"  in: {head['operation']}",
    ]
    for step in parsed[1:]:
        lines.append(
            f"  caused by [{step['code']}] {step['operation']}: {step['message']}"
        )
    return {
        "code": head["code"],
        "root_code": parsed[-1]["code"],
        "depth": len(parsed),
        "steps": tuple(parsed),
        "report": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# 2. rsai_aggregate -- every failure, not the first one
# ---------------------------------------------------------------------------


def rsai_aggregate(
    errors: Sequence[Mapping[str, Any]],
    *,
    limit: int = 0,
) -> dict[str, Any]:
    """Collect validation failures without short-circuiting.

    Each error is ``{"code": <E00n>, "message": <str>, "path": <str>}``;
    ``path`` is optional and identifies which item failed. Duplicates
    (same code, message, and path) are collapsed, the rest are sorted
    deterministically by code then path then message, and the headline code
    is the lowest one so a caller who prints only the code still sees the most
    structural problem.

    ``limit`` keeps the first N entries in the report but never changes
    ``count``, so a UI can truncate while a machine-readable field stays
    complete. An empty sequence is not an error: it returns ``code=None``.
    """
    if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)):
        _fail("E003", "errors must be a sequence of mappings")
    cap = _int(limit, "limit", minimum=0) if limit else 0

    seen: dict[tuple[str, str, str], dict[str, str]] = {}
    for index, raw in enumerate(errors):
        if not isinstance(raw, Mapping):
            _fail("E003", f"error {index} must be a mapping")
        code = _code(raw.get("code"), f"error {index} code")
        message = _text(raw.get("message"), f"error {index} message")
        path = raw.get("path", "")
        if not isinstance(path, str):
            _fail("E003", f"error {index} path must be a str")
        seen.setdefault(
            (code, path, message), {"code": code, "message": message, "path": path}
        )

    ordered = sorted(seen.values(), key=lambda e: (e["code"], e["path"], e["message"]))
    shown = ordered[:cap] if cap else ordered
    report_lines = [
        f"error[{e['code']}] {e['path']}: {e['message']}"
        if e["path"]
        else f"error[{e['code']}] {e['message']}"
        for e in shown
    ]
    truncated = len(shown) < len(ordered)
    if truncated:
        report_lines.append(f"... {len(ordered) - len(shown)} more")
    return {
        "code": ordered[0]["code"] if ordered else None,
        "count": len(ordered),
        "shown": len(shown),
        "truncated": truncated,
        "errors": tuple(shown),
        "report": "\n".join(report_lines),
    }


# ---------------------------------------------------------------------------
# 3. rsai_retry_plan -- a schedule with cancellation checkpoints
# ---------------------------------------------------------------------------


def rsai_retry_plan(
    attempts: int,
    *,
    base_ms: int = 50,
    cap_ms: int = 1_000,
    factor: int = 2,
) -> dict[str, Any]:
    """Plan exponential backoff with an explicit cancellation checkpoint.

    The schedule has ``attempts - 1`` waits, because a wait after the final
    attempt is wasted work. Each wait is ``min(base_ms * factor**n, cap_ms)``,
    and ``checkpoints`` names where a cancellation is observed: before each
    attempt and during each wait. Nothing is "in flight" during a wait, which
    is why a cancellation there cannot lose a completed attempt; a
    cancellation during an attempt is only safe if that attempt is
    idempotent, and ``idempotent_required`` says so instead of pretending.

    ``attempts >= 1``, ``base_ms >= 1``, ``cap_ms >= base_ms``, and
    ``factor >= 1`` are required; anything else raises ``E001``/``E003``.
    """
    n = _int(attempts, "attempts", minimum=1)
    base = _int(base_ms, "base_ms", minimum=1)
    cap = _int(cap_ms, "cap_ms", minimum=1)
    if cap < base:
        _fail("E001", f"cap_ms {cap} is below base_ms {base}, so the schedule shrinks")
    growth = _int(factor, "factor", minimum=1)

    waits: list[int] = []
    for n_index in range(n - 1):
        wait = base * (growth**n_index)
        waits.append(min(wait, cap))

    checkpoints = tuple(
        [f"before-attempt-{i + 1}" for i in range(n)]
        + [f"during-wait-{i + 1}" for i in range(len(waits))]
    )
    return {
        "attempts": n,
        "waits_ms": tuple(waits),
        "total_ms": sum(waits),
        "checkpoints": checkpoints,
        "idempotent_required": True,
        "note": (
            "a cancellation during a wait loses no work; a cancellation during "
            "an attempt is safe only when that attempt is idempotent"
        ),
    }


def rsai_deadline_plan(
    budget_ms: int, steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Split one time budget across steps without over-committing it.

    Each step is ``{"name": <ident>, "min_ms": <int>, "share": <int>}``; ``share``
    is a relative weight and defaults to 1. Every step gets at least
    ``min_ms``; the remainder is distributed by weight, and the leftover of an
    integer division goes to the earliest step so the sum is exactly the
    budget. If the minimums already exceed the budget the plan raises ``E002``
    instead of silently shaving a step that cannot run in less.

    The returned ``slack_ms`` is what is left after the minimums, which is the
    number a caller should compare against jitter rather than trusting a
    wall-clock estimate.
    """
    total = _int(budget_ms, "budget_ms", minimum=1)
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        _fail("E003", "steps must be a sequence of mappings")
    if not steps:
        _fail("E003", "steps must not be empty")

    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(steps):
        if not isinstance(raw, Mapping):
            _fail("E003", f"step {index} must be a mapping")
        parsed.append(
            {
                "name": _ident(raw.get("name"), f"step {index} name"),
                "min_ms": _int(raw.get("min_ms", 0), f"step {index} min_ms"),
                "share": _int(raw.get("share", 1), f"step {index} share", minimum=1),
            }
        )

    minimums = sum(s["min_ms"] for s in parsed)
    if minimums > total:
        _fail(
            "E002",
            f"the minimum time for {len(parsed)} step(s) is {minimums} ms, "
            f"above the budget of {total} ms; raise the budget or drop a step",
        )

    slack = total - minimums
    weight = sum(s["share"] for s in parsed)
    allocations = [s["min_ms"] + slack * s["share"] // weight for s in parsed]
    remainder = total - sum(allocations)
    for offset in range(remainder):
        allocations[offset % len(allocations)] += 1

    return {
        "budget_ms": total,
        "minimum_ms": minimums,
        "slack_ms": slack,
        "allocations": tuple(
            {"name": s["name"], "ms": ms} for s, ms in zip(parsed, allocations)
        ),
        "note": "allocations sum to the budget exactly; slack is not a speed claim",
    }


def rsai_cancel_scope(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Plan commit points so a cancelled task cannot leave partial state.

    A step is ``{"name": <ident>, "commit": <ident>}``: the work is done under
    ``name`` and only becomes visible at ``commit``. The plan runs commits in
    order and rollbacks in reverse, and it names the only safe cancellation
    window: after a commit and before the next step's commit, because every
    other point has either nothing committed or a pending mutation.

    A step without a ``commit`` is rejected with ``E002``: a step that mutates
    and cannot name its commit point is exactly the one that loses work on
    drop.
    """
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        _fail("E003", "steps must be a sequence of mappings")
    if not steps:
        _fail("E003", "steps must not be empty")

    parsed: list[dict[str, str]] = []
    for index, raw in enumerate(steps):
        if not isinstance(raw, Mapping):
            _fail("E003", f"step {index} must be a mapping")
        commit = raw.get("commit")
        if commit is None:
            _fail(
                "E002",
                f"step {index} names no commit point; a step that mutates without "
                "a commit point loses work when the future is dropped",
            )
        parsed.append(
            {
                "name": _ident(raw.get("name"), f"step {index} name"),
                "commit": _ident(commit, f"step {index} commit"),
            }
        )

    names = [s["name"] for s in parsed]
    if len(set(names)) != len(names):
        _fail("E003", f"step names must be unique, got {names}")

    return {
        "steps": tuple(parsed),
        "commit_order": tuple(s["commit"] for s in parsed),
        "rollback_order": tuple(s["commit"] for s in reversed(parsed)),
        "safe_cancel_windows": tuple(
            f"after {parsed[i]['commit']} before {parsed[i + 1]['name']}"
            for i in range(len(parsed) - 1)
        )
        + (f"after {parsed[-1]['commit']}",),
    }


# ---------------------------------------------------------------------------
# 4. rsai_gil_plan -- allow_threads discipline
# ---------------------------------------------------------------------------


def rsai_gil_plan(released: Sequence[str], held: Sequence[str]) -> dict[str, Any]:
    """Check that no Python-object operation sits inside a released region.

    ``released`` and ``held`` are operation names. Anything that touches a
    ``PyObject`` -- a dict mutation, an attribute read, a call into Python --
    must be in ``held``, because ``allow_threads`` gives up the GIL and PEP 667
    makes the resulting race observable (a dict view can change under you).

    An operation that appears in both lists raises ``E002``, and so does a
    ``released`` operation whose name is in ``_PY_OBJECT_OPS``: that is the
    case the plan exists to catch, not to document.
    """
    for value, what in ((released, "released"), (held, "held")):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            _fail("E003", f"{what} must be a sequence of str")

    released_ops = [_ident(op, "released op") for op in released]
    held_ops = [_ident(op, "held op") for op in held]
    both = sorted(set(released_ops) & set(held_ops))
    if both:
        _fail(
            "E002",
            f"operation(s) {both} are declared both released and held; "
            "an operation is either inside allow_threads or it is not",
        )

    forbidden = sorted(set(released_ops) & _PY_OBJECT_OPS)
    if forbidden:
        _fail(
            "E002",
            f"{forbidden} touch a Python object while the GIL is released; "
            "move them into the held region or drop the allow_threads block",
        )

    return {
        "released": tuple(released_ops),
        "held": tuple(held_ops),
        "forbidden": (),
        "requires_reacquire": tuple(sorted(set(held_ops) & _PY_OBJECT_OPS)),
        "note": (
            "compile the released region, reacquire the GIL, then touch Python: "
            "PEP 667 makes a released-region dict mutation observable"
        ),
    }


_PY_OBJECT_OPS = frozenset(
    {
        "dict_set",
        "dict_view",
        "attr_get",
        "attr_set",
        "call_python",
        "list_append",
        "import_module",
        "py_iter",
    }
)


# ---------------------------------------------------------------------------
# 5. rsai_convert_result / rsai_convert_enum -- typed PyO3 mappings
# ---------------------------------------------------------------------------


_RESULT_SHAPES: dict[tuple[str, str], dict[str, str]] = {
    ("String", "PillarError"): {"ok_py": "str", "err_py": "raise PillarError"},
    ("i64", "PillarError"): {"ok_py": "int", "err_py": "raise PillarError"},
    ("bool", "PillarError"): {"ok_py": "bool", "err_py": "raise PillarError"},
    ("Vec<String>", "PillarError"): {
        "ok_py": "list[str]",
        "err_py": "raise PillarError",
    },
}


def rsai_convert_result(ok_type: str, err_type: str) -> dict[str, Any]:
    """Map a Rust ``Result<T, E>`` onto a PyO3 return shape.

    The mapping is explicit rather than inferred: a ``Result`` becomes either
    a value or a raised exception, and the pair must be one this boundary
    knows. An unknown pair raises ``E003`` with the supported pairs listed,
    which is the difference between a contract and a guess.

    ``?`` inside the boundary is not modelled: the emitted advice is to convert
    at the outermost call, because a ``?`` that crosses ``allow_threads``
    needs the GIL back before raising.
    """
    ok = _text(ok_type, "ok_type")
    err = _text(err_type, "err_type")
    shape = _RESULT_SHAPES.get((ok, err))
    if shape is None:
        supported = ", ".join(f"Result<{o}, {e}>" for o, e in sorted(_RESULT_SHAPES))
        _fail(
            "E003",
            f"no boundary mapping for Result<{ok}, {err}>; supported: {supported}",
        )
    return {
        "ok_type": ok,
        "err_type": err,
        "ok_py": shape["ok_py"],
        "err_py": shape["err_py"],
        "emitted": f"-> PyResult<{shape['ok_py']}> // Err({err}) -> Err(PyErr)",
        "note": "convert at the outermost call, after the GIL is back",
    }


def rsai_convert_enum(
    name: str,
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map a Rust enum onto a Python-visible enum: names, payloads, and arms.

    Each variant is ``{"name": <ident>, "payload": <type or null>}``. A
    payload variant becomes a class with that attribute; a unit variant
    becomes a constant. Duplicate names raise ``E003``, an empty variant list
    raises ``E003``, and a payload type that is not a known boundary type
    raises ``E003`` rather than producing a broken class.

    No conversion is invented for a variant that carries a reference: a
    ``&T`` payload cannot outlive the call, so the plan refuses it with
    ``E002`` and points at cloning or taking ownership instead.
    """
    enum_name = _ident(name, "enum name")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        _fail("E003", "variants must be a sequence of mappings")
    if not variants:
        _fail("E003", "variants must not be empty")

    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(variants):
        if not isinstance(raw, Mapping):
            _fail("E003", f"variant {index} must be a mapping")
        variant = _ident(raw.get("name"), f"variant {index} name")
        payload = raw.get("payload")
        if payload is not None:
            payload = _text(payload, f"variant {index} payload")
            if payload.startswith("&"):
                _fail(
                    "E002",
                    f"variant `{variant}` carries `{payload}`, which cannot outlive "
                    "the call; clone it or store an owned value",
                )
            if payload not in _BOUNDARY_TYPES:
                _fail(
                    "E003",
                    f"variant `{variant}` payload `{payload}` is not a boundary type; "
                    f"supported: {', '.join(sorted(_BOUNDARY_TYPES))}",
                )
        parsed.append({"name": variant, "payload": payload})

    names = [v["name"] for v in parsed]
    if len(set(names)) != len(names):
        _fail("E003", f"variant names must be unique, got {names}")

    arms = []
    for variant in parsed:
        if variant["payload"] is None:
            arms.append(f"{enum_name}::{variant['name']} => Py::new(py, Unit)?")
        else:
            arms.append(f"{enum_name}::{variant['name']}(v) => Py::new(py, With(v))?")
    return {
        "name": enum_name,
        "variants": tuple(parsed),
        "unit_variants": tuple(v["name"] for v in parsed if v["payload"] is None),
        "payload_variants": tuple(
            v["name"] for v in parsed if v["payload"] is not None
        ),
        "emitted": tuple(arms),
    }


_BOUNDARY_TYPES = frozenset(
    {"i64", "f64", "bool", "String", "Vec<String>", "Vec<i64>", "PillarError"}
)
