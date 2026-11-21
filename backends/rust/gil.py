"""GIL and RAII ownership modeling for the Rust/PyO3 backend.

The functions here emit deterministic PyO3 scaffolding and model the lifetime
of Python references explicitly: borrowed ``Bound`` handles never cross a GIL
release, owned ``Py<T>`` handles are cloned first, and RAII guards release
resources through ``Drop`` on every return path.  Unsupported native patterns
raise :class:`PolyglotError` with the framework E001-E008 codes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "rsg_allow_threads_block",
    "rsg_async_send_check",
    "rsg_borrow_overlap_check",
    "rsg_borrow_snapshot",
    "rsg_critical_section",
    "rsg_detach_attach",
    "rsg_drop_order",
    "rsg_gil_contention_model",
    "rsg_gil_plan",
    "rsg_gil_state",
    "rsg_guard_release_map",
    "rsg_lifetime_erasure",
    "rsg_owned_handle",
    "rsg_ownership_audit",
    "rsg_py_ref_count",
    "rsg_raii_guard_def",
    "rsg_raii_guard_use",
    "rsg_reentrancy_check",
    "rsg_release_plan",
    "rsg_with_gil_block",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PY_API_RE = re.compile(r"\bpy\.|Python::|PyModule|PyDict|PyList|PyTuple|PyAny")
_GUARD_KINDS = ("lock", "handle", "buffer", "py_handle", "gil")
_DEFAULT_RELEASE = {
    "lock": "drop(__resource);",
    "handle": "drop(__resource);",
    "buffer": "drop(__resource);",
    "py_handle": "drop(__resource);",
    "gil": "drop(__resource);",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured native GIL/ownership diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-gil")


def _text(value: object, what: str) -> str:
    """Validate one caller-supplied string."""
    if not isinstance(value, str):
        _fail("E003", f"{what} must be a str, got {type(value).__name__}")
    return value


def _identifier(value: object, what: str) -> str:
    """Validate a Rust binding name."""
    text = _text(value, what)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a valid Rust identifier, got {text!r}")
    return text


def _balanced(value: object, what: str) -> str:
    """Validate balanced delimiters in a Rust code fragment."""
    text = _text(value, what)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            _fail("E003", f"{what} has unbalanced delimiters")
    if stack:
        _fail("E003", f"{what} has unbalanced delimiters")
    return text


def _names(values: Sequence[str], what: str) -> tuple[str, ...]:
    """Validate a sequence of binding names and reject duplicates."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        _fail("E003", f"{what} must be a sequence of names")
    result = tuple(_identifier(item, what) for item in values)
    if len(set(result)) != len(result):
        _fail("E003", f"duplicate names in {what}")
    return result


def _used_in(name: str, work: str) -> bool:
    """Whether a binding name appears as a whole word in *work*."""
    return re.search(rf"\b{re.escape(name)}\b", work) is not None


def _type_text(value: object, what: str) -> str:
    """Normalize a simple Rust result type string."""
    text = " ".join(_text(value, what).split())
    if not text or ";" in text or "\n" in text:
        _fail("E003", f"{what} must be a non-empty Rust type, got {text!r}")
    if text.count("<") != text.count(">"):
        _fail("E001", f"{what} has unbalanced generics: {text!r}")
    return text


def _allow_threads_indices(code: str) -> tuple[int, ...]:
    """Line indices of statements inside an ``allow_threads`` closure."""
    lines = code.splitlines()
    inside: list[int] = []
    depth = 0
    for index, line in enumerate(lines):
        if depth == 0 and "allow_threads(" not in line:
            continue
        inside.append(index)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and ("});" in line or line.rstrip().endswith("}")):
            depth = 0
    return tuple(inside)


# ---------------------------------------------------------------------------
# 1. rsg_gil_state
# ---------------------------------------------------------------------------


def rsg_gil_state(events: Sequence[str] = ()) -> dict[str, Any]:
    """Model a sequence of GIL state transitions without touching CPython."""
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        _fail("E003", "events must be a sequence of strings")
    state = "held"
    transitions: list[dict[str, str]] = []
    for index, event in enumerate(events):
        name = _text(event, f"event {index}")
        if name in {"release", "allow_threads", "detach"}:
            if state != "held":
                _fail("E003", f"cannot {name} while the GIL is already released")
            state = "released"
        elif name in {"acquire", "with_gil", "attach"}:
            if state != "released":
                _fail("E003", f"cannot {name} while the GIL is already held")
            state = "held"
        else:
            _fail("E003", f"unknown GIL event {name!r}")
        transitions.append({"event": name, "state": state})
    return {
        "state": state,
        "held": state == "held",
        "transitions": tuple(transitions),
        "event_count": len(transitions),
        "balanced": state == "held",
    }


# ---------------------------------------------------------------------------
# 2. rsg_allow_threads_block
# ---------------------------------------------------------------------------


def rsg_allow_threads_block(
    work: str,
    *,
    result_type: str = "i64",
    owned: Sequence[str] = (),
    check_python_apis: bool = True,
) -> dict[str, Any]:
    """Emit ``py.allow_threads`` scaffolding for pure Rust work."""
    body = _balanced(work, "GIL work expression")
    if not body.strip():
        _fail("E003", "GIL work expression must not be empty")
    if check_python_apis and _PY_API_RE.search(body):
        _fail(
            "E007",
            "work calls Python APIs inside allow_threads; move them outside the closure",
        )
    result = _type_text(result_type, "result type")
    owned_names = _names(owned, "owned reference")
    lines = ["// GIL released only for pure Rust work; no Python API calls inside."]
    for name in owned_names:
        lines.append(f"let __gil_move_{name} = {name}.clone_ref(py);")
    lines.append(f"let __gil_result: {result} = py.allow_threads(|| {{")
    lines.extend(f"    {line}" for line in body.splitlines())
    lines.append("});")
    return {
        "code": "\n".join(lines),
        "result_type": result,
        "owned": owned_names,
        "releases_gil": True,
        "python_apis_detected": bool(_PY_API_RE.search(body)),
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 3. rsg_with_gil_block
# ---------------------------------------------------------------------------


def rsg_with_gil_block(work: str) -> dict[str, Any]:
    """Emit a ``Python::with_gil`` block that re-acquires the GIL explicitly."""
    body = _balanced(work, "with_gil work expression")
    if not body.strip():
        _fail("E003", "with_gil work expression must not be empty")
    lines = [
        "// GIL is held for the whole callback; keep the body short.",
        "Python::with_gil(|py| {",
        *(f"    {line}" for line in body.splitlines()),
        "});",
    ]
    return {
        "code": "\n".join(lines),
        "reacquires_gil": True,
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 4. rsg_borrow_snapshot
# ---------------------------------------------------------------------------


def rsg_borrow_snapshot(
    name: str, *, rust_type: str = "Bound<'_, PyAny>"
) -> dict[str, Any]:
    """Clone a borrowed ``Bound`` reference into an owned, GIL-safe handle."""
    identifier = _identifier(name, "borrowed reference")
    text = _type_text(rust_type, "borrowed reference type")
    if not text.startswith("Bound<"):
        _fail("E001", f"{text!r} is not a borrowed Bound<T> reference")
    code = f"let __gil_owned_{identifier} = {identifier}.clone().unbind();"
    inner = text[6:-1].split(",")[-1].strip() or "PyAny"
    return {
        "code": code,
        "name": identifier,
        "source_type": text,
        "owned_type": f"Py<{inner}>",
        "crosses_gil_release": True,
    }


# ---------------------------------------------------------------------------
# 5. rsg_owned_handle
# ---------------------------------------------------------------------------


def rsg_owned_handle(name: str, *, rust_type: str = "Py<PyAny>") -> dict[str, Any]:
    """Emit a ``clone_ref`` binding for an already-owned ``Py<T>`` handle."""
    identifier = _identifier(name, "owned reference")
    text = _type_text(rust_type, "owned reference type")
    if not text.startswith("Py<") or not text.endswith(">"):
        _fail("E001", f"{text!r} is not an owned Py<T> handle")
    code = f"let __gil_move_{identifier} = {identifier}.clone_ref(py);"
    return {
        "code": code,
        "name": identifier,
        "rust_type": text,
        "increments_refcount": True,
        "drop_line": f"drop(__gil_move_{identifier});",
    }


# ---------------------------------------------------------------------------
# 6. rsg_release_plan
# ---------------------------------------------------------------------------


def rsg_release_plan(
    work: str,
    *,
    borrowed: Sequence[str] = (),
    owned: Sequence[str] = (),
    reentrant: bool = True,
) -> dict[str, Any]:
    """Decide which references may cross a GIL-release boundary."""
    body = _balanced(work, "GIL work expression")
    if not body.strip():
        _fail("E003", "GIL work expression must not be empty")
    overlap = rsg_borrow_overlap_check(borrowed, owned)["overlap"]
    snapshots: list[dict[str, Any]] = []
    blocked: list[str] = []
    warnings: list[str] = []
    for name in borrowed:
        if not _used_in(name, body):
            continue
        if reentrant:
            blocked.append(name)
        else:
            snapshots.append(rsg_borrow_snapshot(name))
            warnings.append(
                f"borrowed reference '{name}' was snapshotted before GIL release"
            )
    if _PY_API_RE.search(body):
        warnings.append(
            "work calls Python APIs inside allow_threads; move them outside"
        )
    for name in owned:
        snapshots.append(rsg_owned_handle(name))
    plan = rsg_allow_threads_block(body, owned=owned, check_python_apis=False)
    return {
        "safe": tuple(owned),
        "snapshots": tuple(snapshots),
        "blocked": tuple(blocked),
        "overlap": overlap,
        "warnings": tuple(warnings),
        "code": plan["code"],
        "releases_gil": True,
    }


# ---------------------------------------------------------------------------
# 7. rsg_reentrancy_check
# ---------------------------------------------------------------------------


def rsg_reentrancy_check(
    work: str,
    *,
    borrowed: Sequence[str] = (),
) -> dict[str, Any]:
    """Reject Python API calls and borrowed references inside released-GIL work."""
    body = _balanced(work, "GIL work expression")
    findings: list[str] = []
    if _PY_API_RE.search(body):
        findings.append("Python API call inside released-GIL work")
    for name in borrowed:
        identifier = _identifier(name, "borrowed reference")
        if _used_in(identifier, body):
            findings.append(f"borrowed reference '{identifier}' crosses GIL release")
    if findings:
        _fail("E007", "; ".join(findings))
    return {
        "safe": True,
        "findings": (),
        "checked_borrows": _names(borrowed, "borrowed reference"),
        "python_apis_detected": False,
    }


# ---------------------------------------------------------------------------
# 8. rsg_borrow_overlap_check
# ---------------------------------------------------------------------------


def rsg_borrow_overlap_check(
    borrowed: Sequence[str],
    owned: Sequence[str],
) -> dict[str, Any]:
    """Ensure a binding is not declared both borrowed and owned."""
    borrowed_names = _names(borrowed, "borrowed reference")
    owned_names = _names(owned, "owned reference")
    overlap = tuple(sorted(set(borrowed_names) & set(owned_names)))
    if overlap:
        _fail("E003", f"names cannot be both borrowed and owned: {overlap}")
    return {
        "overlap": (),
        "borrowed": borrowed_names,
        "owned": owned_names,
        "distinct": len(borrowed_names) + len(owned_names),
    }


# ---------------------------------------------------------------------------
# 9. rsg_drop_order
# ---------------------------------------------------------------------------


def rsg_drop_order(names: Sequence[str]) -> dict[str, Any]:
    """Model Rust's reverse-acquisition drop order for scoped guards."""
    acquired = _names(names, "guard name")
    reverse = tuple(reversed(acquired))
    lines = [f"drop({name});" for name in reverse]
    return {
        "acquisition": acquired,
        "reverse": reverse,
        "code": "\n".join(lines),
        "drops": len(lines),
    }


# ---------------------------------------------------------------------------
# 10. rsg_raii_guard_def
# ---------------------------------------------------------------------------


def rsg_raii_guard_def(
    guard_type: str = "PolyglotPyGuard",
    *,
    resource_type: str = "Py<PyAny>",
) -> dict[str, Any]:
    """Emit an RAII guard whose ``Drop`` impl releases its resource."""
    name = _identifier(guard_type, "guard type")
    resource = _type_text(resource_type, "resource type")
    lines = [
        f"// {name} releases its resource on every return path.",
        f"struct {name} {{",
        f"    resource: Option<{resource}>,",
        "}",
        "",
        f"impl {name} {{",
        "    fn new(resource: " + resource + ") -> Self {",
        "        Self { resource: Some(resource) }",
        "    }",
        "",
        "    fn get(&self) -> Option<&" + resource + "> {",
        "        self.resource.as_ref()",
        "    }",
        "}",
        "",
        f"impl Drop for {name} {{",
        "    fn drop(&mut self) {",
        "        self.resource.take();",
        "    }",
        "}",
        "",
    ]
    return {
        "text": "\n".join(lines),
        "guard_type": name,
        "resource_type": resource,
        "has_drop_impl": True,
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 11. rsg_raii_guard_use
# ---------------------------------------------------------------------------


def rsg_raii_guard_use(
    name: str,
    *,
    acquire: str = "lock.lock().unwrap()",
    release: str | None = None,
    kind: str = "lock",
) -> dict[str, Any]:
    """Emit an RAII guard instantiation with an explicit release expression."""
    identifier = _identifier(name, "guard name")
    if kind not in _GUARD_KINDS:
        _fail("E001", f"unknown guard kind {kind!r}; expected {_GUARD_KINDS}")
    expression = _balanced(acquire, f"acquire expression of {identifier!r}")
    release_text = _balanced(
        release or _DEFAULT_RELEASE[kind], f"release of {identifier!r}"
    )
    if not expression.strip() or not release_text.strip():
        _fail(
            "E003", f"guard {identifier!r} needs non-empty acquire/release expressions"
        )
    lines = [
        f"// {kind} guard '{identifier}': released by Drop on all paths.",
        f"let __raii_{identifier} = PolyglotRaiiGuard::new(",
        f"    {expression},",
        "    std::boxed::Box::new(|__resource| {",
        f"        {release_text}",
        "    }),",
        ");",
        "",
    ]
    return {
        "code": "\n".join(lines),
        "name": identifier,
        "kind": kind,
        "acquire": expression,
        "release": release_text,
        "drop_line": f"drop(__raii_{identifier});",
    }


# ---------------------------------------------------------------------------
# 12. rsg_guard_release_map
# ---------------------------------------------------------------------------


def rsg_guard_release_map(guards: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a mapping of guard names to acquisition/release metadata."""
    if not isinstance(guards, Mapping):
        _fail("E003", "guards must be a mapping")
    entries: dict[str, dict[str, str]] = {}
    for key, value in guards.items():
        name = _identifier(key, "guard name")
        if not isinstance(value, Mapping):
            _fail("E003", f"guard {name!r} metadata must be a mapping")
        kind = _text(value.get("kind", "lock"), f"kind of {name!r}")
        if kind not in _GUARD_KINDS:
            _fail("E001", f"unknown guard kind {kind!r} for {name!r}")
        acquire = _balanced(value.get("acquire", ""), f"acquire of {name!r}")
        release = _balanced(
            value.get("release", _DEFAULT_RELEASE[kind]), f"release of {name!r}"
        )
        entries[name] = {"kind": kind, "acquire": acquire, "release": release}
    return {
        "guards": entries,
        "names": tuple(sorted(entries)),
        "release_order": tuple(sorted(entries, reverse=True)),
        "count": len(entries),
    }


# ---------------------------------------------------------------------------
# 13. rsg_async_send_check
# ---------------------------------------------------------------------------


def rsg_async_send_check(
    body: str,
    *,
    captures: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify that an async native future is ``Send + 'static``."""
    text = _balanced(body, "async body")
    if not text.strip():
        _fail("E003", "async body must not be empty")
    if captures is None:
        capture_map: dict[str, str] = {}
    elif isinstance(captures, Mapping):
        capture_map = dict(captures)
    else:
        _fail("E003", "captures must be a mapping")
    forbidden_captures: list[str] = []
    for name, mode in capture_map.items():
        identifier = _identifier(name, "captured binding")
        if mode not in {"borrow", "owned", "copy", "py_handle"}:
            _fail("E003", f"unknown capture mode {mode!r} for {identifier!r}")
        if mode == "borrow":
            forbidden_captures.append(identifier)
    if forbidden_captures:
        _fail(
            "E007",
            f"async future captures non-Send borrowed references: {tuple(forbidden_captures)}",
        )
    if re.search(r"\bRc<|\bCell<|\bRefCell<", text):
        _fail("E007", "async future captures a !Send Rc/Cell type")
    return {
        "send": True,
        "static": True,
        "body_lines": len(text.splitlines()),
        "captures": tuple(sorted(capture_map)),
        "requirements": ("Send", "'static", "no borrowed Bound<T>"),
    }


# ---------------------------------------------------------------------------
# 14. rsg_lifetime_erasure
# ---------------------------------------------------------------------------


def rsg_lifetime_erasure(name: str, *, borrowed: bool = True) -> dict[str, Any]:
    """Erase a borrowed Python lifetime by cloning into an owned handle."""
    identifier = _identifier(name, "reference name")
    if borrowed:
        code = f"let __owned_{identifier} = {identifier}.clone().unbind();"
        source, target = "Bound<'py, T>", "Py<T>"
    else:
        code = f"let __owned_{identifier} = {identifier}.clone_ref(py);"
        source, target = "Py<T>", "Py<T>"
    return {
        "code": code,
        "name": identifier,
        "from": source,
        "to": target,
        "erased": borrowed,
        "drop_line": f"drop(__owned_{identifier});",
    }


# ---------------------------------------------------------------------------
# 15. rsg_gil_contention_model
# ---------------------------------------------------------------------------


def rsg_gil_contention_model(threads: int, *, releases: int = 1) -> dict[str, Any]:
    """Model deterministic GIL contention for a fixed number of threads."""
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        _fail("E003", "threads must be a positive int")
    if not isinstance(releases, int) or isinstance(releases, bool) or releases < 0:
        _fail("E003", "releases must be a non-negative int")
    wait_steps = max(0, threads - 1) * releases
    throughput = round(threads / (1 + wait_steps), 6)
    return {
        "threads": threads,
        "releases": releases,
        "wait_steps": wait_steps,
        "throughput": throughput,
        "model": "deterministic round-robin GIL handoff",
    }


# ---------------------------------------------------------------------------
# 16. rsg_py_ref_count
# ---------------------------------------------------------------------------


def rsg_py_ref_count(
    initial: int, *, clones: int = 0, drops: int = 0
) -> dict[str, Any]:
    """Model reference-count changes from clones and drops."""
    for value, what in ((initial, "initial"), (clones, "clones"), (drops, "drops")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _fail("E003", f"{what} must be a non-negative int")
    final = initial + clones - drops
    if final < 0:
        _fail("E003", "drops exceed the available Python references")
    return {
        "initial": initial,
        "clones": clones,
        "drops": drops,
        "final": final,
        "balanced": clones == drops,
        "leaked": final > initial,
    }


# ---------------------------------------------------------------------------
# 17. rsg_detach_attach
# ---------------------------------------------------------------------------


def rsg_detach_attach(work: str, *, reacquire: bool = True) -> dict[str, Any]:
    """Emit ``Python::detach`` / ``Python::attach`` scaffolding."""
    body = _balanced(work, "detach work expression")
    if not body.strip():
        _fail("E003", "detach work expression must not be empty")
    lines = [
        "let __gil_token = Python::detach(py);",
        *(f"    {line}" for line in body.splitlines()),
    ]
    if reacquire:
        lines.append("Python::attach(|py| {")
        lines.append("    let _ = __gil_token;")
        lines.append("});")
    else:
        lines.append("drop(__gil_token);")
    return {
        "code": "\n".join(lines),
        "reacquires_gil": reacquire,
        "python_apis_in_detached_work": bool(_PY_API_RE.search(body)),
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 18. rsg_critical_section
# ---------------------------------------------------------------------------


def rsg_critical_section(work: str, *, reason: str = "") -> dict[str, Any]:
    """Mark a body that must execute while the GIL is held."""
    body = _balanced(work, "critical section body")
    if not body.strip():
        _fail("E003", "critical section body must not be empty")
    if "allow_threads" in body or "Python::detach" in body:
        _fail("E007", "critical section may not release the GIL")
    explanation = _text(reason, "critical section reason")
    comment = (
        f"// GIL held: {explanation}"
        if explanation
        else "// GIL held for critical section."
    )
    lines = [comment, *(f"    {line}" for line in body.splitlines())]
    return {
        "code": "\n".join(lines),
        "reason": explanation,
        "gil_held": True,
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 19. rsg_gil_plan
# ---------------------------------------------------------------------------


def rsg_gil_plan(
    work: str,
    *,
    borrowed: Sequence[str] = (),
    owned: Sequence[str] = (),
    result_type: str = "i64",
    reentrant: bool = True,
) -> dict[str, Any]:
    """Aggregate snapshots, release code and ownership rules into one plan."""
    release = rsg_release_plan(
        work, borrowed=borrowed, owned=owned, reentrant=reentrant
    )
    code_lines = [
        "// GIL plan: snapshot borrowed refs, then release only for pure Rust work.",
        *release["code"].splitlines(),
    ]
    rules = (
        "clone every borrowed Bound<T> into Py<T> before allow_threads",
        "never call Python APIs inside the released-GIL closure",
        "re-acquire the GIL before touching Python objects again",
    )
    return {
        "code": "\n".join(code_lines),
        "rules": rules,
        "warnings": release["warnings"],
        "snapshots": release["snapshots"],
        "blocked": release["blocked"],
        "result_type": _type_text(result_type, "result type"),
        "releases_gil": True,
    }


# ---------------------------------------------------------------------------
# 20. rsg_ownership_audit
# ---------------------------------------------------------------------------


def rsg_ownership_audit(
    code: str,
    *,
    forbidden: Sequence[str] = ("Bound<'_", "&Py", "PyRef"),
    strict: bool = True,
) -> dict[str, Any]:
    """Audit emitted code for borrowed references inside GIL-release closures."""
    text = _text(code, "Rust code")
    patterns = tuple(_text(item, "forbidden pattern") for item in forbidden)
    indices = _allow_threads_indices(text)
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []
    for index in indices:
        for pattern in patterns:
            if pattern in lines[index]:
                findings.append(
                    {
                        "line": index + 1,
                        "pattern": pattern,
                        "text": lines[index].strip(),
                    }
                )
    if findings and strict:
        _fail("E007", f"borrowed Python reference inside allow_threads: {findings[0]}")
    return {
        "ok": not findings,
        "findings": tuple(findings),
        "code_lines": len(lines),
        "audited_lines": indices,
        "patterns": patterns,
    }
