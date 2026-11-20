"""FRAME 10 documentation-sync helpers (functions 91-93).

These helpers treat the checked-in source and Markdown files as the two sides
of a contract.  They never rewrite documentation: each returns a deterministic
report so CI, a CLI, or a test can decide what to do with the drift.

The implementation modules are read as source with :mod:`ast` rather than
imported from the inspected root.  That keeps the report stable when the
inspected tree is a temporary copy and avoids executing documentation tooling
just to check a table.  The one runtime probe used below (``platform_support``)
is imported defensively and is reported as unavailable when absent.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import re
import sys
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

_STATUSES = ("foundation", "modelled", "rejected")

_CAPABILITY_ROW_RE = re.compile(
    r"^\|\s*(?P<number>\d+)\s*\|\s*`(?P<name>[A-Za-z0-9_]+)`\s*"
    r"\|\s*(?P<status>foundation|modelled|rejected)\s*\|"
    r"\s*(?P<boundary>.*?)\s*\|\s*$"
)
_CAPABILITY_COUNTS_RE = re.compile(
    r"classifies the (?P<total>\d+) functions as "
    r"(?P<foundation>\d+) foundation, "
    r"(?P<modelled>\d+) modelled, and "
    r"(?P<rejected>\d+) rejected"
)
_BACKEND_CHOICES_RE = re.compile(r"--backend(?:=|\s+)(?P<choices>[^\s`\]]+)")
_COMMAND_LINE_RE = re.compile(
    r"(?m)^\s*[>*#`\-]*\s*polyglot\s+(?P<command>[a-z][a-z0-9_-]*)"
)
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")


def _root_path(root: Path | str | None) -> Path:
    """Normalize an optional root argument to an absolute :class:`Path`."""
    return Path(root).resolve() if root is not None else _DEFAULT_ROOT


def _read_text(path: Path) -> str | None:
    """Return UTF-8 text for ``path``, or ``None`` when it is absent/unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError, UnicodeError:
        return None


def _relative(root: Path, path: Path) -> str:
    """Return a stable, POSIX-style path relative to ``root``."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _source_document(path: Path, text: str | None) -> dict[str, Any]:
    """Describe a source file that was read by a sync helper."""
    return {
        "path": path.as_posix(),
        "present": text is not None,
        "size": len(text) if text is not None else 0,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if text is not None
        else None,
    }


def _doc_paths(root: Path, *, include_project_docs: bool = True) -> list[Path]:
    """Return Markdown documentation in a deterministic order."""
    paths: list[Path] = sorted((root / "docs").glob("*.md"))
    if include_project_docs:
        for name in ("README.md", "GUIDE.md"):
            path = root / name
            if path.is_file():
                paths.append(path)
    return paths


def _drift(kind: str, *, path: str = "", **details: Any) -> dict[str, Any]:
    """Build one structured drift record."""
    return {"kind": kind, "path": path, **details}


def _as_text(node: ast.expr | None) -> str | None:
    """Return a string literal, ``None`` literal, or a source expression."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return None if node.value is None else str(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    return ast.unparse(node)


def _parse_capability_registry(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``runtime/capabilities.py`` into plain capability records.

    Returns ``(rows, source)``.  Malformed rows are returned with an ``error``
    field so the caller can report them instead of silently under-counting.
    """
    path = root / "runtime" / "capabilities.py"
    text = _read_text(path)
    source = _source_document(path, text)
    if text is None:
        return [], source
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        source["error"] = f"SyntaxError: {error.msg}"
        return [], source

    rows_node: ast.expr | None = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_ROWS":
                rows_node = node.value
                break
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_ROWS"
            for target in node.targets
        ):
            rows_node = node.value
            break
    if not isinstance(rows_node, (ast.Tuple, ast.List)):
        source["error"] = "no literal _ROWS assignment found"
        return [], source

    fields = (
        "name",
        "subsystem",
        "module",
        "attr",
        "status",
        "code",
        "summary",
        "boundary",
    )
    rows: list[dict[str, Any]] = []
    for number, element in enumerate(rows_node.elts, start=1):
        if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) != len(
            fields
        ):
            rows.append(
                {
                    "number": number,
                    "error": f"expected {len(fields)} fields, got {len(element.elts)}"
                    if isinstance(element, (ast.Tuple, ast.List))
                    else "row is not a literal tuple",
                }
            )
            continue
        values = dict(
            zip(fields, (_as_text(item) for item in element.elts), strict=True)
        )
        values["number"] = number
        status = values.get("status")
        if status not in _STATUSES:
            values["error"] = f"unknown status {status!r}"
        rows.append(values)
    return rows, source


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count capabilities by status, including a total."""
    counts: dict[str, int] = {status: 0 for status in _STATUSES}
    for row in rows:
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return {"total": len(rows), **counts}


def docs_sync_capability_matrix(*, root: Path | str | None = None) -> dict[str, Any]:
    """Report drift between the capability registry and the documented matrix.

    The registry in ``runtime/capabilities.py`` is the source of truth.  Every
    Markdown file under ``docs/`` is scanned for capability-table rows, status
    values, boundaries, and the summary count sentence.  Each disagreement is
    returned as a structured record; ``ok`` is true only when no drift exists.

    Parameters
    ----------
    root:
        Repository root to inspect.  Defaults to the directory containing
        ``docs/`` in this checkout.
    """
    base = _root_path(root)
    rows, source = _parse_capability_registry(base)
    expected = {row["name"]: row for row in rows if isinstance(row.get("name"), str)}
    counts = _status_counts(rows)
    drift: list[dict[str, Any]] = []

    if source.get("error") is not None:
        drift.append(_drift("registry-unreadable", detail=source["error"]))
    for row in rows:
        if row.get("error") is not None:
            drift.append(
                _drift(
                    "registry-row-invalid",
                    path=source["path"],
                    number=row["number"],
                    detail=row["error"],
                )
            )
    if not expected:
        drift.append(
            _drift(
                "registry-empty", path=source["path"], detail="no capability rows found"
            )
        )

    documented: list[dict[str, Any]] = []
    doc_paths = _doc_paths(base, include_project_docs=False)
    if not doc_paths:
        drift.append(
            _drift(
                "matrix-missing", path="docs/", detail="no Markdown files to inspect"
            )
        )
    for path in doc_paths:
        text = _read_text(path)
        if text is None:
            continue
        rel = _relative(base, path)
        seen: dict[str, dict[str, Any]] = {}
        doc_counts: dict[str, int] = {status: 0 for status in _STATUSES}
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = _CAPABILITY_ROW_RE.match(line)
            if match is None:
                continue
            number = int(match.group("number"))
            name = match.group("name")
            status = match.group("status")
            boundary = " ".join(match.group("boundary").split())
            doc_counts[status] += 1
            entry = {
                "number": number,
                "name": name,
                "status": status,
                "boundary": boundary,
                "line": line_number,
            }
            if name in seen:
                drift.append(
                    _drift(
                        "duplicate-row",
                        path=rel,
                        name=name,
                        detail=f"first at line {seen[name]['line']}",
                        line=line_number,
                    )
                )
            seen[name] = entry
            target = expected.get(name)
            if target is None:
                drift.append(
                    _drift("unknown-capability", path=rel, name=name, line=line_number)
                )
                continue
            if target["number"] != number:
                drift.append(
                    _drift(
                        "number-mismatch",
                        path=rel,
                        name=name,
                        expected=target["number"],
                        actual=number,
                        line=line_number,
                    )
                )
            if target.get("status") != status:
                drift.append(
                    _drift(
                        "status-mismatch",
                        path=rel,
                        name=name,
                        expected=target.get("status"),
                        actual=status,
                        line=line_number,
                    )
                )
            if " ".join(str(target.get("boundary") or "").split()) != boundary:
                drift.append(
                    _drift(
                        "boundary-mismatch",
                        path=rel,
                        name=name,
                        expected=target.get("boundary"),
                        actual=boundary,
                        line=line_number,
                    )
                )
        count_match = _CAPABILITY_COUNTS_RE.search(text)
        if seen or count_match is not None:
            for name in sorted(set(expected) - set(seen)):
                drift.append(_drift("missing-row", path=rel, name=name))
        documented_counts = (
            {
                "total": int(count_match.group("total")),
                "foundation": int(count_match.group("foundation")),
                "modelled": int(count_match.group("modelled")),
                "rejected": int(count_match.group("rejected")),
            }
            if count_match is not None
            else None
        )
        if count_match is not None and documented_counts != counts:
            drift.append(
                _drift(
                    "count-mismatch",
                    path=rel,
                    expected=counts,
                    actual=documented_counts,
                )
            )
        if seen:
            documented.append(
                {
                    "path": rel,
                    "rows": len(seen),
                    "counts": {**doc_counts, "total": len(seen)},
                    "declared_counts": documented_counts,
                    "capabilities": sorted(seen),
                }
            )
    if not any(item["rows"] for item in documented):
        drift.append(
            _drift(
                "matrix-empty", path="docs/", detail="no capability table rows found"
            )
        )

    return {
        "function": "docs_sync_capability_matrix",
        "ok": not drift,
        "root": base.as_posix(),
        "sources_read": [source],
        "expected_counts": counts,
        "documented": documented,
        "drift": drift,
        "summary": (
            f"registry: {counts['total']} capabilities "
            f"({counts['foundation']} foundation, {counts['modelled']} modelled, "
            f"{counts['rejected']} rejected); "
            f"{len(drift)} drift record(s) across {len(documented)} doc(s)"
        ),
    }


def _dotted_name(node: ast.expr | None) -> str:
    """Return a dotted source name for an attribute/name expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal(node: ast.expr | None) -> Any:
    """Return a literal value from an AST node, or ``None``."""
    try:
        return ast.literal_eval(node) if node is not None else None
    except ValueError, TypeError, SyntaxError:
        return None


def _cli_facts(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract commands, flags, and backend names from CLI source code."""
    path = root / "api" / "cli.py"
    text = _read_text(path)
    source = _source_document(path, text)
    commands: dict[str, dict[str, Any]] = {}
    drift: list[dict[str, Any]] = []
    if text is None:
        return {"commands": commands, "source": source}, [
            _drift("cli-source-missing", path=source["path"])
        ]
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return {"commands": commands, "source": source}, [
            _drift("cli-source-invalid", path=source["path"], detail=error.msg)
        ]

    build_parser = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
        ),
        None,
    )
    if build_parser is None:
        return {"commands": commands, "source": source}, [
            _drift("cli-parser-missing", path=source["path"])
        ]

    variables: dict[str, str] = {}
    for node in ast.walk(build_parser):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        name = _dotted_name(call.func)
        if name.endswith(".add_parser") and call.args:
            command = _literal(call.args[0])
            if not isinstance(command, str):
                continue
            aliases: list[str] = []
            for keyword in call.keywords:
                if keyword.arg == "aliases":
                    value = _literal(keyword.value)
                    if isinstance(value, (list, tuple)):
                        aliases = [item for item in value if isinstance(item, str)]
            commands[command] = {"aliases": aliases, "flags": set(), "backends": set()}
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables[target.id] = command
        elif name.endswith(".add_mutually_exclusive_group"):
            command = variables.get(_dotted_name(call.func.value))
            for target in node.targets:
                if isinstance(target, ast.Name) and command is not None:
                    variables[target.id] = command

    for call in ast.walk(build_parser):
        if not isinstance(call, ast.Call):
            continue
        name = _dotted_name(call.func)
        if name.endswith(".add_argument") and call.args:
            command = variables.get(_dotted_name(call.func.value))
            flag = _literal(call.args[0])
            if command is None or not isinstance(flag, str):
                continue
            is_option = flag.startswith("--")
            if is_option:
                commands[command]["flags"].add(flag)
            choices: set[str] = set()
            for keyword in call.keywords:
                if keyword.arg == "choices":
                    value = _literal(keyword.value)
                    if isinstance(value, (list, tuple)):
                        choices = {item for item in value if isinstance(item, str)}
            if is_option and flag == "--backend":
                commands[command]["backends"].update(choices)
        elif name == "_add_error_format" and call.args:
            command = variables.get(_dotted_name(call.args[0]))
            if command is not None:
                commands[command]["flags"].add("--error-format")

    base_path = root / "backends" / "base.py"
    base_text = _read_text(base_path)
    backend_names: set[str] = set()
    for command in commands.values():
        backend_names.update(command["backends"])
    if base_text is not None:
        try:
            base_tree = ast.parse(base_text)
        except SyntaxError as error:
            drift.append(
                _drift(
                    "backend-registry-invalid",
                    path=_relative(root, base_path),
                    detail=error.msg,
                )
            )
        else:
            get_backend = next(
                (
                    node
                    for node in base_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_backend"
                ),
                None,
            )
            if get_backend is not None:
                for node in ast.walk(get_backend):
                    if (
                        isinstance(node, ast.Compare)
                        and isinstance(node.left, ast.Name)
                        and node.left.id == "name"
                    ):
                        for comparator in node.comparators:
                            value = _literal(comparator)
                            if isinstance(value, str):
                                backend_names.add(value)
    else:
        drift.append(
            _drift("backend-registry-missing", path=_relative(root, base_path))
        )

    for name in sorted(backend_names - union_backend_choices(commands)):
        drift.append(
            _drift(
                "backend-without-cli-choice",
                path=_relative(root, base_path),
                name=name,
            )
        )
    return (
        {
            "commands": {
                name: {
                    "aliases": data["aliases"],
                    "flags": sorted(data["flags"]),
                    "backends": sorted(data["backends"]),
                }
                for name, data in sorted(commands.items())
            },
            "backend_names": sorted(backend_names),
            "flags": sorted(
                {flag for data in commands.values() for flag in data["flags"]}
            ),
            "source": source,
        },
        drift,
    )


def union_backend_choices(commands: dict[str, dict[str, Any]]) -> set[str]:
    """Return the union of all ``--backend`` choices declared by commands."""
    result: set[str] = set()
    for data in commands.values():
        result.update(data.get("backends", ()))
    return result


def docs_sync_cli_and_backend_flags(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    """Report drift between CLI/backend implementation and its documentation.

    Commands, aliases, options, and backend names are extracted from
    ``api/cli.py`` and ``backends/base.py``.  Root-level guides and every file
    under ``docs/`` are scanned for those names.  Usage lines that advertise a
    finite ``--backend`` choice list are checked against the command's real
    choices, and ``polyglot`` usage lines cannot advertise unknown flags.

    Parameters
    ----------
    root:
        Repository root to inspect.  Defaults to this checkout.
    """
    base = _root_path(root)
    implementation, drift = _cli_facts(base)
    documented = {"commands": set(), "flags": set(), "backends": set()}
    implemented_commands = set(implementation["commands"])
    implemented_flags = set(implementation["flags"])
    implemented_backends = set(implementation["backend_names"])
    sources: list[dict[str, Any]] = [implementation["source"]]

    for path in _doc_paths(base):
        text = _read_text(path)
        if text is None:
            continue
        rel = _relative(base, path)
        sources.append(_source_document(path, text))
        for command in implemented_commands:
            if re.search(rf"\bpolyglot\s+{re.escape(command)}(?:\s|`|$)", text):
                documented["commands"].add(command)
        for flag in implemented_flags:
            if re.search(rf"{re.escape(flag)}(?=[=\s`)\]>,]|$)", text):
                documented["flags"].add(flag)
        for backend in implemented_backends:
            if re.search(rf"\b{re.escape(backend)}\b", text, re.IGNORECASE):
                documented["backends"].add(backend)
        for line_number, line in enumerate(text.splitlines(), start=1):
            command_match = _COMMAND_LINE_RE.search(line)
            for flag in _FLAG_RE.findall(line):
                if command_match is not None and flag not in implemented_flags:
                    drift.append(
                        _drift(
                            "unknown-flag",
                            path=rel,
                            flag=flag,
                            command=command_match.group("command"),
                            line=line_number,
                        )
                    )
            choice_match = _BACKEND_CHOICES_RE.search(line)
            if choice_match is None or "|" not in choice_match.group("choices"):
                continue
            advertised = {
                item.strip()
                for item in choice_match.group("choices").rstrip("])>,").split("|")
                if item.strip()
            }
            command = command_match.group("command") if command_match else ""
            expected = (
                set(implementation["commands"].get(command, {}).get("backends", ()))
                or implemented_backends
            )
            if advertised != expected:
                drift.append(
                    _drift(
                        "backend-choices-mismatch",
                        path=rel,
                        command=command,
                        expected=sorted(expected),
                        actual=sorted(advertised),
                        line=line_number,
                    )
                )

    for command in sorted(implemented_commands - documented["commands"]):
        drift.append(_drift("undocumented-command", command=command))
    for flag in sorted(implemented_flags - documented["flags"]):
        drift.append(_drift("undocumented-flag", flag=flag))
    for backend in sorted(implemented_backends - documented["backends"]):
        drift.append(_drift("undocumented-backend", name=backend))

    return {
        "function": "docs_sync_cli_and_backend_flags",
        "ok": not drift,
        "root": base.as_posix(),
        "sources_read": sources,
        "implementation": implementation,
        "documented": {key: sorted(value) for key, value in documented.items()},
        "drift": drift,
        "summary": (
            f"{len(implementation['commands'])} command(s), "
            f"{len(implementation['flags'])} flag(s), "
            f"{len(implementation['backend_names'])} backend(s); "
            f"{len(drift)} drift record(s)"
        ),
    }


_PLATFORM_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "map_jit",
        "title": "Owned executable pages / MAP_JIT",
        "keywords": ("MAP_JIT", "foreign __TEXT", "E007"),
        "platforms": ("darwin/arm64", "owned RW->RX pages elsewhere"),
        "runtime_requirement": "process-owned pages; foreign loaded text is never patched",
        "mechanism": (
            "An ExecutableRegion token owns the page; an owned sled is written "
            "with an icache flush and re-protected after the write."
        ),
        "boundary": (
            "GOT/PLT cell swaps use one aligned atomic store; foreign __TEXT "
            "and ambiguous relocations are rejected with E007/E008."
        ),
        "model_only": False,
        "probe_key": "executable_region",
    },
    {
        "id": "pep_667",
        "title": "CPython fast-local migration at safe points",
        "keywords": ("PEP 667", "f_code", "fast-local"),
        "platforms": ("CPython >= 3.13",),
        "runtime_requirement": "writable frame locals; future-entry code swap only",
        "mechanism": (
            "The target function's __code__ changes for future calls; live "
            "frames keep the code object they captured and fast locals are "
            "copied through the write-through frame.f_locals mapping."
        ),
        "boundary": (
            "Mid-instruction f_code pointer replacement, layout/closure "
            "mismatches, and interpreters without writable locals are refused."
        ),
        "model_only": False,
        "probe_key": "frame_locals",
    },
    {
        "id": "toctou",
        "title": "Raw-address time-of-check/time-of-use window",
        "keywords": ("TOCTOU", "raw", "mapping"),
        "platforms": ("all",),
        "runtime_requirement": "caller owns the lifetime of every raw range or cell",
        "mechanism": (
            "Raw slot/journal addresses are checked for NULL, alignment, and "
            "an apparent live mapping immediately before use."
        ),
        "boundary": (
            "The check cannot pin another thread's memory: a TOCTOU window "
            "remains, so owned tokens are preferred over raw addresses."
        ),
        "model_only": False,
        "probe_key": None,
    },
    {
        "id": "model_only",
        "title": "Modelled pillar paths",
        "keywords": ("modelled", "in-process"),
        "platforms": ("all",),
        "runtime_requirement": "deterministic in-process model; no privileged mechanism",
        "mechanism": (
            "Capabilities marked modelled are callable Python models used for "
            "reasoning and parity tests."
        ),
        "boundary": (
            "A model is never a substitute for kernel RCU, real accelerator "
            "dispatch, eBPF/MPK enforcement, or threshold signatures."
        ),
        "model_only": True,
        "probe_key": None,
    },
)


def _hotpatch_probe() -> tuple[dict[str, Any], str, str | None]:
    """Return the runtime support probe and its availability status."""
    try:
        module = importlib.import_module("runtime.hotpatch")
        support = module.platform_support()
        probe = dict(support) if hasattr(support, "items") else {"value": support}
        return probe, "available", None
    except Exception as error:  # optional subsystem must not break documentation
        return {}, "unavailable", f"{type(error).__name__}: {error}"


def docs_record_platform_constraints(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    """Return structured records for platform and runtime constraints.

    Four records are produced: ``map_jit``, ``pep_667``, ``toctou``, and
    ``model_only``.  Each record states the mechanism, the honest boundary, the
    platforms/runtime it depends on, and which documentation files mention it.
    The modelled-capability list is derived from the inspected registry, and
    ``runtime.hotpatch.platform_support()`` is included when importable.

    Parameters
    ----------
    root:
        Repository root whose docs and capability registry are inspected.
    """
    base = _root_path(root)
    rows, source = _parse_capability_registry(base)
    counts = _status_counts(rows)
    modelled = sorted(
        row["name"] for row in rows if row.get("status") == "modelled" and "name" in row
    )
    probe, probe_status, probe_error = _hotpatch_probe()

    docs: list[tuple[str, str]] = []
    sources: list[dict[str, Any]] = []
    for path in _doc_paths(base, include_project_docs=True):
        text = _read_text(path)
        if text is None:
            continue
        rel = _relative(base, path)
        docs.append((rel, text))
        sources.append(_source_document(path, text))

    records: list[dict[str, Any]] = []
    for spec in _PLATFORM_RECORDS:
        keywords = tuple(spec["keywords"])
        cited = [
            rel
            for rel, text in docs
            if all(keyword.lower() in text.lower() for keyword in keywords)
        ]
        entry = {
            "id": spec["id"],
            "title": spec["title"],
            "keywords": list(keywords),
            "platforms": list(spec["platforms"]),
            "runtime_requirement": spec["runtime_requirement"],
            "mechanism": spec["mechanism"],
            "boundary": spec["boundary"],
            "model_only": spec["model_only"],
            "documented": bool(cited),
            "sources": cited,
            "markdown": (
                f"- **{spec['title']}**: {spec['mechanism']} Boundary: "
                f"{spec['boundary']}"
            ),
        }
        key = spec["probe_key"]
        if key is not None:
            entry["probe"] = {
                "key": key,
                "status": probe_status,
                "value": probe.get(key) if probe_status == "available" else None,
                "error": probe_error,
            }
        if spec["id"] == "model_only":
            entry["capabilities"] = modelled
            entry["capability_count"] = len(modelled)
        records.append(entry)

    return {
        "function": "docs_record_platform_constraints",
        "ok": all(record["documented"] for record in records),
        "root": base.as_posix(),
        "sources_read": [source, *sources],
        "runtime": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "hotpatch_probe": probe,
            "hotpatch_probe_status": probe_status,
            "hotpatch_probe_error": probe_error,
        },
        "registry": {"source": source, "counts": counts, "modelled": modelled},
        "records": records,
        "summary": (
            f"{len(records)} platform/runtime boundary record(s); "
            f"{sum(record['documented'] for record in records)} documented"
        ),
    }


__all__ = [
    "docs_record_platform_constraints",
    "docs_sync_capability_matrix",
    "docs_sync_cli_and_backend_flags",
    "union_backend_choices",
]
