"""Deterministic build-cache keys, scoped directories, and pruning helpers.

Every digest in this module is SHA-256 over canonical JSON; Python's
per-process ``hash()`` is never used.  Callers pass explicit repository or
cache roots, and pruning is serialized with an exclusive ``O_EXCL`` lock file
so concurrent pruners cannot evict the same entry twice.  Invalid API inputs
raise :class:`~compiler.frontend.errors.PolyglotError` with the E001-E008
surface; a busy lock is an E004 tool failure.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "bk_assert_no_hash_usage",
    "bk_canonical_json",
    "bk_canonicalize_env",
    "bk_canonicalize_flags",
    "bk_classify_cache_entry",
    "bk_combine_keys",
    "bk_describe_cache_entry",
    "bk_hash_payload",
    "bk_key_for_command",
    "bk_key_for_env",
    "bk_key_for_manifest",
    "bk_key_for_platform",
    "bk_key_for_profile",
    "bk_key_for_sources",
    "bk_key_for_toolchain",
    "bk_measure_cache_usage",
    "bk_plan_prune",
    "bk_prune_cache_entries",
    "bk_scoped_cache_dir",
    "bk_scoped_cargo_target_dir",
    "bk_scoped_pip_cache",
    "bk_scoped_pytest_cache",
    "bk_scoped_python_paths",
    "bk_scoped_ruff_cache",
    "bk_validate_cache_key",
]

DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_STALE_LOCK_SECONDS = 300.0
PRUNE_LOCK_NAME = ".prune.lock"

_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{64}\Z")
_HASH_CALL_RE = re.compile(r"\bbuiltins\s*\.\s*hash\s*\(|(?<![.\w])hash\s*\(")


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured build-cache diagnostic."""
    raise PolyglotError(code, message, phase="build-ci")


def _text(value: object, name: str) -> str:
    """Require a non-empty string, raising E003 otherwise."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{name} must be a non-empty str")
    return value


def _count(value: object, name: str) -> int:
    """Require a non-negative integer (bool is not accepted)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("E003", f"{name} must be a non-negative int")
    return value


def _seconds(value: object, name: str) -> float:
    """Require a finite non-negative number of seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E003", f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        _fail("E003", f"{name} must be finite and non-negative")
    return result


def _path(value: object, name: str) -> Path:
    """Require a usable path-like value, expanding ``~`` only."""
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    _fail("E003", f"{name} must be a str or pathlib.Path")


def _component(value: object, name: str) -> str:
    """Validate a single cache path component without sanitizing it."""
    text = _text(value, name)
    if text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        _fail("E003", f"{name} must be a safe path component: {text!r}")
    return text


def _filename(value: object, name: str) -> str:
    """Validate a single file or directory name, allowing a leading dot."""
    text = _text(value, name)
    if text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        _fail("E003", f"{name} must be a plain filename: {text!r}")
    return text


def _sequence(value: object, name: str) -> list[Any]:
    """Require a non-string sequence, returning a mutable copy."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail("E003", f"{name} must be a sequence")
    return list(value)


def _env(value: object, name: str = "env") -> dict[str, str]:
    """Require a mapping whose keys and values are both strings."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_text(key, f"{name} key")] = _text(item, f"{name}[{key!r}]")
    return dict(sorted(result.items()))


def _jsonable(value: Any, where: str) -> Any:
    """Convert *value* into a JSON-safe structure or raise E003."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("E003", f"{where} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = _text(key, f"{where} key")
            result[child] = _jsonable(item, f"{where}.{child}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, f"{where}[{index}]") for index, item in enumerate(value)
        ]
    _fail("E003", f"{where} has unsupported type {type(value).__name__}")


def _canonical_text(payload: Any) -> str:
    """Serialize *payload* with sorted keys and compact separators."""
    safe = _jsonable(payload, "payload")
    try:
        return json.dumps(
            safe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("E003", f"payload is not canonicalizable: {error}")


def _digest(payload: Any) -> str:
    """Return the SHA-256 hex digest of canonical JSON for *payload*."""
    return hashlib.sha256(_canonical_text(payload).encode("utf-8")).hexdigest()


def _relative_name(root: Path, path: Path) -> str:
    """Return a POSIX relative name for *path* below *root*."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        _fail("E003", f"path is outside the cache root: {path}")


def _entry_size(path: Path) -> int:
    """Recursive byte size of one cache entry, excluding nothing."""
    try:
        if path.is_dir() and not path.is_symlink():
            return sum(_entry_size(child) for child in path.iterdir())
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
    except OSError as error:
        _fail("E004", f"cannot stat cache entry {path}: {error}")
    return 0


def _entry_digest(path: Path) -> str:
    """Content digest of a file entry or a sorted directory manifest."""
    try:
        if path.is_dir() and not path.is_symlink():
            files = sorted(
                (child for child in path.rglob("*") if child.is_file()),
                key=lambda child: _relative_name(path, child),
            )
            return _digest(
                [[_relative_name(path, child), _entry_digest(child)] for child in files]
            )
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        _fail("E004", f"cannot read cache entry {path}: {error}")


def _remove_entry(path: Path) -> None:
    """Remove a file, symlink, or directory cache entry."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as error:
        _fail("E004", f"cannot remove cache entry {path}: {error}")


def _acquire_prune_lock(lock_path: Path, *, now: float | None, stale: float) -> int:
    """Create an exclusive lock file, raising E004 when another pruner holds it."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    lock_exists = False
    try:
        return os.open(str(lock_path), flags, 0o600)
    except FileExistsError:
        lock_exists = True
    except OSError as error:
        _fail("E004", f"cannot create prune lock {lock_path}: {error}")
    if now is not None and lock_exists:
        try:
            age = now - lock_path.stat().st_mtime
        except OSError:
            age = stale + 1.0
        if age > stale:
            try:
                lock_path.unlink(missing_ok=True)
                return os.open(str(lock_path), flags, 0o600)
            except OSError as error:
                _fail("E004", f"cannot reclaim stale prune lock: {error}")
    _fail("E004", f"cache prune lock is busy: {lock_path}")


def _release_prune_lock(lock_fd: int, lock_path: Path) -> None:
    """Close and delete an exclusively held prune lock."""
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)


def _prune_report(root: Path, lock_name: str) -> dict[str, Any]:
    """Empty prune report with stable keys."""
    return {
        "root": str(root),
        "lock_name": lock_name,
        "removed": [],
        "expired_count": 0,
        "oversized_count": 0,
        "freed_bytes": 0,
        "retained_bytes": 0,
    }


def bk_assert_no_hash_usage(source: object) -> dict[str, Any]:
    """Report bare ``hash()`` calls in Python source text.

    Only calls to the builtin are flagged: ``hashlib.sha256(...)`` and
    ``obj.hash(...)`` are legitimate deterministic constructs.  The scan is
    line-based so reported locations are stable across processes.
    """
    text = _text(source, "source")
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _HASH_CALL_RE.search(line):
            findings.append({"line": line_number, "text": line.strip()})
    return {
        "ok": not findings,
        "findings": findings,
        "scanned_lines": len(text.splitlines()),
    }


def bk_canonical_json(payload: Any) -> dict[str, Any]:
    """Return canonical JSON text and its digest for an arbitrary payload."""
    text = _canonical_text(payload)
    return {
        "text": text,
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
    }


def bk_canonicalize_env(
    env: object, *, allowed_prefixes: Sequence[str] = ()
) -> dict[str, Any]:
    """Normalize an environment mapping into sorted string pairs.

    Keys outside *allowed_prefixes* are dropped when prefixes are supplied.
    The fingerprint hashes the resulting pairs, never a Python ``dict`` hash.
    """
    entries = list(_env(env).items())
    prefixes = tuple(
        _text(item, "prefix")
        for item in _sequence(allowed_prefixes, "allowed_prefixes")
    )
    if prefixes:
        entries = [
            (key, value)
            for key, value in entries
            if any(key.startswith(prefix) for prefix in prefixes)
        ]
    pairs = [[key, value] for key, value in entries]
    return {
        "entries": pairs,
        "fingerprint": _digest(pairs),
        "count": len(pairs),
    }


def bk_canonicalize_flags(flags: object) -> dict[str, Any]:
    """Normalize feature flags from a mapping or a sequence of names.

    Boolean values become ``"1"``/``"0"`` and ``None`` becomes ``""`` so the
    same logical flag set hashes identically regardless of its input form.
    """
    if isinstance(flags, str):
        items: list[tuple[Any, Any]] = [(flags, True)]
    elif isinstance(flags, Mapping):
        items = list(flags.items())
    else:
        items = [(item, True) for item in _sequence(flags, "flags")]
    normalized: dict[str, str] = {}
    for key, value in items:
        name = _text(key, "flag key").strip().lower()
        if not name:
            _fail("E003", "flag keys must not be blank")
        if isinstance(value, bool):
            rendered = "1" if value else "0"
        elif value is None:
            rendered = ""
        elif isinstance(value, (int, float, str)):
            rendered = str(value).strip().lower()
        else:
            _fail("E003", f"flag {name!r} has unsupported value type")
        normalized[name] = rendered
    pairs = [[key, normalized[key]] for key in sorted(normalized)]
    return {
        "entries": pairs,
        "fingerprint": _digest(pairs),
        "count": len(pairs),
    }


def bk_classify_cache_entry(
    name: object, *, temp_names: Sequence[str] = (), temp_suffixes: Sequence[str] = ()
) -> dict[str, Any]:
    """Classify one cache entry name as temporary or retained."""
    text = _text(name, "name")
    lowered = text.lower()
    names = tuple(
        _text(item, "temp name") for item in _sequence(temp_names, "temp_names")
    )
    suffixes = tuple(
        _text(item, "temp suffix") for item in _sequence(temp_suffixes, "temp_suffixes")
    )
    reason = "retained"
    if any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in names):
        reason = "temporary name"
    elif any(lowered.endswith(suffix.lower()) for suffix in suffixes):
        reason = "temporary suffix"
    return {"name": text, "temporary": reason != "retained", "reason": reason}


def bk_combine_keys(parts: object, *, label: str = "combined") -> dict[str, Any]:
    """Combine already-computed scoped keys into one stable composite key."""
    values = _sequence(parts, "parts")
    if not values:
        _fail("E003", "parts must not be empty")
    rendered = [_text(item, "cache key") for item in values]
    for item in rendered:
        if len(item) > 256 or not _COMPONENT_RE.fullmatch(item):
            _fail("E003", f"malformed cache key component: {item!r}")
    prefix = _component(label, "label")
    return {
        "key": f"{prefix}-{_digest(sorted(rendered))}",
        "parts": rendered,
        "count": len(rendered),
    }


def bk_describe_cache_entry(
    path: object, *, now: float | None = None
) -> dict[str, Any]:
    """Describe one cache entry with its age, size, and content digest."""
    entry = _path(path, "path")
    if not entry.exists():
        _fail("E006", f"cache entry is missing: {entry}")
    try:
        stat = entry.stat()
    except OSError as error:
        _fail("E004", f"cannot stat cache entry {entry}: {error}")
    current = time.time() if now is None else _seconds(now, "now")
    return {
        "path": str(entry),
        "name": entry.name,
        "size_bytes": _entry_size(entry),
        "age_seconds": max(0.0, current - stat.st_mtime),
        "modified": stat.st_mtime,
        "is_dir": entry.is_dir(),
        "digest": _entry_digest(entry),
    }


def bk_hash_payload(payload: Any) -> dict[str, Any]:
    """Hash a JSON-compatible payload with a canonical, process-independent form."""
    text = _canonical_text(payload)
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
    }


def bk_key_for_command(
    args: object, *, root: object, env: object | None = None
) -> dict[str, Any]:
    """Derive a key from an argv tuple, explicit working directory, and env."""
    argv = tuple(_text(item, "argv item") for item in _sequence(args, "args"))
    if not argv:
        _fail("E003", "args must not be empty")
    cwd = _path(root, "root")
    payload = {
        "argv": list(argv),
        "cwd": str(cwd),
        "env": [] if env is None else list(_env(env).items()),
    }
    return {
        "key": f"cmd-{_digest(payload)}",
        "argv": list(argv),
        "cwd": str(cwd),
        "digest": _digest(payload),
    }


def bk_key_for_env(
    env: object, *, allowed_prefixes: Sequence[str] = ()
) -> dict[str, Any]:
    """Derive a key from selected environment variables."""
    canonical = bk_canonicalize_env(env, allowed_prefixes=allowed_prefixes)
    return {
        "key": f"env-{canonical['fingerprint']}",
        "entries": canonical["entries"],
        "count": canonical["count"],
    }


def bk_key_for_manifest(manifest: object) -> dict[str, Any]:
    """Derive a key from a structured manifest of names and metadata."""
    if not isinstance(manifest, Mapping):
        _fail("E003", "manifest must be a mapping")
    digest = _digest(manifest)
    return {
        "key": f"manifest-{digest}",
        "digest": digest,
        "entry_count": len(manifest),
    }


def bk_key_for_platform(
    os_name: object, arch: object, *, abi: object = "none"
) -> dict[str, Any]:
    """Derive a normalized key for an OS, architecture, and ABI tuple."""
    normalized_os = _text(os_name, "os_name").strip().lower()
    normalized_arch = _text(arch, "arch").strip().lower()
    normalized_abi = _text(abi, "abi").strip().lower()
    payload = {
        "os": normalized_os,
        "arch": normalized_arch,
        "abi": normalized_abi,
    }
    return {
        "key": f"platform-{_digest(payload)}",
        "os": normalized_os,
        "arch": normalized_arch,
        "abi": normalized_abi,
    }


def bk_key_for_profile(profile: object, *, scope_key: object) -> dict[str, Any]:
    """Derive a key for one build profile inside an existing scope."""
    name = _component(profile, "profile")
    scope = _component(scope_key, "scope_key")
    payload = {"profile": name, "scope_key": scope}
    return {
        "key": f"profile-{_digest(payload)}",
        "profile": name,
        "scope_key": scope,
    }


def bk_key_for_sources(
    root: object,
    *,
    paths: Sequence[str | Path] | None = None,
    include: Sequence[str] = ("*",),
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Hash source file contents into a stable key for the given root.

    Explicit relative paths are preferred; otherwise files are discovered with
    ``rglob`` and filtered by POSIX-style glob patterns.  Every discovered file
    must exist, otherwise E006 is raised instead of silently producing a key.
    """
    base = _path(root, "root")
    if not base.is_dir():
        _fail("E006", f"source root is not a directory: {base}")
    includes = tuple(
        _text(item, "include pattern") for item in _sequence(include, "include")
    )
    excludes = tuple(
        _text(item, "exclude pattern") for item in _sequence(exclude, "exclude")
    )
    if paths is None:
        candidates = sorted(
            (item for item in base.rglob("*") if item.is_file()),
            key=lambda item: _relative_name(base, item),
        )
    else:
        candidates = [_path(item, "path") for item in _sequence(paths, "paths")]
    files: list[list[str]] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else base / candidate
        relative = _relative_name(base, resolved)
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in includes):
            continue
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in excludes):
            continue
        if not resolved.is_file():
            _fail("E006", f"source file is missing: {resolved}")
        files.append([relative, _entry_digest(resolved)])
    return {
        "key": f"sources-{_digest(files)}",
        "files": files,
        "count": len(files),
        "root": str(base.resolve()),
    }


def bk_key_for_toolchain(toolchain: object) -> dict[str, Any]:
    """Derive a key from normalized compiler and tool version strings."""
    if not isinstance(toolchain, Mapping):
        _fail("E003", "toolchain must be a mapping")
    components: dict[str, str] = {}
    for key, value in toolchain.items():
        name = _text(key, "toolchain key").strip().lower()
        components[name] = _text(value, f"toolchain[{key!r}]").strip().lower()
    return {
        "key": f"toolchain-{_digest(components)}",
        "components": dict(sorted(components.items())),
        "digest": _digest(components),
    }


def bk_measure_cache_usage(cache_root: object) -> dict[str, Any]:
    """Summarize recursive byte usage of every entry under a cache root."""
    root = _path(cache_root, "cache_root")
    if not root.is_dir():
        _fail("E006", f"cache root is not a directory: {root}")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    measured = [
        {"name": item.name, "size_bytes": _entry_size(item)} for item in entries
    ]
    return {
        "root": str(root),
        "entries": measured,
        "count": len(measured),
        "total_bytes": sum(item["size_bytes"] for item in measured),
    }


def bk_plan_prune(
    entries: object,
    *,
    max_age_seconds: object = DEFAULT_MAX_AGE_SECONDS,
    max_total_bytes: object = DEFAULT_MAX_TOTAL_BYTES,
    now: object = 0.0,
) -> dict[str, Any]:
    """Plan eviction from pre-measured entries without touching the filesystem."""
    age_limit = _seconds(max_age_seconds, "max_age_seconds")
    budget = _count(max_total_bytes, "max_total_bytes")
    current = _seconds(now, "now")
    described: list[dict[str, Any]] = []
    for entry in _sequence(entries, "entries"):
        if not isinstance(entry, Mapping):
            _fail("E003", "each entry must be a mapping")
        name = _text(entry.get("name"), "entry name")
        described.append(
            {
                "name": name,
                "age_seconds": _seconds(entry.get("age_seconds", 0), "age_seconds"),
                "size_bytes": _count(entry.get("size_bytes", 0), "size_bytes"),
                "modified": entry.get("modified", current),
            }
        )
    expired = [item for item in described if item["age_seconds"] > age_limit]
    expired_names = {item["name"] for item in expired}
    survivors = [item for item in described if item["name"] not in expired_names]
    total = sum(item["size_bytes"] for item in survivors)
    oversized: list[str] = []
    for item in sorted(survivors, key=lambda row: (row["modified"], row["name"])):
        if total <= budget:
            break
        oversized.append(item["name"])
        total -= item["size_bytes"]
    retained = [
        item["name"]
        for item in sorted(survivors, key=lambda row: row["name"])
        if item["name"] not in set(oversized)
    ]
    freed = sum(item["size_bytes"] for item in expired)
    freed += sum(
        item["size_bytes"] for item in survivors if item["name"] in set(oversized)
    )
    return {
        "expired": sorted(expired_names),
        "oversized": sorted(oversized),
        "retained": retained,
        "freed_bytes": freed,
        "retained_bytes": max(0, total),
        "now": current,
    }


def bk_prune_cache_entries(
    cache_root: object,
    *,
    max_age_seconds: object = DEFAULT_MAX_AGE_SECONDS,
    max_total_bytes: object = DEFAULT_MAX_TOTAL_BYTES,
    now: float | None = None,
    stale_lock_seconds: object = DEFAULT_STALE_LOCK_SECONDS,
    lock_name: str = PRUNE_LOCK_NAME,
) -> dict[str, Any]:
    """Evict expired then oldest oversized entries under an exclusive lock.

    The lock file is created with ``O_CREAT | O_EXCL`` and always removed in a
    ``finally`` block.  Supplying *now* makes both expiry and stale-lock
    reclamation deterministic; a fresh foreign lock raises E004.
    """
    age_limit = _seconds(max_age_seconds, "max_age_seconds")
    budget = _count(max_total_bytes, "max_total_bytes")
    stale = _seconds(stale_lock_seconds, "stale_lock_seconds")
    name = _filename(lock_name, "lock_name")
    root = _path(cache_root, "cache_root")
    report = _prune_report(root, name)
    if not root.is_dir():
        return report
    current = time.time() if now is None else _seconds(now, "now")
    lock_path = root / name
    lock_fd = _acquire_prune_lock(lock_path, now=current, stale=stale)
    try:
        entries = sorted(
            (path for path in root.iterdir() if path.name != name),
            key=lambda path: path.name,
        )
        survivors: list[Path] = []
        for path in entries:
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if current - modified > age_limit:
                size = _entry_size(path)
                _remove_entry(path)
                report["removed"].append(path.name)
                report["expired_count"] += 1
                report["freed_bytes"] += size
            else:
                survivors.append(path)
        sized = [(path, _entry_size(path), path.stat().st_mtime) for path in survivors]
        total = sum(item[1] for item in sized)
        for path, size, _modified in sorted(
            sized, key=lambda item: (item[2], item[0].name)
        ):
            if total <= budget:
                break
            _remove_entry(path)
            report["removed"].append(path.name)
            report["oversized_count"] += 1
            report["freed_bytes"] += size
            total -= size
        report["retained_bytes"] = max(0, total)
        return report
    finally:
        _release_prune_lock(lock_fd, lock_path)


def bk_scoped_cache_dir(
    cache_root: object, *, scope_key: object, component: object
) -> dict[str, Any]:
    """Create and describe a dedicated directory for one cache component."""
    root = _path(cache_root, "cache_root")
    scope = _component(scope_key, "scope_key")
    part = _component(component, "component")
    target = root / "scopes" / scope / part
    created = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    return {
        "path": str(target),
        "scope_key": scope,
        "component": part,
        "created": created,
        "exists": target.is_dir(),
    }


def bk_scoped_cargo_target_dir(
    cache_root: object,
    *,
    scope_key: object,
    profile: object,
    env: object | None = None,
) -> dict[str, Any]:
    """Create a profile-specific Cargo target dir and return its env snapshot."""
    root = _path(cache_root, "cache_root")
    scope = _component(scope_key, "scope_key")
    name = _component(profile, "profile")
    target = root / "cargo" / scope / name
    target.mkdir(parents=True, exist_ok=True)
    updated = {} if env is None else _env(env)
    updated["CARGO_TARGET_DIR"] = str(target)
    return {
        "path": str(target),
        "env": dict(sorted(updated.items())),
        "scope_key": scope,
        "profile": name,
    }


def bk_scoped_pip_cache(
    cache_root: object, *, scope_key: object, env: object | None = None
) -> dict[str, Any]:
    """Create a scope-specific pip cache and return its env snapshot."""
    root = _path(cache_root, "cache_root")
    scope = _component(scope_key, "scope_key")
    target = root / "pip" / scope
    target.mkdir(parents=True, exist_ok=True)
    updated = {} if env is None else _env(env)
    updated["PIP_CACHE_DIR"] = str(target)
    return {
        "path": str(target),
        "env": dict(sorted(updated.items())),
        "scope_key": scope,
    }


def bk_scoped_pytest_cache(
    cache_root: object, *, scope_key: object, env: object | None = None
) -> dict[str, Any]:
    """Create a scope-specific pytest cache and return its env snapshot."""
    root = _path(cache_root, "cache_root")
    scope = _component(scope_key, "scope_key")
    target = root / "pytest" / scope
    target.mkdir(parents=True, exist_ok=True)
    updated = {} if env is None else _env(env)
    updated["PYTEST_ADDOPTS"] = f"-o cache_dir={target}"
    return {
        "path": str(target),
        "env": dict(sorted(updated.items())),
        "scope_key": scope,
    }


def bk_scoped_python_paths(
    cache_root: object,
    *,
    scope_key: object,
    workspace: object,
    venv_name: object = ".venv",
) -> dict[str, Any]:
    """Plan isolated Python paths that never overlap the workspace."""
    root = _path(cache_root, "cache_root").resolve()
    work = _path(workspace, "workspace").resolve()
    if root == work or root.is_relative_to(work) or work.is_relative_to(root):
        _fail("E003", f"cache_root must not overlap workspace: {root} vs {work}")
    scope = _component(scope_key, "scope_key")
    venv = _filename(venv_name, "venv_name")
    base = root / "python" / scope
    paths = {
        "venv": base / venv,
        "python": base / venv / "bin" / "python",
        "pip_cache": base / "pip-cache",
        "build_cache": base / "build",
        "pytest_cache": base / "pytest-cache",
        "pycache": base / "pycache",
    }
    for key in ("pip_cache", "build_cache", "pytest_cache", "pycache"):
        paths[key].mkdir(parents=True, exist_ok=True)
    env = {
        "VIRTUAL_ENV": str(paths["venv"]),
        "PIP_CACHE_DIR": str(paths["pip_cache"]),
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
    }
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "env": env,
        "scope_key": scope,
        "workspace": str(work),
    }


def bk_scoped_ruff_cache(
    cache_root: object, *, scope_key: object, env: object | None = None
) -> dict[str, Any]:
    """Create a scope-specific Ruff cache and return its env snapshot."""
    root = _path(cache_root, "cache_root")
    scope = _component(scope_key, "scope_key")
    target = root / "ruff" / scope
    target.mkdir(parents=True, exist_ok=True)
    updated = {} if env is None else _env(env)
    updated["RUFF_CACHE_DIR"] = str(target)
    return {
        "path": str(target),
        "env": dict(sorted(updated.items())),
        "scope_key": scope,
    }


def bk_validate_cache_key(
    key: object, *, prefix: object | None = None
) -> dict[str, Any]:
    """Validate that a cache key has a stable prefix and a 64-hex SHA-256 body."""
    value = _text(key, "key")
    match = _KEY_RE.fullmatch(value)
    if match is None:
        _fail("E003", f"malformed cache key: {value!r}")
    actual_prefix, digest = value.rsplit("-", 1)
    if prefix is not None and _text(prefix, "prefix") != actual_prefix:
        _fail("E003", f"cache key prefix mismatch: {actual_prefix!r}")
    return {
        "valid": True,
        "prefix": actual_prefix,
        "digest": digest,
        "length": len(value),
    }
