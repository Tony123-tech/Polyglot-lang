"""Scoped Zig build orchestration for the experimental backend.

The module owns the real toolchain path: generated-module caching, a
``build.zig`` graph, local/global cache isolation, optional ``zig fmt`` and
cold/warm timing metadata.  It never pretends a cached artifact is a fresh
compilation; callers get an explicit ``cache_hit`` flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_OPTIMIZE_MODES = {
    "debug": "Debug",
    "release_fast": "ReleaseFast",
    "release_safe": "ReleaseSafe",
    "release_small": "ReleaseSmall",
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ZigBuildError(RuntimeError):
    """Raised when the Zig toolchain rejects a build."""


@dataclass(frozen=True)
class ZigBuildResult:
    """One built artifact plus the facts needed to audit the build."""

    artifact: Path
    source: Path
    build_file: Path | None
    command: tuple[str, ...]
    elapsed_ms: float
    cache_hit: bool
    formatted: bool
    optimize: str
    incremental: bool
    incremental_fallback: bool = False
    stderr: str = ""
    local_cache: Path | None = None
    global_cache: Path | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly timing/cache record."""
        return {
            "artifact": str(self.artifact),
            "source": str(self.source),
            "build_file": str(self.build_file) if self.build_file else None,
            "command": list(self.command),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cache_hit": self.cache_hit,
            "formatted": self.formatted,
            "optimize": self.optimize,
            "incremental": self.incremental,
            "incremental_fallback": self.incremental_fallback,
            "stderr": self.stderr,
            "local_cache": str(self.local_cache) if self.local_cache else None,
            "global_cache": str(self.global_cache) if self.global_cache else None,
            "extra": dict(self.extra),
        }


def zig_available() -> bool:
    """Return whether a ``zig`` executable is available on PATH."""
    return shutil.which("zig") is not None


def zig_version() -> str:
    """Return the installed Zig version, or an empty string when absent."""
    if not zig_available():
        return ""
    result = subprocess.run(
        ["zig", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def normalize_optimize(value: str | None) -> str:
    """Normalize CLI/profile spellings to a Zig optimize mode."""
    if value is None or value == "":
        return "ReleaseFast"
    key = value.strip().lower().replace("-", "_")
    if key in _OPTIMIZE_MODES:
        return _OPTIMIZE_MODES[key]
    for mode in _OPTIMIZE_MODES.values():
        if mode.lower() == key:
            return mode
    raise ValueError(f"unsupported Zig optimize mode: {value!r}")


def _safe_scope(scope: str | os.PathLike[str] | None) -> str:
    raw = str(scope) if scope is not None else "default"
    cleaned = _SAFE_NAME.sub("_", Path(raw).name).strip("._") or "scope"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:32]}-{digest}"


def scoped_cache_dirs(
    cache_root: str | os.PathLike[str],
    scope: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    """Return isolated local/global Zig cache directories for *scope*.

    Parallel compilations and independent test suites therefore never share a
    mutable cache entry.  The directories are created eagerly so a caller can
    inspect or remove them without racing another process.
    """
    root = Path(cache_root) / "scopes" / _safe_scope(scope)
    local = root / "local"
    global_ = root / "global"
    local.mkdir(parents=True, exist_ok=True)
    global_.mkdir(parents=True, exist_ok=True)
    return local, global_


def _zig_string(value: str | os.PathLike[str]) -> str:
    """Quote a path/string for generated Zig source."""
    return json.dumps(str(value), ensure_ascii=False)


def render_build_zig(
    module_name: str,
    *,
    c_sources: Sequence[str] = (),
    include_dirs: Sequence[str] = (),
    link_libs: Sequence[str] = (),
    link_libc: bool = False,
) -> str:
    """Render a small ``build.zig`` for one dynamic library."""
    if not _SAFE_MODULE.fullmatch(module_name):
        raise ValueError(f"invalid Zig module name: {module_name!r}")
    c_files = ", ".join(_zig_string(path) for path in c_sources)
    c_block = ""
    if c_sources:
        c_block += (
            "    lib.addCSourceFiles(.{ .files = &.{ "
            f"{c_files} "
            '}, .flags = &.{"-std=c11"} });\n'
        )
    for include in include_dirs:
        c_block += f"    lib.addIncludePath(b.path({_zig_string(include)}));\n"
    if link_libc or c_sources:
        c_block += "    lib.linkLibC();\n"
    for library in link_libs:
        c_block += f"    lib.linkSystemLibrary({_zig_string(library)});\n"
    return f"""const std = @import("std");

pub fn build(b: *std.Build) void {{
    const target = b.standardTargetOptions(.{{}});
    const optimize = b.standardOptimizeOption(.{{}});
    const lib = b.addLibrary(.{{
        .name = {_zig_string(module_name)},
        .root_module = b.createModule(.{{
            .root_source_file = b.path({_zig_string(module_name + ".zig")}),
            .target = target,
            .optimize = optimize,
        }}),
        .linkage = .dynamic,
    }});
{c_block}    b.installArtifact(lib);
}}
"""


def _artifact_candidates(out_dir: Path, module_name: str) -> tuple[Path, ...]:
    return (
        out_dir / "lib" / f"lib{module_name}.dylib",
        out_dir / "lib" / f"lib{module_name}.so",
        out_dir / "lib" / f"lib{module_name}.dll",
        out_dir / "bin" / f"lib{module_name}.dylib",
        out_dir / "bin" / f"lib{module_name}.so",
        out_dir / "bin" / f"lib{module_name}.dll",
        out_dir / f"lib{module_name}.dylib",
        out_dir / f"lib{module_name}.so",
        out_dir / f"lib{module_name}.dll",
    )


def format_zig_source(
    source: str, work_dir: str | os.PathLike[str]
) -> tuple[str, bool]:
    """Format *source* with ``zig fmt`` when available.

    The original text is returned unchanged when the formatter is missing or
    rejects the program, so callers still get the compiler's diagnostic.
    """
    if not zig_available():
        return source, False
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # A fixed probe filename races with parallel codegen/build calls, so use
    # a process-unique temporary file inside the caller's work directory.
    handle, name = tempfile.mkstemp(
        prefix="_format_probe_",
        suffix=".zig",
        dir=str(directory),
    )
    os.close(handle)
    candidate = Path(name)
    try:
        candidate.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["zig", "fmt", str(candidate)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return source, False
        return candidate.read_text(encoding="utf-8"), True
    finally:
        candidate.unlink(missing_ok=True)


def check_zig_format(path: str | os.PathLike[str]) -> bool:
    """Return whether *path* already passes ``zig fmt --check``."""
    if not zig_available():
        return False
    result = subprocess.run(
        ["zig", "fmt", "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _content_digests(paths: Sequence[str]) -> tuple[str, ...]:
    """Return stable (path, content-hash) pairs for cache-key inputs.

    Missing files are represented by their name alone so the build still
    produces a clear toolchain error instead of silently reusing a stale key.
    """
    rows: list[str] = []
    for raw in paths:
        path = Path(raw)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = "missing"
        rows.append(f"{raw}:{digest}")
    return tuple(rows)


def _cache_key(
    source: str,
    module_name: str,
    optimize: str,
    version: str,
    extra: Sequence[str] = (),
) -> str:
    payload = "\0".join((version, module_name, optimize, *extra, source))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_dynamic_library(
    source: str,
    module_name: str,
    *,
    cache_root: str | os.PathLike[str],
    scope: str | os.PathLike[str] | None = None,
    optimize: str = "ReleaseFast",
    incremental: bool = False,
    c_sources: Sequence[str] = (),
    include_dirs: Sequence[str] = (),
    link_libs: Sequence[str] = (),
    link_libc: bool = False,
    format_source: bool = True,
) -> ZigBuildResult:
    """Build a Zig dynamic library through a generated ``build.zig`` graph.

    Warm builds reuse the scoped cache and report ``cache_hit=True``.  The
    optional ``-fincremental`` flag is retried without it because some Zig
    releases have platform-specific incremental bugs (for example Mach-O
    archive loading on macOS).  No timing is synthesized: ``elapsed_ms`` is
    wall-clock time around the real command.
    """
    if not zig_available():
        raise ZigBuildError("zig is not installed or not on PATH")
    mode = normalize_optimize(optimize)
    version = zig_version()
    formatted = False
    if format_source:
        source, formatted = format_zig_source(source, Path(cache_root) / "_fmt")
    key = _cache_key(
        source,
        module_name,
        mode,
        version,
        tuple(c_sources)
        + _content_digests(c_sources)
        + tuple(include_dirs)
        + tuple(link_libs)
        + (("incremental",) if incremental else ()),
    )
    local_cache, global_cache = scoped_cache_dirs(cache_root, scope)
    build_dir = Path(cache_root) / "builds" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{module_name}.zig"
    source_path.write_text(source, encoding="utf-8")
    build_file = build_dir / "build.zig"
    build_file.write_text(
        render_build_zig(
            module_name,
            c_sources=c_sources,
            include_dirs=include_dirs,
            link_libs=link_libs,
            link_libc=link_libc,
        ),
        encoding="utf-8",
    )
    out_dir = build_dir / "out"
    existing = next(
        (path for path in _artifact_candidates(out_dir, module_name) if path.is_file()),
        None,
    )
    if existing is not None and existing.stat().st_size > 0:
        return ZigBuildResult(
            artifact=existing,
            source=source_path,
            build_file=build_file,
            command=(),
            elapsed_ms=0.0,
            cache_hit=True,
            formatted=formatted,
            optimize=mode,
            incremental=incremental,
            local_cache=local_cache,
            global_cache=global_cache,
        )

    command = [
        "zig",
        "build",
        f"-Doptimize={mode}",
        "--prefix",
        str(out_dir),
        "--cache-dir",
        str(local_cache),
    ]
    if incremental:
        command.append("-fincremental")
    env = os.environ.copy()
    env["ZIG_LOCAL_CACHE_DIR"] = str(local_cache)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=build_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    fallback = False
    if result.returncode != 0 and incremental:
        # Zig 0.17.1 currently aborts on Mach-O incremental library builds.
        # Retry once without the flag so callers get a working artifact while
        # the result records that the requested incremental mode was not used.
        retry_command = [item for item in command if item != "-fincremental"]
        result = subprocess.run(
            retry_command,
            cwd=build_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        command = retry_command
        fallback = True
    elapsed = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise ZigBuildError(
            "Zig build failed:\n"
            + (result.stderr or result.stdout or "no compiler output")
        )
    artifact = next(
        (path for path in _artifact_candidates(out_dir, module_name) if path.is_file()),
        None,
    )
    if artifact is None or artifact.stat().st_size == 0:
        raise ZigBuildError(
            f"Zig build produced no dynamic library under {out_dir}; "
            f"checked {', '.join(str(p) for p in _artifact_candidates(out_dir, module_name))}"
        )
    return ZigBuildResult(
        artifact=artifact,
        source=source_path,
        build_file=build_file,
        command=tuple(command),
        elapsed_ms=elapsed,
        cache_hit=False,
        formatted=formatted,
        optimize=mode,
        incremental=incremental,
        incremental_fallback=fallback,
        stderr=result.stderr,
        local_cache=local_cache,
        global_cache=global_cache,
    )


def build_dynamic_library_direct(
    source: str,
    module_name: str,
    *,
    cache_root: str | os.PathLike[str],
    scope: str | os.PathLike[str] | None = None,
    optimize: str = "ReleaseFast",
) -> ZigBuildResult:
    """Fallback path using ``zig build-lib`` instead of ``zig build``."""
    if not zig_available():
        raise ZigBuildError("zig is not installed or not on PATH")
    mode = normalize_optimize(optimize)
    version = zig_version()
    key = _cache_key(source, module_name, mode, version, ("direct",))
    local_cache, global_cache = scoped_cache_dirs(cache_root, scope)
    build_dir = Path(cache_root) / "direct" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{module_name}.zig"
    source_path.write_text(source, encoding="utf-8")
    suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    artifact = build_dir / f"lib{module_name}{suffix}"
    if artifact.is_file() and artifact.stat().st_size > 0:
        return ZigBuildResult(
            artifact=artifact,
            source=source_path,
            build_file=None,
            command=(),
            elapsed_ms=0.0,
            cache_hit=True,
            formatted=False,
            optimize=mode,
            incremental=False,
            local_cache=local_cache,
            global_cache=global_cache,
        )
    command = [
        "zig",
        "build-lib",
        str(source_path),
        "-dynamic",
        "-O",
        mode,
        f"-femit-bin={artifact}",
        "--cache-dir",
        str(local_cache),
        "--global-cache-dir",
        str(global_cache),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise ZigBuildError(
            "Zig build-lib failed:\n"
            + (result.stderr or result.stdout or "no compiler output")
        )
    return ZigBuildResult(
        artifact=artifact,
        source=source_path,
        build_file=None,
        command=tuple(command),
        elapsed_ms=elapsed,
        cache_hit=False,
        formatted=False,
        optimize=mode,
        incremental=False,
        stderr=result.stderr,
        local_cache=local_cache,
        global_cache=global_cache,
    )


def clear_scope(
    cache_root: str | os.PathLike[str], scope: str | os.PathLike[str]
) -> None:
    """Remove one isolated scope directory."""
    path = Path(cache_root) / "scopes" / _safe_scope(scope)
    shutil.rmtree(path, ignore_errors=True)
