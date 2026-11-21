"""C and C++ interop through ``zig cc`` / ``zig c++``.

Zig ships a Clang-compatible driver, so the experimental backend can compile a
small C/C++ translation unit without requiring a second toolchain.  This module
deliberately keeps the invocation explicit:

* source paths are passed as-is;
* include/library/define flags are never inferred from the environment;
* local and global Zig caches are scoped, so parallel test runs do not race;
* every result records wall-clock timing and whether an artifact was reused.

It does not fake a Python extension.  The produced file is an ordinary shared
library that callers load with :mod:`ctypes` (or link from Zig itself).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from zig.build import (
    ZigBuildError,
    _content_digests,
    normalize_optimize,
    scoped_cache_dirs,
    zig_available,
    zig_version,
)

__all__ = [
    "CcBuildResult",
    "compile_shared",
    "default_shared_name",
]


_OPT_FLAGS = {
    "Debug": "-O0",
    "ReleaseSafe": "-O2",
    "ReleaseFast": "-O3",
    "ReleaseSmall": "-Os",
}


@dataclass(frozen=True)
class CcBuildResult:
    """One ``zig cc``/``zig c++`` invocation and its measured outcome."""

    artifact: Path
    sources: tuple[Path, ...]
    command: tuple[str, ...]
    elapsed_ms: float
    cache_hit: bool
    compiler: str
    optimize: str
    local_cache: Path
    global_cache: Path
    stderr: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly audit record."""
        return {
            "artifact": str(self.artifact),
            "sources": [str(path) for path in self.sources],
            "command": list(self.command),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cache_hit": self.cache_hit,
            "compiler": self.compiler,
            "optimize": self.optimize,
            "local_cache": str(self.local_cache),
            "global_cache": str(self.global_cache),
            "stderr": self.stderr,
            "extra": dict(self.extra),
        }


def default_shared_name(module_name: str) -> str:
    """Return the platform shared-library name for *module_name*."""
    if sys.platform == "darwin":
        return f"lib{module_name}.dylib"
    if os.name == "nt":
        return f"{module_name}.dll"
    return f"lib{module_name}.so"


def _clean_module_name(module_name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in module_name
    ).strip("._-")
    return cleaned or "polyglot_cc"


def _key(
    sources: Sequence[Path],
    compiler: str,
    optimize: str,
    args: Sequence[str],
    output_name: str,
) -> str:
    payload = "\0".join(
        (
            zig_version(),
            compiler,
            optimize,
            output_name,
            *args,
            *_content_digests([str(path) for path in sources]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compile_shared(
    sources: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    cache_root: str | os.PathLike[str] = ".polyglot-cache/zig-cc",
    scope: str | os.PathLike[str] | None = None,
    module_name: str | None = None,
    output: str | os.PathLike[str] | None = None,
    cxx: bool = False,
    optimize: str = "ReleaseFast",
    include_dirs: Sequence[str | os.PathLike[str]] = (),
    library_dirs: Sequence[str | os.PathLike[str]] = (),
    link_libs: Sequence[str] = (),
    defines: Sequence[str] = (),
    flags: Sequence[str] = (),
    std: str | None = None,
    force: bool = False,
) -> CcBuildResult:
    """Compile *sources* into a shared library with ``zig cc``/``zig c++``.

    ``cache_root``/``scope`` isolate the compiler caches.  When ``output`` is
    omitted the artifact lives inside the scoped build directory, which makes
    cache-key reuse safe.  A caller-supplied output is rebuilt whenever the
    generated cache key changes.
    """
    if not zig_available():
        raise ZigBuildError("zig is not installed or not on PATH")
    if isinstance(sources, (str, os.PathLike)):
        raw_sources: list[str | os.PathLike[str]] = [sources]
    else:
        raw_sources = list(sources)
    if not raw_sources:
        raise ValueError("compile_shared requires at least one source file")
    source_paths = tuple(Path(path).expanduser() for path in raw_sources)
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("C/C++ source file(s) not found: " + ", ".join(missing))

    compiler = "c++" if cxx else "cc"
    standard = std or ("c++20" if cxx else "c11")
    mode = normalize_optimize(optimize)
    module = _clean_module_name(module_name or source_paths[0].stem)
    output_name = (
        Path(output).name if output is not None else default_shared_name(module)
    )
    normalized = (
        [f"-std={standard}", _OPT_FLAGS[mode], "-shared", "-fPIC"]
        + [f"-I{Path(path)}" for path in include_dirs]
        + [f"-L{Path(path)}" for path in library_dirs]
        + [f"-l{library}" for library in link_libs]
        + [f"-D{item}" for item in defines]
        + list(flags)
    )
    key = _key(source_paths, compiler, mode, normalized, output_name)
    local_cache, global_cache = scoped_cache_dirs(cache_root, scope)
    build_dir = Path(cache_root) / "cc" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact = (
        Path(output).expanduser()
        if output is not None
        else build_dir / default_shared_name(module)
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    stamp = build_dir / f"{artifact.name}.key"
    reusable = (
        not force
        and artifact.is_file()
        and artifact.stat().st_size > 0
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8").strip() == key
    )
    if reusable:
        return CcBuildResult(
            artifact=artifact,
            sources=source_paths,
            command=(),
            elapsed_ms=0.0,
            cache_hit=True,
            compiler=f"zig {compiler}",
            optimize=mode,
            local_cache=local_cache,
            global_cache=global_cache,
        )

    command = [
        "zig",
        compiler,
        *normalized,
        "-o",
        str(artifact),
        *(str(path) for path in source_paths),
    ]
    env = os.environ.copy()
    env["ZIG_LOCAL_CACHE_DIR"] = str(local_cache)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        artifact.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        raise ZigBuildError(
            f"zig {compiler} failed:\n"
            + (result.stderr or result.stdout or "no compiler output")
        )
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ZigBuildError(f"zig {compiler} produced no artifact at {artifact}")
    stamp.write_text(key, encoding="utf-8")
    return CcBuildResult(
        artifact=artifact,
        sources=source_paths,
        command=tuple(command),
        elapsed_ms=elapsed,
        cache_hit=False,
        compiler=f"zig {compiler}",
        optimize=mode,
        local_cache=local_cache,
        global_cache=global_cache,
        stderr=result.stderr,
        extra={"key": key},
    )
