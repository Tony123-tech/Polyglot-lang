"""Deterministic build provenance recording.

Generates build provenance records (toolchain versions, target triple, isolated
cache scope keys, input/output hashes) inside scoped temp/build directories
(``CARGO_TARGET_DIR``, isolated Zig cache, temp paths) without leaving
untracked artifacts in the repository.

Every record is written under ``<cache_root>/provenance/<scope_key>/`` so
parallel runs never collide and the repository working tree stays clean.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.build_ci.framework import (
    ci_compute_cache_scope_key,
    ci_isolate_cargo_target_dir,
)

__all__ = [
    "BuildProvenance",
    "ProvenanceRecorder",
    "default_scope_dimensions",
    "record_build_provenance",
]

BuildFn = Callable[[dict[str, str], dict[str, str], Path], list[Path]]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toolchain_version(binary: str) -> tuple[str, bool]:
    """Return ``(version, available)`` for a toolchain binary."""
    executable = shutil.which(binary)
    if executable is None:
        return "", False
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (result.stdout or result.stderr or "").strip()
        return text.splitlines()[0] if text else "", True
    except OSError, subprocess.SubprocessError:
        return "", True


def default_scope_dimensions(*, backend: str = "python") -> dict[str, str]:
    """Return the canonical scope dimensions for one backend."""
    return {
        "os": platform.system() or "unknown",
        "arch": platform.machine() or "unknown",
        "compiler": backend,
        "compiler_version": sys.version.split()[0],
    }


@dataclass
class BuildProvenance:
    """One recorded build provenance record."""

    schema: str = "polyglot-build-provenance/v1"
    run_id: str = ""
    timestamp: float = 0.0
    backend: str = "python"
    toolchain: dict[str, Any] = None
    target: dict[str, Any] = None
    cache: dict[str, Any] = None
    inputs: dict[str, str] = None
    outputs: dict[str, str] = None
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.toolchain is None:
            self.toolchain = {}
        if self.target is None:
            self.target = {}
        if self.cache is None:
            self.cache = {}
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = {}
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "backend": self.backend,
            "toolchain": self.toolchain,
            "target": self.target,
            "cache": self.cache,
            "inputs": dict(sorted(self.inputs.items())),
            "outputs": dict(sorted(self.outputs.items())),
            "deterministic": self.deterministic,
        }

    def to_json(self, *, pretty: bool = False) -> str:
        indent = 2 if pretty else None
        separators = None if pretty else (",", ":")
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
        )


class ProvenanceRecorder:
    """Records build provenance inside isolated, scoped directories."""

    def __init__(
        self,
        *,
        cache_root: Path | str,
        scope_key: str | None = None,
        backend: str = "python",
        scope_dimensions: dict[str, str] | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.backend = backend
        self.scope_dimensions = dict(
            scope_dimensions or default_scope_dimensions(backend=backend)
        )
        if scope_key is None:
            self.scope_key = ci_compute_cache_scope_key(
                os_name=self.scope_dimensions["os"],
                arch=self.scope_dimensions["arch"],
                compiler_name=self.scope_dimensions["compiler"],
                compiler_version=self.scope_dimensions["compiler_version"],
                extra={"backend": backend},
            )
        else:
            self.scope_key = scope_key
        self.provenance_dir = self.cache_root / "provenance" / self.scope_key
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Toolchain / target discovery
    # ------------------------------------------------------------------

    def _capture_toolchain(self) -> dict[str, Any]:
        toolchain: dict[str, Any] = {}
        toolchain["python"] = {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "available": True,
        }
        for binary in ("rustc", "cargo"):
            version, available = _toolchain_version(binary)
            toolchain[binary] = {"version": version, "available": available}
        zig_version, zig_available = _toolchain_version("zig")
        toolchain["zig"] = {
            "version": zig_version,
            "available": zig_available,
        }
        return toolchain

    def _capture_target(self) -> dict[str, Any]:
        return {
            "machine": platform.machine() or "unknown",
            "system": platform.system() or "unknown",
            "processor": platform.processor() or "unknown",
            "python_platform": sys.platform,
        }

    # ------------------------------------------------------------------
    # Isolated environment
    # ------------------------------------------------------------------

    def _isolated_environment(self, *, work_dir: Path) -> dict[str, str]:
        """Build an isolated environment for one build run.

        Sets an isolated ``CARGO_TARGET_DIR`` and isolated Zig cache directories
        so the build never touches the repository or another run's cache.
        """
        env = os.environ.copy()
        target_dir = ci_isolate_cargo_target_dir(
            profile="provenance",
            cache_root=self.cache_root / "cargo",
            scope_key=self.scope_key,
            env=env,
        )
        zig_cache = self.cache_root / "zig" / self.scope_key
        (zig_cache / "local").mkdir(parents=True, exist_ok=True)
        (zig_cache / "global").mkdir(parents=True, exist_ok=True)
        env["ZIG_LOCAL_CACHE_DIR"] = str(zig_cache / "local")
        env["ZIG_GLOBAL_CACHE_DIR"] = str(zig_cache / "global")
        env["CARGO_TARGET_DIR"] = str(target_dir)
        # Force deterministic, offline-ish builds where possible.
        env.setdefault("CARGO_NET_OFFLINE", "1")
        env.setdefault("RUSTC_BOOTSTRAP", "1")
        return env

    def _cache_record(self, *, work_dir: Path) -> dict[str, Any]:
        return {
            "scope_key": self.scope_key,
            "scope_dimensions": dict(self.scope_dimensions),
            "cache_root": str(self.cache_root),
            "provenance_dir": str(self.provenance_dir),
            "cargo_target_dir": str(
                self.cache_root / "cargo" / self.scope_key / "provenance"
            ),
            "zig_local_cache": str(self.cache_root / "zig" / self.scope_key / "local"),
            "zig_global_cache": str(
                self.cache_root / "zig" / self.scope_key / "global"
            ),
            "work_dir": str(work_dir),
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        build_fn: BuildFn,
        inputs: dict[str, str],
        *,
        run_label: str | None = None,
    ) -> BuildProvenance:
        """Run *build_fn* in an isolated environment and record its provenance.

        ``inputs`` maps a logical name to source text.  Each input is hashed,
        *build_fn* is invoked with ``(inputs, env, output_dir)`` and must return
        the list of output artifact paths it produced.  Every output is hashed.
        The full record is written to JSON under the provenance directory and
        returned.
        """
        work_dir = Path(tempfile.mkdtemp(prefix=f"polyglot-provenance-{self.backend}-"))
        output_dir = work_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_hashes: dict[str, str] = {}
        for name, text in inputs.items():
            input_hashes[name] = _sha256_text(text)

        env = self._isolated_environment(work_dir=work_dir)
        try:
            output_paths = build_fn(inputs, env, output_dir)
        finally:
            pass

        output_hashes: dict[str, str] = {}
        for path in output_paths:
            path = Path(path)
            key = str(path)
            if path.is_file():
                output_hashes[key] = _sha256_file(path)
            elif path.is_dir():
                # Hash the directory's files recursively, stable order.
                combined = hashlib.sha256()
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file():
                        combined.update(file_path.read_bytes())
                output_hashes[key] = combined.hexdigest()

        provenance = BuildProvenance(
            run_id=str(uuid.uuid4()),
            timestamp=time.time(),
            backend=self.backend,
            toolchain=self._capture_toolchain(),
            target=self._capture_target(),
            cache=self._cache_record(work_dir=work_dir),
            inputs=input_hashes,
            outputs=output_hashes,
            deterministic=True,
        )

        record_path = self.provenance_dir / f"{provenance.run_id}.json"
        record_path.write_text(provenance.to_json(pretty=True), encoding="utf-8")

        # Clean up the scratch work directory; never leave repo artifacts.
        import shutil as _shutil

        _shutil.rmtree(work_dir, ignore_errors=True)

        return provenance


def record_build_provenance(
    *,
    cache_root: Path | str,
    build_fn: BuildFn,
    inputs: dict[str, str],
    scope_key: str | None = None,
    backend: str = "python",
    scope_dimensions: dict[str, str] | None = None,
    run_label: str | None = None,
) -> BuildProvenance:
    """Convenience: record one build's provenance."""
    recorder = ProvenanceRecorder(
        cache_root=cache_root,
        scope_key=scope_key,
        backend=backend,
        scope_dimensions=scope_dimensions,
    )
    return recorder.record(build_fn, inputs, run_label=run_label)
