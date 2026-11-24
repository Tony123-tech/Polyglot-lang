"""
Native backend: compile Polyglot AST to a real Rust extension module.

Pipeline:
    Program -> RustCodegen -> lib.rs
    lib.rs -> cargo build --release -> .so / .dylib / .pyd
    .so -> importlib -> namespace dict

Caches builds by SHA-256 of the generated source. `target/` is preserved
across builds for incremental compilation. Rust compile errors are parsed
and re-raised as `RustCompileError` with source line mapping.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from backends.base import BackendBase
from backends.errors import InvalidIntegerParseLiteralError
from backends.rust.codegen import InvalidCastError, RustCodegen
from compiler.core.ast_nodes import Program

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass
class RustCompileError(RuntimeError):
    """Structured Cargo error with generated and source locations."""

    message: str
    rust_line: int | None = None
    rust_col: int | None = None
    source_line: int | None = None
    source_col: int | None = None
    source_line_text: str | None = None
    raw_stderr: str = ""
    generated_source: str = ""
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    span: tuple[int, int] | None = None
    code: str = "E004"

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.format())

    @property
    def error_line(self) -> int | None:
        return self.rust_line

    @property
    def error_col(self) -> int | None:
        return self.rust_col

    @property
    def rust_message(self) -> str:
        return self.raw_stderr

    def format(self) -> str:
        line = self.source_line if self.source_line is not None else self.rust_line
        col = self.source_col if self.source_col is not None else self.rust_col
        at = (
            f"at: line {line}, column {col}"
            if line is not None and col is not None
            else "at: <unknown>"
        )
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        reason = (
            self.message.splitlines()[0] if self.message else "rust compilation failed"
        )
        help_text = "inspect the generated Rust and fix the reported issue"
        note = "RustCompileError"
        if self.error_code:
            note += f" [{self.error_code}]"
        return (
            f"error[{self.code}]: {reason}\n"
            f"  reason: {reason}\n"
            f"  help: {help_text}\n"
            f"  note: {note}\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()

    def pretty(self) -> str:
        lines = [self.format()]
        if self.source_line is not None:
            lines.append(
                f"Polyglot source at {self.source_line}:{self.source_col or 1}"
            )
            if self.source_line_text is not None:
                lines.append(f"  {self.source_line_text}")
        if self.error_line is not None:
            src_lines = self.generated_source.splitlines()
            start = max(0, self.error_line - 3)
            end = min(len(src_lines), self.error_line + 2)
            for i in range(start, end):
                marker = ">>" if i + 1 == self.error_line else "  "
                lines.append(f"{marker} {i + 1:4d} | {src_lines[i]}")
        if self.raw_stderr:
            lines.append("")
            lines.append("Rust stderr:")
            if self.source_line is None:
                lines.extend(self.rust_message.splitlines()[:10])
            else:
                lines.append(self.rust_message)
        return "\n".join(lines)


class TypeErrorForBoundary(RustCompileError):
    """Pre-Cargo error for an implicit integer boundary conversion."""

    code = "E001"

    def __init__(
        self,
        source_type: str,
        expected_type: str,
        line: int,
        col: int,
        source_line_text: str | None = None,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.source_type = source_type
        self.expected_type = expected_type
        self.line = line
        self.col = col
        self.span = span
        message = self.format_boundary()
        super().__init__(
            message=message,
            source_line=line,
            source_col=col,
            source_line_text=source_line_text,
            span=span,
            code="E001",
        )

    def format_boundary(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{self.code}]: type mismatch at line {self.line}, column {self.col}\n"
            f"  found:    {self.source_type}\n"
            f"  expected: {self.expected_type}\n"
            f"  help: write `as {self.expected_type}` here\n"
            "  note: `as` may wrap or truncate the value; check the range first\n"
            f"  {at}"
        )

    def format(self) -> str:
        return self.format_boundary()

    def __str__(self) -> str:
        return self.format_boundary()


_ERROR_HEAD_RE = re.compile(r"(?m)^\s*error(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>[^\n]+)")
_ERROR_LOCATION_RE = re.compile(
    r"(?m)^\s*-->\s*(?P<file>[^\n]+?):(?P<line>\d+):(?P<col>\d+)\s*$"
)
_WARNING_RE = re.compile(r"(?m)^\s*warning(?:\[[^\]]+\])?:\s*(?P<msg>[^\n]+)")
_DIAGNOSTIC_START_RE = re.compile(r"^(?:error(?:\[[^\]]+\])?:|warning(?:\[[^\]]+\])?:)")
# rustc colours its diagnostics when stdout is a terminal or when
# CARGO_TERM_COLOR says so, which is always the case on a GitHub runner. The
# escape sequences sit in front of the `error[E0308]:` marker that the regexes
# above anchor on, so without this the whole diagnostic is missed and the
# failure degrades to a bare "Cargo build failed".
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

_EXTENSION_PATTERNS = ("*.so", "*.pyd", "*.dylib")
_MACOS_CARGO_CONFIG = (
    "[target.aarch64-apple-darwin]\n"
    'rustflags = ["-C", "link-arg=-undefined", '
    '"-C", "link-arg=dynamic_lookup"]\n'
    "\n"
    "[target.x86_64-apple-darwin]\n"
    'rustflags = ["-C", "link-arg=-undefined", '
    '"-C", "link-arg=dynamic_lookup"]\n'
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class NativeBackend(BackendBase):
    name = "native"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        trusted: bool = True,
    ) -> None:
        super().__init__(trusted=trusted)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "polyglot-native")
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.cache_dir.chmod(0o700)
        except OSError:
            pass
        # Diagnostics for the most recent build (useful to tests/tools).
        self.last_build: dict[str, Any] = {}

    # ---- public API --------------------------------------------------------

    def compile(
        self,
        program: Program,
        module_name: str = "polyglot_native",
        *,
        incremental: bool = False,
        cache_scope: str | None = None,
    ) -> dict[str, Any]:
        self._check_trust()
        extension = self.build(
            program,
            module_name=module_name,
            incremental=incremental,
            cache_scope=cache_scope,
        )
        mod = self._load_extension(extension, module_name=module_name)
        return mod.__dict__

    def build(
        self,
        program: Program,
        module_name: str = "polyglot_native",
        *,
        incremental: bool = False,
        cache_scope: str | None = None,
    ) -> Path:
        """Compile ``program`` and return the built extension module path.

        Builds are content-addressed by the full generated Rust source. With
        ``incremental=True`` a stable per-scope working directory is kept so
        Cargo can reuse per-function compilation artifacts when only one
        top-level item changes.
        """
        self._check_trust()
        from backends.move_check import check_moves

        check_moves(program)
        codegen = RustCodegen(module_name=module_name)
        try:
            source = codegen.generate(program)
        except InvalidIntegerParseLiteralError as error:
            raise RustCompileError(
                message=f"invalid literal for i64 parse: {error.value!r}",
                source_line=error.line,
                source_col=error.col,
                source_line_text=program.source_lines.get(error.line),
                span=error.span,
                code="E008",
            ) from error
        except InvalidCastError as error:
            raise RustCompileError(
                message=error.message,
                source_line=error.line,
                source_col=error.col,
                source_line_text=program.source_lines.get(error.line),
                code="E008",
            ) from error
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        build_dir = self.cache_dir / key

        if self._is_built(build_dir, module_name):
            self.last_build = {
                "source_hash": key,
                "reused": True,
                "incremental": incremental,
            }
            return self._find_extension(build_dir, module_name=module_name)

        if incremental:
            return self._build_incremental(
                source,
                codegen,
                module_name=module_name,
                key=key,
                cache_scope=cache_scope,
            )

        self._build(
            source,
            build_dir,
            codegen.line_map,
            codegen.source_line_text_map,
            module_name=module_name,
        )
        self.last_build = {
            "source_hash": key,
            "reused": False,
            "incremental": False,
        }
        return self._find_extension(build_dir, module_name=module_name)

    # ---- build steps -------------------------------------------------------

    def _is_built(self, build_dir: Path, module_name: str) -> bool:
        """True only when this cache entry was built for ``module_name``.

        The build directory is seeded with another build's ``target/`` to keep
        dependencies warm, so an extension file may exist without belonging to
        this source. The manifest written after a successful build is the only
        trustworthy signal.
        """
        manifest_path = build_dir / ".polyglot-manifest.json"
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            return False
        if manifest.get("module") != module_name:
            return False
        if manifest.get("source_hash") != build_dir.name:
            return False
        extension = manifest.get("extension")
        if not isinstance(extension, str) or not extension:
            return False
        for base in (build_dir, build_dir / "target" / "release"):
            if (base / extension).is_file():
                return True
        return False

    # ---- incremental build -------------------------------------------------

    def _incremental_scope_dir(self, scope: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope).strip("._-") or "scope"
        digest = hashlib.sha256(scope.encode()).hexdigest()[:8]
        return self.cache_dir / "incremental" / f"{safe[:40]}-{digest}"

    @contextmanager
    def _scope_lock(self, scope_dir: Path):
        scope_dir.mkdir(parents=True, exist_ok=True)
        with open(scope_dir / ".lock", "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _build_incremental(
        self,
        source: str,
        codegen: RustCodegen,
        *,
        module_name: str,
        key: str,
        cache_scope: str | None,
    ) -> Path:
        scope_dir = self._incremental_scope_dir(cache_scope or module_name)
        with self._scope_lock(scope_dir):
            # Another process may have finished this exact build while we
            # waited for the lock.
            build_dir = self.cache_dir / key
            if self._is_built(build_dir, module_name):
                self.last_build = {
                    "source_hash": key,
                    "reused": True,
                    "incremental": True,
                }
                return self._find_extension(build_dir, module_name=module_name)

            manifest_path = scope_dir / ".polyglot-manifest.json"
            previous = _read_json(manifest_path)
            previous_fragments = previous.get("fragments")
            if not isinstance(previous_fragments, dict):
                previous_fragments = {}
            changed = sorted(
                name
                for name, digest in codegen.fragment_hashes.items()
                if previous_fragments.get(name) != digest
            )
            removed = sorted(set(previous_fragments) - set(codegen.fragment_hashes))

            if not (scope_dir / "target").is_dir():
                self._seed_incremental_target(scope_dir)
            self._write_project_files(scope_dir, module_name)
            fragment_starts = self._write_fragments(scope_dir, codegen)
            lib_rs = self._assemble_lib_rs(codegen)
            _write_if_changed(scope_dir / "src" / "lib.rs", lib_rs)

            env = _build_env()
            # sccache refuses to run with CARGO_INCREMENTAL=1. Incremental
            # builds trade wrapper caching for sub-second local rebuilds; the
            # seed step above keeps prebuilt dependencies warm.
            env.pop("RUSTC_WRAPPER", None)
            env["CARGO_INCREMENTAL"] = "1"
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=scope_dir,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                raise _parse_cargo_error(
                    result.stdout,
                    result.stderr,
                    source,
                    codegen.line_map,
                    codegen.source_line_text_map,
                    fragment_starts=fragment_starts,
                )

            extension = self._find_extension(scope_dir, module_name=module_name)
            build_dir.mkdir(parents=True, exist_ok=True)
            artifact = build_dir / extension.name
            temporary = build_dir / f".{extension.name}.{os.getpid()}.tmp"
            shutil.copy2(extension, temporary)
            os.replace(temporary, artifact)

            manifest = {
                "source_hash": key,
                "module": module_name,
                "fragments": dict(codegen.fragment_hashes),
                "extension": extension.name,
            }
            _write_json(build_dir / ".polyglot-manifest.json", manifest)
            _write_json(
                manifest_path,
                {**manifest, "changed": changed, "removed": removed},
            )
            self.last_build = {
                "source_hash": key,
                "reused": False,
                "incremental": True,
                "scope": scope_dir.name,
                "changed": changed,
                "removed": removed,
            }
            return artifact

    def _write_project_files(self, scope_dir: Path, module_name: str) -> None:
        _write_if_changed(scope_dir / "Cargo.toml", self._cargo_toml(module_name))
        if sys.platform == "darwin":
            cargo_dir = scope_dir / ".cargo"
            cargo_dir.mkdir(exist_ok=True)
            _write_if_changed(cargo_dir / "config.toml", _MACOS_CARGO_CONFIG)

    def _write_fragments(self, scope_dir: Path, codegen: RustCodegen) -> dict[str, int]:
        """Write each content-addressed fragment and return diagnostic offsets.

        Fragment files whose hash is already present are left untouched, so
        Cargo sees stable inputs for the functions that did not change.
        """
        fragment_dir = scope_dir / "src" / "fragments"
        fragment_dir.mkdir(parents=True, exist_ok=True)
        starts: dict[str, int] = {}
        for name, text in codegen.fragments.items():
            filename = _fragment_filename(name, codegen.fragment_hashes[name])
            path = fragment_dir / filename
            if not path.exists():
                path.write_text(text + "\n", encoding="utf-8")
            start = codegen.fragment_spans[name][0]
            rel = path.relative_to(scope_dir).as_posix()
            starts[rel] = start
            starts[filename] = start
            starts[path.as_posix()] = start
            starts[str(path.resolve())] = start
        return starts

    @staticmethod
    def _assemble_lib_rs(codegen: RustCodegen) -> str:
        lines = [codegen.header_source, ""]
        for name in codegen.fragments:
            filename = _fragment_filename(name, codegen.fragment_hashes[name])
            lines.append(f'include!("fragments/{filename}");')
        lines.append("")
        return "\n".join(lines) + "\n"

    def _seed_incremental_target(self, scope_dir: Path) -> None:
        """Best-effort copy of an existing ``target/`` to warm dependencies.

        Cargo will still rebuild the crate itself, but prebuilt dependency
        artifacts (PyO3 and friends) are reused. This only runs once per
        incremental scope and is deliberately silent on failure.
        """
        target = scope_dir / "target"
        if target.exists():
            return
        candidates: list[tuple[float, Path]] = []
        search_roots = [self.cache_dir]
        default_cache = Path.home() / ".cache" / "polyglot-native"
        if default_cache != self.cache_dir and default_cache.is_dir():
            search_roots.append(default_cache)
        children: list[Path] = []
        for root in search_roots:
            try:
                children.extend(root.iterdir())
            except OSError:
                continue
        for child in children:
            if child == scope_dir:
                continue
            if child.name == "incremental":
                try:
                    nested = list(child.iterdir())
                except OSError:
                    nested = []
                for path in nested:
                    if path != scope_dir:
                        candidate = _target_candidate(path)
                        if candidate is not None:
                            candidates.append(candidate)
                continue
            candidate = _target_candidate(child)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return
        _, source_target = max(candidates)
        lock_source = source_target.parent / "Cargo.lock"
        if lock_source.is_file() and not (scope_dir / "Cargo.lock").exists():
            try:
                shutil.copy2(lock_source, scope_dir / "Cargo.lock")
            except OSError:
                pass
        seed = scope_dir / ".target-seed"
        shutil.rmtree(seed, ignore_errors=True)
        try:
            if sys.platform == "darwin":
                command = ["cp", "-Rc", str(source_target), str(seed)]
            else:
                command = [
                    "cp",
                    "-a",
                    "--reflink=auto",
                    str(source_target),
                    str(seed),
                ]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except OSError, subprocess.CalledProcessError:
            shutil.rmtree(seed, ignore_errors=True)
            return
        if seed.is_dir() and not target.exists():
            seed.rename(target)
        else:
            shutil.rmtree(seed, ignore_errors=True)

    def _build(
        self,
        source: str,
        build_dir: Path,
        line_map: dict[int, tuple[int, int]],
        source_line_text_map: dict[int, str] | None = None,
        module_name: str = "polyglot_native",
    ) -> None:
        first_build = not build_dir.exists()
        if first_build:
            build_dir.mkdir(parents=True)
        if not (build_dir / "target").is_dir():
            self._seed_incremental_target(build_dir)

        (build_dir / "Cargo.toml").write_text(
            self._cargo_toml(module_name), encoding="utf-8"
        )

        if sys.platform == "darwin":
            cargo_dir = build_dir / ".cargo"
            cargo_dir.mkdir(exist_ok=True)
            (cargo_dir / "config.toml").write_text(
                "[target.aarch64-apple-darwin]\n"
                'rustflags = ["-C", "link-arg=-undefined", '
                '"-C", "link-arg=dynamic_lookup"]\n'
                "\n"
                "[target.x86_64-apple-darwin]\n"
                'rustflags = ["-C", "link-arg=-undefined", '
                '"-C", "link-arg=dynamic_lookup"]\n',
                encoding="utf-8",
            )

        src_dir = build_dir / "src"
        src_dir.mkdir(exist_ok=True)
        (src_dir / "lib.rs").write_text(source, encoding="utf-8")

        env = _build_env()
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            raise _parse_cargo_error(
                result.stdout,
                result.stderr,
                source,
                line_map,
                source_line_text_map,
            )

        # Record that this cache entry really belongs to this source and module.
        # Without it a seeded ``target/`` directory can masquerade as a finished
        # build and hand back another program's extension module.
        extension = _find_extension_path(
            build_dir, module_name
        ) or _find_extension_path(build_dir / "target" / "release", module_name)
        if extension is not None:
            _write_json(
                build_dir / ".polyglot-manifest.json",
                {
                    "source_hash": build_dir.name,
                    "module": module_name,
                    "extension": extension.name,
                },
            )

    def _cargo_toml(self, module_name: str = "polyglot_native") -> str:
        return f"""\
[package]
name = "{module_name}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{module_name}"
crate-type = ["cdylib"]

[dependencies]
pyo3 = {{ version = "0.26", features = ["extension-module"] }}
futures = "0.3"
mio = {{ version = "1", features = ["os-poll", "os-ext"] }}
"""

    # ---- load steps --------------------------------------------------------

    def _find_extension(self, build_dir: Path, module_name: str | None = None) -> Path:
        found = _find_extension_path(build_dir, module_name)
        if found is not None:
            return found
        release = build_dir / "target" / "release"
        found = _find_extension_path(release, module_name)
        if found is not None:
            return found
        raise RuntimeError(f"No extension module found in {release}")

    def _load_extension(self, path: Path, module_name: str = "polyglot_native"):
        mod_name = module_name
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        loader = importlib.machinery.ExtensionFileLoader(mod_name, str(path))
        spec = importlib.util.spec_from_file_location(
            mod_name, str(path), loader=loader
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load extension from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
    env.setdefault("CARGO_PROFILE_RELEASE_DEBUG", "false")
    # Diagnostics are parsed, not read by a person, so ask for no colour: the
    # escape sequences would otherwise depend on where the build ran (a runner
    # exports CARGO_TERM_COLOR=always, a developer's shell does not).
    env["CARGO_TERM_COLOR"] = "never"
    if shutil.which("sccache"):
        env["RUSTC_WRAPPER"] = "sccache"
    return env


def _find_extension_path(
    directory: Path, module_name: str | None = None
) -> Path | None:
    if not directory.is_dir():
        return None
    expected = {module_name, f"lib{module_name}"} if module_name is not None else None
    for pattern in _EXTENSION_PATTERNS:
        matches = sorted(directory.glob(pattern))
        if expected is not None:
            matches = [path for path in matches if path.stem in expected]
        if matches:
            return matches[0]
    return None


def _target_candidate(directory: Path) -> tuple[float, Path] | None:
    target = directory / "target"
    if not target.is_dir():
        return None
    try:
        return (target.stat().st_mtime, target)
    except OSError:
        return None


def _fragment_filename(name: str, digest: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or "fragment"
    return f"{safe[:80]}-{digest}.rs"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lookup_fragment_start(file: str, starts: dict[str, int]) -> int | None:
    normalized = file.replace("\\", "/")
    for candidate in (normalized, normalized.removeprefix("./")):
        if candidate in starts:
            return starts[candidate]
    return starts.get(normalized.rsplit("/", 1)[-1])


def _parse_cargo_error(
    stdout: str,
    stderr: str,
    source: str,
    line_map: dict[int, tuple[int, int]] | None = None,
    source_line_text_map: dict[int, str] | None = None,
    fragment_starts: dict[str, int] | None = None,
) -> RustCompileError:
    combined = _deduplicate_error_blocks(
        _ANSI_ESCAPE_RE.sub("", f"{stdout}\n{stderr}".strip())
    )
    head = _ERROR_HEAD_RE.search(combined)
    location = _ERROR_LOCATION_RE.search(
        combined,
        head.end() if head is not None else 0,
    )
    location_file = location.group("file") if location else None
    rust_line = int(location.group("line")) if location else None
    if rust_line is not None and location_file is not None and fragment_starts:
        fragment_start = _lookup_fragment_start(location_file, fragment_starts)
        if fragment_start is not None:
            rust_line += fragment_start
    rust_col = int(location.group("col")) if location else None
    source_origin = (
        line_map.get(rust_line)
        if line_map is not None and rust_line is not None
        else None
    )
    return RustCompileError(
        message=head.group("msg") if head else "Cargo build failed",
        raw_stderr=combined,
        generated_source=source,
        rust_line=rust_line,
        rust_col=rust_col,
        source_line=source_origin[0] if source_origin else None,
        source_col=source_origin[1] if source_origin else None,
        source_line_text=(
            source_line_text_map.get(rust_line)
            if source_line_text_map is not None and rust_line is not None
            else None
        ),
        error_code=head.group("code") if head else None,
        warnings=[match.group("msg") for match in _WARNING_RE.finditer(combined)],
    )


def _deduplicate_error_blocks(output: str) -> str:
    lines = output.splitlines()
    result: list[str] = []
    seen_errors: set[tuple[str, ...]] = set()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("error"):
            result.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not _DIAGNOSTIC_START_RE.match(lines[end]):
            end += 1
        block = tuple(lines[index:end])
        if block not in seen_errors:
            result.extend(block)
            seen_errors.add(block)
        index = end
    return "\n".join(result)
