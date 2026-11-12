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
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ast_nodes import Program
from backends.base import BackendBase
from backends.errors import InvalidIntegerParseLiteralError
from backends.rust.codegen import RustCodegen

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

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    @property
    def error_line(self) -> int | None:
        return self.rust_line

    @property
    def error_col(self) -> int | None:
        return self.rust_col

    @property
    def rust_message(self) -> str:
        return self.raw_stderr

    def __str__(self) -> str:
        location = ""
        if self.rust_line is not None:
            location = f" at line {self.rust_line}"
            if self.rust_col is not None:
                location += f":{self.rust_col}"
        code = f" [{self.error_code}]" if self.error_code else ""
        return f"Rust compile error: {self.message}{code}{location}"

    def pretty(self) -> str:
        lines = [str(self)]
        if self.source_line is not None:
            lines.append(
                f"Polyglot source at {self.source_line}:"
                f"{self.source_col or 1}"
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
        lines.append("")
        lines.append("Rust stderr:")
        lines.append(self.rust_message)
        return "\n".join(lines)


class TypeErrorForBoundary(RustCompileError):
    """Pre-Cargo error for an implicit integer boundary conversion."""

    def __init__(
        self,
        source_type: str,
        expected_type: str,
        line: int,
        col: int,
        source_line_text: str | None = None,
    ) -> None:
        message = (
            f"error: type mismatch at line {line}, column {col}\n"
            f"  found:    {source_type}\n"
            f"  expected: {expected_type}\n"
            f"  help: write `as {expected_type}` here\n"
            "  note: `as` may wrap or truncate the value; check the range first"
        )
        super().__init__(
            message=message,
            source_line=line,
            source_col=col,
            source_line_text=source_line_text,
        )
        self.source_type = source_type
        self.expected_type = expected_type
        self.line = line
        self.col = col

    def __str__(self) -> str:
        return self.message


_ERROR_HEAD_RE = re.compile(
    r"(?m)^\s*error(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>[^\n]+)"
)
_ERROR_LOCATION_RE = re.compile(
    r"(?m)^\s*-->\s*src/lib\.rs:(?P<line>\d+):(?P<col>\d+)\s*$"
)
_WARNING_RE = re.compile(
    r"(?m)^\s*warning(?:\[[^\]]+\])?:\s*(?P<msg>[^\n]+)"
)
_DIAGNOSTIC_START_RE = re.compile(
    r"^(?:error(?:\[[^\]]+\])?:|warning(?:\[[^\]]+\])?:)"
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
        self.cache_dir = Path(
            cache_dir or Path.home() / ".cache" / "polyglot-native"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.cache_dir.chmod(0o700)
        except OSError:
            pass

    # ---- public API --------------------------------------------------------

    def compile(self, program: Program) -> dict[str, Any]:
        self._check_trust()
        codegen = RustCodegen()
        try:
            source = codegen.generate(program)
        except InvalidIntegerParseLiteralError as error:
            raise RustCompileError(
                message=f"invalid literal for i64 parse: {error.value!r}",
                source_line=error.line,
                source_col=error.col,
                source_line_text=program.source_lines.get(error.line),
            ) from error
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        build_dir = self.cache_dir / key

        if not self._is_built(build_dir):
            self._build(
                source,
                build_dir,
                codegen.line_map,
                codegen.source_line_text_map,
            )

        ext = self._find_extension(build_dir)
        mod = self._load_extension(ext)
        return mod.__dict__

    # ---- build steps -------------------------------------------------------

    def _is_built(self, build_dir: Path) -> bool:
        release = build_dir / "target" / "release"
        if not release.exists():
            return False
        return (
            any(release.glob("*.so"))
            or any(release.glob("*.dylib"))
            or any(release.glob("*.pyd"))
        )

    def _build(
        self,
        source: str,
        build_dir: Path,
        line_map: dict[int, tuple[int, int]],
        source_line_text_map: dict[int, str] | None = None,
    ) -> None:
        first_build = not build_dir.exists()
        if first_build:
            build_dir.mkdir(parents=True)

        (build_dir / "Cargo.toml").write_text(
            self._cargo_toml(), encoding="utf-8"
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
                result.stdout, result.stderr, source, line_map,
                source_line_text_map,
            )

    def _cargo_toml(self) -> str:
        return """\
[package]
name = "polyglot_native"
version = "0.1.0"
edition = "2021"

[lib]
name = "polyglot_native"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.26", features = ["extension-module"] }
futures = "0.3"
"""

    # ---- load steps --------------------------------------------------------

    def _find_extension(self, build_dir: Path) -> Path:
        release = build_dir / "target" / "release"
        for pattern in ("*.so", "*.pyd", "*.dylib"):
            matches = list(release.glob(pattern))
            if matches:
                return matches[0]
        raise RuntimeError(f"No extension module found in {release}")

    def _load_extension(self, path: Path):
        mod_name = "polyglot_native"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        loader = importlib.machinery.ExtensionFileLoader(
            mod_name, str(path)
        )
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
    if shutil.which("sccache"):
        env["RUSTC_WRAPPER"] = "sccache"
    return env


def _parse_cargo_error(
    stdout: str,
    stderr: str,
    source: str,
    line_map: dict[int, tuple[int, int]] | None = None,
    source_line_text_map: dict[int, str] | None = None,
) -> RustCompileError:
    combined = _deduplicate_error_blocks(
        f"{stdout}\n{stderr}".strip()
    )
    head = _ERROR_HEAD_RE.search(combined)
    location = _ERROR_LOCATION_RE.search(
        combined, head.end() if head is not None else 0,
    )
    rust_line = int(location.group("line")) if location else None
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
