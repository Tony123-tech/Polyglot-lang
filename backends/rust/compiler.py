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
import subprocess
import sys
from pathlib import Path
from typing import Any

from ast_nodes import Program
from backends.base import Backend
from backends.rust.codegen import RustCodegen

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RustCompileError(RuntimeError):
    """Raised when cargo build fails."""

    def __init__(
        self,
        rust_message: str,
        generated_source: str,
        error_line: int | None = None,
        error_col: int | None = None,
        error_code: str | None = None,
    ) -> None:
        location = ""
        if error_line is not None:
            location = f" at line {error_line}"
            if error_col is not None:
                location += f":{error_col}"
        code = f" [{error_code}]" if error_code else ""
        summary = f"Rust compile error{code}{location}"
        super().__init__(summary)
        self.rust_message = rust_message
        self.generated_source = generated_source
        self.error_line = error_line
        self.error_col = error_col
        self.error_code = error_code

    def pretty(self) -> str:
        lines = [str(self)]
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


_ERROR_RE = re.compile(
    r"error(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>[^\n]+)\n"
    r"\s*-->\s*(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+)"
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class NativeBackend(Backend):
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
        source = RustCodegen().generate(program)
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        build_dir = self.cache_dir / key

        if not self._is_built(build_dir):
            self._build(source, build_dir)

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

    def _build(self, source: str, build_dir: Path) -> None:
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
            raise _parse_cargo_error(result.stdout, result.stderr, source)

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
    return env


def _parse_cargo_error(
    stdout: str, stderr: str, source: str
) -> RustCompileError:
    match = _ERROR_RE.search(stderr)
    if match is None:
        return RustCompileError(
            rust_message=f"{stdout}\n{stderr}".strip(),
            generated_source=source,
        )
    return RustCompileError(
        rust_message=stderr.strip(),
        generated_source=source,
        error_line=int(match.group("line")),
        error_col=int(match.group("col")),
        error_code=match.group("code"),
    )
