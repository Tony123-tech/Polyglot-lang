"""Self-synthesizing compiler foundations.

This is a *controlled* meta-compiler, not an unbounded self-modifying system.
It can:

* collect compile-failure and runtime-profile observations;
* synthesize a candidate optimization pass from a small, auditable template
  library (or a caller-supplied strategy source);
* validate the candidate in an isolated subprocess against a corpus of
  programs, comparing observable results with the unoptimized baseline;
* keep a content-addressed, versioned history with optional git commits; and
* promote or roll back a strategy explicitly.

It deliberately does **not** rewrite the running translator/codegen in place.
A candidate is a normal Python source file.  Promotion is an explicit,
auditable step.  This preserves the compiler's own integrity while still
providing the feedback loop and sandboxed evolution mechanics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ast_nodes import Node
from runtime.diagnostics import PillarError

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# AST map
# ---------------------------------------------------------------------------


def map_ast(value: Any, transform: Any) -> Any:
    """Bottom-up structural map over an AST.

    ``transform`` receives each rebuilt node and may return a replacement or
    ``None`` to keep the original.  Non-AST values are left alone.
    """
    if isinstance(value, Node):
        updates: dict[str, Any] = {}
        for info in dataclasses.fields(value):
            if info.name in {"line", "col", "span"}:
                continue
            updates[info.name] = map_ast(getattr(value, info.name), transform)
        rebuilt = dataclasses.replace(value, **updates)
        replacement = transform(rebuilt)
        return rebuilt if replacement is None else replacement
    if isinstance(value, list):
        return [map_ast(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_ast(item, transform) for item in value)
    if isinstance(value, dict):
        return {key: map_ast(item, transform) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Auditable pass templates
# ---------------------------------------------------------------------------


TEMPLATES: dict[str, str] = {
    "fold_add_zero": """\
from ast_nodes import Binary, LitInt


def transform(node):
    if isinstance(node, Binary) and node.op == "+":
        if isinstance(node.right, LitInt) and node.right.value == 0:
            return node.left
        if isinstance(node.left, LitInt) and node.left.value == 0:
            return node.right
    return None
""",
    "fold_mul_one": """\
from ast_nodes import Binary, LitInt


def transform(node):
    if isinstance(node, Binary) and node.op == "*":
        if isinstance(node.right, LitInt) and node.right.value == 1:
            return node.left
        if isinstance(node.left, LitInt) and node.left.value == 1:
            return node.right
    return None
""",
    "fold_if_literal": """\
from ast_nodes import Block, IfExpr, LitBool


def transform(node):
    if isinstance(node, IfExpr) and isinstance(node.cond, LitBool):
        branch = node.then_block if node.cond.value else node.else_branch
        if isinstance(branch, Block):
            return branch
        if branch is None:
            return Block(node.line, node.col, [], None)
        return branch
    return None
""",
}

_TEMPLATE_FOR_OBSERVATION = {
    "compile_failure": ("fold_add_zero", "fold_mul_one", "fold_if_literal"),
    "slow_function": ("fold_if_literal", "fold_mul_one", "fold_add_zero"),
}

_IDENTITY = "def transform(node):\n    return None\n"


# ---------------------------------------------------------------------------
# Observations and candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    kind: str
    detail: str
    value: float = 0.0


@dataclass(frozen=True)
class PassCandidate:
    name: str
    version: int
    template: str
    source: str
    parent: int | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def filename(self) -> str:
        return f"v{self.version:04d}.py"


@dataclass
class Evaluation:
    ok: bool
    results: list[dict[str, Any]] = field(default_factory=list)
    candidate_ms: float = 0.0
    baseline_ms: float = 0.0
    error: str | None = None

    @property
    def faster(self) -> bool:
        return self.ok and self.candidate_ms < self.baseline_ms


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


_RUNNER = r"""
import json
import sys
import time
import traceback

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)

from ast_nodes import Program
from backends.python.translator import PythonBackend
from compiler.evolution import map_ast
from lexer import lex
from parser import parse


def identity(node):
    return None


def run_case(source, transform):
    program = parse(lex(source))
    started = time.perf_counter()
    mapped = map_ast(program, transform)
    elapsed = (time.perf_counter() - started) * 1000.0
    namespace = PythonBackend(trusted=True).compile(mapped)
    return namespace["main"](), elapsed


def main():
    candidate_path, corpus_path = sys.argv[2], sys.argv[3]
    source = open(candidate_path, encoding="utf-8").read()
    corpus = json.load(open(corpus_path, encoding="utf-8"))
    namespace = {}
    exec(compile(source, candidate_path, "exec"), namespace)
    transform = namespace["transform"]
    results = []
    candidate_ms = 0.0
    baseline_ms = 0.0
    try:
        for case in corpus:
            expected, baseline_elapsed = run_case(case, identity)
            got, elapsed = run_case(case, transform)
            candidate_ms += elapsed
            baseline_ms += baseline_elapsed
            results.append({"expected": expected, "got": got})
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()}))
        return
    print(json.dumps({
        "ok": all(item["expected"] == item["got"] for item in results),
        "results": results,
        "candidate_ms": candidate_ms,
        "baseline_ms": baseline_ms,
    }))


main()
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EvolutionEngine:
    """Observe, synthesize, sandbox-validate, and version compiler passes."""

    def __init__(
        self,
        history_dir: str | Path | None = None,
        *,
        timeout: float = 20.0,
        use_git: bool = False,
    ) -> None:
        self.history_dir = Path(
            history_dir or Path.home() / ".cache" / "polyglot-evolution"
        )
        self.history_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.timeout = timeout
        self.use_git = use_git
        self._observations: list[Observation] = []
        self._runner = _RUNNER

    # -- feedback ----------------------------------------------------------

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def observe_compile_failure(self, stderr: str, code: str = "E004") -> None:
        detail = (stderr or "unknown compiler failure").strip()
        self._observations.append(
            Observation("compile_failure", f"{code}: {detail[:4000]}")
        )

    def observe_profile(self, name: str, milliseconds: float, budget_ms: float) -> None:
        if milliseconds <= budget_ms:
            return
        self._observations.append(
            Observation(
                "slow_function",
                f"{name} took {milliseconds:.3f}ms (budget {budget_ms:.3f}ms)",
                value=milliseconds,
            )
        )

    # -- synthesis ---------------------------------------------------------

    def propose(self, template: str | None = None) -> PassCandidate:
        if not self._observations:
            raise PillarError(
                "E003",
                "no compiler observation is available to drive evolution",
                help_text="call observe_compile_failure() or observe_profile() first",
            )
        if template is None:
            last = self._observations[-1]
            choices = _TEMPLATE_FOR_OBSERVATION.get(last.kind, tuple(TEMPLATES))
            used = {manifest["template"] for manifest in self.history()}
            template = next((name for name in choices if name not in used), choices[0])
        if template not in TEMPLATES:
            raise PillarError(
                "E003",
                f"unknown pass template {template!r}",
                help_text=f"choose one of {sorted(TEMPLATES)}",
            )
        source = TEMPLATES[template]
        self._self_compile(source)
        current = self.current()
        version = 1 if current is None else current.version + 1
        return PassCandidate(
            name=template,
            version=version,
            template=template,
            source=source,
            parent=None if current is None else current.version,
        )

    def _self_compile(self, source: str) -> None:
        try:
            compile(source, "<evolution-candidate>", "exec")
        except SyntaxError as error:
            raise PillarError(
                "E004",
                "synthesized pass does not compile",
                reason=str(error),
                help_text="fix the template before promoting it",
                line=error.lineno or 0,
                col=error.offset or 0,
            ) from error

    # -- sandbox validation ------------------------------------------------

    def evaluate(
        self,
        candidate: PassCandidate,
        corpus: list[str],
    ) -> Evaluation:
        if not corpus:
            raise PillarError(
                "E003",
                "the validation corpus is empty",
                help_text="supply at least one program with a `main` function",
            )
        self._self_compile(candidate.source)
        with tempfile.TemporaryDirectory(prefix="polyglot-evolution-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate.py"
            corpus_path = tmp_path / "corpus.json"
            runner_path = tmp_path / "runner.py"
            candidate_path.write_text(candidate.source, encoding="utf-8")
            corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
            runner_path.write_text(self._runner, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(runner_path),
                        str(ROOT),
                        str(candidate_path),
                        str(corpus_path),
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return Evaluation(
                    ok=False,
                    error=f"candidate timed out after {self.timeout:g}s",
                )
            if completed.returncode != 0:
                return Evaluation(
                    ok=False,
                    error=completed.stderr.strip() or completed.stdout.strip(),
                )
            try:
                payload = json.loads(completed.stdout.strip())
            except json.JSONDecodeError:
                return Evaluation(
                    ok=False,
                    error=f"invalid sandbox output: {completed.stdout.strip()[:2000]}",
                )
            return Evaluation(
                ok=bool(payload.get("ok")),
                results=list(payload.get("results") or []),
                candidate_ms=float(payload.get("candidate_ms") or 0.0),
                baseline_ms=float(payload.get("baseline_ms") or 0.0),
                error=payload.get("error"),
            )

    # -- history -----------------------------------------------------------

    def promote(self, candidate: PassCandidate, evaluation: Evaluation) -> int:
        if not evaluation.ok:
            raise PillarError(
                "E002",
                "cannot promote an unverified compiler pass",
                reason=evaluation.error or "the sandbox corpus produced a mismatch",
                help_text="fix the candidate or strengthen the corpus",
            )
        if evaluation.error:
            raise PillarError(
                "E004",
                "sandbox reported an error for the candidate",
                reason=evaluation.error,
            )
        self.history_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        source_path = self.history_dir / candidate.filename
        manifest_path = self.history_dir / f"v{candidate.version:04d}.json"
        source_path.write_text(candidate.source, encoding="utf-8")
        manifest = {
            "name": candidate.name,
            "version": candidate.version,
            "template": candidate.template,
            "parent": candidate.parent,
            "digest": candidate.digest,
            "created_at": candidate.created_at,
            "candidate_ms": evaluation.candidate_ms,
            "baseline_ms": evaluation.baseline_ms,
            "corpus_results": evaluation.results,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.history_dir / "current.json").write_text(
            json.dumps({"version": candidate.version}), encoding="utf-8"
        )
        if self.use_git:
            self._git_commit(candidate)
        return candidate.version

    def current(self) -> PassCandidate | None:
        pointer = self.history_dir / "current.json"
        if not pointer.is_file():
            return None
        try:
            version = int(json.loads(pointer.read_text(encoding="utf-8"))["version"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise PillarError(
                "E008",
                "evolution history pointer is malformed",
                reason=str(error),
                help_text="delete current.json and re-promote a candidate",
            ) from error
        return self._load(version)

    def history(self) -> list[dict[str, Any]]:
        if not self.history_dir.is_dir():
            return []
        manifests: list[dict[str, Any]] = []
        for path in sorted(self.history_dir.glob("v*.json")):
            try:
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                raise PillarError(
                    "E008",
                    f"corrupt evolution manifest: {path.name}",
                    reason=str(error),
                    help_text="remove the corrupt manifest and re-promote",
                ) from error
        return manifests

    def rollback(self, version: int) -> PassCandidate:
        candidate = self._load(version)
        (self.history_dir / "current.json").write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
        return candidate

    def _load(self, version: int) -> PassCandidate:
        manifest_path = self.history_dir / f"v{version:04d}.json"
        source_path = self.history_dir / f"v{version:04d}.py"
        if not manifest_path.is_file() or not source_path.is_file():
            raise PillarError(
                "E006",
                f"compiler pass version {version} is not in the history",
                help_text="inspect history() for available versions",
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = source_path.read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != manifest.get("digest"):
            raise PillarError(
                "E008",
                f"compiler pass version {version} failed its digest check",
                help_text="the history file was modified outside the engine",
            )
        return PassCandidate(
            name=manifest["name"],
            version=manifest["version"],
            template=manifest["template"],
            source=source,
            parent=manifest.get("parent"),
            created_at=float(manifest.get("created_at") or 0.0),
        )

    def _git_commit(self, candidate: PassCandidate) -> None:
        git = shutil.which("git")
        if git is None:
            return
        if not (self.history_dir / ".git").is_dir():
            subprocess.run(
                [git, "init", "-q", str(self.history_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
        subprocess.run(
            [git, "-C", str(self.history_dir), "add", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            [
                git,
                "-C",
                str(self.history_dir),
                "commit",
                "-q",
                "-m",
                f"evolution: promote {candidate.name} v{candidate.version:04d}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )


def meta_cache_line_struct_pack(
    fields: Mapping[str, int],
    *,
    cache_line: int = 64,
) -> tuple[tuple[str, int], ...]:
    """Model a cache-friendly field order from declared field sizes.

    The real mechanism would need L1/L2 miss telemetry and a target ABI; this
    deterministic model only provides the layout plan.  Oversized fields are
    kept whole, and ties are broken by name for reproducibility.
    """
    if cache_line < 1:
        raise PillarError("E003", "cache_line must be positive")
    ordered = sorted(fields.items(), key=lambda item: (-item[1], item[0]))
    return tuple((name, size) for name, size in ordered)


__all__ = [
    "TEMPLATES",
    "Evaluation",
    "EvolutionEngine",
    "Observation",
    "PassCandidate",
    "map_ast",
    "meta_cache_line_struct_pack",
]
