"""Hard isolated-build and differential-parity helpers (Frame C, CI half).

Every helper is deterministic and keeps all generated state under a
caller-supplied root outside the repository working tree.  Divergence
artifacts are written only for a genuine parity failure.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runtime.diagnostics import PillarError

Span = tuple[int, int] | None


def _fail(
    code: str, message: str, *, span: Span = None, reason: str | None = None
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text="use a fresh isolated cache root and a fixed seed",
        note="tools.build_ci/hard",
        span=span,
    )


def ci_scoped_backend_environment(
    root: Path | str,
    *,
    backend: str,
    seed: int,
    case_id: str,
    extra: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an isolated environment and path set for one backend run."""
    if backend not in {"python", "native", "wasm", "jit"}:
        raise _fail("E004", f"unknown backend {backend!r}")
    base = Path(root).resolve()
    if not base.is_absolute():
        raise _fail("E004", "cache root must be resolvable to an absolute path")
    scope = hashlib.sha256(f"{backend}:{seed}:{case_id}".encode()).hexdigest()[:16]
    cache = base / backend / scope
    target = cache / "target"
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYTHONPYCACHEPREFIX"] = str(cache / "pycache")
    env["POLYGLOT_CACHE_SCOPE"] = scope
    env["POLYGLOT_WASM_CACHE"] = str(cache / "wasm")
    env["POLYGLOT_JIT_CACHE"] = str(cache / "jit")
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return {
        "backend": backend,
        "seed": seed,
        "case_id": case_id,
        "scope": scope,
        "cache": cache,
        "target_dir": target,
        "env": env,
    }


def ci_seed_reproducible_matrix(
    cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    backends: Sequence[str],
) -> list[dict[str, Any]]:
    """Return a stable backend/case matrix with deterministic scope keys."""
    matrix: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        if not case_id:
            raise _fail("E003", "parity case needs an id")
        for backend in sorted(set(backends)):
            scope = ci_scoped_backend_environment(
                Path("/tmp/polyglot-parity-scope"),
                backend=backend,
                seed=seed,
                case_id=case_id,
            )["scope"]
            matrix.append(
                {
                    "id": case_id,
                    "backend": backend,
                    "seed": seed,
                    "scope": scope,
                    "source": str(case.get("source", "")),
                }
            )
    return matrix


def ci_persist_divergence(
    report: Mapping[str, Any],
    *,
    directory: Path | str,
    case: Mapping[str, Any],
    seed: int,
) -> Path | None:
    """Write ``seed + source + normalized outputs`` only on real divergence."""
    mismatches = report.get("mismatches", ())
    if report.get("passed", True) or not mismatches:
        return None
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    case_id = str(case.get("id", "case"))
    safe_id = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in case_id
    )
    source = str(case.get("source", ""))
    stem = f"{seed}-{safe_id}"
    poly_path = target / f"{stem}.poly"
    json_path = target / f"{stem}.json"
    poly_path.write_text(source, encoding="utf-8")
    payload = {
        "seed": seed,
        "case": case_id,
        "source": source,
        "mismatches": list(mismatches),
    }
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return json_path


__all__ = [
    "ci_persist_divergence",
    "ci_scoped_backend_environment",
    "ci_seed_reproducible_matrix",
]
