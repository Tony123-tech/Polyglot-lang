"""Deterministic build verification.

Verifies that repeated builds with identical inputs and isolated caches produce
deterministic outputs and zero cross-run state leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .recorder import (
    BuildFn,
    BuildProvenance,
    record_build_provenance,
)

__all__ = [
    "BuildDeterminismResult",
    "BuildIntegrityResult",
    "verify_cache_isolation",
    "verify_deterministic_build",
    "verify_provenance_records",
]


class BuildDeterminismResult:
    """Outcome of a determinism verification run."""

    def __init__(
        self,
        *,
        clean: bool,
        runs: int,
        fingerprints: list[str],
        inputs_hash: str,
        records: list[BuildProvenance],
        message: str = "",
    ) -> None:
        self.clean = clean
        self.runs = runs
        self.fingerprints = fingerprints
        self.inputs_hash = inputs_hash
        self.records = records
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "runs": self.runs,
            "fingerprints": self.fingerprints,
            "inputs_hash": self.inputs_hash,
            "message": self.message,
            "records": [record.to_dict() for record in self.records],
        }


class BuildIntegrityResult:
    """Outcome of a cache-isolation verification run."""

    def __init__(
        self,
        *,
        clean: bool,
        scope_keys: list[str],
        records: list[BuildProvenance],
        message: str = "",
    ) -> None:
        self.clean = clean
        self.scope_keys = scope_keys
        self.records = records
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "scope_keys": self.scope_keys,
            "message": self.message,
            "records": [record.to_dict() for record in self.records],
        }


def _outputs_fingerprint(record: BuildProvenance) -> str:
    """Stable fingerprint of one record's outputs (path-independent)."""
    import hashlib

    payload = "|".join(
        f"{Path(path).name}={digest}" for path, digest in sorted(record.outputs.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_deterministic_build(
    *,
    cache_root: Path | str,
    build_fn: BuildFn,
    inputs: dict[str, str],
    backend: str = "python",
    runs: int = 2,
    scope_key: str | None = None,
) -> BuildDeterminismResult:
    """Run *build_fn* *runs* times with identical inputs and isolated caches.

    Each run gets a fresh isolated cache scope.  All output hashes must match
    across runs (deterministic outputs), and no run may observe another run's
    cache state (zero cross-run leakage).

    Returns a :class:`BuildDeterminismResult`; ``clean`` is False on any
    divergence or leakage.
    """
    runs = max(runs, 1)
    records: list[BuildProvenance] = []
    fingerprints: list[str] = []

    for index in range(runs):
        run_scope = scope_key if scope_key is not None else f"{backend}-run-{index}"
        record = record_build_provenance(
            cache_root=cache_root,
            build_fn=build_fn,
            inputs=inputs,
            backend=backend,
            scope_key=run_scope,
        )
        records.append(record)
        fingerprints.append(_outputs_fingerprint(record))

    clean = True
    messages: list[str] = []

    if len(set(fingerprints)) != 1:
        clean = False
        messages.append(
            f"non-deterministic outputs: {len(set(fingerprints))} distinct "
            f"fingerprints across {runs} run(s)"
        )

    # Cross-run state leakage: every run must use a distinct, isolated cache
    # directory and must not reuse another run's outputs.
    cache_dirs = [record.cache["cargo_target_dir"] for record in records]
    if len(set(cache_dirs)) != len(cache_dirs):
        clean = False
        messages.append("cache directories are not isolated across runs")

    # Inputs must be identical across runs.
    input_fingerprints = [
        "|".join(f"{k}={v}" for k, v in sorted(record.inputs.items()))
        for record in records
    ]
    if len(set(input_fingerprints)) != 1:
        clean = False
        messages.append("inputs differ across runs (determinism violated)")

    message = "clean" if clean else "; ".join(messages)
    return BuildDeterminismResult(
        clean=clean,
        runs=runs,
        fingerprints=fingerprints,
        inputs_hash=input_fingerprints[0] if input_fingerprints else "",
        records=records,
        message=message,
    )


def verify_cache_isolation(
    *,
    cache_root: Path | str,
    build_fn: BuildFn,
    inputs: dict[str, str],
    backend: str = "python",
    scope_keys: list[str] | None = None,
) -> BuildIntegrityResult:
    """Verify that builds with different scope keys do not leak cache state.

    Each scope key gets its own isolated cache.  Outputs must be deterministic
    per scope (same inputs -> same outputs) and scopes must not share cache
    directories.
    """
    keys = list(scope_keys or [f"{backend}-scope-a", f"{backend}-scope-b"])
    records: list[BuildProvenance] = []
    fingerprints: list[str] = []

    for key in keys:
        record = record_build_provenance(
            cache_root=cache_root,
            build_fn=build_fn,
            inputs=inputs,
            backend=backend,
            scope_key=key,
        )
        records.append(record)
        fingerprints.append(_outputs_fingerprint(record))

    clean = True
    messages: list[str] = []

    # Same inputs across scopes must produce identical outputs.
    if len(set(fingerprints)) != 1:
        clean = False
        messages.append("outputs differ across isolated scopes with identical inputs")

    # Each scope must have a distinct cache directory.
    cache_dirs = [record.cache["cargo_target_dir"] for record in records]
    if len(set(cache_dirs)) != len(cache_dirs):
        clean = False
        messages.append("cache directories are shared across scopes")

    message = "clean" if clean else "; ".join(messages)
    return BuildIntegrityResult(
        clean=clean,
        scope_keys=keys,
        records=records,
        message=message,
    )


def verify_provenance_records(records: list[BuildProvenance]) -> dict[str, Any]:
    """Validate a list of recorded provenance records structurally.

    Checks that every record has a run_id, timestamp, toolchain, target, cache
    scope key, inputs, and outputs.  Returns a summary dict.
    """
    import hashlib

    issues: list[str] = []
    for index, record in enumerate(records):
        if not record.run_id:
            issues.append(f"record {index}: missing run_id")
        if record.timestamp <= 0:
            issues.append(f"record {index}: invalid timestamp")
        if not record.toolchain:
            issues.append(f"record {index}: missing toolchain")
        if not record.target:
            issues.append(f"record {index}: missing target")
        if not record.cache.get("scope_key"):
            issues.append(f"record {index}: missing cache scope key")
        if not record.inputs:
            issues.append(f"record {index}: no inputs recorded")
        if not record.outputs:
            issues.append(f"record {index}: no outputs recorded")

    payload = "|".join(
        f"{i}:{record.run_id}:{record.cache.get('scope_key')}"
        for i, record in enumerate(records)
    )
    return {
        "record_count": len(records),
        "issues": issues,
        "clean": not issues,
        "fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
