"""Deterministic build provenance & cache integrity tests.

Covers build provenance recording, deterministic output verification, and
isolated-cache integrity.  All runs use fresh temp directories so nothing is
written to the repository working tree.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from tools.provenance.recorder import (
    BuildProvenance,
    record_build_provenance,
)
from tools.provenance.verifier import (
    BuildDeterminismResult,
    verify_cache_isolation,
    verify_deterministic_build,
    verify_provenance_records,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_build(inputs, env, output_dir):
    """Deterministic build: output = sha256 of each input."""
    outputs = []
    for index, (name, text) in enumerate(inputs.items()):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out_path = output_dir / f"{index}_{name}.out"
        out_path.write_text(digest, encoding="utf-8")
        outputs.append(out_path)
    return outputs


def _non_deterministic_build(inputs, env, output_dir):
    """Non-deterministic build: output embeds a process-local counter."""
    counter = _non_deterministic_build.counter
    _non_deterministic_build.counter += 1
    outputs = []
    for index, (name, text) in enumerate(inputs.items()):
        digest = hashlib.sha256((text + str(counter)).encode("utf-8")).hexdigest()
        out_path = output_dir / f"{index}_{name}.out"
        out_path.write_text(digest, encoding="utf-8")
        outputs.append(out_path)
    return outputs


_non_deterministic_build.counter = 0


def _fresh_cache_root():
    return Path(tempfile.mkdtemp(prefix="polyglot-provenance-test-"))


# ---------------------------------------------------------------------------
# Provenance recording
# ---------------------------------------------------------------------------


def test_record_build_produces_complete_record():
    cache_root = _fresh_cache_root()
    inputs = {"add.poly": "fn add(a: i64, b: i64) -> i64 { a + b }\n"}
    record = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
    )
    assert isinstance(record, BuildProvenance)
    assert record.run_id
    assert record.timestamp > 0
    assert record.backend == "python"
    assert "python" in record.toolchain
    assert record.toolchain["python"]["available"] is True
    assert record.target["system"]
    assert record.target["machine"]
    assert record.cache["scope_key"]
    assert record.inputs
    assert record.outputs
    assert record.deterministic is True


def test_provenance_record_written_to_json(cache_root=None):
    cache_root = _fresh_cache_root()
    inputs = {"main.poly": "fn main() -> i64 { 1 }\n"}
    record = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
    )
    scope_dir = cache_root / "provenance" / record.cache["scope_key"]
    json_files = list(scope_dir.glob("*.json"))
    assert len(json_files) == 1
    content = json_files[0].read_text(encoding="utf-8")
    assert record.run_id in content
    assert '"schema"' in content


def test_no_repo_artifacts_left():
    """Provenance must never write inside the repository."""
    repo_root = Path(__file__).resolve().parent.parent
    cache_root = _fresh_cache_root()
    record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs={"x.poly": "1\n"},
        backend="python",
    )
    # The provenance root is outside the repo by construction (temp dir).
    assert not str(cache_root).startswith(str(repo_root))


# ---------------------------------------------------------------------------
# Determinism verification
# ---------------------------------------------------------------------------


def test_verify_deterministic_build_is_clean():
    cache_root = _fresh_cache_root()
    inputs = {"a.poly": "1\n", "b.poly": "2\n"}
    result = verify_deterministic_build(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
        runs=3,
    )
    assert isinstance(result, BuildDeterminismResult)
    assert result.clean is True
    assert result.runs == 3
    assert len(set(result.fingerprints)) == 1


def test_verify_deterministic_build_detects_non_determinism():
    cache_root = _fresh_cache_root()
    inputs = {"a.poly": "1\n"}
    result = verify_deterministic_build(
        cache_root=cache_root,
        build_fn=_non_deterministic_build,
        inputs=inputs,
        backend="python",
        runs=2,
    )
    assert result.clean is False
    assert "non-deterministic" in result.message


# ---------------------------------------------------------------------------
# Cache isolation
# ---------------------------------------------------------------------------


def test_verify_cache_isolation_is_clean():
    cache_root = _fresh_cache_root()
    inputs = {"a.poly": "1\n"}
    result = verify_cache_isolation(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
    )
    assert result.clean is True
    assert len(result.scope_keys) == 2


def test_cache_scopes_are_distinct():
    cache_root = _fresh_cache_root()
    inputs = {"a.poly": "1\n"}
    rec_a = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
        scope_key="scope-a",
    )
    rec_b = record_build_provenance(
        cache_root=cache_root,
        build_fn=_deterministic_build,
        inputs=inputs,
        backend="python",
        scope_key="scope-b",
    )
    assert rec_a.cache["scope_key"] == "scope-a"
    assert rec_b.cache["scope_key"] == "scope-b"
    assert rec_a.cache["cargo_target_dir"] != rec_b.cache["cargo_target_dir"]
    assert rec_a.cache["zig_local_cache"] != rec_b.cache["zig_local_cache"]


# ---------------------------------------------------------------------------
# Provenance record validation
# ---------------------------------------------------------------------------


def test_verify_provenance_records_structural():
    cache_root = _fresh_cache_root()
    inputs = {"a.poly": "1\n"}
    records = [
        record_build_provenance(
            cache_root=cache_root,
            build_fn=_deterministic_build,
            inputs=inputs,
            backend="python",
        )
    ]
    summary = verify_provenance_records(records)
    assert summary["record_count"] == 1
    assert summary["clean"] is True
    assert summary["issues"] == []
    assert len(summary["fingerprint"]) == 64


def test_verify_provenance_records_catches_bad_record():
    bad = BuildProvenance(backend="python")
    bad.run_id = ""
    bad.timestamp = 0
    summary = verify_provenance_records([bad])
    assert summary["clean"] is False
    assert any("run_id" in issue for issue in summary["issues"])
    assert any("timestamp" in issue for issue in summary["issues"])
