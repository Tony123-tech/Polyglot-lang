"""Behavioral tests for FRAME 10 functions 91-100.

The harnesses in ``tests/regression_harness.py`` are deliberately not collected
by pytest.  These tests call them directly.  ``SOURCES`` is exported so
``conftest.py`` can compile the native probes once for the optional
``@pytest.mark.native`` checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docs.framework import (
    docs_record_platform_constraints,
    docs_sync_capability_matrix,
    docs_sync_cli_and_backend_flags,
)
from tests.regression_harness import (
    F10_SOURCES,
)
from tests.regression_harness import (
    test_hotpatch_delegation_and_rollback as harness_hotpatch_delegation_and_rollback,
)
from tests.regression_harness import (
    test_isolated_cache_no_contamination as harness_isolated_cache_no_contamination,
)
from tests.regression_harness import (
    test_recursive_generic_enums as harness_recursive_generic_enums,
)
from tests.regression_harness import (
    test_recursive_generic_structs as harness_recursive_generic_structs,
)
from tests.regression_harness import (
    test_scaffold_drift_detection as harness_scaffold_drift_detection,
)
from tests.regression_harness import (
    test_trait_bounds_and_impls as harness_trait_bounds_and_impls,
)
from tests.regression_harness import (
    test_typed_pillar_error_parity as harness_typed_pillar_error_parity,
)

# Exposed for conftest.py's session-wide native build. The prefixed F10 names
# cannot collide with probes from other test modules.
SOURCES = dict(F10_SOURCES)


def _write(root: Path, relative: str, text: str) -> Path:
    """Write a UTF-8 fixture file and return its path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _capability_source() -> str:
    """Return a small literal capability registry for documentation tests."""
    return """
_ROWS: tuple[tuple[str, str, str, str, str, str, str, str], ...] = (
    ("alpha", "core", "runtime.alpha", "alpha", "foundation", "E001", "alpha", "owned token"),
    ("beta", "core", "runtime.beta", "beta", "modelled", "E002", "beta", "in-process model"),
)
""".lstrip()


def _cli_source() -> str:
    """Return a minimal argparse CLI source with one documented command."""
    return """
def build_parser():
    parser = argparse.ArgumentParser(prog="polyglot")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run")
    run_p.add_argument("file")
    run_p.add_argument("--backend", choices=["python", "native"])
    run_p.add_argument("--verbose", action="store_true")
    return parser
""".lstrip()


def _backend_source() -> str:
    """Return a minimal backend registry with two backend names."""
    return """
def get_backend(name="python"):
    if name == "python":
        return None
    if name == "native":
        return None
    raise ValueError(name)
""".lstrip()


def _fake_native_namespace() -> dict[str, Any]:
    """Return callable stand-ins with the expected native probe results."""
    return {
        "f10_recursive_struct": lambda: 3,
        "f10_recursive_enum": lambda: 2,
        "f10_box_bounds": lambda: 24,
        "f10_identity_bounds": lambda: 42,
        "f10_impl_omit_generics": lambda: 42,
    }


# ---------------------------------------------------------------------------
# Functions 91-93: documentation sync
# ---------------------------------------------------------------------------


def test_capability_matrix_detects_documented_status_drift(tmp_path: Path) -> None:
    _write(tmp_path, "runtime/capabilities.py", _capability_source())
    doc = _write(
        tmp_path,
        "docs/unified_pillars.md",
        """# Capabilities

| # | capability | status | boundary |
|---:|---|---|---|
| 1 | `alpha` | foundation | owned token |
| 2 | `beta` | modelled | in-process model |

The registry classifies the 2 functions as 1 foundation, 1 modelled, and 0 rejected.
""",
    )
    aligned = docs_sync_capability_matrix(root=tmp_path)
    assert aligned["ok"] is True, aligned["summary"]
    assert aligned["expected_counts"] == {
        "total": 2,
        "foundation": 1,
        "modelled": 1,
        "rejected": 0,
    }
    assert [item["path"] for item in aligned["documented"]] == [
        "docs/unified_pillars.md"
    ]

    doc.write_text(
        doc.read_text(encoding="utf-8").replace(
            "| 1 | `alpha` | foundation |", "| 1 | `alpha` | modelled |"
        ),
        encoding="utf-8",
    )
    drifted = docs_sync_capability_matrix(root=tmp_path)
    assert drifted["ok"] is False
    assert any(
        record["kind"] == "status-mismatch" and record["name"] == "alpha"
        for record in drifted["drift"]
    )


def test_cli_sync_detects_unknown_usage_flag(tmp_path: Path) -> None:
    _write(tmp_path, "api/cli.py", _cli_source())
    _write(tmp_path, "backends/base.py", _backend_source())
    doc = _write(
        tmp_path,
        "docs/cli.md",
        """# CLI

Use `polyglot run file.poly --backend python --verbose`.
The backends are python and native.
""",
    )
    aligned = docs_sync_cli_and_backend_flags(root=tmp_path)
    assert aligned["ok"] is True, aligned["summary"]
    assert aligned["implementation"]["commands"] == {
        "run": {
            "aliases": [],
            "flags": ["--backend", "--verbose"],
            "backends": ["native", "python"],
        }
    }

    doc.write_text(
        doc.read_text(encoding="utf-8")
        + "\npolyglot run file.poly --backend python --bogus\n",
        encoding="utf-8",
    )
    drifted = docs_sync_cli_and_backend_flags(root=tmp_path)
    assert drifted["ok"] is False
    assert any(
        record["kind"] == "unknown-flag" and record["flag"] == "--bogus"
        for record in drifted["drift"]
    )


def test_platform_constraints_record_boundaries(tmp_path: Path) -> None:
    _write(tmp_path, "runtime/capabilities.py", _capability_source())
    _write(
        tmp_path,
        "docs/quantum_hotpatch.md",
        "MAP_JIT patches owned pages; foreign __TEXT is rejected with E007.",
    )
    _write(
        tmp_path,
        "docs/experimental_pillars.md",
        "PEP 667 uses fast-local write-through; f_code is never swapped mid-instruction. "
        "TOCTOU checks cannot pin raw mappings. Modelled paths are in-process.",
    )
    result = docs_record_platform_constraints(root=tmp_path)
    assert result["ok"] is True, result["summary"]
    assert [record["id"] for record in result["records"]] == [
        "map_jit",
        "pep_667",
        "toctou",
        "model_only",
    ]
    by_id = {record["id"]: record for record in result["records"]}
    assert "foreign __TEXT" in by_id["map_jit"]["boundary"]
    assert "future-entry" in by_id["pep_667"]["runtime_requirement"]
    assert "TOCTOU window" in by_id["toctou"]["boundary"]
    assert by_id["model_only"]["capabilities"] == ["beta"]
    assert "boundary" in by_id["model_only"]["markdown"].lower()


# ---------------------------------------------------------------------------
# Functions 94-96: recursion, bounds, and native parity
# ---------------------------------------------------------------------------


def test_recursive_generic_structs_harness_reports_parity() -> None:
    result = harness_recursive_generic_structs(
        native_namespace=_fake_native_namespace()
    )
    assert result["ok"] is True, result["summary"]
    assert result["expected"] == 3
    assert result["python"]["value"] == result["native"]["value"] == 3
    assert result["native"]["source"] == "injected-native-namespace"
    monomorph = result["checks"]["monomorphization"]
    assert monomorph["status"] == "ok"
    assert "F10Node" in monomorph["names"]
    assert monomorph["residual_generics"] == []


@pytest.mark.native
def test_recursive_generic_structs_native(native_ns: dict[str, Any]) -> None:
    result = harness_recursive_generic_structs(native_namespace=native_ns)
    assert result["native"]["status"] == "ok", result["summary"]
    assert result["ok"] is True


def test_recursive_generic_enums_harness_reports_parity() -> None:
    result = harness_recursive_generic_enums(native_namespace=_fake_native_namespace())
    assert result["ok"] is True, result["summary"]
    assert result["expected"] == 2
    assert result["python"]["value"] == result["native"]["value"] == 2
    monomorph = result["checks"]["monomorphization"]
    assert monomorph["status"] == "ok"
    assert "F10Tree" in monomorph["names"]


@pytest.mark.native
def test_recursive_generic_enums_native(native_ns: dict[str, Any]) -> None:
    result = harness_recursive_generic_enums(native_namespace=native_ns)
    assert result["native"]["status"] == "ok", result["summary"]
    assert result["ok"] is True


def test_trait_bounds_harness_reports_all_probes() -> None:
    result = harness_trait_bounds_and_impls(native_namespace=_fake_native_namespace())
    assert result["ok"] is True, result["summary"]
    assert result["parity"] is True
    assert result["checks"]["bounds"] == {
        "status": "ok",
        "struct": "F10Box",
        "parameter": "T",
        "bounds": ["Clone"],
    }
    probes = result["checks"]["probes"]
    assert sorted(probes) == [
        "f10_box_bounds",
        "f10_identity_bounds",
        "f10_impl_omit_generics",
    ]
    assert all(probe["ok"] for probe in probes.values())


@pytest.mark.native
def test_trait_bounds_native(native_ns: dict[str, Any]) -> None:
    result = harness_trait_bounds_and_impls(native_namespace=native_ns)
    assert all(
        probe["native"]["status"] == "ok"
        for probe in result["checks"]["probes"].values()
    ), result["summary"]
    assert result["ok"] is True


def test_native_harnesses_skip_by_default() -> None:
    for harness in (
        harness_recursive_generic_structs,
        harness_recursive_generic_enums,
        harness_trait_bounds_and_impls,
    ):
        result = harness()
        if harness is harness_trait_bounds_and_impls:
            probes = result["checks"]["probes"].values()
            assert all(probe["native"]["status"] == "skipped" for probe in probes)
        else:
            assert result["native"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Functions 97-100: rollback, errors, scaffold drift, and cache isolation
# ---------------------------------------------------------------------------


def test_hotpatch_delegation_and_rollback_harness() -> None:
    result = harness_hotpatch_delegation_and_rollback()
    assert result["ok"] is True, result["summary"]
    checks = result["checks"]
    assert checks["delegation"]["status"] == "ok"
    assert (
        checks["delegation"]["entries"]["hotpatch_rollback_journal_commit"][
            "actual_module"
        ]
        == "runtime.hotpatch.journal"
    )
    assert checks["manager_rollback"]["values"] == [1, 2, 1]
    assert checks["manager_rollback"]["restored_generation"] == 1
    assert checks["journal_rollback"]["value_after_rollback"] == 1


def test_typed_pillar_error_parity_harness() -> None:
    span = (3, 9)
    result = harness_typed_pillar_error_parity(span=span)
    assert result["ok"] is True, result["summary"]
    assert result["span"] == [3, 9]
    assert sorted(result["cases"]) == [
        "invalid_backend",
        "rejected_capability",
        "supported_failure",
    ]
    for name, case in result["cases"].items():
        assert case["matches"] is True, name
        assert case["python"]["error"]["code"] == case["expected_code"]
        assert case["native"]["error"] == case["python"]["error"]


def test_scaffold_drift_detection_harness(tmp_path: Path) -> None:
    result = harness_scaffold_drift_detection(tmp_dir=tmp_path)
    assert result["ok"] is True, result["summary"]
    assert result["generated_files"] == ["Cargo.toml", "src/lib.rs"]
    assert result["clean_drift"] == []
    assert "src/lib.rs" in result["mutated_drift"]


def test_isolated_cache_no_contamination_harness(tmp_path: Path) -> None:
    result = harness_isolated_cache_no_contamination(tmp_dir=tmp_path)
    assert result["ok"] is True, result["summary"]
    assert sorted(result["scopes"]) == ["f10-scope-a", "f10-scope-b"]
    assert result["cross_contamination"] == []
    assert sorted(result["build_keys"]) == ["f10_cache_a", "f10_cache_b"]
    assert len(set(result["build_keys"].values())) == 2
