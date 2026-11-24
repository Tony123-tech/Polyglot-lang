"""The native cache must not accept artifacts it did not build itself.

A cache entry is seeded with another build's ``target/`` directory to keep
dependencies warm. That directory can contain extension modules from an
unrelated program, so the presence of a ``.dylib`` must never be read as
"this program is already compiled".
"""

from __future__ import annotations

import json
from pathlib import Path

from backends.rust.compiler import NativeBackend

MODULE = "polyglot_native"


def _backend(tmp_path: Path) -> NativeBackend:
    return NativeBackend(cache_dir=tmp_path / "cache")


def _seed_foreign_artifact(build_dir: Path) -> Path:
    release = build_dir / "target" / "release"
    release.mkdir(parents=True, exist_ok=True)
    foreign = release / f"lib{MODULE}.dylib"
    foreign.write_bytes(b"not a real module")
    return foreign


def test_foreign_artifact_without_manifest_is_not_reused(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    build_dir = backend.cache_dir / "0123456789abcdef"
    _seed_foreign_artifact(build_dir)

    assert backend._is_built(build_dir, MODULE) is False


def test_manifest_for_another_module_is_not_reused(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    build_dir = backend.cache_dir / "0123456789abcdef"
    _seed_foreign_artifact(build_dir)
    (build_dir / ".polyglot-manifest.json").write_text(
        json.dumps(
            {
                "source_hash": build_dir.name,
                "module": "hello",
                "extension": f"lib{MODULE}.dylib",
            }
        ),
        encoding="utf-8",
    )

    assert backend._is_built(build_dir, MODULE) is False


def test_manifest_for_a_different_source_dir_is_not_reused(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    build_dir = backend.cache_dir / "0123456789abcdef"
    _seed_foreign_artifact(build_dir)
    (build_dir / ".polyglot-manifest.json").write_text(
        json.dumps(
            {
                "source_hash": "ffffffffffffffff",
                "module": MODULE,
                "extension": f"lib{MODULE}.dylib",
            }
        ),
        encoding="utf-8",
    )

    assert backend._is_built(build_dir, MODULE) is False


def test_manifest_with_missing_extension_is_not_reused(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    build_dir = backend.cache_dir / "0123456789abcdef"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / ".polyglot-manifest.json").write_text(
        json.dumps(
            {
                "source_hash": build_dir.name,
                "module": MODULE,
                "extension": f"lib{MODULE}.dylib",
            }
        ),
        encoding="utf-8",
    )

    assert backend._is_built(build_dir, MODULE) is False


def test_matching_manifest_is_reused(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    build_dir = backend.cache_dir / "0123456789abcdef"
    _seed_foreign_artifact(build_dir)
    (build_dir / ".polyglot-manifest.json").write_text(
        json.dumps(
            {
                "source_hash": build_dir.name,
                "module": MODULE,
                "extension": f"lib{MODULE}.dylib",
            }
        ),
        encoding="utf-8",
    )

    assert backend._is_built(build_dir, MODULE) is True
