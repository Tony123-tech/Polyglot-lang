"""Boundary-safe isolation for causal replay artifacts.

Replay and trace artifacts must never be written into the worktree, and the
pure-Python replay model must never touch executable memory, live PEP 667
frame locals, raw addresses, or privileged paths.  This module enforces both
rules:

* an artifact root must resolve inside an allowed temporary/cache root;
* every artifact path is containment-checked before it is created;
* ``MAP_JIT``, PEP 667 live frames, raw-address TOCTOU, and privileged paths
  are blocked with ``E007`` unless the caller explicitly asks for a
  *model-only* marker (and, for privileged paths, an explicit opt-in);
* the returned report owns its scoped directory and can clean it up.

The report is a model, not a sandbox escape: it never executes native code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn

from runtime.causal_replay.graph import _jsonable
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
Feature = Literal[
    "map_jit",
    "pep667",
    "raw_address",
    "toctou",
    "model_only",
    "privileged",
]
_VALID_FEATURES: frozenset[str] = frozenset(
    {"map_jit", "pep667", "raw_address", "toctou", "model_only", "privileged"}
)
_SCOPE_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    span: Span = None,
) -> NoReturn:
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "use a scoped directory under the system temp/cache root and keep "
        "platform-specific execution out of the replay model",
        note="runtime.causal_replay/isolation",
        span=span,
    )


def _allowed_roots() -> tuple[Path, ...]:
    temp = Path(tempfile.gettempdir()).expanduser().resolve()
    cache_setting = os.environ.get("XDG_CACHE_HOME")
    cache = (
        Path(cache_setting).expanduser() if cache_setting else Path.home() / ".cache"
    )
    roots = [temp]
    try:
        roots.append(cache.resolve())
    except OSError:
        pass
    return tuple(dict.fromkeys(roots))


def _within(path: Path, roots: tuple[Path, ...], *, allow_root: bool = False) -> bool:
    for root in roots:
        if path == root:
            if allow_root:
                return True
            continue
        try:
            if path.is_relative_to(root):
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class BoundaryStatus:
    feature: str
    requested: bool
    supported: bool
    permitted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "requested": self.requested,
            "supported": self.supported,
            "permitted": self.permitted,
            "reason": self.reason,
        }


@dataclass
class BoundaryIsolationReport:
    artifact_root: Path
    scope: str
    created: bool
    boundaries: tuple[BoundaryStatus, ...]
    artifacts: list[Path] = field(default_factory=list)
    blocked: tuple[str, ...] = ()
    _owned: bool = field(default=False, repr=False)
    _roots: tuple[Path, ...] = field(default=(), repr=False)

    def _resolve_artifact(self, name: str) -> Path:
        if not isinstance(name, str) or not name.strip():
            _fail("E003", "artifact name must be a non-empty string")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            _fail(
                "E007",
                f"artifact path {name!r} escapes the scoped artifact root",
                reason="artifact names must be relative and may not contain '..'",
            )
        root = self.artifact_root.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            _fail(
                "E007",
                f"artifact path {name!r} escapes the scoped artifact root",
            )
        if target.exists() and target.is_symlink():
            _fail(
                "E007",
                f"artifact path {name!r} is a symlink",
                reason="replay artifacts must not follow links outside the scope",
            )
        return target

    def reserve(self, name: str) -> Path:
        target = self._resolve_artifact(name)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.parent.resolve().is_relative_to(self.artifact_root.resolve()):
            _fail("E007", f"artifact parent for {name!r} escapes the scope")
        target.touch(mode=0o600, exist_ok=True)
        try:
            target.chmod(0o600)
        except OSError:
            pass
        if target not in self.artifacts:
            self.artifacts.append(target)
        return target

    def write_text(self, name: str, text: str) -> Path:
        if not isinstance(text, str):
            _fail("E003", "artifact text must be a str")
        target = self._resolve_artifact(name)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_text(text, encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
        if target not in self.artifacts:
            self.artifacts.append(target)
        return target

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(
            name,
            json.dumps(_jsonable(payload), sort_keys=True, indent=2) + "\n",
        )

    def cleanup(self) -> bool:
        """Remove the scoped directory if this report created it."""
        if not self._owned:
            return False
        root = self.artifact_root.resolve()
        if not _within(root, self._roots):
            _fail("E007", "refusing to remove a directory outside the allowed roots")
        if root.is_symlink():
            _fail("E007", "refusing to remove a symlinked artifact root")
        shutil.rmtree(root, ignore_errors=False)
        self.artifacts.clear()
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_root": str(self.artifact_root),
            "scope": self.scope,
            "created": self.created,
            "boundaries": [item.as_dict() for item in self.boundaries],
            "artifacts": [str(item) for item in self.artifacts],
            "blocked": list(self.blocked),
            "caveats": (
                "artifacts are confined to a scoped temp/cache directory and "
                "never written to the worktree",
                "MAP_JIT, live PEP 667 frames, raw-address TOCTOU, and "
                "privileged paths are never executed by this model",
                "the report is an isolation boundary, not an OS sandbox",
            ),
        }


def _normalize_feature(feature: str) -> str:
    return feature.strip().lower().replace("-", "_")


def _feature_status(
    feature: str,
    *,
    platform: str,
    allow_model_only: bool,
    allow_privileged: bool,
) -> BoundaryStatus:
    if feature == "map_jit":
        supported = platform.startswith("darwin")
        return BoundaryStatus(
            feature,
            True,
            supported,
            allow_model_only,
            "MAP_JIT executable memory is never mapped; model-only marker"
            if allow_model_only
            else "MAP_JIT executable memory is not available to the replay model",
        )
    if feature == "pep667":
        supported = sys.version_info >= (3, 13)
        return BoundaryStatus(
            feature,
            True,
            supported,
            allow_model_only,
            "PEP 667 live frame locals are never accessed; model-only marker"
            if allow_model_only
            else "PEP 667 live frame locals are not captured",
        )
    if feature in {"raw_address", "toctou"}:
        return BoundaryStatus(
            feature,
            True,
            False,
            allow_model_only,
            "raw addresses/TOCTOU windows are modeled but never dereferenced"
            if allow_model_only
            else "raw-address TOCTOU replay is outside the model-only boundary",
        )
    if feature == "model_only":
        return BoundaryStatus(
            feature,
            True,
            True,
            True,
            "the replay model is model-only by construction",
        )
    if feature == "privileged":
        return BoundaryStatus(
            feature,
            True,
            False,
            allow_privileged,
            "privileged paths require explicit opt-in and are not executed"
            if not allow_privileged
            else "privileged path requested; the replay model records but never executes it",
        )
    _fail("E003", f"unknown boundary feature {feature!r}")


def replay_enforce_boundary_isolation(
    *,
    artifact_root: Path | str | None = None,
    scope: str = "default",
    requested_features: list[str] | tuple[str, ...] = (),
    artifact_names: list[str] | tuple[str, ...] = (),
    allow_model_only: bool = False,
    allow_privileged: bool = False,
    platform: str | None = None,
    strict: bool = True,
    span: Span = None,
) -> BoundaryIsolationReport:
    """Create a scoped artifact root and enforce replay platform boundaries.

    ``artifact_root`` must resolve inside the system temp directory or the
    user cache directory; otherwise ``E007`` is raised.  When it is omitted a
    fresh private directory is created below the system temp root and the
    returned report owns it (``cleanup()`` removes it).

    Requested ``map_jit``, ``pep667``, ``raw_address``, and ``toctou``
    features are blocked unless ``allow_model_only`` is set, in which case they
    are recorded as model-only markers.  ``privileged`` requires
    ``allow_privileged``.  Blocked features raise ``E007`` when ``strict`` is
    true; otherwise they are listed in ``report.blocked``.
    """
    if not isinstance(scope, str) or not _SCOPE_RE.fullmatch(scope):
        _fail(
            "E003",
            "scope must be a simple identifier (letters, digits, '.', '_', '-')",
            span=span,
        )
    roots = _allowed_roots()
    if not roots:
        _fail("E007", "no temporary/cache root is available for replay artifacts")

    created = False
    if artifact_root is None:
        base = roots[0] / "polyglot-causal-replay"
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = Path(tempfile.mkdtemp(prefix=f"{scope}-", dir=base)).resolve()
        created = True
    else:
        root = Path(artifact_root).expanduser().resolve()
        if not _within(root, roots):
            _fail(
                "E007",
                "artifact_root is outside the allowed temp/cache roots",
                reason=f"resolved {root} is not below {roots}",
                help_text="pass a directory below the system temp or ~/.cache",
                span=span,
            )
        if root.exists() and root.is_symlink():
            _fail("E007", "artifact_root must not be a symlink", span=span)
        existed = root.exists()
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            _fail(
                "E007",
                "artifact_root could not be created",
                reason=f"{type(error).__name__}: {error}",
                span=span,
            )
        created = not existed

    try:
        root.chmod(0o700)
    except OSError:
        pass

    platform_name = platform or sys.platform
    boundaries: list[BoundaryStatus] = []
    blocked: list[str] = []
    for requested in requested_features:
        if not isinstance(requested, str) or not requested.strip():
            _fail("E003", "requested features must be non-empty strings", span=span)
        feature = _normalize_feature(requested)
        if feature not in _VALID_FEATURES:
            _fail("E003", f"unknown boundary feature {requested!r}", span=span)
        status = _feature_status(
            feature,
            platform=platform_name,
            allow_model_only=allow_model_only,
            allow_privileged=allow_privileged,
        )
        boundaries.append(status)
        if not status.permitted:
            blocked.append(feature)

    report = BoundaryIsolationReport(
        artifact_root=root,
        scope=scope,
        created=created,
        boundaries=tuple(boundaries),
        blocked=tuple(blocked),
        _owned=created,
        _roots=roots,
    )

    if strict and blocked:
        _fail(
            "E007",
            "requested replay features cross a platform/runtime boundary",
            reason=f"blocked features: {', '.join(blocked)}",
            help_text="remove the feature, use allow_model_only=True for a "
            "model-only marker, or opt into privileged paths explicitly",
            span=span,
        )
    for name in artifact_names:
        report.reserve(name)
    return report


__all__ = [
    "BoundaryIsolationReport",
    "BoundaryStatus",
    "Feature",
    "replay_enforce_boundary_isolation",
]
