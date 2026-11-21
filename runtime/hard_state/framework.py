"""Frame J — points-to sets, live schema migration, and quarantined replay.

This is an explicitly *bounded* model of hard state:

* points-to sets are computed from caller-declared objects, pointers, and
  access traces; raw addresses and FFI are modeled, never dereferenced;
* schema migration accepts only total, unambiguous field maps with a rollback
  plan; lossy or ambiguous transitions are rejected with ``E008``; and
* fault capsules are deterministic, stored only below a caller-supplied temp
  root, and replayed with an injected function.  NATIVE/MAP_JIT and PEP 667
  dynamic-frame paths remain declared boundaries.

No claim of zero aliasing risk, universal aliasing decidability, or zero-risk
zero-downtime migration is made.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from runtime.diagnostics import PillarError
from tools.diff_fuzz import normalize_error

Span = tuple[int, int] | None
Language = Literal["python", "rust", "zig"]
PointerKind = Literal["reference", "borrow", "raw", "slice", "allocator", "handle"]
HazardKind = Literal[
    "mutable-alias", "toctou", "executable-memory", "untracked-ffi", "dangling"
]
Op = Literal["assign", "check", "write", "use", "epoch"]

MUTABLE_KINDS: frozenset[PointerKind] = frozenset(
    {"reference", "borrow", "raw", "slice", "allocator"}
)
SAFE_WIDENINGS: frozenset[tuple[str, str]] = frozenset(
    {
        ("i8", "i16"),
        ("i16", "i32"),
        ("i32", "i64"),
        ("u8", "u16"),
        ("u16", "u32"),
        ("u32", "u64"),
        ("f32", "f64"),
        ("str", "str?"),
    }
)


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text or "declare an explicit mapping or quarantine the fault",
        note="runtime.hard_state/frame-j",
        span=span,
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=repr)
        return items
    return repr(value)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Points-to and alias analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryObject:
    object_id: str
    language: Language
    type_name: str
    size_bytes: int
    mutable: bool = True
    mapped_jit: bool = False
    model_only: bool = True

    def __post_init__(self) -> None:
        if not self.object_id or self.size_bytes < 0:
            raise _fail("E003", "memory objects require an id and non-negative size")


@dataclass(frozen=True)
class PointerFact:
    name: str
    language: Language
    kind: PointerKind
    points_to: tuple[str, ...]
    offset: int = 0
    length: int | None = None
    borrowed_from: str | None = None
    epoch: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.points_to:
            raise _fail("E003", "pointer facts require a name and at least one target")
        if self.offset < 0 or (self.length is not None and self.length < 0):
            raise _fail("E003", "pointer offset and length must be non-negative")


@dataclass(frozen=True)
class Access:
    op: Op
    target: str
    source: str | None = None
    value: int | None = None


@dataclass(frozen=True)
class AliasFinding:
    kind: HazardKind
    left: str
    right: str
    objects: tuple[str, ...]
    message: str
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "objects": list(self.objects),
            "message": self.message,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class PointsToReport:
    epoch: int
    points_to: dict[str, tuple[str, ...]]
    alias_sets: tuple[tuple[str, ...], ...]
    hazards: tuple[AliasFinding, ...]
    diagnostics: tuple[dict[str, str], ...]

    @property
    def safe(self) -> bool:
        return not self.hazards and not self.diagnostics

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "points_to": {
                name: list(targets) for name, targets in self.points_to.items()
            },
            "alias_sets": [list(group) for group in self.alias_sets],
            "hazards": [item.as_dict() for item in self.hazards],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "safe": self.safe,
            "caveats": (
                "points-to sets are a declared over-approximation, not a proof of zero aliasing",
                "raw addresses and FFI are modeled only and never dereferenced",
                "MAP_JIT executable-memory aliases are reported as hazards",
            ),
        }


def _overlaps(
    left: PointerFact, right: PointerFact, objects: Mapping[str, MemoryObject]
) -> bool:
    for object_id in set(left.points_to) & set(right.points_to):
        obj = objects[object_id]
        left_start, right_start = left.offset, right.offset
        left_end = (
            left.offset + left.length if left.length is not None else obj.size_bytes
        )
        right_end = (
            right.offset + right.length if right.length is not None else obj.size_bytes
        )
        left_end = min(left_end, obj.size_bytes)
        right_end = min(right_end, obj.size_bytes)
        if max(left_start, right_start) < min(left_end, right_end):
            return True
    return False


def hard_analyze_points_to_and_alias_sets(
    objects: Sequence[MemoryObject],
    pointers: Sequence[PointerFact],
    accesses: Sequence[Access] = (),
    *,
    epoch: int = 0,
    strict: bool = False,
    span: Span = None,
) -> PointsToReport:
    """Compute flow-sensitive points-to sets and flag aliasing/TOCTOU hazards.

    ``assign`` access records widen a pointer's points-to set to the source
    set; ``check``/``use`` pairs delimit a TOCTOU window.  A ``write`` from an
    aliasing pointer or an ``epoch`` change before ``use`` is a hazard.  With
    ``strict=True`` hazards raise ``E008`` (or ``E007`` for native boundaries).
    """
    object_map = {item.object_id: item for item in objects}
    pointer_map: dict[str, PointerFact] = {item.name: item for item in pointers}
    for pointer in pointers:
        for target in pointer.points_to:
            if target not in object_map:
                raise _fail(
                    "E006",
                    f"pointer {pointer.name!r} targets missing object {target!r}",
                    span=span,
                )
    if epoch < 0:
        raise _fail("E003", "epoch must be non-negative", span=span)

    current_epoch = epoch
    hazards: list[AliasFinding] = []
    diagnostics: list[dict[str, str]] = []
    pending_checks: dict[str, int] = {}

    for index, access in enumerate(accesses):
        if access.op == "epoch":
            for pending in pending_checks.values():
                hazards.append(
                    AliasFinding(
                        "toctou",
                        f"check@{pending}",
                        f"epoch@{current_epoch}",
                        (),
                        "epoch changed between check and use",
                        span,
                    )
                )
            pending_checks.clear()
            current_epoch = (
                access.value if access.value is not None else current_epoch + 1
            )
            continue
        if access.op == "assign":
            source = pointer_map.get(access.source or "")
            if source is None:
                raise _fail(
                    "E006",
                    f"assignment source pointer {access.source!r} not declared",
                    span=span,
                )
            previous = pointer_map.get(access.target)
            if previous is None:
                raise _fail(
                    "E006",
                    f"assignment target pointer {access.target!r} not declared",
                    span=span,
                )
            pointer_map[access.target] = PointerFact(
                name=previous.name,
                language=previous.language,
                kind=previous.kind,
                points_to=tuple(
                    dict.fromkeys((*previous.points_to, *source.points_to))
                ),
                offset=previous.offset,
                length=previous.length,
                borrowed_from=previous.borrowed_from,
                epoch=current_epoch,
            )
            continue
        if access.op == "check":
            pending_checks[access.target] = index
            continue
        if access.op == "write":
            for checked, check_index in tuple(pending_checks.items()):
                checked_ptr = pointer_map.get(checked)
                writer = pointer_map.get(access.target)
                if (
                    checked_ptr is not None
                    and writer is not None
                    and _overlaps(checked_ptr, writer, object_map)
                ):
                    hazards.append(
                        AliasFinding(
                            "toctou",
                            checked,
                            access.target,
                            tuple(
                                sorted(
                                    set(checked_ptr.points_to) & set(writer.points_to)
                                )
                            ),
                            f"aliasing write at access {index} invalidates check {check_index}",
                            span,
                        )
                    )
            continue
        if access.op == "use":
            pending_checks.pop(access.target, None)

    ordered = sorted(pointer_map.values(), key=lambda item: item.name)
    alias_sets: list[tuple[str, ...]] = []
    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            if not _overlaps(left, right, object_map):
                continue
            alias_sets.append((left.name, right.name))
            shared = tuple(sorted(set(left.points_to) & set(right.points_to)))
            mutable_objects = [
                object_map[item] for item in shared if object_map[item].mutable
            ]
            if mutable_objects and (
                left.kind in MUTABLE_KINDS or right.kind in MUTABLE_KINDS
            ):
                hazards.append(
                    AliasFinding(
                        "mutable-alias",
                        left.name,
                        right.name,
                        shared,
                        f"mutable {left.language}/{left.kind} aliases "
                        f"{right.language}/{right.kind}",
                        span,
                    )
                )
            for obj in mutable_objects:
                if obj.mapped_jit:
                    hazards.append(
                        AliasFinding(
                            "executable-memory",
                            left.name,
                            right.name,
                            (obj.object_id,),
                            "MAP_JIT executable object has a mutable alias",
                            span,
                        )
                    )
                if (left.kind == "raw" or right.kind == "raw") and not obj.model_only:
                    hazards.append(
                        AliasFinding(
                            "untracked-ffi",
                            left.name,
                            right.name,
                            (obj.object_id,),
                            "raw pointer reaches a non-model-only FFI object",
                            span,
                        )
                    )

    report = PointsToReport(
        epoch=current_epoch,
        points_to={name: pointer.points_to for name, pointer in pointer_map.items()},
        alias_sets=tuple(alias_sets),
        hazards=tuple(hazards),
        diagnostics=tuple(diagnostics),
    )
    if strict and hazards:
        codes = {
            "executable-memory": "E008",
            "untracked-ffi": "E007",
        }
        code = next(
            (codes[hazard.kind] for hazard in hazards if hazard.kind in codes), "E008"
        )
        raise _fail(
            code,
            f"points-to analysis found {len(hazards)} hazard(s)",
            span=span,
            reason=hazards[0].message,
            help_text="split the mutable borrows or add an explicit TOCTOU guard",
        )
    return report


# ---------------------------------------------------------------------------
# Live state schema migration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    optional: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise _fail("E003", "field specs require a name")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "optional": self.optional,
            "default": _jsonable(self.default),
        }


@dataclass(frozen=True)
class Schema:
    version: int
    kind: Literal["struct", "enum", "table"]
    fields: tuple[FieldSpec, ...]
    variants: tuple[str, ...] = ()

    def field_map(self) -> dict[str, FieldSpec]:
        return {item.name: item for item in self.fields}

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "fields": [item.as_dict() for item in self.fields],
            "variants": list(self.variants),
        }


@dataclass(frozen=True)
class MigrationStep:
    field: str
    action: Literal["copy", "rename", "default", "convert", "drop"]
    source: str | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "action": self.action,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MigrationPlan:
    old_version: int
    new_version: int
    safe: bool
    atomic: bool
    steps: tuple[MigrationStep, ...]
    rollback: tuple[MigrationStep, ...]
    diagnostics: tuple[dict[str, str], ...]

    def digest(self) -> str:
        return self.as_dict()["digest"]

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "safe": self.safe,
            "atomic": self.atomic,
            "steps": [item.as_dict() for item in self.steps],
            "rollback": [item.as_dict() for item in self.rollback],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "caveats": (
                "safe means total and unambiguous for the declared fields only",
                "live readers are assumed pinned to the old epoch during migration",
                "arbitrary hot-patch invariants are not proved",
            ),
        }
        payload["digest"] = _digest(payload)
        return payload


def _coerce_schema(value: Schema | Mapping[str, Any]) -> Schema:
    if isinstance(value, Schema):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload["fields"] = tuple(
            FieldSpec(**item) if isinstance(item, Mapping) else item
            for item in payload.get("fields", ())
        )
        return Schema(**payload)
    raise _fail("E003", "schema must be a Schema or mapping")


def hard_plan_live_state_schema_migration(
    old: Schema | Mapping[str, Any],
    new: Schema | Mapping[str, Any],
    *,
    mapping: Mapping[str, str] | None = None,
    renames: Mapping[str, str] | None = None,
    epoch: int = 0,
    allow_lossy: bool = False,
    span: Span = None,
) -> MigrationPlan:
    """Plan a total field mapping between two schema versions.

    Ambiguous targets, missing required fields, kind narrowing, removed enum
    variants, and unsafe epoch transitions are rejected with ``E008``.  A
    rollback plan is always produced for accepted transitions; a rejected plan
    is marked non-atomic and must not be applied to live state.
    """
    old_schema, new_schema = _coerce_schema(old), _coerce_schema(new)
    if epoch < 0:
        raise _fail("E003", "migration epoch must be non-negative", span=span)
    if old_schema.kind != new_schema.kind:
        raise _fail(
            "E008",
            f"cannot migrate {old_schema.kind} schema to {new_schema.kind}",
            span=span,
            reason="schema kinds differ",
        )
    if new_schema.version <= old_schema.version:
        raise _fail(
            "E008",
            "new schema version must be strictly greater than the old version",
            span=span,
        )

    diagnostics: list[dict[str, str]] = []
    old_fields = old_schema.field_map()
    explicit = dict(mapping or {})
    rename_map = dict(renames or {})
    steps: list[MigrationStep] = []
    rollback: list[MigrationStep] = []
    used_sources: set[str] = set()

    for target in new_schema.fields:
        source_name = (
            explicit.get(target.name) or rename_map.get(target.name) or target.name
        )
        source = old_fields.get(source_name)
        if source is None:
            if target.optional:
                steps.append(
                    MigrationStep(
                        target.name, "default", None, f"default={target.default!r}"
                    )
                )
                rollback.append(
                    MigrationStep(target.name, "drop", None, "remove default")
                )
                continue
            diagnostics.append(
                {
                    "code": "E008",
                    "message": f"new field {target.name!r} has no source in schema {old_schema.version}",
                    "kind": "lossy",
                }
            )
            steps.append(
                MigrationStep(target.name, "default", None, "unmapped-required")
            )
            continue
        if source_name in used_sources:
            diagnostics.append(
                {
                    "code": "E008",
                    "message": f"source field {source_name!r} maps to more than one target",
                    "kind": "ambiguous",
                }
            )
        used_sources.add(source_name)
        if source.kind != target.kind:
            if (source.kind, target.kind) in SAFE_WIDENINGS:
                action, detail = "convert", f"{source.kind} -> {target.kind}"
            else:
                diagnostics.append(
                    {
                        "code": "E008",
                        "message": f"field {target.name!r} changes {source.kind} -> {target.kind}",
                        "kind": "lossy",
                    }
                )
                action, detail = "convert", "unproved conversion"
        elif source_name != target.name:
            action, detail = "rename", f"{source_name} -> {target.name}"
        else:
            action, detail = "copy", f"{source.kind}"
        steps.append(MigrationStep(target.name, action, source_name, detail))
        rollback.append(
            MigrationStep(
                source_name,
                "rename" if action == "rename" else "copy",
                target.name,
                f"restore {source.kind} from {target.name}",
            )
        )

    dropped = sorted(set(old_fields) - used_sources)
    for name in dropped:
        if old_fields[name].optional:
            steps.append(MigrationStep(name, "drop", name, "optional source dropped"))
            continue
        diagnostics.append(
            {
                "code": "E008",
                "message": f"required source field {name!r} is not carried forward",
                "kind": "lossy",
            }
        )
        steps.append(MigrationStep(name, "drop", name, "required source dropped"))

    removed_variants = sorted(set(old_schema.variants) - set(new_schema.variants))
    for variant in removed_variants:
        diagnostics.append(
            {
                "code": "E008",
                "message": f"enum variant {variant!r} would be lost",
                "kind": "lossy",
            }
        )

    if allow_lossy:
        diagnostics = [item for item in diagnostics if "ambiguous" not in item["kind"]]

    safe = not diagnostics
    atomic = safe and bool(rollback)
    if not safe and not allow_lossy:
        # Keep the diagnostics; the caller decides whether to quarantine.
        atomic = False

    plan = MigrationPlan(
        old_version=old_schema.version,
        new_version=new_schema.version,
        safe=safe,
        atomic=atomic,
        steps=tuple(steps),
        rollback=tuple(rollback),
        diagnostics=tuple(diagnostics),
    )
    if not safe and not allow_lossy:
        raise _fail(
            "E008",
            f"schema migration {old_schema.version} -> {new_schema.version} is unsafe",
            span=span,
            reason="; ".join(item["message"] for item in diagnostics[:3]),
            help_text="add an explicit mapping/default or quarantine the transition",
        )
    return plan


# ---------------------------------------------------------------------------
# Quarantine and deterministic replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateCapsule:
    capsule_id: str
    transaction_id: str
    snapshot: dict[str, Any]
    fault: dict[str, Any]
    epoch: int
    jit_mode: Literal["model", "native"]
    pep667_frame: bool
    model_only: bool
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "transaction_id": self.transaction_id,
            "snapshot": _jsonable(self.snapshot),
            "fault": _jsonable(self.fault),
            "epoch": self.epoch,
            "jit_mode": self.jit_mode,
            "pep667_frame": self.pep667_frame,
            "model_only": self.model_only,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ReplayReport:
    capsule_id: str
    deterministic: bool
    matched: bool
    steps: tuple[str, ...]
    outcomes: tuple[Any, ...]
    artifacts: tuple[str, ...]
    diagnostics: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "deterministic": self.deterministic,
            "matched": self.matched,
            "steps": list(self.steps),
            "outcomes": [_jsonable(item) for item in self.outcomes],
            "artifacts": list(self.artifacts),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "caveats": (
                "replay is deterministic only for the injected replay function",
                "native MAP_JIT and PEP 667 frame locals remain explicit boundaries",
            ),
        }


def hard_quarantine_and_replay_faults(
    transaction_id: str,
    snapshot: Mapping[str, Any],
    fault: Mapping[str, Any],
    *,
    replay: Any | None = None,
    tmp_root: Path | str | None = None,
    jit_mode: Literal["model", "native"] = "model",
    pep667_frame: bool = False,
    model_only: bool = True,
    allow_native: bool = False,
    span: Span = None,
) -> tuple[StateCapsule, ReplayReport]:
    """Quarantine a fault capsule and replay it deterministically.

    The capsule is written only below ``tmp_root``; when no root is supplied no
    artifact is written.  Native/MAP_JIT execution requires ``allow_native``,
    and a PEP 667 dynamic frame is recorded as an uninspectable boundary.
    """
    if not transaction_id:
        raise _fail("E003", "quarantine requires a transaction id", span=span)
    if jit_mode == "native" and not allow_native:
        raise _fail(
            "E007",
            "native MAP_JIT replay is not enabled",
            span=span,
            reason="this frame never maps executable memory implicitly",
            help_text="pass jit_mode='model' or explicitly opt in with allow_native=True",
        )
    if not model_only and any(
        "raw" in str(key).lower() or "ffi" in str(key).lower() for key in snapshot
    ):
        raise _fail(
            "E007",
            "raw/FFI snapshot entries cannot cross the model-only boundary",
            span=span,
        )

    capsule_id = f"capsule-{_digest({'transaction': transaction_id, 'fault': _jsonable(fault)})[:16]}"
    payload = {
        "capsule_id": capsule_id,
        "transaction_id": transaction_id,
        "snapshot": _jsonable(snapshot),
        "fault": _jsonable(fault),
        "epoch": fault.get("epoch", 0),
        "jit_mode": jit_mode,
        "pep667_frame": pep667_frame,
        "model_only": model_only,
    }
    capsule = StateCapsule(**payload, digest=_digest(payload))
    diagnostics: list[dict[str, str]] = []
    if pep667_frame:
        diagnostics.append(
            {
                "code": "E008",
                "message": "PEP 667 dynamic frame locals are not captured in the snapshot",
                "kind": "boundary",
            }
        )

    artifacts: list[str] = []
    if tmp_root is not None:
        root = Path(tmp_root).expanduser().resolve()
        capsule_dir = root / "hard_state" / transaction_id
        capsule_dir.mkdir(parents=True, exist_ok=True)
        artifact = capsule_dir / f"{capsule_id}.json"
        artifact.write_text(
            json.dumps(capsule.as_dict(), sort_keys=True), encoding="utf-8"
        )
        artifacts.append(str(artifact))

    steps: list[str] = []
    outcomes: tuple[Any, ...] = ()
    deterministic = True
    matched = False
    if replay is not None:
        first = replay(dict(capsule.snapshot))
        second = replay(dict(capsule.snapshot))
        first_normalized = tuple(
            normalize_error(item) if isinstance(item, BaseException | dict) else item
            for item in (first if isinstance(first, tuple) else (first,))
        )
        second_normalized = tuple(
            normalize_error(item) if isinstance(item, BaseException | dict) else item
            for item in (second if isinstance(second, tuple) else (second,))
        )
        deterministic = first_normalized == second_normalized
        outcomes = first_normalized
        expected = fault.get("expected_outcomes")
        if expected is not None:
            expected_tuple = tuple(
                expected if isinstance(expected, tuple) else (expected,)
            )
            matched = deterministic and outcomes == expected_tuple
        else:
            matched = deterministic
        steps = ["snapshot-restored", "replay-run", "normalized-compared"]

    report = ReplayReport(
        capsule_id=capsule_id,
        deterministic=deterministic,
        matched=matched,
        steps=tuple(steps),
        outcomes=outcomes,
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
    )
    return capsule, report
