"""Model-only guards for hot-patch requests and live runtime state.

Every function in this module validates inputs and returns a structured
verdict.  A raw address is only ever an opaque key into an in-process model
registry: it is never dereferenced, and no page-protection flag, executable
text, ``f_code`` pointer or monitoring callback is modified.  Requests that
would require rewriting a loaded image or using an unsupported native seam are
refused with ``E007``, preserving the legacy hot-patch diagnostic mapping.

Diagnostics use :class:`compiler.frontend.errors.PolyglotError` with the shared
``E001``-``E008`` codes: ``E001`` for ABI/identity mismatches, ``E002`` for
soft semantic regressions, ``E003`` for API misuse, ``E005`` for dependency
cycles, ``E006`` for missing symbols or model entries, and ``E008`` for
malformed journal or audit state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "hpg_abi_shape",
    "hpg_address_generation_bump",
    "hpg_address_guard_check",
    "hpg_address_guard_token",
    "hpg_address_tag",
    "hpg_audit_trail_guard",
    "hpg_boundary_report",
    "hpg_capability_guard",
    "hpg_change_scope_guard",
    "hpg_check_raw_address",
    "hpg_dependency_guard",
    "hpg_epoch_guard",
    "hpg_frame_eval_slot",
    "hpg_frame_layout_check",
    "hpg_hook_plan",
    "hpg_hook_slot_conflict",
    "hpg_journal_guard",
    "hpg_layout_compat",
    "hpg_loaded_image_guard",
    "hpg_native_pattern_guard",
    "hpg_page_protection_guard",
    "hpg_reactor_registration_guard",
    "hpg_reactor_snapshot_diff",
    "hpg_symbol_resolution_guard",
    "hpg_version_token_check",
]

_MODEL_BOUNDARY = (
    "Model-only guard: addresses are state keys, never pointers; no foreign "
    "text is rewritten and no page protection or runtime callback is changed."
)
_FORBIDDEN_PATTERNS = frozenset(
    {
        "mmap",
        "MAP_JIT",
        "map_jit",
        "mprotect",
        "VirtualProtect",
        "loaded_image_text",
        "f_code_rewrite",
        "dlsym_override",
        "page_protection_change",
    }
)
_SAFE_PATTERNS = frozenset(
    {"model", "python", "python-bridge", "selector", "cooperative", "journal"}
)
_REWRITE_OPERATIONS = frozenset({"rewrite", "patch", "text-write", "mprotect"})
_EVENT_NAMES = frozenset({"read", "write", "error"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured :class:`PolyglotError` for a failed guard check."""
    raise PolyglotError(
        code,
        message,
        reason=reason or message,
        note="runtime.hotpatch.guards",
    )


def _identifier(value: object, label: str) -> str:
    """Return a non-empty, whitespace-free identifier."""
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        _fail("E003", f"{label} must be a non-empty identifier without whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    """Return an integer at least *minimum*, or raise E003."""
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("E003", f"{label} must be an integer")
    if value < minimum:
        _fail("E003", f"{label} must be at least {minimum}")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Validate and copy a string-keyed mapping of JSON-compatible values."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E003", f"{label} keys must be strings")
        result[key] = _jsonable(item, f"{label}[{key!r}]")
    return result


def _jsonable(value: object, label: str) -> Any:
    """Return a JSON-compatible copy of *value* or reject it with E003."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("E003", f"{label} must be a finite number")
        return value
    if isinstance(value, list):
        return [
            _jsonable(item, f"{label}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, f"{label}[{key!r}]")
            for key, item in value.items()
        }
    _fail("E003", f"{label} must contain only JSON-compatible values")


def _strings(value: object, label: str) -> list[str]:
    """Validate a sequence of identifiers and return a deduplicated list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("E003", f"{label} must be a sequence of identifiers")
    result: list[str] = []
    for item in value:
        name = _identifier(item, label)
        if name not in result:
            result.append(name)
    return result


def _digest(payload: object) -> str:
    """SHA-256 hex digest of a canonical JSON payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _address(value: object, label: str = "address") -> int:
    """Validate a raw address used solely as a model-state key."""
    return _integer(value, label)


def _address_registry(registry: object) -> dict[int, dict[str, Any]]:
    """Validate a registry mapping address keys to model-tag records."""
    if not isinstance(registry, Mapping):
        _fail("E003", "registry must be a mapping of addresses to model tags")
    result: dict[int, dict[str, Any]] = {}
    for address, raw in registry.items():
        key = _address(address, "registry address")
        tag = _mapping(raw, f"registry[{key}]")
        if "tag" not in tag or "namespace" not in tag:
            _fail("E008", f"registry[{key}] needs 'tag' and 'namespace'")
        tag.setdefault("generation", 1)
        tag.setdefault("loaded_image", False)
        tag["generation"] = _integer(tag["generation"], "generation", minimum=1)
        if not isinstance(tag["loaded_image"], bool):
            _fail("E008", f"registry[{key}].loaded_image must be a bool")
        result[key] = tag
    return result


def _token_record(token: object) -> dict[str, Any]:
    """Validate an address guard token."""
    record = _mapping(token, "token")
    required = ("address", "generation", "epoch", "digest")
    missing = [key for key in required if key not in record]
    if missing:
        _fail("E008", f"malformed guard token: missing {missing[0]!r}")
    record["address"] = _address(record["address"])
    record["generation"] = _integer(record["generation"], "generation", minimum=1)
    record["epoch"] = _integer(record["epoch"], "epoch")
    expected = _digest(
        {
            "address": record["address"],
            "generation": record["generation"],
            "epoch": record["epoch"],
        }
    )
    if record["digest"] != expected:
        _fail("E008", "guard token digest mismatch")
    return record


def _parse_signature(signature: object, label: str) -> dict[str, Any]:
    """Parse a small C-style signature into return and parameter types."""
    if not isinstance(signature, str) or not signature:
        _fail("E003", f"{label} must be a non-empty signature string")
    compact = "".join(signature.split())
    if compact.count("(") != 1 or not compact.endswith(")"):
        _fail("E001", f"malformed signature {signature!r}")
    return_type, params = compact.split("(", 1)
    parameters = [] if params == "()" else params[:-1].split(",")
    for item in [return_type, *parameters]:
        if not item or not all(ch.isalnum() or ch == "_" for ch in item):
            _fail("E001", f"invalid type token {item!r} in {signature!r}")
    return {
        "signature": compact,
        "return_type": return_type,
        "parameters": parameters,
        "arity": len(parameters),
    }


def _size_map(value: object, label: str) -> dict[str, int]:
    """Validate a field-to-byte-size mapping."""
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for field, size in raw.items():
        result[_identifier(field, f"{label} field")] = _integer(
            size, f"{label}[{field!r}]", minimum=0
        )
    return result


def _registration(raw: object, index: int) -> dict[str, Any]:
    """Normalize one reactor registration record."""
    record = _mapping(raw, f"registrations[{index}]")
    required = ("fd", "events", "callback")
    missing = [key for key in required if key not in record]
    if missing:
        _fail("E003", f"registrations[{index}] is missing {missing[0]!r}")
    fd = _integer(record["fd"], f"registrations[{index}].fd")
    events = _strings(record["events"], f"registrations[{index}].events")
    for event in events:
        if event not in _EVENT_NAMES:
            _fail("E003", f"unknown selector event {event!r}")
    return {
        "fd": fd,
        "events": sorted(events),
        "callback": _identifier(record["callback"], "callback"),
        "backend": record.get("backend", "selector"),
    }


def _graph(value: object, label: str) -> dict[str, list[str]]:
    """Validate a dependency graph mapping."""
    raw = _mapping(value, label)
    return {key: _strings(item, f"{label}[{key!r}]") for key, item in raw.items()}


def _has_cycle(graph: Mapping[str, Sequence[str]]) -> bool:
    """Return whether a dependency graph contains a cycle."""
    markers: dict[str, int] = {}

    def visit(node: str) -> bool:
        markers[node] = 1
        for neighbour in sorted(graph.get(node, [])):
            if neighbour not in graph:
                continue
            if markers.get(neighbour) == 1:
                return True
            if markers.get(neighbour, 0) == 0 and visit(neighbour):
                return True
        markers[node] = 2
        return False

    for node in sorted(graph):
        if markers.get(node, 0) == 0 and visit(node):
            return True
    return False


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def hpg_abi_shape(signature: object, *, expected: object = None) -> dict[str, Any]:
    """Parse a C-style ABI signature and compare it with an expected shape."""
    actual = _parse_signature(signature, "signature")
    wanted = None if expected is None else _parse_signature(expected, "expected")
    matches = wanted is None or actual["signature"] == wanted["signature"]
    if not matches:
        _fail(
            "E001",
            f"ABI signature mismatch: expected {wanted['signature']}, got {actual['signature']}",
        )
    return {
        **actual,
        "expected": None if wanted is None else wanted["signature"],
        "matches": matches,
        "model_only": True,
    }


def hpg_address_generation_bump(
    registry: object, address: object, *, by: int = 1
) -> dict[str, Any]:
    """Bump the model generation of one registered raw-address key."""
    entries = _address_registry(registry)
    key = _address(address)
    if key not in entries:
        _fail("E006", f"address key {key} is not registered")
    step = _integer(by, "by", minimum=1)
    previous = entries[key]["generation"]
    entries[key]["generation"] = previous + step
    return {
        "registry": {addr: item for addr, item in entries.items()},
        "address": key,
        "previous": previous,
        "generation": entries[key]["generation"],
        "model_only": True,
    }


def hpg_address_guard_check(
    token: object,
    *,
    generation: object = None,
    epoch: object = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Check a TOCTOU token against observed model generation and epoch."""
    record = _token_record(token)
    reasons: list[str] = []
    observed_generation = (
        None if generation is None else _integer(generation, "generation", minimum=1)
    )
    observed_epoch = None if epoch is None else _integer(epoch, "epoch")
    if observed_generation is not None and observed_generation != record["generation"]:
        reasons.append(
            f"generation changed: token {record['generation']} vs observed {observed_generation}"
        )
    if observed_epoch is not None and observed_epoch != record["epoch"]:
        reasons.append(
            f"epoch changed: token {record['epoch']} vs observed {observed_epoch}"
        )
    ok = not reasons
    if strict and not ok:
        _fail("E001", "; ".join(reasons))
    return {
        "address": record["address"],
        "token_generation": record["generation"],
        "token_epoch": record["epoch"],
        "observed_generation": observed_generation,
        "observed_epoch": observed_epoch,
        "ok": ok,
        "reasons": reasons,
        "model_only": True,
    }


def hpg_address_guard_token(
    address: object, *, generation: int = 1, epoch: int = 0, namespace: object = "model"
) -> dict[str, Any]:
    """Create a digest-bound TOCTOU token for a raw-address model key."""
    key = _address(address)
    name = _identifier(namespace, "namespace")
    gen = _integer(generation, "generation", minimum=1)
    era = _integer(epoch, "epoch")
    payload = {"address": key, "generation": gen, "epoch": era}
    return {
        **payload,
        "namespace": name,
        "key": f"{name}:0x{key:x}",
        "digest": _digest(payload),
        "model_only": True,
    }


def hpg_address_tag(
    address: object,
    *,
    namespace: object = "model",
    tag: object = "state-key",
    generation: int = 1,
    loaded_image: bool = False,
) -> dict[str, Any]:
    """Build a model tag record for an opaque raw-address key."""
    key = _address(address)
    name = _identifier(namespace, "namespace")
    label = _identifier(tag, "tag")
    gen = _integer(generation, "generation", minimum=1)
    if not isinstance(loaded_image, bool):
        _fail("E003", "loaded_image must be a bool")
    return {
        "address": key,
        "key": f"{name}:0x{key:x}",
        "namespace": name,
        "tag": label,
        "generation": gen,
        "loaded_image": loaded_image,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_audit_trail_guard(
    record: object, *, required: object = ("event", "actor", "target")
) -> dict[str, Any]:
    """Validate one audit record's required fields and sequence number."""
    fields = _strings(required, "required")
    entry = _mapping(record, "record")
    missing = [field for field in fields if field not in entry]
    if missing:
        _fail("E008", f"malformed audit record: missing {missing[0]!r}")
    seq = entry.get("seq", 0)
    entry["seq"] = _integer(seq, "seq")
    for field in fields:
        entry[field] = _identifier(entry[field], field)
    return {
        "record": entry,
        "fields": fields,
        "digest": _digest(entry),
        "model_only": True,
    }


def hpg_boundary_report(decisions: object) -> dict[str, Any]:
    """Aggregate guard decisions into one model-only boundary report."""
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        _fail("E003", "decisions must be a sequence")
    records: list[dict[str, Any]] = []
    codes: dict[str, int] = {}
    allowed = 0
    for index, raw in enumerate(decisions):
        item = _mapping(raw, f"decisions[{index}]")
        name = _identifier(item.get("name"), f"decisions[{index}].name")
        if not isinstance(item.get("allowed"), bool):
            _fail("E003", f"decisions[{index}].allowed must be a bool")
        code = item.get("code", "OK" if item["allowed"] else "E007")
        if not isinstance(code, str):
            _fail("E003", f"decisions[{index}].code must be a string")
        codes[code] = codes.get(code, 0) + 1
        allowed += int(item["allowed"])
        records.append({"name": name, "allowed": item["allowed"], "code": code})
    return {
        "decisions": records,
        "total": len(records),
        "allowed": allowed,
        "denied": len(records) - allowed,
        "codes": dict(sorted(codes.items())),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_capability_guard(
    capability: object, *, backend: object = "python", supported: object = ()
) -> dict[str, Any]:
    """Check a capability id and backend against a model support set."""
    name = _identifier(capability, "capability")
    kind = _identifier(backend, "backend")
    if kind not in ("python", "native"):
        _fail("E003", f"unknown backend {kind!r}")
    entries = _strings(supported, "supported")
    if name not in entries:
        _fail("E006", f"capability {name!r} is not present in the support set")
    active = kind == "python" or name.endswith("_native")
    if kind == "native" and not active:
        _fail("E007", f"no native adapter is modelled for {name!r}")
    return {
        "capability": name,
        "backend": kind,
        "active": active,
        "supported": entries,
        "model_only": True,
    }


def hpg_change_scope_guard(
    changes: object, *, allowed_prefixes: object = (), strict: bool = False
) -> dict[str, Any]:
    """Split change keys into accepted and rejected model scopes."""
    prefixes = _strings(allowed_prefixes, "allowed_prefixes")
    if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)):
        _fail("E003", "changes must be a sequence")
    accepted: list[str] = []
    rejected: list[str] = []
    for index, raw in enumerate(changes):
        key = (
            raw
            if isinstance(raw, str)
            else _mapping(raw, f"changes[{index}]").get("key")
        )
        name = _identifier(key, f"changes[{index}].key")
        if any(name.startswith(prefix) for prefix in prefixes):
            accepted.append(name)
        else:
            rejected.append(name)
    if strict and rejected:
        _fail("E003", f"change keys outside scope: {', '.join(rejected)}")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "prefixes": prefixes,
        "model_only": True,
    }


def hpg_check_raw_address(
    address: object,
    *,
    registry: object = None,
    generation: object = None,
    allow_loaded_image: bool = False,
) -> dict[str, Any]:
    """Validate an opaque raw-address key against a model registry."""
    key = _address(address)
    if not isinstance(allow_loaded_image, bool):
        _fail("E003", "allow_loaded_image must be a bool")
    tag = hpg_address_tag(key, tag="state-key")
    if registry is not None:
        entries = _address_registry(registry)
        if key not in entries:
            _fail("E006", f"address key {key} is not registered")
        registered = entries[key]
        if registered["loaded_image"] and not allow_loaded_image:
            _fail("E007", "unsupported loaded-image rewrite of a raw address")
        tag = {
            "address": key,
            "key": f"{registered['namespace']}:0x{key:x}",
            "namespace": registered["namespace"],
            "tag": registered["tag"],
            "generation": registered["generation"],
            "loaded_image": registered["loaded_image"],
            "model_only": True,
            "boundary": _MODEL_BOUNDARY,
        }
    observed = (
        None if generation is None else _integer(generation, "generation", minimum=1)
    )
    if observed is not None and observed != tag["generation"]:
        _fail("E001", f"generation mismatch for address key {key}")
    return {
        **tag,
        "observed_generation": observed,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_dependency_guard(node: object, *, graph: object = None) -> dict[str, Any]:
    """Check a node's dependency closure for self-loops and cycles."""
    name = _identifier(node, "node")
    edges = _graph(graph or {}, "graph")
    if name in edges.get(name, []):
        _fail("E005", f"node {name!r} depends on itself")
    probe: dict[str, list[str]] = {key: list(value) for key, value in edges.items()}
    probe.setdefault(name, [])
    if _has_cycle(probe):
        _fail("E005", f"dependency graph containing {name!r} has a cycle")
    ancestors: list[str] = []
    pending = list(probe[name])
    while pending:
        current = pending.pop(0)
        if current not in ancestors:
            ancestors.append(current)
            pending.extend(probe.get(current, []))
    return {
        "node": name,
        "ancestors": sorted(ancestors),
        "cycle_free": True,
        "model_only": True,
    }


def hpg_epoch_guard(
    epoch: object, *, current: object = 0, strict: bool = False
) -> dict[str, Any]:
    """Validate an epoch value and report cooperative epoch regression."""
    value = _integer(epoch, "epoch")
    now = _integer(current, "current")
    regression = value < now
    if regression and strict:
        _fail("E002", f"epoch {value} regressed from current epoch {now}")
    return {
        "epoch": value,
        "current": now,
        "delta": value - now,
        "regression": regression,
        "ok": not regression,
        "model_only": True,
    }


def hpg_frame_eval_slot(
    slot: object, *, version: object = (3, 13), tool_id: int = 2
) -> dict[str, Any]:
    """Validate a model monitoring tool-id slot for a Python version."""
    number = _integer(slot, "slot")
    if number > 5:
        _fail("E003", "slot must be between 0 and 5")
    tool = _integer(tool_id, "tool_id", minimum=1)
    if tool > 5:
        _fail("E003", "tool_id must be between 1 and 5")
    if not isinstance(version, (tuple, list)) or len(version) < 2:
        _fail("E003", "version must be a (major, minor) sequence")
    major = _integer(version[0], "version major")
    minor = _integer(version[1], "version minor")
    if (major, minor) < (3, 12):
        _fail("E007", "frame-evaluation slots need Python 3.12 or newer")
    return {
        "slot": number,
        "tool_id": tool,
        "version": f"{major}.{minor}",
        "supported": True,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_frame_layout_check(
    layout: object, *, frame_size: object = None
) -> dict[str, Any]:
    """Check field offsets and sizes for overlap in a model frame layout."""
    raw = _mapping(layout, "layout")
    fields: list[dict[str, Any]] = []
    for field, spec in raw.items():
        name = _identifier(field, "layout field")
        if isinstance(spec, Mapping):
            item = _mapping(spec, f"layout[{field}]")
            offset = _integer(item.get("offset"), f"layout[{field}].offset")
            size = _integer(item.get("size"), f"layout[{field}].size", minimum=1)
        else:
            _fail("E003", f"layout[{field}] must define 'offset' and 'size'")
        fields.append(
            {"field": name, "offset": offset, "size": size, "end": offset + size}
        )
    fields.sort(key=lambda item: (item["offset"], item["field"]))
    for previous, current in pairwise(fields):
        if current["offset"] < previous["end"]:
            _fail(
                "E001",
                f"frame layout fields {previous['field']} and {current['field']} overlap",
            )
    total = fields[-1]["end"] if fields else 0
    limit = None if frame_size is None else _integer(frame_size, "frame_size")
    if limit is not None and total > limit:
        _fail("E001", f"frame layout needs {total} bytes but frame_size is {limit}")
    return {
        "fields": fields,
        "total_size": total,
        "frame_size": limit,
        "model_only": True,
    }


def hpg_hook_plan(
    symbol: object,
    *,
    slot: int = 0,
    callback: object = "python_bridge",
    tool_id: int = 2,
    version: object = (3, 13),
) -> dict[str, Any]:
    """Build a model frame-evaluation hook plan without installing a hook."""
    name = _identifier(symbol, "symbol")
    target = _identifier(callback, "callback")
    if target not in ("python_bridge", "model_callback", "traced_callback"):
        _fail("E007", f"unsupported frame-evaluation callback {target!r}")
    checked = hpg_frame_eval_slot(slot, version=version, tool_id=tool_id)
    plan = {
        "symbol": name,
        "slot": checked["slot"],
        "callback": target,
        "tool_id": checked["tool_id"],
        "version": checked["version"],
    }
    return {
        **plan,
        "digest": _digest(plan),
        "installed": False,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_hook_slot_conflict(plans: object) -> dict[str, Any]:
    """Detect two hook plans using the same slot for different targets."""
    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        _fail("E003", "plans must be a sequence")
    owners: dict[int, list[dict[str, Any]]] = {}
    for index, raw in enumerate(plans):
        plan = _mapping(raw, f"plans[{index}]")
        if "slot" not in plan or "symbol" not in plan:
            _fail("E003", f"plans[{index}] needs 'slot' and 'symbol'")
        slot = _integer(plan["slot"], f"plans[{index}].slot")
        owners.setdefault(slot, []).append(
            {
                "slot": slot,
                "symbol": _identifier(plan["symbol"], "symbol"),
                "callback": _identifier(
                    plan.get("callback", "python_bridge"), "callback"
                ),
            }
        )
    conflicts: list[dict[str, Any]] = []
    for slot in sorted(owners):
        group = sorted(owners[slot], key=lambda item: item["symbol"])
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if (
                    left["symbol"] != right["symbol"]
                    or left["callback"] != right["callback"]
                ):
                    conflicts.append({"slot": slot, "left": left, "right": right})
    return {
        "slots": {str(slot): owners[slot] for slot in sorted(owners)},
        "conflicts": conflicts,
        "count": len(conflicts),
        "model_only": True,
    }


def hpg_journal_guard(journal: object) -> dict[str, Any]:
    """Validate the shape and digest chain of a model rollback journal."""
    if journal is None:
        return {"ok": True, "count": 0, "head": None, "model_only": True}
    if not isinstance(journal, Sequence) or isinstance(journal, (str, bytes)):
        _fail("E008", "journal must be a sequence of entries")
    head: str | None = None
    for index, raw in enumerate(journal):
        entry = _mapping(raw, f"journal[{index}]")
        required = ("seq", "action", "digest")
        missing = [key for key in required if key not in entry]
        if missing:
            _fail("E008", f"journal[{index}] is missing {missing[0]!r}")
        if entry["seq"] != index + 1:
            _fail("E008", f"journal[{index}] has a non-sequential seq")
        if not isinstance(entry["digest"], str) or not entry["digest"]:
            _fail("E008", f"journal[{index}] has an invalid digest")
        if index and entry.get("prev") != head:
            _fail("E008", f"journal[{index}] breaks the digest chain")
        head = entry["digest"]
    return {"ok": True, "count": len(journal), "head": head, "model_only": True}


def hpg_layout_compat(
    layout: object, *, expected: object = None, allow_growth: bool = False
) -> dict[str, Any]:
    """Compare a model layout with an expected layout without touching bytes."""
    actual = _size_map(layout, "layout")
    wanted = {} if expected is None else _size_map(expected, "expected")
    if not isinstance(allow_growth, bool):
        _fail("E003", "allow_growth must be a bool")
    added = sorted(set(actual) - set(wanted))
    removed = sorted(set(wanted) - set(actual))
    resized = sorted(
        field for field in set(actual) & set(wanted) if actual[field] != wanted[field]
    )
    if removed:
        _fail("E001", f"layout fields removed: {', '.join(removed)}")
    if resized:
        _fail("E001", f"layout fields resized: {', '.join(resized)}")
    if added and not allow_growth:
        _fail(
            "E001", f"layout fields appended without allow_growth: {', '.join(added)}"
        )
    return {
        "fields": sorted(actual),
        "added": added,
        "removed": removed,
        "resized": resized,
        "compatible": True,
        "model_only": True,
    }


def hpg_loaded_image_guard(
    target: object, *, operation: object = "inspect", loaded_image: bool = False
) -> dict[str, Any]:
    """Refuse loaded-image rewrites with E007; model inspection is allowed."""
    name = _identifier(target, "target")
    action = _identifier(operation, "operation")
    if not isinstance(loaded_image, bool):
        _fail("E003", "loaded_image must be a bool")
    if loaded_image and action in _REWRITE_OPERATIONS:
        _fail("E007", f"unsupported loaded-image rewrite of {name!r} ({action})")
    return {
        "target": name,
        "operation": action,
        "loaded_image": loaded_image,
        "allowed": True,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_native_pattern_guard(
    pattern: object, *, backend: object = "python"
) -> dict[str, Any]:
    """Classify a native pattern as model-safe or unsupported (E007)."""
    name = _identifier(pattern, "pattern")
    kind = _identifier(backend, "backend")
    if name in _FORBIDDEN_PATTERNS:
        _fail("E007", f"unsupported native pattern {name!r}")
    if name not in _SAFE_PATTERNS:
        _fail("E007", f"native pattern {name!r} is not modelled")
    return {
        "pattern": name,
        "backend": kind,
        "classification": "model-safe",
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_page_protection_guard(
    *, protection: object = "none", address: object = None
) -> dict[str, Any]:
    """Model a page-protection request without changing any page flags."""
    name = _identifier(protection, "protection")
    key = None if address is None else _address(address)
    if name not in ("none", "model-flag"):
        _fail("E007", f"unsupported page-protection change {name!r}")
    return {
        "protection": name,
        "address": key,
        "changed": False,
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_reactor_registration_guard(
    registration: object, *, allowed_events: object = None
) -> dict[str, Any]:
    """Validate a selector registration record in the in-process model."""
    record = _registration(registration, 0)
    allowed = (
        None if allowed_events is None else _strings(allowed_events, "allowed_events")
    )
    for event in record["events"]:
        if allowed is not None and event not in allowed:
            _fail("E003", f"event {event!r} is not allowed")
    if record["backend"] in ("native_mio", "raw_epoll"):
        _fail("E007", f"unsupported reactor backend {record['backend']!r}")
    return {
        **record,
        "digest": _digest(record),
        "model_only": True,
        "boundary": _MODEL_BOUNDARY,
    }


def hpg_reactor_snapshot_diff(before: object, after: object) -> dict[str, Any]:
    """Diff two selector-registration snapshots keyed by model fd."""
    if not isinstance(before, Sequence) or isinstance(before, (str, bytes)):
        _fail("E003", "before must be a sequence of registrations")
    if not isinstance(after, Sequence) or isinstance(after, (str, bytes)):
        _fail("E003", "after must be a sequence of registrations")
    old = {
        item["fd"]: item
        for item in (_registration(raw, i) for i, raw in enumerate(before))
    }
    new = {
        item["fd"]: item
        for item in (_registration(raw, i) for i, raw in enumerate(after))
    }
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        fd for fd in set(old) & set(new) if _digest(old[fd]) != _digest(new[fd])
    )
    return {
        "added": [new[fd] for fd in added],
        "removed": [old[fd] for fd in removed],
        "changed": [{"fd": fd, "before": old[fd], "after": new[fd]} for fd in changed],
        "count": len(added) + len(removed) + len(changed),
        "model_only": True,
    }


def hpg_symbol_resolution_guard(
    symbol: object, *, symbols: object = ()
) -> dict[str, Any]:
    """Check that a symbol exists in the model resolution table."""
    name = _identifier(symbol, "symbol")
    candidates = _strings(symbols, "symbols")
    if name not in candidates:
        _fail("E006", f"symbol {name!r} is not in the model resolution table")
    return {
        "symbol": name,
        "resolved": True,
        "candidates": candidates,
        "model_only": True,
    }


def hpg_version_token_check(
    token: object, *, expected: object = None, strict: bool = True
) -> dict[str, Any]:
    """Compare a version token with its expected value."""
    if not isinstance(token, (str, int)) or isinstance(token, bool):
        _fail("E003", "token must be a str or int")
    matches = expected is None or token == expected
    if not matches and strict:
        _fail("E001", f"stale version token {token!r}: expected {expected!r}")
    return {
        "token": token,
        "expected": expected,
        "matches": matches,
        "model_only": True,
    }
