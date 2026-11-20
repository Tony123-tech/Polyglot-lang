"""Session-scoped telemetry collection, redaction, and terminal formatting.

The collector is deliberately explicit: callers record raw observations with the
``record_*`` methods, and all formatting happens in :meth:`snapshot`.  That
keeps repeated snapshots stable and lets the CLI render the same payload as
plain text or JSON.  The heavy lifting for measurements is delegated to the
existing diagnostics framework helpers; capabilities and their honest
boundaries come verbatim from :mod:`runtime.hotpatch`.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from diagnostics.framework import (
    diag_export_benchmark_summary,
    diag_record_cache_hit_metrics,
    diag_record_compile_phase_timing,
    diag_redact_sensitive_paths,
)
from runtime.hotpatch import CAPABILITIES, platform_support

_SCHEMA = "polyglot-telemetry/v1"
_EMPTY_NOTE = "no telemetry recorded in this session"
_DEFAULT_CAPACITY = 200
_RECENT_LIMIT = 20
_CAPABILITY_FIELDS = ("mechanism", "boundary", "diagnostic")
_SECTIONS = frozenset({"all", "telemetry", "timings", "cache", "hotpatch"})
_ANSI = {"heading": "36", "value": "32", "warn": "33", "reset": "0"}
_BENCHMARK_KEYS = frozenset({"benchmark", "benchmarks"})
_MONO_RESERVED = frozenset(
    {
        "backend",
        "benchmark",
        "benchmarks",
        "families",
        "growth_nodes",
        "inflation_report",
        "program",
        "report",
        "schema",
        "specializations",
        "total_growth_nodes",
        "total_instantiations",
        "total_specializations",
        "total_specialized_nodes",
        "warnings",
    }
)
_EPOCH_PATTERN = re.compile(
    r"epoch\s*(?:[:=]|is|advanced(?:\s+to)?|moved\s+to|->)?\s*(\d+)",
    re.IGNORECASE,
)
_RESIDUAL_PATH_PATTERNS = (
    re.compile(r"<tmp>[/\\][^\s\"']+"),
    re.compile(r"<workspace>[/\\][^\s\"']+"),
    re.compile(r"~[/\\][^\s\"']+"),
    re.compile(r"(?<![\w<])(?:/Users|/home)[/\\][^\s\"']+", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _count(value: Any) -> int:
    return _optional_int(value) or 0


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Return a deterministic, JSON-safe copy of an explicit observation."""
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, os.PathLike):
        return diag_redact_sensitive_paths(os.fspath(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, _depth=_depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [
            _jsonable(item, _depth=_depth + 1)
            for item in sorted(value, key=lambda item: str(item))
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item, _depth=_depth + 1) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict(), _depth=_depth + 1)
    return str(value)


def _scrub_residual_paths(value: Any) -> Any:
    """Drop path remainders left after the framework's prefix redaction."""
    if isinstance(value, str):
        text = value
        for pattern, replacement in zip(
            _RESIDUAL_PATH_PATTERNS, ("<tmp>", "<workspace>", "~", "~"), strict=True
        ):
            text = pattern.sub(replacement, text)
        return text
    if isinstance(value, Mapping):
        return {key: _scrub_residual_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_residual_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_residual_paths(item) for item in value)
    return value


def _safe_scope(scope: Any) -> str:
    """Return a stable scope label that can never be an absolute path."""
    if scope is None:
        return "default"
    text = str(scope).strip()
    if not text:
        return "default"
    path_like = (
        text.startswith(("/", "\\", "~"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", text))
        or "/" in text
        or "\\" in text
    )
    if path_like:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        return f"<redacted:{digest}>"
    return text


def _event_dict(event: TelemetryEvent | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(event, TelemetryEvent):
        return event.to_dict()
    if isinstance(event, Mapping):
        return dict(event)
    return {}


def _payload_of(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = raw.get("payload")
    return payload if isinstance(payload, Mapping) else raw


def _merged_bucket() -> dict[str, Any]:
    return {"hits": 0, "misses": 0, "rebuild_reasons": {}}


def _merge_bucket(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    target["hits"] += _count(source.get("hits"))
    target["misses"] += _count(source.get("misses"))
    reasons = source.get("rebuild_reasons")
    if isinstance(reasons, Mapping):
        for reason, count in reasons.items():
            key = str(reason)
            target["rebuild_reasons"][key] = _count(
                target["rebuild_reasons"].get(key)
            ) + _count(count)


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    hits = _count(bucket.get("hits"))
    misses = _count(bucket.get("misses"))
    total = hits + misses
    bucket["hits"] = hits
    bucket["misses"] = misses
    bucket["total"] = total
    bucket["hit_rate"] = round(hits / total, 4) if total else 0.0
    reasons = bucket.get("rebuild_reasons")
    bucket["rebuild_reasons"] = {
        str(reason): _count(count)
        for reason, count in sorted(
            (reasons or {}).items(), key=lambda pair: str(pair[0])
        )
    }
    bucket["rebuild_total"] = sum(bucket["rebuild_reasons"].values())
    return bucket


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryEvent:
    """One raw observation in the session ring."""

    kind: str
    backend: str
    payload: dict[str, Any]
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "payload": dict(_jsonable(self.payload)),
            "sequence": self.sequence,
        }


class TelemetryCollector:
    """Thread-safe, bounded collector of raw session observations."""

    def __init__(
        self, capacity: int = _DEFAULT_CAPACITY, *, max_events: int | None = None
    ) -> None:
        size = _DEFAULT_CAPACITY if max_events is None else max_events
        if capacity != _DEFAULT_CAPACITY and max_events is None:
            size = capacity
        limit = _optional_int(size)
        if limit is None or limit < 1:
            raise ValueError("capacity must be a positive integer")
        self._max_events = limit
        self._lock = threading.RLock()
        self._events: deque[TelemetryEvent] = deque(maxlen=limit)
        self._sequence = 0

    @property
    def capacity(self) -> int:
        return self._max_events

    def _record(self, kind: str, backend: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            event = TelemetryEvent(
                kind=kind,
                backend=str(backend),
                payload=dict(_jsonable(payload)),
                sequence=self._sequence,
            )
            self._events.append(event)

    def record_compile(
        self,
        *,
        backend: str,
        samples: Mapping[str, float | Mapping[str, Any]],
        source_length: int | None = None,
        monomorph: Mapping[str, Any] | None = None,
    ) -> None:
        """Store raw compile samples; phase formatting happens in ``snapshot``."""
        self._record(
            "compile",
            backend,
            {
                "samples": samples,
                "source_length": source_length,
                "monomorph": monomorph,
            },
        )

    def record_cache(
        self,
        *,
        samples: Mapping[str, Any] | Sequence[Any] | None = None,
        backend: str = "python",
        hits: int | None = None,
        misses: int | None = None,
        rebuild_reasons: Mapping[str, int] | Sequence[str] | None = None,
        scope: str | None = None,
    ) -> None:
        """Store one raw cache observation (or set of observations)."""
        self._record(
            "cache",
            backend,
            {
                "samples": samples,
                "backend": backend,
                "hits": hits,
                "misses": misses,
                "rebuild_reasons": rebuild_reasons,
                "scope": scope,
            },
        )

    def record_hotpatch(
        self,
        event: str,
        *,
        capability: str | None = None,
        epoch: int | None = None,
        diagnostic: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record an explicit hotpatch activity (nothing is probed here)."""
        payload: dict[str, Any] = {
            "event": str(event),
            "capability": capability,
            "epoch": epoch,
            "diagnostic": diagnostic,
            "detail": detail,
        }
        # RcuStats-shaped objects expose ``epoch``; keep their field if supplied.
        stats = _epoch_from_stats(detail)
        if stats is not None:
            payload["stats_epoch"] = stats
            payload["detail"] = None
        self._record(
            "hotpatch",
            capability or "hotpatch",
            payload,
        )

    def record_monomorph(self, summary: Mapping[str, Any]) -> None:
        """Store an inflation summary, a family map, or a parsed program."""
        normalized = _normalize_monomorph_summary(summary)
        self._record(
            "monomorph",
            str(normalized.get("backend") or "compiler"),
            {"summary": normalized},
        )

    def record_benchmark(
        self,
        results: Mapping[str, Any] | Sequence[Any],
        *,
        baseline: str = "python",
        unit: str = "ms",
        workload: str = "default",
    ) -> None:
        """Store raw benchmark results for formatting via the framework helper."""
        self._record(
            "benchmark",
            "compiler",
            {
                "results": results,
                "baseline": baseline,
                "unit": unit,
                "workload": workload,
            },
        )

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest raw events (redacted) as dictionaries."""
        count = max(0, _count(limit))
        with self._lock:
            selected = list(reversed(self._events))[:count]
        redacted = diag_redact_sensitive_paths([event.to_dict() for event in selected])
        return list(_scrub_residual_paths(redacted))

    def _events_of_kind(self, kinds: frozenset[str]) -> list[TelemetryEvent]:
        with self._lock:
            return [event for event in self._events if event.kind in kinds]

    def snapshot(self, *, redact: bool = True) -> dict[str, Any]:
        """Build a deterministic, JSON-safe view of everything recorded so far."""
        with self._lock:
            events = list(self._events)
        compile_events = [event for event in events if event.kind == "compile"]
        benchmark_events = [event for event in events if event.kind == "benchmark"]
        cache_events = [event for event in events if event.kind == "cache"]
        hotpatch_events = [event for event in events if event.kind == "hotpatch"]
        monomorph_events = [event for event in events if event.kind == "monomorph"]
        payload: dict[str, Any] = {
            "schema": _SCHEMA,
            "available": bool(events),
            "note": "" if events else _EMPTY_NOTE,
            "timings": _build_timings(compile_events + benchmark_events),
            "cache": _build_cache(cache_events),
            "monomorph": _build_monomorph(monomorph_events + compile_events),
            "hotpatch": hotpatch_status(hotpatch_events),
            "recent_events": [
                event.to_dict() for event in reversed(events[-_RECENT_LIMIT:])
            ],
        }
        if redact:
            return dict(_scrub_residual_paths(diag_redact_sensitive_paths(payload)))
        return payload

    def reset(self) -> None:
        """Forget all recorded observations and restart the sequence."""
        with self._lock:
            self._events.clear()
            self._sequence = 0


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _split_compile_samples(
    samples: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    timing: dict[str, Any] = {}
    benchmarks = None
    for key, value in samples.items():
        name = str(key)
        if name in _BENCHMARK_KEYS:
            benchmarks = value
            continue
        timing[name] = value
    return timing, benchmarks


def _build_timings(events: Sequence[TelemetryEvent]) -> dict[str, Any]:
    merged_seconds: dict[str, float] = {}
    samples = 0
    benchmark_payload: Mapping[str, Any] | None = None
    for event in events:
        payload = _payload_of(event.to_dict())
        if event.kind == "benchmark":
            benchmark_payload = payload
            continue
        samples += 1
        raw_samples = payload.get("samples")
        timing_samples, benchmarks = _split_compile_samples(
            raw_samples if isinstance(raw_samples, Mapping) else {}
        )
        if benchmarks is not None:
            benchmark_payload = {
                "results": benchmarks,
                "baseline": payload.get("benchmark_baseline", "python"),
                "unit": "ms",
                "workload": "default",
            }
        if not timing_samples:
            continue
        report = diag_record_compile_phase_timing(timing_samples)
        for phase, milliseconds in report["phase_ms"].items():
            value = _optional_float(milliseconds)
            if value is not None:
                merged_seconds[phase] = merged_seconds.get(phase, 0.0) + value / 1000.0

    if merged_seconds:
        report = diag_record_compile_phase_timing(merged_seconds)
        timings: dict[str, Any] = {
            "phases": report["phases"],
            "phase_ms": report["phase_ms"],
            "total_ms": report["total_ms"],
            "critical_phase": report["critical_phase"],
            "samples": samples,
        }
    else:
        timings = {
            "phases": {},
            "phase_ms": {},
            "total_ms": None,
            "critical_phase": None,
            "samples": samples,
        }

    if benchmark_payload is not None:
        results = benchmark_payload.get("results", benchmark_payload)
        try:
            timings["benchmarks"] = diag_export_benchmark_summary(
                results,
                baseline=str(benchmark_payload.get("baseline", "python")),
                unit=str(benchmark_payload.get("unit", "ms")),
                workload=str(benchmark_payload.get("workload", "default")),
            )
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as error:
            timings["benchmarks"] = {"error": str(error)}
    return timings


def _build_cache(events: Sequence[TelemetryEvent]) -> dict[str, Any]:
    backends: dict[str, dict[str, Any]] = {}
    scopes: dict[str, dict[str, Any]] = {}
    reasons: dict[str, int] = {}
    for event in events:
        raw = event.to_dict()
        payload = _payload_of(raw)
        report = diag_record_cache_hit_metrics(
            samples=payload.get("samples"),
            backend=str(payload.get("backend") or "python"),
            hits=payload.get("hits"),
            misses=payload.get("misses"),
            rebuild_reasons=payload.get("rebuild_reasons"),
        )
        scope = _safe_scope(payload.get("scope"))
        scope_bucket = scopes.setdefault(
            scope,
            {
                "scope": scope,
                "hits": 0,
                "misses": 0,
                "rebuild_reasons": {},
                "backends": {},
            },
        )
        scope_bucket["hits"] += _count(report.get("hits"))
        scope_bucket["misses"] += _count(report.get("misses"))
        for reason, count in (report.get("rebuild_reasons") or {}).items():
            key = str(reason)
            scope_bucket["rebuild_reasons"][key] = _count(
                scope_bucket["rebuild_reasons"].get(key)
            ) + _count(count)
            reasons[key] = _count(reasons.get(key)) + _count(count)
        for name, info in (report.get("backends") or {}).items():
            bucket = backends.setdefault(str(name), _merged_bucket())
            _merge_bucket(bucket, info)
            scope_backend = scope_bucket["backends"].setdefault(
                str(name), _merged_bucket()
            )
            _merge_bucket(scope_backend, info)

    backend_view = {name: _finalize_bucket(backends[name]) for name in sorted(backends)}
    scope_view = [_finalize_bucket(scopes[name]) for name in sorted(scopes)]
    total_hits = sum(bucket["hits"] for bucket in backend_view.values())
    total_misses = sum(bucket["misses"] for bucket in backend_view.values())
    total = total_hits + total_misses
    return {
        "backends": backend_view,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "hit_rate": round(total_hits / total, 4) if total else 0.0,
        "rebuild_reasons": {str(reason): reasons[reason] for reason in sorted(reasons)},
        "scopes": scope_view,
    }


def _family_entry(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        count = _count(
            value.get(
                "instantiations",
                value.get("specializations", value.get("count", 0)),
            )
        )
        generic = _count(value.get("generic_nodes"))
        specialized = _count(value.get("specialized_nodes"))
        growth = _count(value.get("growth_nodes", max(0, specialized - generic)))
    else:
        count = 1 if isinstance(value, str) else _count(value)
        generic = 0
        specialized = 0
        growth = 0
    return {
        "family": name,
        "instantiations": count,
        "specializations": count,
        "generic_nodes": generic,
        "specialized_nodes": specialized,
        "growth_nodes": growth,
    }


def _families_from_sequence(items: Sequence[Any]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, Mapping):
            name = item.get("family", item.get("name"))
            if name is None:
                continue
            key = str(name)
            families[key] = _family_entry(key, item)
        elif isinstance(item, str):
            families[item] = _family_entry(item, 1)
    return families


def _normalize_monomorph_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize inflation reports, family maps, and parsed programs."""
    data = dict(summary)
    report_object = data.get("inflation_report", data.get("report"))
    as_dict = getattr(report_object, "as_dict", None)
    if callable(as_dict):
        data = dict(as_dict())
    elif "program" in data and hasattr(data["program"], "items"):
        from compiler.types.framework import monomorph_emit_inflation_report

        limits = {
            key: _count(data[key])
            for key in ("max_instantiations", "max_growth_nodes")
            if key in data
        }
        data = monomorph_emit_inflation_report(data["program"], **limits).as_dict()

    families: dict[str, dict[str, Any]] = {}
    raw_families = data.get("families")
    if isinstance(raw_families, Mapping):
        for name, value in raw_families.items():
            key = str(name)
            families[key] = _family_entry(key, value)
    elif isinstance(raw_families, Sequence) and not isinstance(
        raw_families, (str, bytes)
    ):
        families.update(_families_from_sequence(raw_families))
    else:
        for name, value in data.items():
            if str(name) in _MONO_RESERVED:
                continue
            key = str(name)
            families[key] = _family_entry(key, value)

    total = _count(
        data.get(
            "total_specializations",
            data.get(
                "total_instantiations",
                sum(f["instantiations"] for f in families.values()),
            ),
        )
    )
    warnings = data.get("warnings")
    warning_list = (
        [str(item) for item in warnings]
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes))
        else []
    )
    return {
        "backend": str(data.get("backend") or "compiler"),
        "families": {name: families[name] for name in sorted(families)},
        "total_specializations": total,
        "samples": 1,
        "warnings": warning_list,
    }


def _build_monomorph(events: Sequence[TelemetryEvent]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    samples = 0
    for event in events:
        if event.kind == "compile":
            candidate = _payload_of(event.to_dict()).get("monomorph")
            if candidate is None:
                continue
            summary = (
                _normalize_monomorph_summary(candidate)
                if isinstance(candidate, Mapping)
                else {"families": {}}
            )
        else:
            payload = _payload_of(event.to_dict())
            summary = payload.get("summary") or {"families": {}}
        samples += 1
        for name, entry in (summary.get("families") or {}).items():
            bucket = families.setdefault(
                str(name),
                {
                    "instantiations": 0,
                    "specializations": 0,
                    "generic_nodes": 0,
                    "specialized_nodes": 0,
                    "growth_nodes": 0,
                },
            )
            for field in tuple(bucket):
                bucket[field] += _count(entry.get(field))
            bucket["specializations"] = bucket["instantiations"]
    return {
        "families": {name: families[name] for name in sorted(families)},
        "total_specializations": sum(
            bucket["instantiations"] for bucket in families.values()
        ),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Hotpatch status (explicit observations only)
# ---------------------------------------------------------------------------


def _epoch_from_stats(stats: Any) -> int | None:
    """Read the ``epoch`` field exposed by ``RcuStats``/``AstEpochManager``."""
    if stats is None:
        return None
    if isinstance(stats, Mapping):
        return _optional_int(stats.get("epoch"))
    return _optional_int(getattr(stats, "epoch", None))


def _epoch_from_text(text: str) -> int | None:
    match = _EPOCH_PATTERN.search(text)
    return _optional_int(match.group(1)) if match else None


def hotpatch_status(
    events: Sequence[TelemetryEvent | Mapping[str, Any]] | None = None,
    *,
    collector: TelemetryCollector | None = None,
) -> dict[str, Any]:
    """Summarize recorded hotpatch activity and declared capabilities.

    Capabilities describe real, honest boundaries from ``runtime.hotpatch``;
    ``active`` only means a caller recorded an event for that capability.  No
    runtime probe or MAP_JIT/PEP 667/TOCTOU claim is made here.
    """
    if events is None:
        source = collector if collector is not None else _collector
        events = source._events_of_kind(frozenset({"hotpatch"}))

    capabilities: dict[str, dict[str, Any]] = {
        name: {
            **{field: str(info.get(field, "")) for field in _CAPABILITY_FIELDS},
            "active": False,
        }
        for name, info in sorted(CAPABILITIES.items())
    }
    epoch: int | None = None
    rollbacks = 0
    migrations = 0
    latest_diagnostic: str | None = None
    activity: list[dict[str, Any]] = []
    ordered = [_event_dict(event) for event in events]
    for raw in ordered:
        payload = _payload_of(raw)
        name = str(payload.get("event") or raw.get("kind") or "")
        detail = payload.get("detail")
        capability = payload.get("capability")
        if capability in capabilities:
            capabilities[capability]["active"] = True
        candidate = _epoch_from_stats(payload.get("stats"))
        if candidate is None:
            candidate = _epoch_from_stats(payload.get("stats_epoch"))
        if candidate is None:
            candidate = _optional_int(payload.get("epoch"))
        if candidate is None:
            candidate = _epoch_from_text(f"{name} {detail or ''}")
        if candidate is not None:
            epoch = candidate
        lowered = f"{name} {detail or ''}".lower()
        if "rollback" in lowered:
            rollbacks += 1
        if "migrat" in lowered:
            migrations += 1
        diagnostic = payload.get("diagnostic")
        if diagnostic:
            latest_diagnostic = str(diagnostic)
        activity.append(
            {
                "kind": str(raw.get("kind") or "hotpatch"),
                "backend": str(raw.get("backend") or capability or "hotpatch"),
                "sequence": _count(raw.get("sequence")),
                "event": name,
                "capability": capability,
                "epoch": _optional_int(payload.get("epoch")),
                "diagnostic": diagnostic,
                "detail": detail,
            }
        )

    return {
        "capabilities": capabilities,
        "platform": platform_support(),
        "epoch": epoch,
        "rollbacks": rollbacks,
        "migrations": migrations,
        "latest_diagnostic": latest_diagnostic,
        "events": list(reversed(activity)),
    }


# ---------------------------------------------------------------------------
# Module-level collector and convenience API
# ---------------------------------------------------------------------------

_collector = TelemetryCollector()


def collector() -> TelemetryCollector:
    """Return the process-wide telemetry collector."""
    return _collector


def snapshot(*, redact: bool = True) -> dict[str, Any]:
    """Return a deterministic snapshot of the process-wide collector."""
    return _collector.snapshot(redact=redact)


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------


def _paint(text: str, color: bool, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[{_ANSI['reset']}m" if color else text


def _heading(text: str, color: bool) -> str:
    return _paint(text, color, _ANSI["heading"])


def _value(text: str, color: bool) -> str:
    return _paint(text, color, _ANSI["value"])


def _warn(text: str, color: bool) -> str:
    return _paint(text, color, _ANSI["warn"])


def _row(label: str, value: Any, color: bool, *, warning: bool = False) -> str:
    text = f"{'-' if value is None or value == '' else value:>10}"
    painted = _warn(text, color) if warning else _value(text, color)
    return f"  {label:<26}{painted}"


def _section_head(payload: Mapping[str, Any], title: str, color: bool) -> list[str]:
    return [
        _heading(title, color),
        _row("schema", str(payload.get("schema", _SCHEMA)), color),
    ]


def _empty_lines(payload: Mapping[str, Any], color: bool, *, empty: bool) -> list[str]:
    if not empty:
        return []
    return [
        _row("status", _EMPTY_NOTE, color, warning=True),
        _row("warnings", "none", color, warning=True),
    ]


def _format_telemetry_section(payload: Mapping[str, Any], color: bool) -> list[str]:
    timings = payload.get("timings") or {}
    monomorph = payload.get("monomorph") or {}
    empty = not payload.get("available") or not (timings or monomorph)
    lines = _section_head(payload, "Telemetry / compile timings", color)
    lines.extend(_empty_lines(payload, color, empty=empty))

    phase_ms = timings.get("phase_ms") or {}
    lines.append("")
    lines.append(_heading("Phases (ms)", color))
    if phase_ms:
        for phase in sorted(phase_ms):
            value = _optional_float(phase_ms[phase])
            text = "-" if value is None else f"{value:.3f}"
            lines.append(f"  {phase!s:<26}{_value(f'{text:>10}', color)}")
    else:
        lines.append(_row("phases", "none", color, warning=True))
    total = _optional_float(timings.get("total_ms"))
    lines.append(_row("total", "-" if total is None else f"{total:.3f}", color))
    lines.append(_row("critical phase", timings.get("critical_phase"), color))
    lines.append(_row("samples", _count(timings.get("samples")), color))
    benchmarks = timings.get("benchmarks")
    if isinstance(benchmarks, Mapping) and "error" not in benchmarks:
        workloads = benchmarks.get("workloads") or {}
        for workload in sorted(workloads):
            group = workloads[workload] or {}
            fastest = group.get("fastest") or {}
            lines.append(
                _row(
                    f"fastest ({workload})",
                    f"{fastest.get('backend', '-')} {fastest.get('opt_level', '-')}",
                    color,
                )
            )
    elif isinstance(benchmarks, Mapping):
        lines.append(
            _row("benchmarks", str(benchmarks.get("error")), color, warning=True)
        )

    lines.append("")
    lines.append(_heading("Monomorph specializations", color))
    families = monomorph.get("families") or {}
    if families:
        for name in sorted(families):
            entry = families[name]
            count = (
                _count(entry.get("instantiations"))
                if isinstance(entry, Mapping)
                else _count(entry)
            )
            growth = (
                _count(entry.get("growth_nodes")) if isinstance(entry, Mapping) else 0
            )
            suffix = f" (growth {growth})" if growth else ""
            lines.append(f"  {name!s:<26}{_value(f'{count:>10}', color)}{suffix}")
    else:
        lines.append(_row("families", "none", color, warning=True))
    lines.append(
        _row(
            "specializations",
            _count(monomorph.get("total_specializations")),
            color,
        )
    )
    return lines


def _format_cache_section(payload: Mapping[str, Any], color: bool) -> list[str]:
    cache = payload.get("cache") or {}
    empty = not payload.get("available") or not (
        cache.get("backends") or cache.get("scopes") or cache.get("rebuild_reasons")
    )
    lines = _section_head(payload, "Cache", color)
    lines.extend(_empty_lines(payload, color, empty=empty))

    rate = _optional_float(cache.get("hit_rate")) or 0.0
    lines.append(_row("hit rate", f"{rate:.4f}", color))
    lines.append(
        _row(
            "hits / misses",
            f"{_count(cache.get('total_hits'))} / {_count(cache.get('total_misses'))}",
            color,
        )
    )

    lines.append("")
    lines.append(_heading("Hits/misses by scope", color))
    scopes = cache.get("scopes") or []
    if scopes:
        lines.append(f"  {'scope':<26}{'hits':>8}{'misses':>8}{'hit rate':>10}")
        for scope in scopes:
            name = str(scope.get("scope", "default"))
            scope_rate = _optional_float(scope.get("hit_rate")) or 0.0
            lines.append(
                f"  {name:<26}"
                f"{_value(f'{_count(scope.get("hits")):>8}', color)}"
                f"{_value(f'{_count(scope.get("misses")):>8}', color)}"
                f"{_value(f'{scope_rate:>10.4f}', color)}"
            )
    else:
        lines.append(_row("scopes", "none", color, warning=True))

    lines.append("")
    lines.append(_heading("Backends", color))
    backends = cache.get("backends") or {}
    if backends:
        for name in sorted(backends):
            info = backends[name] or {}
            backend_rate = _optional_float(info.get("hit_rate")) or 0.0
            lines.append(
                f"  {name!s:<26}"
                f"{_value(f'{_count(info.get("hits")):>6}', color)} / "
                f"{_value(f'{_count(info.get("misses")):<6}', color)}"
                f"{_value(f'{backend_rate:>10.4f}', color)}"
            )
    else:
        lines.append(_row("backends", "none", color, warning=True))

    reasons = cache.get("rebuild_reasons") or {}
    lines.append("")
    lines.append(_heading("Rebuild reasons", color))
    if reasons:
        for reason in sorted(reasons):
            lines.append(
                f"  {reason!s:<26}{_value(f'{_count(reasons[reason]):>10}', color)}"
            )
    else:
        lines.append(_row("reasons", "none", color, warning=True))
    return lines


def _format_hotpatch_section(payload: Mapping[str, Any], color: bool) -> list[str]:
    hotpatch = payload.get("hotpatch") or {}
    capabilities = hotpatch.get("capabilities") or {}
    empty = not payload.get("available") or not hotpatch.get("events")
    lines = _section_head(payload, "Hotpatch", color)
    lines.extend(_empty_lines(payload, color, empty=empty))

    epoch = hotpatch.get("epoch")
    active = sum(
        1
        for info in capabilities.values()
        if isinstance(info, Mapping) and info.get("active")
    )
    lines.append(_row("RCU epoch", epoch, color))
    lines.append(_row("active capabilities", f"{active}/{len(capabilities)}", color))
    lines.append(_row("rollbacks", _count(hotpatch.get("rollbacks")), color))
    lines.append(_row("migrations", _count(hotpatch.get("migrations")), color))
    lines.append(
        _row(
            "latest diagnostic",
            hotpatch.get("latest_diagnostic") or "none",
            color,
            warning=bool(hotpatch.get("latest_diagnostic")),
        )
    )
    platform = hotpatch.get("platform") or {}
    lines.append(_row("platform", platform.get("platform", "-"), color))
    lines.append("")
    lines.append(_heading("Capabilities (declared boundaries)", color))
    for name in sorted(capabilities):
        info = capabilities[name] or {}
        marker = "+" if info.get("active") else "-"
        lines.append(f"  [{marker}] {name}")
        lines.append(f"      mechanism: {info.get('mechanism', '-')}")
        lines.append(f"      boundary: {info.get('boundary', '-')}")
        lines.append(f"      diagnostic: {info.get('diagnostic', '-')}")
    return lines


def format_terminal(
    payload: dict[str, Any], *, section: str = "all", color: bool = False
) -> str:
    """Render a telemetry snapshot as aligned, plain terminal text."""
    requested = "telemetry" if section == "timings" else section
    if requested not in {"all", "telemetry", "cache", "hotpatch"}:
        raise ValueError(
            "section must be one of all, telemetry, timings, cache, hotpatch"
        )
    lines: list[str] = []
    blocks: list[list[str]] = []
    if requested in {"all", "telemetry"}:
        blocks.append(_format_telemetry_section(payload, color))
    if requested in {"all", "cache"}:
        blocks.append(_format_cache_section(payload, color))
    if requested in {"all", "hotpatch"}:
        blocks.append(_format_hotpatch_section(payload, color))
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend(block)
    return "\n".join(lines)


__all__ = [
    "TelemetryCollector",
    "TelemetryEvent",
    "collector",
    "format_terminal",
    "hotpatch_status",
    "snapshot",
]
