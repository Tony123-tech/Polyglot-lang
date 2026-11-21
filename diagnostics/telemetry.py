"""Deterministic telemetry helpers for the framework rollout.

Telemetry is represented as a plain, JSON-safe ``dict``.  All samples are
explicit numbers, explicit mappings, or callables measured with a caller-
supplied clock/wrapper; this module never reads the wall clock, probes the
environment, or starts a thread.  Backend benchmark helpers compute *models*
from supplied durations rather than executing backend code.  Invalid inputs
raise :class:`~compiler.frontend.errors.PolyglotError`: ``E003`` for API misuse,
``E004`` for toolchain failures, ``E005`` for trace cycles, ``E006`` for missing
trace artifacts, ``E007`` for unsupported benchmark targets, and ``E008`` for
malformed serialized telemetry.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "dtele_benchmark_compare",
    "dtele_benchmark_model",
    "dtele_cache_metrics",
    "dtele_counter",
    "dtele_critical_path",
    "dtele_deserialize",
    "dtele_digest",
    "dtele_gauge",
    "dtele_histogram",
    "dtele_merge",
    "dtele_new",
    "dtele_percentile",
    "dtele_phase_timings",
    "dtele_rate",
    "dtele_redact_tags",
    "dtele_regression",
    "dtele_sample",
    "dtele_serialize",
    "dtele_span_duration",
    "dtele_stats",
    "dtele_summary",
    "dtele_threshold_check",
    "dtele_timing",
    "dtele_trace_span",
    "dtele_trace_tree",
]

_TELEMETRY_SCHEMA: Final = "polyglot-telemetry/v2"
_BENCHMARK_SCHEMA: Final = "polyglot-benchmark-model/v2"
_CANONICAL_PHASES: Final[tuple[str, ...]] = (
    "lexing",
    "parsing",
    "typecheck",
    "monomorphization",
    "codegen",
)
_PHASE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lex": "lexing",
        "tokenize": "lexing",
        "parse": "parsing",
        "type": "typecheck",
        "types": "typecheck",
        "mono": "monomorphization",
        "emit": "codegen",
    }
)
_BACKEND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"py": "python", "cpython": "python", "rs": "rust", "native": "rust"}
)
_SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset(
    {"python", "rust", "zig", "wasm"}
)
_UNIT_SCALE: Final[Mapping[str, float]] = MappingProxyType(
    {
        "s": 1.0,
        "sec": 1.0,
        "seconds": 1.0,
        "ms": 0.001,
        "msec": 0.001,
        "milliseconds": 0.001,
        "us": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanoseconds": 1e-9,
    }
)
_TELEMETRY_KEYS: Final[tuple[str, ...]] = (
    "schema",
    "name",
    "tags",
    "samples",
    "counters",
    "gauges",
    "events",
)


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured telemetry error for invalid input."""
    raise PolyglotError(code, message, phase="telemetry")


def _text(value: object, label: str) -> str:
    """Validate a required string argument."""
    if not isinstance(value, str):
        _fail("E003", f"{label} must be a str")
    return value


def _integer(value: object, label: str) -> int:
    """Validate an integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{label} must be an int")
    return value


def _nonnegative(value: object, label: str) -> int:
    """Validate a non-negative integer argument."""
    number = _integer(value, label)
    if number < 0:
        _fail("E003", f"{label} must not be negative")
    return number


def _finite(value: object, label: str) -> float:
    """Validate a finite numeric argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E003", f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail("E003", f"{label} must be a finite number")
    return number


def _scale(unit: str) -> float:
    """Resolve a duration unit to seconds."""
    text = _text(unit, "unit").strip().lower()
    if text not in _UNIT_SCALE:
        _fail("E003", f"unknown duration unit {unit!r}")
    return _UNIT_SCALE[text]


def _seconds(value: object, *, unit: str = "s", label: str = "duration") -> float:
    """Convert an explicit numeric or mapping sample to non-negative seconds."""
    if isinstance(value, Mapping):
        raw = value.get("failure") or value.get("error")
        if raw:
            _fail("E004", f"toolchain failure in {label}: {raw}")
        for key, scale in (
            ("seconds", 1.0),
            ("duration_seconds", 1.0),
            ("ms", 0.001),
            ("duration_ms", 0.001),
            ("duration", _scale(unit)),
        ):
            if key in value:
                result = _finite(value[key], key) * scale
                break
        else:
            _fail("E003", f"{label} mapping has no duration field")
    else:
        result = _finite(value, label) * _scale(unit)
    if result < 0:
        _fail("E003", f"{label} must not be negative")
    return result


def _tags(value: object, label: str = "tags") -> dict[str, str]:
    """Validate a string-to-string tag mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    return {
        _text(str(key), f"{label} key"): _text(item, f"{label}[{key}]")
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _telemetry(value: object) -> dict[str, Any]:
    """Validate a telemetry mapping and return a defensive copy."""
    if not isinstance(value, Mapping):
        _fail("E003", "telemetry must be a mapping")
    raw: Mapping[str, Any] = value
    missing = [key for key in _TELEMETRY_KEYS if key not in raw]
    if missing:
        _fail("E003", f"telemetry is missing keys: {', '.join(missing)}")
    if not isinstance(raw["samples"], Mapping) or not isinstance(
        raw["counters"], Mapping
    ):
        _fail("E003", "samples and counters must be mappings")
    if not isinstance(raw["gauges"], Mapping):
        _fail("E003", "gauges must be a mapping")
    samples: dict[str, list[float]] = {}
    for name, values in raw["samples"].items():
        if not isinstance(values, (list, tuple)):
            _fail("E003", f"sample series {name!r} must be a sequence")
        samples[_text(str(name), "metric name")] = [
            _finite(item, f"sample[{name}]") for item in values
        ]
    counters = {
        _text(str(name), "counter name"): _nonnegative(item, f"counter[{name}]")
        for name, item in raw["counters"].items()
    }
    gauges = {
        _text(str(name), "gauge name"): _finite(item, f"gauge[{name}]")
        for name, item in raw["gauges"].items()
    }
    if not isinstance(raw["events"], (list, tuple)):
        _fail("E003", "events must be a sequence")
    return {
        "schema": _text(raw["schema"], "telemetry.schema"),
        "name": _text(raw["name"], "telemetry.name"),
        "tags": _tags(raw["tags"]),
        "samples": dict(sorted(samples.items())),
        "counters": dict(sorted(counters.items())),
        "gauges": dict(sorted(gauges.items())),
        "events": [dict(event) for event in raw["events"]],
    }


def _phase(value: object) -> str:
    """Normalize a compiler phase name through the shared alias table."""
    text = _text(value, "phase").strip().lower()
    return _PHASE_ALIASES.get(text, text or "compile")


def _backend(value: object) -> str:
    """Normalize a backend name through the shared alias table."""
    text = _text(value, "backend").strip().lower()
    return _BACKEND_ALIASES.get(text, text or "python")


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-safe value with stable key and separator ordering."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dtele_new(
    name: str = "default",
    tags: Mapping[str, str] | None = None,
    *,
    schema: str = _TELEMETRY_SCHEMA,
) -> dict[str, Any]:
    """Create an empty telemetry accumulator with caller-supplied identity."""
    return {
        "schema": _text(schema, "schema"),
        "name": _text(name, "name"),
        "tags": _tags(tags),
        "samples": {},
        "counters": {},
        "gauges": {},
        "events": [],
    }


def dtele_sample(
    telemetry: Mapping[str, Any],
    metric: str,
    value: object,
    *,
    unit: str = "s",
    tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Append an explicit numeric sample to one metric series."""
    state = _telemetry(telemetry)
    name = _text(metric, "metric")
    if not name.strip():
        _fail("E003", "metric must not be empty")
    measured = _seconds(value, unit=unit, label=f"sample[{name}]")
    state["samples"].setdefault(name, []).append(measured)
    state["samples"] = dict(sorted(state["samples"].items()))
    event = {"kind": "sample", "metric": name, "seconds": measured, "tags": _tags(tags)}
    if tags:
        state["events"].append(event)
    return {
        "telemetry": state,
        "metric": name,
        "seconds": measured,
        "count": len(state["samples"][name]),
    }


def dtele_counter(
    telemetry: Mapping[str, Any], key: str, amount: int = 1
) -> dict[str, Any]:
    """Increase a named counter by a caller-supplied integer delta."""
    state = _telemetry(telemetry)
    name = _text(key, "counter key")
    delta = _integer(amount, "amount")
    previous = state["counters"].get(name, 0)
    state["counters"][name] = previous + delta
    state["counters"] = dict(sorted(state["counters"].items()))
    return {
        "telemetry": state,
        "key": name,
        "value": state["counters"][name],
        "delta": delta,
    }


def dtele_gauge(
    telemetry: Mapping[str, Any], key: str, value: object
) -> dict[str, Any]:
    """Set a named gauge to a finite numeric reading."""
    state = _telemetry(telemetry)
    name = _text(key, "gauge key")
    reading = _finite(value, "gauge value")
    previous = state["gauges"].get(name)
    state["gauges"][name] = reading
    state["gauges"] = dict(sorted(state["gauges"].items()))
    return {
        "telemetry": state,
        "key": name,
        "value": reading,
        "previous": previous,
    }


def dtele_timing(
    telemetry: Mapping[str, Any],
    phase: str,
    sample: object,
    *,
    clock: Callable[[], float] | None = None,
    wrapper: Callable[[str, object], object] | None = None,
    unit: str = "s",
) -> dict[str, Any]:
    """Record a phase duration from an explicit sample or injected clock."""
    state = _telemetry(telemetry)
    name = _phase(phase)
    observed = callable(sample)
    if observed:
        if wrapper is not None:
            outcome = wrapper(name, sample)
            measured = _seconds(outcome, unit=unit, label=f"timing[{name}]")
        elif clock is not None:
            started = _finite(clock(), "clock start")
            result = sample()
            ended = _finite(clock(), "clock end")
            measured = ended - started
            if result is None:
                measured = max(0.0, measured)
            if measured < 0:
                _fail("E003", f"{name} clock moved backwards")
        else:
            _fail("E003", "callable timing samples require clock or wrapper")
    else:
        measured = _seconds(sample, unit=unit, label=f"timing[{name}]")
    added = dtele_sample(state, f"timing.{name}", measured)
    state = added["telemetry"]
    state["events"].append(
        {"kind": "timing", "phase": name, "seconds": measured, "observed": observed}
    )
    return {
        "telemetry": state,
        "phase": name,
        "seconds": measured,
        "observed": observed,
        "alias": _text(phase, "phase"),
    }


def dtele_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two telemetry accumulators without mutating either input."""
    first, second = _telemetry(left), _telemetry(right)
    if first["schema"] != second["schema"]:
        _fail("E003", "cannot merge telemetry with different schemas")
    merged = dtele_new(
        first["name"], {**first["tags"], **second["tags"]}, schema=first["schema"]
    )
    for name in sorted(set(first["samples"]) | set(second["samples"])):
        merged["samples"][name] = [
            *first["samples"].get(name, []),
            *second["samples"].get(name, []),
        ]
    for name in sorted(set(first["counters"]) | set(second["counters"])):
        merged["counters"][name] = first["counters"].get(name, 0) + second[
            "counters"
        ].get(name, 0)
    merged["gauges"] = dict(sorted({**first["gauges"], **second["gauges"]}.items()))
    merged["events"] = [*first["events"], *second["events"]]
    return {
        "telemetry": merged,
        "left_name": first["name"],
        "right_name": second["name"],
        "sample_series": len(merged["samples"]),
        "event_count": len(merged["events"]),
    }


def dtele_percentile(
    values: Sequence[object], percentile: object, *, unit: str = "s"
) -> dict[str, Any]:
    """Return a nearest-rank percentile of explicit numeric samples."""
    if not isinstance(values, (list, tuple)) or not values:
        _fail("E003", "values must be a non-empty sequence")
    rank = _finite(percentile, "percentile")
    if not 0.0 <= rank <= 100.0:
        _fail("E003", "percentile must be between 0 and 100")
    measured = sorted(_finite(item, "value") for item in values)
    index = max(0, math.ceil(rank / 100.0 * len(measured)) - 1)
    selected = measured[min(index, len(measured) - 1)]
    return {
        "percentile": rank,
        "value": selected,
        "seconds": selected * _scale(unit),
        "method": "nearest-rank",
        "count": len(measured),
    }


def dtele_stats(values: Sequence[object], *, unit: str = "s") -> dict[str, Any]:
    """Compute deterministic summary statistics for explicit samples."""
    if not isinstance(values, (list, tuple)) or not values:
        _fail("E003", "values must be a non-empty sequence")
    measured = [_finite(item, "value") for item in values]
    scale = _scale(unit)
    return {
        "count": len(measured),
        "mean": statistics.fmean(measured),
        "median": statistics.median(measured),
        "min": min(measured),
        "max": max(measured),
        "stdev": statistics.pstdev(measured),
        "seconds": statistics.fmean(measured) * scale,
    }


def dtele_histogram(
    values: Sequence[object],
    *,
    boundaries: Sequence[object] = (0.001, 0.01, 0.1, 1.0),
    unit: str = "s",
) -> dict[str, Any]:
    """Bucket explicit samples into caller-defined duration boundaries."""
    if not isinstance(values, (list, tuple)):
        _fail("E003", "values must be a sequence")
    if not isinstance(boundaries, (list, tuple)):
        _fail("E003", "boundaries must be a sequence")
    edges = [_finite(item, "boundary") for item in boundaries]
    if not edges or any(right <= left for left, right in itertools.pairwise(edges)):
        _fail("E003", "boundaries must be strictly increasing")
    buckets: list[dict[str, Any]] = [
        {"lower": lower, "upper": upper, "count": 0}
        for lower, upper in zip([0.0, *edges], edges)
    ]
    underflow = overflow = 0
    for item in values:
        value = _finite(item, "value")
        if value < edges[0]:
            underflow += 1
        elif value >= edges[-1]:
            overflow += 1
        else:
            for index, bucket in enumerate(buckets):
                if bucket["lower"] <= value < bucket["upper"]:
                    bucket["count"] += 1
                    break
    return {
        "buckets": buckets,
        "underflow": underflow,
        "overflow": overflow,
        "count": len(values),
        "unit": unit,
    }


def dtele_rate(hits: object, misses: object, *, label: str = "cache") -> dict[str, Any]:
    """Compute a hit rate from explicit non-negative hit and miss counts."""
    name = _text(label, "label")
    hit_count = _nonnegative(hits, "hits")
    miss_count = _nonnegative(misses, "misses")
    total = hit_count + miss_count
    return {
        "label": name,
        "hits": hit_count,
        "misses": miss_count,
        "total": total,
        "rate": round(hit_count / total, 6) if total else 0.0,
    }


def dtele_phase_timings(
    samples: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    *,
    phases: Sequence[str] | None = None,
    unit: str = "s",
) -> dict[str, Any]:
    """Build a canonically ordered per-pass timing report from explicit data."""
    if samples is None:
        pairs: Sequence[tuple[str, object]] = ()
    elif isinstance(samples, Mapping):
        pairs = list(samples.items())
    elif isinstance(samples, (list, tuple)):
        pairs = list(samples)
    else:
        _fail("E003", "samples must be a mapping, a sequence, or None")
    supplied: dict[str, tuple[str, object]] = {}
    for raw_name, sample in pairs:
        name = _phase(raw_name)
        supplied[name] = (_text(str(raw_name), "phase alias"), sample)
    if phases is None:
        ordered = list(_CANONICAL_PHASES)
    else:
        if not isinstance(phases, (list, tuple)):
            _fail("E003", "phases must be a sequence")
        ordered = list(dict.fromkeys(_phase(item) for item in phases))
    for name in sorted(supplied):
        if name not in ordered:
            ordered.append(name)
    report: dict[str, Any] = {}
    measured: list[str] = []
    total = 0.0
    for name in ordered:
        if name not in supplied:
            report[name] = {
                "seconds": None,
                "ms": None,
                "alias": name,
                "observed": False,
            }
            continue
        alias, sample = supplied[name]
        duration = _seconds(sample, unit=unit, label=f"timing[{name}]")
        measured.append(name)
        total += duration
        report[name] = {
            "seconds": duration,
            "ms": duration * 1000.0,
            "alias": alias,
            "observed": True,
        }
    critical = max(
        measured, key=lambda name: (report[name]["seconds"], name), default=None
    )
    return {
        "schema": "polyglot-phase-timings/v2",
        "phases": report,
        "measured": measured,
        "total_seconds": total,
        "critical_phase": critical,
    }


def dtele_cache_metrics(
    observations: Sequence[object], *, backend: str = "python"
) -> dict[str, Any]:
    """Summarize explicit cache observations by backend and rebuild reason."""
    if not isinstance(observations, (list, tuple)):
        _fail("E003", "observations must be a sequence")
    default = _backend(backend)
    counters: dict[str, dict[str, Any]] = {
        name: {"hits": 0, "misses": 0, "reasons": {}} for name in [default]
    }
    for item in observations:
        name, is_hit, reason = default, False, None
        if isinstance(item, bool):
            is_hit = item
        elif isinstance(item, str):
            lowered = item.strip().lower()
            if lowered in {"hit", "cached", "true"}:
                is_hit = True
            elif lowered in {"miss", "cold", "false"}:
                is_hit = False
            else:
                reason = item
        elif isinstance(item, Mapping):
            name = _backend(item.get("backend") or default)
            raw = item.get("hit", item.get("cache_hit"))
            is_hit = (
                bool(raw)
                if isinstance(raw, bool)
                else str(raw).strip().lower() in {"hit", "true", "1", "cached"}
            )
            reason = item.get("reason") or item.get("rebuild_reason")
        else:
            _fail("E003", f"unsupported cache observation {item!r}")
        bucket = counters.setdefault(name, {"hits": 0, "misses": 0, "reasons": {}})
        if is_hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1
            if reason is not None:
                text = _text(str(reason), "reason")
                bucket["reasons"][text] = bucket["reasons"].get(text, 0) + 1
    summaries: dict[str, Any] = {}
    total_hits = total_misses = 0
    merged_reasons: dict[str, int] = {}
    for name in sorted(counters):
        bucket = counters[name]
        count = bucket["hits"] + bucket["misses"]
        total_hits += bucket["hits"]
        total_misses += bucket["misses"]
        reasons = dict(sorted(bucket["reasons"].items()))
        for reason, value in reasons.items():
            merged_reasons[reason] = merged_reasons.get(reason, 0) + value
        summaries[name] = {
            "hits": bucket["hits"],
            "misses": bucket["misses"],
            "total": count,
            "hit_rate": round(bucket["hits"] / count, 6) if count else 0.0,
            "rebuild_reasons": reasons,
        }
    total = total_hits + total_misses
    return {
        "schema": "polyglot-cache-metrics/v2",
        "backends": summaries,
        "hits": total_hits,
        "misses": total_misses,
        "total": total,
        "hit_rate": round(total_hits / total, 6) if total else 0.0,
        "rebuild_reasons": dict(sorted(merged_reasons.items())),
    }


def dtele_benchmark_model(
    records: Sequence[Mapping[str, object]],
    *,
    baseline: str = "python:O0",
    unit: str = "ms",
    precision: int = 3,
) -> dict[str, Any]:
    """Compare modeled backend durations without executing any backend."""
    if not isinstance(records, (list, tuple)) or not records:
        _fail("E003", "records must be a non-empty sequence")
    digits = _nonnegative(precision, "precision")
    scale = _scale(unit)
    measurements: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            _fail("E003", "each benchmark record must be a mapping")
        name = _backend(record.get("backend") or "python")
        if name not in _SUPPORTED_BACKENDS:
            _fail("E007", f"unsupported benchmark backend {name!r}")
        level = _text(str(record.get("opt_level") or "O0"), "opt_level").strip().upper()
        duration = _finite(
            record.get("duration_ms", record.get("duration")), "duration"
        )
        if duration < 0:
            _fail("E003", "duration must not be negative")
        measurements.append(
            {
                "backend": name,
                "opt_level": level,
                "duration_ms": duration * scale * 1000.0,
                "workload": _text(str(record.get("workload") or "default"), "workload"),
            }
        )
    pieces = baseline.split(":")
    base_name = _backend(pieces[0])
    base_level = pieces[1].upper() if len(pieces) > 1 else None
    candidates = [
        item
        for item in measurements
        if item["backend"] == base_name
        and (base_level is None or item["opt_level"] == base_level)
    ]
    fallback = not candidates
    if not candidates:
        candidates = list(measurements)
    reference = min(candidates, key=lambda item: (item["duration_ms"], item["backend"]))
    comparisons: list[dict[str, Any]] = []
    for item in sorted(
        measurements,
        key=lambda entry: (
            entry["workload"],
            entry["backend"],
            entry["opt_level"],
        ),
    ):
        duration = item["duration_ms"]
        comparisons.append(
            {
                **item,
                "speedup": round(reference["duration_ms"] / duration, digits)
                if duration > 0
                else None,
                "delta_ms": round(duration - reference["duration_ms"], digits),
            }
        )
    return {
        "schema": _BENCHMARK_SCHEMA,
        "baseline": f"{base_name}:{base_level}" if base_level else base_name,
        "baseline_fallback": fallback,
        "baseline_duration_ms": round(reference["duration_ms"], digits),
        "comparisons": comparisons,
        "count": len(comparisons),
    }


def dtele_benchmark_compare(
    baseline: Sequence[object], candidate: Sequence[object], *, tolerance: object = 0.05
) -> dict[str, Any]:
    """Compare two explicit timing distributions with a relative tolerance."""
    first = dtele_stats(baseline, unit="s")
    second = dtele_stats(candidate, unit="s")
    limit = _finite(tolerance, "tolerance")
    if limit < 0:
        _fail("E003", "tolerance must not be negative")
    reference = first["mean"]
    observed = second["mean"]
    ratio = observed / reference if reference > 0 else None
    return {
        "baseline_mean": reference,
        "candidate_mean": observed,
        "ratio": ratio,
        "faster": observed < reference,
        "within_tolerance": ratio is not None and abs(ratio - 1.0) <= limit,
        "tolerance": limit,
    }


def dtele_trace_span(
    span_id: str,
    name: str,
    *,
    start: object,
    end: object | None = None,
    duration: object | None = None,
    parent: str | None = None,
) -> dict[str, Any]:
    """Create an explicit trace span with a start, end, and duration."""
    identifier = _text(span_id, "span_id")
    if not identifier.strip():
        _fail("E003", "span_id must not be empty")
    label = _text(name, "name")
    beginning = _finite(start, "start")
    if beginning < 0:
        _fail("E003", "start must not be negative")
    if end is not None:
        ending = _finite(end, "end")
        if ending < beginning:
            _fail("E003", "end must not precede start")
        measured = ending - beginning
    elif duration is not None:
        measured = _finite(duration, "duration")
        if measured < 0:
            _fail("E003", "duration must not be negative")
        ending = beginning + measured
    else:
        _fail("E003", "trace span requires end or duration")
    return {
        "span_id": identifier,
        "name": label,
        "start": beginning,
        "end": ending,
        "duration": measured,
        "parent": parent,
    }


def dtele_span_duration(span: Mapping[str, object]) -> dict[str, Any]:
    """Validate and derive the duration of one explicit trace span."""
    if not isinstance(span, Mapping):
        _fail("E003", "span must be a mapping")
    normalized = dtele_trace_span(
        str(span.get("span_id", "")),
        str(span.get("name", "span")),
        start=span.get("start"),
        end=span.get("end"),
        duration=span.get("duration"),
        parent=span.get("parent"),
    )
    declared = span.get("duration")
    consistent = True
    if declared is not None:
        consistent = abs(_finite(declared, "duration") - normalized["duration"]) < 1e-9
        if not consistent:
            _fail("E003", "declared span duration disagrees with start/end")
    return {
        "span_id": normalized["span_id"],
        "name": normalized["name"],
        "start": normalized["start"],
        "end": normalized["end"],
        "duration": normalized["duration"],
        "parent": normalized["parent"],
        "consistent": consistent,
    }


def dtele_trace_tree(
    spans: Sequence[Mapping[str, object]], *, root: str | None = None
) -> dict[str, Any]:
    """Build a trace hierarchy, rejecting cycles and missing parent artifacts."""
    if not isinstance(spans, (list, tuple)):
        _fail("E003", "spans must be a sequence")
    normalized = [dtele_span_duration(span) for span in spans]
    by_id = {span["span_id"]: span for span in normalized}
    if len(by_id) != len(normalized):
        _fail("E003", "trace span ids must be unique")
    children: dict[str, list[str]] = {identifier: [] for identifier in by_id}
    for span in normalized:
        parent = span["parent"]
        if parent is None:
            continue
        if parent not in by_id:
            _fail("E006", f"missing parent trace span {parent!r}")
        children[parent].append(span["span_id"])
    for child_ids in children.values():
        child_ids.sort()

    def visit(identifier: str, stack: tuple[str, ...]) -> int:
        if identifier in stack:
            _fail("E005", f"trace cycle at span {identifier!r}")
        depth = 0
        for child in children[identifier]:
            depth = max(depth, 1 + visit(child, (*stack, identifier)))
        return depth

    # Validate every component, including components disconnected from any
    # root, so a cycle can never hide simply because no root was supplied.
    for identifier in sorted(by_id):
        visit(identifier, ())
    if root is not None:
        if root not in by_id:
            _fail("E006", f"missing root trace span {root!r}")
        roots = [root]
    else:
        roots = sorted(
            identifier for identifier in by_id if by_id[identifier]["parent"] is None
        )
    depth = max((visit(identifier, ()) for identifier in roots), default=0)
    return {
        "nodes": {
            identifier: {
                **by_id[identifier],
                "children": children[identifier],
            }
            for identifier in sorted(by_id)
        },
        "roots": roots,
        "count": len(by_id),
        "depth": depth,
    }


def dtele_critical_path(spans: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Return the highest-duration parent-to-child path through a trace."""
    tree = dtele_trace_tree(spans)
    best: tuple[float, tuple[str, ...]] = (0.0, ())

    def walk(identifier: str, path: tuple[str, ...], total: float) -> None:
        nonlocal best
        candidate_path = (*path, identifier)
        candidate_total = total + tree["nodes"][identifier]["duration"]
        if candidate_total > best[0] or (
            candidate_total == best[0] and candidate_path < best[1]
        ):
            best = (candidate_total, candidate_path)
        for child in tree["nodes"][identifier]["children"]:
            walk(child, candidate_path, candidate_total)

    for identifier in tree["roots"]:
        walk(identifier, (), 0.0)
    return {
        "span_ids": list(best[1]),
        "duration": best[0],
        "branches": len(tree["roots"]),
    }


def dtele_threshold_check(
    metrics: Mapping[str, object],
    thresholds: Mapping[str, object],
    *,
    tolerance: object = 0.0,
) -> dict[str, Any]:
    """Compare observed metrics against explicit upper or lower obligations."""
    values = {
        _text(str(name), "metric name"): _finite(value, f"metric[{name}]")
        for name, value in metrics.items()
    }
    if not isinstance(thresholds, Mapping):
        _fail("E003", "thresholds must be a mapping")
    slack = _finite(tolerance, "tolerance")
    if slack < 0:
        _fail("E003", "tolerance must not be negative")
    violations: list[dict[str, Any]] = []
    checked = 0
    for name in sorted(thresholds):
        if name not in values:
            _fail("E003", f"no observed metric for threshold {name!r}")
        spec = thresholds[name]
        if isinstance(spec, Mapping):
            maximum = spec.get("max")
            minimum = spec.get("min")
        else:
            maximum, minimum = spec, None
        observed = values[name]
        if maximum is not None:
            limit = _finite(maximum, f"max[{name}]")
            checked += 1
            if observed > limit + slack:
                violations.append(
                    {
                        "metric": name,
                        "kind": "max",
                        "observed": observed,
                        "limit": limit,
                    }
                )
        if minimum is not None:
            limit = _finite(minimum, f"min[{name}]")
            checked += 1
            if observed < limit - slack:
                violations.append(
                    {
                        "metric": name,
                        "kind": "min",
                        "observed": observed,
                        "limit": limit,
                    }
                )
    return {
        "passes": not violations,
        "violations": violations,
        "checked": checked,
        "tolerance": slack,
    }


def dtele_regression(
    baseline_samples: Sequence[object],
    candidate_samples: Sequence[object],
    *,
    tolerance: object = 0.05,
) -> dict[str, Any]:
    """Detect a relative regression between two explicit sample sets."""
    comparison = dtele_benchmark_compare(
        baseline_samples, candidate_samples, tolerance=tolerance
    )
    ratio = comparison["ratio"]
    regressed = ratio is not None and ratio > 1.0 + comparison["tolerance"]
    return {
        "regressed": regressed,
        "baseline_mean": comparison["baseline_mean"],
        "candidate_mean": comparison["candidate_mean"],
        "delta_ratio": None if ratio is None else ratio - 1.0,
        "tolerance": comparison["tolerance"],
    }


def dtele_redact_tags(
    tags: Mapping[str, str], *, prefixes: Sequence[str] = ()
) -> dict[str, Any]:
    """Redact explicit textual prefixes in tag values without environment I/O."""
    values = _tags(tags)
    if not isinstance(prefixes, (list, tuple)):
        _fail("E003", "prefixes must be a sequence")
    cleaned = [
        (_text(prefix, "prefix"), f"<redacted:{index}>")
        for index, prefix in enumerate(prefixes)
    ]
    redacted: list[str] = []
    result: dict[str, str] = {}
    for key in sorted(values):
        value = values[key]
        for prefix, placeholder in cleaned:
            if prefix and prefix in value:
                value = value.replace(prefix, placeholder)
                redacted.append(key)
        result[key] = value
    return {"tags": result, "redacted": sorted(set(redacted)), "count": len(redacted)}


def dtele_serialize(
    telemetry: Mapping[str, Any], *, pretty: bool = False
) -> dict[str, Any]:
    """Serialize telemetry to a stable JSON payload with sorted keys."""
    state = _telemetry(telemetry)
    if not isinstance(pretty, bool):
        _fail("E003", "pretty must be a bool")
    payload = json.dumps(
        state,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return {
        "payload": payload,
        "schema": state["schema"],
        "bytes": len(payload.encode("utf-8")),
    }


def dtele_deserialize(payload: str) -> dict[str, Any]:
    """Parse serialized telemetry, reporting malformed state as E008."""
    text = _text(payload, "payload")
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as error:
        _fail("E008", f"malformed telemetry payload: {error}")
    if not isinstance(parsed, dict):
        _fail("E008", "telemetry payload must decode to an object")
    try:
        return _telemetry(parsed)
    except PolyglotError as error:
        _fail("E008", f"malformed telemetry payload: {error.diagnostic.message}")


def dtele_digest(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint telemetry with a SHA-256 of its canonical JSON form."""
    payload = dtele_serialize(telemetry)["payload"].encode("utf-8")
    return {
        "digest": hashlib.sha256(payload).hexdigest(),
        "algorithm": "sha256",
        "bytes": len(payload),
        "schema": _TELEMETRY_SCHEMA,
    }


def dtele_summary(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize telemetry counts and names in a stable report."""
    state = _telemetry(telemetry)
    return {
        "schema": state["schema"],
        "name": state["name"],
        "tags": state["tags"],
        "sample_count": sum(len(values) for values in state["samples"].values()),
        "metric_names": sorted(state["samples"]),
        "counter_total": sum(state["counters"].values()),
        "counter_names": sorted(state["counters"]),
        "gauges": state["gauges"],
        "event_count": len(state["events"]),
    }
