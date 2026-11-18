"""Honest, dependency-free foundations for the distributed agent swarm pillar.

This module implements functions 91-100 as deterministic, in-memory reference
models.  It deliberately does **not** provide sockets, real multi-node
deployment, external ML, or real Ed25519 threshold signatures.  Signed
envelopes use HMAC-SHA256 with per-signer keys derived from a caller-supplied
secret, which the standard library can actually provide; requesting an Ed25519
threshold signature is rejected with E007 instead of being silently faked.

Every public entry point accepts ``span``, ``line`` and ``col`` so a caller
that owns an AST can attach exact character spans to any E001-E008 failure.
All simulations use sorted iteration, so the same input always gives the same
result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from runtime.diagnostics import PillarError

REFERENCE_SECRET: Final[bytes] = b"polyglot-lang/swarm/reference-secret-v1"
SCHEME_HMAC_DIGEST: Final[str] = "hmac-digest"
SCHEME_ED25519_THRESHOLD: Final[str] = "ed25519-threshold"
REMOVE_MARKER: Final[object] = object()

# ---------------------------------------------------------------------------
# Shared location and validation plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Location:
    span: tuple[int, int] | None
    line: int
    col: int

    def fail(self, code: str, message: str, **details: Any) -> PillarError:
        return PillarError(
            code,
            message,
            span=self.span,
            line=self.line,
            col=self.col,
            **details,
        )

    def child(
        self,
        span: tuple[int, int] | None,
        line: int,
        col: int,
    ) -> _Location:
        """Prefer a nested source position, falling back field by field."""
        return _Location(
            span if span is not None else self.span,
            line if line else self.line,
            col if col else self.col,
        )


def _validate_span(span: tuple[int, int] | None, *, loc: _Location) -> None:
    if span is None:
        return
    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in span
        )
        or span[0] > span[1]
    ):
        raise loc.fail(
            "E003",
            "source span must be a (start, end) pair of non-negative integers",
            reason=f"got {span!r}",
            help_text="use the exact character offsets of the originating AST node",
        )


def _make_location(
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> _Location:
    probe = _Location(None, line, col)
    if isinstance(line, bool) or not isinstance(line, int) or line < 0:
        raise probe.fail("E003", "line must be a non-negative integer")
    if isinstance(col, bool) or not isinstance(col, int) or col < 0:
        raise probe.fail("E003", "col must be a non-negative integer")
    _validate_span(span, loc=probe)
    return _Location(span, line, col)


def _require_mapping(value: Any, *, label: str, loc: _Location) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise loc.fail(
            "E001",
            f"{label} must be a mapping",
            reason=f"got {type(value).__name__}",
        )
    return value


def _require_sequence(value: Any, *, label: str, loc: _Location) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise loc.fail(
            "E001",
            f"{label} must be a sequence",
            reason=f"got {type(value).__name__}",
        )
    return value


def _string(
    value: Any,
    *,
    label: str,
    loc: _Location,
    code: str = "E003",
) -> str:
    if not isinstance(value, str):
        raise loc.fail(
            "E001",
            f"{label} must be a string",
            reason=f"got {type(value).__name__}",
        )
    if not value:
        raise loc.fail(code, f"{label} must not be empty")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    loc: _Location,
    minimum: int = 0,
    code: str = "E003",
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise loc.fail(
            "E001",
            f"{label} must be an integer",
            reason=f"got {type(value).__name__}",
        )
    if value < minimum:
        raise loc.fail(code, f"{label} must be >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    label: str,
    loc: _Location,
    minimum: float | None = None,
    maximum: float | None = None,
    code: str = "E008",
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise loc.fail(
            "E001",
            f"{label} must be a real number",
            reason=f"got {type(value).__name__}",
        )
    number = float(value)
    if not math.isfinite(number):
        raise loc.fail("E008", f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise loc.fail(code, f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise loc.fail(code, f"{label} must be <= {maximum}")
    return number


def _canonical_bytes(value: Any, *, label: str, loc: _Location) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as error:
        raise loc.fail(
            "E008",
            f"{label} contains a non-finite number",
            reason=str(error),
        ) from error
    except TypeError as error:
        raise loc.fail(
            "E001",
            f"{label} is not a JSON-compatible value",
            reason=str(error),
        ) from error


def _digest(value: Any, *, label: str, loc: _Location) -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label, loc=loc)).hexdigest()


def _round(value: float) -> float:
    return round(value, 12)


# ---------------------------------------------------------------------------
# 91. Gossip epidemic sync
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GossipExchange:
    round: int
    source: str
    target: str
    pushed: tuple[str, ...]
    pulled: tuple[str, ...]
    digest_ok: bool


@dataclass(frozen=True)
class GossipSyncResult:
    states: dict[str, dict[str, int]]
    rounds: int
    converged: bool
    messages: int
    exchanges: tuple[GossipExchange, ...]
    digests: dict[str, str]


def swarm_gossip_epidemic_sync(
    registry: Mapping[str, Mapping[str, int]],
    *,
    rounds: int = 8,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> GossipSyncResult:
    """Reconcile per-node key/version maps by deterministic push/pull gossip.

    Peers are paired on a rotating ring over sorted node ids.  Each exchange
    sends a digest plus pushed entries one way and pulled entries the other;
    versions merge with ``max`` (anti-entropy).  This is an in-memory model,
    not a network transport.
    """
    loc = _make_location(span, line, col)
    data = _require_mapping(registry, label="node registry", loc=loc)
    count = _integer(rounds, label="rounds", loc=loc)

    nodes = list(data)
    if not nodes:
        raise loc.fail("E003", "node registry is empty")
    for node in nodes:
        _string(node, label="node id", loc=loc, code="E008")
    nodes = sorted(nodes)

    states: dict[str, dict[str, int]] = {}
    for node in nodes:
        entries = _require_mapping(data[node], label=f"state of {node!r}", loc=loc)
        clean: dict[str, int] = {}
        for key, version in entries.items():
            _string(key, label="gossip key", loc=loc, code="E008")
            clean[key] = _integer(
                version,
                label=f"version of {key!r} on {node!r}",
                loc=loc,
                code="E008",
            )
        states[node] = clean

    def converged() -> bool:
        first = states[nodes[0]]
        return all(states[node] == first for node in nodes[1:])

    exchanges: list[GossipExchange] = []
    messages = 0
    done = converged()
    executed = 0
    for round_index in range(count):
        if done:
            break
        offset = round_index % len(nodes)
        seen_pairs: set[tuple[str, str]] = set()
        for index in range(len(nodes)):
            left_node = nodes[index]
            right_node = nodes[(index + 1 + offset) % len(nodes)]
            if left_node == right_node:
                continue
            pair = tuple(sorted((left_node, right_node)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            a, b = pair
            left, right = states[a], states[b]
            pushed = tuple(
                sorted(
                    key for key, version in left.items() if right.get(key, -1) < version
                )
            )
            pulled = tuple(
                sorted(
                    key for key, version in right.items() if left.get(key, -1) < version
                )
            )
            for key in pushed:
                right[key] = max(right.get(key, -1), left[key])
            for key in pulled:
                left[key] = max(left.get(key, -1), right[key])
            # Both advertised digests are checked before entries are merged.
            digest_ok = all(
                _digest(states[node], label=f"gossip state of {node!r}", loc=loc)
                for node in (a, b)
            )
            messages += 2
            exchanges.append(
                GossipExchange(
                    round=round_index + 1,
                    source=a,
                    target=b,
                    pushed=pushed,
                    pulled=pulled,
                    digest_ok=bool(digest_ok),
                )
            )
        executed += 1
        done = converged()

    return GossipSyncResult(
        states={node: dict(states[node]) for node in nodes},
        rounds=executed,
        converged=done,
        messages=messages,
        exchanges=tuple(exchanges),
        digests={
            node: _digest(states[node], label=f"gossip state of {node!r}", loc=loc)
            for node in nodes
        },
    )


# ---------------------------------------------------------------------------
# 92. Neural panic detector (heuristic, no external ML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetrySample:
    latency_ms: float
    error_rate: float
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class PanicAssessment:
    node: str
    status: str
    score: float
    streak: int
    peak_latency_ratio: float
    peak_error_rate: float
    reasons: tuple[str, ...]
    triggered_at: int | None


@dataclass(frozen=True)
class PanicReport:
    assessments: tuple[PanicAssessment, ...]
    panics: tuple[str, ...]
    warnings: tuple[str, ...]
    nominal: tuple[str, ...]
    scores: dict[str, float]


def swarm_neural_panic_detector(
    telemetry: Mapping[str, Sequence[TelemetrySample | tuple[float, float]]],
    *,
    warning_latency_ratio: float = 1.75,
    panic_latency_ratio: float = 3.0,
    warning_error_rate: float = 0.05,
    panic_error_rate: float = 0.40,
    panic_streak: int = 3,
    baseline_window: int = 8,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> PanicReport:
    """Flag latency/error anomalies with a fixed heuristic, never a trained model.

    The baseline is the mean latency of each node's first ``baseline_window``
    samples.
    A node panics on a critical sample or after ``panic_streak`` consecutive
    warnings.  Finite input validation uses E008; configuration mistakes use
    E003.
    """
    loc = _make_location(span, line, col)
    data = _require_mapping(telemetry, label="telemetry", loc=loc)
    warn_ratio = _number(
        warning_latency_ratio,
        label="warning_latency_ratio",
        loc=loc,
        minimum=1.0,
        code="E003",
    )
    panic_ratio = _number(
        panic_latency_ratio,
        label="panic_latency_ratio",
        loc=loc,
        minimum=warn_ratio,
        code="E003",
    )
    warn_error = _number(
        warning_error_rate,
        label="warning_error_rate",
        loc=loc,
        minimum=0.0,
        maximum=1.0,
        code="E003",
    )
    panic_error = _number(
        panic_error_rate,
        label="panic_error_rate",
        loc=loc,
        minimum=warn_error,
        maximum=1.0,
        code="E003",
    )
    streak_limit = _integer(panic_streak, label="panic_streak", loc=loc, minimum=1)
    window = _integer(baseline_window, label="baseline_window", loc=loc, minimum=1)
    nodes = list(data)
    if not nodes:
        raise loc.fail("E003", "telemetry contains no nodes")
    for node in nodes:
        _string(node, label="telemetry node id", loc=loc, code="E008")
    nodes = sorted(nodes)

    assessments: list[PanicAssessment] = []
    for node in nodes:
        raw_samples = _require_sequence(
            data[node], label=f"telemetry of {node!r}", loc=loc
        )
        if not raw_samples:
            raise loc.fail("E003", f"telemetry of {node!r} has no samples")
        samples: list[tuple[float, float]] = []
        for index, raw in enumerate(raw_samples):
            sample_loc = loc
            if isinstance(raw, TelemetrySample):
                latency = raw.latency_ms
                error_rate = raw.error_rate
                sample_loc = loc.child(raw.span, raw.line, raw.col)
            elif isinstance(raw, tuple) and len(raw) == 2:
                latency, error_rate = raw
            else:
                raise loc.fail(
                    "E001",
                    f"telemetry sample {index} of {node!r} must be a sample or pair",
                    reason=f"got {type(raw).__name__}",
                )
            samples.append(
                (
                    _number(
                        latency,
                        label=f"latency_ms of sample {index}",
                        loc=sample_loc,
                        minimum=0.0,
                    ),
                    _number(
                        error_rate,
                        label=f"error_rate of sample {index}",
                        loc=sample_loc,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                )
            )

        calibration = samples[:window]
        base_latency = sum(item[0] for item in calibration) / len(calibration)
        streak = 0
        status = "nominal"
        reasons: list[str] = []
        triggered_at: int | None = None
        peak_ratio = 0.0
        peak_errors = 0.0
        for index, (latency, error_rate) in enumerate(samples):
            ratio = latency / max(base_latency, 1.0)
            peak_ratio = max(peak_ratio, ratio)
            peak_errors = max(peak_errors, error_rate)
            warning = ratio >= warn_ratio or error_rate >= warn_error
            critical = (ratio >= panic_ratio and error_rate >= warn_error) or (
                error_rate >= panic_error
            )
            if warning or critical:
                streak += 1
            else:
                streak = 0
            if critical:
                status = "panic"
                reason = (
                    f"error_rate {error_rate:.3f} at sample {index}"
                    if error_rate >= panic_error
                    else f"latency ratio {ratio:.3f} at sample {index}"
                )
                reasons.append(reason)
                triggered_at = index
                break
            if warning and streak >= streak_limit:
                status = "panic"
                reasons.append(f"{streak} consecutive warnings at sample {index}")
                triggered_at = index
                break
            if warning:
                status = "warning"
        score = min(
            1.0,
            max(peak_ratio / panic_ratio, peak_errors / panic_error),
        )
        assessments.append(
            PanicAssessment(
                node=node,
                status=status,
                score=_round(score),
                streak=streak,
                peak_latency_ratio=_round(peak_ratio),
                peak_error_rate=_round(peak_errors),
                reasons=tuple(reasons),
                triggered_at=triggered_at,
            )
        )

    by_status = {
        status: tuple(item.node for item in assessments if item.status == status)
        for status in ("panic", "warning", "nominal")
    }
    return PanicReport(
        assessments=tuple(assessments),
        panics=by_status["panic"],
        warnings=by_status["warning"],
        nominal=by_status["nominal"],
        scores={item.node: item.score for item in assessments},
    )


# ---------------------------------------------------------------------------
# 93. Autonomous patch synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyCheck:
    name: str
    predicate: Callable[[Mapping[str, Any]], bool]
    description: str = ""
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class PatchOperation:
    path: str
    kind: str
    old: Any = None
    new: Any = None


@dataclass(frozen=True)
class CandidatePatch:
    identifier: str
    target: str
    operations: tuple[PatchOperation, ...]
    post_state: dict[str, Any]
    digest: str
    algorithm: str = "sha256-canonical-json"


@dataclass(frozen=True)
class PropertyResult:
    name: str
    passed: bool
    detail: str
    span: tuple[int, int] | None
    line: int
    col: int


@dataclass(frozen=True)
class PatchSynthesisResult:
    candidate: CandidatePatch
    checks: tuple[PropertyResult, ...]
    validated: bool


def swarm_autonomous_patch_synth(
    target: Mapping[str, Any],
    proposal: Mapping[str, Any],
    checks: Sequence[PropertyCheck],
    *,
    identifier: str = "autonomous-patch",
    target_name: str = "cluster-state",
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> PatchSynthesisResult:
    """Build one deterministic candidate patch and validate it against checks.

    The proposal is a flat mapping; ``REMOVE_MARKER`` deletes a key.  Checks run
    against the post-patch state.  A rejected check is an unmet semantic
    obligation (E002), not a silently accepted patch.
    """
    loc = _make_location(span, line, col)
    before = dict(_require_mapping(target, label="target state", loc=loc))
    proposed = dict(_require_mapping(proposal, label="proposal", loc=loc))
    raw_checks = _require_sequence(checks, label="property checks", loc=loc)
    name = _string(identifier, label="identifier", loc=loc)
    target_id = _string(target_name, label="target_name", loc=loc)
    if not raw_checks:
        raise loc.fail("E003", "at least one property check is required")
    for key in before:
        _string(key, label="target state key", loc=loc, code="E008")
    _canonical_bytes(before, label="target state", loc=loc)

    post = dict(before)
    for key, value in proposed.items():
        _string(key, label="proposal key", loc=loc, code="E008")
        if value is REMOVE_MARKER:
            post.pop(key, None)
        else:
            post[key] = value
    _canonical_bytes(post, label="post-patch state", loc=loc)

    operations: list[PatchOperation] = []
    for key in sorted(set(before) | set(proposed)):
        old = before.get(key)
        new = proposed.get(key)
        if key not in proposed:
            continue
        if new is REMOVE_MARKER:
            if key in before:
                operations.append(PatchOperation(key, "remove", old=old))
        elif key not in before:
            operations.append(PatchOperation(key, "add", new=new))
        elif _canonical_bytes(old, label=f"old value of {key!r}", loc=loc) != (
            _canonical_bytes(new, label=f"new value of {key!r}", loc=loc)
        ):
            operations.append(PatchOperation(key, "update", old=old, new=new))
    if not operations:
        raise loc.fail("E003", "proposal does not change the target state")

    digest = _digest(
        {
            "identifier": name,
            "target": target_id,
            "operations": [
                {"path": op.path, "kind": op.kind, "old": op.old, "new": op.new}
                for op in operations
            ],
            "post_state": post,
        },
        label="candidate patch",
        loc=loc,
    )
    candidate = CandidatePatch(
        identifier=name,
        target=target_id,
        operations=tuple(operations),
        post_state=post,
        digest=digest,
    )

    results: list[PropertyResult] = []
    seen_names: set[str] = set()
    for raw_check in raw_checks:
        if not isinstance(raw_check, PropertyCheck):
            raise loc.fail("E001", "each property check must be a PropertyCheck")
        check_loc = _make_location(raw_check.span, raw_check.line, raw_check.col)
        check_loc = loc.child(check_loc.span, check_loc.line, check_loc.col)
        check_name = _string(
            raw_check.name,
            label="property check name",
            loc=check_loc,
        )
        if check_name in seen_names:
            raise check_loc.fail("E003", f"duplicate property check {check_name!r}")
        seen_names.add(check_name)
        if not callable(raw_check.predicate):
            raise check_loc.fail("E001", "property predicate must be callable")
        try:
            outcome = raw_check.predicate(dict(post))
        except Exception as error:
            raise check_loc.fail(
                "E002",
                f"property check {check_name!r} raised an exception",
                reason=str(error),
            ) from error
        if not isinstance(outcome, bool):
            raise check_loc.fail(
                "E002",
                f"property check {check_name!r} did not return a bool",
                reason=f"got {type(outcome).__name__}",
            )
        results.append(
            PropertyResult(
                name=check_name,
                passed=outcome,
                detail=raw_check.description,
                span=check_loc.span,
                line=check_loc.line,
                col=check_loc.col,
            )
        )
        if not outcome:
            raise check_loc.fail(
                "E002",
                f"candidate patch violates property {check_name!r}",
                reason=raw_check.description or "property check returned False",
                help_text="strengthen the candidate or correct the property check",
            )

    return PatchSynthesisResult(
        candidate=candidate,
        checks=tuple(results),
        validated=True,
    )


# ---------------------------------------------------------------------------
# 94. Byzantine quorum signing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedEnvelope:
    signer: str
    payload_digest: str
    signature: str
    algorithm: str = "hmac-sha256/v1"

    def verify(self, secret: bytes) -> bool:
        """Re-derive the HMAC; this is a digest envelope, not a threshold sig."""
        expected = _signature_for(self.payload_digest, self.signer, secret)
        return hmac.compare_digest(self.signature, expected)


def _signer_key(secret: bytes, signer: str) -> bytes:
    context = b"polyglot-lang/swarm/v1/signer/" + signer.encode("utf-8")
    return hmac.new(secret, context, hashlib.sha256).digest()


def _signature_for(payload_digest: str, signer: str, secret: bytes) -> str:
    return hmac.new(
        _signer_key(secret, signer),
        payload_digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _make_envelope(payload_digest: str, signer: str, secret: bytes) -> SignedEnvelope:
    return SignedEnvelope(
        signer=signer,
        payload_digest=payload_digest,
        signature=_signature_for(payload_digest, signer, secret),
    )


@dataclass(frozen=True)
class QuorumCertificate:
    payload_digest: str
    transition: dict[str, Any]
    envelopes: tuple[SignedEnvelope, ...]
    f: int
    threshold: int
    algorithm: str
    pbft_safe: bool
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    def signers(self) -> tuple[str, ...]:
        return tuple(sorted({envelope.signer for envelope in self.envelopes}))

    def verify(self, secret: bytes) -> bool:
        """Independent verification: digest, every signature, and quorum size."""
        if len(self.signers()) < self.threshold:
            return False
        if any(not envelope.verify(secret) for envelope in self.envelopes):
            return False
        try:
            expected = hashlib.sha256(
                json.dumps(
                    self.transition,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        except TypeError, ValueError:
            return False
        return hmac.compare_digest(expected, self.payload_digest)


def swarm_byzantine_quorum_sign(
    transition: Mapping[str, Any] | str,
    signers: Sequence[str],
    *,
    f: int = 1,
    secret: bytes = REFERENCE_SECRET,
    scheme: str = SCHEME_HMAC_DIGEST,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> QuorumCertificate:
    """Sign a transition with per-signer HMAC envelopes and 2f+1 quorum math.

    The standard library has no Ed25519 threshold-signature primitive.  Pass
    ``scheme="ed25519-threshold"`` to get an explicit E007 instead of a fake
    threshold signature.  The HMAC path is real cryptography but is a simple
    envelope aggregation, not a distributed key-generation protocol.
    """
    loc = _make_location(span, line, col)
    if scheme == SCHEME_ED25519_THRESHOLD:
        raise loc.fail(
            "E007",
            "real Ed25519 threshold signatures are not available",
            reason="the Python standard library provides HMAC but no threshold signing",
            help_text="use scheme='hmac-digest' for the reference envelope model",
        )
    if scheme != SCHEME_HMAC_DIGEST:
        raise loc.fail("E003", f"unknown signing scheme {scheme!r}")
    if isinstance(secret, (bytes, bytearray)):
        key = bytes(secret)
    else:
        raise loc.fail("E001", "secret must be bytes")
    if not key:
        raise loc.fail("E003", "secret must not be empty")

    faults = _integer(f, label="f", loc=loc)
    raw_signers = _require_sequence(signers, label="signers", loc=loc)
    unique_signers = sorted(
        {
            _string(signer, label="signer id", loc=loc, code="E008")
            for signer in raw_signers
        }
    )
    threshold = 2 * faults + 1
    if len(unique_signers) < threshold:
        raise loc.fail(
            "E003",
            f"need {threshold} distinct signers for f={faults}, got {len(unique_signers)}",
            reason="quorum size is 2f+1",
        )
    if len(raw_signers) != len(unique_signers):
        raise loc.fail("E003", "duplicate signer ids are not allowed")

    if isinstance(transition, str):
        body: dict[str, Any] = {"payload": transition}
    else:
        body = dict(_require_mapping(transition, label="transition", loc=loc))
    payload_digest = _digest(body, label="transition", loc=loc)
    envelopes = tuple(
        _make_envelope(payload_digest, signer, key) for signer in unique_signers
    )
    return QuorumCertificate(
        payload_digest=payload_digest,
        transition=body,
        envelopes=envelopes,
        f=faults,
        threshold=threshold,
        algorithm=SCHEME_HMAC_DIGEST,
        pbft_safe=len(unique_signers) >= 3 * faults + 1,
        span=loc.span,
        line=loc.line,
        col=loc.col,
    )


# ---------------------------------------------------------------------------
# 95. Sybil-resistant reputation
# ---------------------------------------------------------------------------


COOPERATIVE: Final[str] = "cooperative"
DEFECTIVE: Final[str] = "defective"
_OUTCOMES: Final[frozenset[str]] = frozenset({COOPERATIVE, DEFECTIVE})


@dataclass(frozen=True)
class ReputationEvent:
    node: str
    outcome: str
    weight: float = 1.0
    reporter: str | None = None
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ReputationUpdate:
    node: str
    outcome: str
    old: float
    new: float
    delta: float
    effective_weight: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ReputationReport:
    scores: dict[str, float]
    quarantined: tuple[str, ...]
    updates: tuple[ReputationUpdate, ...]
    ignored: tuple[str, ...]
    quarantine_threshold: float


def swarm_sybil_reputation_score(
    reputation: Mapping[str, float] | None,
    events: Sequence[ReputationEvent],
    *,
    quarantine_threshold: float = 0.25,
    initial: float = 0.5,
    reward: float = 0.05,
    penalty: float = 0.20,
    vote_cap: int = 1,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> ReputationReport:
    """Update reputations with identity-capped, reporter-weighted feedback.

    A quarantined reporter's votes are dropped and each reporter has at most
    ``vote_cap`` accepted votes.  This limits Sybil amplification, but it is a
    reputation heuristic: it does not prove node identities.
    """
    loc = _make_location(span, line, col)
    threshold = _number(
        quarantine_threshold,
        label="quarantine_threshold",
        loc=loc,
        minimum=0.0,
        maximum=1.0,
        code="E003",
    )
    start = _number(
        initial, label="initial", loc=loc, minimum=0.0, maximum=1.0, code="E003"
    )
    gain = _number(
        reward, label="reward", loc=loc, minimum=0.0, maximum=1.0, code="E003"
    )
    loss = _number(
        penalty, label="penalty", loc=loc, minimum=0.0, maximum=1.0, code="E003"
    )
    cap = _integer(vote_cap, label="vote_cap", loc=loc, minimum=1)

    source: Mapping[Any, Any] = (
        {}
        if reputation is None
        else _require_mapping(reputation, label="reputation", loc=loc)
    )
    scores: dict[str, float] = {}
    for node, score in source.items():
        _string(node, label="reputation node id", loc=loc, code="E008")
        scores[node] = _number(
            score,
            label=f"reputation of {node!r}",
            loc=loc,
            minimum=0.0,
            maximum=1.0,
        )
    raw_events = _require_sequence(events, label="events", loc=loc)
    if not scores and not raw_events:
        raise loc.fail("E003", "no nodes or events supplied")

    updates: list[ReputationUpdate] = []
    ignored: list[str] = []
    votes: dict[str, int] = {}
    for index, event in enumerate(raw_events):
        if not isinstance(event, ReputationEvent):
            raise loc.fail("E001", f"event {index} must be a ReputationEvent")
        event_loc = _make_location(event.span, event.line, event.col)
        event_loc = loc.child(event_loc.span, event_loc.line, event_loc.col)
        node = _string(
            event.node, label=f"event {index} node", loc=event_loc, code="E008"
        )
        outcome = _string(
            event.outcome,
            label=f"event {index} outcome",
            loc=event_loc,
            code="E003",
        )
        if outcome not in _OUTCOMES:
            raise event_loc.fail("E003", f"unknown reputation outcome {outcome!r}")
        weight = _number(
            event.weight,
            label=f"event {index} weight",
            loc=event_loc,
            minimum=0.0,
        )
        scores.setdefault(node, start)
        if event.reporter is not None:
            scores.setdefault(event.reporter, start)

        if event.reporter is None:
            factor = 0.5
            reason = "untrusted self-attestation"
        elif scores[event.reporter] < threshold:
            factor = 0.0
            reason = f"reporter {event.reporter!r} is quarantined"
        elif votes.get(event.reporter, 0) >= cap:
            factor = 0.0
            reason = f"reporter {event.reporter!r} exhausted vote cap"
        else:
            factor = 1.0
            reason = f"endorsed by {event.reporter!r}"
            votes[event.reporter] = votes.get(event.reporter, 0) + 1

        delta = _round(factor * weight * (gain if outcome == COOPERATIVE else -loss))
        old = scores[node]
        new = _round(min(1.0, max(0.0, old + delta)))
        scores[node] = new
        accepted = factor > 0.0
        if not accepted:
            ignored.append(reason)
        updates.append(
            ReputationUpdate(
                node=node,
                outcome=outcome,
                old=old,
                new=new,
                delta=_round(new - old),
                effective_weight=_round(factor),
                accepted=accepted,
                reason=reason,
            )
        )

    return ReputationReport(
        scores=scores,
        quarantined=tuple(
            sorted(node for node, score in scores.items() if score < threshold)
        ),
        updates=tuple(updates),
        ignored=tuple(ignored),
        quarantine_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 96. Stigmergy task routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    identifier: str
    kind: str
    demand: float = 1.0
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class Route:
    task: str
    agent: str
    score: float
    pheromone_before: float
    load_before: float
    load_after: float
    exploration: bool


@dataclass(frozen=True)
class StigmergyResult:
    routes: tuple[Route, ...]
    pheromones: dict[str, dict[str, float]]
    loads: dict[str, float]
    evaporations: int


def swarm_stigmergy_task_routing(
    tasks: Sequence[Task],
    agents: Mapping[str, Mapping[str, float]],
    *,
    loads: Mapping[str, float] | None = None,
    capacities: Mapping[str, float] | None = None,
    evaporation: float = 0.1,
    pheromone_weight: float = 1.0,
    load_weight: float = 1.0,
    deposit: float = 0.2,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> StigmergyResult:
    """Route tasks by a pheromone-minus-load gradient and reinforce the choice.

    Tasks are processed in a stable order (largest demand first).  The chosen
    agent receives a pheromone deposit for the task kind, and all pheromones
    evaporate by a fixed factor.  This is an in-memory simulation of trail
    dynamics, not a distributed scheduler.
    """
    loc = _make_location(span, line, col)
    raw_tasks = _require_sequence(tasks, label="tasks", loc=loc)
    raw_agents = _require_mapping(agents, label="agents", loc=loc)
    if not raw_tasks:
        raise loc.fail("E003", "no tasks to route")
    if not raw_agents:
        raise loc.fail("E003", "no agents available")
    decay = _number(
        evaporation, label="evaporation", loc=loc, minimum=0.0, maximum=1.0, code="E003"
    )
    if decay >= 1.0:
        raise loc.fail("E003", "evaporation must be less than 1.0")
    pheromone_factor = _number(
        pheromone_weight, label="pheromone_weight", loc=loc, minimum=0.0, code="E003"
    )
    load_factor = _number(
        load_weight, label="load_weight", loc=loc, minimum=0.0, code="E003"
    )
    trail_deposit = _number(deposit, label="deposit", loc=loc, minimum=0.0, code="E003")

    agent_ids = list(raw_agents)
    for agent in agent_ids:
        _string(agent, label="agent id", loc=loc, code="E008")
    agent_ids = sorted(agent_ids)
    load_source = (
        {} if loads is None else _require_mapping(loads, label="loads", loc=loc)
    )
    capacity_source = (
        {}
        if capacities is None
        else _require_mapping(capacities, label="capacities", loc=loc)
    )
    current_loads: dict[str, float] = {}
    capacities_by_agent: dict[str, float] = {}
    trails: dict[str, dict[str, float]] = {}
    for agent in agent_ids:
        raw_trails = _require_mapping(
            raw_agents[agent], label=f"pheromones of {agent!r}", loc=loc
        )
        trails[agent] = {}
        for kind, level in raw_trails.items():
            _string(kind, label="task kind", loc=loc, code="E008")
            trails[agent][kind] = _number(
                level,
                label=f"pheromone for {kind!r} on {agent!r}",
                loc=loc,
                minimum=0.0,
            )
        current_loads[agent] = _number(
            load_source.get(agent, 0.0),
            label=f"load of {agent!r}",
            loc=loc,
            minimum=0.0,
        )
        capacities_by_agent[agent] = _number(
            capacity_source.get(agent, 1.0),
            label=f"capacity of {agent!r}",
            loc=loc,
            minimum=0.000001,
            code="E003",
        )

    prepared: list[tuple[str, str, float, _Location]] = []
    seen_tasks: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Task):
            raise loc.fail("E001", f"task {index} must be a Task")
        task_loc = _make_location(raw.span, raw.line, raw.col)
        task_loc = loc.child(task_loc.span, task_loc.line, task_loc.col)
        identifier = _string(
            raw.identifier, label=f"task {index} id", loc=task_loc, code="E008"
        )
        if identifier in seen_tasks:
            raise task_loc.fail("E003", f"duplicate task id {identifier!r}")
        seen_tasks.add(identifier)
        kind = _string(
            raw.kind, label=f"kind of task {identifier!r}", loc=task_loc, code="E008"
        )
        demand = _number(
            raw.demand,
            label=f"demand of task {identifier!r}",
            loc=task_loc,
            minimum=0.0,
        )
        prepared.append((identifier, kind, demand, task_loc))
    prepared.sort(key=lambda item: (-item[2], item[0]))

    routes: list[Route] = []
    evaporations = 0
    for identifier, kind, demand, task_loc in prepared:
        eligible = [agent for agent in agent_ids if kind in trails[agent]]
        if not eligible:
            raise task_loc.fail(
                "E006",
                f"no agent advertises pheromone for task kind {kind!r}",
                reason=f"task {identifier!r} cannot be routed",
            )
        best_agent = ""
        best_score = float("-inf")
        for agent in eligible:
            before = trails[agent][kind]
            score = pheromone_factor * before - load_factor * (
                current_loads[agent] / capacities_by_agent[agent]
            )
            if score > best_score or (score == best_score and agent < best_agent):
                best_agent = agent
                best_score = score
        chosen_level = trails[best_agent][kind]
        load_before = current_loads[best_agent]
        current_loads[best_agent] = _round(load_before + demand)
        routes.append(
            Route(
                task=identifier,
                agent=best_agent,
                score=_round(best_score),
                pheromone_before=_round(chosen_level),
                load_before=_round(load_before),
                load_after=current_loads[best_agent],
                exploration=chosen_level == 0.0,
            )
        )
        trails[best_agent][kind] = _round(
            (chosen_level + trail_deposit) * (1.0 - decay)
        )
        for agent in agent_ids:
            for trail_kind, level in trails[agent].items():
                if agent == best_agent and trail_kind == kind:
                    continue
                trails[agent][trail_kind] = _round(level * (1.0 - decay))
                evaporations += 1

    return StigmergyResult(
        routes=tuple(routes),
        pheromones={agent: dict(trails[agent]) for agent in agent_ids},
        loads=dict(current_loads),
        evaporations=evaporations,
    )


# ---------------------------------------------------------------------------
# 97. Canary rollout state machine
# ---------------------------------------------------------------------------

_CANARY_PHASES: Final[tuple[str, ...]] = ("idle", "canary", "promoted", "aborted")
_CANARY_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    "idle": frozenset({"canary"}),
    "canary": frozenset({"promote", "abort"}),
    "promoted": frozenset(),
    "aborted": frozenset(),
}
_CANARY_RESULTS: Final[dict[tuple[str, str], str]] = {
    ("idle", "canary"): "canary",
    ("canary", "promote"): "promoted",
    ("canary", "abort"): "aborted",
}


@dataclass(frozen=True)
class RolloutProbe:
    name: str
    evaluate: Callable[[Mapping[str, Any]], bool]
    required: bool = True
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ProbeResult:
    name: str
    passed: bool
    required: bool
    detail: str
    span: tuple[int, int] | None
    line: int
    col: int


@dataclass(frozen=True)
class RolloutState:
    phase: str
    sample_percent: int
    buckets: int
    canary_buckets: tuple[int, ...]
    log: tuple[str, ...]


@dataclass(frozen=True)
class CanaryRolloutResult:
    state: RolloutState
    promoted: bool
    aborted: bool
    probes: tuple[ProbeResult, ...]
    digest: str
    merged_state: dict[str, Any]


class CanaryStateMachine:
    """Explicit four-phase state machine; illegal transitions raise E003."""

    def __init__(self, sample_percent: int) -> None:
        self.phase = "idle"
        self.sample_percent = sample_percent
        self.canary_buckets = tuple(range(sample_percent))
        self.log: list[str] = []

    def transition(self, event: str, *, loc: _Location) -> None:
        allowed = _CANARY_TRANSITIONS.get(self.phase, frozenset())
        if event not in allowed:
            raise loc.fail(
                "E003",
                f"invalid canary transition {self.phase!r} -> {event!r}",
                reason=f"allowed events from {self.phase!r}: {sorted(allowed)!r}",
            )
        before = self.phase
        self.phase = _CANARY_RESULTS[(before, event)]
        self.log.append(f"{before}->{event}")


def swarm_canary_rollout_verify(
    current: Mapping[str, Any],
    canary: Mapping[str, Any],
    probes: Sequence[RolloutProbe],
    *,
    sample_percent: int = 1,
    expected_digest: str | None = None,
    initial_phase: str = "canary",
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> CanaryRolloutResult:
    """Run a 1% canary through probe verification, then promote or abort.

    The default ``sample_percent=1`` maps to bucket 0 of 100.  Probes are pure
    callables over the merged state; a required probe failure aborts.  An
    expected-digest mismatch is reported as malformed rollout state (E008).
    """
    loc = _make_location(span, line, col)
    base = dict(_require_mapping(current, label="current state", loc=loc))
    override = dict(_require_mapping(canary, label="canary state", loc=loc))
    raw_probes = _require_sequence(probes, label="probes", loc=loc)
    if not raw_probes:
        raise loc.fail("E003", "at least one rollout probe is required")
    percentage = _integer(
        sample_percent,
        label="sample_percent",
        loc=loc,
        minimum=1,
    )
    if percentage > 100:
        raise loc.fail("E003", "sample_percent must be <= 100")
    if initial_phase not in _CANARY_PHASES:
        raise loc.fail("E003", f"unknown initial phase {initial_phase!r}")
    if initial_phase in {"promoted", "aborted"}:
        raise loc.fail("E003", f"cannot start a rollout in phase {initial_phase!r}")
    for key in list(base) + list(override):
        _string(key, label="rollout state key", loc=loc, code="E008")
    _canonical_bytes(base, label="current state", loc=loc)
    _canonical_bytes(override, label="canary state", loc=loc)
    for key in sorted(override):
        if key not in base:
            raise loc.fail(
                "E003",
                f"canary key {key!r} is absent from the current state",
            )

    merged = dict(base)
    merged.update(override)
    digest = _digest(
        {"current": base, "canary": override, "sample_percent": percentage},
        label="rollout plan",
        loc=loc,
    )
    if expected_digest is not None and not hmac.compare_digest(
        str(expected_digest), digest
    ):
        raise loc.fail(
            "E008",
            "rollout plan failed its digest check",
            reason=f"expected {expected_digest!r}, computed {digest!r}",
        )

    machine = CanaryStateMachine(percentage)
    machine.transition("canary", loc=loc)
    results: list[ProbeResult] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_probes):
        if not isinstance(raw, RolloutProbe):
            raise loc.fail("E001", f"probe {index} must be a RolloutProbe")
        probe_loc = _make_location(raw.span, raw.line, raw.col)
        probe_loc = loc.child(probe_loc.span, probe_loc.line, probe_loc.col)
        probe_name = _string(
            raw.name, label=f"probe {index} name", loc=probe_loc, code="E008"
        )
        if probe_name in seen_names:
            raise probe_loc.fail("E003", f"duplicate probe name {probe_name!r}")
        seen_names.add(probe_name)
        if not callable(raw.evaluate):
            raise probe_loc.fail("E001", "probe evaluate must be callable")
        try:
            outcome = raw.evaluate(dict(merged))
        except Exception as error:
            outcome = False
            detail = f"probe raised {type(error).__name__}: {error}"
        else:
            detail = "ok" if outcome else "probe returned False"
        if not isinstance(outcome, bool):
            raise probe_loc.fail(
                "E002",
                f"probe {probe_name!r} did not return a bool",
                reason=f"got {type(outcome).__name__}",
            )
        results.append(
            ProbeResult(
                name=probe_name,
                passed=outcome,
                required=raw.required,
                detail=detail,
                span=probe_loc.span,
                line=probe_loc.line,
                col=probe_loc.col,
            )
        )
    failed = [result for result in results if not result.passed and result.required]
    if failed:
        machine.transition("abort", loc=loc)
    else:
        machine.transition("promote", loc=loc)

    return CanaryRolloutResult(
        state=RolloutState(
            phase=machine.phase,
            sample_percent=percentage,
            buckets=100,
            canary_buckets=machine.canary_buckets,
            log=tuple(machine.log),
        ),
        promoted=machine.phase == "promoted",
        aborted=machine.phase == "aborted",
        probes=tuple(results),
        digest=digest,
        merged_state=merged,
    )


# ---------------------------------------------------------------------------
# 98. Epidemic rollback kill switch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KillSwitch:
    identifier: str
    epoch: int
    reason: str
    issuer: str
    nonce: str
    target_digest: str
    signature: SignedEnvelope | None = None
    digest: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "epoch": self.epoch,
            "reason": self.reason,
            "issuer": self.issuer,
            "nonce": self.nonce,
            "target_digest": self.target_digest,
        }


@dataclass(frozen=True)
class KillBroadcastResult:
    kill_switch: KillSwitch
    deliveries: dict[str, int]
    verifications: dict[str, bool]
    verified: tuple[str, ...]
    rounds: int
    quorum: int
    quorum_met: bool


def _coerce_envelope(value: Any, *, loc: _Location) -> SignedEnvelope:
    if isinstance(value, SignedEnvelope):
        return value
    data = _require_mapping(value, label="signature envelope", loc=loc)
    return SignedEnvelope(
        signer=_string(data.get("signer"), label="signer", loc=loc, code="E008"),
        payload_digest=_string(
            data.get("payload_digest"), label="payload_digest", loc=loc, code="E008"
        ),
        signature=_string(
            data.get("signature"), label="signature", loc=loc, code="E008"
        ),
        algorithm=str(data.get("algorithm", "hmac-sha256/v1")),
    )


def _coerce_kill_switch(value: Any, *, loc: _Location) -> KillSwitch:
    if isinstance(value, KillSwitch):
        return value
    data = _require_mapping(value, label="kill switch", loc=loc)
    signature = data.get("signature")
    return KillSwitch(
        identifier=_string(
            data.get("identifier"), label="kill switch identifier", loc=loc, code="E008"
        ),
        epoch=_integer(
            data.get("epoch"), label="kill switch epoch", loc=loc, code="E008"
        ),
        reason=_string(data.get("reason"), label="kill reason", loc=loc, code="E008"),
        issuer=_string(
            data.get("issuer"), label="kill switch issuer", loc=loc, code="E008"
        ),
        nonce=_string(
            data.get("nonce"), label="kill switch nonce", loc=loc, code="E008"
        ),
        target_digest=_string(
            data.get("target_digest"), label="target_digest", loc=loc, code="E008"
        ),
        signature=None if signature is None else _coerce_envelope(signature, loc=loc),
        digest=str(data.get("digest", "")),
    )


def _epidemic_broadcast(
    issuer: str,
    recipients: Sequence[str],
    rounds: int,
    fanout: int,
) -> dict[str, int]:
    infected = {issuer}
    deliveries: dict[str, int] = {}
    frontier = [issuer]
    for round_index in range(1, rounds + 1):
        next_frontier: list[str] = []
        for node in frontier:
            pending = [target for target in recipients if target not in infected]
            for target in pending[:fanout]:
                infected.add(target)
                deliveries[target] = round_index
                next_frontier.append(target)
        frontier = next_frontier
        if all(target in deliveries for target in recipients):
            break
    return deliveries


def swarm_epidemic_rollback_kill(
    kill_switch: KillSwitch | Mapping[str, Any],
    recipients: Sequence[str],
    *,
    secret: bytes = REFERENCE_SECRET,
    quorum: int | None = None,
    rounds: int = 8,
    fanout: int = 2,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> KillBroadcastResult:
    """Broadcast and verify a signed kill-switch object by epidemic diffusion.

    Signatures are HMAC envelopes over a canonical digest of the switch body.
    A quorum of verified acknowledgements is required; the model uses no real
    network and makes no threshold-signature claim.
    """
    loc = _make_location(span, line, col)
    if isinstance(secret, (bytes, bytearray)):
        key = bytes(secret)
    else:
        raise loc.fail("E001", "secret must be bytes")
    if not key:
        raise loc.fail("E003", "secret must not be empty")
    max_rounds = _integer(rounds, label="rounds", loc=loc, minimum=1)
    spread = _integer(fanout, label="fanout", loc=loc, minimum=1)
    raw_recipients = _require_sequence(recipients, label="recipients", loc=loc)
    targets = sorted(
        {
            _string(recipient, label="recipient", loc=loc, code="E008")
            for recipient in raw_recipients
        }
    )
    if not targets:
        raise loc.fail("E003", "no kill-switch recipients")
    if len(raw_recipients) != len(targets):
        raise loc.fail("E003", "duplicate recipients are not allowed")

    switch = _coerce_kill_switch(kill_switch, loc=loc)
    body_digest = _digest(switch.body(), label="kill switch body", loc=loc)
    if switch.digest and not hmac.compare_digest(switch.digest, body_digest):
        raise loc.fail(
            "E008",
            "kill switch digest does not match its body",
            reason=f"expected {body_digest!r}, got {switch.digest!r}",
        )
    if switch.signature is None:
        signature = _make_envelope(body_digest, switch.issuer, key)
    else:
        signature = switch.signature
        if signature.signer != switch.issuer:
            raise loc.fail(
                "E008",
                "kill switch signature has a different issuer",
                reason=f"{signature.signer!r} != {switch.issuer!r}",
            )
        if not hmac.compare_digest(signature.payload_digest, body_digest):
            raise loc.fail("E008", "kill switch signature covers a different body")
        if not signature.verify(key):
            raise loc.fail("E008", "kill switch signature failed verification")
    signed = replace(switch, digest=body_digest, signature=signature)

    needed = (
        len(targets)
        if quorum is None
        else _integer(quorum, label="quorum", loc=loc, minimum=1)
    )
    if needed > len(targets) + 1:
        raise loc.fail("E003", "quorum exceeds the number of kill-switch participants")
    deliveries = _epidemic_broadcast(signed.issuer, targets, max_rounds, spread)
    verifications = {
        target: (
            target in deliveries
            and signed.signature is not None
            and signed.signature.verify(key)
        )
        for target in targets
    }
    verified = tuple(sorted(target for target, ok in verifications.items() if ok))
    quorum_met = len(verified) >= needed
    if not quorum_met:
        raise loc.fail(
            "E008",
            "kill switch did not reach its verification quorum",
            reason=f"{len(verified)} verified of {needed} required",
            help_text="increase the number of gossip rounds or fanout",
        )
    return KillBroadcastResult(
        kill_switch=signed,
        deliveries=deliveries,
        verifications=verifications,
        verified=verified,
        rounds=max(deliveries.values(), default=0),
        quorum=needed,
        quorum_met=True,
    )


# ---------------------------------------------------------------------------
# 99. Emergent topology healing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class HealedTopology:
    survivors: tuple[str, ...]
    removed: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]
    total_weight: float
    loss_ratio: float
    connected: bool


def _coerce_topology(value: Any, *, loc: _Location) -> Topology:
    if isinstance(value, Topology):
        topology = value
    else:
        data = _require_mapping(value, label="topology", loc=loc)
        node_set: set[str] = set()
        for node in data:
            node_set.add(_string(node, label="topology node id", loc=loc, code="E008"))
        raw_edges: list[tuple[str, str, float]] = []
        for node, neighbours in data.items():
            adjacency = _require_mapping(
                neighbours, label=f"neighbours of {node!r}", loc=loc
            )
            for neighbour, weight in adjacency.items():
                peer = _string(neighbour, label="neighbour id", loc=loc, code="E008")
                node_set.add(peer)
                raw_edges.append(
                    (
                        node,
                        peer,
                        _number(
                            weight,
                            label=f"edge weight {node!r}-{peer!r}",
                            loc=loc,
                            minimum=0.0,
                        ),
                    )
                )
        topology = Topology(
            nodes=tuple(sorted(node_set)),
            edges=tuple(raw_edges),
        )

    for node in topology.nodes:
        _string(node, label="topology node id", loc=loc, code="E008")
    nodes = sorted(topology.nodes)
    if len(nodes) != len(set(nodes)):
        raise loc.fail("E008", "topology contains duplicate node ids")
    if not nodes:
        raise loc.fail("E003", "topology has no nodes")
    node_set = set(nodes)
    merged: dict[tuple[str, str], float] = {}
    for edge in topology.edges:
        if not isinstance(edge, tuple) or len(edge) != 3:
            raise loc.fail("E008", "each topology edge must be a (a, b, weight) triple")
        a = _string(edge[0], label="edge endpoint", loc=loc, code="E008")
        b = _string(edge[1], label="edge endpoint", loc=loc, code="E008")
        if a not in node_set or b not in node_set:
            raise loc.fail("E008", f"edge {a!r}-{b!r} has an unknown endpoint")
        if a == b:
            continue
        weight = _number(
            edge[2],
            label=f"edge weight {a!r}-{b!r}",
            loc=loc,
            minimum=0.0,
        )
        key = tuple(sorted((a, b)))
        merged[key] = min(merged.get(key, weight), weight)
    return Topology(
        nodes=tuple(nodes),
        edges=tuple((a, b, weight) for (a, b), weight in sorted(merged.items())),
    )


def swarm_emergent_topology_heal(
    topology: Topology | Mapping[str, Mapping[str, float]],
    lost: Iterable[str],
    *,
    max_loss_ratio: float = 0.49,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> HealedTopology:
    """Reconstruct an MST after up to 49% node loss using Kruskal's algorithm.

    A larger loss ratio is E008 rather than a silent best-effort tree; so is a
    survivor graph that is disconnected after the loss.  Weights are caller
    supplied and are not a measurement of real network latency.
    """
    loc = _make_location(span, line, col)
    limit = _number(
        max_loss_ratio,
        label="max_loss_ratio",
        loc=loc,
        minimum=0.0,
        maximum=0.49,
        code="E003",
    )
    graph = _coerce_topology(topology, loc=loc)
    if isinstance(lost, (str, bytes)):
        raise loc.fail("E001", "lost must be an iterable of node ids")
    missing = tuple(lost)
    removed = sorted(
        _string(node, label="lost node id", loc=loc, code="E008") for node in missing
    )
    if len(removed) != len(set(removed)):
        raise loc.fail("E003", "duplicate node ids in lost set")
    unknown = sorted(set(removed) - set(graph.nodes))
    if unknown:
        raise loc.fail("E008", f"lost set contains unknown nodes: {unknown!r}")
    loss_ratio = len(removed) / len(graph.nodes)
    if loss_ratio > limit:
        raise loc.fail(
            "E008",
            f"node loss {loss_ratio:.3f} exceeds the {limit:.3f} recovery limit",
            reason=f"{len(removed)} of {len(graph.nodes)} nodes were lost",
            help_text="restore nodes or raise the documented 49% bound explicitly",
        )
    survivors = tuple(sorted(set(graph.nodes) - set(removed)))
    if not survivors:
        raise loc.fail("E008", "no nodes survived the topology loss")

    parent = {node: node for node in survivors}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        parent[max(root_a, root_b)] = min(root_a, root_b)
        return True

    available = [
        edge for edge in graph.edges if edge[0] in parent and edge[1] in parent
    ]
    available.sort(key=lambda edge: (edge[2], edge[0], edge[1]))
    mst: list[tuple[str, str, float]] = []
    for edge in available:
        if union(edge[0], edge[1]):
            mst.append(edge)
            if len(mst) == len(survivors) - 1:
                break
    connected = len(mst) == len(survivors) - 1
    if not connected:
        raise loc.fail(
            "E008",
            "survivor topology is partitioned; MST reconstruction failed",
            reason=f"only {len(mst)} of {len(survivors) - 1} MST edges could be built",
        )
    return HealedTopology(
        survivors=survivors,
        removed=tuple(removed),
        edges=tuple(mst),
        total_weight=_round(sum(edge[2] for edge in mst)),
        loss_ratio=_round(loss_ratio),
        connected=True,
    )


# ---------------------------------------------------------------------------
# 100. Cluster-wide invariant audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeState:
    values: dict[str, Any]
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class Invariant:
    identifier: str
    description: str
    check: Callable[[Mapping[str, Any]], bool]
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class AuditRecord:
    node: str
    identifier: str
    code: str
    status: str
    detail: str
    span: tuple[int, int] | None
    line: int
    col: int


@dataclass(frozen=True)
class AuditReport:
    records: tuple[AuditRecord, ...]
    ok: bool
    passed: int
    failed: int
    errors: int
    nodes: tuple[str, ...]
    invariants: tuple[str, ...]


def swarm_global_invariant_audit(
    cluster_state: Mapping[str, NodeState | Mapping[str, Any]],
    invariants: Sequence[Invariant],
    *,
    strict: bool = True,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> AuditReport:
    """Check every invariant on every node and emit E008 audit records.

    Records carry the most precise available source span (invariant, then
    node, then call site).  With ``strict=True`` the first non-passing record
    also raises PillarError E008, with the full report attached as
    ``audit_report``; ``strict=False`` only returns the report.
    """
    loc = _make_location(span, line, col)
    data = _require_mapping(cluster_state, label="cluster state", loc=loc)
    raw_invariants = _require_sequence(invariants, label="invariants", loc=loc)
    if not data:
        raise loc.fail("E003", "cluster state has no nodes")
    if not raw_invariants:
        raise loc.fail("E003", "at least one invariant is required")

    checks: list[tuple[Invariant, _Location]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_invariants):
        if not isinstance(raw, Invariant):
            raise loc.fail("E001", f"invariant {index} must be an Invariant")
        inv_loc = _make_location(raw.span, raw.line, raw.col)
        inv_loc = loc.child(inv_loc.span, inv_loc.line, inv_loc.col)
        identifier = _string(
            raw.identifier, label=f"invariant {index} id", loc=inv_loc, code="E008"
        )
        if identifier in identifiers:
            raise inv_loc.fail("E003", f"duplicate invariant {identifier!r}")
        identifiers.add(identifier)
        _string(
            raw.description,
            label=f"description of invariant {identifier!r}",
            loc=inv_loc,
            code="E008",
        )
        if not callable(raw.check):
            raise inv_loc.fail("E001", "invariant check must be callable")
        checks.append((raw, inv_loc))

    records: list[AuditRecord] = []
    for node in list(data):
        _string(node, label="cluster node id", loc=loc, code="E008")
    for node in sorted(data):
        raw_state = data[node]
        if isinstance(raw_state, NodeState):
            values = raw_state.values
            node_loc = _make_location(raw_state.span, raw_state.line, raw_state.col)
            node_loc = loc.child(node_loc.span, node_loc.line, node_loc.col)
        elif isinstance(raw_state, Mapping):
            values = raw_state
            node_loc = loc
        else:
            raise loc.fail(
                "E008",
                f"malformed state for cluster node {node!r}",
                reason=f"got {type(raw_state).__name__}",
            )
        for invariant, inv_loc in checks:
            record_loc = _Location(
                inv_loc.span if inv_loc.span is not None else node_loc.span,
                inv_loc.line if inv_loc.line else node_loc.line,
                inv_loc.col if inv_loc.col else node_loc.col,
            )
            try:
                outcome = invariant.check(dict(values))
            except Exception as error:
                status = "error"
                detail = f"check raised {type(error).__name__}: {error}"
            else:
                if not isinstance(outcome, bool):
                    status = "error"
                    detail = f"check returned {type(outcome).__name__}, not bool"
                elif outcome:
                    status = "pass"
                    detail = invariant.description
                else:
                    status = "fail"
                    detail = invariant.description
            records.append(
                AuditRecord(
                    node=node,
                    identifier=invariant.identifier,
                    code="E008" if status != "pass" else "",
                    status=status,
                    detail=detail,
                    span=record_loc.span,
                    line=record_loc.line,
                    col=record_loc.col,
                )
            )

    passed = sum(record.status == "pass" for record in records)
    failed = sum(record.status == "fail" for record in records)
    errors = sum(record.status == "error" for record in records)
    report = AuditReport(
        records=tuple(records),
        ok=failed == 0 and errors == 0,
        passed=passed,
        failed=failed,
        errors=errors,
        nodes=tuple(sorted(data)),
        invariants=tuple(item.identifier for item, _ in checks),
    )
    if not report.ok and strict:
        first = next(record for record in records if record.status != "pass")
        error = _Location(first.span, first.line, first.col).fail(
            "E008",
            f"invariant {first.identifier!r} failed on node {first.node!r}",
            reason=first.detail,
            help_text="inspect the returned audit_report for all cluster records",
        )
        error.audit_report = report
        raise error
    return report


__all__ = [
    "COOPERATIVE",
    "DEFECTIVE",
    "REFERENCE_SECRET",
    "REMOVE_MARKER",
    "SCHEME_ED25519_THRESHOLD",
    "SCHEME_HMAC_DIGEST",
    "AuditRecord",
    "AuditReport",
    "CanaryRolloutResult",
    "CanaryStateMachine",
    "CandidatePatch",
    "GossipExchange",
    "GossipSyncResult",
    "HealedTopology",
    "Invariant",
    "KillBroadcastResult",
    "KillSwitch",
    "NodeState",
    "PanicAssessment",
    "PanicReport",
    "PatchOperation",
    "PatchSynthesisResult",
    "ProbeResult",
    "PropertyCheck",
    "PropertyResult",
    "QuorumCertificate",
    "ReputationEvent",
    "ReputationReport",
    "ReputationUpdate",
    "RolloutProbe",
    "RolloutState",
    "Route",
    "SignedEnvelope",
    "StigmergyResult",
    "Task",
    "TelemetrySample",
    "Topology",
    "swarm_autonomous_patch_synth",
    "swarm_byzantine_quorum_sign",
    "swarm_canary_rollout_verify",
    "swarm_emergent_topology_heal",
    "swarm_epidemic_rollback_kill",
    "swarm_global_invariant_audit",
    "swarm_gossip_epidemic_sync",
    "swarm_neural_panic_detector",
    "swarm_stigmergy_task_routing",
    "swarm_sybil_reputation_score",
]
