"""Deterministic reference tests for swarm functions 91-100."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import swarm
from runtime.diagnostics import PillarError
from swarm import swarm_global_invariant_audit
from swarm.agent import (
    COOPERATIVE,
    DEFECTIVE,
    REMOVE_MARKER,
    SCHEME_ED25519_THRESHOLD,
    Invariant,
    KillSwitch,
    NodeState,
    PropertyCheck,
    ReputationEvent,
    RolloutProbe,
    Task,
    TelemetrySample,
    Topology,
    swarm_autonomous_patch_synth,
    swarm_byzantine_quorum_sign,
    swarm_canary_rollout_verify,
    swarm_emergent_topology_heal,
    swarm_epidemic_rollback_kill,
    swarm_gossip_epidemic_sync,
    swarm_neural_panic_detector,
    swarm_stigmergy_task_routing,
    swarm_sybil_reputation_score,
)

# ---------------------------------------------------------------------------
# 91. Gossip epidemic sync
# ---------------------------------------------------------------------------


def test_gossip_converges_via_anti_entropy():
    result = swarm_gossip_epidemic_sync(
        {
            "a": {"x": 1},
            "b": {"y": 3},
            "c": {"x": 5, "z": 2},
            "d": {},
        },
        rounds=12,
    )
    assert result.converged
    assert result.rounds > 0
    expected = {"x": 5, "y": 3, "z": 2}
    assert all(state == expected for state in result.states.values())
    assert result.messages == 2 * len(result.exchanges)
    assert all(exchange.digest_ok for exchange in result.exchanges)
    assert len(set(result.digests.values())) == 1


def test_gossip_already_converged_performs_no_rounds():
    result = swarm_gossip_epidemic_sync(
        {"a": {"x": 1}, "b": {"x": 1}},
        rounds=10,
    )
    assert result.converged
    assert result.rounds == 0
    assert result.messages == 0
    assert result.exchanges == ()


def test_gossip_versions_never_regress():
    result = swarm_gossip_epidemic_sync(
        {"a": {"x": 9}, "b": {"x": 2}},
        rounds=3,
    )
    assert result.states == {"a": {"x": 9}, "b": {"x": 9}}


def test_gossip_rejects_empty_registry_and_bad_versions():
    with pytest.raises(PillarError) as empty:
        swarm_gossip_epidemic_sync({})
    assert empty.value.code == "E003"
    with pytest.raises(PillarError) as shape:
        swarm_gossip_epidemic_sync({"a": []})
    assert shape.value.code == "E001"
    with pytest.raises(PillarError) as version:
        swarm_gossip_epidemic_sync({"a": {"x": -1}})
    assert version.value.code == "E008"
    with pytest.raises(PillarError) as rounds:
        swarm_gossip_epidemic_sync({"a": {}}, rounds=-1)
    assert rounds.value.code == "E003"


def test_gossip_reports_exact_span_and_line():
    with pytest.raises(PillarError) as error:
        swarm_gossip_epidemic_sync(
            {"a": {"x": "bad"}},
            span=(4, 9),
            line=7,
            col=2,
        )
    assert error.value.code == "E001"
    assert error.value.span == (4, 9)
    assert (error.value.line, error.value.col) == (7, 2)
    assert "span 4..9" in error.value.pretty()
    with pytest.raises(PillarError) as malformed:
        swarm_gossip_epidemic_sync({"a": {}}, span=(9, 4))
    assert malformed.value.code == "E003"


# ---------------------------------------------------------------------------
# 92. Neural panic detector
# ---------------------------------------------------------------------------


def test_panic_detector_accepts_nominal_telemetry():
    report = swarm_neural_panic_detector(
        {"n1": [(10.0, 0.01), (11.0, 0.01), (9.0, 0.02), (10.0, 0.01)]},
        baseline_window=2,
    )
    assert report.panics == ()
    assert report.warnings == ()
    assert report.nominal == ("n1",)
    assert report.assessments[0].status == "nominal"
    assert 0.0 <= report.scores["n1"] < 1.0


def test_panic_detector_flags_latency_and_error_spike():
    report = swarm_neural_panic_detector(
        {"n1": [(10.0, 0.01), (10.0, 0.01), (40.0, 0.2)]},
        baseline_window=2,
    )
    assessment = report.assessments[0]
    assert assessment.status == "panic"
    assert report.panics == ("n1",)
    assert assessment.triggered_at == 2
    assert assessment.peak_latency_ratio == 4.0
    assert "latency ratio" in assessment.reasons[0]


def test_panic_detector_flags_critical_error_rate():
    report = swarm_neural_panic_detector(
        {"n1": [(10.0, 0.01), (10.0, 0.01), (10.0, 0.6)]},
        baseline_window=2,
    )
    assert report.panics == ("n1",)
    assert "error_rate" in report.assessments[0].reasons[0]


def test_panic_detector_panics_after_warning_streak():
    report = swarm_neural_panic_detector(
        {
            "n1": [
                (10.0, 0.01),
                (10.0, 0.01),
                (18.0, 0.06),
                (18.0, 0.06),
                (18.0, 0.06),
            ]
        },
        baseline_window=2,
        panic_streak=3,
    )
    assessment = report.assessments[0]
    assert assessment.status == "panic"
    assert "consecutive warnings" in assessment.reasons[0]
    assert assessment.triggered_at == 4


def test_panic_detector_validates_samples_and_configuration():
    with pytest.raises(PillarError) as nan:
        swarm_neural_panic_detector({"n1": [(10.0, float("nan"))]})
    assert nan.value.code == "E008"
    with pytest.raises(PillarError) as shape:
        swarm_neural_panic_detector({"n1": ["bad"]})
    assert shape.value.code == "E001"
    with pytest.raises(PillarError) as config:
        swarm_neural_panic_detector(
            {"n1": [(1.0, 0.0)]},
            warning_latency_ratio=4.0,
            panic_latency_ratio=2.0,
        )
    assert config.value.code == "E003"
    with pytest.raises(PillarError) as empty:
        swarm_neural_panic_detector({})
    assert empty.value.code == "E003"


def test_panic_detector_uses_sample_span_for_bad_values():
    sample = TelemetrySample(
        latency_ms=-1.0,
        error_rate=0.0,
        span=(15, 25),
        line=11,
        col=6,
    )
    with pytest.raises(PillarError) as error:
        swarm_neural_panic_detector({"n1": [sample]})
    assert error.value.code == "E008"
    assert error.value.span == (15, 25)
    assert (error.value.line, error.value.col) == (11, 6)


# ---------------------------------------------------------------------------
# 93. Autonomous patch synthesis
# ---------------------------------------------------------------------------


def test_patch_synthesis_builds_a_validated_candidate():
    result = swarm_autonomous_patch_synth(
        {"a": 1, "b": 2, "c": 3},
        {"b": 5, "c": REMOVE_MARKER, "d": 7},
        [
            PropertyCheck("b_is_five", lambda state: state["b"] == 5),
            PropertyCheck("c_removed", lambda state: "c" not in state),
            PropertyCheck("d_added", lambda state: state["d"] == 7),
        ],
    )
    assert result.validated
    assert [op.kind for op in result.candidate.operations] == [
        "update",
        "remove",
        "add",
    ]
    assert result.candidate.post_state == {"a": 1, "b": 5, "d": 7}
    assert len(result.candidate.digest) == 64
    assert all(check.passed for check in result.checks)


def test_patch_synthesis_rejects_a_failed_property_with_its_span():
    check = PropertyCheck(
        "b_is_large",
        lambda state: state["b"] > 100,
        "b must exceed 100",
        span=(11, 19),
        line=9,
        col=4,
    )
    with pytest.raises(PillarError) as error:
        swarm_autonomous_patch_synth(
            {"b": 2},
            {"b": 5},
            [check],
            span=(0, 4),
            line=1,
            col=1,
        )
    assert error.value.code == "E002"
    assert error.value.span == (11, 19)
    assert (error.value.line, error.value.col) == (9, 4)
    assert "b_is_large" in str(error.value)


def test_patch_synthesis_wraps_predicate_exceptions_as_e002():
    def explode(state):
        raise RuntimeError("probe unavailable")

    with pytest.raises(PillarError) as error:
        swarm_autonomous_patch_synth(
            {"x": 1},
            {"x": 2},
            [PropertyCheck("boom", explode)],
        )
    assert error.value.code == "E002"
    assert "boom" in str(error.value)


def test_patch_synthesis_rejects_malformed_and_empty_inputs():
    with pytest.raises(PillarError) as unsupported:
        swarm_autonomous_patch_synth(
            {"x": 1},
            {"x": {1, 2}},
            [PropertyCheck("ok", lambda state: True)],
        )
    assert unsupported.value.code == "E001"
    with pytest.raises(PillarError) as nonfinite:
        swarm_autonomous_patch_synth(
            {"x": 1},
            {"x": float("inf")},
            [PropertyCheck("ok", lambda state: True)],
        )
    assert nonfinite.value.code == "E008"
    with pytest.raises(PillarError) as no_checks:
        swarm_autonomous_patch_synth({"x": 1}, {"x": 2}, [])
    assert no_checks.value.code == "E003"
    with pytest.raises(PillarError) as no_change:
        swarm_autonomous_patch_synth(
            {"x": 1},
            {"x": 1},
            [PropertyCheck("ok", lambda state: True)],
        )
    assert no_change.value.code == "E003"


# ---------------------------------------------------------------------------
# 94. Byzantine quorum signing
# ---------------------------------------------------------------------------


def test_quorum_sign_produces_verifiable_hmac_envelopes():
    certificate = swarm_byzantine_quorum_sign(
        {"op": "commit", "epoch": 1},
        ["n1", "n2", "n3"],
        f=1,
    )
    assert certificate.threshold == 3
    assert certificate.algorithm == "hmac-digest"
    assert certificate.signers() == ("n1", "n2", "n3")
    assert len(certificate.envelopes) == 3
    assert all(
        envelope.algorithm == "hmac-sha256/v1" for envelope in certificate.envelopes
    )
    assert certificate.verify(swarm.REFERENCE_SECRET)


def test_quorum_math_scales_with_f():
    five = swarm_byzantine_quorum_sign({"op": "commit"}, ["a", "b", "c", "d", "e"], f=2)
    seven = swarm_byzantine_quorum_sign(
        {"op": "commit"}, ["a", "b", "c", "d", "e", "f", "g"], f=2
    )
    assert five.threshold == 5
    assert five.pbft_safe is False
    assert seven.threshold == 5
    assert seven.pbft_safe is True
    assert five.verify(swarm.REFERENCE_SECRET)
    assert seven.verify(swarm.REFERENCE_SECRET)


def test_quorum_requires_2f_plus_1_distinct_signers():
    with pytest.raises(PillarError) as insufficient:
        swarm_byzantine_quorum_sign({"op": "x"}, ["n1", "n2"], f=1)
    assert insufficient.value.code == "E003"
    assert "need 3 distinct signers" in str(insufficient.value)
    with pytest.raises(PillarError) as duplicate:
        swarm_byzantine_quorum_sign({"op": "x"}, ["n1", "n1", "n2", "n3"], f=1)
    assert duplicate.value.code == "E003"


def test_threshold_signatures_are_e007_not_faked():
    with pytest.raises(PillarError) as error:
        swarm_byzantine_quorum_sign(
            {"op": "x"},
            ["n1", "n2", "n3"],
            scheme=SCHEME_ED25519_THRESHOLD,
            span=(2, 12),
            line=5,
            col=9,
        )
    assert error.value.code == "E007"
    assert error.value.span == (2, 12)
    assert (error.value.line, error.value.col) == (5, 9)
    assert "threshold signatures" in str(error.value)


def test_quorum_certificate_detects_tampering():
    certificate = swarm_byzantine_quorum_sign({"op": "commit"}, ["n1", "n2", "n3"], f=1)
    bad_signature = dataclasses.replace(certificate.envelopes[0], signature="00" * 32)
    tampered = dataclasses.replace(
        certificate,
        envelopes=(bad_signature, *certificate.envelopes[1:]),
    )
    assert not tampered.verify(swarm.REFERENCE_SECRET)
    assert not certificate.verify(b"another-secret")
    replayed = dataclasses.replace(certificate, transition={"op": "rollback"})
    assert not replayed.verify(swarm.REFERENCE_SECRET)
    with pytest.raises(PillarError) as unknown:
        swarm_byzantine_quorum_sign({"op": "x"}, ["a", "b", "c"], scheme="rsa")
    assert unknown.value.code == "E003"


# ---------------------------------------------------------------------------
# 95. Sybil reputation
# ---------------------------------------------------------------------------


def test_reputation_updates_are_deterministic_and_clamped():
    report = swarm_sybil_reputation_score(
        {"observer": 0.9},
        [
            ReputationEvent("a", COOPERATIVE, reporter="observer"),
            ReputationEvent("b", DEFECTIVE, reporter="observer"),
        ],
        vote_cap=2,
    )
    assert report.scores["a"] == 0.55
    assert report.scores["b"] == 0.3
    assert report.quarantined == ()
    assert [update.delta for update in report.updates] == [0.05, -0.2]
    high = swarm_sybil_reputation_score(
        {"observer": 0.9},
        [ReputationEvent("a", COOPERATIVE, weight=10.0, reporter="observer")],
    )
    assert high.scores["a"] == 1.0


def test_reputation_quarantines_below_threshold():
    report = swarm_sybil_reputation_score(
        {"bad": 0.5, "observer": 0.9},
        [ReputationEvent("bad", DEFECTIVE, weight=2.0, reporter="observer")],
        quarantine_threshold=0.25,
    )
    assert report.scores["bad"] == 0.1
    assert report.quarantined == ("bad",)


def test_quarantined_reporter_cannot_amplify_sybil_votes():
    report = swarm_sybil_reputation_score(
        {"sybil": 0.1},
        [ReputationEvent("victim", DEFECTIVE, weight=5.0, reporter="sybil")],
    )
    update = report.updates[0]
    assert report.scores["victim"] == 0.5
    assert update.accepted is False
    assert update.effective_weight == 0.0
    assert "quarantined" in update.reason
    assert "quarantined" in report.ignored[0]


def test_reputation_vote_cap_and_self_attestation():
    capped = swarm_sybil_reputation_score(
        {"trusted": 0.9},
        [
            ReputationEvent("a", COOPERATIVE, reporter="trusted"),
            ReputationEvent("b", COOPERATIVE, reporter="trusted"),
        ],
        vote_cap=1,
    )
    assert [update.accepted for update in capped.updates] == [True, False]
    assert "vote cap" in capped.ignored[0]
    self_report = swarm_sybil_reputation_score(
        None,
        [ReputationEvent("a", COOPERATIVE)],
    )
    assert self_report.updates[0].effective_weight == 0.5
    assert self_report.scores["a"] == 0.525


def test_reputation_validates_values_and_uses_event_spans():
    with pytest.raises(PillarError) as nonfinite:
        swarm_sybil_reputation_score({"a": float("inf")}, [])
    assert nonfinite.value.code == "E008"
    with pytest.raises(PillarError) as empty:
        swarm_sybil_reputation_score(None, [])
    assert empty.value.code == "E003"
    event = ReputationEvent(
        "a",
        "unknown",
        span=(20, 28),
        line=13,
        col=3,
    )
    with pytest.raises(PillarError) as bad_outcome:
        swarm_sybil_reputation_score(None, [event])
    assert bad_outcome.value.code == "E003"
    assert bad_outcome.value.span == (20, 28)
    assert (bad_outcome.value.line, bad_outcome.value.col) == (13, 3)


# ---------------------------------------------------------------------------
# 96. Stigmergy task routing
# ---------------------------------------------------------------------------


def test_stigmergy_follows_pheromone_minus_load_gradient():
    result = swarm_stigmergy_task_routing(
        [Task("t1", "io", 1.0), Task("t2", "io", 1.0)],
        {"fast": {"io": 0.9}, "slow": {"io": 0.2}},
        loads={"fast": 9.0, "slow": 0.0},
        capacities={"fast": 10.0, "slow": 1.0},
    )
    assert [route.agent for route in result.routes] == ["slow", "fast"]
    assert result.routes[0].score == 0.2
    assert result.loads == {"fast": 10.0, "slow": 1.0}
    assert result.pheromones["slow"]["io"] > 0.2
    assert result.pheromones["fast"]["io"] > 0.9


def test_stigmergy_deposits_and_evaporates_trails():
    result = swarm_stigmergy_task_routing(
        [Task("t", "io", 1.0)],
        {"a": {"io": 1.0}, "b": {"io": 0.0}},
        evaporation=0.5,
        deposit=0.4,
    )
    assert result.routes[0].agent == "a"
    assert result.pheromones["a"]["io"] == 0.7
    assert result.pheromones["b"]["io"] == 0.0
    assert result.evaporations == 1
    assert result.routes[0].exploration is False


def test_stigmergy_explores_when_all_trails_are_zero():
    result = swarm_stigmergy_task_routing(
        [Task("t", "cpu", 1.0)],
        {"a": {"cpu": 0.0}, "b": {"cpu": 0.0}},
        load_weight=0.0,
    )
    assert result.routes[0].agent == "a"
    assert result.routes[0].exploration is True


def test_stigmergy_reports_missing_pheromone_with_task_span():
    task = Task(
        "t",
        "cpu",
        1.0,
        span=(6, 14),
        line=17,
        col=2,
    )
    with pytest.raises(PillarError) as error:
        swarm_stigmergy_task_routing([task], {"a": {"io": 0.5}})
    assert error.value.code == "E006"
    assert error.value.span == (6, 14)
    assert (error.value.line, error.value.col) == (17, 2)


def test_stigmergy_validates_configuration_and_trail_values():
    with pytest.raises(PillarError) as decay:
        swarm_stigmergy_task_routing(
            [Task("t", "io")], {"a": {"io": 0.5}}, evaporation=1.0
        )
    assert decay.value.code == "E003"
    with pytest.raises(PillarError) as bad_trail:
        swarm_stigmergy_task_routing(
            [Task("t", "io")],
            {"a": {"io": -0.5}},
            span=(1, 5),
            line=2,
            col=3,
        )
    assert bad_trail.value.code == "E008"
    assert bad_trail.value.span == (1, 5)
    with pytest.raises(PillarError) as no_agents:
        swarm_stigmergy_task_routing([Task("t", "io")], {})
    assert no_agents.value.code == "E003"


# ---------------------------------------------------------------------------
# 97. Canary rollout state machine
# ---------------------------------------------------------------------------


def test_canary_promotes_when_required_probes_pass():
    result = swarm_canary_rollout_verify(
        {"mode": "stable", "version": 1},
        {"mode": "canary"},
        [RolloutProbe("mode", lambda state: state["mode"] == "canary")],
    )
    assert result.promoted
    assert not result.aborted
    assert result.state.phase == "promoted"
    assert result.state.sample_percent == 1
    assert result.state.buckets == 100
    assert result.state.canary_buckets == (0,)
    assert result.state.log == ("idle->canary", "canary->promote")
    assert result.merged_state["mode"] == "canary"
    assert len(result.digest) == 64


def test_canary_aborts_on_failed_required_probe():
    result = swarm_canary_rollout_verify(
        {"healthy": True},
        {"healthy": False},
        [
            RolloutProbe("health", lambda state: state["healthy"]),
            RolloutProbe(
                "optional",
                lambda state: False,
                required=False,
            ),
        ],
    )
    assert result.aborted
    assert not result.promoted
    assert result.state.phase == "aborted"
    assert result.state.log == ("idle->canary", "canary->abort")
    assert [probe.passed for probe in result.probes] == [False, False]


def test_canary_validates_sample_percent_and_digest():
    good = swarm_canary_rollout_verify(
        {"v": 1},
        {"v": 2},
        [RolloutProbe("ok", lambda state: True)],
    )
    with pytest.raises(PillarError) as mismatch:
        swarm_canary_rollout_verify(
            {"v": 1},
            {"v": 2},
            [RolloutProbe("ok", lambda state: True)],
            expected_digest="0" * 64,
            span=(3, 11),
            line=6,
            col=1,
        )
    assert mismatch.value.code == "E008"
    assert mismatch.value.span == (3, 11)
    assert (mismatch.value.line, mismatch.value.col) == (6, 1)
    accepted = swarm_canary_rollout_verify(
        {"v": 1},
        {"v": 2},
        [RolloutProbe("ok", lambda state: True)],
        expected_digest=good.digest,
    )
    assert accepted.promoted
    with pytest.raises(PillarError) as zero:
        swarm_canary_rollout_verify(
            {"v": 1},
            {"v": 2},
            [RolloutProbe("ok", lambda state: True)],
            sample_percent=0,
        )
    assert zero.value.code == "E003"
    with pytest.raises(PillarError) as too_many:
        swarm_canary_rollout_verify(
            {"v": 1},
            {"v": 2},
            [RolloutProbe("ok", lambda state: True)],
            sample_percent=101,
        )
    assert too_many.value.code == "E003"


def test_canary_validates_phases_probes_and_spans():
    with pytest.raises(PillarError) as phase:
        swarm_canary_rollout_verify(
            {"v": 1},
            {"v": 2},
            [RolloutProbe("ok", lambda state: True)],
            initial_phase="promoted",
        )
    assert phase.value.code == "E003"
    bad_probe = RolloutProbe(
        "not_bool",
        lambda state: 1,
        span=(30, 40),
        line=21,
        col=8,
    )
    with pytest.raises(PillarError) as result_type:
        swarm_canary_rollout_verify({"v": 1}, {"v": 2}, [bad_probe])
    assert result_type.value.code == "E002"
    assert result_type.value.span == (30, 40)
    assert (result_type.value.line, result_type.value.col) == (21, 8)
    with pytest.raises(PillarError) as unknown_key:
        swarm_canary_rollout_verify(
            {"v": 1},
            {"missing": 2},
            [RolloutProbe("ok", lambda state: True)],
        )
    assert unknown_key.value.code == "E003"


# ---------------------------------------------------------------------------
# 98. Epidemic rollback kill switch
# ---------------------------------------------------------------------------


def _kill_switch() -> KillSwitch:
    return KillSwitch(
        identifier="kill-1",
        epoch=3,
        reason="neural panic",
        issuer="leader",
        nonce="nonce-1",
        target_digest="a" * 64,
    )


def test_kill_switch_broadcasts_and_verifies_all_recipients():
    recipients = [f"n{index}" for index in range(7)]
    result = swarm_epidemic_rollback_kill(_kill_switch(), recipients)
    assert result.verified == tuple(sorted(recipients))
    assert all(result.verifications.values())
    assert result.quorum == 7
    assert result.quorum_met
    assert 0 < result.rounds <= 3
    assert len(result.kill_switch.digest) == 64
    assert result.kill_switch.signature is not None
    assert result.kill_switch.signature.verify(swarm.REFERENCE_SECRET)


def test_kill_switch_accepts_a_mapping_payload():
    result = swarm_epidemic_rollback_kill(
        {
            "identifier": "kill-2",
            "epoch": 1,
            "reason": "quorum lost",
            "issuer": "leader",
            "nonce": "nonce-2",
            "target_digest": "b" * 64,
        },
        ["n1"],
    )
    assert result.verified == ("n1",)
    assert result.kill_switch.identifier == "kill-2"


def test_kill_switch_rejects_tampered_digest_and_issuer():
    with pytest.raises(PillarError) as tampered:
        swarm_epidemic_rollback_kill(
            dataclasses.replace(_kill_switch(), digest="0" * 64),
            ["n1"],
            span=(8, 16),
            line=4,
            col=7,
        )
    assert tampered.value.code == "E008"
    assert tampered.value.span == (8, 16)
    assert (tampered.value.line, tampered.value.col) == (4, 7)
    delivered = swarm_epidemic_rollback_kill(_kill_switch(), ["n1", "n2"])
    forged = dataclasses.replace(
        delivered.kill_switch,
        signature=dataclasses.replace(
            delivered.kill_switch.signature,
            signer="attacker",
        ),
    )
    with pytest.raises(PillarError) as wrong_issuer:
        swarm_epidemic_rollback_kill(forged, ["n1", "n2"])
    assert wrong_issuer.value.code == "E008"
    assert "issuer" in str(wrong_issuer.value)


def test_kill_switch_requires_verification_quorum():
    with pytest.raises(PillarError) as short:
        swarm_epidemic_rollback_kill(
            _kill_switch(),
            ["n1", "n2", "n3", "n4"],
            rounds=1,
            quorum=4,
        )
    assert short.value.code == "E008"
    assert "quorum" in str(short.value)
    with pytest.raises(PillarError) as duplicate:
        swarm_epidemic_rollback_kill(_kill_switch(), ["n1", "n1"])
    assert duplicate.value.code == "E003"


# ---------------------------------------------------------------------------
# 99. Emergent topology healing
# ---------------------------------------------------------------------------


def _dense_topology(count: int = 10) -> Topology:
    nodes = tuple(f"n{index}" for index in range(count))
    edges = tuple(
        (nodes[i], nodes[j], float(abs(i - j)))
        for i in range(count)
        for j in range(i + 1, count)
    )
    return Topology(nodes=nodes, edges=edges)


def test_topology_heals_mst_after_40_percent_loss():
    topology = _dense_topology()
    healed = swarm_emergent_topology_heal(
        topology,
        ["n0", "n1", "n2", "n3"],
    )
    assert healed.loss_ratio == 0.4
    assert len(healed.survivors) == 6
    assert healed.connected
    assert len(healed.edges) == 5
    assert healed.total_weight == 5.0
    assert all(node in healed.survivors for edge in healed.edges for node in edge[:2])


def test_topology_heal_accepts_adjacency_mapping():
    adjacency = {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 2.0},
        "c": {"b": 2.0},
    }
    healed = swarm_emergent_topology_heal(adjacency, [])
    assert healed.survivors == ("a", "b", "c")
    assert healed.total_weight == 3.0


def test_topology_rejects_greater_than_49_percent_loss():
    with pytest.raises(PillarError) as error:
        swarm_emergent_topology_heal(
            _dense_topology(),
            ["n0", "n1", "n2", "n3", "n4"],
            span=(12, 24),
            line=15,
            col=5,
        )
    assert error.value.code == "E008"
    assert error.value.span == (12, 24)
    assert (error.value.line, error.value.col) == (15, 5)
    assert "exceeds" in str(error.value)


def test_topology_rejects_partitioned_survivors():
    topology = Topology(
        nodes=("a", "b", "c", "d", "e", "f"),
        edges=(
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("c", "d", 1.0),
            ("d", "e", 1.0),
            ("e", "f", 1.0),
        ),
    )
    with pytest.raises(PillarError) as partitioned:
        swarm_emergent_topology_heal(topology, ["c"])
    assert partitioned.value.code == "E008"
    assert "partitioned" in str(partitioned.value)


def test_topology_validates_loss_and_weights():
    topology = _dense_topology(4)
    with pytest.raises(PillarError) as unknown:
        swarm_emergent_topology_heal(topology, ["missing"])
    assert unknown.value.code == "E008"
    with pytest.raises(PillarError) as duplicate:
        swarm_emergent_topology_heal(topology, ["n0", "n0"])
    assert duplicate.value.code == "E003"
    with pytest.raises(PillarError) as bad_weight:
        swarm_emergent_topology_heal(
            Topology(("a", "b"), (("a", "b", float("nan")),)),
            [],
        )
    assert bad_weight.value.code == "E008"
    with pytest.raises(PillarError) as limit:
        swarm_emergent_topology_heal(topology, [], max_loss_ratio=0.5)
    assert limit.value.code == "E003"


# ---------------------------------------------------------------------------
# 100. Global invariant audit
# ---------------------------------------------------------------------------

_REPLICAS = Invariant(
    "replicas>=3",
    "at least three replicas",
    lambda state: state["replicas"] >= 3,
)


def test_audit_passes_for_every_node():
    report = swarm_global_invariant_audit(
        {
            "n1": NodeState({"replicas": 3}),
            "n2": {"replicas": 4},
        },
        [_REPLICAS],
    )
    assert report.ok
    assert report.passed == 2
    assert report.failed == 0
    assert report.errors == 0
    assert report.invariants == ("replicas>=3",)
    assert all(record.code == "" for record in report.records)


def test_strict_audit_raises_e008_with_invariant_span():
    invariant = dataclasses.replace(
        _REPLICAS,
        span=(40, 55),
        line=18,
        col=9,
    )
    with pytest.raises(PillarError) as error:
        swarm_global_invariant_audit(
            {"n1": {"replicas": 3}, "n2": {"replicas": 2}},
            [invariant],
        )
    assert error.value.code == "E008"
    assert error.value.span == (40, 55)
    assert (error.value.line, error.value.col) == (18, 9)
    assert "replicas>=3" in str(error.value)
    assert error.value.audit_report.failed == 1
    assert error.value.audit_report.ok is False


def test_non_strict_audit_prefers_node_spans_and_call_site_spans():
    node = NodeState(
        {"replicas": 2},
        span=(2, 8),
        line=3,
        col=6,
    )
    report = swarm_global_invariant_audit(
        {"n1": node},
        [_REPLICAS],
        strict=False,
    )
    record = report.records[0]
    assert record.status == "fail"
    assert record.code == "E008"
    assert record.span == (2, 8)
    assert (record.line, record.col) == (3, 6)

    plain = swarm_global_invariant_audit(
        {"n1": {"replicas": 2}},
        [_REPLICAS],
        strict=False,
        span=(20, 30),
        line=9,
        col=2,
    )
    assert plain.records[0].span == (20, 30)
    assert (plain.records[0].line, plain.records[0].col) == (9, 2)


def test_audit_records_check_exceptions_as_e008_errors():
    broken = Invariant("broken", "always errors", lambda state: 1 / 0)
    report = swarm_global_invariant_audit(
        {"n1": {"replicas": 3}},
        [broken],
        strict=False,
    )
    assert report.errors == 1
    assert not report.ok
    assert report.records[0].status == "error"
    assert report.records[0].code == "E008"
    assert "ZeroDivisionError" in report.records[0].detail


def test_audit_validates_cluster_state_and_invariants():
    with pytest.raises(PillarError) as malformed:
        swarm_global_invariant_audit({"n1": 42}, [_REPLICAS], strict=False)
    assert malformed.value.code == "E008"
    with pytest.raises(PillarError) as empty:
        swarm_global_invariant_audit({}, [_REPLICAS])
    assert empty.value.code == "E003"
    with pytest.raises(PillarError) as duplicate:
        swarm_global_invariant_audit({"n1": {}}, [_REPLICAS, _REPLICAS])
    assert duplicate.value.code == "E003"
    with pytest.raises(PillarError) as bad_span:
        swarm_global_invariant_audit(
            {"n1": {}},
            [dataclasses.replace(_REPLICAS, span=(8, 3))],
        )
    assert bad_span.value.code == "E003"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

_PUBLIC_FUNCTIONS = (
    "swarm_gossip_epidemic_sync",
    "swarm_neural_panic_detector",
    "swarm_autonomous_patch_synth",
    "swarm_byzantine_quorum_sign",
    "swarm_sybil_reputation_score",
    "swarm_stigmergy_task_routing",
    "swarm_canary_rollout_verify",
    "swarm_epidemic_rollback_kill",
    "swarm_emergent_topology_heal",
    "swarm_global_invariant_audit",
)


def test_all_ten_public_functions_are_exported_with_span_parameters():
    for name in _PUBLIC_FUNCTIONS:
        function = getattr(swarm, name)
        assert callable(function)
        assert name in swarm.__all__
        parameters = inspect.signature(function).parameters
        assert {"span", "line", "col"} <= set(parameters)
        assert parameters["span"].annotation == "tuple[int, int] | None"
