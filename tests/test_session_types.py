"""Unit, property, and adversarial tests for bounded multiparty session types."""

from __future__ import annotations

import random

import pytest

from compiler.session_types import (
    ChoiceStep,
    EndStep,
    GlobalProtocol,
    LinearOperation,
    LocalAction,
    LocalBranch,
    MessageStep,
    RepeatStep,
    session_synthesize_and_project_protocols,
    session_verify_linear_channel_consumption,
)
from runtime.diagnostics import PillarError

SPAN = (4, 12)


def _crossed_losers() -> dict[str, object]:
    """Both roles wait for a message the other never sends: a real deadlock."""

    return {
        "name": "crossed",
        "roles": ["A", "B"],
        "locals": {
            "A": [
                {"kind": "recv", "peer": "B", "label": "n"},
                {"kind": "send", "peer": "B", "label": "m"},
            ],
            "B": [
                {"kind": "recv", "peer": "A", "label": "m"},
                {"kind": "send", "peer": "A", "label": "n"},
            ],
        },
    }


def _two_party_choice() -> GlobalProtocol:
    return GlobalProtocol(
        "two_party_choice",
        ("A", "B"),
        (
            MessageStep("A", "B", "req"),
            ChoiceStep(
                "B",
                (
                    LocalBranch("ok", (MessageStep("B", "A", "resp"), EndStep())),
                    LocalBranch("err", (MessageStep("B", "A", "fail"), EndStep())),
                ),
            ),
        ),
    )


def test_projection_produces_local_endpoints_and_proves_progress() -> None:
    report = session_synthesize_and_project_protocols(_two_party_choice())

    assert report.deadlock_free and report.progress
    assert not report.bounded
    assert report.variants == 2
    assert [endpoint.role for endpoint in report.endpoints] == ["A", "B"]
    a, b = report.endpoints
    assert a.text == "A: !req(B); offer { ok: ?resp(B) & err: ?fail(B) }"
    assert b.text == "B: ?req(A); choose { ok: !resp(A) + err: !fail(A) }"
    assert len(a.digest) == 24
    assert [item.kind for item in report.diagnostics] == ["proof"]
    assert report.digest() == report.digest()


def test_mapping_protocol_is_normalized() -> None:
    protocol = {
        "name": "ping",
        "roles": ["client", "server"],
        "steps": [
            {
                "kind": "message",
                "sender": "client",
                "receiver": "server",
                "label": "ping",
                "payload": "i64",
            },
            {"kind": "end"},
        ],
    }
    report = session_synthesize_and_project_protocols(protocol, span=SPAN)

    assert report.deadlock_free and report.progress
    assert report.protocol == "ping"
    assert "client: !ping(server)" in report.endpoints[0].text


def test_roles_are_inferred_when_omitted() -> None:
    protocol = (
        MessageStep("x", "y", "hello"),
        MessageStep("y", "x", "bye"),
    )
    report = session_synthesize_and_project_protocols(protocol)
    assert report.roles == ("x", "y")
    assert report.deadlock_free and report.progress


def test_reachable_deadlock_is_reported_with_trace() -> None:
    report = session_synthesize_and_project_protocols(_crossed_losers())

    assert not report.deadlock_free
    assert not report.progress
    assert report.diagnostics[0].code == "E002"
    assert report.diagnostics[0].kind == "deadlock"
    assert report.traces
    assert any("blocked state" in step for step in report.traces[0])


def test_deadlock_strict_mode_raises_pillar_error() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            _crossed_losers(), strict=True, span=SPAN
        )
    assert excinfo.value.code == "E002"
    assert excinfo.value.span == SPAN


def test_independent_local_choices_are_all_explored() -> None:
    # One safe interleaving (A sends, B receives) and one deadlocked
    # interleaving (both receive) must make the whole session fail.
    protocol = {
        "name": "independent",
        "roles": ["A", "B"],
        "locals": {
            "A": [
                {
                    "kind": "choice",
                    "branches": {
                        "send_first": [{"kind": "send", "peer": "B", "label": "m"}],
                        "recv_first": [{"kind": "recv", "peer": "B", "label": "m"}],
                    },
                }
            ],
            "B": [
                {
                    "kind": "choice",
                    "branches": {
                        "recv_first": [{"kind": "recv", "peer": "A", "label": "m"}],
                        "send_first": [{"kind": "send", "peer": "A", "label": "m"}],
                    },
                }
            ],
        },
    }
    report = session_synthesize_and_project_protocols(protocol)
    assert report.variants == 4
    assert not report.deadlock_free
    assert report.bounds["variants_from"] == "locals"


def test_independent_local_choice_bound_yields_unknown() -> None:
    branches = {
        f"b{index}": [{"kind": "send", "peer": "B", "label": f"m{index}"}]
        for index in range(4)
    }
    other = {
        f"b{index}": [{"kind": "recv", "peer": "A", "label": f"m{index}"}]
        for index in range(4)
    }
    protocol = {
        "name": "wide_local",
        "roles": ["A", "B"],
        "locals": {
            "A": [{"kind": "choose", "branches": branches}],
            "B": [{"kind": "offer", "branches": other}],
        },
    }
    report = session_synthesize_and_project_protocols(protocol, max_variants=8)
    assert report.bounded
    assert report.diagnostics[0].kind == "variant-bound"
    assert report.diagnostics[0].code == "E008"
    assert report.bounds["variants_from"] == "locals"


def test_buffered_send_before_receive_is_deadlock_free() -> None:
    protocol = {
        "name": "pipeline",
        "roles": ["A", "B"],
        "locals": {
            "A": [{"kind": "send", "peer": "B", "label": "m"}],
            "B": [{"kind": "recv", "peer": "A", "label": "m"}],
        },
    }
    report = session_synthesize_and_project_protocols(protocol)
    assert report.deadlock_free and report.progress
    assert [item.kind for item in report.diagnostics] == ["proof"]


def test_buffer_bound_is_reported_as_unknown() -> None:
    # A sends twice before B receives once, so a buffer of one overflows.
    protocol = {
        "name": "overflow",
        "roles": ["A", "B"],
        "locals": {
            "A": [
                {"kind": "send", "peer": "B", "label": "m"},
                {"kind": "send", "peer": "B", "label": "m"},
            ],
            "B": [
                {"kind": "recv", "peer": "A", "label": "m"},
                {"kind": "recv", "peer": "A", "label": "m"},
            ],
        },
    }
    report = session_synthesize_and_project_protocols(protocol, max_buffer=1)
    assert report.bounded
    assert report.diagnostics[0].code == "E008"
    assert "buffer bound" in report.diagnostics[0].message


def test_global_protocols_are_total_orders_and_deadlock_free() -> None:
    protocol = GlobalProtocol(
        "sequence",
        ("A", "B"),
        (
            MessageStep("A", "B", "request"),
            MessageStep("B", "A", "reply"),
            MessageStep("A", "B", "ack"),
        ),
    )
    report = session_synthesize_and_project_protocols(protocol)
    assert report.deadlock_free and report.progress
    assert report.bounds["variants_from"] == "global"


def test_local_endpoint_validation() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            {
                "name": "unknown_peer",
                "roles": ["A"],
                "locals": {"A": [{"kind": "send", "peer": "Z", "label": "x"}]},
            },
            strict=True,
        )
    assert excinfo.value.code == "E001"

    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            {
                "name": "self",
                "roles": ["A"],
                "locals": {"A": [{"kind": "send", "peer": "A", "label": "x"}]},
            },
            strict=True,
        )
    assert excinfo.value.code == "E007"


def test_local_action_objects_are_accepted() -> None:
    protocol = GlobalProtocol(
        "objects",
        ("A", "B"),
        (),
        (
            (
                "A",
                (
                    LocalAction("send", "B", "ping"),
                    LocalAction("receive", "B", "pong"),
                ),
            ),
            (
                "B",
                (
                    LocalAction("receive", "A", "ping"),
                    LocalAction("send", "A", "pong"),
                ),
            ),
        ),
    )
    report = session_synthesize_and_project_protocols(protocol)
    assert report.deadlock_free and report.progress
    assert report.endpoints[0].text == "A: !ping(B); ?pong(B)"


def test_variant_bound_yields_e008_and_unknown() -> None:
    branches = tuple(
        LocalBranch(
            f"b{index}",
            (MessageStep("A", "B", f"m{index}"), MessageStep("B", "A", "ack")),
        )
        for index in range(6)
    )
    protocol = GlobalProtocol("wide", ("A", "B"), (ChoiceStep("A", branches),))
    report = session_synthesize_and_project_protocols(protocol, max_variants=4)

    assert report.bounded
    assert not report.deadlock_free and not report.progress
    assert report.diagnostics[0].code == "E008"
    assert report.diagnostics[0].kind == "variant-bound"
    assert report.bounds["max_variants"] == 4


def test_repeat_bound_is_rejected_before_exploration() -> None:
    protocol = GlobalProtocol(
        "looped",
        ("A", "B"),
        (RepeatStep(1000, (MessageStep("A", "B", "tick"),)),),
    )
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(protocol, strict=True)
    assert excinfo.value.code == "E008"


def test_bounded_repeat_projects_and_terminates() -> None:
    protocol = GlobalProtocol(
        "bounded",
        ("A", "B"),
        (
            RepeatStep(
                3,
                (
                    MessageStep("A", "B", "data"),
                    MessageStep("B", "A", "ack"),
                ),
            ),
            EndStep(),
        ),
    )
    report = session_synthesize_and_project_protocols(protocol)
    assert report.deadlock_free and report.progress
    assert "loop(3)" in report.endpoints[0].text


def test_unknown_role_and_self_message_are_rejected() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            GlobalProtocol("u", ("A",), (MessageStep("A", "Z", "x"),)),
            strict=True,
        )
    assert excinfo.value.code == "E001"

    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            GlobalProtocol("s", ("A",), (MessageStep("A", "A", "x"),)),
            strict=True,
        )
    assert excinfo.value.code == "E007"


def test_steps_after_end_are_rejected() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            GlobalProtocol(
                "late",
                ("A", "B"),
                (EndStep(), MessageStep("A", "B", "x")),
            ),
            strict=True,
        )
    assert excinfo.value.code == "E003"


def test_choice_needs_two_unique_branches() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_synthesize_and_project_protocols(
            GlobalProtocol(
                "one",
                ("A", "B"),
                (ChoiceStep("A", (LocalBranch("only", ()),)),),
            ),
            strict=True,
        )
    assert excinfo.value.code == "E003"


def test_non_positive_bounds_are_rejected() -> None:
    for kwargs in (
        {"max_variants": 0},
        {"max_states": 0},
        {"max_steps": 0},
        {"timeout_ms": 0},
    ):
        with pytest.raises(PillarError) as excinfo:
            session_synthesize_and_project_protocols(
                _two_party_choice(), strict=True, **kwargs
            )
        assert excinfo.value.code == "E008"


def test_projection_is_deterministic_across_runs() -> None:
    first = session_synthesize_and_project_protocols(_two_party_choice())
    second = session_synthesize_and_project_protocols(_two_party_choice())
    assert first.as_dict() == second.as_dict()


def test_linear_trace_accepts_balanced_session() -> None:
    report = session_verify_linear_channel_consumption(
        [
            {"kind": "open", "channel": "c", "endpoints": ["A", "B"]},
            {"kind": "send", "channel": "c", "role": "A"},
            {"kind": "recv", "channel": "c", "role": "B"},
            {"kind": "close", "channel": "c", "role": "A"},
        ],
        span=SPAN,
    )
    assert report.accepted
    assert not report.diagnostics
    (channel,) = report.channels
    assert channel.state == "closed"
    assert channel.tokens == 0
    assert channel.operations == ("open", "send", "recv", "close")
    assert len(report.digest()) == 24


def test_duplicate_capability_is_structured_error() -> None:
    report = session_verify_linear_channel_consumption(
        [
            LinearOperation("open", "c"),
            LinearOperation("duplicate", "c"),
            LinearOperation("close", "c"),
        ]
    )
    assert not report.accepted
    assert report.diagnostics[0].code == "E001"
    assert "duplicated" in report.diagnostics[0].message


def test_operations_before_open_and_after_close() -> None:
    before = session_verify_linear_channel_consumption([LinearOperation("send", "c")])
    assert before.diagnostics[0].code == "E006"

    after = session_verify_linear_channel_consumption(
        [
            LinearOperation("open", "c"),
            LinearOperation("close", "c"),
            LinearOperation("send", "c"),
        ]
    )
    assert after.diagnostics[0].code == "E001"
    assert "operation after closed" in after.diagnostics[0].message


def test_unconsumed_token_is_reported_as_leak() -> None:
    report = session_verify_linear_channel_consumption(
        [LinearOperation("open", "c"), LinearOperation("send", "c")]
    )
    assert not report.accepted
    assert any("still open" in item.message for item in report.diagnostics)
    assert report.diagnostics[-1].code == "E002"


def test_split_requires_both_tokens_to_be_closed() -> None:
    half = session_verify_linear_channel_consumption(
        [
            LinearOperation("open", "c"),
            LinearOperation("split", "c"),
            LinearOperation("close", "c"),
        ]
    )
    assert not half.accepted
    assert "still open" in half.diagnostics[-1].message

    whole = session_verify_linear_channel_consumption(
        [
            LinearOperation("open", "c"),
            LinearOperation("split", "c"),
            LinearOperation("close", "c"),
            LinearOperation("close", "c"),
        ]
    )
    assert whole.accepted
    assert whole.channels[0].tokens == 0


def test_affine_drop_policy() -> None:
    trace = [LinearOperation("open", "c"), LinearOperation("drop", "c")]
    assert session_verify_linear_channel_consumption(trace).accepted

    strict = session_verify_linear_channel_consumption(trace, allow_affine=False)
    assert not strict.accepted
    assert strict.diagnostics[0].code == "E001"


def test_unknown_operation_and_bad_bounds_are_rejected() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_verify_linear_channel_consumption(
            [{"kind": "teleport", "channel": "c"}], strict=True
        )
    assert excinfo.value.code == "E003"

    with pytest.raises(PillarError) as excinfo:
        session_verify_linear_channel_consumption(
            [LinearOperation("open", "c")], max_operations=0, strict=True
        )
    assert excinfo.value.code == "E008"

    with pytest.raises(PillarError) as excinfo:
        session_verify_linear_channel_consumption(
            [LinearOperation("open", "c"), LinearOperation("open", "d")],
            max_operations=1,
            strict=True,
        )
    assert excinfo.value.code == "E008"


def test_endpoint_ownership_is_enforced() -> None:
    report = session_verify_linear_channel_consumption(
        [
            {"kind": "open", "channel": "c", "endpoints": ["A"]},
            {"kind": "send", "channel": "c", "role": "B"},
            {"kind": "close", "channel": "c", "role": "A"},
        ]
    )
    assert not report.accepted
    assert report.diagnostics[0].code == "E001"
    assert "not an endpoint" in report.diagnostics[0].message


def test_strict_linear_violation_raises_first_error() -> None:
    with pytest.raises(PillarError) as excinfo:
        session_verify_linear_channel_consumption(
            [LinearOperation("open", "c"), LinearOperation("open", "c")],
            strict=True,
            span=SPAN,
        )
    assert excinfo.value.code == "E001"
    assert excinfo.value.span == SPAN


def test_random_balanced_traces_are_accepted() -> None:
    rng = random.Random(20261122)
    for _ in range(50):
        channels = [f"c{index}" for index in range(rng.randint(1, 4))]
        operations: list[LinearOperation] = []
        for channel in channels:
            operations.append(LinearOperation("open", channel))
            for _ in range(rng.randint(0, 3)):
                operations.append(
                    LinearOperation(rng.choice(("send", "recv")), channel)
                )
            operations.append(LinearOperation("close", channel))
        rng.shuffle(operations)
        # Shuffling can legitimately reorder open/close per channel, so only
        # assert determinism and structural well-formedness here.
        first = session_verify_linear_channel_consumption(operations)
        second = session_verify_linear_channel_consumption(operations)
        assert first.as_dict() == second.as_dict()

    # An ordered walk over every channel is always accepted.
    operations = []
    for channel in channels:
        operations.extend(
            [
                LinearOperation("open", channel),
                LinearOperation("send", channel),
                LinearOperation("close", channel),
            ]
        )
    assert session_verify_linear_channel_consumption(operations).accepted


def test_random_crossed_protocols_always_deadlock() -> None:
    rng = random.Random(7)
    for _ in range(20):
        a_to_b = f"a{rng.randint(0, 99)}"
        b_to_a = f"b{rng.randint(0, 99)}"
        protocol = {
            "name": "random_cross",
            "roles": ["A", "B"],
            "locals": {
                "A": [
                    {"kind": "recv", "peer": "B", "label": b_to_a},
                    {"kind": "send", "peer": "B", "label": a_to_b},
                ],
                "B": [
                    {"kind": "recv", "peer": "A", "label": a_to_b},
                    {"kind": "send", "peer": "A", "label": b_to_a},
                ],
            },
        }
        report = session_synthesize_and_project_protocols(protocol)
        assert not report.deadlock_free


def test_linear_operation_mapping_normalization() -> None:
    report = session_verify_linear_channel_consumption(
        [
            {"kind": "acquire", "name": "ch", "endpoints": "A"},
            {"kind": "write", "channel": "ch", "role": "A", "payload": "x"},
            {"kind": "release", "channel": "ch"},
        ]
    )
    assert report.accepted
    assert report.channels[0].operations == ("acquire", "write", "release")
