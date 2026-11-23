# Elixir / OTP actor runtime (experimental)

`runtime/actor_otp/` brings Erlang/OTP's signature strengths to
`polyglot-lang` as a deterministic, pure-Python runtime model:

- **Share-nothing actor isolation** — each actor owns its state and mailbox;
  messages are deep-copied at the send boundary, so an actor can never mutate
  another actor's state through a shared reference.
- **OTP supervision strategies** — `one_for_one`, `one_for_all`, and
  `rest_for_one` with restart-intensity budgets and explicit escalation
  (`E004`) when the budget is exceeded.
- **Let-it-crash fault isolation** — crashes become structured `E001`–`E008`
  `CrashReport` values and bounded telemetry events, never an unhandled
  exception that takes down unrelated subsystems.

These modules are a **runtime model**. They do not spawn OS threads, do not
provide real BEAM/OTP semantics, and are not part of the Python/native parity
contract.

- Normative rules: [SPECIFICATION.md](../SPECIFICATION.md) Section 8.15
- Boundary entry: SPEC 10.44
- Tests: `tests/test_actor_otp.py`

## Requirements

None beyond Python 3.14.8. The model is deterministic: time is supplied by
the caller as a float, and no wall-clock sleeping is used.

```python
from runtime.actor_otp import (
    spawn_actor,
    Strategy,
    Supervisor,
    RestartIntensity,
    Telemetry,
    classify_fault,
)
```

## Isolated actor processes

An actor owns its `state` and a `Mailbox`. The `handler` callable is invoked
as `handler(state, payload)` for each message and returns the new state (or
`None` to keep the current state).

```python
def counter(state, payload):
    if payload == "increment":
        return state + 1
    return None


actor = spawn_actor(0, counter)
actor.send("increment")
actor.send("increment")
actor.run()
assert actor.state == 2
```

### Isolation guarantee

State and payloads are deep-copied at the boundary:

```python
actor = spawn_actor([], lambda s, p: s + [p])
original = [1, 2]
actor.send(original)
original.append(99)  # mutate after send
actor.run()
assert actor.state == [[1, 2]]  # the copy was taken at send time
```

- `spawn_actor` deep-copies the initial state, so later caller mutations do
  not leak in.
- `Mailbox.send` deep-copies the payload.
- A `BorrowOnly` or borrowed value cannot cross the boundary as a shared
  mutable reference; the model copies by value.

### Mailbox semantics

- `send(payload, sender=...)` appends a deep copy; a full bounded mailbox
  raises `E001`.
- `receive()` pops the oldest message.
- `receive_until(predicate)` drains until a predicate matches and **preserves
  the rest** in order.
- `stop()` marks the actor dead and discards pending messages; sending to a
  dead actor raises `E001`.

### Let-it-crash

If the handler raises, the actor is marked dead and the exception propagates.
The supervisor is responsible for catching it and restarting the actor from
its initial state — defensive error handling inside the actor is unnecessary.

## Supervision

A `Supervisor` tracks restart history per child and enforces a global
`RestartIntensity` budget.

```python
from runtime.actor_otp import RestartIntensity, Strategy, Supervisor

sup = Supervisor("root", Strategy.ONE_FOR_ONE, RestartIntensity(5, 60.0))
sup.add_child("worker1", lambda: spawn_actor(0, counter))
sup.add_child("worker2", lambda: spawn_actor(0, counter))

outcome = sup.restart("worker1", now=1.0)
assert outcome.restarted == ("worker1",)
assert outcome.escalated is False
```

### Strategies

| Strategy        | Restarts                                             |
| --------------- | ---------------------------------------------------- |
| `one_for_one`   | Only the crashed child                               |
| `one_for_all`   | Every child                                          |
| `rest_for_one`  | The crashed child and every child started after it   |

### Restart intensity

`RestartIntensity(max_restarts, window_seconds)` bounds restarts. When more
than `max_restarts` restarts fall within `window_seconds` of the most recent
restart, the supervisor raises `E004` and escalates:

```python
sup = Supervisor("root", Strategy.ONE_FOR_ONE, RestartIntensity(2, 60.0))
sup.add_child("w", lambda: spawn_actor(0, counter))
sup.restart("w", now=1.0)
sup.restart("w", now=2.0)
sup.restart("w", now=3.0)  # error[E004]: restart intensity exceeded
```

A restart outside the window does not count against the budget, so a fresh
start is possible after escalation.

## Crash reports and telemetry

`classify_fault` maps an exception to a structured `CrashReport` with an
`E001`–`E008` code, the traceback, and a boundary-safety flag:

```python
from runtime.actor_otp import classify_fault

try:
    raise ValueError("bad input")
except Exception as exc:
    report = classify_fault(exc, actor_name="worker1")

assert report.code == "E001"
assert report.boundary_safe is True
```

| Exception              | Code   |
| ---------------------- | ------ |
| `ValueError`, `TypeError` | `E001` |
| `RecursionError`       | `E002` |
| `MemoryError`, `TimeoutError` | `E004` |
| `ImportError`          | `E006` |
| `SystemExit`, `KeyboardInterrupt`, `OSError` | `E007` |
| any other              | `E004` |
| `PolyglotError`        | its own code |

### Platform boundary guard

A fault handler that attempts to touch a forbidden platform boundary raises
`E007` instead of silently proceeding. The guarded keywords are `MAP_JIT`,
`PEP 667`, `f_locals`, `raw address`, `TOCTOU`, and `model-only`:

```python
try:
    raise ValueError("attempted to access MAP_JIT region")
except Exception as exc:
    classify_fault(exc)  # error[E007]: forbidden platform boundary
```

### Bounded telemetry

`Telemetry` is a ring buffer capped at 10,000 events. When full, the oldest
event is dropped — telemetry never grows unbounded and never crashes the host.

```python
from runtime.actor_otp import Telemetry

tel = Telemetry(max_events=3)
tel.record("a", "crash", "something failed", code="E001", timestamp=1.0)
tel.record("a", "restart", "restarted", timestamp=2.0)
tel.record("b", "crash", "other failure", code="E002", timestamp=3.0)
tel.record("b", "crash", "overflow", code="E004", timestamp=4.0)  # drops oldest
assert len(tel) == 3
```

## Tests

`tests/test_actor_otp.py` covers:

- actor spawn, message delivery, and state updates;
- isolation guarantees (initial-state copy, payload deep-copy at send);
- sending to dead actors, handler crashes, mailbox capacity, and
  `receive_until` preservation;
- all three supervision strategies and restart-set contents;
- restart-intensity exhaustion, window reset, and zero-budget escalation;
- crash classification for each exception class and boundary-violation
  rejection;
- telemetry bounding, per-actor filtering, and clearing.

Run it directly with:

```sh
pytest -q tests/test_actor_otp.py
```