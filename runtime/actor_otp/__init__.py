"""Elixir/BEAM/OTP-inspired actor isolation, supervision, and fault recovery.

Three submodules bring OTP's signature strengths to ``polyglot-lang`` while
respecting platform/runtime boundaries (``MAP_JIT``, PEP 667, raw-address
TOCTOU, and model-only paths are never touched):

* :mod:`runtime.actor_otp.process` — lightweight, isolated actor processes
  with message passing (copying/transferring values, never sharing mutable
  raw pointers across actors) and structured mailboxes.
* :mod:`runtime.actor_otp.supervisor` — OTP-style supervision strategies
  (``one_for_one``, ``one_for_all``, ``rest_for_one``) with restart
  intensity/window limits and explicit escalation when budgets are exceeded.
* :mod:`runtime.actor_otp.crash` — let-it-crash fault isolation, structured
  ``E001``–``E008`` fault reports, and bounded telemetry/audit events.

All code is pure Python and deterministic: time is supplied by the caller
or via an injected clock, and no wall-clock sleeping is used.
"""

from __future__ import annotations

from runtime.actor_otp.crash import (
    CrashReport,
    Telemetry,
    TelemetryEvent,
    classify_fault,
)
from runtime.actor_otp.process import (
    Actor,
    ActorId,
    Mailbox,
    ProcessRegistry,
    spawn_actor,
)
from runtime.actor_otp.supervisor import (
    RestartIntensity,
    RestartOutcome,
    Strategy,
    SupervisionTree,
    Supervisor,
)

__all__ = [
    "Actor",
    "ActorId",
    "CrashReport",
    "Mailbox",
    "ProcessRegistry",
    "RestartIntensity",
    "RestartOutcome",
    "Strategy",
    "SupervisionTree",
    "Supervisor",
    "Telemetry",
    "TelemetryEvent",
    "classify_fault",
    "spawn_actor",
]
