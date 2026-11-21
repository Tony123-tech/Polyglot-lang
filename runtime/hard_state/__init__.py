"""Hard state frame (Frame J).

The frame models aliasing and live-state migration with explicit boundaries.
It never claims zero aliasing risk or zero-risk hot-patching: unsupported raw
addresses, FFI, executable memory, and dynamic PEP 667 frames are reported as
hazards or structured diagnostics.
"""

from __future__ import annotations

from runtime.hard_state.framework import (
    Access,
    AliasFinding,
    FieldSpec,
    MemoryObject,
    MigrationPlan,
    MigrationStep,
    PointerFact,
    PointsToReport,
    ReplayReport,
    Schema,
    StateCapsule,
    hard_analyze_points_to_and_alias_sets,
    hard_plan_live_state_schema_migration,
    hard_quarantine_and_replay_faults,
)

__all__ = [
    "Access",
    "AliasFinding",
    "FieldSpec",
    "MemoryObject",
    "MigrationPlan",
    "MigrationStep",
    "PointerFact",
    "PointsToReport",
    "ReplayReport",
    "Schema",
    "StateCapsule",
    "hard_analyze_points_to_and_alias_sets",
    "hard_plan_live_state_schema_migration",
    "hard_quarantine_and_replay_faults",
]
