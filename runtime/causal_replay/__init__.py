"""Causal time-travel replay, delta slicing, and boundary isolation.

The package records deterministic happens-before relations with vector and
Lamport clocks, slices the minimal structurally responsible past of a failure,
and confines replay artifacts to scoped temp/cache directories.  It never
orders events by wall-clock time and never touches ``MAP_JIT``, PEP 667 live
frames, raw addresses, or privileged paths.

See ``docs/causal_replay.md`` and SPEC 8.17.
"""

from __future__ import annotations

from runtime.causal_replay.graph import (
    CausalGraph,
    EventKind,
    ReplayEvent,
    VectorClock,
    replay_record_causal_happens_before,
)
from runtime.causal_replay.isolation import (
    BoundaryIsolationReport,
    BoundaryStatus,
    Feature,
    replay_enforce_boundary_isolation,
)
from runtime.causal_replay.slicing import (
    CausalSlice,
    Minimality,
    SliceStatus,
    replay_slice_minimal_failure_causality,
)

__all__ = [
    "BoundaryIsolationReport",
    "BoundaryStatus",
    "CausalGraph",
    "CausalSlice",
    "EventKind",
    "Feature",
    "Minimality",
    "ReplayEvent",
    "SliceStatus",
    "VectorClock",
    "replay_enforce_boundary_isolation",
    "replay_record_causal_happens_before",
    "replay_slice_minimal_failure_causality",
]
