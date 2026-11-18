"""Experimental runtime foundations for the advanced pillar prototypes.

Nothing in this package is imported by the normal ``polyglot`` CLI path.
The modules are opt-in and are documented in ``docs/experimental_pillars.md``.
"""

from __future__ import annotations

from runtime.capabilities import (
    CAPABILITIES,
    CAPABILITY_NAMES,
    Capability,
    CapabilityStatus,
    capability_namespace,
    capability_signature,
    describe,
    install_capability_bridge,
    invoke,
    reject,
)
from runtime.diagnostics import ERROR_CODES, PillarError
from runtime.native_bindings import generate_lib_rs, write_crate

__all__ = [
    "CAPABILITIES",
    "CAPABILITY_NAMES",
    "ERROR_CODES",
    "Capability",
    "CapabilityStatus",
    "PillarError",
    "capability_namespace",
    "capability_signature",
    "describe",
    "generate_lib_rs",
    "install_capability_bridge",
    "invoke",
    "reject",
    "write_crate",
]
