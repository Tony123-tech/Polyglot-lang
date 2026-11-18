"""Quantum hot-patching subsystem (experimental pillar I).

The package implements five cooperating mechanisms:

``hotpatch_live_dylib_trampoline``
    Atomic GOT/PLT pointer swaps and owned-page AArch64/x86_64 jump sleds.
``hotpatch_cpython_frame_eval``
    Safe-point function-bytecode swaps with live fast-local migration.
``hotpatch_mio_reactor_state``
    Selector / native-``mio`` descriptor migration with zero socket loss.
``hotpatch_rcu_ast_epoch``
    Epoch-based reclamation for immutable AST roots.
``hotpatch_rollback_journal_commit``
    Before-image object/buffer journaling with instant rollback.

The honest boundaries are part of the API: foreign executable text, live
``f_code`` pointer replacement, and OS page CoW without ``fork`` are refused
with ``E007``/``E008`` rather than approximated.  See
``docs/quantum_hotpatch.md``.
"""

from __future__ import annotations

import sys
from typing import Any

from runtime.hotpatch.errors import (
    ByteSpan,
    HotpatchError,
    abi_error,
    byte_offsets,
    relocation_error,
    span_from_line_col,
    unsupported,
)
from runtime.hotpatch.frames import (
    FrameEvalMigrator,
    FrameLayout,
    FramePatch,
    FrameSnapshot,
    SafepointDispatcher,
    hotpatch_cpython_frame_eval,
    install_safepoint,
)
from runtime.hotpatch.journal import (
    Journal,
    RollbackJournal,
    Snapshot,
    hotpatch_rollback_journal_commit,
)
from runtime.hotpatch.rcu import (
    AstEpochManager,
    RcuDomain,
    RcuStats,
    hotpatch_rcu_ast_epoch,
    wait_for_epoch,
)
from runtime.hotpatch.reactor import (
    MioBridgeMigration,
    MioReactorBridge,
    ReactorMigration,
    ReactorState,
    capture_selector,
    hotpatch_mio_reactor_state,
    migrate_mio_bridge,
    migrate_selector,
)
from runtime.hotpatch.trampoline import (
    ExecutableRegion,
    GotSlot,
    LiveTrampoline,
    TextPatch,
    TrampolinePatch,
    encode_jump,
    find_got_slots,
    hotpatch_live_dylib_trampoline,
    parse_signature,
    resolve_symbol,
    signature_text,
)

__version__ = "0.1.0"

#: Status metadata for the five subsystem entry points.  ``mechanism`` is the
#: real primitive, ``boundary`` is what is deliberately refused.
CAPABILITIES: dict[str, dict[str, Any]] = {
    "hotpatch_live_dylib_trampoline": {
        "mechanism": "atomic GOT/PLT cell swap; owned MAP_JIT inline sled",
        "boundary": "foreign __TEXT is refused (E007)",
        "diagnostic": "E008 for relocation/ABI errors",
    },
    "hotpatch_cpython_frame_eval": {
        "mechanism": "future-entry code swap + PEP 667 fast-local migration",
        "boundary": "no mid-instruction f_code rewrite (E007)",
        "diagnostic": "E008 for layout/closure mismatches",
    },
    "hotpatch_mio_reactor_state": {
        "mechanism": "selector re-registration and fd adoption; mio bridge seam",
        "boundary": "requires a live selector or native bridge",
        "diagnostic": "E008 for closed descriptors",
    },
    "hotpatch_rcu_ast_epoch": {
        "mechanism": "epoch counters, read guards, grace-period reclamation",
        "boundary": "cooperative readers; no kernel RCU syscall",
        "diagnostic": "E008 on grace-period timeout",
    },
    "hotpatch_rollback_journal_commit": {
        "mechanism": "before-image attribute/buffer snapshots and undo log",
        "boundary": "nested containers need deep=True or attribute replacement",
        "diagnostic": "E008 on verification/undo failure",
    },
}


def platform_support() -> dict[str, bool]:
    """Report which primitives are available in this process."""
    return {
        "platform": sys.platform,
        "executable_region": ExecutableRegion.available(),
        "got_discovery": bool(resolve_symbol("malloc"))
        and (sys.platform == "darwin" or sys.platform.startswith("linux")),
        "frame_locals": hasattr(sys, "monitoring") and sys.version_info >= (3, 13),
        "selector": True,
    }


__all__ = [
    "CAPABILITIES",
    "AstEpochManager",
    "ByteSpan",
    "ExecutableRegion",
    "FrameEvalMigrator",
    "FrameLayout",
    "FramePatch",
    "FrameSnapshot",
    "GotSlot",
    "HotpatchError",
    "Journal",
    "LiveTrampoline",
    "MioBridgeMigration",
    "MioReactorBridge",
    "RcuDomain",
    "RcuStats",
    "ReactorMigration",
    "ReactorState",
    "RollbackJournal",
    "SafepointDispatcher",
    "Snapshot",
    "TextPatch",
    "TrampolinePatch",
    "abi_error",
    "byte_offsets",
    "capture_selector",
    "encode_jump",
    "find_got_slots",
    "hotpatch_cpython_frame_eval",
    "hotpatch_live_dylib_trampoline",
    "hotpatch_mio_reactor_state",
    "hotpatch_rcu_ast_epoch",
    "hotpatch_rollback_journal_commit",
    "install_safepoint",
    "migrate_mio_bridge",
    "migrate_selector",
    "parse_signature",
    "platform_support",
    "relocation_error",
    "resolve_symbol",
    "signature_text",
    "span_from_line_col",
    "unsupported",
    "wait_for_epoch",
]
