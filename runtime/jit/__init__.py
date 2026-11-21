"""Speculative JIT orchestration frame.

This package sits above :mod:`backends.jit` (the AArch64 encoder and the
``MAP_JIT`` memory manager).  It adds hotness tracking, guards, deoptimization
snapshots, inline-cache patching, invalidation, and structured ``E008``
diagnostics.  Every privileged operation is model-only unless the caller
explicitly opts in on a supported host.
"""

from runtime.jit.framework import (
    CodeBlock,
    DeoptFrame,
    GuardFailure,
    GuardReport,
    GuardSpec,
    IcSite,
    InvalidationReport,
    JitState,
    JitTraceRecord,
    jit_compile_hot_trace,
    jit_deoptimize_to_interpreter_frame,
    jit_emit_diagnostic_e008,
    jit_emit_type_and_bounds_guards,
    jit_invalidate_on_redefinition_or_hotpatch,
    jit_manage_executable_memory_rx,
    jit_patch_ic_and_call_sites,
    new_jit_state,
    release_executable_memory_block,
)

__all__ = [
    "CodeBlock",
    "DeoptFrame",
    "GuardFailure",
    "GuardReport",
    "GuardSpec",
    "IcSite",
    "InvalidationReport",
    "JitState",
    "JitTraceRecord",
    "jit_compile_hot_trace",
    "jit_deoptimize_to_interpreter_frame",
    "jit_emit_diagnostic_e008",
    "jit_emit_type_and_bounds_guards",
    "jit_invalidate_on_redefinition_or_hotpatch",
    "jit_manage_executable_memory_rx",
    "jit_patch_ic_and_call_sites",
    "new_jit_state",
    "release_executable_memory_block",
]
