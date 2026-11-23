"""Capability registry for the 10 "impossible" engineering pillars.

This module is the single source of truth for the 100 functions in the unified
directive.  It deliberately separates three things that are easy to conflate:

``FOUNDATION``
    A real, testable implementation exists (possibly a deliberately small
    one) and can be called through :func:`invoke`.
``MODELLED``
    A deterministic in-process model exists.  It is useful for reasoning and
    tests but is not a substitute for the privileged/hardware mechanism named
    in the directive.  The model is exposed through :func:`invoke`.
``REJECTED``
    The requested mechanism cannot be provided safely or portably by this
    project (unstable CPython internals, loaded-image text patching, raw
    ``mprotect``/eBPF/MPK/CFI from a pure-Python runtime, real GPU/ANE
    dispatch, ...).  Both backends reject it with the *same* structured
    ``PillarError`` and the caller-supplied character span.

Both backends receive the identical namespace from :func:`capability_namespace`.
For Python this is a direct host bridge; for the native backend it is the same
host bridge until the opt-in PyO3 scaffold in ``native_pillars/`` is built.
This is a parity guarantee about observable behaviour, not a claim that a
userspace module can rewrite executable text or rewind a live PyO3 stack.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from typing import Any

from runtime.diagnostics import PillarError

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class CapabilityStatus(str, Enum):
    FOUNDATION = "foundation"
    MODELLED = "modelled"
    REJECTED = "rejected"


SUBSYSTEMS: tuple[str, ...] = (
    "hotpatch",
    "evolution",
    "synthesis",
    "timetravel",
    "transmute",
    "migrate",
    "dsm",
    "sandbox",
    "accelerate",
    "swarm",
)


@dataclass(frozen=True)
class Capability:
    """One entry in the 100-function directive."""

    number: int
    name: str
    subsystem: str
    module: str
    attr: str | None
    status: CapabilityStatus
    code: str
    summary: str
    boundary: str

    @property
    def supported(self) -> bool:
        return self.status is not CapabilityStatus.REJECTED

    @property
    def backends(self) -> tuple[str, ...]:
        return ("python", "native") if self.supported else ()


# (name, subsystem, module, attr, status, code, summary, boundary)
_ROWS: tuple[tuple[str, str, str, str | None, CapabilityStatus, str, str, str], ...] = (
    # I. Quantum & zero-downtime memory hot-patching
    (
        "hotpatch_live_dylib_trampoline",
        "hotpatch",
        "runtime.hotpatch.trampoline",
        "hotpatch_live_dylib_trampoline",
        CapabilityStatus.FOUNDATION,
        "E008",
        "GOT/PLT cell swaps and owned-page inline JMP sleds",
        "Writable GOT/PLT cells and owned MAP_JIT sleds are patched atomically; foreign __TEXT is refused (E007) and ABI/relocation errors are E008.",
    ),
    (
        "hotpatch_cpython_frame_eval",
        "hotpatch",
        "runtime.hotpatch.frames",
        "hotpatch_cpython_frame_eval",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Swap function bytecode at a safe point and migrate fast locals",
        "Future-entry code swaps with PEP 667 fast-local migration are real; mid-instruction f_code pointer replacement is refused (E007), layout/closure mismatches are E008.",
    ),
    (
        "hotpatch_mio_reactor_state",
        "hotpatch",
        "runtime.hotpatch.reactor",
        "hotpatch_mio_reactor_state",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Preserve the selector/mio reactor across a patch",
        "Registrations are re-armed on a fresh selector and mio bridge fds are adopted so sockets stay open; closed descriptors are E008.",
    ),
    (
        "hotpatch_rcu_ast_epoch",
        "hotpatch",
        "runtime.hotpatch.rcu",
        "hotpatch_rcu_ast_epoch",
        CapabilityStatus.MODELLED,
        "E008",
        "RCU epoch reclamation for live AST nodes",
        "Cooperative user-space epoch counters, read guards and grace-period reclamation are implemented; no kernel RCU syscall is claimed, timeouts are E008.",
    ),
    (
        "hotpatch_shadow_heap_reconcile",
        "hotpatch",
        "backends.hotpatch",
        "diff_programs",
        CapabilityStatus.MODELLED,
        "E001",
        "Reconcile divergent live object layouts",
        "Layout diffs are computed from ASTs; page-fault driven reconciliation is rejected.",
    ),
    (
        "hotpatch_tls_slot_relocate",
        "hotpatch",
        "backends.hotpatch",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Relocate TLS offsets in a reloaded cdylib",
        "TLS layout is fixed at load time; no stable ABI exists for offset relocation.",
    ),
    (
        "hotpatch_rollback_journal_commit",
        "hotpatch",
        "runtime.hotpatch.journal",
        "hotpatch_rollback_journal_commit",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Commit/rollback a before-image patch journal",
        "Attribute/buffer before-images and an undo log restore state; nested containers need deep=True or attribute replacement, verification/undo failures are E008.",
    ),
    (
        "hotpatch_abi_shim_synthesize",
        "hotpatch",
        "backends.hotpatch",
        None,
        CapabilityStatus.REJECTED,
        "E001",
        "Generate runtime assembly FFI thunks",
        "ABI-changing signatures are rejected; only ABI-compatible dispatch is supported.",
    ),
    (
        "hotpatch_async_waker_rebind",
        "hotpatch",
        "backends.hotpatch",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Rebind Waker vtables to new code segments",
        "Rust Waker vtables are immutable after construction; futures keep their original code.",
    ),
    (
        "hotpatch_gil_free_quiesce",
        "hotpatch",
        "backends.hotpatch",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Lock-free quiescence across PEP 684 sub-interpreters",
        "The patch manager uses a documented quiescent-point protocol, not lock-free sub-interpreter pauses.",
    ),
    # II. Self-synthesizing neuromorphic meta-compiler
    (
        "meta_evolve_ast_rewrite_rules",
        "evolution",
        "compiler.evolution",
        "EvolutionEngine.propose",
        CapabilityStatus.FOUNDATION,
        "E004",
        "Synthesize auditable AST rewrite candidates",
        "Candidates come from a fixed template library and require sandbox validation.",
    ),
    (
        "meta_cargo_stderr_auto_repair",
        "evolution",
        "compiler.evolution",
        "EvolutionEngine.observe_compile_failure",
        CapabilityStatus.FOUNDATION,
        "E004",
        "Record rustc diagnostics for pass synthesis",
        "Errors are observed and classified; generated code is never silently rewritten in place.",
    ),
    (
        "meta_superoptimize_bytecode_seq",
        "evolution",
        "compiler.evolution",
        None,
        CapabilityStatus.REJECTED,
        "E002",
        "SMT-derived shortest equivalent bytecode",
        "No SMT solver dependency exists; equivalence is checked by corpus replay only.",
    ),
    (
        "meta_bootstrap_self_compile",
        "evolution",
        "compiler.evolution",
        None,
        CapabilityStatus.REJECTED,
        "E004",
        "Bit-reproducible compiler self-hosting",
        "The compiler is not self-hosting; deterministic bootstrap is not available.",
    ),
    (
        "meta_monomorph_explosion_prune",
        "evolution",
        "compiler.analysis.monomorph",
        "monomorphize",
        CapabilityStatus.FOUNDATION,
        "E003",
        "Deduplicate monomorphized generic instances",
        "Concrete mangled declarations are deduplicated with a recursion limit.",
    ),
    (
        "meta_profile_guided_inliner",
        "evolution",
        "compiler.evolution",
        None,
        CapabilityStatus.REJECTED,
        "E004",
        "Cross-language LTO of hot FFI boundaries",
        "No LLVM bitcode pipeline is wired into the PyO3 backend.",
    ),
    (
        "meta_speculative_loop_vectorize",
        "evolution",
        "compiler.evolution",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Synthesize AVX-512/NEON from list comprehensions",
        "Vector intrinsics require backend-specific codegen that this project does not emit.",
    ),
    (
        "meta_cache_line_struct_pack",
        "evolution",
        "compiler.evolution",
        "meta_cache_line_struct_pack",
        CapabilityStatus.MODELLED,
        "E003",
        "Reorder fields using cache-miss telemetry",
        "Field order is modelled from declarations; no hardware counter feedback is collected.",
    ),
    (
        "meta_deadlock_graph_mutate",
        "evolution",
        "proofs.synthesis",
        "verify_lock_order",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Detect lock-order inversions",
        "A cycle is proven from the declared order; rewriting to atomics is out of scope.",
    ),
    (
        "meta_codegen_parity_fuzzer",
        "evolution",
        "compiler.evolution",
        "EvolutionEngine.evaluate",
        CapabilityStatus.FOUNDATION,
        "E004",
        "Replay candidate passes against a corpus",
        "The sandbox compares observable main() results before promotion.",
    ),
    # III. Bi-directional formal verification & synthesis
    (
        "verify_natural_lang_invariant",
        "synthesis",
        "proofs.synthesis",
        "interpret_spec",
        CapabilityStatus.FOUNDATION,
        "E003",
        "Parse controlled invariant language",
        "A small grammar is supported; unrestricted natural language is rejected.",
    ),
    (
        "synthesize_proven_ast_body",
        "synthesis",
        "proofs.synthesis",
        "synthesize",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Enumerate and prove a candidate body",
        "Proofs hold only in the documented i64 interval/polynomial fragment.",
    ),
    (
        "prove_ffi_memory_safety",
        "synthesis",
        "proofs.synthesis",
        None,
        CapabilityStatus.REJECTED,
        "E002",
        "Prove no raw pointer outlives Py<T>",
        "Pointer lifetimes are not represented in the .poly type system.",
    ),
    (
        "verify_termination_ranking_fn",
        "synthesis",
        "compiler.analysis.termination",
        "check_function",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Check declared termination",
        "The existing #[terminates] checker validates structural recursion; ranking synthesis is not claimed.",
    ),
    (
        "extract_counterexample_trace",
        "synthesis",
        "proofs.synthesis",
        "extract_counterexample_trace",
        CapabilityStatus.MODELLED,
        "E002",
        "Convert a failed obligation into a test",
        "Failed interval checks record the offending expression, not a full SMT model.",
    ),
    (
        "prove_constant_time_crypto",
        "synthesis",
        "proofs.synthesis",
        None,
        CapabilityStatus.REJECTED,
        "E002",
        "Prove secret-independent branching",
        "No taint/symbolic-execution engine is available for cryptographic code.",
    ),
    (
        "verify_linear_type_consumption",
        "synthesis",
        "proofs.synthesis",
        None,
        CapabilityStatus.REJECTED,
        "E002",
        "Enforce affine resource usage",
        "Linear types are not part of the language; closure is enforced by the runtime helper only.",
    ),
    (
        "synthesize_lock_free_cas_loop",
        "synthesis",
        "distributed.dsm",
        "dsm_atomic_cross_node_cas",
        CapabilityStatus.MODELLED,
        "E002",
        "Synthesize a verified CAS state machine",
        "A coordinator-simulated CAS is provided; generated lock-free Rust is not.",
    ),
    (
        "prove_equivalence_across_backends",
        "synthesis",
        "proofs.synthesis",
        "compile_both",
        CapabilityStatus.FOUNDATION,
        "E004",
        "Cross-backend compilation of a proved program",
        "Emission is guarded by a certificate; semantic equivalence is tested, not proved for all programs.",
    ),
    (
        "verify_no_panics_reachable",
        "synthesis",
        "proofs.synthesis",
        "verify_bounds",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Prove literal indices and emitted asserts",
        "Dynamic indices remain unknown; overflow behavior remains platform-defined.",
    ),
    # IV. Cross-FFI time-travel debugger
    (
        "ttd_record_deterministic_replay",
        "timetravel",
        "debug.timetravel",
        "TimeTravelDebugger.trace",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Record deterministic transitions",
        "Only registered transitions are replayable; syscalls/RDTSC are not captured.",
    ),
    (
        "ttd_reverse_step_instruction",
        "timetravel",
        "debug.timetravel",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Reverse single machine instructions across FFI",
        "Instruction-level reverse execution requires a hypervisor or ptrace implementation.",
    ),
    (
        "ttd_snapshot_dual_heaps",
        "timetravel",
        "debug.timetravel",
        "snapshot_object_graph",
        CapabilityStatus.MODELLED,
        "E008",
        "Freeze Python and native heaps coherently",
        "Python object graphs are captured by identity; native state comes from a provider protocol.",
    ),
    (
        "ttd_retroactive_var_mutate",
        "timetravel",
        "debug.timetravel",
        "TimeTravelDebugger.modify",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Edit a past variable and replay dependents",
        "Replay is only as sound as the registered transition functions.",
    ),
    (
        "ttd_causal_dataflow_slice",
        "timetravel",
        "debug.timetravel",
        "NativeDiff.render",
        CapabilityStatus.MODELLED,
        "E003",
        "Render a key-level causal differential",
        "No byte-level dynamic slicing across Rust and Python is performed.",
    ),
    (
        "ttd_rr_hardware_perf_counters",
        "timetravel",
        "debug.timetravel",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Synchronize PMU counters with bytecode events",
        "Hardware counters are not exposed by the runtime and are not portable.",
    ),
    (
        "ttd_time_travel_watchpoint",
        "timetravel",
        "debug.timetravel",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Jump to the cycle of the last address write",
        "No memory watchpoint/trap infrastructure exists in the Python runtime.",
    ),
    (
        "ttd_fork_timeline_branch",
        "timetravel",
        "debug.timetravel",
        "TimeTravelDebugger.travel_to",
        CapabilityStatus.MODELLED,
        "E006",
        "Branch a timeline from a checkpoint",
        "Multiple independent debugger instances are the supported branching model.",
    ),
    (
        "ttd_serialize_closure_upvalues",
        "timetravel",
        "debug.timetravel",
        None,
        CapabilityStatus.REJECTED,
        "E008",
        "Serialize live closures and async state machines",
        "Generator and PyO3 async frames cannot be portably reconstructed.",
    ),
    (
        "ttd_emit_flamegraph_diff",
        "timetravel",
        "debug.timetravel",
        "NativeDiff.render",
        CapabilityStatus.MODELLED,
        "E003",
        "Summarize state divergence after a mutation",
        "A textual differential is emitted; no sampled flamegraph is produced.",
    ),
    # V. Speculative dynamic-to-native JIT transmutation
    (
        "jit_type_lattice_speculate",
        "transmute",
        "runtime.transmute",
        "signature_of",
        CapabilityStatus.FOUNDATION,
        "E001",
        "Profile exact runtime type signatures",
        "Signatures are exact type tuples, not an inferred lattice.",
    ),
    (
        "jit_background_cranelift_emit",
        "transmute",
        "runtime.transmute",
        "SpeculativeTransmuter",
        CapabilityStatus.MODELLED,
        "E004",
        "Compile a signature on a worker thread",
        "A pluggable specializer runs in the background; Cranelift is not bundled.",
    ),
    (
        "jit_osr_on_stack_replacement",
        "transmute",
        "runtime.transmute",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Replace a running frame mid-iteration",
        "Live CPython frames are never rewritten; in-flight calls keep their resolved callable.",
    ),
    (
        "jit_nanosecond_deopt_bailout",
        "transmute",
        "runtime.transmute",
        "SpeculativeTransmuter.deoptimize",
        CapabilityStatus.MODELLED,
        "E001",
        "Guard miss falls back to the original callable",
        "The guard is a tuple compare plus a Python call; no nanosecond latency is claimed.",
    ),
    (
        "jit_unbox_pyobject_floats",
        "transmute",
        "runtime.transmute",
        None,
        CapabilityStatus.REJECTED,
        "E001",
        "Keep IEEE-754 floats in SIMD registers across loops",
        "The Python reference path boxes values; no register allocation is available.",
    ),
    (
        "jit_inline_cache_polymorphic",
        "transmute",
        "runtime.transmute",
        "jit_inline_cache_polymorphic",
        CapabilityStatus.MODELLED,
        "E001",
        "Model a 4-way polymorphic inline cache",
        "The dispatch table is modelled with an active-signature guard only.",
    ),
    (
        "jit_escape_analysis_scalar_repl",
        "transmute",
        "runtime.transmute",
        None,
        CapabilityStatus.REJECTED,
        "E002",
        "Scalar-replace non-escaping structs",
        "No escape analysis or stack allocation is performed by the Python backend.",
    ),
    (
        "jit_guard_elision_dominator",
        "transmute",
        "runtime.transmute",
        "SpeculativeTransmuter.path",
        CapabilityStatus.MODELLED,
        "E002",
        "Report the currently installed guard",
        "Dominator-based elision is not implemented; the active signature is authoritative.",
    ),
    (
        "jit_gc_safepoint_map_emit",
        "transmute",
        "runtime.transmute",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Emit stack maps for native JIT frames",
        "No JIT stack frames are visible to CPython's cyclic GC.",
    ),
    (
        "jit_trace_tree_stitching",
        "transmute",
        "runtime.transmute",
        "SpeculativeTransmuter.profile",
        CapabilityStatus.MODELLED,
        "E001",
        "Record hot call-site signatures",
        "Profiles are counters; trace trees and side-exits are not generated.",
    ),
    # VI. Transparent cross-node distributed state migration
    (
        "migrate_freeze_async_coro",
        "migrate",
        "distributed.tasks",
        "CheckpointableTask.to_payload",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Freeze an await journal",
        "Migration is cooperative replay from a checkpoint, not a live stack copy.",
    ),
    (
        "migrate_zero_copy_rdma_ship",
        "migrate",
        "distributed.transport",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Stream stacks over RDMA",
        "TCP/in-memory transports are supplied; pinned RDMA buffers are not.",
    ),
    (
        "migrate_rehydrate_native_stack",
        "migrate",
        "distributed.native_bridge",
        "from_native_value",
        CapabilityStatus.MODELLED,
        "E001",
        "Rebuild concrete native layouts on the target",
        "Layouts are reconstructed via public constructors, not raw stack frames.",
    ),
    (
        "migrate_socket_fd_proxy_tunnel",
        "migrate",
        "distributed.connections",
        "AnchorRegistry",
        CapabilityStatus.FOUNDATION,
        "E006",
        "Keep sockets anchored across migration",
        "Channels are rebound to an anchored endpoint; no kernel fd duplication is used.",
    ),
    (
        "migrate_closure_env_closure_pack",
        "migrate",
        "distributed.transmute",
        "encode_state",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Encode a versioned, digest-checked state",
        "Only registered, wire-safe values are transported.",
    ),
    (
        "migrate_deterministic_rng_sync",
        "migrate",
        "distributed.transmute",
        None,
        CapabilityStatus.REJECTED,
        "E007",
        "Clone cryptographic PRNG state across nodes",
        "PRNG internals are deliberately opaque and are not transported.",
    ),
    (
        "migrate_predictive_load_evict",
        "migrate",
        "distributed.cluster",
        "Cluster.migrate",
        CapabilityStatus.MODELLED,
        "E006",
        "Migrate a task between nodes",
        "Migration is explicit; thermal/CPU prediction is not implemented.",
    ),
    (
        "migrate_causal_message_buffer",
        "migrate",
        "distributed.actor",
        "Actor",
        CapabilityStatus.MODELLED,
        "E008",
        "Buffer and replay actor messages",
        "A mailbox is replayed in recorded order; causal delivery is bounded by that order.",
    ),
    (
        "migrate_heterogeneous_isa_bridge",
        "migrate",
        "distributed.native_bridge",
        None,
        CapabilityStatus.REJECTED,
        "E001",
        "Translate live frames between ARM64 and x86_64",
        "Only architecture-neutral value layouts are portable.",
    ),
    (
        "migrate_rollback_on_link_drop",
        "migrate",
        "distributed.cluster",
        "Cluster.migrate",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Abort a failed transfer and resume locally",
        "Prepare uses digest/handler validation before commit; the source remains authoritative until then.",
    ),
    # VII. Distributed shared memory & consensus
    (
        "dsm_virtual_page_fault_fetch",
        "dsm",
        "distributed.dsm",
        "dsm_virtual_page_fault_fetch",
        CapabilityStatus.MODELLED,
        "E007",
        "Simulate remote page fetch",
        "Real userfaultfd/Mach exception ports are rejected with E007.",
    ),
    (
        "dsm_compile_struct_to_crdt",
        "dsm",
        "distributed.dsm",
        "dsm_compile_struct_to_crdt",
        CapabilityStatus.FOUNDATION,
        "E001",
        "Lower struct fields to CRDTs",
        "LWW-Register, OR-Set and PN-Counter are implemented in-process.",
    ),
    (
        "dsm_raft_quorum_future_yield",
        "dsm",
        "distributed.dsm",
        "dsm_raft_quorum_future_yield",
        CapabilityStatus.MODELLED,
        "E008",
        "Drive a Raft-like log to quorum",
        "A single-process consensus model is provided; no real cluster quorum is contacted.",
    ),
    (
        "dsm_merkle_tree_state_diff",
        "dsm",
        "distributed.dsm",
        "dsm_merkle_tree_state_diff",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Diff Merkle-rooted page digests",
        "Rolling hashes identify divergent pages; wire transfer is out of scope.",
    ),
    (
        "dsm_vector_clock_causality",
        "dsm",
        "distributed.dsm",
        "dsm_vector_clock_causality",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Stamp mutations with HLC/vector clocks",
        "Causal anomalies are detected from recorded clocks.",
    ),
    (
        "dsm_split_brain_partition_fence",
        "dsm",
        "distributed.dsm",
        "dsm_split_brain_partition_fence",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Fence a stale primary",
        "Fencing tokens are checked; actual network partition detection is supplied by the caller.",
    ),
    (
        "dsm_mvcc_snapshot_isolation",
        "dsm",
        "distributed.dsm",
        "dsm_mvcc_snapshot_isolation",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Provide versioned snapshot reads",
        "Reads see a consistent version; conflicts are rejected deterministically.",
    ),
    (
        "dsm_false_sharing_page_split",
        "dsm",
        "distributed.dsm",
        "dsm_false_sharing_page_split",
        CapabilityStatus.MODELLED,
        "E003",
        "Split hot pages on ping-pong invalidation",
        "Invalidations come from recorded telemetry, not kernel page faults.",
    ),
    (
        "dsm_distributed_gc_lease",
        "dsm",
        "distributed.dsm",
        "dsm_distributed_gc_lease",
        CapabilityStatus.FOUNDATION,
        "E006",
        "Lease-based distributed reclamation",
        "Expired leases are reported; reclamation is explicit.",
    ),
    (
        "dsm_atomic_cross_node_cas",
        "dsm",
        "distributed.dsm",
        "dsm_atomic_cross_node_cas",
        CapabilityStatus.MODELLED,
        "E008",
        "Coordinate a cross-node CAS",
        "A 2PC-style coordinator is simulated in-process.",
    ),
    # VIII. Hardware-enforced zero-cost sandboxing
    (
        "sandbox_mprotect_heap_isolate",
        "sandbox",
        "security.sandbox",
        "sandbox_mprotect_heap_isolate",
        CapabilityStatus.MODELLED,
        "E007",
        "Describe mprotect guard-page isolation",
        "A declarative policy is emitted; no mprotect call is made.",
    ),
    (
        "sandbox_ebpf_syscall_synthesize",
        "sandbox",
        "security.sandbox",
        "sandbox_ebpf_syscall_synthesize",
        CapabilityStatus.MODELLED,
        "E007",
        "Describe a syscall allowlist",
        "Portable JSON policy is emitted; real eBPF/Seatbelt generation is rejected.",
    ),
    (
        "sandbox_pku_mpk_domain_switch",
        "sandbox",
        "security.sandbox",
        "sandbox_pku_mpk_domain_switch",
        CapabilityStatus.MODELLED,
        "E007",
        "Model MPK/POE domain tokens",
        "No protection-key instructions are executed.",
    ),
    (
        "sandbox_fat_pointer_bounds_verify",
        "sandbox",
        "security.sandbox",
        "sandbox_fat_pointer_bounds_verify",
        CapabilityStatus.FOUNDATION,
        "E001",
        "Check tagged capability bounds",
        "Spatial/temporal checks are real Python-level checks.",
    ),
    (
        "sandbox_cfi_shadow_stack_emit",
        "sandbox",
        "security.sandbox",
        "sandbox_cfi_shadow_stack_emit",
        CapabilityStatus.MODELLED,
        "E007",
        "Describe CFI landing pads",
        "No shadow stack or landing pad is emitted in machine code.",
    ),
    (
        "sandbox_spectre_lfence_inject",
        "sandbox",
        "security.sandbox",
        "sandbox_spectre_lfence_inject",
        CapabilityStatus.MODELLED,
        "E007",
        "Annotate speculation barriers",
        "Annotations are emitted; machine barriers are not injected.",
    ),
    (
        "sandbox_deterministic_fuel_meter",
        "sandbox",
        "security.sandbox",
        "sandbox_deterministic_fuel_meter",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Meter loop fuel deterministically",
        "Fuel exhaustion raises a structured termination diagnostic.",
    ),
    (
        "sandbox_taint_flow_enforce",
        "sandbox",
        "security.sandbox",
        "sandbox_taint_flow_enforce",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Propagate taint and block sinks",
        "Taint is tracked per bit/field; exec/eval sinks are rejected.",
    ),
    (
        "sandbox_zeroize_secret_drop",
        "sandbox",
        "security.sandbox",
        "sandbox_zeroize_secret_drop",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Best-effort secret scrubbing",
        "Python cannot guarantee compiler-unoptimizable zeroization; the caveat is explicit.",
    ),
    (
        "sandbox_violation_trap_recover",
        "sandbox",
        "security.sandbox",
        "sandbox_violation_trap_recover",
        CapabilityStatus.MODELLED,
        "E008",
        "Journal and recover violations",
        "Signals are not caught on a dedicated alternate stack.",
    ),
    # IX. Heterogeneous NPU/GPU autonomous JIT offloading
    (
        "accel_detect_tensor_polyhedral",
        "accelerate",
        "compiler.heterogeneous",
        "accel_detect_tensor_polyhedral",
        CapabilityStatus.FOUNDATION,
        "E003",
        "Recognize affine loop nests",
        "A conservative affine detector is implemented; general polyhedral analysis is not.",
    ),
    (
        "accel_lower_ast_to_mlir",
        "accelerate",
        "compiler.heterogeneous",
        "accel_lower_ast_to_mlir",
        CapabilityStatus.MODELLED,
        "E003",
        "Emit textual MLIR-like IR",
        "The output is a readable IR descriptor, not a validated MLIR module.",
    ),
    (
        "accel_emit_metal_cuda_spirv",
        "accelerate",
        "compiler.heterogeneous",
        "accel_emit_metal_cuda_spirv",
        CapabilityStatus.MODELLED,
        "E007",
        "Emit kernel source text",
        "Real compilation and device dispatch are rejected; #[gpu]/#[npu] are feature-gated.",
    ),
    (
        "accel_zero_copy_unified_mem",
        "accelerate",
        "compiler.heterogeneous",
        "accel_zero_copy_unified_mem",
        CapabilityStatus.MODELLED,
        "E007",
        "Model a zero-copy buffer view",
        "No unified-memory mapping is created.",
    ),
    (
        "accel_npu_ane_graph_compile",
        "accelerate",
        "compiler.heterogeneous",
        "accel_npu_ane_graph_compile",
        CapabilityStatus.MODELLED,
        "E007",
        "Partition a matrix subgraph",
        "A partition descriptor is emitted; ANE dispatch is rejected.",
    ),
    (
        "accel_kernel_fusion_epilogue",
        "accelerate",
        "compiler.heterogeneous",
        "accel_kernel_fusion_epilogue",
        CapabilityStatus.FOUNDATION,
        "E003",
        "Fuse epilogue descriptors",
        "Fusion is a deterministic IR-level transformation.",
    ),
    (
        "accel_async_stream_pipelining",
        "accelerate",
        "compiler.heterogeneous",
        "accel_async_stream_pipelining",
        CapabilityStatus.MODELLED,
        "E003",
        "Model stream/event dependencies",
        "Dependencies are modelled; no DMA engine is driven.",
    ),
    (
        "accel_auto_tune_tile_sizes",
        "accelerate",
        "compiler.heterogeneous",
        "accel_auto_tune_tile_sizes",
        CapabilityStatus.FOUNDATION,
        "E003",
        "Search tile sizes with a cost model",
        "The search is deterministic and hardware-independent.",
    ),
    (
        "accel_mixed_precision_quantize",
        "accelerate",
        "compiler.heterogeneous",
        "accel_mixed_precision_quantize",
        CapabilityStatus.FOUNDATION,
        "E001",
        "Quantize with an error bound",
        "Out-of-bound quantization is rejected instead of silently losing accuracy.",
    ),
    (
        "accel_nanosecond_cpu_fallback",
        "accelerate",
        "compiler.heterogeneous",
        "accel_nanosecond_cpu_fallback",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Select a CPU fallback plan",
        "The fallback is correctness-first; latency is reported honestly.",
    ),
    # X. Distributed agent swarm & autonomous self-healing
    (
        "swarm_gossip_epidemic_sync",
        "swarm",
        "swarm.agent",
        "swarm_gossip_epidemic_sync",
        CapabilityStatus.FOUNDATION,
        "E006",
        "Anti-entropy gossip rounds",
        "Gossip runs over an in-memory registry, not a real network.",
    ),
    (
        "swarm_neural_panic_detector",
        "swarm",
        "swarm.agent",
        "swarm_neural_panic_detector",
        CapabilityStatus.MODELLED,
        "E003",
        "Heuristic anomaly detection",
        "No neural model or external ML dependency is used.",
    ),
    (
        "swarm_autonomous_patch_synth",
        "swarm",
        "swarm.agent",
        "swarm_autonomous_patch_synth",
        CapabilityStatus.FOUNDATION,
        "E002",
        "Synthesize and validate a patch candidate",
        "Candidates are checked against declared properties before rollout.",
    ),
    (
        "swarm_byzantine_quorum_sign",
        "swarm",
        "swarm.agent",
        "swarm_byzantine_quorum_sign",
        CapabilityStatus.MODELLED,
        "E007",
        "Sign transitions and check quorum",
        "HMAC/digest envelopes model signatures; Ed25519 threshold signatures are rejected.",
    ),
    (
        "swarm_sybil_reputation_score",
        "swarm",
        "swarm.agent",
        "swarm_sybil_reputation_score",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Score and quarantine peers",
        "Reputation is computed from recorded evidence.",
    ),
    (
        "swarm_stigmergy_task_routing",
        "swarm",
        "swarm.agent",
        "swarm_stigmergy_task_routing",
        CapabilityStatus.FOUNDATION,
        "E006",
        "Route by pheromone load gradients",
        "Routing is a deterministic in-process model.",
    ),
    (
        "swarm_canary_rollout_verify",
        "swarm",
        "swarm.agent",
        "swarm_canary_rollout_verify",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Verify a canary before global rollout",
        "Promotion requires a clean canary error tally.",
    ),
    (
        "swarm_epidemic_rollback_kill",
        "swarm",
        "swarm.agent",
        "swarm_epidemic_rollback_kill",
        CapabilityStatus.MODELLED,
        "E007",
        "Broadcast a patch kill switch",
        "Kill-switch tokens are verified in-process; no cryptographic multicast is used.",
    ),
    (
        "swarm_emergent_topology_heal",
        "swarm",
        "swarm.agent",
        "swarm_emergent_topology_heal",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Rebuild an MST after node loss",
        "Up to 49% loss is supported; larger loss is rejected.",
    ),
    (
        "swarm_global_invariant_audit",
        "swarm",
        "swarm.agent",
        "swarm_global_invariant_audit",
        CapabilityStatus.FOUNDATION,
        "E008",
        "Audit cluster invariants with spans",
        "Drift produces structured E008 audit records.",
    ),
)


CAPABILITIES: tuple[Capability, ...] = tuple(
    Capability(
        number=index,
        name=name,
        subsystem=subsystem,
        module=module,
        attr=attr,
        status=status,
        code=code,
        summary=summary,
        boundary=boundary,
    )
    for index, (
        name,
        subsystem,
        module,
        attr,
        status,
        code,
        summary,
        boundary,
    ) in enumerate(_ROWS, start=1)
)

BY_NAME: Mapping[str, Capability] = {cap.name: cap for cap in CAPABILITIES}
BY_SUBSYSTEM: Mapping[str, tuple[Capability, ...]] = {
    subsystem: tuple(cap for cap in CAPABILITIES if cap.subsystem == subsystem)
    for subsystem in SUBSYSTEMS
}
CAPABILITY_NAMES: tuple[str, ...] = tuple(cap.name for cap in CAPABILITIES)


# ---------------------------------------------------------------------------
# Diagnostics and dispatch
# ---------------------------------------------------------------------------


_BACKENDS: frozenset[str] = frozenset({"python", "native"})


def span_of(node: Any) -> tuple[int, int] | None:
    """Return the character span carried by an AST node, if any.

    Malformed spans are ignored rather than raising from an error-formatting
    path: a diagnostic must never be the source of a new exception.
    """
    span = getattr(node, "span", None)
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            return (int(span[0]), int(span[1]))
        except TypeError, ValueError:
            return None
    return None


def _location(span: tuple[int, int] | None = None, node: Any = None) -> dict[str, Any]:
    """Build the location kwargs for a :class:`PillarError`.

    A node's real ``line``/``col`` take precedence over the legacy
    ``col = span[0]`` fallback so a source-aware node is never downgraded to a
    raw character offset presented as a column.
    """
    span = span if span is not None else span_of(node)
    if span is None:
        return {}
    location: dict[str, Any] = {"span": span, "line": 0, "col": span[0]}
    if node is not None:
        line = getattr(node, "line", None)
        col = getattr(node, "col", None)
        if isinstance(line, int) and line >= 0:
            location["line"] = line
        if isinstance(col, int) and col >= 0:
            location["col"] = col
    return location


def _check_backend(
    backend: str,
    *,
    span: tuple[int, int] | None = None,
    node: Any = None,
) -> None:
    """Validate a backend name once, at every public dispatch entry point."""
    if backend not in _BACKENDS:
        raise PillarError(
            "E003",
            f"unknown backend `{backend}`",
            reason="capabilities are exposed on 'python' and 'native' only",
            **_location(span, node),
        )


def reject(
    name: str,
    *,
    backend: str = "python",
    span: tuple[int, int] | None = None,
    node: Any = None,
    reason: str | None = None,
) -> PillarError:
    """Build the canonical structured rejection for a capability.

    The returned error is a value; dispatch entry points raise it.  This keeps
    ``reject()`` usable as a diagnostic builder while namespace stubs and
    :func:`invoke` share one raising contract.
    """
    _check_backend(backend, span=span, node=node)
    cap = BY_NAME.get(name)
    if cap is None:
        raise PillarError(
            "E006",
            f"unknown capability `{name}`",
            reason="the directive defines exactly 100 capability names",
            help_text="choose a name from runtime.capabilities.CAPABILITY_NAMES",
            **_location(span, node),
        )
    if cap.status is CapabilityStatus.REJECTED:
        code = cap.code
        message = f"capability `{name}` is not available"
    elif backend not in cap.backends:
        # Defensive: a future capability could expose only one backend.
        code = "E007"
        message = f"capability `{name}` is not available on the `{backend}` backend"
    else:
        code = cap.code
        message = f"capability `{name}` failed"
    return PillarError(
        code,
        message,
        reason=reason or cap.boundary,
        help_text=(
            "use the documented host bridge or an equivalent supported capability"
            if cap.supported
            else "remove the unsupported request and use the supported fallback"
        ),
        note=f"{cap.subsystem}:{cap.status.value}",
        **_location(span, node),
    )


@cache
def resolve(cap: Capability) -> Callable[..., Any] | None:
    """Import a capability implementation lazily; rejected entries return None.

    Invariants:

    * ``None`` means *the optional module is absent* or *no adapter is
      declared* -- never "the module imported badly".  A genuine import error
      inside an installed module propagates so a broken adapter cannot masquer
      as an unimplemented capability.
    * Results are cached.  ``resolve.cache_clear()`` is available for hosts
      that install modules at runtime.
    """
    if cap.attr is None or not cap.supported:
        return None
    try:
        spec = importlib.util.find_spec(cap.module)
    except ImportError, AttributeError, ValueError:
        # A missing/broken parent package is treated like an absent opt-in
        # module; errors raised by the module itself are never swallowed.
        return None
    if spec is None:
        return None
    module = importlib.import_module(cap.module)
    target: Any = module
    for part in cap.attr.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            return None
    return target if callable(target) else None


def _callable_location(
    fn: Callable[..., Any], kwargs: dict[str, Any], args: tuple[Any, ...]
) -> dict[str, Any]:
    """Return the location kwargs that can be injected without a conflict.

    ``span``/``node`` are injected as keywords.  A target whose matching
    parameter is positional-only, or is already satisfied by a positional
    argument, must not receive a duplicate value -- otherwise a diagnostic
    helper would turn a perfectly good call into a ``TypeError``.
    """
    try:
        signature = inspect.signature(fn)
    except TypeError, ValueError:  # builtins without an introspectable signature
        return {}
    positional = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    injected: dict[str, Any] = {}
    for key, value in kwargs.items():
        parameter = signature.parameters.get(key)
        if parameter is None or parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            continue
        if key in positional and positional.index(key) < len(args):
            continue
        injected[key] = value
    return injected


def invoke(
    name: str,
    *args: Any,
    backend: str = "python",
    span: tuple[int, int] | None = None,
    node: Any = None,
    **kwargs: Any,
) -> Any:
    """Call a capability or raise its canonical structured diagnostic.

    ``span`` (or the span carried by ``node``) is propagated to implementations
    that accept it, so a security or consensus failure keeps its exact source
    location on both backends.  Injection is skipped when it would collide with
    a positional argument; the original call then proceeds unchanged.
    """
    _check_backend(backend, span=span, node=node)
    cap = BY_NAME.get(name)
    if cap is None or not cap.supported:
        raise reject(name, backend=backend, span=span, node=node)
    fn = resolve(cap)
    if fn is None:
        raise PillarError(
            "E006",
            f"capability `{name}` has no callable adapter",
            reason=cap.boundary,
            help_text="use capability_namespace() to list callable capabilities",
            **_location(span, node),
        )
    if span is None:
        span = span_of(node)
    injected: dict[str, Any] = {}
    if span is not None:
        injected["span"] = span
    if node is not None:
        injected["node"] = node
    if injected:
        kwargs.update(_callable_location(fn, injected, args))
    return fn(*args, **kwargs)


def _rejection(name: str, *args: Any, backend: str, **kwargs: Any) -> Any:
    """Namespace stub for an unavailable capability.

    Unlike :func:`reject` (a diagnostic *builder*), a namespace entry must
    raise, exactly like the :func:`invoke` entries beside it.  Positional
    arguments are accepted and ignored so the 100-name surface has one calling
    convention.
    """
    del args
    raise reject(
        name,
        backend=backend,
        span=kwargs.get("span"),
        node=kwargs.get("node"),
        reason=kwargs.get("reason"),
    )


def capability_namespace(backend: str = "python") -> dict[str, Callable[..., Any]]:
    """Return the identical 100-name surface for either backend.

    A fresh mapping is returned on every call so callers may install it into a
    host namespace without mutating a shared cache.  Adapter resolution is
    cached by :func:`resolve`, so the 100-name scan is cheap after the first
    call.
    """
    _check_backend(backend)
    return {
        cap.name: (
            partial(invoke, cap.name, backend=backend)
            if cap.supported and resolve(cap) is not None
            else partial(_rejection, cap.name, backend=backend)
        )
        for cap in CAPABILITIES
    }


def install_capability_bridge(
    namespace: dict[str, Any],
    *,
    backend: str = "python",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge the capability surface into a backend result namespace.

    This is deliberately explicit: pillar capabilities are opt-in and never
    injected into the normal ``polyglot run`` / ``polyglot build`` path.
    """
    for name, fn in capability_namespace(backend).items():
        if overwrite or name not in namespace:
            namespace[name] = fn
    return namespace


def describe(name: str) -> str:
    cap = BY_NAME.get(name)
    if cap is None:
        raise reject(name)
    return (
        f"{cap.number:>3}. [{cap.status.value}] {cap.name}\n"
        f"     subsystem: {cap.subsystem}\n"
        f"     signature: {capability_signature(name)}\n"
        f"     summary: {cap.summary}\n"
        f"     boundary: {cap.boundary}"
    )


def capability_signature(name: str, *, backend: str = "python") -> str:
    """Return the callable signature for a capability.

    Rejected or unadapted capabilities report their structured rejection
    contract instead of a Python signature.  ``backend`` is accepted for
    symmetry with :func:`capability_namespace`; both backends share the same
    host bridge and therefore the same signature.
    """
    _check_backend(backend)
    cap = BY_NAME.get(name)
    if cap is None:
        raise reject(name)
    fn = resolve(cap)
    if fn is None:
        return f"{name}(*args, **kwargs) -> PillarError[{cap.code}]"
    try:
        return f"{name}{inspect.signature(fn)}"
    except TypeError, ValueError:  # pragma: no cover - builtins
        return f"{name}(*args, **kwargs)"


__all__ = [
    "BY_NAME",
    "BY_SUBSYSTEM",
    "CAPABILITIES",
    "CAPABILITY_NAMES",
    "SUBSYSTEMS",
    "Capability",
    "CapabilityStatus",
    "capability_namespace",
    "capability_signature",
    "describe",
    "install_capability_bridge",
    "invoke",
    "reject",
    "resolve",
    "span_of",
]
