# Unified impossible-engineering pillar surface

This document is the honest contract for the 10 pillars / 100 functions in
the unified directive. It is an **opt-in** layer: nothing here is imported
by `polyglot run` or `polyglot build`, and no existing conformance test
depends on it.

## Parity and honesty policy

Dual-backend parity means both backends expose the same 100 names and emit
byte-identical structured diagnostics for the same request. It does **not**
mean that a userspace module can patch foreign loaded executable text, replace
a running frame's `f_code` mid-instruction, invoke raw `mprotect`/eBPF/MPK,
dispatch to an ANE, or guarantee nanosecond de-optimisation. Supported safe-point
PEP 667 migration and owned-page sleds are explicit foundation/modelled
capabilities, not those privileged or unstable operations. Unsupported requests
are rejected uniformly with `E007` (or the domain-specific code below) and
preserve the caller's exact character span.

Status values:

- `foundation`: a real, tested implementation is callable.
- `modelled`: a deterministic in-process model is callable; it is not the
  privileged/hardware mechanism named in the directive.
- `rejected`: both backends raise the same structured `PillarError`.

All failures use `runtime.diagnostics.PillarError` (`E001`–`E008`) with
`span`, `line`, and `col` fields. `runtime/capabilities.py` is the single
source of truth; `runtime/native_bindings.py` generates the opt-in PyO3
scaffold from it.

## Using the capability surface

```python
from runtime.capabilities import (
    capability_signature,
    install_capability_bridge,
    invoke,
)

sig = invoke("jit_type_lattice_speculate", (1, 2.0), backend="python")
namespace = {}
install_capability_bridge(namespace, backend="native")
assert len(namespace) == 100
print(capability_signature("jit_type_lattice_speculate"))
```

`capability_signature(name)` returns the adapter's type signature, or the
structured rejection contract when a capability is unavailable. Both backends
share the same host bridge and therefore the same signatures.

For a capability that cannot be provided, pass the source span so the
diagnostic points at the offending syntax:

```python
invoke("hotpatch_tls_slot_relocate", span=(12, 37))  # error[E007]
```

The accelerator feature gate (`runtime/feature_gates.py`) is wired into both
backends: `#[gpu]` and `#[npu]` are rejected with `E007` and the attribute's
exact span, matching the `accel_*` capability boundary below.

## Native binding scaffold

`native_pillars/` contains a generated, opt-in `cdylib` crate with one
`#[pyfunction]` per capability. It is deliberately not built by CI and does
not change `polyglot build`. Regenerate it after editing the registry:

```python
from runtime.native_bindings import write_crate

write_crate("native_pillars")
```

Supported capabilities in the scaffold report `E006` and direct callers to
the explicit host bridge; rejected capabilities report the registry's code.
This is a binding surface, not a claim that the host bridge has been lowered
to Rust.

## Capability matrix

The registry classifies the 100 functions as 43 foundation, 35 modelled,
and 22 rejected. Regenerate this section after registry edits.

### I. Quantum & zero-downtime memory hot-patching

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 1 | `hotpatch_live_dylib_trampoline` | foundation | Writable GOT/PLT cells and owned MAP_JIT sleds are patched atomically; foreign __TEXT is refused (E007) and ABI/relocation errors are E008. |
| 2 | `hotpatch_cpython_frame_eval` | foundation | Future-entry code swaps with PEP 667 fast-local migration are real; mid-instruction f_code pointer replacement is refused (E007), layout/closure mismatches are E008. |
| 3 | `hotpatch_mio_reactor_state` | foundation | Registrations are re-armed on a fresh selector and mio bridge fds are adopted so sockets stay open; closed descriptors are E008. |
| 4 | `hotpatch_rcu_ast_epoch` | modelled | Cooperative user-space epoch counters, read guards and grace-period reclamation are implemented; no kernel RCU syscall is claimed, timeouts are E008. |
| 5 | `hotpatch_shadow_heap_reconcile` | modelled | Layout diffs are computed from ASTs; page-fault driven reconciliation is rejected. |
| 6 | `hotpatch_tls_slot_relocate` | rejected | TLS layout is fixed at load time; no stable ABI exists for offset relocation. |
| 7 | `hotpatch_rollback_journal_commit` | foundation | Attribute/buffer before-images and an undo log restore state; nested containers need deep=True or attribute replacement, verification/undo failures are E008. |
| 8 | `hotpatch_abi_shim_synthesize` | rejected | ABI-changing signatures are rejected; only ABI-compatible dispatch is supported. |
| 9 | `hotpatch_async_waker_rebind` | rejected | Rust Waker vtables are immutable after construction; futures keep their original code. |
| 10 | `hotpatch_gil_free_quiesce` | rejected | The patch manager uses a documented quiescent-point protocol, not lock-free sub-interpreter pauses. |

### II. Self-synthesizing neuromorphic meta-compiler

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 11 | `meta_evolve_ast_rewrite_rules` | foundation | Candidates come from a fixed template library and require sandbox validation. |
| 12 | `meta_cargo_stderr_auto_repair` | foundation | Errors are observed and classified; generated code is never silently rewritten in place. |
| 13 | `meta_superoptimize_bytecode_seq` | rejected | No SMT solver dependency exists; equivalence is checked by corpus replay only. |
| 14 | `meta_bootstrap_self_compile` | rejected | The compiler is not self-hosting; deterministic bootstrap is not available. |
| 15 | `meta_monomorph_explosion_prune` | foundation | Concrete mangled declarations are deduplicated with a recursion limit. |
| 16 | `meta_profile_guided_inliner` | rejected | No LLVM bitcode pipeline is wired into the PyO3 backend. |
| 17 | `meta_speculative_loop_vectorize` | rejected | Vector intrinsics require backend-specific codegen that this project does not emit. |
| 18 | `meta_cache_line_struct_pack` | modelled | Field order is modelled from declarations; no hardware counter feedback is collected. |
| 19 | `meta_deadlock_graph_mutate` | foundation | A cycle is proven from the declared order; rewriting to atomics is out of scope. |
| 20 | `meta_codegen_parity_fuzzer` | foundation | The sandbox compares observable main() results before promotion. |

### III. Bi-directional formal verification & synthesis

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 21 | `verify_natural_lang_invariant` | foundation | A small grammar is supported; unrestricted natural language is rejected. |
| 22 | `synthesize_proven_ast_body` | foundation | Proofs hold only in the documented i64 interval/polynomial fragment. |
| 23 | `prove_ffi_memory_safety` | rejected | Pointer lifetimes are not represented in the .poly type system. |
| 24 | `verify_termination_ranking_fn` | foundation | The existing #[terminates] checker validates structural recursion; ranking synthesis is not claimed. |
| 25 | `extract_counterexample_trace` | modelled | Failed interval checks record the offending expression, not a full SMT model. |
| 26 | `prove_constant_time_crypto` | rejected | No taint/symbolic-execution engine is available for cryptographic code. |
| 27 | `verify_linear_type_consumption` | rejected | Linear types are not part of the language; closure is enforced by the runtime helper only. |
| 28 | `synthesize_lock_free_cas_loop` | modelled | A coordinator-simulated CAS is provided; generated lock-free Rust is not. |
| 29 | `prove_equivalence_across_backends` | foundation | Emission is guarded by a certificate; semantic equivalence is tested, not proved for all programs. |
| 30 | `verify_no_panics_reachable` | foundation | Dynamic indices remain unknown; overflow behavior remains platform-defined. |

### IV. Cross-FFI time-travel debugger

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 31 | `ttd_record_deterministic_replay` | foundation | Only registered transitions are replayable; syscalls/RDTSC are not captured. |
| 32 | `ttd_reverse_step_instruction` | rejected | Instruction-level reverse execution requires a hypervisor or ptrace implementation. |
| 33 | `ttd_snapshot_dual_heaps` | modelled | Python object graphs are captured by identity; native state comes from a provider protocol. |
| 34 | `ttd_retroactive_var_mutate` | foundation | Replay is only as sound as the registered transition functions. |
| 35 | `ttd_causal_dataflow_slice` | modelled | No byte-level dynamic slicing across Rust and Python is performed. |
| 36 | `ttd_rr_hardware_perf_counters` | rejected | Hardware counters are not exposed by the runtime and are not portable. |
| 37 | `ttd_time_travel_watchpoint` | rejected | No memory watchpoint/trap infrastructure exists in the Python runtime. |
| 38 | `ttd_fork_timeline_branch` | modelled | Multiple independent debugger instances are the supported branching model. |
| 39 | `ttd_serialize_closure_upvalues` | rejected | Generator and PyO3 async frames cannot be portably reconstructed. |
| 40 | `ttd_emit_flamegraph_diff` | modelled | A textual differential is emitted; no sampled flamegraph is produced. |

### V. Speculative dynamic-to-native transmutation

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 41 | `jit_type_lattice_speculate` | foundation | Signatures are exact type tuples, not an inferred lattice. |
| 42 | `jit_background_cranelift_emit` | modelled | A pluggable specializer runs in the background; Cranelift is not bundled. |
| 43 | `jit_osr_on_stack_replacement` | rejected | Live CPython frames are never rewritten; in-flight calls keep their resolved callable. |
| 44 | `jit_nanosecond_deopt_bailout` | modelled | The guard is a tuple compare plus a Python call; no nanosecond latency is claimed. |
| 45 | `jit_unbox_pyobject_floats` | rejected | The Python reference path boxes values; no register allocation is available. |
| 46 | `jit_inline_cache_polymorphic` | modelled | The dispatch table is modelled with an active-signature guard only. |
| 47 | `jit_escape_analysis_scalar_repl` | rejected | No escape analysis or stack allocation is performed by the Python backend. |
| 48 | `jit_guard_elision_dominator` | modelled | Dominator-based elision is not implemented; the active signature is authoritative. |
| 49 | `jit_gc_safepoint_map_emit` | rejected | No JIT stack frames are visible to CPython's cyclic GC. |
| 50 | `jit_trace_tree_stitching` | modelled | Profiles are counters; trace trees and side-exits are not generated. |

### VI. Transparent cross-node state migration

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 51 | `migrate_freeze_async_coro` | foundation | Migration is cooperative replay from a checkpoint, not a live stack copy. |
| 52 | `migrate_zero_copy_rdma_ship` | rejected | TCP/in-memory transports are supplied; pinned RDMA buffers are not. |
| 53 | `migrate_rehydrate_native_stack` | modelled | Layouts are reconstructed via public constructors, not raw stack frames. |
| 54 | `migrate_socket_fd_proxy_tunnel` | foundation | Channels are rebound to an anchored endpoint; no kernel fd duplication is used. |
| 55 | `migrate_closure_env_closure_pack` | foundation | Only registered, wire-safe values are transported. |
| 56 | `migrate_deterministic_rng_sync` | rejected | PRNG internals are deliberately opaque and are not transported. |
| 57 | `migrate_predictive_load_evict` | modelled | Migration is explicit; thermal/CPU prediction is not implemented. |
| 58 | `migrate_causal_message_buffer` | modelled | A mailbox is replayed in recorded order; causal delivery is bounded by that order. |
| 59 | `migrate_heterogeneous_isa_bridge` | rejected | Only architecture-neutral value layouts are portable. |
| 60 | `migrate_rollback_on_link_drop` | foundation | Prepare uses digest/handler validation before commit; the source remains authoritative until then. |

### VII. Distributed shared memory & consensus

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 61 | `dsm_virtual_page_fault_fetch` | modelled | Real userfaultfd/Mach exception ports are rejected with E007. |
| 62 | `dsm_compile_struct_to_crdt` | foundation | LWW-Register, OR-Set and PN-Counter are implemented in-process. |
| 63 | `dsm_raft_quorum_future_yield` | modelled | A single-process consensus model is provided; no real cluster quorum is contacted. |
| 64 | `dsm_merkle_tree_state_diff` | foundation | Rolling hashes identify divergent pages; wire transfer is out of scope. |
| 65 | `dsm_vector_clock_causality` | foundation | Causal anomalies are detected from recorded clocks. |
| 66 | `dsm_split_brain_partition_fence` | foundation | Fencing tokens are checked; actual network partition detection is supplied by the caller. |
| 67 | `dsm_mvcc_snapshot_isolation` | foundation | Reads see a consistent version; conflicts are rejected deterministically. |
| 68 | `dsm_false_sharing_page_split` | modelled | Invalidations come from recorded telemetry, not kernel page faults. |
| 69 | `dsm_distributed_gc_lease` | foundation | Expired leases are reported; reclamation is explicit. |
| 70 | `dsm_atomic_cross_node_cas` | modelled | A 2PC-style coordinator is simulated in-process. |

### VIII. Hardware-enforced sandboxing

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 71 | `sandbox_mprotect_heap_isolate` | modelled | A declarative policy is emitted; no mprotect call is made. |
| 72 | `sandbox_ebpf_syscall_synthesize` | modelled | Portable JSON policy is emitted; real eBPF/Seatbelt generation is rejected. |
| 73 | `sandbox_pku_mpk_domain_switch` | modelled | No protection-key instructions are executed. |
| 74 | `sandbox_fat_pointer_bounds_verify` | foundation | Spatial/temporal checks are real Python-level checks. |
| 75 | `sandbox_cfi_shadow_stack_emit` | modelled | No shadow stack or landing pad is emitted in machine code. |
| 76 | `sandbox_spectre_lfence_inject` | modelled | Annotations are emitted; machine barriers are not injected. |
| 77 | `sandbox_deterministic_fuel_meter` | foundation | Fuel exhaustion raises a structured termination diagnostic. |
| 78 | `sandbox_taint_flow_enforce` | foundation | Taint is tracked per bit/field; exec/eval sinks are rejected. |
| 79 | `sandbox_zeroize_secret_drop` | foundation | Python cannot guarantee compiler-unoptimizable zeroization; the caveat is explicit. |
| 80 | `sandbox_violation_trap_recover` | modelled | Signals are not caught on a dedicated alternate stack. |

### IX. Heterogeneous NPU/GPU offloading

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 81 | `accel_detect_tensor_polyhedral` | foundation | A conservative affine detector is implemented; general polyhedral analysis is not. |
| 82 | `accel_lower_ast_to_mlir` | modelled | The output is a readable IR descriptor, not a validated MLIR module. |
| 83 | `accel_emit_metal_cuda_spirv` | modelled | Real compilation and device dispatch are rejected; #[gpu]/#[npu] are feature-gated. |
| 84 | `accel_zero_copy_unified_mem` | modelled | No unified-memory mapping is created. |
| 85 | `accel_npu_ane_graph_compile` | modelled | A partition descriptor is emitted; ANE dispatch is rejected. |
| 86 | `accel_kernel_fusion_epilogue` | foundation | Fusion is a deterministic IR-level transformation. |
| 87 | `accel_async_stream_pipelining` | modelled | Dependencies are modelled; no DMA engine is driven. |
| 88 | `accel_auto_tune_tile_sizes` | foundation | The search is deterministic and hardware-independent. |
| 89 | `accel_mixed_precision_quantize` | foundation | Out-of-bound quantization is rejected instead of silently losing accuracy. |
| 90 | `accel_nanosecond_cpu_fallback` | foundation | The fallback is correctness-first; latency is reported honestly. |

### X. Distributed agent swarm & self-healing

| # | Function | Status | Boundary |
| ---: | --- | --- | --- |
| 91 | `swarm_gossip_epidemic_sync` | foundation | Gossip runs over an in-memory registry, not a real network. |
| 92 | `swarm_neural_panic_detector` | modelled | No neural model or external ML dependency is used. |
| 93 | `swarm_autonomous_patch_synth` | foundation | Candidates are checked against declared properties before rollout. |
| 94 | `swarm_byzantine_quorum_sign` | modelled | HMAC/digest envelopes model signatures; Ed25519 threshold signatures are rejected. |
| 95 | `swarm_sybil_reputation_score` | foundation | Reputation is computed from recorded evidence. |
| 96 | `swarm_stigmergy_task_routing` | foundation | Routing is a deterministic in-process model. |
| 97 | `swarm_canary_rollout_verify` | foundation | Promotion requires a clean canary error tally. |
| 98 | `swarm_epidemic_rollback_kill` | modelled | Kill-switch tokens are verified in-process; no cryptographic multicast is used. |
| 99 | `swarm_emergent_topology_heal` | foundation | Up to 49% loss is supported; larger loss is rejected. |
| 100 | `swarm_global_invariant_audit` | foundation | Drift produces structured E008 audit records. |

## Tests

```sh
.venv/bin/python -m pytest -q tests/test_capabilities.py tests/test_pillar_feature_gates.py
.venv/bin/ruff check runtime/ tests/test_capabilities.py
```
