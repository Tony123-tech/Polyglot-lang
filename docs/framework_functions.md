# Framework functions (100)

The `framework_functions` package is the single registry for the 100
high-priority functions. Each function has a stable number, a unique name, and
a canonical implementation module. Look one up with
`framework_functions.get_function(4)` or `get_function("lex_source_with_spans")`,
and call it with `framework_functions.invoke(4, ...)`.
`validate_registry()` asserts that all 100 entries resolve to callables.

## Frame 1 — Lexing, Parsing & Source Spans (`compiler/frontend`)

1. `lex_source_with_spans`
2. `lex_raw_and_interpolated_strings`
3. `lex_recover_unterminated_token`
4. `parse_pratt_expr_with_precedence`
5. `parse_generic_type_params`
6. `parse_pattern_match_arms`
7. `parse_macro_invocations`
8. `parse_error_sync_recovery`
9. `ast_node_span_normalize`
10. `ast_pretty_print_canonical`

## Frame 2 — Type Checking, Traits & Monomorphization (`compiler/types`)

11. `typecheck_infer_expressions`
12. `typecheck_enforce_trait_bounds`
13. `typecheck_detect_occurs_check_cycle`
14. `monomorph_cache_placeholder_register`
15. `monomorph_resolve_recursive_enum`
16. `monomorph_resolve_recursive_struct`
17. `monomorph_unify_cache_keys`
18. `monomorph_propagate_vec_elem_type`
19. `monomorph_prune_dead_specializations`
20. `monomorph_emit_inflation_report`

## Frame 3 — Lowering, IR & Optimization Passes (`compiler/ir`)

21. `ir_lower_ast_to_hir`
22. `ir_lower_hir_to_mir`
23. `ir_fold_constant_expressions`
24. `ir_eliminate_dead_blocks`
25. `ir_validate_control_flow_graph`
26. `ir_check_borrow_and_move_state`
27. `ir_insert_raii_drop_points`
28. `ir_inline_small_pure_functions`
29. `ir_hoist_loop_invariant_checks`
30. `ir_verify_dual_backend_invariants`

## Frame 4 — Python Backend & CPython Interop (`backends/python`)

31. `py_emit_module_header`
32. `py_emit_struct_and_enum_classes`
33. `py_emit_trait_dispatch_adapters`
34. `py_emit_match_lowering`
35. `py_emit_macro_expansions`
36. `py_bind_native_extension_module`
37. `py_marshal_result_and_errors`
38. `py_guard_frame_and_trace_state`
39. `py_format_with_ruff_compatible_style`
40. `py_run_parity_harness`

## Frame 5 — Rust / PyO3 Native Backend (`backends/rust`)

41. `rs_generate_pyo3_scaffold`
42. `rs_verify_scaffold_matches_generator`
43. `rs_emit_monomorphized_structs`
44. `rs_emit_trait_impls_and_bounds`
45. `rs_map_python_args_to_rust`
46. `rs_map_rust_results_to_py`
47. `rs_raise_typed_pillar_error`
48. `rs_manage_gil_and_py_refs`
49. `rs_emit_raii_guards`
50. `rs_run_cargo_check_and_clippy`

## Frame 6 — Hot-Patching & Live Runtime State (`runtime/hotpatch`)

51. `hotpatch_registry_lookup_capability`
52. `hotpatch_validate_target_abi`
53. `hotpatch_guard_raw_address_toctou`
54. `hotpatch_apply_patch_transaction`
55. `hotpatch_rollback_transaction`
56. `hotpatch_update_frame_eval_hook`
57. `hotpatch_migrate_reactor_registrations`
58. `hotpatch_advance_rcu_epoch`
59. `hotpatch_delegate_legacy_backend`
60. `hotpatch_emit_diagnostic_e008`

## Frame 7 — Concurrency, Async & Resource Lifecycles (`runtime/concurrency`)

61. `conc_spawn_scoped_task`
62. `conc_acquire_lock_ordered`
63. `conc_release_guard_on_drop`
64. `conc_channel_bounded_backpressure`
65. `conc_cancel_token_propagate`
66. `conc_timeout_future_cleanly`
67. `conc_reactor_poll_wakeup`
68. `conc_worker_pool_graceful_drain`
69. `conc_detect_poisoned_state`
70. `conc_audit_open_handles`

## Frame 8 — Diagnostics, Error Reporting & Telemetry (`diagnostics`)

71. `diag_create_diagnostic_context`
72. `diag_create_pillar_error`
73. `diag_map_error_code_e001_to_e008`
74. `diag_format_terminal_snippet`
75. `diag_format_json_payload`
76. `diag_aggregate_without_truncating_root_cause`
77. `diag_redact_sensitive_paths`
78. `diag_record_compile_phase_timing`
79. `diag_record_cache_hit_metrics`
80. `diag_export_benchmark_summary`

## Frame 9 — Build Cache, Isolation & CI Gates (`tools/build_ci`)

81. `ci_compute_cache_scope_key`
82. `ci_isolate_cargo_target_dir`
83. `ci_isolate_python_venv_paths`
84. `ci_prune_stale_cache_entries`
85. `ci_verify_ruff_format_and_lint`
86. `ci_verify_pytest_parity_suites`
87. `ci_verify_rust_checks`
88. `ci_verify_docs_sync_and_lint`
89. `ci_verify_clean_git_worktree`
90. `ci_generate_release_gate_report`

## Frame 10 — Documentation, Examples & Regression Harness (`docs`, `regression`)

91. `docs_sync_capability_matrix`
92. `docs_sync_cli_and_backend_flags`
93. `docs_record_platform_constraints`
94. `test_recursive_generic_structs`
95. `test_recursive_generic_enums`
96. `test_trait_bounds_and_impls`
97. `test_hotpatch_delegation_and_rollback`
98. `test_typed_pillar_error_parity`
99. `test_scaffold_drift_detection`
100. `test_isolated_cache_no_contamination`

## Naming notes

- The request spelled function 4 `parse_ Pratt_expr_with_precedence` with an
  embedded space; the canonical name removes it.
- The request's Frame 8 heading says 71–80 but enumerates only 72–80. Number 71
  is filled by `diag_create_diagnostic_context`, a real context constructor for
  downstream diagnostics, so all 100 slots are occupied without renaming any
  supplied function.
- Functions 94–100 resolve to `regression.harness`, which is shipped in wheels;
  `tests/regression_harness.py` is a compatibility re-export so in-repo tests keep
  their original import path. This keeps `validate_registry()` working in a
  fresh install, where `tests/` is intentionally absent.

## 1,000-function rollout manifest

The registry above is the original 100-function surface. The ten-batch
expansion to 1,000 functions is tracked separately by
`compiler.framework_manifest`, which exposes one entry per zero-padded id
`0001`–`1000`. Every entry records its batch, canonical name, implementation
module, status, and focused test module, so progress is machine-verifiable:

```python
from compiler import framework_manifest as manifest

manifest.validate()  # asserts a complete, gap-free, uniquely named rollout
manifest.coverage_summary()
# {'total': 1000, 'implemented': 1000, 'pending': 0,
#  'with_test_coverage': 1000, 'batches_completed': 10}
manifest.implemented_count()  # 1000
manifest.get("0001").test_module  # 'tests.test_framework_f1'
```

All ten batches are implemented. Batch 1 (`0001`–`0100`, Frontend Lexing,
Parsing, Spans & AST) lives in `compiler.frontend`; ids `0001`–`0010`
intentionally reuse the names above. Each of batches 2–10 reuses the ten legacy
FRAME functions of its area and adds ninety new helpers in disjoint modules.
The row tables are owned by the per-area `rollout` modules (for example
`compiler.types.rollout`) and imported by the manifest, so ids, callables, and
test modules cannot drift. Batch 6 uses the real top-level `zig` package; there
is no `backends.zig` package. Frontend failures use `PolyglotError` from
`compiler.frontend.errors`, a specialization of `PillarError` on the shared
`E001`–`E008` surface.

## Production frames (JIT, WASM, and hard compiler/runtime)

The 1,000-function manifest above tracks the original registry. Three later
frames add production-grade implementations that compose with it rather than
duplicating it. They are documented here for discoverability; their canonical
sources are the named modules.

### Frame A — speculative JIT (`runtime.jit`)

| Function | Contract |
| --- | --- |
| `jit_compile_hot_trace` | Per-key hotness counting; compiles at the threshold via `backends.jit` (or records a model trace); rejects shape changes with `E008`. |
| `jit_emit_type_and_bounds_guards` | Evaluates type / non-null / bounds guards against concrete values and returns a `GuardReport`. |
| `jit_deoptimize_to_interpreter_frame` | Bails out to the interpreter with live values, stack, and span preserved; guard exhaustion raises `E008`. |
| `jit_manage_executable_memory_rx` | Owns `MAP_JIT` blocks (or models them), enforces W^X immutability, and refuses to reclaim pinned blocks. |
| `jit_patch_ic_and_call_sites` | Copy-on-write, version-checked inline-cache/call-site publication; defers patches that would mutate live code. |
| `jit_invalidate_on_redefinition_or_hotpatch` | Advances version/epoch, invalidates traces, clears sites, and reclaims or defers code blocks. |
| `jit_emit_diagnostic_e008` | Structured, span-carrying `E008` diagnostic factory. |

Privileged native codegen and `MAP_JIT` allocation require `allow_native` /
`opt_in_executable_memory` and a supported host; otherwise all seven functions
operate in model mode. See SPEC 13.1.

### Frame B — WASM production frame (`backends.wasm.framework`)

| Function | Contract |
| --- | --- |
| `wasm_lower_ir_to_module` | Lowers source/AST/HIR/MIR to WAT and `.wasm`; unsupported nodes raise `E003`, missing runtime `E007`. |
| `wasm_layout_linear_memory_and_stack` | Aligned bump layout plus a 16-byte-aligned shadow stack; enforces the wasm32 4 GiB ceiling (`E008`). |
| `wasm_marshal_adt_and_result_abi` | Round-trip encoding of struct / tagged `Result` / string / slice values into linear-memory bytes. |
| `wasm_bind_host_imports_and_exports` | Validated host-import/export signatures; delegates exports to real `wasmtime` host functions. |
| `wasm_guard_trap_and_oom_paths` | Structured bounds/stack/OOM checks; `E008` in strict mode. |
| `wasm_isolate_instance_state` | Fresh `Engine`/`Store`/`Memory` per run with caller-scoped module caching. |
| `wasm_verify_dual_backend_parity` | Normalized Python/Rust-PyO3/WASM outcome comparison with explicit boundary skips. |

Before WAT emission, `backends.wasm.validation.wasm_preflight_validate` reuses
the shared typed errors so `E001`–`E008` codes and messages match the Python and
native backends, and `backends.wasm.aggregates.lower_wasm_aggregates`
monomorphizes generics and specializes compile-time structs, vectors, enums and
bounded methods into scalar expressions. Scalar bindings remain real WASM
locals; a dynamic aggregate ABI is an explicit `E003` boundary. See SPEC 13.2.

### Frame C — hard compiler/runtime frames

| Module | Hard frames |
| --- | --- |
| `compiler.types.hard` | Bounded mutually recursive generic expansion with alias unification; trait coherence/projection/HRTB checks. |
| `compiler.ir.hard` | Dominance and loop-header validation; branch-aware move tracking; deterministic RAII cleanup. |
| `runtime.hotpatch.hard` | RCU reader/writer invariants, descriptor migration, raw-address TOCTOU re-validation, verified atomic rollback. |
| `tools.build_ci.hard` | Isolated per-backend build/cache environments, seed-reproducible matrix, divergence-only artifacts. |

These frames compose with `tools.three_way_parity`, which runs a fixed
seed-reproducible Python/Rust-PyO3/WASM matrix under isolated caches and writes
artifacts only for genuine divergence. All 12 cases now run on all three
backends for seeds `1`, `42`, `9999`, and `2026` with zero skips. The release
gate's `hard-frames-parity` stage (between `pytest-core` and `diff-fuzz`) runs
the JIT/WASM/hard-frame suites plus those per-seed parity checks; its runner is
injectable so tests stay hermetic. See SPEC 13.3–13.4.

## Research frames (H–K)

These frames add bounded verification tooling without changing JIT or WASM
behavior. They are intentionally separate from the 100-function registry, and
none of them claims an undecidable property unconditionally: exhausted bounds,
timeouts, and `Unknown` results are reported as structured `E001`–`E008`
diagnostics. See SPEC 9.21 and 13.3.8.

| Module | Public functions | Boundary |
| --- | --- | --- |
| `compiler.formal` (H) | `formal_sym_exec_bounded_paths`, `formal_check_relational_refinement`, `formal_synthesize_infeasible_path_witness`, `formal_require_bounded_proof` | Step/unroll/path/time budgets; interval solver leaves nonlinear guards `Unknown`; witnesses minimize only recorded symbolic atoms. |
| `compiler.termination` (I) | `term_check_structural_and_lexicographic_variants`, `term_infer_loop_bound_certificates`, `term_enforce_resource_fuel_contracts` | Variants/subterm fields are caller-declared; uncounted loops are `UnboundedOrUnknown`; fuel is charged through the replayable sandbox meter. |
| `runtime.hard_state` (J) | `hard_analyze_points_to_and_alias_sets`, `hard_plan_live_state_schema_migration`, `hard_quarantine_and_replay_faults` | Points-to sets are declared over-approximations; only total, unambiguous migrations are safe; `MAP_JIT`/raw FFI/PEP 667 frames are hazards or explicit boundaries. |
| `tools.adversarial` (K) | `adv_mutate_ast_and_type_constraints`, `adv_run_mutation_score_gate`, `adv_verify_spec_oracle_and_report` | Generation is deterministic and grammar-bounded, not exhaustive; mutation score measures test sensitivity; artifacts live only under caller-supplied roots. |

The release gate's `research-frames` stage (between `hard-frames-parity` and
`diff-fuzz`) runs the four focused suites and an injectable in-process
spec-oracle check. The default oracle evaluates the real frame functions and
records bounded caveats; no native, JIT, or WASM code is spawned.
