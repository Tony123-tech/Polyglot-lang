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
