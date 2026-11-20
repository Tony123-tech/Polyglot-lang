"""Unified registry for the 100 high-priority framework functions.

The registry is intentionally thin: it maps a stable 1-based number and name to
the canonical implementation module.  Each frame module owns its own logic and
tests; this package only guarantees discoverability and a uniform invocation
surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

FRAME_TITLES: dict[int, str] = {
    1: "Lexing, Parsing & Source Spans",
    2: "Type Checking, Traits & Monomorphization",
    3: "Lowering, IR & Optimization Passes",
    4: "Python Backend & CPython Interop",
    5: "Rust / PyO3 Native Backend",
    6: "Hot-Patching & Live Runtime State",
    7: "Concurrency, Async & Resource Lifecycles",
    8: "Diagnostics, Error Reporting & Telemetry",
    9: "Build Cache, Isolation & CI Gates",
    10: "Documentation, Examples & Regression Harness",
}

# (number, function name, frame, module path)
_ROWS: tuple[tuple[int, str, int, str], ...] = (
    (1, "lex_source_with_spans", 1, "compiler.frontend.framework"),
    (2, "lex_raw_and_interpolated_strings", 1, "compiler.frontend.framework"),
    (3, "lex_recover_unterminated_token", 1, "compiler.frontend.framework"),
    (4, "parse_pratt_expr_with_precedence", 1, "compiler.frontend.framework"),
    (5, "parse_generic_type_params", 1, "compiler.frontend.framework"),
    (6, "parse_pattern_match_arms", 1, "compiler.frontend.framework"),
    (7, "parse_macro_invocations", 1, "compiler.frontend.framework"),
    (8, "parse_error_sync_recovery", 1, "compiler.frontend.framework"),
    (9, "ast_node_span_normalize", 1, "compiler.frontend.framework"),
    (10, "ast_pretty_print_canonical", 1, "compiler.frontend.framework"),
    (11, "typecheck_infer_expressions", 2, "compiler.types.framework"),
    (12, "typecheck_enforce_trait_bounds", 2, "compiler.types.framework"),
    (13, "typecheck_detect_occurs_check_cycle", 2, "compiler.types.framework"),
    (14, "monomorph_cache_placeholder_register", 2, "compiler.types.framework"),
    (15, "monomorph_resolve_recursive_enum", 2, "compiler.types.framework"),
    (16, "monomorph_resolve_recursive_struct", 2, "compiler.types.framework"),
    (17, "monomorph_unify_cache_keys", 2, "compiler.types.framework"),
    (18, "monomorph_propagate_vec_elem_type", 2, "compiler.types.framework"),
    (19, "monomorph_prune_dead_specializations", 2, "compiler.types.framework"),
    (20, "monomorph_emit_inflation_report", 2, "compiler.types.framework"),
    (21, "ir_lower_ast_to_hir", 3, "compiler.ir.framework"),
    (22, "ir_lower_hir_to_mir", 3, "compiler.ir.framework"),
    (23, "ir_fold_constant_expressions", 3, "compiler.ir.framework"),
    (24, "ir_eliminate_dead_blocks", 3, "compiler.ir.framework"),
    (25, "ir_validate_control_flow_graph", 3, "compiler.ir.framework"),
    (26, "ir_check_borrow_and_move_state", 3, "compiler.ir.framework"),
    (27, "ir_insert_raii_drop_points", 3, "compiler.ir.framework"),
    (28, "ir_inline_small_pure_functions", 3, "compiler.ir.framework"),
    (29, "ir_hoist_loop_invariant_checks", 3, "compiler.ir.framework"),
    (30, "ir_verify_dual_backend_invariants", 3, "compiler.ir.framework"),
    (31, "py_emit_module_header", 4, "backends.python.framework"),
    (32, "py_emit_struct_and_enum_classes", 4, "backends.python.framework"),
    (33, "py_emit_trait_dispatch_adapters", 4, "backends.python.framework"),
    (34, "py_emit_match_lowering", 4, "backends.python.framework"),
    (35, "py_emit_macro_expansions", 4, "backends.python.framework"),
    (36, "py_bind_native_extension_module", 4, "backends.python.framework"),
    (37, "py_marshal_result_and_errors", 4, "backends.python.framework"),
    (38, "py_guard_frame_and_trace_state", 4, "backends.python.framework"),
    (39, "py_format_with_ruff_compatible_style", 4, "backends.python.framework"),
    (40, "py_run_parity_harness", 4, "backends.python.framework"),
    (41, "rs_generate_pyo3_scaffold", 5, "backends.rust.framework"),
    (42, "rs_verify_scaffold_matches_generator", 5, "backends.rust.framework"),
    (43, "rs_emit_monomorphized_structs", 5, "backends.rust.framework"),
    (44, "rs_emit_trait_impls_and_bounds", 5, "backends.rust.framework"),
    (45, "rs_map_python_args_to_rust", 5, "backends.rust.framework"),
    (46, "rs_map_rust_results_to_py", 5, "backends.rust.framework"),
    (47, "rs_raise_typed_pillar_error", 5, "backends.rust.framework"),
    (48, "rs_manage_gil_and_py_refs", 5, "backends.rust.framework"),
    (49, "rs_emit_raii_guards", 5, "backends.rust.framework"),
    (50, "rs_run_cargo_check_and_clippy", 5, "backends.rust.framework"),
    (51, "hotpatch_registry_lookup_capability", 6, "runtime.hotpatch.framework"),
    (52, "hotpatch_validate_target_abi", 6, "runtime.hotpatch.framework"),
    (53, "hotpatch_guard_raw_address_toctou", 6, "runtime.hotpatch.framework"),
    (54, "hotpatch_apply_patch_transaction", 6, "runtime.hotpatch.framework"),
    (55, "hotpatch_rollback_transaction", 6, "runtime.hotpatch.framework"),
    (56, "hotpatch_update_frame_eval_hook", 6, "runtime.hotpatch.framework"),
    (57, "hotpatch_migrate_reactor_registrations", 6, "runtime.hotpatch.framework"),
    (58, "hotpatch_advance_rcu_epoch", 6, "runtime.hotpatch.framework"),
    (59, "hotpatch_delegate_legacy_backend", 6, "runtime.hotpatch.framework"),
    (60, "hotpatch_emit_diagnostic_e008", 6, "runtime.hotpatch.framework"),
    (61, "conc_spawn_scoped_task", 7, "runtime.concurrency.framework"),
    (62, "conc_acquire_lock_ordered", 7, "runtime.concurrency.framework"),
    (63, "conc_release_guard_on_drop", 7, "runtime.concurrency.framework"),
    (64, "conc_channel_bounded_backpressure", 7, "runtime.concurrency.framework"),
    (65, "conc_cancel_token_propagate", 7, "runtime.concurrency.framework"),
    (66, "conc_timeout_future_cleanly", 7, "runtime.concurrency.framework"),
    (67, "conc_reactor_poll_wakeup", 7, "runtime.concurrency.framework"),
    (68, "conc_worker_pool_graceful_drain", 7, "runtime.concurrency.framework"),
    (69, "conc_detect_poisoned_state", 7, "runtime.concurrency.framework"),
    (70, "conc_audit_open_handles", 7, "runtime.concurrency.framework"),
    (71, "diag_create_diagnostic_context", 8, "diagnostics.framework"),
    (72, "diag_create_pillar_error", 8, "diagnostics.framework"),
    (73, "diag_map_error_code_e001_to_e008", 8, "diagnostics.framework"),
    (74, "diag_format_terminal_snippet", 8, "diagnostics.framework"),
    (75, "diag_format_json_payload", 8, "diagnostics.framework"),
    (76, "diag_aggregate_without_truncating_root_cause", 8, "diagnostics.framework"),
    (77, "diag_redact_sensitive_paths", 8, "diagnostics.framework"),
    (78, "diag_record_compile_phase_timing", 8, "diagnostics.framework"),
    (79, "diag_record_cache_hit_metrics", 8, "diagnostics.framework"),
    (80, "diag_export_benchmark_summary", 8, "diagnostics.framework"),
    (81, "ci_compute_cache_scope_key", 9, "tools.build_ci.framework"),
    (82, "ci_isolate_cargo_target_dir", 9, "tools.build_ci.framework"),
    (83, "ci_isolate_python_venv_paths", 9, "tools.build_ci.framework"),
    (84, "ci_prune_stale_cache_entries", 9, "tools.build_ci.framework"),
    (85, "ci_verify_ruff_format_and_lint", 9, "tools.build_ci.framework"),
    (86, "ci_verify_pytest_parity_suites", 9, "tools.build_ci.framework"),
    (87, "ci_verify_rust_checks", 9, "tools.build_ci.framework"),
    (88, "ci_verify_docs_sync_and_lint", 9, "tools.build_ci.framework"),
    (89, "ci_verify_clean_git_worktree", 9, "tools.build_ci.framework"),
    (90, "ci_generate_release_gate_report", 9, "tools.build_ci.framework"),
    (91, "docs_sync_capability_matrix", 10, "docs.framework"),
    (92, "docs_sync_cli_and_backend_flags", 10, "docs.framework"),
    (93, "docs_record_platform_constraints", 10, "docs.framework"),
    (94, "test_recursive_generic_structs", 10, "regression.harness"),
    (95, "test_recursive_generic_enums", 10, "regression.harness"),
    (96, "test_trait_bounds_and_impls", 10, "regression.harness"),
    (97, "test_hotpatch_delegation_and_rollback", 10, "regression.harness"),
    (98, "test_typed_pillar_error_parity", 10, "regression.harness"),
    (99, "test_scaffold_drift_detection", 10, "regression.harness"),
    (100, "test_isolated_cache_no_contamination", 10, "regression.harness"),
)


@dataclass(frozen=True)
class FrameworkFunction:
    number: int
    name: str
    frame: int
    module: str

    @property
    def frame_title(self) -> str:
        return FRAME_TITLES[self.frame]

    @property
    def callable(self):
        return getattr(import_module(self.module), self.name)


FUNCTIONS: tuple[FrameworkFunction, ...] = tuple(
    FrameworkFunction(number, name, frame, module)
    for number, name, frame, module in _ROWS
)

_BY_NUMBER = {item.number: item for item in FUNCTIONS}
_BY_NAME = {item.name: item for item in FUNCTIONS}


def get_function(key: int | str) -> FrameworkFunction:
    """Resolve a function by its 1-based number or exact name."""
    if isinstance(key, int):
        try:
            return _BY_NUMBER[key]
        except KeyError as error:
            raise KeyError(f"no framework function numbered {key}") from error
    try:
        return _BY_NAME[key]
    except KeyError as error:
        raise KeyError(f"no framework function named {key!r}") from error


def invoke(key: int | str, *args: Any, **kwargs: Any) -> Any:
    """Resolve and call a framework function."""
    return get_function(key).callable(*args, **kwargs)


def iter_functions(frame: int | None = None) -> Iterator[FrameworkFunction]:
    """Iterate over all functions, optionally restricted to one frame."""
    for item in FUNCTIONS:
        if frame is None or item.frame == frame:
            yield item


def validate_registry() -> None:
    """Assert the registry is complete and every entry resolves to a callable."""
    numbers = [item.number for item in FUNCTIONS]
    names = [item.name for item in FUNCTIONS]
    assert numbers == list(range(1, 101)), (
        "function numbers must be 1..100 without gaps"
    )
    assert len(set(names)) == 100, "function names must be unique"
    for item in FUNCTIONS:
        assert callable(item.callable), f"{item.number}: {item.name} is not callable"


__all__ = [
    "FRAME_TITLES",
    "FUNCTIONS",
    "FrameworkFunction",
    "get_function",
    "invoke",
    "iter_functions",
    "validate_registry",
]
