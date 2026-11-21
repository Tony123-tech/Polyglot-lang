"""Machine-verifiable manifest for the 1,000-function framework rollout.

The rollout is delivered in ten controlled batches of 100 functions.  This
module is the single source of truth for progress toward ``1000/1000``: every
stable zero-padded id records its owning batch, canonical function name,
implementation module, status, and the focused test module that covers it.

All ten batches are implemented.  Batch 1 (ids ``0001``–``0100``) lives under
:mod:`compiler.frontend`; each of batches 2–10 reuses the ten legacy FRAME
functions for its area and adds ninety focused helpers in disjoint modules.
The row tables are owned by the corresponding ``rollout`` module (for example
:mod:`compiler.types.rollout`) and imported here, so the manifest cannot drift
from the delivered implementations.  The Zig batch uses the real top-level
:mod:`zig` package; there is no ``backends.zig`` package.

The legacy :mod:`framework_functions` registry remains the public API for the
original 100 high-priority functions; ids ``0001``–``0010`` deliberately reuse
its names, but the two registries are otherwise independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module

from backends.python.rollout import FRAMEWORK_ROWS as PYTHON_ROWS
from backends.rust.rollout import FRAMEWORK_ROWS as RUST_ROWS
from compiler.ir.rollout import FRAMEWORK_ROWS as IR_ROWS
from compiler.types.rollout import FRAMEWORK_ROWS as TYPES_ROWS
from diagnostics.rollout import FRAMEWORK_ROWS as DIAGNOSTICS_ROWS
from runtime.concurrency.rollout import FRAMEWORK_ROWS as CONCURRENCY_ROWS
from runtime.hotpatch.rollout import FRAMEWORK_ROWS as HOTPATCH_ROWS
from tools.build_ci.rollout import FRAMEWORK_ROWS as BUILD_CI_ROWS
from zig.rollout import FRAMEWORK_ROWS as ZIG_ROWS

BATCH_TITLES: dict[int, str] = {
    1: "Frontend Lexing, Parsing, Spans & AST",
    2: "Type Inference, Traits, Lifetimes & Monomorphization",
    3: "HIR/MIR Lowering, IR Passes & Optimizations",
    4: "Python Backend & CPython Interop",
    5: "Rust / PyO3 Native Backend",
    6: "Zig Backend & Idiomatic Codegen",
    7: "Hot-Patching & Live Runtime State",
    8: "Concurrency, Async, IO & Resource Lifecycles",
    9: "Diagnostics, Telemetry, Terminal & Visualizer",
    10: "Build Caches, Packaging, CI Gates & Parity Tests",
}

# (function name, implementation module, category, focused test module)
_BATCH1_ROWS: tuple[tuple[str, str, str, str], ...] = (
    # 0001-0010: existing FRAME 1 surface.
    (
        "lex_source_with_spans",
        "compiler.frontend.framework",
        "lexing",
        "tests.test_framework_f1",
    ),
    (
        "lex_raw_and_interpolated_strings",
        "compiler.frontend.framework",
        "lexing",
        "tests.test_framework_f1",
    ),
    (
        "lex_recover_unterminated_token",
        "compiler.frontend.framework",
        "lexing",
        "tests.test_framework_f1",
    ),
    (
        "parse_pratt_expr_with_precedence",
        "compiler.frontend.framework",
        "parsing",
        "tests.test_framework_f1",
    ),
    (
        "parse_generic_type_params",
        "compiler.frontend.framework",
        "parsing",
        "tests.test_framework_f1",
    ),
    (
        "parse_pattern_match_arms",
        "compiler.frontend.framework",
        "parsing",
        "tests.test_framework_f1",
    ),
    (
        "parse_macro_invocations",
        "compiler.frontend.framework",
        "parsing",
        "tests.test_framework_f1",
    ),
    (
        "parse_error_sync_recovery",
        "compiler.frontend.framework",
        "parsing",
        "tests.test_framework_f1",
    ),
    (
        "ast_node_span_normalize",
        "compiler.frontend.framework",
        "spans",
        "tests.test_framework_f1",
    ),
    (
        "ast_pretty_print_canonical",
        "compiler.frontend.framework",
        "ast",
        "tests.test_framework_f1",
    ),
    # 0011-0035: lexing.
    (
        "lex_tokenize",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_count_by_type",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_keywords",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_identifiers",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_literals",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_operators",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_validate_identifier",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_decode_numeric",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_decode_string_escapes",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_detect_bom",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_normalize_newlines",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_split_lines",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_scan_comments",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_balance_delimiters",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_indent_levels",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_source_fingerprint",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_token_window",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_find_token",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_token_distance",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_validate_source",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_keyword_boundary",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_char_class",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_escape_table",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_measure_token_widths",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    (
        "lex_tokenize_with_offsets",
        "compiler.frontend.lexing",
        "lexing",
        "tests.test_frontend_lexing",
    ),
    # 0036-0070: parsing.
    (
        "parse_source_program",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_items",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_function_signature",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_struct_fields",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_enum_variants",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_trait_methods",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_impl_methods",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_type_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_generic_args",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_where_predicates",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_attributes",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_visibility",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_let_binding",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_match_arms_simple",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_if_chain",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_loop_body",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_call_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_path_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_range_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_literal_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_unary_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_binary_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_block_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_lambda_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_return_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_tuple_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_array_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_index_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_field_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_try_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_cast_expr",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_pattern_simple",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_arm_guard_simple",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_statement_kind",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    (
        "parse_expression_depth",
        "compiler.frontend.parsing",
        "parsing",
        "tests.test_frontend_parsing",
    ),
    # 0071-0090: spans.
    (
        "span_offset_to_line_col",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_line_col_to_offset",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_char_to_byte",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_byte_to_char",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    ("span_make", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_union", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_intersect", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_contains", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_overlaps", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_compare", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_shift", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_clamp", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    (
        "span_merge_adjacent",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_from_token",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_from_ast_node",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    (
        "span_line_range",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    ("span_text", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    (
        "span_utf8_validate",
        "compiler.frontend.spans",
        "spans",
        "tests.test_frontend_spans",
    ),
    ("span_format", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    ("span_validate", "compiler.frontend.spans", "spans", "tests.test_frontend_spans"),
    # 0091-0100: AST utilities.
    ("ast_walk", "compiler.frontend.ast_utils", "ast", "tests.test_frontend_ast_utils"),
    (
        "ast_node_count",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_depth",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_find_by_type",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_find_by_predicate",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_to_json",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_node_kinds",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_collect_symbols",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_collect_literals",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
    (
        "ast_node_at_offset",
        "compiler.frontend.ast_utils",
        "ast",
        "tests.test_frontend_ast_utils",
    ),
)

# Each rollout module owns exactly 100 rows in id order.  Importing them here
# keeps the manifest authoritative without duplicating nine row tables.
_BATCH_ROWS: dict[int, tuple[tuple[str, str, str, str], ...]] = {
    1: _BATCH1_ROWS,
    2: TYPES_ROWS,
    3: IR_ROWS,
    4: PYTHON_ROWS,
    5: RUST_ROWS,
    6: ZIG_ROWS,
    7: HOTPATCH_ROWS,
    8: CONCURRENCY_ROWS,
    9: DIAGNOSTICS_ROWS,
    10: BUILD_CI_ROWS,
}


@dataclass(frozen=True)
class ManifestEntry:
    """One stable id in the 1,000-function rollout."""

    id: str
    number: int
    batch: int
    name: str | None
    module: str
    category: str
    status: str
    test_module: str | None

    @property
    def is_implemented(self) -> bool:
        return self.status == "implemented"

    @property
    def callable(self):
        """Resolve the implementation; pending ids raise a clear error."""
        if not self.is_implemented or self.name is None:
            raise ValueError(
                f"function {self.id} is {self.status} and has no callable yet"
            )
        return getattr(import_module(self.module), self.name)


def _build_entries() -> tuple[ManifestEntry, ...]:
    entries: list[ManifestEntry] = []
    for batch in range(1, 11):
        rows = _BATCH_ROWS[batch]
        if len(rows) != 100:
            raise ValueError(f"batch {batch} declares {len(rows)} rows, expected 100")
        for offset, (name, module, category, test_module) in enumerate(rows, 1):
            number = (batch - 1) * 100 + offset
            entries.append(
                ManifestEntry(
                    id=f"{number:04d}",
                    number=number,
                    batch=batch,
                    name=name,
                    module=module,
                    category=category,
                    status="implemented",
                    test_module=test_module,
                )
            )
    return tuple(entries)


FUNCTIONS: tuple[ManifestEntry, ...] = _build_entries()
_BY_ID = {entry.id: entry for entry in FUNCTIONS}
_BY_NUMBER = {entry.number: entry for entry in FUNCTIONS}


def get(entry_id: str | int) -> ManifestEntry:
    """Resolve an entry by zero-padded id (``"0007"``) or integer (``7``)."""
    key = f"{entry_id:04d}" if isinstance(entry_id, int) else entry_id
    try:
        return _BY_ID[key]
    except KeyError as error:
        raise KeyError(f"no framework manifest id {entry_id!r}") from error


def get_by_number(number: int) -> ManifestEntry:
    """Resolve an entry by its 1-based number."""
    try:
        return _BY_NUMBER[number]
    except KeyError as error:
        raise KeyError(f"no framework manifest number {number}") from error


def iter_batch(batch: int) -> Iterator[ManifestEntry]:
    """Yield the 100 entries belonging to *batch* in id order."""
    for entry in FUNCTIONS:
        if entry.batch == batch:
            yield entry


def implemented_count(*, batch: int | None = None) -> int:
    """Return the number of implemented entries, optionally for one batch."""
    return sum(
        entry.is_implemented
        for entry in FUNCTIONS
        if batch is None or entry.batch == batch
    )


def coverage_summary() -> dict[str, int]:
    """Return machine-readable counts for progress reporting."""
    implemented = [entry for entry in FUNCTIONS if entry.is_implemented]
    covered = [entry for entry in implemented if entry.test_module is not None]
    return {
        "total": len(FUNCTIONS),
        "implemented": len(implemented),
        "pending": len(FUNCTIONS) - len(implemented),
        "with_test_coverage": len(covered),
        "batches_completed": len(
            {entry.batch for entry in implemented if entry.is_implemented}
        ),
    }


def validate() -> None:
    """Assert the manifest is a complete, gap-free, uniquely named rollout."""
    assert len(FUNCTIONS) == 1000, "manifest must contain exactly 1000 ids"
    assert [entry.number for entry in FUNCTIONS] == list(range(1, 1001))
    assert [entry.id for entry in FUNCTIONS] == [f"{n:04d}" for n in range(1, 1001)]
    assert all(entry.is_implemented for entry in FUNCTIONS), (
        "the rollout is complete; no id may remain pending"
    )
    implemented_names = [entry.name for entry in FUNCTIONS if entry.is_implemented]
    assert len(implemented_names) == len(set(implemented_names)), (
        "implemented function names must be unique"
    )
    for entry in FUNCTIONS:
        assert entry.batch == (entry.number - 1) // 100 + 1
        if entry.is_implemented:
            assert entry.name and entry.test_module
            assert callable(entry.callable), f"{entry.id}: {entry.name} not callable"


def next_batch() -> int | None:
    """Return the first batch that is not fully implemented, or ``None``."""
    for batch in range(1, 11):
        if implemented_count(batch=batch) < 100:
            return batch
    return None


__all__ = [
    "BATCH_TITLES",
    "FUNCTIONS",
    "ManifestEntry",
    "coverage_summary",
    "get",
    "get_by_number",
    "implemented_count",
    "iter_batch",
    "next_batch",
    "validate",
]
