"""Behavioral tests for the batch-6 Zig backend framework rollout.

All functions are exercised as pure models: emitted source is never compiled,
addresses are only opaque mapping keys, and no executable memory is touched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from compiler.frontend.errors import PolyglotError
from zig import abi, codegen, framework
from zig.rollout import FRAMEWORK_ROWS

_MODULES: dict[str, Any] = {
    "zig.abi": abi,
    "zig.codegen": codegen,
    "zig.framework": framework,
}


def _code(function: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Return the diagnostic code raised by *function* for bad inputs."""
    with pytest.raises(PolyglotError) as error:
        function(*args, **kwargs)
    return error.value.code


# ---------------------------------------------------------------------------
# Rollout metadata
# ---------------------------------------------------------------------------


def test_rollout_has_one_hundred_unique_ordered_rows():
    assert len(FRAMEWORK_ROWS) == 100
    assert [row[0] for row in FRAMEWORK_ROWS] == list(
        dict.fromkeys(row[0] for row in FRAMEWORK_ROWS)
    )
    assert all(row[3] == "tests.test_zig_rollout" for row in FRAMEWORK_ROWS)
    assert {row[1] for row in FRAMEWORK_ROWS} == {
        "zig.framework",
        "zig.codegen",
        "zig.abi",
    }
    assert {row[2] for row in FRAMEWORK_ROWS} == {"framework", "codegen", "abi"}


def test_all_rollout_names_resolve_to_documented_callables():
    for name, module_name, _category, _test_module in FRAMEWORK_ROWS:
        function = getattr(_MODULES[module_name], name)
        assert callable(function), name
        assert function.__doc__, name
        assert function.__name__ == name


# ---------------------------------------------------------------------------
# zig.framework (40 functions)
# ---------------------------------------------------------------------------


def test_zf_build_profile_models_debug_flags():
    result = framework.zf_build_profile("debug")
    assert result["flags"] == ("-ODebug", "-g")
    assert result["debug_symbols"] is True
    assert result["model_only"] is True


def test_zf_cache_entry_hashes_source_bytes_and_artifacts():
    result = framework.zf_cache_entry("k", source="aé", artifacts=("x",))
    assert result["key"] == "k"
    assert result["byte_length"] == 3
    assert result["artifacts"] == ("x",)
    assert len(result["source_hash"]) == 16


def test_zf_cache_evict_removes_key_and_trims_oldest_entries():
    cache: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    result = framework.zf_cache_evict(cache, "b", max_entries=1)
    assert result["removed"] is True
    assert result["evicted"] == ("a",)
    assert result["remaining"] == ("c",)
    assert cache == {"c": 3}


def test_zf_cache_key_is_stable_and_profile_sensitive():
    first = framework.zf_cache_key("mod", source="x")
    again = framework.zf_cache_key("mod", source="x")
    other = framework.zf_cache_key("mod", source="x", profile="debug")
    assert first == again
    assert first["key"] != other["key"]
    assert first["length"] == 16


def test_zf_cache_lookup_reports_hit_and_checked_count():
    assert framework.zf_cache_lookup({"k": 7}, "k") == {
        "key": "k",
        "hit": True,
        "entry": 7,
        "checked": 1,
    }
    assert framework.zf_cache_lookup({}, "k")["hit"] is False


def test_zf_dependency_graph_normalizes_edges_and_nodes():
    result = framework.zf_dependency_graph([("base", "mid"), ("mid", "top")])
    assert result["nodes"] == ("base", "mid", "top")
    assert result["edges"] == (("base", "mid"), ("mid", "top"))
    assert result["indegree"]["mid"] == 1
    assert result["adjacency"]["base"] == ("mid",)


def test_zf_dependency_order_uses_stable_topological_levels():
    result = framework.zf_dependency_order(
        {"edges": [("base", "mid"), ("mid", "top"), ("base", "side")]}
    )
    assert result["order"] == ("base", "mid", "side", "top")
    assert result["levels"] == (("base",), ("mid", "side"), ("top",))
    assert result["acyclic"] is True


def test_zf_diagnostic_renders_code_and_location():
    result = framework.zf_diagnostic("E001", "bad", line=3, col=5)
    assert result["rendered"] == "error[E001]: bad"
    assert result["location"] == "3:5"
    assert result["hint"] is None


def test_zf_diagnostic_format_renders_text_with_optional_color():
    diagnostics = [{"code": "E003", "message": "wrong type"}]
    plain = framework.zf_diagnostic_format(diagnostics)
    colored = framework.zf_diagnostic_format(diagnostics, color=True)
    assert plain["text"] == "error[E003]: wrong type"
    assert plain["codes"] == ("E003",)
    assert colored["lines"] == ("\x1b[31merror[E003]\x1b[0m: wrong type",)


def test_zf_dispatch_table_deduplicates_and_detects_signature_conflicts():
    result = framework.zf_dispatch_table(
        [("f", "i64 (i64)"), ("f", "i64(i64)"), ("g", "void()")]
    )
    assert result["count"] == 2
    assert [slot["index"] for slot in result["slots"]] == [0, 1]
    assert result["by_symbol"]["f"]["signature"] == "i64(i64)"
    repeated = framework.zf_dispatch_table(
        [("f", "i64()"), ("f", "i64()")], deduplicate=False
    )
    assert repeated["collisions"] == ("f",)
    assert repeated["count"] == 2


def test_zf_emit_context_starts_with_clean_accumulators():
    context = framework.zf_emit_context("m", indent=2)
    assert context["pad"] == "  "
    assert context["lines"] == []
    assert context["helpers"] == []
    assert context["model_only"] is True


def test_zf_error_surface_classifies_e007_as_hard_failure():
    result = framework.zf_error_surface("E007", operation="simd")
    assert result["meaning"] == "unsupported platform feature"
    assert result["recoverable"] is False
    assert result["retryable"] is False
    assert framework.zf_error_surface("E004", operation="emit")["retryable"] is True


def test_zf_escape_identifier_prefixes_reserved_and_numeric_names():
    assert framework.zf_escape_identifier("type") == {
        "name": "type",
        "escaped": "_type",
        "changed": True,
        "reserved": True,
    }
    assert framework.zf_escape_identifier("9lives")["escaped"] == "_9lives"
    assert framework.zf_escape_identifier("ok")["changed"] is False


def test_zf_escape_string_renders_zig_escapes_and_byte_length():
    result = framework.zf_escape_string('a"b\n')
    assert result["literal"] == '"a\\"b\\n"'
    assert result["escapes"] == (('"', '\\"'), ("\n", "\\n"))
    assert result["byte_length"] == 4


def test_zf_feature_gate_disables_debug_parallel_and_rejects_wasm_simd():
    enabled = framework.zf_feature_gate("simd", target="native", profile="release_fast")
    disabled = framework.zf_feature_gate("parallel")
    assert enabled["enabled"] is True
    assert enabled["reason"] is None
    assert disabled["enabled"] is False
    assert "debug" in disabled["reason"]
    assert _code(framework.zf_feature_gate, "simd", target="wasm32") == "E007"


def test_zf_int_bounds_returns_signed_two_complement_extremes():
    assert framework.zf_int_bounds("i8") == {
        "type": "i8",
        "min": -128,
        "max": 127,
        "bits": 8,
        "signed": True,
    }
    assert framework.zf_int_bounds("u3")["max"] == 7


def test_zf_integer_widths_scales_byte_sizes_with_word():
    result = framework.zf_integer_widths(word=4)
    assert result["word"] == 4
    assert result["byte_sizes"] == (1, 1, 1, 1, 2, 4, 8, 16)
    assert result["max_lanes"] == 64


def test_zf_layout_align_rounds_arbitrary_integer_widths():
    result = framework.zf_layout_align("u24")
    assert result == {"type": "u24", "align": 4, "size": 3, "kind": "int"}


def test_zf_layout_offsets_inserts_padding_between_fields():
    result = framework.zf_layout_offsets([("a", "u8"), ("b", "i32")])
    assert [row["offset"] for row in result["fields"]] == [0, 4]
    assert result["size"] == 8
    assert result["align"] == 4
    assert result["padding"] == 3


def test_zf_layout_size_models_wide_integers():
    assert framework.zf_layout_size("u128") == {
        "type": "u128",
        "size": 16,
        "align": 16,
        "bits": 128,
    }


def test_zf_layout_validate_accepts_matching_size_and_alignment():
    result = framework.zf_layout_validate(
        {
            "name": "Pair",
            "fields": [("a", "u8"), ("b", "i32")],
            "size": 8,
            "align": 4,
        }
    )
    assert result["valid"] is True
    assert result["padding"] == 3
    assert result["offsets"][1]["offset"] == 4


def test_zf_module_fingerprint_sorts_sources_before_hashing():
    result = framework.zf_module_fingerprint("m", sources={"b.zig": "y", "a.zig": "x"})
    assert result["files"] == ("a.zig", "b.zig")
    assert result["byte_total"] == 2
    assert result["digest_kind"] == "sha256-16"
    assert len(result["fingerprint"]) == 16


def test_zf_module_io_collects_a_sorted_interface():
    result = framework.zf_module_io("m", inputs=["a"], outputs=["b"])
    assert result["interface"] == ("a", "b")
    assert result["arity"] == (1, 1)


def test_zf_module_manifest_lists_sections_in_stable_order():
    result = framework.zf_module_manifest(
        {
            "module": "m",
            "target": "native",
            "profile": "debug",
            "features": ["simd"],
        }
    )
    assert result["sections"] == ("header", "types", "functions", "exports")
    assert result["layout_size"] is None
    assert result["features"] == ("simd",)


def test_zf_module_plan_sorts_features_and_derives_flags():
    result = framework.zf_module_plan("m", features=["simd", "cabi"])
    assert result["features"] == ("cabi", "simd")
    assert result["imports"] == ("std",)
    assert result["flags"] == ("-femit-bin",)
    assert result["output_name"] == "m.zig"
    assert result["model_only"] is True


def test_zf_module_stats_counts_lines_bytes_and_blank_lines():
    assert framework.zf_module_stats("a\n\nb") == {
        "lines": 3,
        "blank_lines": 1,
        "characters": 4,
        "bytes": 4,
        "max_line_length": 1,
        "trailing_newline": False,
    }


def test_zf_name_mangle_uses_c_namespace_convention():
    result = framework.zf_name_mangle("add")
    assert result["mangled"] == "polyglot__add"
    assert result["length"] == 13
    assert framework.zf_name_mangle("add", abi="zig")["mangled"] == "polyglot.add"


def test_zf_op_cost_model_multiplies_vectorizable_operations():
    result = framework.zf_op_cost_model("add", lanes=4)
    assert result["base_cost"] == 1
    assert result["cost"] == 4
    assert framework.zf_op_cost_model("load", lanes=4)["cost"] == 2


def test_zf_optimize_plan_orders_passes_and_estimates_size():
    result = framework.zf_optimize_plan(1000, passes=["inline", "constant_fold"])
    assert result["passes"] == ("constant_fold", "inline")
    assert result["estimated_size"] == 500
    assert framework.zf_optimize_plan(1000)["estimated_size"] == 1000


def test_zf_parallel_plan_partitions_items_into_contiguous_chunks():
    result = framework.zf_parallel_plan(2, items=10, chunk_size=4)
    assert result["chunks"] == ((0, 4), (4, 8), (8, 10))
    assert result["used_workers"] == 2
    assert result["balanced"] is False


def test_zf_pass_pipeline_sorts_by_compiler_stage():
    result = framework.zf_pass_pipeline(["emit", "parse", "opt"])
    assert result["order"] == ("parse", "opt", "emit")
    assert result["count"] == 3


def test_zf_pass_trace_shrinks_only_at_the_opt_pass():
    result = framework.zf_pass_trace({"order": ["parse", "opt"]}, source_size=100)
    assert [step["output_size"] for step in result["steps"]] == [100, 90]
    assert result["final_size"] == 90
    assert result["total_steps"] == 2


def test_zf_resource_budget_returns_named_limits():
    result = framework.zf_resource_budget(max_memory=32)
    assert result["max_memory_mb"] == 32
    assert result["limits"] == ("lines", "memory", "symbols")
    assert result["model_only"] is True


def test_zf_saturating_model_clamps_integer_overflow():
    result = framework.zf_saturating_model("+|", 100, 100, type_name="i8")
    assert result["raw"] == 200
    assert result["value"] == 127
    assert result["saturated"] is True
    assert framework.zf_saturating_model("-|", 10, 3)["saturated"] is False


def test_zf_source_map_tracks_utf8_byte_offsets():
    result = framework.zf_source_map("a\né")
    assert result["lines"][1]["byte_start"] == 2
    assert result["lines"][1]["byte_end"] == 4
    assert result["total_bytes"] == 4


def test_zf_symbol_registry_sorts_names_and_keeps_metadata():
    result = framework.zf_symbol_registry([("f", 2, "i64"), ("a", 0, "void")])
    assert result["names"] == ("a", "f")
    assert result["by_name"]["f"] == {"arity": 2, "result": "i64"}
    assert result["count"] == 2


def test_zf_target_model_models_wasm32_word_size():
    result = framework.zf_target_model("wasm32")
    assert result["word"] == 4
    assert result["features"] == ("cabi",)
    assert result["endian"] == "little"


def test_zf_type_canonical_canonicalizes_vector_spacing():
    result = framework.zf_type_canonical("@Vector(4,i32)")
    assert result["canonical"] == "@Vector(4, i32)"
    assert result["kind"] == "vector"
    assert result["size"] == 16


def test_zf_type_layout_includes_pointer_element_layout():
    result = framework.zf_type_layout("[*]const u8")
    assert result["kind"] == "pointer"
    assert result["element"]["type"] == "u8"
    assert result["size"] == 8
    assert result["align"] == 8


def test_zf_validation_report_counts_passed_and_failed_checks():
    result = framework.zf_validation_report([("a", True), ("b", False)])
    assert result["passed"] == ("a",)
    assert result["failed"] == ("b",)
    assert result["ok"] is False
    assert result["summary"] == "1/2 checks passed"


# ---------------------------------------------------------------------------
# zig.codegen (30 functions)
# ---------------------------------------------------------------------------


def test_zgc_alignof_expr_emits_builtin():
    result = codegen.zgc_alignof_expr("u32")
    assert result["expr"] == "@alignOf(u32)"
    assert result["align"] == 4
    assert result["builtin"] is True


def test_zgc_array_literal_counts_and_types_elements():
    result = codegen.zgc_array_literal([1, 2, 3], type_name="i16")
    assert result["expr"] == "[3]i16{ 1, 2, 3 }"
    assert result["items"] == (1, 2, 3)
    assert result["length"] == 3


def test_zgc_binary_expr_parenthesizes_saturating_operator():
    result = codegen.zgc_binary_expr("+|", "a", "b")
    assert result["expr"] == "(a +| b)"
    assert result["associative"] is True
    assert "_zig_sat" not in result["expr"]


def test_zgc_block_emits_labeled_indented_source():
    result = codegen.zgc_block(["x = 1;"], label="loop")
    assert result["source"] == "loop: {\n    x = 1;\n}"
    assert result["lines"][0] == "loop: {"
    assert result["count"] == 1


def test_zgc_cast_expr_selects_integer_builtin():
    result = codegen.zgc_cast_expr("i64", "x", kind="int")
    assert result["expr"] == "@intCast(x)"
    assert result["builtin"] == "@intCast"
    assert codegen.zgc_cast_expr("i64", "x")["expr"] == "@as(i64, x)"


def test_zgc_comment_renders_each_line_with_prefix():
    result = codegen.zgc_comment("hi\nthere")
    assert result["source"] == "// hi\n// there"
    assert result["lines"] == ("// hi", "// there")
    assert codegen.zgc_comment("hi", style="block")["source"] == "/* hi */"


def test_zgc_const_decl_is_immutable_and_annotated():
    result = codegen.zgc_const_decl("N", "4", type_name="usize")
    assert result["source"] == "const N: usize = 4;"
    assert result["mutable"] is False


def test_zgc_enum_decl_supports_explicit_discriminants():
    result = codegen.zgc_enum_decl("Color", [("Red", 1), "Green"])
    assert result["variants"] == ("Red", "Green")
    assert "Red = 1," in result["source"]
    assert result["tag"] == "u8"


def test_zgc_export_shim_forwards_to_the_model_function():
    result = codegen.zgc_export_shim(
        "add", params=[("a", "i64"), ("b", "i64")], result="i64"
    )
    assert result["symbol"] == "__cabi_add"
    assert result["callconv"] == "c"
    assert "return add(a, b);" in result["source"]
    assert "callconv(.c) i64" in result["source"]


def test_zgc_expr_stmt_terminates_expression():
    assert codegen.zgc_expr_stmt("f(1)") == {
        "source": "f(1);",
        "expr": "f(1)",
        "terminated": True,
    }


def test_zgc_fn_decl_emits_parameters_body_and_visibility():
    result = codegen.zgc_fn_decl(
        "main", params=[("n", "i64")], result="i64", body=["return n;"]
    )
    assert result["source"] == "fn main(n: i64) i64 {\n    return n;\n}"
    assert result["line_count"] == 3
    assert result["public"] is False
    assert codegen.zgc_fn_decl("f", is_pub=True)["source"].startswith("pub fn f(")


def test_zgc_for_loop_emits_inclusive_while_range():
    result = codegen.zgc_for_loop("i", "0", "4", inclusive=True)
    assert result["source"].startswith("var i: usize = 0;")
    assert "while (i <= 4) : (i += 1) {" in result["source"]
    assert result["inclusive"] is True


def test_zgc_helper_lib_emits_integer_safe_helpers_only():
    result = codegen.zgc_helper_lib(features=["simd", "saturating"])
    assert result["features"] == ("saturating", "simd")
    assert result["helpers"] == ("_zig_sat_add", "_zig_simd_sum4")
    assert "@reduce(.Add, values)" in result["source"]
    assert "@Vector(4, i32)" in result["source"]
    assert "f32" not in result["source"]


def test_zgc_if_expr_is_exhaustive():
    result = codegen.zgc_if_expr("x > 0", "1", "0")
    assert result["expr"] == "if (x > 0) 1 else 0"
    assert result["exhaustive"] is True


def test_zgc_import_decl_derives_alias_from_last_path_part():
    result = codegen.zgc_import_decl("std.math")
    assert result["source"] == 'const math = @import("std.math");'
    assert result["alias"] == "math"
    assert result["parts"] == ("std", "math")


def test_zgc_index_expr_keeps_operands_opaque():
    result = codegen.zgc_index_expr("xs", "i")
    assert result["expr"] == "xs[i]"
    assert result["bounds_checked_model"] is True


def test_zgc_integer_literal_uses_radix_prefix():
    result = codegen.zgc_integer_literal(255, radix=16)
    assert result["expr"] == "0xff"
    assert result["digit_count"] == 2
    assert codegen.zgc_integer_literal(-5)["expr"] == "-5"


def test_zgc_match_expr_emits_switch_with_default_arm():
    result = codegen.zgc_match_expr("x", [("0", "1"), ("1", "2")], otherwise="0")
    assert result["exhaustive"] is True
    assert "else => 0," in result["expr"]
    assert result["arms"] == (("0", "1"), ("1", "2"))


def test_zgc_module_header_includes_optional_std_import():
    result = codegen.zgc_module_header("m", std_import=True, features=["simd"])
    assert result["features"] == ("simd",)
    assert result["imports"] == ('const std = @import("std");',)
    assert result["lines"][0].startswith("// Generated")


def test_zgc_module_source_assembles_imports_and_functions():
    result = codegen.zgc_module_source(
        "m", functions=["fn f() void {}"], imports=["std"]
    )
    assert result["function_count"] == 1
    assert 'const std = @import("std");' in result["source"]
    assert result["source"].endswith("fn f() void {}")
    assert len(result["fingerprint"]) == 16


def test_zgc_ptr_type_models_many_item_const_pointer():
    result = codegen.zgc_ptr_type("i32")
    assert result["expr"] == "[*]const i32"
    assert result["qualifier"] == "const"
    assert result["size"] == 8
    assert codegen.zgc_ptr_type("i32", const=False, many=False)["expr"] == "*i32"


def test_zgc_return_stmt_distinguishes_bare_and_value_returns():
    assert codegen.zgc_return_stmt() == {
        "source": "return;",
        "expr": None,
        "void": True,
    }
    assert codegen.zgc_return_stmt("x")["source"] == "return x;"


def test_zgc_simd_expr_uses_integer_only_vector_builtins():
    result = codegen.zgc_simd_expr("splat", 4, values=[3])
    assert result["expr"] == "@splat(3)"
    assert result["type"] == "@Vector(4, i32)"
    assert result["float_safe"] is True
    reduced = codegen.zgc_simd_expr("reduce_add", 4, values=[1, 2, 3, 4])
    assert "@reduce(.Add, @as(@Vector(4, i32)" in reduced["expr"]


def test_zgc_slice_type_qualifies_element_type():
    result = codegen.zgc_slice_type("u8", const=False)
    assert result["expr"] == "[]u8"
    assert result["size"] == 16
    assert codegen.zgc_slice_type("u8")["expr"] == "[]const u8"


def test_zgc_struct_decl_supports_packed_structs():
    result = codegen.zgc_struct_decl("Pair", [("a", "i32"), ("b", "u8")], packed=True)
    assert result["kind"] == "packed struct"
    assert result["fields"] == ("a", "b")
    assert "    a: i32," in result["source"]


def test_zgc_switch_expr_has_an_explicit_default():
    result = codegen.zgc_switch_expr("x", [("0", "1")])
    assert result["expr"] == "switch (x) { 0 => 1, else => unreachable }"
    assert result["exhaustive"] is True


def test_zgc_type_expr_reports_vector_layout():
    result = codegen.zgc_type_expr("@Vector(4,i32)")
    assert result["expr"] == "@Vector(4, i32)"
    assert result["kind"] == "vector"
    assert result["size"] == 16
    assert result["align"] == 4


def test_zgc_unary_expr_emits_prefix_operator():
    result = codegen.zgc_unary_expr("~", "x")
    assert result["expr"] == "~x"
    assert result["prefix"] is True


def test_zgc_var_decl_tracks_mutability():
    result = codegen.zgc_var_decl("x", "1", type_name="u8")
    assert result["source"] == "var x: u8 = 1;"
    assert result["mutable"] is True
    assert codegen.zgc_var_decl("x", "1", constant=True)["source"] == "const x = 1;"


def test_zgc_while_loop_emits_defer_before_body():
    result = codegen.zgc_while_loop("i < 3", body=["i += 1;"], defer_expr="cleanup()")
    assert result["lines"][1] == "    defer cleanup();"
    assert result["body_count"] == 1
    assert result["source"].startswith("while (i < 3) {")


# ---------------------------------------------------------------------------
# zig.abi (30 functions)
# ---------------------------------------------------------------------------


def test_zabi_abi_signature_canonicalizes_parameter_types():
    result = abi.zabi_abi_signature("add", ["i32", "i32"], "i32")
    assert result["canonical"] == "add(i32,i32)->i32"
    assert result["params"] == ("i32", "i32")
    assert len(result["fingerprint"]) == 16


def test_zabi_alignment_reports_integer_alignment():
    result = abi.zabi_alignment("u16")
    assert result["align"] == 2
    assert result["size"] == 2
    assert result["kind"] == "int"


def test_zabi_arg_slots_align_arguments_to_word_boundaries():
    result = abi.zabi_arg_slots({"params": ["u8", "u32"], "result": "void"})
    assert [slot["offset"] for slot in result["slots"]] == [0, 4]
    assert result["bytes"] == 8
    assert result["align"] == 4


def test_zabi_bitcast_plan_accepts_equal_width_types():
    result = abi.zabi_bitcast_plan("u32", "i32")
    assert result["size"] == 4
    assert result["allowed"] is True
    assert result["expr"] == "@as(i32, @bitCast(value))"


def test_zabi_buffer_layout_multiplies_capacity_by_element_size():
    result = abi.zabi_buffer_layout(4, elem_type="u16")
    assert result["expr"] == "[4]u16"
    assert result["size"] == 8
    assert result["align"] == 2


def test_zabi_cabi_type_widens_small_integers():
    result = abi.zabi_cabi_type("i32")
    assert result["cabi_type"] == "i64"
    assert abi.zabi_cabi_type("usize")["cabi_type"] == "usize"
    assert abi.zabi_cabi_type("f64")["cabi_type"] == "f64"


def test_zabi_call_frame_reserves_argument_and_return_slots():
    result = abi.zabi_call_frame({"params": ["u8"], "result": "i64"})
    assert result["args_size"] == 8
    assert result["return_slot"] == 8
    assert result["frame_size"] == 16
    assert result["align"] == 8


def test_zabi_calling_convention_selects_callconv_expression():
    result = abi.zabi_calling_convention("c")
    assert result["name"] == ".c"
    assert result["expr"] == "callconv(.c)"
    assert result["preserves"] == ("callee_saved_registers",)


def test_zabi_canonical_signature_normalizes_whitespace():
    result = abi.zabi_canonical_signature("i64 (i64, ptr)")
    assert result["canonical"] == "i64(i64,ptr)"
    assert result["params"] == ("i64", "ptr")
    assert result["result"] == "i64"


def test_zabi_data_layout_models_little_endian_words():
    result = abi.zabi_data_layout("native")
    assert result["word"] == 8
    assert result["endian"] == "little"
    assert result["stack_direction"] == "down"


def test_zabi_enum_repr_uses_explicit_and_implicit_values():
    result = abi.zabi_enum_repr(["A", ("B", 4)])
    assert result["variants"] == ("A", "B")
    assert result["max_value"] == 4
    assert result["expr"] == "enum(u8)"


def test_zabi_error_map_translates_e006_to_missing_artifact():
    result = abi.zabi_error_map("E006")
    assert result["errno"] == 2
    assert result["name"] == "missing_artifact"
    assert result["recoverable"] is True


def test_zabi_export_registry_prefixes_symbol_names():
    result = abi.zabi_export_registry([("f", "i64(i64)->i64")])
    assert result["symbols"] == ("__cabi_f",)
    assert result["by_symbol"]["__cabi_f"]["signature"] == "i64(i64)"
    assert result["count"] == 1


def test_zabi_field_offsets_pads_fields_for_alignment():
    result = abi.zabi_field_offsets([("a", "u8"), ("b", "u32")])
    assert [row["offset"] for row in result["fields"]] == [0, 4]
    assert result["size"] == 8
    assert result["padding"] == 3


def test_zabi_float_policy_is_scalar_only():
    result = abi.zabi_float_policy("f32")
    assert result["precision"] == "single"
    assert result["vector_ok"] is False
    assert "integer-safe" in result["note"]


def test_zabi_int_conversion_wraps_into_destination_width():
    result = abi.zabi_int_conversion(-1, target="u8")
    assert result["wrapped"] == 255
    assert result["overflow"] is True
    assert abi.zabi_int_conversion(7, target="u8")["wrapped"] == 7


def test_zabi_layout_compare_reports_declared_size_drift():
    result = abi.zabi_layout_compare(
        {"size": 8, "align": 4, "fields": [("a", "u32")]},
        {"size": 4, "align": 4, "fields": [("a", "u32")]},
    )
    assert result["equal"] is False
    assert result["size_delta"] == -4
    assert result["differences"] == ("size",)


def test_zabi_layout_fingerprint_depends_on_shape():
    result = abi.zabi_layout_fingerprint(
        {"size": 8, "align": 4, "fields": [("a", "u32")]}
    )
    assert result["fields"] == ("a",)
    assert result["size"] == 8
    assert len(result["fingerprint"]) == 16


def test_zabi_opaque_address_is_an_opaque_mapping_key():
    result = abi.zabi_opaque_address(0x1000, state={"0x1000": {"v": 1}})
    assert result["key"] == "0x1000"
    assert result["present"] is True
    assert result["dereferenced"] is False
    assert result["model_only"] is True
    assert abi.zabi_opaque_address(0x1000)["present"] is False


def test_zabi_param_class_distinguishes_integers_from_pointers():
    result = abi.zabi_param_class("i64")
    assert result["class"] == "int"
    assert result["by_value"] is True
    assert abi.zabi_param_class("[*]const u8")["class"] == "pointer"


def test_zabi_pointer_width_tracks_target_word():
    assert abi.zabi_pointer_width("wasm32") == {
        "target": "wasm32",
        "bits": 32,
        "bytes": 4,
        "model_only": True,
    }
    assert abi.zabi_pointer_width("native")["bits"] == 64


def test_zabi_return_class_uses_struct_return_for_large_vectors():
    result = abi.zabi_return_class("@Vector(16, i32)")
    assert result["class"] == "sret"
    assert result["registers"] == 0
    assert abi.zabi_return_class("void")["class"] == "void"


def test_zabi_serialization_format_has_versioned_header():
    result = abi.zabi_serialization_format("json")
    assert result["header"] == "PGLY-json-1"
    assert result["checksum"] == "crc32-model"
    assert result["version"] == 1


def test_zabi_shim_body_casts_arguments_at_the_boundary():
    result = abi.zabi_shim_body("add", [("a", "i32"), ("b", "i32")], "i32")
    assert result["symbol"] == "__cabi_add"
    assert result["arg_count"] == 2
    assert "@as(i32, @intCast(a))" in result["source"]
    assert "callconv(.c) i32" in result["source"]


def test_zabi_sizeof_counts_vector_bits():
    result = abi.zabi_sizeof("@Vector(4,i32)")
    assert result["size"] == 16
    assert result["bits"] == 128
    assert result["kind"] == "vector"


def test_zabi_slice_repr_is_pointer_plus_length():
    result = abi.zabi_slice_repr("u8")
    assert result["expr"] == "[]const u8"
    assert result["ptr_type"] == "[*]const u8"
    assert result["len_type"] == "usize"
    assert result["size"] == 16


def test_zabi_struct_repr_packs_fields_without_padding():
    result = abi.zabi_struct_repr("S", [("a", "u8"), ("b", "u32")], packed=True)
    assert result["expr"] == "packed struct"
    assert result["size"] == 5
    assert result["align"] == 1
    assert result["padding"] == 0


def test_zabi_symbol_name_includes_abi_and_version():
    result = abi.zabi_symbol_name("add")
    assert result["mangled"] == "__cabi_polyglot_add_v1"
    assert abi.zabi_symbol_name("add", abi="zig")["mangled"] == "polyglot.add"


def test_zabi_target_spec_limits_wasm_abis():
    result = abi.zabi_target_spec("wasm32")
    assert result["word"] == 4
    assert result["abis"] == ("c",)
    assert result["model_only"] is True


def test_zabi_type_class_describes_primitive_layout():
    result = abi.zabi_type_class("bool")
    assert result["class"] == "bool"
    assert result["size"] == 1
    assert result["signed"] is None


# ---------------------------------------------------------------------------
# E001-E008 diagnostic coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "args", "kwargs", "code"),
    [
        (framework.zf_type_layout, ("NoSuchType",), {}, "E001"),
        (abi.zabi_bitcast_plan, ("u8", "u32"), {}, "E001"),
        (framework.zf_module_manifest, ({},), {}, "E002"),
        (codegen.zgc_binary_expr, ("**", "a", "b"), {}, "E003"),
        (framework.zf_optimize_plan, (10_000_001,), {}, "E004"),
        (codegen.zgc_module_source, ("m",), {"functions": ["x" * 1_000_001]}, "E004"),
        (
            framework.zf_dependency_order,
            ({"edges": [("a", "b"), ("b", "a")]},),
            {},
            "E005",
        ),
        (abi.zabi_opaque_address, (0x10,), {"state": {}}, "E006"),
        (framework.zf_feature_gate, ("simd",), {"target": "wasm32"}, "E007"),
        (codegen.zgc_simd_expr, ("splat", 4), {"type_name": "f32"}, "E007"),
        (abi.zabi_canonical_signature, ("not a signature",), {}, "E008"),
        (abi.zabi_serialization_format, ("json",), {"version": 3}, "E008"),
    ],
)
def test_error_codes_are_structured(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    code: str,
) -> None:
    assert _code(function, *args, **kwargs) == code


@pytest.mark.parametrize(
    ("function", "args", "kwargs"),
    [
        (framework.zf_module_plan, (123,), {}),
        (framework.zf_cache_lookup, ([], "k"), {}),
        (codegen.zgc_fn_decl, ("f",), {"params": "bad"}),
        (codegen.zgc_array_literal, ("123",), {}),
        (abi.zabi_field_offsets, (None,), {}),
        (abi.zabi_error_map, ("E010",), {"operation": "emit"}),
    ],
)
def test_invalid_api_inputs_raise_e003(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    assert _code(function, *args, **kwargs) == "E003"


def test_layout_mismatch_and_malformed_layout_state_are_separate_codes():
    assert (
        _code(
            framework.zf_layout_validate,
            {"name": "S", "fields": [], "size": 4, "align": 1},
        )
        == "E001"
    )
    assert (
        _code(
            framework.zf_layout_validate,
            {"name": "S", "fields": [], "size": "bad", "align": 1},
        )
        == "E008"
    )
    assert _code(framework.zf_layout_validate, {"name": "S"}) == "E002"


def test_conflicting_signatures_and_duplicate_symbols_are_rejected():
    assert (
        _code(
            framework.zf_dispatch_table,
            [("f", "i64()"), ("f", "void()")],
        )
        == "E001"
    )
    assert (
        _code(
            abi.zabi_export_registry,
            [("f", "i64()"), ("f", "void()")],
        )
        == "E001"
    )
    assert (
        _code(framework.zf_symbol_registry, [("f", 1, "i64"), ("f", 2, "i64")])
        == "E003"
    )
