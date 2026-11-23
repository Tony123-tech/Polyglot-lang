"""Behavioral tests for the twenty source-span helpers.

Every test uses a source string containing multibyte UTF-8 characters so the
character offsets and UTF-8 byte offsets cannot be accidentally conflated.
"""

from __future__ import annotations

import pytest

import compiler.core.ast_nodes as ast
from compiler.frontend import spans
from compiler.frontend.errors import PolyglotError
from compiler.core.lexer import Token, TokenType

SOURCE = "aé日🙂\nb"


def test_span_offset_to_line_col_multibyte_and_clamping():
    assert spans.span_offset_to_line_col(SOURCE, 2) == [1, 3]
    assert spans.span_offset_to_line_col(SOURCE, 5) == [2, 1]
    assert spans.span_offset_to_line_col(SOURCE, -3) == [1, 1]
    assert spans.span_offset_to_line_col(SOURCE, 99) == [2, 2]
    with pytest.raises(PolyglotError) as error:
        spans.span_offset_to_line_col(SOURCE, 1.5)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_offset_to_line_col(b"a", 0)


def test_span_line_col_to_offset_inverse_and_bad_types():
    assert spans.span_line_col_to_offset(SOURCE, 1, 3) == 2
    assert spans.span_line_col_to_offset(SOURCE, 2, 1) == 5
    assert spans.span_line_col_to_offset(SOURCE, 9, 1) == len(SOURCE)
    assert spans.span_line_col_to_offset(SOURCE, 0, 4) == 0
    with pytest.raises(PolyglotError) as error:
        spans.span_line_col_to_offset(SOURCE, "1", 1)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_line_col_to_offset(SOURCE, 1, None)


def test_span_char_to_byte_counts_utf8_prefixes():
    assert [spans.span_char_to_byte(SOURCE, index) for index in range(7)] == [
        0,
        1,
        3,
        6,
        10,
        11,
        12,
    ]
    assert spans.span_char_to_byte(SOURCE, 99) == 12
    assert spans.span_char_to_byte(SOURCE, -1) == 0
    with pytest.raises(PolyglotError) as error:
        spans.span_char_to_byte(SOURCE, "2")
    assert error.value.code == "E003"


def test_span_byte_to_char_rejects_splits_and_out_of_range():
    byte_offsets = (0, 1, 3, 6, 10, 11, 12)
    assert [spans.span_byte_to_char(SOURCE, value) for value in byte_offsets] == list(
        range(7)
    )
    with pytest.raises(PolyglotError) as error:
        spans.span_byte_to_char(SOURCE, 2)
    assert error.value.code == "E008"
    assert (error.value.line, error.value.col) == (1, 2)
    with pytest.raises(PolyglotError) as error:
        spans.span_byte_to_char(SOURCE, 13)
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        spans.span_byte_to_char(SOURCE, -1)
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        spans.span_byte_to_char(SOURCE, "6")
    assert error.value.code == "E003"


def test_span_make_char_byte_and_line_col_record():
    assert spans.span_make(SOURCE, 1, 4) == {
        "char_span": [1, 4],
        "byte_span": [1, 10],
        "line_col_span": [[1, 2], [1, 5]],
        "text": "é日🙂",
    }
    assert spans.span_make(SOURCE, 4, 1)["char_span"] == [1, 4]
    assert spans.span_make(SOURCE, -2, 99)["char_span"] == [0, len(SOURCE)]
    with pytest.raises(PolyglotError) as error:
        spans.span_make(SOURCE, "1", 4)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_make(None, 1, 4)


def test_span_union_uses_multibyte_char_boundaries():
    left = spans.span_make(SOURCE, 1, 3)["char_span"]
    right = spans.span_make(SOURCE, 2, 4)["char_span"]
    assert spans.span_union(left, right) == [1, 4]
    assert SOURCE[1:4] == "é日🙂"
    assert spans.span_union((3, 5), (0, 2)) == [0, 5]
    with pytest.raises(PolyglotError) as error:
        spans.span_union((1,), (2, 3))
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_union((1, 3), (4, 2))


def test_span_intersect_overlap_and_touching():
    left = spans.span_make(SOURCE, 1, 3)["char_span"]
    right = spans.span_make(SOURCE, 2, 4)["char_span"]
    assert spans.span_intersect(left, right) == [2, 3]
    assert SOURCE[2:3] == "日"
    assert spans.span_intersect((1, 2), (2, 4)) is None
    assert spans.span_intersect((0, 1), (2, 3)) is None
    with pytest.raises(PolyglotError) as error:
        spans.span_intersect((1, 3), ("2", 4))
    assert error.value.code == "E003"


def test_span_contains_touching_boundaries():
    outer = spans.span_make(SOURCE, 0, 4)["char_span"]
    inner = spans.span_make(SOURCE, 1, 3)["char_span"]
    assert spans.span_contains(outer, inner) is True
    assert SOURCE[0:4] == "aé日🙂"
    assert spans.span_contains(inner, outer) is False
    assert spans.span_contains((1, 3), (1, 3)) is True
    assert spans.span_contains((1, 3), (2, 3)) is True
    assert spans.span_contains((1, 3), (3, 4)) is False
    assert spans.span_contains((1, 3), (0, 2)) is False
    with pytest.raises(PolyglotError) as error:
        spans.span_contains((1, 3), (1, 2, 3))
    assert error.value.code == "E003"


def test_span_overlaps_shares_at_least_one_character():
    left = spans.span_make(SOURCE, 1, 3)["char_span"]
    right = spans.span_make(SOURCE, 2, 4)["char_span"]
    assert spans.span_overlaps(left, right) is True
    assert spans.span_overlaps((1, 2), (2, 3)) is False
    assert spans.span_overlaps((0, 2), (1, 3)) is True
    with pytest.raises(PolyglotError) as error:
        spans.span_overlaps((1, 3), 4)
    assert error.value.code == "E003"


def test_span_compare_orders_by_start_then_end():
    left = spans.span_make(SOURCE, 1, 4)["char_span"]
    right = spans.span_make(SOURCE, 2, 3)["char_span"]
    assert SOURCE[1:4] == "é日🙂"
    assert spans.span_compare(left, right) == -1
    assert spans.span_compare(right, left) == 1
    assert spans.span_compare((1, 3), (1, 4)) == -1
    assert spans.span_compare((1, 4), (1, 3)) == 1
    assert spans.span_compare((1, 3), (1, 3)) == 0
    with pytest.raises(PolyglotError) as error:
        spans.span_compare((1, 3), (2, 1))
    assert error.value.code == "E003"


def test_span_shift_clamps_at_zero():
    span = spans.span_make(SOURCE, 1, 4)["char_span"]
    assert spans.span_shift(span, 1) == [2, 5]
    assert SOURCE[2:5] == "日🙂\n"
    assert spans.span_shift(span, -2) == [0, 2]
    assert spans.span_shift((1, 3), -99) == [0, 0]
    with pytest.raises(PolyglotError) as error:
        spans.span_shift((1, 3), "1")
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_shift((3, 1), 1)


def test_span_clamp_outside_ranges():
    span = spans.span_make(SOURCE, 1, 4)["char_span"]
    assert SOURCE[span[0] : span[1]] == "é日🙂"
    assert spans.span_clamp(span, 2, 3) == [2, 3]
    assert spans.span_clamp(span, 0, 2) == [1, 2]
    assert spans.span_clamp(span, 4, 6) == [4, 4]
    assert spans.span_clamp(span, 5, 6) == [5, 5]
    assert spans.span_clamp((5, 6), 0, 2) == [2, 2]
    with pytest.raises(PolyglotError) as error:
        spans.span_clamp(span, 5, 2)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_clamp(span, 0, "6")


def test_span_merge_adjacent_sorts_and_merges_touching():
    items = [
        spans.span_make(SOURCE, 2, 4)["char_span"],
        [0, 1],
        spans.span_make(SOURCE, 1, 2)["char_span"],
        [4, 5],
    ]
    assert spans.span_merge_adjacent(items) == [[0, 5]]
    assert SOURCE[0:5] == "aé日🙂\n"
    assert spans.span_merge_adjacent([[2, 4], [0, 2], [5, 6]]) == [[0, 4], [5, 6]]
    assert spans.span_merge_adjacent([]) == []
    with pytest.raises(PolyglotError) as error:
        spans.span_merge_adjacent(5)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_merge_adjacent([[1, 2], [3]])


def test_span_from_token_uses_raw_lexeme_and_utf8():
    source = 'x = "é🙂";'
    token = Token(TokenType.STRING, "é🙂", 1, 5, 4)
    record = spans.span_from_token(source, token)
    assert record["type"] == "STRING"
    assert record["value"] == "é🙂"
    assert record["char_span"] == [4, 8]
    assert record["byte_span"] == [4, 12]
    assert record["line_col_span"] == [[1, 5], [1, 9]]
    assert record["text"] == '"é🙂"'
    assert spans.span_from_token(source, Token(TokenType.EOF, "", 1, 9, 9))[
        "char_span"
    ] == [9, 9]
    with pytest.raises(PolyglotError) as error:
        spans.span_from_token(source, "STRING")
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_from_token(None, token)


def test_span_from_ast_node_valid_missing_and_malformed():
    node = ast.Node(1, 1, span=(1, 4))
    assert spans.span_from_ast_node(node) == [1, 4]
    assert SOURCE[1:4] == "é日🙂"
    assert spans.span_from_ast_node(ast.Node(1, 1)) is None
    assert spans.span_from_ast_node(object()) is None
    assert spans.span_from_ast_node({"span": (1, 2)}) is None
    malformed = ast.Node(1, 1)
    malformed.span = "oops"
    with pytest.raises(PolyglotError) as error:
        spans.span_from_ast_node(malformed)
    assert error.value.code == "E003"


def test_span_line_range_whole_lines_and_newlines():
    assert spans.span_line_range(SOURCE, 1, 1) == [0, 5]
    assert SOURCE[0:5] == "aé日🙂\n"
    assert spans.span_line_range(SOURCE, 2, 2) == [5, 6]
    assert spans.span_line_range(SOURCE, 1, 2) == [0, 6]
    assert spans.span_line_range(SOURCE, 4, 5) == [5, 6]
    with pytest.raises(PolyglotError) as error:
        spans.span_line_range(SOURCE, 0, 1)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_line_range(SOURCE, 2, 1)
    with pytest.raises(PolyglotError):
        spans.span_line_range(SOURCE, 1, "2")


def test_span_text_slice_and_validation():
    span = spans.span_make(SOURCE, 1, 4)["char_span"]
    assert spans.span_text(SOURCE, span) == "é日🙂"
    assert spans.span_text(SOURCE, (-2, 99)) == SOURCE
    with pytest.raises(PolyglotError) as error:
        spans.span_text(SOURCE, (4, 1))
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_text(SOURCE, (1, "4"))


def test_span_utf8_validate_boundaries():
    assert spans.span_utf8_validate(SOURCE, (1, 3)) == {
        "valid": True,
        "reason": None,
        "char_span": [1, 3],
        "byte_span": [1, 6],
    }
    assert spans.span_utf8_validate(SOURCE, (1, 2)) == {
        "valid": False,
        "reason": "splits multibyte character",
        "char_span": [1, 2],
        "byte_span": [1, 3],
    }
    assert spans.span_utf8_validate(SOURCE, (1, 4))["valid"] is False
    assert spans.span_utf8_validate(SOURCE, (2, 4))["valid"] is False
    with pytest.raises(PolyglotError) as error:
        spans.span_utf8_validate(SOURCE, (1, None))
    assert error.value.code == "E003"


def test_span_format_canonical():
    span = spans.span_make(SOURCE, 1, 4)["char_span"]
    assert SOURCE[1:4] == "é日🙂"
    assert spans.span_format(span) == "[1, 4)"
    assert spans.span_format((0, 0)) == "[0, 0)"
    with pytest.raises(PolyglotError) as error:
        spans.span_format((1,))
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError):
        spans.span_format((4, 1))


def test_span_validate_never_raises():
    assert spans.span_validate(SOURCE, (1, 3)) is True
    assert spans.span_validate(SOURCE, (3, 6)) is True
    assert spans.span_validate(SOURCE, (1, 2)) is False
    assert spans.span_validate(SOURCE, (1, 4)) is False
    assert spans.span_validate(SOURCE, (2, 4)) is False
    assert spans.span_validate(SOURCE, (5, 3)) is False
    assert spans.span_validate(SOURCE, (0, 99)) is False
    assert spans.span_validate(SOURCE, (0, "1")) is False
    assert spans.span_validate(SOURCE, 5) is False
    assert spans.span_validate(None, (0, 1)) is False
