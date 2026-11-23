"""Behavioral tests for the ``compiler.frontend.lexing`` helper surface."""

from __future__ import annotations

import hashlib

import pytest

from compiler.frontend import lexing
from compiler.frontend.errors import PolyglotError
from compiler.core.lexer import KEYWORDS, TokenType

_PUBLIC_NAMES = [
    "lex_tokenize",
    "lex_count_by_type",
    "lex_keywords",
    "lex_identifiers",
    "lex_literals",
    "lex_operators",
    "lex_validate_identifier",
    "lex_decode_numeric",
    "lex_decode_string_escapes",
    "lex_detect_bom",
    "lex_normalize_newlines",
    "lex_split_lines",
    "lex_scan_comments",
    "lex_balance_delimiters",
    "lex_indent_levels",
    "lex_source_fingerprint",
    "lex_token_window",
    "lex_find_token",
    "lex_token_distance",
    "lex_validate_source",
    "lex_keyword_boundary",
    "lex_char_class",
    "lex_escape_table",
    "lex_measure_token_widths",
    "lex_tokenize_with_offsets",
]


def test_all_public_functions_are_importable():
    assert len(_PUBLIC_NAMES) == 25
    for name in _PUBLIC_NAMES:
        function = getattr(lexing, name)
        assert callable(function)
        assert function.__doc__


def test_lex_tokenize_records_every_token_and_eof():
    source = 'let x = "a\\n";'
    records = lexing.lex_tokenize(source)

    assert [record["type"] for record in records] == [
        "LET",
        "IDENT",
        "EQ",
        "STRING",
        "SEMI",
        "EOF",
    ]
    assert [record["char_span"] for record in records] == [
        [0, 3],
        [4, 5],
        [6, 7],
        [8, 13],
        [13, 14],
        [14, 14],
    ]
    assert records[3]["value"] == "a\n"
    assert source[8:13] == '"a\\n"'
    assert records[3]["byte_span"] == [8, 13]
    assert records[-1]["byte_span"] == [len(source), len(source)]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_tokenize(123)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_count_by_type_sorted_and_excludes_eof():
    counts = lexing.lex_count_by_type("if x else { let y = true; }")

    assert counts == {
        "ELSE": 1,
        "EQ": 1,
        "IDENT": 2,
        "IF": 1,
        "LBRACE": 1,
        "LET": 1,
        "RBRACE": 1,
        "SEMI": 1,
        "TRUE": 1,
    }
    assert list(counts) == sorted(counts)
    with pytest.raises(PolyglotError) as error:
        lexing.lex_count_by_type(None)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_keywords_uses_lexer_keyword_map_and_source_spans():
    assert lexing._KEYWORD_TYPES == frozenset(KEYWORDS.values())
    assert TokenType.TRUE in lexing._KEYWORD_TYPES

    keywords = lexing.lex_keywords("let x = true;\nif x { return false; }")

    assert keywords == [
        {"keyword": "let", "line": 1, "col": 1, "char_span": [0, 3]},
        {"keyword": "true", "line": 1, "col": 9, "char_span": [8, 12]},
        {"keyword": "if", "line": 2, "col": 1, "char_span": [14, 16]},
        {"keyword": "return", "line": 2, "col": 8, "char_span": [21, 27]},
        {"keyword": "false", "line": 2, "col": 15, "char_span": [28, 33]},
    ]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_keywords(b"let")  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_identifiers_unique_in_first_appearance_order():
    assert lexing.lex_identifiers("let alpha = beta + alpha;") == ["alpha", "beta"]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_identifiers([])  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_literals_use_raw_source_and_reject_unterminated_lexemes():
    source = "let n = 0xFF_FF; let s = \"é\\n\"; let c = '\\t'; let t = true;"
    literals = lexing.lex_literals(source)

    assert [literal["kind"] for literal in literals] == [
        "number",
        "string",
        "char",
        "bool",
    ]
    assert literals[0]["raw"] == "0xFF_FF"
    assert literals[0]["value"] == "0xFF_FF"
    assert literals[1]["raw"] == '"é\\n"'
    assert literals[1]["value"] == "é\n"
    assert literals[1]["char_span"] == [25, 30]
    assert literals[2]["raw"] == "'\\t'"
    assert literals[2]["value"] == "\t"
    assert literals[3]["raw"] == "true"
    assert literals[3]["char_span"] == [54, 58]

    with pytest.raises(PolyglotError) as error:
        lexing.lex_literals('let s = "oops')
    assert error.value.code == "E001"
    assert error.value.span == (8, 9)
    with pytest.raises(PolyglotError) as error:
        lexing.lex_literals(1.5)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_operators_excludes_tokens_from_other_categories():
    records = lexing.lex_operators("let x = a + b; f(1, 2);")

    assert [(record["op"], record["type"]) for record in records] == [
        ("=", "EQ"),
        ("+", "PLUS"),
        (";", "SEMI"),
        ("(", "LPAREN"),
        (",", "COMMA"),
        (")", "RPAREN"),
        (";", "SEMI"),
    ]
    assert records[0]["char_span"] == [6, 7]
    assert all(
        TokenType[record["type"]] not in lexing._NON_OPERATOR_TYPES
        for record in records
    )
    assert not any(record["op"] == "let" for record in records)
    with pytest.raises(PolyglotError) as error:
        lexing.lex_operators({})  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_validate_identifier_accepts_valid_and_rejects_reserved_words():
    assert lexing.lex_validate_identifier("foo_1") == {
        "name": "foo_1",
        "valid": True,
        "reason": None,
        "code": None,
    }
    assert lexing.lex_validate_identifier("_")["valid"] is True

    for bad in ("", "1abc", "a-b", "let", "a b"):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_validate_identifier(bad)
        assert error.value.code == "E001"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_validate_identifier(7)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_decode_numeric_handles_radices_underscores_and_floats():
    assert lexing.lex_decode_numeric("1_000") == {
        "raw": "1_000",
        "value": 1000,
        "kind": "int",
        "radix": 10,
    }
    assert lexing.lex_decode_numeric("0xFF_FF") == {
        "raw": "0xFF_FF",
        "value": 65535,
        "kind": "int",
        "radix": 16,
    }
    assert lexing.lex_decode_numeric("0o17")["value"] == 15
    assert lexing.lex_decode_numeric("0b1010")["radix"] == 2
    assert lexing.lex_decode_numeric("1_000.5e-2") == {
        "raw": "1_000.5e-2",
        "value": 10.005,
        "kind": "float",
        "radix": 10,
    }
    assert lexing.lex_decode_numeric("1.")["value"] == 1.0

    for bad in ("", "0x", "1__0", "1.2.3", "nan", "0b2", "1e"):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_decode_numeric(bad)
        assert error.value.code == "E008"


def test_lex_decode_string_escapes_decodes_all_forms_with_spans():
    result = lexing.lex_decode_string_escapes(r"a\n\x41\u{1F600}\u00e9")

    assert result["value"] == "a\nA😀é"
    assert result["char_length"] == 5
    assert result["escapes"] == [
        {"raw": "\\n", "value": "\n", "char_span": [1, 3]},
        {"raw": "\\x41", "value": "A", "char_span": [3, 7]},
        {"raw": "\\u{1F600}", "value": "😀", "char_span": [7, 16]},
        {"raw": "\\u00e9", "value": "é", "char_span": [16, 22]},
    ]
    for bad in (r"\q", r"\xZZ", r"\u12", r"\u{110000}", "\\"):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_decode_string_escapes(bad)
        assert error.value.code == "E008"


def test_lex_detect_bom_reports_leading_or_misplaced_bom():
    assert lexing.lex_detect_bom("\ufefflet x = 1;") == {
        "encoding": "utf-8",
        "bom_length": 1,
        "content_start": 1,
    }
    assert lexing.lex_detect_bom("let x = 1;") == {
        "encoding": None,
        "bom_length": 0,
        "content_start": 0,
    }
    with pytest.raises(PolyglotError) as error:
        lexing.lex_detect_bom("a\ufeffb")
    assert error.value.code == "E001"
    assert error.value.span == (1, 2)
    with pytest.raises(PolyglotError) as error:
        lexing.lex_detect_bom(123)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_normalize_newlines_counts_each_crlf_and_cr_once():
    assert lexing.lex_normalize_newlines("a\r\nb\rc") == {
        "source": "a\nb\nc",
        "replacements": 2,
        "line_count": 3,
    }
    assert lexing.lex_normalize_newlines("a\nb")["replacements"] == 0
    with pytest.raises(PolyglotError) as error:
        lexing.lex_normalize_newlines(None)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_split_lines_excludes_newlines_and_keeps_byte_spans():
    assert lexing.lex_split_lines("ab\né") == [
        {
            "index": 0,
            "text": "ab",
            "char_span": [0, 2],
            "byte_span": [0, 2],
            "line_no": 1,
        },
        {
            "index": 1,
            "text": "é",
            "char_span": [3, 4],
            "byte_span": [3, 5],
            "line_no": 2,
        },
    ]
    assert lexing.lex_split_lines("a\n")[1]["char_span"] == [2, 2]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_split_lines(1)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_scan_comments_nests_blocks_and_diagnoses_unterminated_comment():
    source = (
        "// hi\n/* a /* b */ c */ let s = \"// not a comment\"; let c = '/*'; /* bad"
    )
    result = lexing.lex_scan_comments(source)

    assert result["comments"] == [
        {"kind": "line", "text": "// hi", "char_span": [0, 5], "nesting": 0},
        {
            "kind": "block",
            "text": "/* a /* b */ c */",
            "char_span": [6, 23],
            "nesting": 2,
        },
    ]
    assert len(result["diagnostics"]) == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["code"] == "E001"
    assert diagnostic["message"] == "unterminated block comment"
    assert diagnostic["char_span"] == [66, len(source)]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_scan_comments(None)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_balance_delimiters_reports_pairs_mismatches_and_unclosed():
    valid = lexing.lex_balance_delimiters('f([1, 2], {x: ")"});')
    assert valid == {
        "balanced": True,
        "pairs": [
            {"open": "[", "close": "]", "char_span": [2, 8]},
            {"open": "{", "close": "}", "char_span": [10, 18]},
            {"open": "(", "close": ")", "char_span": [1, 19]},
        ],
        "unclosed": [],
        "unexpected": [],
        "diagnostics": [],
    }
    assert lexing.lex_balance_delimiters('"(" // )')["balanced"] is True

    mismatch = lexing.lex_balance_delimiters("(]")
    assert mismatch["balanced"] is False
    assert mismatch["unexpected"] == [{"close": "]", "char_span": [1, 2]}]
    assert mismatch["diagnostics"][0]["code"] == "E003"
    assert mismatch["diagnostics"][0]["char_span"] == [1, 2]

    unclosed = lexing.lex_balance_delimiters("f(")
    assert unclosed["unclosed"] == [{"open": "(", "char_span": [1, 2]}]
    assert unclosed["diagnostics"][0]["message"] == "unclosed delimiter '('"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_balance_delimiters(3)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_indent_levels_skips_blank_lines_and_weights_tabs():
    assert lexing.lex_indent_levels("a\n\t  b\n\n    c") == [
        {"line": 1, "spaces": 0, "tabs": 0, "width": 0, "text_start": 0},
        {"line": 2, "spaces": 2, "tabs": 1, "width": 6, "text_start": 3},
        {"line": 4, "spaces": 4, "tabs": 0, "width": 4, "text_start": 4},
    ]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_indent_levels(False)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_source_fingerprint_is_deterministic_after_newline_normalization():
    crlf = lexing.lex_source_fingerprint("let x = 1;\r\nlet y = 2;")
    lf = lexing.lex_source_fingerprint("let x = 1;\nlet y = 2;")
    assert crlf == lf
    expected = (
        "LET:let\nIDENT:x\nEQ:=\nNUMBER:1\nSEMI:;\n"
        "LET:let\nIDENT:y\nEQ:=\nNUMBER:2\nSEMI:;\nEOF:"
    )
    assert crlf == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert crlf == "31a105232adc60977f958c5ce1ca885a8b5769fae2b105e4ede8f49fcd23b85c"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_source_fingerprint(7)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_token_window_selects_nearest_token_and_inclusive_radius():
    source = "let x = 1;"
    window = lexing.lex_token_window(source, 6, 1)
    assert [record["value"] for record in window] == ["x", "=", "1"]
    assert lexing.lex_token_window(source, 5, 0)[0]["value"] == "x"
    assert lexing.lex_token_window(source, len(source), 0)[0]["type"] == "EOF"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_token_window(source, -1)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_token_window(source, len(source) + 1)
    assert error.value.code == "E003"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_token_window(source, 0, -1)
    assert error.value.code == "E003"


def test_lex_find_token_returns_nth_match_or_none():
    assert lexing.lex_find_token("a + a + a", "a")["char_span"] == [0, 1]
    assert lexing.lex_find_token("a + a + a", "a", nth=1)["char_span"] == [4, 5]
    assert lexing.lex_find_token("a + a + a", "a", nth=2)["char_span"] == [8, 9]
    assert lexing.lex_find_token("a + a", "a", nth=3) is None
    with pytest.raises(PolyglotError) as error:
        lexing.lex_find_token("a", "a", nth=-1)
    assert error.value.code == "E003"


def test_lex_token_distance_requires_second_token_after_first():
    assert lexing.lex_token_distance("a + b + c", "a", "c") == 4
    assert lexing.lex_token_distance("a + a", "a", "a") == 2
    assert lexing.lex_token_distance("b + a", "a", "b") is None
    assert lexing.lex_token_distance("a", "a", "z") is None
    with pytest.raises(PolyglotError) as error:
        lexing.lex_token_distance(None, "a", "b")  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_validate_source_reports_lexical_problems_without_raising():
    assert lexing.lex_validate_source("let x = 1;") == []
    diagnostics = lexing.lex_validate_source('let x = "oops\nlet y = $;')

    assert diagnostics == [
        {
            "code": "E001",
            "message": "unterminated string",
            "line": 1,
            "col": 9,
            "char_span": [8, 13],
            "kind": "STRING",
        },
        {
            "code": "E001",
            "message": "unexpected character '$'",
            "line": 2,
            "col": 9,
            "char_span": [22, 23],
            "kind": "UNKNOWN",
        },
    ]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_validate_source(12)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_keyword_boundary_detects_inside_edge_and_non_keyword_offsets():
    assert lexing.lex_keyword_boundary("if (x) {}", 1) == {
        "offset": 1,
        "at_boundary": False,
        "keyword": "if",
        "char_span": [0, 2],
    }
    assert lexing.lex_keyword_boundary("if (x) {}", 2) == {
        "offset": 2,
        "at_boundary": True,
        "keyword": "if",
        "char_span": [0, 2],
    }
    assert lexing.lex_keyword_boundary("if (x) {}", 5) == {
        "offset": 5,
        "at_boundary": False,
        "keyword": None,
        "char_span": [5, 5],
    }
    for bad in (-1, 10):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_keyword_boundary("if (x) {}", bad)
        assert error.value.code == "E003"


def test_lex_char_class_classifies_each_category():
    assert lexing.lex_char_class("5") == "digit"
    assert lexing.lex_char_class("é") == "alpha"
    assert lexing.lex_char_class("½") == "alnum"
    assert lexing.lex_char_class("\n") == "newline"
    assert lexing.lex_char_class(" ") == "space"
    assert lexing.lex_char_class("+") == "punct"
    assert lexing.lex_char_class("€") == "other"
    for bad in ("", "ab"):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_char_class(bad)
        assert error.value.code == "E001"


def test_lex_escape_table_returns_a_fresh_copy_each_call():
    table = lexing.lex_escape_table()
    assert table == {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "0": "\0",
        "\\": "\\",
        '"': '"',
        "'": "'",
    }
    table["n"] = "changed"
    assert lexing.lex_escape_table()["n"] == "\n"


def test_lex_measure_token_widths_counts_decoded_chars_and_utf8_bytes():
    assert lexing.lex_measure_token_widths('let café = "é";') == [
        {"type": "LET", "value": "let", "chars": 3, "bytes": 3, "char_span": [0, 3]},
        {
            "type": "IDENT",
            "value": "café",
            "chars": 4,
            "bytes": 5,
            "char_span": [4, 8],
        },
        {"type": "EQ", "value": "=", "chars": 1, "bytes": 1, "char_span": [9, 10]},
        {
            "type": "STRING",
            "value": "é",
            "chars": 1,
            "bytes": 2,
            "char_span": [11, 14],
        },
        {"type": "SEMI", "value": ";", "chars": 1, "bytes": 1, "char_span": [14, 15]},
        {"type": "EOF", "value": "", "chars": 0, "bytes": 0, "char_span": [15, 15]},
    ]
    with pytest.raises(PolyglotError) as error:
        lexing.lex_measure_token_widths([])  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_lex_tokenize_with_offsets_shifts_lines_and_columns_including_eof():
    records = lexing.lex_tokenize_with_offsets("let x = 1;", 10, 5)

    assert records[0] == {
        "type": "LET",
        "value": "let",
        "line": 11,
        "col": 6,
        "char_span": [0, 3],
        "byte_span": [0, 3],
        "original_line": 1,
        "original_col": 1,
    }
    assert records[1]["col"] == 10
    assert records[1]["original_col"] == 5
    assert records[-1]["type"] == "EOF"
    assert records[-1]["original_line"] == 1
    assert records[-1]["char_span"] == [10, 10]
    for bad in (0, -1):
        with pytest.raises(PolyglotError) as error:
            lexing.lex_tokenize_with_offsets("x", bad, 1)
        assert error.value.code == "E003"
    with pytest.raises(PolyglotError) as error:
        lexing.lex_tokenize_with_offsets("x", 1, 0)
    assert error.value.code == "E003"
