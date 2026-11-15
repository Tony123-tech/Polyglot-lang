"""N4 native and Python string tests."""

import pytest

from backends.python.translator import compile_rust
from lexer import lex
from parser import parse

SOURCES = {
    "string_len": """
fn string_len(value: String) -> i64 {
    value.len() as i64
}
""",
    "string_chars": """
fn string_chars(value: String) -> Vec<char> {
    value.chars()
}
""",
    "string_chars_rev": """
fn string_chars_rev(value: String) -> Vec<char> {
    value.chars().rev()
}
""",
    "string_char_indices": """
fn string_char_indices(value: String) -> Vec<(i64, char)> {
    value.char_indices()
}
""",
    "string_trim_matches": """
fn string_trim_matches(value: String) -> String {
    value.trim_matches(|c| c == 'x')
}
""",
    "string_trim_matches_side": """
fn string_trim_start_matches(value: String) -> String {
    value.trim_start_matches(|c| c == 'x')
}
fn string_trim_end_matches(value: String) -> String {
    value.trim_end_matches(|c| c == 'x')
}
""",
    "string_bytes": """
fn string_bytes(value: String) -> Vec<u8> {
    value.bytes()
}
""",
    "string_split": """
fn string_split(value: String) -> Vec<String> {
    value.split(",")
}
""",
    "string_splitn": """
fn string_splitn(value: String) -> Vec<String> {
    value.splitn(2, ",")
}
fn string_rsplitn(value: String) -> Vec<String> {
    value.rsplitn(2, ",")
}
""",
    "string_rsplit": """
fn string_rsplit(value: String) -> Vec<String> {
    value.rsplit(",")
}
""",
    "string_matches": """
fn string_matches(value: String) -> Vec<String> {
    value.matches("ab")
}
fn string_rmatches(value: String) -> Vec<String> {
    value.rmatches("ab")
}
""",
    "string_join": """
fn string_join(values: Vec<String>, separator: String) -> String {
    values.join(separator)
}
""",
    "string_replace": """
fn string_replace(value: String) -> String {
    value.replace("a", "x")
}
""",
    "string_trim": """
fn string_trim(value: String) -> String {
    value.trim()
}
""",
    "string_starts_with": """
fn string_starts_with(value: String, prefix: String) -> bool {
    value.starts_with(prefix)
}
""",
    "string_ends_with": """
fn string_ends_with(value: String, suffix: String) -> bool {
    value.ends_with(suffix)
}
""",
    "string_to_uppercase": """
fn string_to_uppercase(value: String) -> String {
    value.to_uppercase()
}
""",
    "string_to_lowercase": """
fn string_to_lowercase(value: String) -> String {
    value.to_lowercase()
}
""",
    "string_parse_i64": """
fn string_parse_i64(value: String) -> i64 {
    value.parse::<i64>()
}
""",
    "string_format_multi": """
fn string_format_multi(name: String, count: i64) -> String {
    format!("{}:{}", name, count)
}
""",
    "string_write": """
fn string_write(value: i64, suffix: String) -> String {
    let mut output: String = format!("");
    write!(&mut output, "{}-{}", value, suffix);
    output
}
""",
    "string_concat": """
fn string_concat(left: String, right: String) -> String {
    left + right
}
""",
    "string_slice": """
fn string_slice(value: String) -> String {
    &value[1..5]
}
""",
    "string_before": """
fn string_before(left: String, right: String) -> bool {
    left < right
}
""",
}


@pytest.mark.native
def test_native_string_len_chars_bytes(native_ns):
    assert native_ns["string_len"]("café") == 5
    assert native_ns["string_chars"]("café") == ["c", "a", "f", "é"]
    assert native_ns["string_bytes"]("café") == b"caf\xc3\xa9"


@pytest.mark.native
def test_native_string_chars_rev(native_ns):
    assert native_ns["string_chars_rev"]("café") == ["é", "f", "a", "c"]


def test_python_string_chars_rev():
    namespace = compile_rust(parse(lex(SOURCES["string_chars_rev"])))

    assert namespace["string_chars_rev"]("café") == ["é", "f", "a", "c"]


@pytest.mark.native
def test_native_string_char_indices(native_ns):
    assert native_ns["string_char_indices"]("café") == [
        (0, "c"),
        (1, "a"),
        (2, "f"),
        (3, "é"),
    ]


def test_python_string_char_indices():
    namespace = compile_rust(parse(lex(SOURCES["string_char_indices"])))

    assert namespace["string_char_indices"]("café") == [
        (0, "c"),
        (1, "a"),
        (2, "f"),
        (3, "é"),
    ]


@pytest.mark.native
def test_native_string_trim_matches(native_ns):
    assert native_ns["string_trim_matches"]("xxabcxx") == "abc"
    assert native_ns["string_trim_matches"]("abc") == "abc"
    assert native_ns["string_trim_matches"]("xxxx") == ""


def test_python_string_trim_matches():
    namespace = compile_rust(parse(lex(SOURCES["string_trim_matches"])))

    assert namespace["string_trim_matches"]("xxabcxx") == "abc"
    assert namespace["string_trim_matches"]("abc") == "abc"
    assert namespace["string_trim_matches"]("xxxx") == ""


@pytest.mark.native
def test_native_string_trim_matches_side(native_ns):
    assert native_ns["string_trim_start_matches"]("xxabcxx") == "abcxx"
    assert native_ns["string_trim_end_matches"]("xxabcxx") == "xxabc"
    assert native_ns["string_trim_start_matches"]("abc") == "abc"


def test_python_string_trim_matches_side_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_trim_matches_side"])))

    assert namespace["string_trim_start_matches"]("xxabcxx") == "abcxx"
    assert namespace["string_trim_end_matches"]("xxabcxx") == "xxabc"
    assert namespace["string_trim_start_matches"]("abc") == "abc"


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("string_len", "café", 5),
        ("string_chars", "café", ["c", "a", "f", "é"]),
        ("string_bytes", "café", b"caf\xc3\xa9"),
    ],
)
def test_python_string_len_chars_bytes_parity(name, value, expected):
    namespace = compile_rust(parse(lex(SOURCES[name])))

    assert namespace[name](value) == expected


@pytest.mark.native
def test_native_string_split_join(native_ns):
    assert native_ns["string_split"]("red,green,blue") == [
        "red",
        "green",
        "blue",
    ]
    assert native_ns["string_join"](["red", "green"], ",") == "red,green"


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("string_split", ("red,green,blue",), ["red", "green", "blue"]),
        ("string_join", (["red", "green"], ","), "red,green"),
    ],
)
def test_python_string_split_join_parity(name, args, expected):
    namespace = compile_rust(parse(lex(SOURCES[name])))

    assert namespace[name](*args) == expected


@pytest.mark.native
def test_native_string_splitn(native_ns):
    assert native_ns["string_splitn"]("a,b,c") == ["a", "b,c"]
    assert native_ns["string_rsplitn"]("a,b,c") == ["c", "a,b"]


def test_python_string_splitn_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_splitn"])))

    assert namespace["string_splitn"]("a,b,c") == ["a", "b,c"]
    assert namespace["string_rsplitn"]("a,b,c") == ["c", "a,b"]


@pytest.mark.native
def test_native_string_rsplit(native_ns):
    assert native_ns["string_rsplit"]("a,b,c") == ["c", "b", "a"]
    assert native_ns["string_rsplit"]("solo") == ["solo"]


def test_python_string_rsplit_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_rsplit"])))

    assert namespace["string_rsplit"]("a,b,c") == ["c", "b", "a"]
    assert namespace["string_rsplit"]("solo") == ["solo"]


@pytest.mark.native
def test_native_string_matches(native_ns):
    assert native_ns["string_matches"]("abxabab") == ["ab", "ab", "ab"]
    assert native_ns["string_matches"]("none") == []
    assert native_ns["string_rmatches"]("abxabab") == ["ab", "ab", "ab"]


def test_python_string_matches_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_matches"])))

    assert namespace["string_matches"]("abxabab") == ["ab", "ab", "ab"]
    assert namespace["string_matches"]("none") == []
    assert namespace["string_rmatches"]("abxabab") == ["ab", "ab", "ab"]


@pytest.mark.native
def test_native_string_replace_trim_prefix_suffix(native_ns):
    assert native_ns["string_replace"]("banana") == "bxnxnx"
    assert native_ns["string_trim"]("  text \n") == "text"
    assert native_ns["string_starts_with"]("polyglot", "poly") is True
    assert native_ns["string_ends_with"]("polyglot", "glot") is True


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("string_replace", ("banana",), "bxnxnx"),
        ("string_trim", ("  text \n",), "text"),
        ("string_starts_with", ("polyglot", "poly"), True),
        ("string_ends_with", ("polyglot", "glot"), True),
    ],
)
def test_python_string_replace_trim_prefix_suffix_parity(
    name,
    args,
    expected,
):
    namespace = compile_rust(parse(lex(SOURCES[name])))

    assert namespace[name](*args) == expected


@pytest.mark.native
def test_native_string_case_conversion(native_ns):
    assert native_ns["string_to_uppercase"]("Straße") == "STRASSE"
    assert native_ns["string_to_lowercase"]("POLYGLOT") == "polyglot"


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("string_to_uppercase", "Straße", "STRASSE"),
        ("string_to_lowercase", "POLYGLOT", "polyglot"),
    ],
)
def test_python_string_case_conversion_parity(name, value, expected):
    namespace = compile_rust(parse(lex(SOURCES[name])))

    assert namespace[name](value) == expected


@pytest.mark.native
def test_native_string_parse_i64(native_ns):
    assert native_ns["string_parse_i64"]("42") == 42


def test_python_string_parse_i64_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_parse_i64"])))

    assert namespace["string_parse_i64"]("42") == 42


@pytest.mark.native
def test_native_format_multiple_arguments(native_ns):
    assert native_ns["string_format_multi"]("items", 3) == "items:3"


def test_python_format_multiple_arguments_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_format_multi"])))

    assert namespace["string_format_multi"]("items", 3) == "items:3"


@pytest.mark.native
def test_native_write_macro_into_string(native_ns):
    assert native_ns["string_write"](42, "ok") == "42-ok"


def test_python_write_macro_into_string_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_write"])))

    assert namespace["string_write"](42, "ok") == "42-ok"


@pytest.mark.native
def test_native_string_concatenation(native_ns):
    assert native_ns["string_concat"]("poly", "glot") == "polyglot"


def test_python_string_concatenation_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_concat"])))

    assert namespace["string_concat"]("poly", "glot") == "polyglot"


@pytest.mark.native
def test_native_string_slice_uses_utf8_byte_offsets(native_ns):
    assert native_ns["string_slice"]("café") == "afé"


def test_python_string_slice_uses_utf8_byte_offsets():
    namespace = compile_rust(parse(lex(SOURCES["string_slice"])))

    assert namespace["string_slice"]("café") == "afé"


@pytest.mark.native
def test_native_string_comparison_and_ordering(native_ns):
    assert native_ns["string_before"]("alpha", "beta") is True
    assert native_ns["string_before"]("beta", "alpha") is False


def test_python_string_comparison_and_ordering_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_before"])))

    assert namespace["string_before"]("alpha", "beta") is True
    assert namespace["string_before"]("beta", "alpha") is False
