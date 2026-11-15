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
    "string_replacen": """
fn string_replacen(value: String) -> String {
    value.replacen("a", "z", 2)
}
""",
    "string_split_at": """
fn string_split_at(value: String) -> (String, String) {
    value.split_at(2)
}
""",
    "string_split_terminator": """
fn string_split_terminator(value: String) -> Vec<String> {
    value.split_terminator(",")
}
""",
    "string_rsplit_terminator": """
fn string_rsplit_terminator(value: String) -> Vec<String> {
    value.rsplit_terminator(",")
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
    "char_is_alphabetic": """
fn char_is_alphabetic(value: char) -> bool {
    value.is_alphabetic()
}
""",
    "char_is_numeric": """
fn char_is_numeric(value: char) -> bool {
    value.is_numeric()
}
""",
    "char_is_alphanumeric": """
fn char_is_alphanumeric(value: char) -> bool {
    value.is_alphanumeric()
}
""",
    "char_is_whitespace": """
fn char_is_whitespace(value: char) -> bool {
    value.is_whitespace()
}
""",
    "char_is_uppercase": """
fn char_is_uppercase(value: char) -> bool {
    value.is_uppercase()
}
""",
    "char_is_lowercase": """
fn char_is_lowercase(value: char) -> bool {
    value.is_lowercase()
}
""",
    "char_to_uppercase": """
fn char_to_uppercase(value: char) -> String {
    value.to_uppercase().to_string()
}
""",
    "char_to_lowercase": """
fn char_to_lowercase(value: char) -> String {
    value.to_lowercase().to_string()
}
""",
    "char_to_digit": """
fn char_to_digit(value: char) -> i64 {
    value.to_digit(16).unwrap_or(0) as i64
}
fn char_to_digit_some(value: char) -> bool {
    value.to_digit(16).is_some()
}
""",
    "char_is_ascii_alphabetic": """
fn char_is_ascii_alphabetic(value: char) -> bool {
    value.is_ascii_alphabetic()
}
""",
    "char_is_ascii_digit": """
fn char_is_ascii_digit(value: char) -> bool {
    value.is_ascii_digit()
}
""",
    "char_is_ascii_alphanumeric": """
fn char_is_ascii_alphanumeric(value: char) -> bool {
    value.is_ascii_alphanumeric()
}
""",
    "char_is_ascii_whitespace": """
fn char_is_ascii_whitespace(value: char) -> bool {
    value.is_ascii_whitespace()
}
""",
    "char_is_ascii_uppercase": """
fn char_is_ascii_uppercase(value: char) -> bool {
    value.is_ascii_uppercase()
}
""",
    "char_is_ascii_lowercase": """
fn char_is_ascii_lowercase(value: char) -> bool {
    value.is_ascii_lowercase()
}
""",
    "char_is_ascii": """
fn char_is_ascii(value: char) -> bool {
    value.is_ascii()
}
""",
    "char_is_ascii_punctuation": """
fn char_is_ascii_punctuation(value: char) -> bool {
    value.is_ascii_punctuation()
}
""",
    "char_len_utf8": """
fn char_len_utf8(value: char) -> usize {
    value.len_utf8()
}
""",
    "char_len_utf16": """
fn char_len_utf16(value: char) -> usize {
    value.len_utf16()
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
def test_native_string_replacen(native_ns):
    assert native_ns["string_replacen"]("banana") == "bznzna"
    assert native_ns["string_replacen"]("xyz") == "xyz"


def test_python_string_replacen_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_replacen"])))

    assert namespace["string_replacen"]("banana") == "bznzna"
    assert namespace["string_replacen"]("xyz") == "xyz"


@pytest.mark.native
def test_native_string_split_at(native_ns):
    assert native_ns["string_split_at"]("banana") == ("ba", "nana")
    assert native_ns["string_split_at"]("ab") == ("ab", "")


def test_python_string_split_at_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_split_at"])))

    assert namespace["string_split_at"]("banana") == ("ba", "nana")
    assert namespace["string_split_at"]("ab") == ("ab", "")


@pytest.mark.native
def test_native_string_split_terminator(native_ns):
    assert native_ns["string_split_terminator"]("a,b,") == ["a", "b"]
    assert native_ns["string_split_terminator"]("a,b") == ["a", "b"]


def test_python_string_split_terminator_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_split_terminator"])))

    assert namespace["string_split_terminator"]("a,b,") == ["a", "b"]
    assert namespace["string_split_terminator"]("a,b") == ["a", "b"]


@pytest.mark.native
def test_native_string_rsplit_terminator(native_ns):
    assert native_ns["string_rsplit_terminator"]("a,b,") == ["b", "a"]
    assert native_ns["string_rsplit_terminator"]("a,b") == ["b", "a"]


def test_python_string_rsplit_terminator_parity():
    namespace = compile_rust(parse(lex(SOURCES["string_rsplit_terminator"])))

    assert namespace["string_rsplit_terminator"]("a,b,") == ["b", "a"]
    assert namespace["string_rsplit_terminator"]("a,b") == ["b", "a"]


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


@pytest.mark.native
def test_native_char_is_alphabetic(native_ns):
    assert native_ns["char_is_alphabetic"]("a") is True
    assert native_ns["char_is_alphabetic"]("é") is True
    assert native_ns["char_is_alphabetic"]("1") is False


def test_python_char_is_alphabetic_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_alphabetic"])))

    assert namespace["char_is_alphabetic"]("a") is True
    assert namespace["char_is_alphabetic"]("é") is True
    assert namespace["char_is_alphabetic"]("1") is False


@pytest.mark.native
def test_native_char_is_numeric(native_ns):
    assert native_ns["char_is_numeric"]("1") is True
    assert native_ns["char_is_numeric"]("٣") is True
    assert native_ns["char_is_numeric"]("a") is False


def test_python_char_is_numeric_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_numeric"])))

    assert namespace["char_is_numeric"]("1") is True
    assert namespace["char_is_numeric"]("٣") is True
    assert namespace["char_is_numeric"]("a") is False


@pytest.mark.native
def test_native_char_is_alphanumeric(native_ns):
    assert native_ns["char_is_alphanumeric"]("a") is True
    assert native_ns["char_is_alphanumeric"]("1") is True
    assert native_ns["char_is_alphanumeric"]("é") is True
    assert native_ns["char_is_alphanumeric"]("_") is False


def test_python_char_is_alphanumeric_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_alphanumeric"])))

    assert namespace["char_is_alphanumeric"]("a") is True
    assert namespace["char_is_alphanumeric"]("1") is True
    assert namespace["char_is_alphanumeric"]("é") is True
    assert namespace["char_is_alphanumeric"]("_") is False


@pytest.mark.native
def test_native_char_is_whitespace(native_ns):
    assert native_ns["char_is_whitespace"](" ") is True
    assert native_ns["char_is_whitespace"]("\t") is True
    assert native_ns["char_is_whitespace"]("a") is False


def test_python_char_is_whitespace_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_whitespace"])))

    assert namespace["char_is_whitespace"](" ") is True
    assert namespace["char_is_whitespace"]("\t") is True
    assert namespace["char_is_whitespace"]("a") is False


@pytest.mark.native
def test_native_char_is_uppercase(native_ns):
    assert native_ns["char_is_uppercase"]("A") is True
    assert native_ns["char_is_uppercase"]("É") is True
    assert native_ns["char_is_uppercase"]("a") is False
    assert native_ns["char_is_uppercase"]("1") is False


def test_python_char_is_uppercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_uppercase"])))

    assert namespace["char_is_uppercase"]("A") is True
    assert namespace["char_is_uppercase"]("É") is True
    assert namespace["char_is_uppercase"]("a") is False
    assert namespace["char_is_uppercase"]("1") is False


@pytest.mark.native
def test_native_char_is_lowercase(native_ns):
    assert native_ns["char_is_lowercase"]("a") is True
    assert native_ns["char_is_lowercase"]("é") is True
    assert native_ns["char_is_lowercase"]("A") is False
    assert native_ns["char_is_lowercase"]("1") is False


def test_python_char_is_lowercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_lowercase"])))

    assert namespace["char_is_lowercase"]("a") is True
    assert namespace["char_is_lowercase"]("é") is True
    assert namespace["char_is_lowercase"]("A") is False
    assert namespace["char_is_lowercase"]("1") is False


@pytest.mark.native
def test_native_char_to_uppercase(native_ns):
    assert native_ns["char_to_uppercase"]("a") == "A"
    assert native_ns["char_to_uppercase"]("ß") == "SS"
    assert native_ns["char_to_uppercase"]("1") == "1"


def test_python_char_to_uppercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_to_uppercase"])))

    assert namespace["char_to_uppercase"]("a") == "A"
    assert namespace["char_to_uppercase"]("ß") == "SS"
    assert namespace["char_to_uppercase"]("1") == "1"


@pytest.mark.native
def test_native_char_to_lowercase(native_ns):
    assert native_ns["char_to_lowercase"]("A") == "a"
    assert native_ns["char_to_lowercase"]("É") == "é"
    assert native_ns["char_to_lowercase"]("1") == "1"


def test_python_char_to_lowercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_to_lowercase"])))

    assert namespace["char_to_lowercase"]("A") == "a"
    assert namespace["char_to_lowercase"]("É") == "é"
    assert namespace["char_to_lowercase"]("1") == "1"


@pytest.mark.native
def test_native_char_to_digit(native_ns):
    assert native_ns["char_to_digit"]("f") == 15
    assert native_ns["char_to_digit"]("F") == 15
    assert native_ns["char_to_digit"]("g") == 0
    assert native_ns["char_to_digit_some"]("a") is True
    assert native_ns["char_to_digit_some"]("g") is False


def test_python_char_to_digit_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_to_digit"])))

    assert namespace["char_to_digit"]("f") == 15
    assert namespace["char_to_digit"]("F") == 15
    assert namespace["char_to_digit"]("g") == 0
    assert namespace["char_to_digit_some"]("a") is True
    assert namespace["char_to_digit_some"]("g") is False


@pytest.mark.native
def test_native_char_is_ascii_alphabetic(native_ns):
    assert native_ns["char_is_ascii_alphabetic"]("a") is True
    assert native_ns["char_is_ascii_alphabetic"]("Z") is True
    assert native_ns["char_is_ascii_alphabetic"]("é") is False
    assert native_ns["char_is_ascii_alphabetic"]("1") is False


def test_python_char_is_ascii_alphabetic_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_alphabetic"])))

    assert namespace["char_is_ascii_alphabetic"]("a") is True
    assert namespace["char_is_ascii_alphabetic"]("Z") is True
    assert namespace["char_is_ascii_alphabetic"]("é") is False
    assert namespace["char_is_ascii_alphabetic"]("1") is False


@pytest.mark.native
def test_native_char_is_ascii_digit(native_ns):
    assert native_ns["char_is_ascii_digit"]("0") is True
    assert native_ns["char_is_ascii_digit"]("9") is True
    assert native_ns["char_is_ascii_digit"]("a") is False
    assert native_ns["char_is_ascii_digit"]("٣") is False


def test_python_char_is_ascii_digit_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_digit"])))

    assert namespace["char_is_ascii_digit"]("0") is True
    assert namespace["char_is_ascii_digit"]("9") is True
    assert namespace["char_is_ascii_digit"]("a") is False
    assert namespace["char_is_ascii_digit"]("٣") is False


@pytest.mark.native
def test_native_char_is_ascii_alphanumeric(native_ns):
    assert native_ns["char_is_ascii_alphanumeric"]("a") is True
    assert native_ns["char_is_ascii_alphanumeric"]("Z") is True
    assert native_ns["char_is_ascii_alphanumeric"]("0") is True
    assert native_ns["char_is_ascii_alphanumeric"]("é") is False
    assert native_ns["char_is_ascii_alphanumeric"]("_") is False


def test_python_char_is_ascii_alphanumeric_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_alphanumeric"])))

    assert namespace["char_is_ascii_alphanumeric"]("a") is True
    assert namespace["char_is_ascii_alphanumeric"]("Z") is True
    assert namespace["char_is_ascii_alphanumeric"]("0") is True
    assert namespace["char_is_ascii_alphanumeric"]("é") is False
    assert namespace["char_is_ascii_alphanumeric"]("_") is False


@pytest.mark.native
def test_native_char_is_ascii_whitespace(native_ns):
    assert native_ns["char_is_ascii_whitespace"](" ") is True
    assert native_ns["char_is_ascii_whitespace"]("\t") is True
    assert native_ns["char_is_ascii_whitespace"]("\n") is True
    assert native_ns["char_is_ascii_whitespace"]("\x0b") is False
    assert native_ns["char_is_ascii_whitespace"]("a") is False


def test_python_char_is_ascii_whitespace_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_whitespace"])))

    assert namespace["char_is_ascii_whitespace"](" ") is True
    assert namespace["char_is_ascii_whitespace"]("\t") is True
    assert namespace["char_is_ascii_whitespace"]("\n") is True
    assert namespace["char_is_ascii_whitespace"]("\x0b") is False
    assert namespace["char_is_ascii_whitespace"]("a") is False


@pytest.mark.native
def test_native_char_is_ascii_uppercase(native_ns):
    assert native_ns["char_is_ascii_uppercase"]("A") is True
    assert native_ns["char_is_ascii_uppercase"]("Z") is True
    assert native_ns["char_is_ascii_uppercase"]("a") is False
    assert native_ns["char_is_ascii_uppercase"]("É") is False


def test_python_char_is_ascii_uppercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_uppercase"])))

    assert namespace["char_is_ascii_uppercase"]("A") is True
    assert namespace["char_is_ascii_uppercase"]("Z") is True
    assert namespace["char_is_ascii_uppercase"]("a") is False
    assert namespace["char_is_ascii_uppercase"]("É") is False


@pytest.mark.native
def test_native_char_is_ascii_lowercase(native_ns):
    assert native_ns["char_is_ascii_lowercase"]("a") is True
    assert native_ns["char_is_ascii_lowercase"]("z") is True
    assert native_ns["char_is_ascii_lowercase"]("A") is False
    assert native_ns["char_is_ascii_lowercase"]("é") is False


def test_python_char_is_ascii_lowercase_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_lowercase"])))

    assert namespace["char_is_ascii_lowercase"]("a") is True
    assert namespace["char_is_ascii_lowercase"]("z") is True
    assert namespace["char_is_ascii_lowercase"]("A") is False
    assert namespace["char_is_ascii_lowercase"]("é") is False


@pytest.mark.native
def test_native_char_is_ascii(native_ns):
    assert native_ns["char_is_ascii"]("a") is True
    assert native_ns["char_is_ascii"]("~") is True
    assert native_ns["char_is_ascii"]("é") is False


def test_python_char_is_ascii_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii"])))

    assert namespace["char_is_ascii"]("a") is True
    assert namespace["char_is_ascii"]("~") is True
    assert namespace["char_is_ascii"]("é") is False


@pytest.mark.native
def test_native_char_is_ascii_punctuation(native_ns):
    assert native_ns["char_is_ascii_punctuation"]("!") is True
    assert native_ns["char_is_ascii_punctuation"](".") is True
    assert native_ns["char_is_ascii_punctuation"]("`") is True
    assert native_ns["char_is_ascii_punctuation"]("a") is False
    assert native_ns["char_is_ascii_punctuation"](" ") is False
    assert native_ns["char_is_ascii_punctuation"]("é") is False


def test_python_char_is_ascii_punctuation_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_is_ascii_punctuation"])))

    assert namespace["char_is_ascii_punctuation"]("!") is True
    assert namespace["char_is_ascii_punctuation"](".") is True
    assert namespace["char_is_ascii_punctuation"]("`") is True
    assert namespace["char_is_ascii_punctuation"]("a") is False
    assert namespace["char_is_ascii_punctuation"](" ") is False
    assert namespace["char_is_ascii_punctuation"]("é") is False


@pytest.mark.native
def test_native_char_len_utf8(native_ns):
    assert native_ns["char_len_utf8"]("a") == 1
    assert native_ns["char_len_utf8"]("é") == 2
    assert native_ns["char_len_utf8"]("€") == 3
    assert native_ns["char_len_utf8"]("𝄞") == 4


def test_python_char_len_utf8_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_len_utf8"])))

    assert namespace["char_len_utf8"]("a") == 1
    assert namespace["char_len_utf8"]("é") == 2
    assert namespace["char_len_utf8"]("€") == 3
    assert namespace["char_len_utf8"]("𝄞") == 4


@pytest.mark.native
def test_native_char_len_utf16(native_ns):
    assert native_ns["char_len_utf16"]("a") == 1
    assert native_ns["char_len_utf16"]("€") == 1
    assert native_ns["char_len_utf16"]("𝄞") == 2


def test_python_char_len_utf16_parity():
    namespace = compile_rust(parse(lex(SOURCES["char_len_utf16"])))

    assert namespace["char_len_utf16"]("a") == 1
    assert namespace["char_len_utf16"]("€") == 1
    assert namespace["char_len_utf16"]("𝄞") == 2
