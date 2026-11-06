"""N4 native and Python string tests."""

import pytest

from backends.python.translator import compile_rust
from lexer import lex
from parser import parse

SOURCES = {
    "string_len": """
fn string_len(value: String) -> i64 {
    value.len()
}
""",
    "string_chars": """
fn string_chars(value: String) -> Vec<char> {
    value.chars()
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
        "red", "green", "blue",
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
    name, args, expected,
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
