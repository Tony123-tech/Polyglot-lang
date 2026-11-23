from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from compiler.core.ast_nodes import FnDecl, Node, Program
from backends.python.translator import PythonBackend
from backends.rust.compiler import NativeBackend
from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import PillarError
from runtime.feature_gates import _walk, validate_program_features


@pytest.mark.parametrize("attribute", ["gpu", "npu"])
def test_accelerator_attributes_are_rejected_identically(attribute, tmp_path):
    source = f"#[{attribute}]\nfn accelerate(x: i64) -> i64 {{ x }}\n"
    program = parse(lex(source))

    with pytest.raises(PillarError) as python_error:
        PythonBackend().compile(program)
    with pytest.raises(PillarError) as native_error:
        NativeBackend(cache_dir=tmp_path).compile(program)

    expected_span = (0, len(f"#[{attribute}]"))
    assert python_error.value.code == "E007"
    assert python_error.value.span == expected_span
    assert str(python_error.value) == str(native_error.value)
    assert "GPU/NPU kernel offloading is not implemented" in str(python_error.value)


def test_validation_uses_the_attribute_location_not_the_function_location():
    program = parse(lex("#[npu]\nfn offload(x: i64) -> i64 { x }\n"))
    with pytest.raises(PillarError) as excinfo:
        validate_program_features(program)
    error = excinfo.value
    assert (error.line, error.col, error.span) == (1, 1, (0, 6))
    assert error.format().splitlines()[0] == (
        "error[E007]: the `#[npu]` accelerator feature is not supported"
    )


@dataclass
class _Container(Node):
    """A synthetic Node with containers that can hold arbitrary children."""

    children: list[Node] = field(default_factory=list)
    lookup: dict[str, Node] = field(default_factory=dict)


def test_walk_handles_cycles_and_shared_subtrees():
    root = _Container(1, 1)
    child = _Container(2, 2)
    root.children.extend([child, child, root])
    child.lookup["back"] = root

    assert list(_walk(root)) == [root, child]


def test_walk_traverses_mapping_values_and_ignores_scalars():
    target = _Container(5, 5)
    root = _Container(1, 1, children=[None, "text", 7, {"key": target}])

    assert list(_walk(root)) == [root, target]


def test_walk_ignores_source_lines_and_other_none_fields():
    program = parse(lex("fn main() -> i64 { 0 }"))
    program.source_lines["not a node"] = "plain text"
    program.span = None

    assert any(isinstance(node, FnDecl) for node in _walk(program))
    validate_program_features(program)  # no accelerator attributes: no error


def test_nested_functions_with_accelerator_attributes_are_still_found():
    source = "fn outer() -> i64 {\n    #[gpu]\n    fn inner() -> i64 { 0 }\n    0\n}\n"
    program: Program = parse(lex(source))
    with pytest.raises(PillarError) as excinfo:
        validate_program_features(program)
    assert excinfo.value.code == "E007"
    assert "gpu" in str(excinfo.value)
