from __future__ import annotations

import pytest

from backends.python.translator import PythonBackend
from backends.rust.compiler import NativeBackend
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError


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
