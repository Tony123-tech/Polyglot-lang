"""N0: native backend can compile and load an empty Program."""

import pytest

from ast_nodes import Program


@pytest.mark.native
def test_native_empty_module(native_backend):
    ns = native_backend.compile(Program(1, 1, []))
    assert isinstance(ns, dict)
