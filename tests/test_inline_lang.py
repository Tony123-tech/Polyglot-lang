import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from inline_lang import InlineLangError, compile_inline_source, inline_lang


def _load_inline(tmp_path: Path, body: str, name: str = "inline_case") -> ModuleType:
    source = "from inline_lang import inline_lang\n\n" + textwrap.dedent(body)
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_single_fn_returns_callable(tmp_path):
    module = _load_inline(
        tmp_path,
        """
        @inline_lang("rust", trusted=True)
        def add(a: i64, b: i64) -> i64:
            return a + b
        """,
    )
    assert callable(module.add)
    assert module.add(2, 3) == 5


def test_multiple_items_returns_namespace():
    namespace = compile_inline_source(
        """
        struct Point {
            x: i64,
            y: i64,
        }

        impl Point {
            fn new(x: i64, y: i64) -> Point {
                Point { x: x, y: y }
            }
        }
        """,
        trusted=True,
    )
    assert isinstance(namespace, dict)
    assert "Point" in namespace


def test_unsupported_language_rejected():
    with pytest.raises(InlineLangError):
        inline_lang("python")(lambda: None)


def test_compile_error_wrapped():
    with pytest.raises(InlineLangError) as excinfo:
        compile_inline_source("fn broken(a: i64) -> i64 { a + }", trusted=True)
    assert excinfo.value.__cause__ is not None


def test_uninspectable_source_rejected():
    with pytest.raises(InlineLangError):
        inline_lang("rust", trusted=True)(eval("lambda: None"))


def test_compile_inline_source_error_maps_to_python_line():
    source = textwrap.dedent(
        """
        fn bad(a: i64) -> i64 {
            a +
        }
        """
    ).strip()
    with pytest.raises(InlineLangError) as excinfo:
        compile_inline_source(source, trusted=True, line_offset=10)
    error = excinfo.value
    assert error.inline_line is not None
    assert error.inline_line >= 11
    assert isinstance(error.__cause__, Exception)


def test_python_closure_supported(tmp_path):
    module = _load_inline(
        tmp_path,
        """
        offset = 10

        @inline_lang("rust", trusted=True)
        def add_offset(a: i64) -> i64:
            return a + offset
        """,
    )
    assert module.add_offset(5) == 15


def test_native_closure_rejected(tmp_path):
    with pytest.raises(InlineLangError):
        _load_inline(
            tmp_path,
            """
            offset = 10

            @inline_lang("rust", backend="native", trusted=True)
            def add_offset(a: i64) -> i64:
                return a + offset
            """,
        )
