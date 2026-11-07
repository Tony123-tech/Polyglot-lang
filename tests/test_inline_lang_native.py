import asyncio
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from inline_lang import InlineLangError


def _load_inline(tmp_path: Path, body: str, name: str = "inline_native") -> ModuleType:
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


@pytest.mark.slow
def test_native_inline_sync_fn(tmp_path):
    module = _load_inline(
        tmp_path,
        """
        @inline_lang("rust", backend="native", trusted=True)
        def add(a: i64, b: i64) -> i64:
            return a + b
        """,
    )
    assert module.add(2, 3) == 5


@pytest.mark.slow
def test_native_inline_async_fn(tmp_path):
    module = _load_inline(
        tmp_path,
        """
        @inline_lang("rust", backend="native", trusted=True)
        async def add(a: i64, b: i64) -> i64:
            a + b
        """,
    )
    assert asyncio.run(module.add(2, 3)) == 5


@pytest.mark.slow
def test_native_inline_closure_rejected(tmp_path):
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
