from __future__ import annotations

from typing import Any

from backends.base import BackendBase
from backends.rust.compiler import NativeBackend
from compiler.core.ast_nodes import Binary, FnDecl, Program, TypeName
from zig.backend import ZigBackend


class AutoBackend(BackendBase):
    name = "auto"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.python_backend = NativeBackend(**kwargs)
        self.native_backend = NativeBackend(**kwargs)
        self.zig_backend = ZigBackend(**kwargs)

    def analyze_ast(self, program: Program) -> str:
        has_simd = False
        has_zig_saturating = False
        has_parallel = False

        def walk(node: Any) -> None:
            nonlocal has_simd, has_zig_saturating, has_parallel
            if isinstance(node, TypeName) and node.name.startswith("Simd"):
                has_simd = True
            elif isinstance(node, Binary):
                if node.op in {"+|", "-|", "*|"}:
                    has_zig_saturating = True
            elif isinstance(node, FnDecl) and any(
                attr.name == "parallel" for attr in node.attributes
            ):
                has_parallel = True

            if hasattr(node, "__dataclass_fields__"):
                for field in node.__dataclass_fields__:
                    val = getattr(node, field)
                    if isinstance(val, list):
                        for item in val:
                            walk(item)
                    elif hasattr(val, "__dataclass_fields__"):
                        walk(val)

        walk(program)

        if has_simd or has_zig_saturating:
            return "zig"
        if has_parallel:
            return "native"
        return "python"

    def compile(
        self, program: Program, module_name: str = "polyglot_auto"
    ) -> dict[str, Any]:
        target = self.analyze_ast(program)

        if target == "zig":
            return self.zig_backend.compile(program, module_name)
        elif target == "native":
            return self.native_backend.compile(program, module_name)
        else:
            return self.python_backend.compile(program, module_name)
