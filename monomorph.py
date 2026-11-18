"""AST monomorphization for Polyglot generics.

The pass turns every generic function/struct instantiation into a concrete
declaration and rewrites all references to the mangled name.  Backends never
see a type parameter afterwards: ``Box<T>`` becomes ``Box_i64`` or
``Box_String`` and ``identity<T>`` becomes ``identity_i64`` (or ``identity``
for the implicit ``i64`` default used by older programs).

Explicit instantiations come from type annotations, struct literals, and
turbofish arguments (``identity::<i64>``).  Missing arguments are inferred
from call arguments, struct fields, and the expected type of a binding.  A
depth limit stops infinitely expanding programs such as
``fn f<T>() { f::<Vec<T>>() }``.
"""

from __future__ import annotations

import copy
from typing import Any

import ast_nodes as ast
from backends.errors import TranslateError

MAX_MONOMORPH_DEPTH = 32

_DEFAULT_TYPE = "i64"
# Names that already exist in Rust's prelude or the generated runtime.  A
# default (bare) instantiation must not shadow them.
_RESERVED_TYPE_NAMES = {
    "Box",
    "BTreeMap",
    "HashMap",
    "HashSet",
    "Option",
    "Result",
    "Self",
    "String",
    "Vec",
}


class MonomorphizationError(TranslateError):
    """A generic program could not be turned into a finite concrete AST."""

    code = "E003"


def _type_name(name: str, args: list[ast.Type] | None = None) -> ast.TypeName:
    return ast.TypeName(0, 0, name, list(args or []))


def _type_key(ty: ast.Type) -> str:
    return ast.type_display_name(ty)


def _atomic_type(name: str) -> ast.TypeName:
    return _type_name(name)


def _mangle_type(ty: ast.Type) -> str:
    if isinstance(ty, ast.TypeName):
        if not ty.args:
            return ty.name
        return ty.name + "_" + "_".join(_mangle_type(arg) for arg in ty.args)
    if isinstance(ty, ast.TypeRef):
        return ("ref_mut_" if ty.mutable else "ref_") + _mangle_type(ty.inner)
    if isinstance(ty, ast.TypeSlice):
        return "slice_" + _mangle_type(ty.inner)
    if isinstance(ty, ast.TypeArray):
        return "array_" + _mangle_type(ty.inner)
    if isinstance(ty, ast.TypeTuple):
        return "tuple_" + "_".join(_mangle_type(item) for item in ty.elems)
    if isinstance(ty, ast.TypeFn):
        parts = [_mangle_type(item) for item in ty.params]
        if ty.ret is not None:
            parts.append("ret_" + _mangle_type(ty.ret))
        return "fn_" + "_".join(parts)
    if isinstance(ty, ast.TypeUnit):
        return "unit"
    if isinstance(ty, ast.TypeImplTrait):
        return "impl_" + "_".join(bound.name for bound in ty.bounds)
    if isinstance(ty, ast.TypeDynTrait):
        return "dyn_" + "_".join(bound.name for bound in ty.bounds)
    return "type"


def _mangle(name: str, args: list[ast.Type]) -> str:
    raw = name + "_" + "_".join(_mangle_type(arg) for arg in args)
    return "".join(char if char.isalnum() or char == "_" else "_" for char in raw)


def _default_args(params: list[ast.GenericParam]) -> list[ast.Type]:
    return [_atomic_type(_DEFAULT_TYPE) for _ in params]


def _types_equal(left: ast.Type | None, right: ast.Type | None) -> bool:
    if left is None or right is None:
        return left is right
    return _type_key(left) == _type_key(right)


def _base_name(ty: ast.Type | None) -> str | None:
    if isinstance(ty, ast.TypeName):
        return ty.name
    if isinstance(ty, ast.TypeRef):
        return _base_name(ty.inner)
    return None


def _substitute_type(
    ty: ast.Type | None, mapping: dict[str, ast.Type]
) -> ast.Type | None:
    """Replace type parameters in ``ty`` (structural, no instantiation)."""
    if ty is None:
        return None
    if isinstance(ty, ast.TypeName):
        if not ty.args and ty.name in mapping:
            return copy.deepcopy(mapping[ty.name])
        return ast.TypeName(
            ty.line,
            ty.col,
            ty.name,
            [_substitute_type(arg, mapping) for arg in ty.args],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeRef):
        return ast.TypeRef(
            ty.line,
            ty.col,
            _substitute_type(ty.inner, mapping),
            ty.mutable,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeSlice):
        return ast.TypeSlice(
            ty.line,
            ty.col,
            _substitute_type(ty.inner, mapping),
            span=ty.span,
        )
    if isinstance(ty, ast.TypeArray):
        return ast.TypeArray(
            ty.line,
            ty.col,
            _substitute_type(ty.inner, mapping),
            ty.size,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeTuple):
        return ast.TypeTuple(
            ty.line,
            ty.col,
            [_substitute_type(item, mapping) for item in ty.elems],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeFn):
        return ast.TypeFn(
            ty.line,
            ty.col,
            [_substitute_type(item, mapping) for item in ty.params],
            _substitute_type(ty.ret, mapping),
            span=ty.span,
        )
    return ty


def _collect_type_params(ty: ast.Type | None) -> set[str]:
    params: set[str] = set()
    if isinstance(ty, ast.TypeName):
        for arg in ty.args:
            params.update(_collect_type_params(arg))
    elif isinstance(ty, (ast.TypeRef, ast.TypeSlice, ast.TypeArray)):
        params.update(_collect_type_params(ty.inner))
    elif isinstance(ty, ast.TypeTuple):
        for item in ty.elems:
            params.update(_collect_type_params(item))
    elif isinstance(ty, ast.TypeFn):
        for item in ty.params:
            params.update(_collect_type_params(item))
        params.update(_collect_type_params(ty.ret))
    return params


class Monomorphizer:
    """Walk a ``Program`` and emit concrete declarations for every instance."""

    def __init__(self, program: ast.Program, *, max_depth: int = MAX_MONOMORPH_DEPTH):
        self.program = program
        self.max_depth = max_depth
        self.generic_fns: dict[str, ast.FnDecl] = {}
        self.generic_structs: dict[str, ast.StructDecl] = {}
        self.generic_enums: dict[str, ast.EnumDecl] = {}
        self.generic_impls: dict[str, list[ast.ImplBlock]] = {}
        self.assoc_methods: dict[str, dict[str, tuple[ast.ImplBlock, ast.FnDecl]]] = {}
        self.instance_source: dict[str, ast.TypeName] = {}
        self.struct_field_types: dict[str, dict[str, ast.Type]] = {}
        self.fn_signatures: dict[
            str, tuple[list[ast.Type | None], ast.Type | None]
        ] = {}
        self.method_signatures: dict[
            tuple[str, str], tuple[list[ast.Type | None], ast.Type | None]
        ] = {}
        self.struct_instances: dict[tuple[Any, ...], ast.StructDecl] = {}
        self.fn_instances: dict[tuple[Any, ...], ast.FnDecl] = {}
        self.enum_cache: dict[tuple[Any, ...], ast.EnumDecl] = {}
        self.generated_structs: list[ast.Item] = []
        self.generated_fns: list[ast.FnDecl] = []
        self.generated_impls: list[ast.ImplBlock] = []
        self.root_items: list[ast.Item] = []
        self._default_alias: tuple[str, list[ast.Type]] | None = None

    # -- public -----------------------------------------------------------

    def run(self) -> ast.Program:
        self._collect()
        # Preserve externally observable generic declarations for names that
        # do not collide with Rust prelude/runtime types.  This keeps the old
        # `GenericHolder` namespace entry while `Box<T>` only appears in its
        # mangled forms.
        for name, decl in self.generic_structs.items():
            if name not in _RESERVED_TYPE_NAMES:
                self.request_struct(name, _default_args(decl.generics), 0, default=True)
        for name, decl in self.generic_enums.items():
            if name not in _RESERVED_TYPE_NAMES:
                self.request_enum(name, _default_args(decl.generics), 0, default=True)

        for item in self.program.items:
            if isinstance(item, ast.StructDecl) and item.generics:
                continue
            if isinstance(item, ast.FnDecl) and item.generics:
                continue
            if isinstance(item, ast.EnumDecl) and item.generics:
                continue
            if isinstance(item, ast.ImplBlock) and item.generics:
                continue
            self.root_items.append(self._transform_root(item))

        result: list[ast.Item] = []
        seen: set[tuple[str, str]] = set()
        for item in (
            self.root_items
            + list(self.generated_structs)
            + list(self.generated_fns)
            + list(self.generated_impls)
        ):
            key = self._output_key(item)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            result.append(item)
        return ast.Program(
            self.program.line,
            self.program.col,
            result,
            self.program.source_lines,
            span=self.program.span,
        )

    # -- collection -------------------------------------------------------

    def _collect(self) -> None:
        for item in self.program.items:
            if isinstance(item, ast.StructDecl):
                if item.generics:
                    self.generic_structs[item.name] = item
            elif isinstance(item, ast.EnumDecl):
                if item.generics:
                    self.generic_enums[item.name] = item
            elif isinstance(item, ast.FnDecl):
                if item.generics:
                    self.generic_fns[item.name] = item
            elif isinstance(item, ast.ImplBlock):
                base = _base_name(item.ty)
                if base is None:
                    continue
                if item.generics or _collect_type_params(item.ty):
                    self.generic_impls.setdefault(base, []).append(item)
                    for method in item.items:
                        if isinstance(method, ast.FnDecl):
                            self.assoc_methods.setdefault(base, {}).setdefault(
                                method.name, (item, method)
                            )

    @staticmethod
    def _output_key(item: ast.Item) -> tuple[str, str] | None:
        if isinstance(item, ast.StructDecl):
            return ("struct", item.name)
        if isinstance(item, ast.FnDecl):
            return ("fn", item.name)
        if isinstance(item, ast.EnumDecl):
            return ("enum", item.name)
        return None

    # -- instantiation requests ------------------------------------------

    def _check_depth(self, depth: int, node: ast.Node) -> None:
        if depth > self.max_depth:
            raise MonomorphizationError(
                "monomorphization recursion limit exceeded "
                f"({self.max_depth} instantiations); the generic program "
                "expands without a finite concrete form",
                node.line,
                node.col,
                getattr(node, "span", None),
            )

    def _struct_key(
        self, name: str, args: list[ast.Type], default: bool
    ) -> tuple[Any, ...]:
        return ("struct", name, tuple(_type_key(arg) for arg in args), default)

    def _fn_key(
        self, name: str, args: list[ast.Type], default: bool
    ) -> tuple[Any, ...]:
        return ("fn", name, tuple(_type_key(arg) for arg in args), default)

    def request_struct(
        self,
        name: str,
        args: list[ast.Type],
        depth: int,
        *,
        default: bool = False,
        node: ast.Node | None = None,
    ) -> str:
        decl = self.generic_structs.get(name)
        if decl is None:
            return name
        self._check_depth(depth, node or decl)
        params = decl.generics
        if not args:
            args = _default_args(params)
            default = True
        if len(args) != len(params):
            raise MonomorphizationError(
                f"generic struct `{name}` expects {len(params)} type "
                f"argument(s), got {len(args)}",
                (node or decl).line,
                (node or decl).col,
                getattr(node or decl, "span", None),
            )
        args = [self.transform_type(arg, {}, depth, node=node) for arg in args]
        if (
            default
            and name not in _RESERVED_TYPE_NAMES
            and all(_types_equal(arg, _atomic_type(_DEFAULT_TYPE)) for arg in args)
        ):
            concrete = name
        else:
            concrete = _mangle(name, args)
            default = False
        key = self._struct_key(name, args, default)
        cached = self.struct_instances.get(key)
        if cached is not None:
            return cached.name

        placeholder = ast.StructDecl(
            decl.line, decl.col, concrete, [], [], decl.is_pub, span=decl.span
        )
        self.struct_instances[key] = placeholder
        self.generated_structs.append(placeholder)
        mapping = {param.name: arg for param, arg in zip(params, args)}
        previous_alias = self._default_alias
        if concrete == name:
            self._default_alias = (name, args)
        try:
            placeholder.fields = [
                (
                    fname,
                    self.transform_type(fty, mapping, depth + 1, node=decl),
                    is_pub,
                )
                for fname, fty, is_pub in decl.fields
            ]
            self.struct_field_types[concrete] = {
                fname: fty for fname, fty, _pub in placeholder.fields
            }
            self.instance_source[concrete] = ast.TypeName(
                decl.line, decl.col, name, args
            )
            for impl in self.generic_impls.get(name, []):
                concrete_impl = self._instantiate_impl(
                    impl, name, args, concrete, depth + 1
                )
                if concrete_impl is not None:
                    self.generated_impls.append(concrete_impl)
        finally:
            self._default_alias = previous_alias
        return concrete

    def request_fn(
        self,
        name: str,
        args: list[ast.Type],
        depth: int,
        *,
        default: bool = False,
        node: ast.Node | None = None,
    ) -> str:
        decl = self.generic_fns.get(name)
        if decl is None:
            return name
        self._check_depth(depth, node or decl)
        params = decl.generics
        if not args:
            args = _default_args(params)
            default = True
        if len(args) != len(params):
            raise MonomorphizationError(
                f"generic function `{name}` expects {len(params)} type "
                f"argument(s), got {len(args)}",
                (node or decl).line,
                (node or decl).col,
                getattr(node or decl, "span", None),
            )
        args = [self.transform_type(arg, {}, depth, node=node) for arg in args]
        if (
            default
            and name not in _RESERVED_TYPE_NAMES
            and all(_types_equal(arg, _atomic_type(_DEFAULT_TYPE)) for arg in args)
        ):
            concrete = name
        else:
            concrete = _mangle(name, args)
            default = False
        key = self._fn_key(name, args, default)
        cached = self.fn_instances.get(key)
        if cached is not None:
            return cached.name

        clone = copy.deepcopy(decl)
        clone.name = concrete
        clone.generics = []
        self.fn_instances[key] = clone
        self.generated_fns.append(clone)
        mapping = {param.name: arg for param, arg in zip(params, args)}
        self._transform_fn(clone, mapping, depth + 1, self_type=None)
        return concrete

    def _instantiate_impl(
        self,
        impl: ast.ImplBlock,
        base: str,
        args: list[ast.Type],
        concrete: str,
        depth: int,
    ) -> ast.ImplBlock | None:
        target = impl.ty if isinstance(impl.ty, ast.TypeName) else None
        if target is None:
            return None
        mapping: dict[str, ast.Type] = {}
        struct_mapping = {
            param.name: arg
            for param, arg in zip(self.generic_structs[base].generics, args)
        }
        if target.args:
            for param, arg in zip(impl.generics, target.args):
                mapping[param.name] = _substitute_type(arg, struct_mapping)
        # An impl parameter that only appears in the target (e.g.
        # `impl<T> Box<T>`) has already been bound above.  Any remaining
        # parameter keeps its position in the struct's parameter list.
        for param in impl.generics:
            if param.name not in mapping and param.name in struct_mapping:
                mapping[param.name] = struct_mapping[param.name]
        concrete_target = ast.TypeName(impl.ty.line, impl.ty.col, concrete, [])
        new_impl = ast.ImplBlock(
            impl.line,
            impl.col,
            concrete_target,
            self.transform_type(impl.trait, mapping, depth, node=impl)
            if impl.trait is not None
            else None,
            [],
            [],
            span=impl.span,
        )
        for item in impl.items:
            if isinstance(item, ast.FnDecl):
                clone = copy.deepcopy(item)
                clone.generics = []
                self._transform_fn(
                    clone,
                    mapping,
                    depth + 1,
                    self_type=ast.TypeName(impl.ty.line, impl.ty.col, concrete, []),
                )
                new_impl.items.append(clone)
                self._record_method_signature(concrete, clone)
        return new_impl

    def _record_method_signature(self, concrete: str, fn: ast.FnDecl) -> None:
        args = [
            self._param_type(param)
            for param in fn.params
            if not (
                isinstance(param.pattern, ast.PatIdent) and param.pattern.name == "self"
            )
        ]
        self.method_signatures[(concrete, fn.name)] = (args, fn.ret)

    @staticmethod
    def _param_type(param: ast.FnParam) -> ast.Type | None:
        if isinstance(param.pattern, ast.PatIdent) and param.pattern.name == "self":
            return None
        return param.ty

    # -- item transformation ---------------------------------------------

    def _transform_root(self, item: ast.Item) -> ast.Item:
        clone = copy.deepcopy(item)
        if isinstance(clone, ast.FnDecl):
            self._transform_fn(clone, {}, 0, self_type=None)
        elif isinstance(clone, ast.StructDecl):
            clone.fields = [
                (
                    fname,
                    self.transform_type(fty, {}, 0, node=clone),
                    is_pub,
                )
                for fname, fty, is_pub in clone.fields
            ]
            self.struct_field_types[clone.name] = {
                fname: fty for fname, fty, _pub in clone.fields
            }
        elif isinstance(clone, ast.EnumDecl):
            for variant in clone.variants:
                if variant.fields is not None:
                    variant.fields = [
                        self.transform_type(fty, {}, 0, node=clone)
                        for fty in variant.fields
                    ]
        elif isinstance(clone, ast.ImplBlock):
            self._transform_impl_root(clone)
        elif isinstance(clone, ast.ConstDecl):
            clone.ty = self.transform_type(clone.ty, {}, 0, node=clone)
            clone.value = self._transform_expr(clone.value, {}, {}, 0, None, clone.ty)
        elif isinstance(clone, ast.TypeAlias):
            clone.ty = self.transform_type(clone.ty, {}, 0, node=clone)
        elif isinstance(clone, ast.ModDecl) and clone.items is not None:
            clone.items = [self._transform_root(sub) for sub in clone.items]
        elif isinstance(clone, ast.TraitDecl):
            for trait_item in clone.items:
                if trait_item.fn is not None:
                    self._transform_fn(trait_item.fn, {}, 0, self_type=None)
        return clone

    def _transform_impl_root(self, impl: ast.ImplBlock) -> None:
        target = self.transform_type(impl.ty, {}, 0, node=impl)
        impl.ty = target
        self_type = target if isinstance(target, ast.TypeName) else None
        for item in impl.items:
            if isinstance(item, ast.FnDecl):
                item.generics = []
                self._transform_fn(item, {}, 0, self_type=self_type)
                if self_type is not None:
                    self._record_method_signature(self_type.name, item)

    def _transform_fn(
        self,
        fn: ast.FnDecl,
        mapping: dict[str, ast.Type],
        depth: int,
        *,
        self_type: ast.TypeName | None,
    ) -> None:
        fn.generics = []
        if self_type is not None:
            mapping = dict(mapping)
            mapping.setdefault("Self", self_type)
        env: dict[str, ast.Type] = {}
        for param in fn.params:
            if isinstance(param.pattern, ast.PatIdent) and param.pattern.name == "self":
                if self_type is not None:
                    env["self"] = self_type
                continue
            param.ty = self.transform_type(param.ty, mapping, depth, node=param)
            if isinstance(param.pattern, ast.PatIdent) and param.ty is not None:
                env[param.pattern.name] = param.ty
        fn.ret = self.transform_type(fn.ret, mapping, depth, node=fn)
        if fn.body is not None:
            fn.body = self._transform_block(fn.body, env, mapping, depth, self_type)
        self.fn_signatures[fn.name] = (
            [self._param_type(param) for param in fn.params],
            fn.ret,
        )

    # -- block / statement transformation -------------------------------

    def _transform_block(
        self,
        block: ast.Block,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
    ) -> ast.Block:
        inner = dict(env)
        for stmt in block.stmts:
            self._transform_stmt(stmt, inner, mapping, depth, self_type)
        if block.tail is not None:
            block.tail = self._transform_expr(
                block.tail, inner, mapping, depth, self_type, None
            )
        return block

    def _transform_stmt(
        self,
        stmt: ast.Stmt,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
    ) -> None:
        if isinstance(stmt, ast.LetStmt):
            stmt.ty = self.transform_type(stmt.ty, mapping, depth, node=stmt)
            stmt.value = (
                self._transform_expr(
                    stmt.value, env, mapping, depth, self_type, stmt.ty
                )
                if stmt.value is not None
                else None
            )
            inferred = stmt.ty or (
                self._infer_expr_type(stmt.value, env)
                if stmt.value is not None
                else None
            )
            if isinstance(stmt.pattern, ast.PatIdent) and inferred is not None:
                env[stmt.pattern.name] = inferred
        elif isinstance(stmt, ast.ExprStmt):
            stmt.expr = self._transform_expr(
                stmt.expr, env, mapping, depth, self_type, None
            )
        elif isinstance(stmt, ast.ItemStmt):
            stmt.item = self._transform_root(stmt.item)

    # -- expression transformation --------------------------------------

    def _transform_expr(
        self,
        expr: ast.Expr,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
        expected: ast.Type | None,
    ) -> ast.Expr:
        if isinstance(expr, ast.StructLiteral):
            return self._transform_struct_literal(
                expr, env, mapping, depth, self_type, expected
            )
        if isinstance(expr, ast.Call):
            return self._transform_call(expr, env, mapping, depth, self_type, expected)
        if isinstance(expr, ast.MethodCall):
            receiver_type = self._infer_expr_type(expr.receiver, env)
            expr.receiver = self._transform_expr(
                expr.receiver, env, mapping, depth, self_type, None
            )
            arg_types: list[ast.Type | None] = []
            signature = None
            if receiver_type is not None:
                class_name = (
                    receiver_type.name
                    if isinstance(receiver_type, ast.TypeName)
                    else None
                )
                if class_name is not None:
                    signature = self.method_signatures.get((class_name, expr.method))
            if signature is not None:
                params, _ret = signature
                arg_types = list(params)
            for index, arg in enumerate(expr.args):
                expected_arg = arg_types[index] if index < len(arg_types) else None
                expr.args[index] = self._transform_expr(
                    arg, env, mapping, depth, self_type, expected_arg
                )
            expr.generic_args = [
                self.transform_type(arg, mapping, depth, node=expr)
                for arg in expr.generic_args
            ]
            return expr
        if isinstance(expr, ast.Field):
            expr.receiver = self._transform_expr(
                expr.receiver, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Index):
            expr.receiver = self._transform_expr(
                expr.receiver, env, mapping, depth, self_type, None
            )
            expr.index = self._transform_expr(
                expr.index, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Unary):
            expr.operand = self._transform_expr(
                expr.operand, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Binary):
            expr.left = self._transform_expr(
                expr.left, env, mapping, depth, self_type, None
            )
            expr.right = self._transform_expr(
                expr.right, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, (ast.Assign, ast.CompoundAssign)):
            expr.target = self._transform_expr(
                expr.target, env, mapping, depth, self_type, None
            )
            expr.value = self._transform_expr(
                expr.value, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Cast):
            expr.expr = self._transform_expr(
                expr.expr, env, mapping, depth, self_type, None
            )
            expr.ty = self.transform_type(expr.ty, mapping, depth, node=expr)
            return expr
        if isinstance(expr, ast.Paren):
            expr.inner = self._transform_expr(
                expr.inner, env, mapping, depth, self_type, expected
            )
            return expr
        if isinstance(expr, ast.Block):
            block_env = dict(env)
            expr = self._transform_block(expr, block_env, mapping, depth, self_type)
            return expr
        if isinstance(expr, (ast.IfExpr, ast.IfLet)):
            return self._transform_if(expr, env, mapping, depth, self_type)
        if isinstance(expr, ast.MatchExpr):
            scrutinee_type = self._infer_expr_type(expr.scrutinee, env)
            expr.scrutinee = self._transform_expr(
                expr.scrutinee, env, mapping, depth, self_type, None
            )
            for arm in expr.arms:
                arm_env = dict(env)
                arm.pattern = self._transform_pattern(
                    arm.pattern,
                    arm_env,
                    mapping,
                    depth,
                    self_type,
                    expected=scrutinee_type,
                )
                if arm.guard is not None:
                    arm.guard = self._transform_expr(
                        arm.guard, arm_env, mapping, depth, self_type, None
                    )
                arm.body = self._transform_expr(
                    arm.body, arm_env, mapping, depth, self_type, expected
                )
            return expr
        if isinstance(expr, (ast.Loop, ast.While, ast.For)):
            if isinstance(expr, ast.While):
                expr.cond = self._transform_expr(
                    expr.cond, env, mapping, depth, self_type, None
                )
            elif isinstance(expr, ast.For):
                expr.iter = self._transform_expr(
                    expr.iter, env, mapping, depth, self_type, None
                )
                body_env = dict(env)
                expr.pattern = self._transform_pattern(
                    expr.pattern, body_env, mapping, depth, self_type
                )
                expr.body = self._transform_block(
                    expr.body, body_env, mapping, depth, self_type
                )
                return expr
            expr.body = self._transform_block(
                expr.body, dict(env), mapping, depth, self_type
            )
            return expr
        if isinstance(expr, ast.Closure):
            closure_env = dict(env)
            for pat, ty in expr.params:
                if ty is not None:
                    ty = self.transform_type(ty, mapping, depth, node=expr)
                if isinstance(pat, ast.PatIdent) and ty is not None:
                    closure_env[pat.name] = ty
            expr.ret = self.transform_type(expr.ret, mapping, depth, node=expr)
            expr.body = self._transform_expr(
                expr.body, closure_env, mapping, depth, self_type, expr.ret
            )
            return expr
        if isinstance(expr, ast.Return):
            expr.value = (
                self._transform_expr(expr.value, env, mapping, depth, self_type, None)
                if expr.value is not None
                else None
            )
            return expr
        if isinstance(expr, ast.Break):
            if expr.value is not None:
                expr.value = self._transform_expr(
                    expr.value, env, mapping, depth, self_type, None
                )
            return expr
        if isinstance(expr, ast.Try):
            expr.expr = self._transform_expr(
                expr.expr, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Await):
            expr.expr = self._transform_expr(
                expr.expr, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.OptionSome):
            expr.value = self._transform_expr(
                expr.value, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.ResultOk):
            expr.value = self._transform_expr(
                expr.value, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.ResultErr):
            expr.value = self._transform_expr(
                expr.value, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.Range):
            if expr.start is not None:
                expr.start = self._transform_expr(
                    expr.start, env, mapping, depth, self_type, None
                )
            if expr.end is not None:
                expr.end = self._transform_expr(
                    expr.end, env, mapping, depth, self_type, None
                )
            return expr
        if isinstance(expr, ast.TupleExpr):
            expr.elems = [
                self._transform_expr(item, env, mapping, depth, self_type, None)
                for item in expr.elems
            ]
            return expr
        if isinstance(expr, ast.ArrayExpr):
            expr.elems = [
                self._transform_expr(item, env, mapping, depth, self_type, None)
                for item in expr.elems
            ]
            return expr
        if isinstance(expr, ast.RepeatArray):
            expr.elem = self._transform_expr(
                expr.elem, env, mapping, depth, self_type, None
            )
            expr.count = self._transform_expr(
                expr.count, env, mapping, depth, self_type, None
            )
            return expr
        if isinstance(expr, ast.MacroCall):
            if isinstance(expr.args, list):
                expr.args = [
                    self._transform_expr(item, env, mapping, depth, self_type, None)
                    for item in expr.args
                ]
            return expr
        if isinstance(expr, ast.Path):
            return self._transform_path(expr, mapping, depth, self_type, expected)
        if (
            isinstance(expr, ast.Ident)
            and expr.name == "Self"
            and self_type is not None
        ):
            return ast.Ident(expr.line, expr.col, self_type.name, span=expr.span)
        return expr

    def _transform_if(
        self,
        expr: ast.IfExpr | ast.IfLet,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
    ) -> ast.IfExpr | ast.IfLet:
        if isinstance(expr, ast.IfLet):
            value_type = self._infer_expr_type(expr.value, env)
            expr.value = self._transform_expr(
                expr.value, env, mapping, depth, self_type, None
            )
            then_env = dict(env)
            expr.pattern = self._transform_pattern(
                expr.pattern,
                then_env,
                mapping,
                depth,
                self_type,
                expected=value_type,
            )
            expr.then_block = self._transform_block(
                expr.then_block, then_env, mapping, depth, self_type
            )
        else:
            expr.cond = self._transform_expr(
                expr.cond, env, mapping, depth, self_type, None
            )
            expr.then_block = self._transform_block(
                expr.then_block, dict(env), mapping, depth, self_type
            )
        if isinstance(expr.else_branch, ast.Block):
            expr.else_branch = self._transform_block(
                expr.else_branch, dict(env), mapping, depth, self_type
            )
        elif isinstance(expr.else_branch, (ast.IfExpr, ast.IfLet)):
            expr.else_branch = self._transform_if(
                expr.else_branch, env, mapping, depth, self_type
            )
        elif expr.else_branch is not None:
            expr.else_branch = self._transform_expr(
                expr.else_branch, env, mapping, depth, self_type, None
            )
        return expr

    def _transform_struct_literal(
        self,
        expr: ast.StructLiteral,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
        expected: ast.Type | None,
    ) -> ast.StructLiteral:
        base = expr.path.split("::")[-1]
        decl = self.generic_structs.get(base)
        explicit = [
            self.transform_type(arg, mapping, depth, node=expr)
            for arg in expr.generic_args
        ]
        args = explicit
        if decl is not None and not args:
            source_expected = self._source_type(expected)
            if (
                isinstance(source_expected, ast.TypeName)
                and source_expected.name == base
                and source_expected.args
            ):
                args = [
                    self.transform_type(arg, mapping, depth, node=expr)
                    for arg in source_expected.args
                ]
            else:
                bindings: dict[str, ast.Type] = {}
                params = {param.name for param in decl.generics}
                for fname, value in expr.fields:
                    field_ty = next(
                        (fty for name, fty, _pub in decl.fields if name == fname),
                        None,
                    )
                    actual = self._infer_expr_type(value, env)
                    if field_ty is not None and actual is not None:
                        self._unify(field_ty, actual, bindings, params)
                args = [
                    self.transform_type(
                        bindings.get(param.name, _atomic_type(_DEFAULT_TYPE)),
                        mapping,
                        depth,
                        node=expr,
                    )
                    for param in decl.generics
                ]
        if decl is not None:
            expected_args = bool(
                isinstance(expected, ast.TypeName)
                and (expected.args or expected.name != base)
            )
            default = not explicit and not expected_args
            concrete = self.request_struct(
                base, args, depth, default=default, node=expr
            )
            expr.path = concrete
            expr.generic_args = []
            field_types = self.struct_field_types.get(concrete, {})
            expr.fields = [
                (
                    fname,
                    self._transform_expr(
                        value,
                        env,
                        mapping,
                        depth,
                        self_type,
                        field_types.get(fname),
                    ),
                )
                for fname, value in expr.fields
            ]
        else:
            expr.fields = [
                (
                    fname,
                    self._transform_expr(value, env, mapping, depth, self_type, None),
                )
                for fname, value in expr.fields
            ]
        if expr.rest is not None:
            expr.rest = self._transform_expr(
                expr.rest, env, mapping, depth, self_type, None
            )
        return expr

    def _transform_call(
        self,
        expr: ast.Call,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
        expected: ast.Type | None,
    ) -> ast.Call:
        func = expr.func
        arg_types = [self._infer_expr_type(arg, env) for arg in expr.args]

        if isinstance(func, ast.Ident) and func.name in self.generic_fns:
            return self._transform_generic_fn_call(
                expr, func.name, [], arg_types, env, mapping, depth, self_type
            )
        if isinstance(func, ast.Path):
            segments = list(func.segments)
            explicit = [
                self.transform_type(arg, mapping, depth, node=func)
                for arg in func.generic_args
            ]
            if len(segments) == 1 and segments[0] in self.generic_fns:
                return self._transform_generic_fn_call(
                    expr,
                    segments[0],
                    explicit,
                    arg_types,
                    env,
                    mapping,
                    depth,
                    self_type,
                )
            if len(segments) >= 2 and segments[0] in self.generic_enums:
                base = segments[0]
                decl = self.generic_enums[base]
                variant = next(
                    (item for item in decl.variants if item.name == segments[-1]),
                    None,
                )
                source_expected = self._source_type(expected)
                expected_args = bool(
                    isinstance(source_expected, ast.TypeName)
                    and source_expected.name == base
                    and source_expected.args
                )
                bindings: dict[str, ast.Type] = {}
                enum_params = {param.name for param in decl.generics}
                field_types = (variant.fields or []) if variant is not None else []
                if expected_args:
                    for param, arg in zip(decl.generics, source_expected.args):
                        bindings[param.name] = arg
                for field_ty, actual in zip(field_types, arg_types):
                    if actual is not None:
                        self._unify(field_ty, actual, bindings, enum_params)
                args = list(explicit)
                if not args:
                    args = [
                        self.transform_type(
                            bindings.get(param.name, _atomic_type(_DEFAULT_TYPE)),
                            mapping,
                            depth,
                            node=expr,
                        )
                        for param in decl.generics
                    ]
                concrete = self.request_enum(
                    base,
                    args,
                    depth,
                    default=not explicit and not expected_args,
                    node=expr,
                )
                expr.func = ast.Path(
                    func.line,
                    func.col,
                    [concrete] + segments[1:],
                    [],
                    span=func.span,
                )
                substitution = {
                    param.name: arg for param, arg in zip(decl.generics, args)
                }
                expected_fields = [
                    self.transform_type(
                        _substitute_type(field_ty, substitution),
                        mapping,
                        depth,
                        node=expr,
                    )
                    for field_ty in field_types
                ]
                expr.args = [
                    self._transform_expr(
                        arg,
                        env,
                        mapping,
                        depth,
                        self_type,
                        expected_fields[index]
                        if index < len(expected_fields)
                        else None,
                    )
                    for index, arg in enumerate(expr.args)
                ]
                return expr
            if len(segments) >= 2 and segments[0] in self.generic_structs:
                base = segments[0]
                method_name = segments[-1]
                method_info = self.assoc_methods.get(base, {}).get(method_name)
                if method_info is not None:
                    impl, method = method_info
                    method_map = self._impl_mapping(impl)
                    bindings: dict[str, ast.Type] = {}
                    struct_params = {
                        param.name for param in self.generic_structs[base].generics
                    }
                    source_expected = self._source_type(expected)
                    if (
                        isinstance(source_expected, ast.TypeName)
                        and source_expected.name == base
                        and source_expected.args
                    ):
                        for param, arg in zip(
                            self.generic_structs[base].generics,
                            source_expected.args,
                        ):
                            bindings[param.name] = arg
                    raw_params = [
                        _substitute_type(param.ty, method_map)
                        for param in method.params
                        if not (
                            isinstance(param.pattern, ast.PatIdent)
                            and param.pattern.name == "self"
                        )
                    ]
                    for param_ty, actual in zip(raw_params, arg_types):
                        if param_ty is not None and actual is not None:
                            self._unify(param_ty, actual, bindings, struct_params)
                    args = explicit
                    if not args:
                        args = [
                            self.transform_type(
                                bindings.get(param.name, _atomic_type(_DEFAULT_TYPE)),
                                mapping,
                                depth,
                                node=expr,
                            )
                            for param in self.generic_structs[base].generics
                        ]
                    expected_args = bool(
                        isinstance(source_expected, ast.TypeName)
                        and source_expected.name == base
                        and source_expected.args
                    )
                    default = not explicit and not expected_args
                    concrete = self.request_struct(
                        base, args, depth, default=default, node=expr
                    )
                    expr.func = ast.Path(
                        func.line,
                        func.col,
                        [concrete] + segments[1:],
                        [],
                        span=func.span,
                    )
                    full_mapping = dict(method_map)
                    for param, arg in zip(self.generic_structs[base].generics, args):
                        full_mapping[param.name] = arg
                    expected_params = [
                        self.transform_type(
                            _substitute_type(param.ty, full_mapping),
                            mapping,
                            depth,
                            node=param,
                        )
                        for param in method.params
                        if not (
                            isinstance(param.pattern, ast.PatIdent)
                            and param.pattern.name == "self"
                        )
                    ]
                    expr.args = [
                        self._transform_expr(
                            arg,
                            env,
                            mapping,
                            depth,
                            self_type,
                            expected_params[index]
                            if index < len(expected_params)
                            else None,
                        )
                        for index, arg in enumerate(expr.args)
                    ]
                    return expr
            expr.func = ast.Path(
                func.line,
                func.col,
                segments,
                explicit,
                span=func.span,
            )
        expr.func = self._transform_expr(
            expr.func, env, mapping, depth, self_type, None
        )
        param_types = self._call_param_types(expr.func)
        expr.args = [
            self._transform_expr(
                arg,
                env,
                mapping,
                depth,
                self_type,
                param_types[index] if index < len(param_types) else None,
            )
            for index, arg in enumerate(expr.args)
        ]
        return expr

    def _transform_generic_fn_call(
        self,
        expr: ast.Call,
        name: str,
        explicit: list[ast.Type],
        arg_types: list[ast.Type | None],
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
    ) -> ast.Call:
        decl = self.generic_fns[name]
        params = decl.generics
        bindings: dict[str, ast.Type] = {}
        param_names = {param.name for param in params}
        if not explicit:
            for param, actual in zip(decl.params, arg_types):
                if param.ty is not None and actual is not None:
                    self._unify(param.ty, actual, bindings, param_names)
            explicit = [
                self.transform_type(
                    bindings.get(param.name, _atomic_type(_DEFAULT_TYPE)),
                    mapping,
                    depth,
                    node=expr,
                )
                for param in params
            ]
        else:
            explicit = [
                self.transform_type(arg, mapping, depth, node=expr) for arg in explicit
            ]
        default = not explicit
        concrete = self.request_fn(name, explicit, depth, default=default, node=expr)
        expr.func = ast.Ident(expr.func.line, expr.func.col, concrete)
        substituted = {param.name: arg for param, arg in zip(params, explicit)}
        expected_params = [
            self.transform_type(
                _substitute_type(param.ty, substituted),
                mapping,
                depth,
                node=param,
            )
            for param in decl.params
        ]
        expr.args = [
            self._transform_expr(
                arg,
                env,
                mapping,
                depth,
                self_type,
                expected_params[index] if index < len(expected_params) else None,
            )
            for index, arg in enumerate(expr.args)
        ]
        return expr

    def _transform_path(
        self,
        expr: ast.Path,
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
        expected: ast.Type | None = None,
    ) -> ast.Path:
        segments = list(expr.segments)
        if segments and segments[0] == "Self":
            if self_type is not None:
                segments[0] = self_type.name
            elif segments[-1] in self.generic_structs:
                segments[-1] = self.request_struct(
                    segments[-1], [], depth, default=True, node=expr
                )
        elif len(segments) >= 2 and segments[0] in self.generic_enums:
            segments[0] = self._concrete_for_pattern(
                segments[0], expected, depth, expr, is_enum=True
            )
        expr.segments = segments
        expr.generic_args = [
            self.transform_type(arg, mapping, depth, node=expr)
            for arg in expr.generic_args
        ]
        return expr

    def _impl_mapping(self, impl: ast.ImplBlock) -> dict[str, ast.Type]:
        mapping: dict[str, ast.Type] = {}
        target = impl.ty if isinstance(impl.ty, ast.TypeName) else None
        if target is None:
            return mapping
        for param, arg in zip(impl.generics, target.args):
            mapping[param.name] = arg
        return mapping

    def _call_param_types(self, func: ast.Expr) -> list[ast.Type | None]:
        name = None
        if isinstance(func, ast.Ident):
            name = func.name
        elif isinstance(func, ast.Path) and len(func.segments) >= 2:
            return self.method_signatures.get(
                (func.segments[-2], func.segments[-1]), ([], None)
            )[0]
        if name is not None:
            return self.fn_signatures.get(name, ([], None))[0]
        return []

    # -- type transformation ---------------------------------------------

    def transform_type(
        self,
        ty: ast.Type | None,
        mapping: dict[str, ast.Type],
        depth: int,
        *,
        node: ast.Node | None = None,
    ) -> ast.Type | None:
        if ty is None:
            return None
        if isinstance(ty, ast.TypeName):
            if not ty.args and ty.name in mapping:
                value = copy.deepcopy(mapping[ty.name])
                return self.transform_type(value, {}, depth, node=node)
            if ty.name == "Self" and node is not None:
                # `Self` is handled by callers that pass the concrete type
                # through the substitution mapping; nothing else to do here.
                pass
            args = [
                self.transform_type(arg, mapping, depth, node=node) for arg in ty.args
            ]
            if ty.name in self.generic_structs:
                if (
                    self._default_alias is not None
                    and self._default_alias[0] == ty.name
                    and args
                    and [_type_key(arg) for arg in args]
                    == [_type_key(arg) for arg in self._default_alias[1]]
                ):
                    return ast.TypeName(
                        ty.line,
                        ty.col,
                        ty.name,
                        [],
                        span=ty.span,
                    )
                concrete = self.request_struct(
                    ty.name, args, depth, default=not args, node=node
                )
                return ast.TypeName(ty.line, ty.col, concrete, [], span=ty.span)
            if ty.name in self.generic_enums:
                # Generic enums are monomorphized like structs; the backends
                # currently accept only concrete variants.
                concrete = self.request_enum(
                    ty.name, args, depth, default=not args, node=node
                )
                return ast.TypeName(ty.line, ty.col, concrete, [], span=ty.span)
            return ast.TypeName(ty.line, ty.col, ty.name, args, span=ty.span)
        if isinstance(ty, ast.TypeRef):
            return ast.TypeRef(
                ty.line,
                ty.col,
                self.transform_type(ty.inner, mapping, depth, node=node),
                ty.mutable,
                span=ty.span,
            )
        if isinstance(ty, ast.TypeSlice):
            return ast.TypeSlice(
                ty.line,
                ty.col,
                self.transform_type(ty.inner, mapping, depth, node=node),
                span=ty.span,
            )
        if isinstance(ty, ast.TypeArray):
            return ast.TypeArray(
                ty.line,
                ty.col,
                self.transform_type(ty.inner, mapping, depth, node=node),
                ty.size,
                span=ty.span,
            )
        if isinstance(ty, ast.TypeTuple):
            return ast.TypeTuple(
                ty.line,
                ty.col,
                [
                    self.transform_type(item, mapping, depth, node=node)
                    for item in ty.elems
                ],
                span=ty.span,
            )
        if isinstance(ty, ast.TypeFn):
            return ast.TypeFn(
                ty.line,
                ty.col,
                [
                    self.transform_type(item, mapping, depth, node=node)
                    for item in ty.params
                ],
                self.transform_type(ty.ret, mapping, depth, node=node),
                span=ty.span,
            )
        return ty

    def request_enum(
        self,
        name: str,
        args: list[ast.Type],
        depth: int,
        *,
        default: bool = False,
        node: ast.Node | None = None,
    ) -> str:
        decl = self.generic_enums.get(name)
        if decl is None:
            return name
        # Generic enums are rare in the current language; emit a concrete copy
        # with the same mangling rules so backends never see a parameter.
        self._check_depth(depth, node or decl)
        if not args:
            args = _default_args(decl.generics)
            default = True
        args = [self.transform_type(arg, {}, depth, node=node) for arg in args]
        concrete = (
            name
            if default
            and name not in _RESERVED_TYPE_NAMES
            and all(_types_equal(arg, _atomic_type(_DEFAULT_TYPE)) for arg in args)
            else _mangle(name, args)
        )
        key = ("enum", name, tuple(_type_key(arg) for arg in args), concrete == name)
        cached = self.enum_cache.get(key)
        if cached is not None:
            return cached.name
        clone = copy.deepcopy(decl)
        clone.name = concrete
        clone.generics = []
        mapping = {param.name: arg for param, arg in zip(decl.generics, args)}
        for variant in clone.variants:
            if variant.fields is not None:
                variant.fields = [
                    self.transform_type(fty, mapping, depth + 1, node=clone)
                    for fty in variant.fields
                ]
        self.enum_cache[key] = clone
        self.generated_structs.append(clone)
        self.instance_source[concrete] = ast.TypeName(decl.line, decl.col, name, args)
        return concrete

    # -- patterns ---------------------------------------------------------

    def _transform_pattern(
        self,
        pattern: ast.Pattern,
        env: dict[str, ast.Type],
        mapping: dict[str, ast.Type],
        depth: int,
        self_type: ast.TypeName | None,
        *,
        expected: ast.Type | None = None,
    ) -> ast.Pattern:
        if isinstance(pattern, ast.PatStruct):
            base = pattern.path.split("::")[0]
            if base in self.generic_structs:
                concrete = self._concrete_for_pattern(
                    base, expected, depth, pattern, is_enum=False
                )
                pattern.path = concrete
            pattern.fields = [
                (
                    fname,
                    self._transform_pattern(sub, env, mapping, depth, self_type),
                )
                for fname, sub in pattern.fields
            ]
        elif isinstance(pattern, ast.PatTuple):
            pattern.elems = [
                self._transform_pattern(sub, env, mapping, depth, self_type)
                for sub in pattern.elems
            ]
        elif isinstance(pattern, ast.PatOr):
            pattern.alts = [
                self._transform_pattern(sub, env, mapping, depth, self_type)
                for sub in pattern.alts
            ]
        elif isinstance(pattern, (ast.PatBinding, ast.PatRef)):
            pattern.inner = self._transform_pattern(
                pattern.inner, env, mapping, depth, self_type
            )
        elif isinstance(pattern, ast.PatEnum):
            segments = pattern.path.split("::")
            base = segments[0]
            if base in self.generic_enums:
                concrete = self._concrete_for_pattern(
                    base, expected, depth, pattern, is_enum=True
                )
                pattern.path = "::".join([concrete] + segments[1:])
            elif base in self.generic_structs:
                pattern.path = self._concrete_for_pattern(
                    base, expected, depth, pattern, is_enum=False
                )
            pattern.args = [
                self._transform_pattern(sub, env, mapping, depth, self_type)
                for sub in pattern.args
            ]
        elif isinstance(pattern, ast.PatPath):
            segments = pattern.path.split("::")
            if len(segments) >= 2 and segments[0] in self.generic_enums:
                segments[0] = self._concrete_for_pattern(
                    segments[0], expected, depth, pattern, is_enum=True
                )
                pattern.path = "::".join(segments)
        elif isinstance(pattern, ast.PatSlice):
            pattern.elems = [
                self._transform_pattern(sub, env, mapping, depth, self_type)
                for sub in pattern.elems
            ]
        elif isinstance(pattern, ast.PatBox):
            pattern.inner = self._transform_pattern(
                pattern.inner, env, mapping, depth, self_type
            )
        elif isinstance(pattern, ast.PatIdent):
            # Patterns do not carry a type; a later unification could refine
            # this, but leaving the parameter unbound is still concrete enough
            # for the backends.
            pass
        return pattern

    def _concrete_for_pattern(
        self,
        base: str,
        expected: ast.Type | None,
        depth: int,
        pattern: ast.Pattern,
        *,
        is_enum: bool,
    ) -> str:
        source = self._source_type(expected)
        if isinstance(source, ast.TypeName) and source.name == base and source.args:
            args = [
                self.transform_type(arg, {}, depth, node=pattern) for arg in source.args
            ]
            if is_enum:
                return self.request_enum(base, args, depth, node=pattern)
            return self.request_struct(base, args, depth, node=pattern)
        if is_enum:
            return self.request_enum(base, [], depth, default=True, node=pattern)
        return self.request_struct(base, [], depth, default=True, node=pattern)

    # -- type inference ---------------------------------------------------

    def _source_type(self, ty: ast.Type | None) -> ast.Type | None:
        if isinstance(ty, ast.TypeName) and not ty.args:
            source = self.instance_source.get(ty.name)
            if source is not None:
                return source
        return ty

    def _unify(
        self,
        param: ast.Type | None,
        actual: ast.Type | None,
        bindings: dict[str, ast.Type],
        params: set[str],
    ) -> None:
        if param is None or actual is None:
            return
        param = self._source_type(param)
        actual = self._source_type(actual)
        if isinstance(param, ast.TypeName):
            if not param.args and param.name in params:
                bindings.setdefault(param.name, copy.deepcopy(actual))
                return
            if isinstance(actual, ast.TypeName) and param.name == actual.name:
                for left, right in zip(param.args, actual.args):
                    self._unify(left, right, bindings, params)
            return
        if (
            isinstance(param, ast.TypeRef)
            and isinstance(actual, ast.TypeRef)
            or isinstance(param, ast.TypeSlice)
            and isinstance(actual, ast.TypeSlice)
        ):
            self._unify(param.inner, actual.inner, bindings, params)
        elif isinstance(param, ast.TypeTuple) and isinstance(actual, ast.TypeTuple):
            for left, right in zip(param.elems, actual.elems):
                self._unify(left, right, bindings, params)
        elif isinstance(param, ast.TypeFn) and isinstance(actual, ast.TypeFn):
            for left, right in zip(param.params, actual.params):
                self._unify(left, right, bindings, params)
            self._unify(param.ret, actual.ret, bindings, params)

    def _infer_expr_type(
        self, expr: ast.Expr | None, env: dict[str, ast.Type]
    ) -> ast.Type | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Paren):
            return self._infer_expr_type(expr.inner, env)
        if isinstance(expr, ast.LitInt):
            return self._integer_literal_type(expr)
        if isinstance(expr, ast.LitFloat):
            return _atomic_type("f64")
        if isinstance(expr, ast.LitStr):
            return _atomic_type("String")
        if isinstance(expr, ast.LitChar):
            return _atomic_type("char")
        if isinstance(expr, ast.LitBool):
            return _atomic_type("bool")
        if isinstance(expr, ast.Ident):
            return env.get(expr.name)
        if isinstance(expr, ast.Path):
            if len(expr.segments) == 1:
                return None
            return None
        if isinstance(expr, ast.Unary):
            if expr.op == "!":
                return _atomic_type("bool")
            inferred = self._infer_expr_type(expr.operand, env)
            if expr.op.startswith("&") and inferred is not None:
                return ast.TypeRef(expr.line, expr.col, inferred, expr.mutable)
            return inferred
        if isinstance(expr, ast.Binary):
            if expr.op in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"}:
                return _atomic_type("bool")
            left = self._infer_expr_type(expr.left, env)
            right = self._infer_expr_type(expr.right, env)
            if (
                left is not None
                and isinstance(left, ast.TypeName)
                and left.name == "String"
            ):
                return left
            return left or right
        if isinstance(expr, ast.Cast):
            return expr.ty
        if isinstance(expr, ast.Call):
            return self._infer_call_type(expr, env)
        if isinstance(expr, ast.MethodCall):
            return self._infer_method_type(expr, env)
        if isinstance(expr, ast.Field):
            receiver = self._infer_expr_type(expr.receiver, env)
            class_name = receiver.name if isinstance(receiver, ast.TypeName) else None
            if class_name is not None:
                return self.struct_field_types.get(class_name, {}).get(expr.name)
            return None
        if isinstance(expr, ast.Index):
            receiver = self._infer_expr_type(expr.receiver, env)
            if isinstance(receiver, ast.TypeName) and receiver.args:
                return receiver.args[0]
            return None
        if isinstance(expr, ast.StructLiteral):
            return ast.TypeName(expr.line, expr.col, expr.path, [])
        if isinstance(expr, ast.IfExpr):
            then_types = self._block_type(expr.then_block, env)
            if isinstance(expr.else_branch, ast.Block):
                return then_types or self._block_type(expr.else_branch, env)
            return then_types or self._infer_expr_type(expr.else_branch, env)
        if isinstance(expr, ast.IfLet):
            then_types = self._block_type(expr.then_block, env)
            if isinstance(expr.else_branch, ast.Block):
                return then_types or self._block_type(expr.else_branch, env)
            return then_types or self._infer_expr_type(expr.else_branch, env)
        if isinstance(expr, ast.MatchExpr):
            for arm in expr.arms:
                inferred = self._infer_expr_type(arm.body, env)
                if inferred is not None:
                    return inferred
            return None
        if isinstance(expr, ast.Block):
            return self._block_type(expr, env)
        if isinstance(expr, ast.Range):
            return self._infer_expr_type(expr.start, env) or self._infer_expr_type(
                expr.end, env
            )
        if isinstance(expr, (ast.ArrayExpr, ast.RepeatArray)):
            if isinstance(expr, ast.ArrayExpr) and expr.elems:
                element = self._infer_expr_type(expr.elems[0], env)
            elif isinstance(expr, ast.RepeatArray):
                element = self._infer_expr_type(expr.elem, env)
            else:
                element = None
            if element is None:
                return None
            return ast.TypeName(expr.line, expr.col, "Vec", [element])
        if isinstance(expr, ast.OptionSome):
            inner = self._infer_expr_type(expr.value, env)
            if inner is None:
                return None
            return ast.TypeName(expr.line, expr.col, "Option", [inner])
        if isinstance(expr, ast.OptionNone):
            return ast.TypeName(expr.line, expr.col, "Option", [])
        if isinstance(expr, ast.ResultOk):
            inner = self._infer_expr_type(expr.value, env)
            if inner is None:
                return None
            return ast.TypeName(expr.line, expr.col, "Result", [inner])
        if isinstance(expr, ast.ResultErr):
            return ast.TypeName(expr.line, expr.col, "Result", [])
        if isinstance(expr, ast.Try):
            return self._infer_expr_type(expr.expr, env)
        return None

    def _infer_call_type(
        self, call: ast.Call, env: dict[str, ast.Type]
    ) -> ast.Type | None:
        func = call.func
        if isinstance(func, ast.Ident):
            signature = self.fn_signatures.get(func.name)
            if signature is not None:
                return signature[1]
            generic = self.generic_fns.get(func.name)
            if generic is not None:
                bindings: dict[str, ast.Type] = {}
                params = {param.name for param in generic.generics}
                for param, arg in zip(generic.params, call.args):
                    actual = self._infer_expr_type(arg, env)
                    if param.ty is not None and actual is not None:
                        self._unify(param.ty, actual, bindings, params)
                return self._substitute_inferred(generic.ret, bindings)
            return None
        if isinstance(func, ast.Path):
            segments = func.segments
            if len(segments) >= 2:
                # Associated function/method on a concrete struct.
                for base in (segments[-2], segments[0]):
                    signature = self.method_signatures.get((base, segments[-1]))
                    if signature is not None:
                        return signature[1]
                base = segments[0]
                decl = self.generic_structs.get(base)
                info = self.assoc_methods.get(base, {}).get(segments[-1])
                if decl is not None and info is not None:
                    impl, method = info
                    bindings: dict[str, ast.Type] = {}
                    params = {param.name for param in decl.generics}
                    method_map = self._impl_mapping(impl)
                    for param, arg in zip(method.params, call.args):
                        actual = self._infer_expr_type(arg, env)
                        raw = _substitute_type(param.ty, method_map)
                        if raw is not None and actual is not None:
                            self._unify(raw, actual, bindings, params)
                    ret = _substitute_type(method.ret, method_map)
                    return self._substitute_inferred(ret, bindings)
            if len(segments) == 1:
                signature = self.fn_signatures.get(segments[0])
                if signature is not None:
                    return signature[1]
        return None

    def _infer_method_type(
        self, call: ast.MethodCall, env: dict[str, ast.Type]
    ) -> ast.Type | None:
        receiver_type = self._infer_expr_type(call.receiver, env)
        class_name = (
            receiver_type.name if isinstance(receiver_type, ast.TypeName) else None
        )
        if class_name is not None:
            signature = self.method_signatures.get((class_name, call.method))
            if signature is not None:
                return signature[1]
        receiver = self._source_type(receiver_type)
        if call.method in {"len", "chars_count", "byte_len"}:
            return _atomic_type("usize")
        if call.method in {"is_empty", "contains", "starts_with", "ends_with"}:
            return _atomic_type("bool")
        if call.method in {"to_string", "trim", "to_uppercase", "to_lowercase"}:
            return _atomic_type("String")
        if (
            call.method == "unwrap"
            and isinstance(receiver, ast.TypeName)
            and receiver.args
        ):
            return receiver.args[0]
        if call.method == "clone":
            return receiver
        return None

    def _block_type(
        self, block: ast.Block, env: dict[str, ast.Type]
    ) -> ast.Type | None:
        inner = dict(env)
        for stmt in block.stmts:
            if isinstance(stmt, ast.LetStmt) and stmt.ty is not None:
                inner[stmt.pattern.name] = (
                    stmt.ty if isinstance(stmt.pattern, ast.PatIdent) else None
                )
        return self._infer_expr_type(block.tail, inner)

    def _substitute_inferred(
        self, ty: ast.Type | None, bindings: dict[str, ast.Type]
    ) -> ast.Type | None:
        return _substitute_type(ty, bindings)

    @staticmethod
    def _integer_literal_type(expr: ast.LitInt) -> ast.TypeName:
        suffixes = (
            "i128",
            "u128",
            "isize",
            "usize",
            "i64",
            "u64",
            "i32",
            "u32",
            "i16",
            "u16",
            "i8",
            "u8",
        )
        raw = expr.raw.lower()
        for suffix in suffixes:
            if raw.endswith(suffix):
                return _atomic_type(suffix)
        return _atomic_type(_DEFAULT_TYPE)


def monomorphize(
    program: ast.Program, *, max_depth: int = MAX_MONOMORPH_DEPTH
) -> ast.Program:
    """Return a copy of ``program`` with all generics instantiated."""
    if not any(
        isinstance(item, (ast.FnDecl, ast.StructDecl, ast.EnumDecl))
        and getattr(item, "generics", None)
        for item in program.items
    ) and not any(
        isinstance(item, ast.ImplBlock) and item.generics for item in program.items
    ):
        return program
    return Monomorphizer(program, max_depth=max_depth).run()


__all__ = [
    "MAX_MONOMORPH_DEPTH",
    "MonomorphizationError",
    "Monomorphizer",
    "monomorphize",
]
