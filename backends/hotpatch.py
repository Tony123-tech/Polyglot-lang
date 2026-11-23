"""Versioned hot-patching foundations for the Python and native backends.

The manager in this module is deliberately conservative:

* It swaps Python code objects and class dictionaries in place, which keeps
  object identity and the existing ``mio``/``PyFuture`` reactor alive.
* It never rewrites a live CPython frame.  A patch is only applied at a
  quiescent point (``active_calls == 0``) unless the caller explicitly forces
  it; suspended coroutines finish on the code object they captured.
* Native symbols are reached through an atomic dispatch table.  The table is
  an ABI-safe indirection point; patching the text of an already loaded
  ``.so``/``.dylib`` is not attempted and is rejected with E007.

The last point is the only sound way to provide native hot-swapping without
depending on a specific linker, macOS ``MAP_JIT`` policy, or PyO3 ABI.  Code
generation can opt into the indirection later by calling
``NativeSymbolTable.resolve`` before each invocation.

This is the first-generation, quiescent-point manager.  The newer
``runtime.hotpatch`` package implements the owned-page trampoline, future-entry
frame migration, reactor, RCU, and journal capabilities routed by
``runtime.capabilities``; prefer it for new work.  Both implementations keep
their distinct, documented boundaries and are not interchangeable.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import threading
import types
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compiler.core.ast_nodes import (
    ConstDecl,
    EnumDecl,
    FnDecl,
    ImplBlock,
    Node,
    Program,
    StructDecl,
    TraitDecl,
    TypeAlias,
    TypeName,
    UseDecl,
)
from runtime.diagnostics import PillarError

# ---------------------------------------------------------------------------
# AST diffing
# ---------------------------------------------------------------------------


class ChangeKind(str, Enum):
    BODY = "body"
    SIGNATURE = "signature"
    STATE_LAYOUT = "state_layout"
    CONSTANT = "constant"
    ADDED = "added"
    REMOVED = "removed"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class ItemChange:
    key: tuple[Any, ...]
    name: str
    kind: ChangeKind
    target: str | None = None
    detail: str = ""
    span: tuple[int, int] | None = None

    @property
    def hot_swappable(self) -> bool:
        return self.kind in {ChangeKind.BODY, ChangeKind.ADDED, ChangeKind.CONSTANT}


@dataclass
class PatchPlan:
    changes: list[ItemChange] = field(default_factory=list)

    @property
    def hot_swappable(self) -> bool:
        return all(change.hot_swappable for change in self.changes)

    @property
    def requires_migration(self) -> bool:
        return any(change.kind is ChangeKind.STATE_LAYOUT for change in self.changes)

    @property
    def incompatible(self) -> list[ItemChange]:
        return [
            change
            for change in self.changes
            if change.kind
            in {
                ChangeKind.SIGNATURE,
                ChangeKind.REMOVED,
                ChangeKind.INCOMPATIBLE,
            }
        ]

    def summary(self) -> str:
        if not self.changes:
            return "no changes"
        return ", ".join(
            f"{change.name}:{change.kind.value}" for change in self.changes
        )


def _normalise(value: Any) -> Any:
    """Structural, location-independent form used by the diff."""
    if isinstance(value, Node):
        result: dict[str, Any] = {}
        for info in dataclasses.fields(value):
            if info.name in {"line", "col", "span", "source_lines"}:
                continue
            result[info.name] = _normalise(getattr(value, info.name))
        return (type(value).__name__, result)
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _normalise(item)) for key, item in value.items()))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _type_name(ty: Any) -> str:
    if isinstance(ty, TypeName):
        if ty.args:
            return f"{ty.name}<{', '.join(_type_name(arg) for arg in ty.args)}>"
        return ty.name
    return _normalise(ty)


def _fn_signature(fn: FnDecl) -> tuple[Any, ...]:
    return (
        tuple(
            (
                param.pattern.__class__.__name__,
                _normalise(param.pattern),
                _type_name(param.ty),
            )
            for param in fn.params
        ),
        _type_name(fn.ret),
        fn.is_async,
        fn.is_unsafe,
        tuple(generic.name for generic in fn.generics),
    )


def _fn_body(fn: FnDecl) -> Any:
    return _normalise(fn.body)


def _index_items(
    program: Program,
) -> dict[tuple[Any, ...], tuple[str, Any, str | None]]:
    """Map a stable key to ``(kind, item, owner_type)``."""
    index: dict[tuple[Any, ...], tuple[str, Any, str | None]] = {}
    impl_counts: dict[tuple[str, str], int] = {}
    for item in program.items:
        if isinstance(item, FnDecl):
            index[("fn", item.name)] = ("fn", item, None)
        elif isinstance(item, StructDecl):
            index[("struct", item.name)] = ("struct", item, None)
        elif isinstance(item, EnumDecl):
            index[("enum", item.name)] = ("enum", item, None)
        elif isinstance(item, TraitDecl):
            index[("trait", item.name)] = ("trait", item, None)
        elif isinstance(item, ConstDecl):
            index[("const", item.name)] = ("const", item, None)
        elif isinstance(item, TypeAlias):
            index[("type", item.name)] = ("type", item, None)
        elif isinstance(item, UseDecl):
            index[("use", item.path)] = ("use", item, None)
        elif isinstance(item, ImplBlock):
            owner = _type_name(item.ty)
            for nested in item.items:
                if isinstance(nested, FnDecl):
                    pair = (owner, nested.name)
                    count = impl_counts.get(pair, 0)
                    impl_counts[pair] = count + 1
                    key = ("method", owner, nested.name, count)
                    index[key] = ("method", nested, owner)
                else:
                    # A trait const/type nested in an impl is still a real
                    # definition; indexing it as unsupported prevents a patch
                    # from silently ignoring the change.
                    key = (
                        "other",
                        type(nested).__name__,
                        getattr(nested, "name", None) or "",
                        impl_counts.get((owner, type(nested).__name__), 0),
                    )
                    impl_counts[(owner, type(nested).__name__)] = key[-1] + 1
                    index[key] = ("other", nested, owner)
        else:
            # ModDecl / IncludeDecl / future item kinds: track them so adding
            # or removing one is reported rather than dropped from the plan.
            name = getattr(item, "name", None) or getattr(item, "path", "") or ""
            key = ("other", type(item).__name__, str(name))
            index[key] = ("other", item, None)
    return index


_SWAPPABLE_KINDS = frozenset(
    {"fn", "method", "const", "struct", "enum", "trait", "type", "use"}
)


def diff_programs(old: Program, new: Program) -> PatchPlan:
    """Return a structured, location-independent diff of two programs."""
    old_index = _index_items(old)
    new_index = _index_items(new)
    plan = PatchPlan()

    for key, (kind, new_item, owner) in new_index.items():
        old_entry = old_index.get(key)
        if old_entry is None:
            supported = kind in _SWAPPABLE_KINDS
            plan.changes.append(
                ItemChange(
                    key=key,
                    name=getattr(new_item, "name", key[-1] if key else "?"),
                    kind=ChangeKind.ADDED if supported else ChangeKind.INCOMPATIBLE,
                    target=owner,
                    detail=(
                        "" if supported else f"{kind} definitions are not hot-swappable"
                    ),
                    span=getattr(new_item, "span", None),
                )
            )
            continue
        _kind, old_item, _owner = old_entry
        if kind == "fn" or kind == "method":
            if _fn_signature(old_item) != _fn_signature(new_item):
                plan.changes.append(
                    ItemChange(
                        key=key,
                        name=new_item.name,
                        kind=ChangeKind.SIGNATURE,
                        target=owner,
                        detail="parameter, return, async, or generic signature changed",
                        span=new_item.span,
                    )
                )
            elif _fn_body(old_item) != _fn_body(new_item):
                plan.changes.append(
                    ItemChange(
                        key=key,
                        name=new_item.name,
                        kind=ChangeKind.BODY,
                        target=owner,
                        span=new_item.span,
                    )
                )
        elif kind == "struct":
            old_fields = [
                (name, _type_name(ty), pub) for name, ty, pub in old_item.fields
            ]
            new_fields = [
                (name, _type_name(ty), pub) for name, ty, pub in new_item.fields
            ]
            if old_fields != new_fields:
                plan.changes.append(
                    ItemChange(
                        key=key,
                        name=new_item.name,
                        kind=ChangeKind.STATE_LAYOUT,
                        detail=f"{len(old_fields)} -> {len(new_fields)} fields",
                        span=new_item.span,
                    )
                )
            elif [generic.name for generic in old_item.generics] != [
                generic.name for generic in new_item.generics
            ]:
                plan.changes.append(
                    ItemChange(
                        key=key,
                        name=new_item.name,
                        kind=ChangeKind.INCOMPATIBLE,
                        detail="generic parameters changed",
                        span=new_item.span,
                    )
                )
        elif kind == "const":
            if _normalise(old_item) != _normalise(new_item):
                plan.changes.append(
                    ItemChange(
                        key=key,
                        name=new_item.name,
                        kind=ChangeKind.CONSTANT,
                        span=new_item.span,
                    )
                )
        elif _normalise(old_item) != _normalise(new_item):
            plan.changes.append(
                ItemChange(
                    key=key,
                    name=getattr(new_item, "name", key[-1] if key else "?"),
                    kind=ChangeKind.INCOMPATIBLE,
                    detail=f"{kind} definitions are not hot-swappable",
                    span=getattr(new_item, "span", None),
                )
            )

    for key, (_kind, old_item, owner) in old_index.items():
        if key not in new_index:
            plan.changes.append(
                ItemChange(
                    key=key,
                    name=getattr(old_item, "name", key[-1] if key else "?"),
                    kind=ChangeKind.REMOVED,
                    target=owner,
                    detail=(
                        ""
                        if _kind in _SWAPPABLE_KINDS
                        else f"{_kind} definitions are not hot-swappable"
                    ),
                    span=getattr(old_item, "span", None),
                )
            )
    return plan


# ---------------------------------------------------------------------------
# Runtime state helpers
# ---------------------------------------------------------------------------


def _unwrap_function(value: Any) -> types.FunctionType | None:
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    return value if isinstance(value, types.FunctionType) else None


def _coerce_default(value: Any) -> Any:
    return copy.deepcopy(value)


_INTEGER_TYPES = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "isize",
        "usize",
        "int",
    }
)
_FLOAT_TYPES = frozenset({"f32", "f64", "float"})
_STRING_TYPES = frozenset({"String", "str", "&str", "char"})
_LIST_TYPES = frozenset({"Vec", "list", "HashSet", "set"})
_DICT_TYPES = frozenset({"HashMap", "dict"})


def _zero_for_type_name(name: str) -> Any:
    """Resolve the zero value of a textual type name."""
    if name in _INTEGER_TYPES:
        return 0
    if name == "bool":
        return False
    if name in _FLOAT_TYPES:
        return 0.0
    if name in _STRING_TYPES:
        return ""
    if name in _LIST_TYPES or name.startswith(("Vec<", "list[", "[")):
        return []
    if name in _DICT_TYPES or name.startswith(("HashMap<", "dict[")):
        return {}
    if name.startswith(("HashSet<", "set[")):
        return set()
    if name.startswith(("Option<", "Optional[")):
        return None
    return _MISSING


def _zero_for_type(ty: Any) -> Any:
    # The Python backend injects ``from __future__ import annotations`` into
    # generated modules, so dataclass field types are usually strings such as
    # ``'int'`` or ``'list[int]'``.  Accept runtime types, generic aliases,
    # and the original Polyglot TypeName nodes as well.
    if isinstance(ty, str):
        return _zero_for_type_name(ty.strip().strip("'\""))
    if ty is bool:
        return False
    if ty is int:
        return 0
    if ty is float:
        return 0.0
    if ty is str:
        return ""
    if ty is list:
        return []
    if ty is dict:
        return {}
    if ty is set:
        return set()
    origin = getattr(ty, "__origin__", None)
    if origin is not None and origin is not ty:
        return _zero_for_type(origin) if isinstance(origin, type) else _MISSING
    name = _type_name(ty)
    if not isinstance(name, str):
        # Complex/nested AST types normalise to tuples; they have no obvious
        # single zero value without an explicit migration default.
        return _MISSING
    return _zero_for_type_name(name)


_MISSING = object()


def _dataclass_field_types(cls: type) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {name: getattr(info, "type", None) for name, info in fields.items()}


@dataclass
class _FunctionState:
    function: types.FunctionType
    code: types.CodeType
    defaults: Any
    kwdefaults: Any
    annotations: dict[str, Any]
    doc: str | None


@dataclass
class _ClassState:
    cls: type
    attributes: dict[str, Any]


@dataclass
class _ObjectState:
    obj: Any
    values: dict[str, Any]


@dataclass
class _UndoState:
    program: Program
    generation: int
    functions: list[_FunctionState]
    classes: list[_ClassState]
    objects: list[_ObjectState]
    namespace_keys: set[str]
    namespace_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchResult:
    generation: int
    plan: PatchPlan
    swapped: int
    migrated_objects: int


class HotpatchManager:
    """Apply body-only patches to a live Python namespace.

    Parameters
    ----------
    program:
        The program that produced ``namespace``.
    namespace:
        The namespace returned by ``PythonBackend.compile``.  If omitted the
        manager compiles ``program`` itself.
    allow_field_drop:
        Permit struct fields to be removed during migration.  The default is
        to reject the patch so live state cannot be silently discarded.
    field_defaults:
        ``{field_name: value}`` used for newly added fields whose type has no
        obvious zero value.  The mapping may also be ``{Struct::field:
        value}`` to disambiguate.
    """

    def __init__(
        self,
        program: Program,
        namespace: dict[str, Any] | None = None,
        *,
        allow_field_drop: bool = False,
        field_defaults: dict[str, Any] | None = None,
    ) -> None:
        if namespace is None:
            from backends.python.translator import PythonBackend

            namespace = PythonBackend(trusted=True).compile(program)
        self._program = program
        self.namespace = namespace
        self.allow_field_drop = allow_field_drop
        self.field_defaults = dict(field_defaults or {})
        self.generation = 1
        self._lock = threading.RLock()
        self._live: list[weakref.ref[Any]] = []
        self._undo: list[_UndoState] = []
        self._active_calls = 0
        self.patch_count = 0
        self.migration_count = 0

    # -- live object registration -----------------------------------------

    def register(self, obj: Any) -> Any:
        """Track ``obj`` so struct layout changes can migrate it in place."""
        if not hasattr(obj, "__dict__"):
            raise PillarError(
                "E001",
                f"cannot register {type(obj).__name__}: it has no mutable __dict__",
                help_text="register instances of generated structs (dataclasses) only",
            )
        with self._lock:
            if any(ref() is obj for ref in self._live):
                return obj
            self._live.append(weakref.ref(obj))
        return obj

    def live_objects(self) -> list[Any]:
        alive: list[Any] = []
        with self._lock:
            for ref in self._live:
                obj = ref()
                if obj is not None:
                    alive.append(obj)
            self._live = [weakref.ref(obj) for obj in alive]
        return alive

    # -- execution barrier -------------------------------------------------

    @contextlib.contextmanager
    def execution(self) -> Iterator[None]:
        """Mark a region in which live frames must not be rewritten."""
        with self._lock:
            self._active_calls += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_calls -= 1

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self.execution():
            if name not in self.namespace:
                raise PillarError(
                    "E006",
                    f"namespace has no callable `{name}`",
                    help_text="compile a program that defines the function first",
                )
            return self.namespace[name](*args, **kwargs)

    # -- patching ----------------------------------------------------------

    def prepare(self, new_program: Program) -> PatchPlan:
        return diff_programs(self._program, new_program)

    def patch(
        self,
        new_program: Program,
        *,
        require_safe_point: bool = True,
        allow_field_drop: bool | None = None,
        field_defaults: dict[str, Any] | None = None,
    ) -> PatchResult:
        from backends.python.translator import PythonBackend

        with self._lock:
            # The plan and the quiescence check must be made with the same
            # lock held for the swap itself.  Otherwise a concurrent
            # execution() or patch() can slip in between the check and the
            # mutation and have its live frames rewritten.
            plan = diff_programs(self._program, new_program)
            incompatible = plan.incompatible
            if incompatible:
                first = incompatible[0]
                raise PillarError(
                    "E001",
                    f"patch is not ABI-compatible: {first.name} ({first.kind.value})",
                    reason=first.detail
                    or "a signature or top-level definition changed",
                    help_text="deploy a new process for signature/layout breaking changes",
                    line=0,
                    col=0,
                    span=first.span,
                )
            if require_safe_point and self._active_calls:
                raise PillarError(
                    "E003",
                    "hot patch requires a quiescent execution point",
                    reason=f"{self._active_calls} call(s) are still active",
                    help_text="finish or suspend the reactor before applying the patch",
                )

            new_namespace = PythonBackend(trusted=True).compile(new_program)
            undo = self._capture_undo(plan, capture_objects=plan.requires_migration)
            swaps = 0
            migrated = 0
            try:
                self._merge_support_names(new_namespace)
                for change in plan.changes:
                    if change.kind in {
                        ChangeKind.BODY,
                        ChangeKind.SIGNATURE,
                        ChangeKind.ADDED,
                    }:
                        if change.target is not None:
                            swaps += self._install_method(change, new_namespace)
                        else:
                            swaps += self._install_function(change, new_namespace)
                    elif change.kind is ChangeKind.CONSTANT:
                        if change.name in new_namespace:
                            self.namespace[change.name] = new_namespace[change.name]
                    elif change.kind is ChangeKind.STATE_LAYOUT:
                        migrated += self._migrate_struct(
                            change,
                            new_namespace,
                            allow_field_drop=(
                                self.allow_field_drop
                                if allow_field_drop is None
                                else allow_field_drop
                            ),
                            field_defaults=field_defaults,
                        )
                self._program = new_program
                self.generation += 1
                self.patch_count += 1
                self.migration_count += migrated
                self._undo.append(undo)
            except BaseException:
                self._restore(undo)
                raise
            return PatchResult(
                generation=self.generation,
                plan=plan,
                swapped=swaps,
                migrated_objects=migrated,
            )

    def rollback(self) -> int:
        """Undo the most recent patch and return the restored generation."""
        with self._lock:
            if not self._undo:
                raise PillarError(
                    "E006",
                    "no hot-patch snapshot is available",
                    help_text="apply at least one patch before rolling back",
                )
            undo = self._undo.pop()
            self._restore(undo)
            self.patch_count += 1
            return self.generation

    # -- internals ---------------------------------------------------------

    def _merge_support_names(self, new_namespace: dict[str, Any]) -> None:
        for key, value in new_namespace.items():
            if key in self.namespace:
                continue
            if key.startswith("_") or isinstance(value, (types.FunctionType, type)):
                self.namespace[key] = value

    def _install_function(
        self, change: ItemChange, new_namespace: dict[str, Any]
    ) -> int:
        old_value = self.namespace.get(change.name)
        new_value = new_namespace.get(change.name)
        if new_value is None:
            return 0
        old_fn = _unwrap_function(old_value)
        new_fn = _unwrap_function(new_value)
        if (
            old_fn is not None
            and new_fn is not None
            and (old_fn.__code__.co_freevars == new_fn.__code__.co_freevars)
        ):
            old_fn.__code__ = new_fn.__code__
            old_fn.__defaults__ = new_fn.__defaults__
            old_fn.__kwdefaults__ = new_fn.__kwdefaults__
            old_fn.__annotations__ = dict(new_fn.__annotations__)
            old_fn.__doc__ = new_fn.__doc__
            return 1
        self.namespace[change.name] = new_value
        return 1

    def _install_method(self, change: ItemChange, new_namespace: dict[str, Any]) -> int:
        target = change.target
        if target is None:
            return 0
        old_cls = self.namespace.get(target)
        new_cls = new_namespace.get(target)
        if not isinstance(old_cls, type) or not isinstance(new_cls, type):
            return 0
        old_value = old_cls.__dict__.get(change.name)
        new_value = new_cls.__dict__.get(change.name)
        if new_value is None:
            return 0
        old_fn = _unwrap_function(old_value)
        new_fn = _unwrap_function(new_value)
        if (
            old_fn is not None
            and new_fn is not None
            and (old_fn.__code__.co_freevars == new_fn.__code__.co_freevars)
        ):
            old_fn.__code__ = new_fn.__code__
            old_fn.__defaults__ = new_fn.__defaults__
            old_fn.__kwdefaults__ = new_fn.__kwdefaults__
            old_fn.__annotations__ = dict(new_fn.__annotations__)
            old_fn.__doc__ = new_fn.__doc__
            return 1
        setattr(old_cls, change.name, new_value)
        return 1

    def _migrate_struct(
        self,
        change: ItemChange,
        new_namespace: dict[str, Any],
        *,
        allow_field_drop: bool,
        field_defaults: dict[str, Any] | None,
    ) -> int:
        old_cls = self.namespace.get(change.name)
        new_cls = new_namespace.get(change.name)
        if not isinstance(old_cls, type) or not isinstance(new_cls, type):
            return 0
        old_fields = _dataclass_field_types(old_cls)
        new_fields = _dataclass_field_types(new_cls)
        added = [name for name in new_fields if name not in old_fields]
        removed = [name for name in old_fields if name not in new_fields]
        if removed and not allow_field_drop:
            raise PillarError(
                "E001",
                f"cannot remove fields from live struct `{change.name}`: {removed}",
                reason="field removal would silently discard live state",
                help_text="pass allow_field_drop=True or keep the fields",
                span=change.span,
            )

        resolved: dict[str, Any] = {}
        defaults = dict(self.field_defaults)
        defaults.update(field_defaults or {})
        for name in added:
            if name in defaults:
                resolved[name] = _coerce_default(defaults[name])
                continue
            scoped = f"{change.name}::{name}"
            if scoped in defaults:
                resolved[name] = _coerce_default(defaults[scoped])
                continue
            ty = new_fields.get(name)
            zero = _zero_for_type(ty) if ty is not None else _MISSING
            if zero is _MISSING:
                raise PillarError(
                    "E001",
                    f"new field `{change.name}::{name}` has no migration default",
                    reason="the field type has no obvious zero value",
                    help_text="supply field_defaults={...} to patch()",
                    span=change.span,
                )
            resolved[name] = zero

        self._copy_class_surface(old_cls, new_cls)
        migrated = 0
        for obj in self.live_objects():
            if not isinstance(obj, old_cls):
                continue
            for name in added:
                if not hasattr(obj, name):
                    setattr(obj, name, copy.deepcopy(resolved[name]))
            for name in removed:
                if hasattr(obj, name):
                    delattr(obj, name)
            migrated += 1
        return migrated

    def _copy_class_surface(self, old_cls: type, new_cls: type) -> None:
        skip = {
            "__dict__",
            "__weakref__",
            "__module__",
            "__qualname__",
            "__doc__",
        }
        for name, value in new_cls.__dict__.items():
            if name in skip:
                continue
            setattr(old_cls, name, value)

    def _capture_undo(
        self, plan: PatchPlan, *, capture_objects: bool = True
    ) -> _UndoState:
        functions: list[_FunctionState] = []
        classes: list[_ClassState] = []
        namespace_values: dict[str, Any] = {}
        seen_fns: set[int] = set()
        seen_classes: set[type] = set()

        def capture(value: Any) -> None:
            fn = _unwrap_function(value)
            if fn is not None and id(fn) not in seen_fns:
                seen_fns.add(id(fn))
                functions.append(
                    _FunctionState(
                        function=fn,
                        code=fn.__code__,
                        defaults=fn.__defaults__,
                        kwdefaults=fn.__kwdefaults__,
                        annotations=dict(fn.__annotations__),
                        doc=fn.__doc__,
                    )
                )

        for change in plan.changes:
            if change.target is None:
                value = self.namespace.get(change.name)
                capture(value)
                if _unwrap_function(value) is None:
                    namespace_values[change.name] = (
                        value if change.name in self.namespace else _MISSING
                    )
            else:
                cls = self.namespace.get(change.target)
                if isinstance(cls, type):
                    capture(cls.__dict__.get(change.name))
                    if cls not in seen_classes:
                        seen_classes.add(cls)
                        classes.append(
                            _ClassState(cls=cls, attributes=dict(cls.__dict__))
                        )

        objects: list[_ObjectState] = []
        if capture_objects:
            # Only layout patches mutate live instances.  Deep-copying every
            # registered object for a body-only patch would make even simple
            # patches fail when a struct happens to hold a non-copyable value
            # such as a lock or a native handle.
            for obj in self.live_objects():
                objects.append(_ObjectState(obj=obj, values=copy.deepcopy(vars(obj))))
        return _UndoState(
            program=self._program,
            generation=self.generation,
            functions=functions,
            classes=classes,
            objects=objects,
            namespace_keys=set(self.namespace),
            namespace_values=namespace_values,
        )

    def _restore(self, undo: _UndoState) -> None:
        for name in list(self.namespace):
            if name not in undo.namespace_keys:
                self.namespace.pop(name, None)
        for name, value in undo.namespace_values.items():
            if value is _MISSING:
                self.namespace.pop(name, None)
            else:
                self.namespace[name] = value
        for state in undo.functions:
            state.function.__code__ = state.code
            state.function.__defaults__ = state.defaults
            state.function.__kwdefaults__ = state.kwdefaults
            state.function.__annotations__ = dict(state.annotations)
            state.function.__doc__ = state.doc
        for class_state in undo.classes:
            current = set(class_state.cls.__dict__)
            for name in current - set(class_state.attributes):
                if name in {"__dict__", "__weakref__"}:
                    continue
                with contextlib.suppress(AttributeError):
                    delattr(class_state.cls, name)
            for name, value in class_state.attributes.items():
                if name in {"__dict__", "__weakref__"}:
                    continue
                setattr(class_state.cls, name, value)
        for object_state in undo.objects:
            object_state.obj.__dict__.clear()
            object_state.obj.__dict__.update(object_state.values)
        self._program = undo.program
        self.generation = undo.generation


# ---------------------------------------------------------------------------
# Native dispatch table
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    signature: Any
    callback: Any
    generation: int


class NativeSymbolTable:
    """Atomic function-pointer indirection for native hot-swapping.

    Generated native code resolves the current callback through this table.
    Because the table swap is a single reference assignment under a lock,
    concurrent callers observe either the old or the new ABI-compatible
    function, never a torn pointer.  Patching executable text is rejected.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, _Slot] = {}
        self.generation = 0
        self.swap_count = 0

    def register(self, name: str, signature: Any, implementation: Any) -> Any:
        with self._lock:
            if name in self._slots:
                raise PillarError(
                    "E001",
                    f"native symbol `{name}` is already registered",
                    help_text="use swap() to replace an existing symbol",
                )
            # Build the callback while the lock is held so concurrent
            # registrations of the same name cannot both run signature()
            # (which may allocate executable trampolines) for a losing slot.
            callback = signature(implementation)
            self.generation += 1
            self._slots[name] = _Slot(signature, callback, self.generation)
            return callback

    def swap(self, name: str, implementation: Any) -> Any:
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                raise PillarError(
                    "E006",
                    f"native symbol `{name}` is not registered",
                    help_text="register an ABI-compatible slot first",
                )
            callback = slot.signature(implementation)
            self.generation += 1
            self.swap_count += 1
            self._slots[name] = _Slot(slot.signature, callback, self.generation)
            return callback

    def resolve(self, name: str) -> Any:
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                raise PillarError(
                    "E006",
                    f"native symbol `{name}` is not registered",
                )
            return slot.callback

    def invoke(self, name: str, *args: Any) -> Any:
        return self.resolve(name)(*args)

    def patch_text(self, *_args: Any, **_kwargs: Any) -> None:
        raise PillarError(
            "E007",
            "in-place .so/.dylib text patching is not supported",
            reason="no portable, ABI-safe way to rewrite loaded executable text",
            help_text="compile native functions against NativeSymbolTable.resolve",
        )

    def snapshot(self) -> dict[str, tuple[int, Any]]:
        with self._lock:
            return {
                name: (slot.generation, slot.callback)
                for name, slot in self._slots.items()
            }


class RcuEpoch:
    """A cooperative epoch counter for AST readers and patch writers.

    This models RCU grace periods in Python. It does not detect kernel-style
    quiescent states; callers must release their read tokens explicitly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.epoch = 0
        self._readers: set[int] = set()
        self._next_token = 1

    def read(self) -> int:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._readers.add(token)
            return token

    def release(self, token: int) -> None:
        with self._lock:
            if token not in self._readers:
                raise PillarError(
                    "E003",
                    f"RCU read token {token} is not active",
                    help_text="release exactly the token returned by read()",
                )
            self._readers.discard(token)

    def advance(self) -> int:
        with self._lock:
            if self._readers:
                raise PillarError(
                    "E002",
                    f"{len(self._readers)} RCU reader(s) still hold a token",
                    help_text="release all read tokens before advancing the epoch",
                )
            self.epoch += 1
            return self.epoch


_DEFAULT_EPOCH = RcuEpoch()


def hotpatch_rcu_ast_epoch(
    action: str = "read",
    *,
    token: int | None = None,
    domain: RcuEpoch | None = None,
) -> int:
    """Enter/exit an AST read epoch or advance the patch epoch."""
    epoch = domain or _DEFAULT_EPOCH
    if action == "read" and token is None:
        return epoch.read()
    if action == "release" and token is not None:
        epoch.release(token)
        return epoch.epoch
    if action == "advance" and token is None:
        return epoch.advance()
    raise PillarError(
        "E003",
        f"invalid RCU epoch action {action!r}",
        reason="use read, release(token=...), or advance",
    )


__all__ = [
    "ChangeKind",
    "HotpatchManager",
    "ItemChange",
    "NativeSymbolTable",
    "PatchPlan",
    "PatchResult",
    "RcuEpoch",
    "diff_programs",
    "hotpatch_rcu_ast_epoch",
]
