"""Bridge between the Polyglot AST/native backend and the wire codec.

The native backend compiles monomorphized structs and enums to Rust structs
behind PyO3.  Raw Rust addresses, vtables, and pointers cannot cross a machine
boundary, so the bridge transports the *concrete layout plus field values* and
rebuilds the value through the extension module's public constructor on the
target node.

Typical use::

    program = monomorphize(parse(lex(source)))
    register_program_layouts(program)

    payload = encode_state(to_native_value("Box_i64", box))
    ...
    box = from_native_value(namespace, decode_state(payload))

The same layout registry is shared with :mod:`distributed.transmute`, so a
snapshot containing a mix of Python dataclasses and native values stays one
self-describing frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ast_nodes import EnumDecl, Program, StructDecl, type_display_name
from distributed.diagnostics import DistributedError
from distributed.transmute import (
    NativeLayout,
    NativeValue,
    native_layout,
    register_native_layout,
)


class Namespace(Protocol):
    def __getitem__(self, name: str) -> Any: ...


def layouts_from_program(program: Program) -> dict[str, NativeLayout]:
    """Return concrete struct/enum layouts declared by ``program``.

    ``program`` should be monomorphized first; generic declarations are kept
    under their original names so callers can inspect them, but only concrete
    declarations are safe to migrate.
    """
    layouts: dict[str, NativeLayout] = {}
    for item in program.items:
        if isinstance(item, StructDecl):
            fields = tuple(
                (name, type_display_name(ty)) for name, ty, _pub in item.fields
            )
            layouts[item.name] = NativeLayout(
                name=item.name,
                kind="struct",
                fields=fields,
                generic_origin=_generic_origin(item.name, item.generics),
            )
        elif isinstance(item, EnumDecl):
            variants = tuple(variant.name for variant in item.variants)
            layouts[item.name] = NativeLayout(
                name=item.name,
                kind="enum",
                variants=variants,
                generic_origin=_generic_origin(item.name, item.generics),
            )
    return layouts


def _generic_origin(name: str, generics: list[Any]) -> str | None:
    if not generics:
        return None
    # Concrete names are ``Base_Arg1_Arg2``; recover the base conservatively.
    return name.split("_", 1)[0]


def register_program_layouts(
    program: Program, *, replace: bool = True
) -> dict[str, NativeLayout]:
    """Register every concrete layout in ``program`` on this node."""
    layouts = layouts_from_program(program)
    for layout in layouts.values():
        register_native_layout(layout, replace=replace)
    return layouts


def to_native_value(layout_name: str, obj: Any) -> NativeValue:
    """Convert a live backend object into its portable ``NativeValue`` form.

    Structs are read by field name; enum variants accept a variant string or a
    ``(variant, *payload)`` tuple.  A ``NativeValue`` is returned unchanged.
    """
    if isinstance(obj, NativeValue):
        return obj
    layout = native_layout(layout_name)
    if layout is None:
        raise DistributedError(
            f"native layout {layout_name!r} is not registered",
            code="E006",
            note="MissingLayoutError",
            help_text="call register_program_layouts(monomorphized_program) first",
            context={"layout": layout_name},
        )
    if layout.kind == "struct":
        fields: dict[str, Any] = {}
        for name, _type_name in layout.fields:
            if isinstance(obj, Mapping):
                if name not in obj:
                    raise DistributedError(
                        f"native struct {layout_name!r} is missing field {name!r}",
                        note="StateMismatchError",
                    )
                fields[name] = obj[name]
            else:
                if not hasattr(obj, name):
                    raise DistributedError(
                        f"native struct {layout_name!r} has no attribute {name!r}",
                        note="StateMismatchError",
                        help_text=(
                            "the Python and native layouts disagree; rebuild "
                            "both from the same monomorphized program"
                        ),
                    )
                fields[name] = getattr(obj, name)
        return NativeValue(layout=layout_name, fields=fields)

    variant, payload = _split_variant(obj)
    if variant not in layout.variants:
        raise DistributedError(
            f"native enum {layout_name!r} has no variant {variant!r}",
            note="StateMismatchError",
            help_text="expected one of: " + ", ".join(layout.variants),
        )
    return NativeValue(layout=layout_name, variant=variant, payload=payload)


def _split_variant(obj: Any) -> tuple[str, Any]:
    if isinstance(obj, str):
        return obj, None
    if isinstance(obj, tuple) and obj:
        first = obj[0]
        name = getattr(first, "name", first if isinstance(first, str) else None)
        if isinstance(name, str):
            values = obj[1:]
            if not values:
                return name, None
            if len(values) == 1:
                return name, values[0]
            return name, tuple(values)
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name, None
    raise DistributedError(
        f"cannot determine the variant of {type(obj).__name__}",
        note="StateMismatchError",
        help_text="pass a variant name or a (variant, *payload) tuple",
    )


def from_native_value(namespace: Namespace, value: NativeValue) -> Any:
    """Rebuild a backend object from ``value`` using ``namespace``."""
    layout = native_layout(value.layout)
    if layout is None:
        raise DistributedError(
            f"native layout {value.layout!r} is not registered",
            code="E006",
            note="MissingLayoutError",
        )
    try:
        cls = namespace[layout.name]
    except KeyError as error:
        raise DistributedError(
            f"namespace has no native type {layout.name!r}",
            code="E006",
            note="MissingTypeError",
            help_text="compile the same monomorphized program on this node",
        ) from error

    if layout.kind == "struct":
        try:
            return cls(**value.fields)
        except TypeError:
            pass
        ordered = [value.fields[name] for name, _ in layout.fields]
        try:
            return cls(*ordered)
        except TypeError:
            pass
        try:
            instance = cls.__new__(cls)
        except TypeError as error:
            raise DistributedError(
                f"cannot reconstruct native struct {layout.name!r}",
                code="E007",
                note="UnsupportedNativeTypeError",
                reason=f"{type(error).__name__}: {error}",
                help_text="expose a public constructor for migration",
            ) from error
        for name, item in value.fields.items():
            setattr(instance, name, item)
        return instance

    if value.variant is None:
        raise DistributedError(
            f"native enum {layout.name!r} has no variant",
            note="StateMismatchError",
        )
    try:
        member = getattr(cls, value.variant)
    except AttributeError as error:
        raise DistributedError(
            f"native enum {layout.name!r} has no variant {value.variant!r}",
            note="StateMismatchError",
        ) from error
    if value.payload is None:
        return member
    if callable(member):
        try:
            return member(value.payload)
        except TypeError:
            pass
    if isinstance(value.payload, tuple):
        return member(*value.payload)
    return member


@dataclass
class NativeBridge:
    """Export/import native values for one compiled namespace."""

    namespace: Namespace
    layouts: dict[str, NativeLayout]

    @classmethod
    def from_program(cls, program: Program, namespace: Namespace) -> NativeBridge:
        return cls(namespace=namespace, layouts=register_program_layouts(program))

    def export(self, layout_name: str, obj: Any) -> NativeValue:
        return to_native_value(layout_name, obj)

    def import_value(self, value: NativeValue) -> Any:
        return from_native_value(self.namespace, value)

    def copy(self, layout_name: str, obj: Any) -> Any:
        """Round-trip a native value through the layout bridge."""
        return self.import_value(self.export(layout_name, obj))


__all__ = [
    "NativeBridge",
    "from_native_value",
    "layouts_from_program",
    "register_program_layouts",
    "to_native_value",
]
