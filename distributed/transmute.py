"""Dual-target serialization bridge for live Python and native state.

The bridge turns a graph of Python objects (including the dataclasses emitted
by the Python backend) or a :class:`NativeValue` snapshot of a PyO3 struct into
one self-describing, versioned byte stream.  It is the transport-neutral core
used by checkpoints, migration envelopes, and the control plane.

Design points
-------------

* **Self-describing.**  Structs carry their concrete type name and field
  *names*.  Decoding therefore tolerates added fields (ignored), removed
  fields (defaults applied), and reordered fields.
* **Cycle safe.**  Compound values get a memo id; back-references encode as a
  compact index.
* **Deterministic.**  Struct fields and set elements are sorted by their
  encoded representation, so two nodes with the same logical state produce
  byte-identical snapshots.
* **Bounded.**  Depth, item count, byte length, and varint width are capped
  before allocation.
* **Native aware.**  :class:`NativeLayout` describes the concrete struct/enum
  layout produced by monomorphization; :class:`NativeValue` is its portable
  value form.  See :mod:`distributed.native_bridge` for the AST adapter.

Anything that cannot be represented raises a structured ``E007`` (unsupported
type) or ``E008`` (malformed state) diagnostic rather than pickling it.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, dataclass, field
from typing import Any, Final

from distributed.diagnostics import DistributedError, wire_error
from distributed.wire import (
    KIND_CHECKPOINT,
    KIND_STATE,
    Reader,
    WireError,
    Writer,
    decode_frame,
    encode_frame,
)

STATE_VERSION: Final = 1
FLAG_TOLERANT: Final = 0x0001
DEFAULT_MAX_ITEMS: Final = 1_000_000
DEFAULT_MAX_DEPTH: Final = 256

# Value tags.  Never renumber an existing tag; append instead.
TAG_NONE: Final = 0
TAG_FALSE: Final = 1
TAG_TRUE: Final = 2
TAG_INT: Final = 3
TAG_FLOAT: Final = 4
TAG_STR: Final = 5
TAG_BYTES: Final = 6
TAG_LIST: Final = 7
TAG_TUPLE: Final = 8
TAG_DICT: Final = 9
TAG_SET: Final = 10
TAG_FROZENSET: Final = 11
TAG_STRUCT: Final = 12
TAG_ENUM: Final = 13
TAG_MEMO_REF: Final = 14
TAG_NATIVE: Final = 15
TAG_REPR: Final = 16
TAG_MAX: Final = TAG_REPR


# ---------------------------------------------------------------------------
# Type registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    has_default: bool = False
    default: Any = None
    default_factory: Callable[[], Any] | None = None

    def fallback(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.has_default:
            return self.default
        raise KeyError(self.name)


@dataclass(frozen=True)
class TypeSpec:
    name: str
    kind: str  # "struct" or "enum"
    cls: type | None = None
    fields: tuple[FieldSpec, ...] = ()
    variants: tuple[str, ...] = ()
    factory: Callable[[dict[str, Any]], Any] | None = None

    def field(self, name: str) -> FieldSpec | None:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        return None


_TYPES: dict[str, TypeSpec] = {}
_CLASSES: dict[type, str] = {}
_REGISTRY_LOCK = threading.RLock()


def register_type(
    cls: type,
    *,
    name: str | None = None,
    kind: str = "struct",
    fields: Iterable[str | tuple[str, Any]] | None = None,
    variants: Iterable[str] | None = None,
    factory: Callable[[dict[str, Any]], Any] | None = None,
    replace: bool = True,
) -> TypeSpec:
    """Register ``cls`` so it can be encoded and decoded by name.

    Dataclasses and ``enum.Enum`` subclasses register their fields/variants
    automatically.  Custom classes may pass an explicit ``factory`` that takes
    the decoded field dictionary.
    """
    resolved_name = name or cls.__name__

    if issubclass(cls, enum.Enum):
        resolved_kind = "enum"
        resolved_variants = tuple(cls.__members__)
        spec = TypeSpec(
            name=resolved_name,
            kind=resolved_kind,
            cls=cls,
            variants=resolved_variants,
        )
    else:
        if dataclasses.is_dataclass(cls):
            specs: list[FieldSpec] = []
            for item in dataclasses.fields(cls):
                if not item.init and item.default is MISSING:
                    continue
                specs.append(
                    FieldSpec(
                        name=item.name,
                        has_default=item.default is not MISSING,
                        default=(None if item.default is MISSING else item.default),
                        default_factory=(
                            None
                            if item.default_factory is MISSING
                            else item.default_factory
                        ),
                    )
                )
            spec = TypeSpec(
                name=resolved_name,
                kind="struct",
                cls=cls,
                fields=tuple(specs),
                factory=factory,
            )
        else:
            if fields is None and factory is None:
                raise DistributedError(
                    f"cannot register {cls.__name__}: not a dataclass or enum",
                    code="E003",
                    note="TypeRegistrationError",
                    help_text=(
                        "pass fields=[...] and a factory=... for custom classes"
                    ),
                )
            specs = []
            for entry in fields or ():
                if isinstance(entry, str):
                    specs.append(FieldSpec(entry))
                else:
                    field_name, default = entry
                    specs.append(
                        FieldSpec(field_name, has_default=True, default=default)
                    )
            spec = TypeSpec(
                name=resolved_name,
                kind=kind,
                cls=cls,
                fields=tuple(specs),
                variants=tuple(variants or ()),
                factory=factory,
            )

    with _REGISTRY_LOCK:
        existing = _TYPES.get(resolved_name)
        if existing is not None and not replace and existing.cls is not cls:
            raise DistributedError(
                f"type {resolved_name!r} is already registered",
                code="E003",
                note="TypeRegistrationError",
                help_text="pass replace=True to override the registration",
            )
        _TYPES[resolved_name] = spec
        _CLASSES[cls] = resolved_name
    return spec


def _auto_register(cls: type) -> TypeSpec:
    with _REGISTRY_LOCK:
        name = _CLASSES.get(cls)
        if name is not None:
            return _TYPES[name]
    if issubclass(cls, enum.Enum) or dataclasses.is_dataclass(cls):
        return register_type(cls)
    raise DistributedError(
        f"type {cls.__module__}.{cls.__qualname__} is not wire-serializable",
        code="E007",
        note="UnsupportedTypeError",
        help_text=(
            "register the type with distributed.transmute.register_type(), "
            "or store a dataclass/NativeValue instead"
        ),
    )


def type_spec(name: str) -> TypeSpec | None:
    with _REGISTRY_LOCK:
        return _TYPES.get(name)


def registered_types() -> dict[str, TypeSpec]:
    with _REGISTRY_LOCK:
        return dict(_TYPES)


def reset_registry() -> None:
    """Clear registrations (tests and isolated embeds only)."""
    with _REGISTRY_LOCK:
        _TYPES.clear()
        _CLASSES.clear()


# ---------------------------------------------------------------------------
# Native layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeLayout:
    """The concrete layout of a monomorphized struct or enum."""

    name: str
    kind: str = "struct"  # "struct" or "enum"
    fields: tuple[tuple[str, str], ...] = ()
    variants: tuple[str, ...] = ()
    generic_origin: str | None = None

    def field_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fields)


@dataclass(frozen=True)
class NativeValue:
    """A portable value of a :class:`NativeLayout`."""

    layout: str
    fields: dict[str, Any] = field(default_factory=dict)
    variant: str | None = None
    payload: Any = None

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)


class PolyglotVariant:
    """Stand-in for the Python backend's payload-enum variant object.

    The Python backend represents a payload enum as a tuple
    ``(variant, *values)`` where ``variant`` is a callable tag.  Identity
    cannot survive a network hop, so equality is defined by the variant name.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *values: Any) -> tuple[Any, ...]:
        return (self, *values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PolyglotVariant):
            return other.name == self.name
        return (
            type(other).__name__ == "_PythonEnumVariant"
            and getattr(other, "name", None) == self.name
        )

    def __hash__(self) -> int:
        return hash(("polyglot-variant", self.name))

    def __repr__(self) -> str:
        return f"PolyglotVariant({self.name!r})"


@dataclass(frozen=True)
class PolyglotEnumValue:
    """Wire form of ``(variant, *values)`` payload-enum tuples."""

    variant: str
    values: tuple[Any, ...] = ()


def _is_variant(value: Any) -> bool:
    return isinstance(value, PolyglotVariant) or (
        type(value).__name__ == "_PythonEnumVariant"
        and isinstance(getattr(value, "name", None), str)
    )


def _decode_polyglot_enum(fields: dict[str, Any]) -> tuple[Any, ...]:
    return (PolyglotVariant(fields["variant"]), *fields.get("values", ()))


register_type(
    PolyglotEnumValue,
    name="PolyglotEnumValue",
    factory=_decode_polyglot_enum,
)


_LAYOUTS: dict[str, NativeLayout] = {}
_LAYOUT_LOCK = threading.RLock()


def register_native_layout(
    layout: NativeLayout, *, replace: bool = True
) -> NativeLayout:
    if layout.kind not in {"struct", "enum"}:
        raise DistributedError(
            f"unknown native layout kind {layout.kind!r}",
            code="E003",
            note="TypeRegistrationError",
        )
    if layout.kind == "enum" and not layout.variants:
        raise DistributedError(
            f"enum layout {layout.name!r} has no variants",
            code="E003",
            note="TypeRegistrationError",
        )
    with _LAYOUT_LOCK:
        if layout.name in _LAYOUTS and not replace:
            raise DistributedError(
                f"native layout {layout.name!r} is already registered",
                code="E003",
                note="TypeRegistrationError",
            )
        _LAYOUTS[layout.name] = layout
    return layout


def native_layout(name: str) -> NativeLayout | None:
    with _LAYOUT_LOCK:
        return _LAYOUTS.get(name)


def native_layouts() -> dict[str, NativeLayout]:
    with _LAYOUT_LOCK:
        return dict(_LAYOUTS)


def reset_native_layouts() -> None:
    with _LAYOUT_LOCK:
        _LAYOUTS.clear()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _Encoder:
    def __init__(
        self,
        writer: Writer,
        *,
        tolerant: bool = False,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        self.writer = writer
        self.tolerant = tolerant
        self.max_depth = max_depth
        self.max_items = max_items
        self.memo: dict[int, int] = {}

    # -- helpers ----------------------------------------------------------
    def _items(self, count: int) -> None:
        if count > self.max_items:
            raise DistributedError(
                f"container has {count} items (limit {self.max_items})",
                note="StateTooLargeError",
                help_text="split the actor state or raise the configured limit",
            )

    def _memoize(self, value: Any) -> int | None:
        ident = id(value)
        existing = self.memo.get(ident)
        if existing is not None:
            return existing
        memo_id = len(self.memo) + 1
        self.memo[ident] = memo_id
        return None

    def value(self, value: Any, depth: int = 0) -> None:
        if depth > self.max_depth:
            raise DistributedError(
                f"value nesting exceeds {self.max_depth} levels",
                note="StateTooDeepError",
                help_text="flatten the graph or raise max_depth",
            )
        writer = self.writer

        if value is None:
            writer.u8(TAG_NONE)
        elif value is True:
            writer.u8(TAG_TRUE)
        elif value is False:
            writer.u8(TAG_FALSE)
        elif isinstance(value, enum.Enum):
            self._enum(value)
        elif isinstance(value, int):
            writer.u8(TAG_INT)
            writer.svarint(value)
        elif isinstance(value, float):
            writer.u8(TAG_FLOAT)
            writer.f64(value)
        elif isinstance(value, str):
            writer.u8(TAG_STR)
            writer.string(value)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            writer.u8(TAG_BYTES)
            writer.raw(value)
        elif isinstance(value, NativeValue):
            self._native(value, depth)
        elif isinstance(value, list):
            self._sequence(TAG_LIST, value, depth)
        elif isinstance(value, tuple) and value and _is_variant(value[0]):
            self._polyglot_enum(value)
        elif isinstance(value, tuple):
            self._sequence(TAG_TUPLE, value, depth)
        elif isinstance(value, frozenset):
            self._set(TAG_FROZENSET, value, depth)
        elif isinstance(value, set):
            self._set(TAG_SET, value, depth)
        elif isinstance(value, dict):
            self._dict(value, depth)
        else:
            self._struct(value, depth)

    # -- compounds --------------------------------------------------------
    def _sequence(self, tag: int, value: Any, depth: int) -> None:
        existing = self._memoize(value)
        if existing is not None:
            self.writer.u8(TAG_MEMO_REF)
            self.writer.varint(existing)
            return
        self._items(len(value))
        self.writer.u8(tag)
        self.writer.varint(self.memo[id(value)])
        self.writer.varint(len(value))
        for item in value:
            self.value(item, depth + 1)

    def _dict(self, value: Mapping[Any, Any], depth: int) -> None:
        existing = self._memoize(value)
        if existing is not None:
            self.writer.u8(TAG_MEMO_REF)
            self.writer.varint(existing)
            return
        self._items(len(value))
        self.writer.u8(TAG_DICT)
        self.writer.varint(self.memo[id(value)])
        self.writer.varint(len(value))
        for key, item in value.items():
            self.value(key, depth + 1)
            self.value(item, depth + 1)

    def _set(self, tag: int, value: Any, depth: int) -> None:
        existing = self._memoize(value)
        if existing is not None:
            self.writer.u8(TAG_MEMO_REF)
            self.writer.varint(existing)
            return
        self._items(len(value))
        ordered = sorted(value, key=_probe)
        self.writer.u8(tag)
        self.writer.varint(self.memo[id(value)])
        self.writer.varint(len(ordered))
        for item in ordered:
            self.value(item, depth + 1)

    def _enum(self, value: enum.Enum) -> None:
        spec = _auto_register(type(value))
        self.writer.u8(TAG_ENUM)
        self.writer.string(spec.name)
        self.writer.string(value.name)

    def _polyglot_enum(self, value: tuple[Any, ...]) -> None:
        variant = value[0]
        name = getattr(variant, "name", None)
        if not isinstance(name, str):
            raise DistributedError(
                "payload-enum variant has no name",
                note="StateMismatchError",
            )
        self._struct(PolyglotEnumValue(name, tuple(value[1:])), 0)

    def _native(self, value: NativeValue, depth: int) -> None:
        layout = native_layout(value.layout)
        if layout is None:
            raise DistributedError(
                f"native layout {value.layout!r} is not registered on this node",
                code="E006",
                note="MissingLayoutError",
                help_text=(
                    "call distributed.native_bridge.register_program_layouts() "
                    "with the same monomorphized program on both nodes"
                ),
                context={"layout": value.layout},
            )
        existing = self._memoize(value)
        if existing is not None:
            self.writer.u8(TAG_MEMO_REF)
            self.writer.varint(existing)
            return
        self.writer.u8(TAG_NATIVE)
        self.writer.varint(self.memo[id(value)])
        self.writer.string(layout.name)
        if layout.kind == "struct":
            self.writer.u8(0)
            known = set(layout.field_names())
            present = {key: item for key, item in value.fields.items()}
            missing = [name for name in layout.field_names() if name not in present]
            if missing:
                raise DistributedError(
                    f"native value {layout.name!r} is missing field(s): "
                    f"{', '.join(missing)}",
                    note="StateMismatchError",
                    help_text="construct the full native value before encoding",
                )
            extra = sorted(set(present) - known)
            self.writer.varint(len(known) + len(extra))
            for name in sorted(present):
                self.writer.string(name)
                self.value(present[name], depth + 1)
        else:
            if value.variant is None or value.variant not in layout.variants:
                raise DistributedError(
                    f"native enum {layout.name!r} has no variant {value.variant!r}",
                    note="StateMismatchError",
                    help_text="use one of: " + ", ".join(layout.variants),
                )
            self.writer.u8(1)
            self.writer.string(value.variant)
            if value.payload is None:
                self.writer.u8(0)
            else:
                self.writer.u8(1)
                self.value(value.payload, depth + 1)

    def _struct(self, value: Any, depth: int) -> None:
        spec = _auto_register(type(value))
        existing = self._memoize(value)
        if existing is not None:
            self.writer.u8(TAG_MEMO_REF)
            self.writer.varint(existing)
            return
        fields: dict[str, Any] = {}
        for item in spec.fields:
            try:
                fields[item.name] = getattr(value, item.name)
            except AttributeError:
                continue
        self.writer.u8(TAG_STRUCT)
        self.writer.varint(self.memo[id(value)])
        self.writer.string(spec.name)
        self.writer.varint(len(fields))
        for name in sorted(fields):
            self.writer.string(name)
            self.value(fields[name], depth + 1)


def _probe(value: Any) -> bytes:
    """Return a deterministic sort key for a set element."""
    writer = Writer()
    try:
        _Encoder(writer).value(value)
    except DistributedError, WireError:
        return repr(value).encode("utf-8", "backslashreplace")
    return writer.build()


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


_PENDING: Final = object()


class _Decoder:
    def __init__(
        self,
        reader: Reader,
        *,
        tolerant: bool = False,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        self.reader = reader
        self.tolerant = tolerant
        self.max_depth = max_depth
        self.max_items = max_items
        self.memo: list[Any] = []

    def value(self, depth: int = 0) -> Any:
        if depth > self.max_depth:
            raise DistributedError(
                f"value nesting exceeds {self.max_depth} levels",
                note="StateTooDeepError",
                help_text="reject the snapshot or raise max_depth",
            )
        tag = self.reader.u8()
        if tag > TAG_MAX:
            raise wire_error(
                f"unknown value tag {tag}",
                help_text="the writer used a newer format; upgrade this node",
            )
        if tag == TAG_NONE:
            return None
        if tag == TAG_TRUE:
            return True
        if tag == TAG_FALSE:
            return False
        if tag == TAG_INT:
            return self.reader.svarint()
        if tag == TAG_FLOAT:
            value = self.reader.f64()
            if not math.isfinite(value):
                # Keep the bits; callers can decide whether NaN/Inf is valid.
                return value
            return value
        if tag == TAG_STR:
            return self.reader.string()
        if tag == TAG_BYTES:
            return self.reader.raw()
        if tag == TAG_MEMO_REF:
            return self._memo_ref()
        if tag == TAG_REPR:
            return self.reader.string()
        if tag in (TAG_LIST, TAG_TUPLE, TAG_SET, TAG_FROZENSET):
            return self._sequence(tag, depth)
        if tag == TAG_DICT:
            return self._dict(depth)
        if tag == TAG_ENUM:
            return self._enum()
        if tag == TAG_STRUCT:
            return self._struct(depth)
        if tag == TAG_NATIVE:
            return self._native(depth)
        raise wire_error(f"unhandled value tag {tag}")

    # -- helpers ----------------------------------------------------------
    def _reserve(self, memo_id: int) -> int:
        if memo_id < 1:
            raise wire_error(f"invalid memo id {memo_id}")
        if memo_id > len(self.memo):
            self.memo.extend([_PENDING] * (memo_id - len(self.memo)))
        return memo_id

    def _store(self, memo_id: int, value: Any) -> None:
        self._reserve(memo_id)
        self.memo[memo_id - 1] = value

    def _memo_ref(self) -> Any:
        index = self.reader.varint()
        if not 1 <= index <= len(self.memo):
            raise wire_error(
                f"memo reference {index} is out of range (have {len(self.memo)})"
            )
        value = self.memo[index - 1]
        if value is _PENDING:
            raise wire_error(
                "cyclic reference to a value that is not constructed yet",
                note="StateMismatchError",
                help_text="avoid cycles through factory-constructed types",
            )
        return value

    def _count(self) -> int:
        count = self.reader.varint()
        if count > self.max_items:
            raise DistributedError(
                f"container has {count} items (limit {self.max_items})",
                note="StateTooLargeError",
            )
        return count

    def _sequence(self, tag: int, depth: int) -> Any:
        memo_id = self.reader.varint()
        count = self._count()
        if tag == TAG_SET:
            container: Any = set()
        else:
            container = []
        # Place the container before decoding children so cycles resolve.
        self._store(memo_id, container)
        items = [self.value(depth + 1) for _ in range(count)]
        if tag == TAG_LIST:
            container.extend(items)
            return container
        if tag == TAG_TUPLE:
            result = tuple(items)
            self._store(memo_id, result)
            return result
        if tag == TAG_SET:
            container.update(items)
            return container
        result = frozenset(items)
        self._store(memo_id, result)
        return result

    def _dict(self, depth: int) -> dict[Any, Any]:
        memo_id = self.reader.varint()
        count = self._count()
        container: dict[Any, Any] = {}
        self._store(memo_id, container)
        for _ in range(count):
            key = self.value(depth + 1)
            container[key] = self.value(depth + 1)
        return container

    def _enum(self) -> Any:
        type_name = self.reader.string()
        member = self.reader.string()
        spec = _require_spec(type_name)
        if spec.kind != "enum" or spec.cls is None:
            raise wire_error(
                f"{type_name!r} is not a registered enum",
                note="StateMismatchError",
            )
        try:
            return spec.cls[member]
        except KeyError as error:
            raise wire_error(
                f"enum {type_name!r} has no member {member!r}",
                note="StateMismatchError",
                help_text="both nodes must run the same enum definition",
                context={"type": type_name, "member": member},
            ) from error

    def _struct(self, depth: int) -> Any:
        memo_id = self.reader.varint()
        type_name = self.reader.string()
        spec = _require_spec(type_name)
        field_count = self._count()

        if spec.kind == "enum":
            raise wire_error(
                f"{type_name!r} is an enum, not a struct",
                note="StateMismatchError",
            )

        if spec.factory is None and spec.cls is not None:
            # Allocate and memoize before children so cycles resolve.
            instance = object.__new__(spec.cls)
            self._store(memo_id, instance)
            decoded: dict[str, Any] = {}
            for _ in range(field_count):
                name = self.reader.string()
                decoded[name] = self.value(depth + 1)
            return _populate(instance, spec, decoded)

        decoded = {}
        for _ in range(field_count):
            name = self.reader.string()
            decoded[name] = self.value(depth + 1)
        instance = _construct(spec, decoded)
        self._store(memo_id, instance)
        return instance

    def _native(self, depth: int) -> NativeValue:
        memo_id = self.reader.varint()
        layout_name = self.reader.string()
        layout = native_layout(layout_name)
        if layout is None:
            raise DistributedError(
                f"native layout {layout_name!r} is not registered on this node",
                code="E006",
                note="MissingLayoutError",
                help_text=(
                    "call distributed.native_bridge.register_program_layouts() "
                    "with the same monomorphized program on both nodes"
                ),
                context={"layout": layout_name},
            )
        kind = self.reader.u8()
        if kind == 0:
            if layout.kind != "struct":
                raise wire_error(
                    f"layout {layout_name!r} is an enum, not a struct",
                    note="StateMismatchError",
                )
            field_count = self._count()
            fields: dict[str, Any] = {}
            for _ in range(field_count):
                name = self.reader.string()
                fields[name] = self.value(depth + 1)
            missing = [name for name in layout.field_names() if name not in fields]
            if missing:
                raise DistributedError(
                    f"native value {layout_name!r} is missing field(s): "
                    f"{', '.join(missing)}",
                    note="StateMismatchError",
                    help_text="both nodes must use the same concrete layout",
                )
            value = NativeValue(layout=layout_name, fields=fields)
            self._store(memo_id, value)
            return value
        if kind == 1:
            variant = self.reader.string()
            if variant not in layout.variants:
                raise wire_error(
                    f"native enum {layout_name!r} has no variant {variant!r}",
                    note="StateMismatchError",
                )
            payload = self.value(depth + 1) if self.reader.u8() else None
            value = NativeValue(layout=layout_name, variant=variant, payload=payload)
            self._store(memo_id, value)
            return value
        raise wire_error(f"unknown native value kind {kind}")


def _require_spec(name: str) -> TypeSpec:
    spec = type_spec(name)
    if spec is not None:
        return spec
    raise DistributedError(
        f"type {name!r} is not registered on this node",
        code="E006",
        note="MissingTypeError",
        help_text=(
            "import/register the type with distributed.transmute.register_type() "
            "before decoding the snapshot"
        ),
        context={"type": name},
    )


def _construct(spec: TypeSpec, decoded: Mapping[str, Any]) -> Any:
    if spec.factory is not None:
        try:
            return spec.factory(dict(decoded))
        except Exception as error:
            raise wire_error(
                f"factory for {spec.name!r} rejected the decoded fields",
                note="StateMismatchError",
                reason=f"{type(error).__name__}: {error}",
            ) from error
    if spec.cls is None:
        raise wire_error(
            f"type {spec.name!r} has no class or factory",
            note="StateMismatchError",
        )
    instance = object.__new__(spec.cls)
    return _populate(instance, spec, decoded)


def _populate(instance: Any, spec: TypeSpec, decoded: Mapping[str, Any]) -> Any:
    values: dict[str, Any] = {}
    for item in spec.fields:
        if item.name in decoded:
            values[item.name] = decoded[item.name]
            continue
        try:
            values[item.name] = item.fallback()
        except KeyError as error:
            raise wire_error(
                f"state is missing required field {item.name!r} of {spec.name!r}",
                note="StateMismatchError",
                help_text=(
                    "both nodes must use compatible layouts, or the field must "
                    "have a default"
                ),
                context={"type": spec.name, "field": item.name},
            ) from error

    for name, value in values.items():
        object.__setattr__(instance, name, value)
    # Dataclass post-init hooks may enforce invariants; run them once.
    post_init = getattr(instance, "__post_init__", None)
    if callable(post_init):
        try:
            post_init()
        except Exception as error:
            raise wire_error(
                f"{spec.name}.__post_init__ rejected the decoded state",
                note="StateMismatchError",
                reason=f"{type(error).__name__}: {error}",
            ) from error
    return instance


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def encode_state(
    value: Any,
    *,
    kind: int = KIND_STATE,
    tolerant: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> bytes:
    """Encode ``value`` into a CRC-checked frame."""
    writer = Writer()
    writer.u16(STATE_VERSION)
    writer.u16(FLAG_TOLERANT if tolerant else 0)
    _Encoder(
        writer,
        tolerant=tolerant,
        max_depth=max_depth,
        max_items=max_items,
    ).value(value)
    return encode_frame(kind, writer.build())


def decode_state(
    data: bytes,
    *,
    expected_kind: int | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> Any:
    """Decode a frame produced by :func:`encode_state`."""
    try:
        frame = decode_frame(data)
    except WireError as error:
        raise wire_error(str(error), reason=error.reason) from error
    if expected_kind is not None and frame.kind != expected_kind:
        raise wire_error(
            f"unexpected frame kind {frame.kind}; expected {expected_kind}",
            note="ProtocolError",
        )
    reader = Reader(frame.payload)
    version = reader.u16()
    flags = reader.u16()
    if version > STATE_VERSION:
        raise wire_error(
            f"state format v{version} is newer than supported v{STATE_VERSION}",
            help_text="upgrade this node before joining the cluster",
        )
    try:
        value = _Decoder(
            reader,
            tolerant=bool(flags & FLAG_TOLERANT),
            max_depth=max_depth,
            max_items=max_items,
        ).value()
        reader.expect_end()
    except WireError as error:
        raise wire_error(str(error), reason=error.reason) from error
    return value


def encode_checkpoint(value: Any, *, tolerant: bool = False) -> bytes:
    return encode_state(value, kind=KIND_CHECKPOINT, tolerant=tolerant)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_deterministic(value: Any) -> bool:
    """Debug helper: encode twice and compare (used by tests)."""
    return encode_state(value) == encode_state(value)


def copy_state(value: Any) -> Any:
    """Round-trip a value through the wire codec."""
    return decode_state(encode_state(value))


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITEMS",
    "FLAG_TOLERANT",
    "STATE_VERSION",
    "TAG_BYTES",
    "TAG_DICT",
    "TAG_ENUM",
    "TAG_FLOAT",
    "TAG_FROZENSET",
    "TAG_INT",
    "TAG_LIST",
    "TAG_MEMO_REF",
    "TAG_NATIVE",
    "TAG_NONE",
    "TAG_SET",
    "TAG_STR",
    "TAG_STRUCT",
    "TAG_TRUE",
    "TAG_TUPLE",
    "FieldSpec",
    "NativeLayout",
    "NativeValue",
    "PolyglotEnumValue",
    "PolyglotVariant",
    "TypeSpec",
    "copy_state",
    "decode_state",
    "digest_bytes",
    "encode_checkpoint",
    "encode_state",
    "native_layout",
    "native_layouts",
    "register_native_layout",
    "register_type",
    "registered_types",
    "reset_native_layouts",
    "reset_registry",
    "type_spec",
]
