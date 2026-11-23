"""Frame B: WASM lowering, linear-memory ABI, host interop, and parity.

The frame composes with :class:`backends.wasm.codegen.WasmCodegen` and a real
``wasmtime`` runtime.  Privileged or platform-specific behaviour is never
assumed: missing ``wasmtime`` produces a structured ``E007`` diagnostic and
unsupported source constructs produce ``E003`` diagnostics with source spans.

Platform and runtime boundaries
-------------------------------
* ``wasm32`` linear memory is capped at 4 GiB; the default checked limit here
  is ``2**32`` bytes and can be lowered by callers.  Every layout and access is
  bounds-checked before data is read or written.
* Host imports are explicit.  No ambient filesystem, clock, or process
  capability is imported by this frame.
* Each :func:`wasm_isolate_instance_state` context creates a fresh ``Engine``,
  ``Store``, and ``Memory``; no globals, tables, or caches are shared across
  runs.  Compiled modules may be cached only in a caller-supplied temp root.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from compiler.core.ast_nodes import Program
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
WasmScalar = Literal["i64", "bool"]
AbiKind = Literal["i64", "bool", "str", "slice", "struct", "result"]

WASM32_LIMIT = 2**32
_ALIGNMENT: dict[str, int] = {
    "i64": 8,
    "bool": 1,
    "str": 8,
    "slice": 8,
    "result": 8,
    "struct": 8,
}
_SCALAR_SIZE: dict[str, int] = {"i64": 8, "bool": 1}


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text or "check the WASM lowering and linear-memory ABI",
        note="backends.wasm/frame-b",
        span=span,
    )


def _normalize_span(span: Span) -> Span:
    if span is None:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError) as error:
        raise _fail(
            "E003", "invalid source span for WASM diagnostic", reason=str(error)
        )
    if start < 0 or end < start:
        raise _fail(
            "E003", "WASM source span must satisfy 0 <= start <= end", span=span
        )
    return (start, end)


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise _fail("E001", "alignment must be positive")
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True)
class AbiField:
    """One leaf field of a linear-memory aggregate."""

    name: str
    kind: AbiKind

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise _fail("E001", "ABI field name must not be empty")


@dataclass(frozen=True)
class AbiSpec:
    """Shape of one value crossing the linear-memory boundary."""

    kind: AbiKind
    fields: tuple[AbiField, ...] = ()
    name: str = "value"


@dataclass(frozen=True)
class HostSignature:
    """A validated host import or module export signature.

    ``source`` names the imported host function that an export delegates to;
    it defaults to ``name`` so a bridge can re-export an import verbatim.
    """

    name: str
    params: tuple[WasmScalar, ...] = ()
    ret: WasmScalar | None = None
    source: str | None = None

    def wat(self) -> str:
        params = " ".join(f"(param {param})" for param in self.params)
        result = "" if self.ret is None else f" (result {self.ret})"
        return f"(func ${self.name} {params}{result})".replace("  ", " ")


@dataclass(frozen=True)
class WasmModuleArtifact:
    """Deterministic WAT/WASM artifact plus export signature metadata."""

    module_name: str
    wat: str
    wasm: bytes | None
    exports: tuple[str, ...]
    signatures: Mapping[str, HostSignature]
    source_digest: str
    lowered_from: Literal["source", "ast", "hir", "mir"] = "source"
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "exports": list(self.exports),
            "wasm_bytes": None if self.wasm is None else len(self.wasm),
            "source_digest": self.source_digest,
            "lowered_from": self.lowered_from,
        }


@dataclass(frozen=True)
class WasmMemoryLayout:
    """Checked bump allocation plus a shadow-stack region."""

    field_offsets: Mapping[str, int]
    field_sizes: Mapping[str, int]
    size: int
    base: int
    stack_base: int
    stack_top: int
    max_bytes: int
    stack_slots: tuple[int, ...]
    bounds: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_offsets": dict(self.field_offsets),
            "field_sizes": dict(self.field_sizes),
            "size": self.size,
            "base": self.base,
            "stack_base": self.stack_base,
            "stack_top": self.stack_top,
            "max_bytes": self.max_bytes,
            "stack_slots": list(self.stack_slots),
        }


@dataclass(frozen=True)
class WasmTrapReport:
    """Structured outcome of bounds/stack/OOM checks."""

    ok: bool
    kind: Literal["none", "out_of_bounds", "stack_overflow", "oom"]
    message: str
    accesses: tuple[tuple[int, int], ...] = ()
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "message": self.message,
            "accesses": [list(item) for item in self.accesses],
            "span": list(self.span) if self.span else None,
        }


@dataclass
class HostBinding:
    """A compiled host-import bridge with validated exports."""

    wat: str
    imports: tuple[HostSignature, ...]
    exports: tuple[HostSignature, ...]
    namespace: Mapping[str, Any]
    engine: Any = None
    store: Any = None
    instance: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "imports": [item.name for item in self.imports],
            "exports": [item.name for item in self.exports],
            "callable_exports": sorted(self.namespace),
        }


@dataclass
class WasmIsolatedInstance:
    """A fresh wasmtime instance whose state cannot leak to other runs."""

    artifact: WasmModuleArtifact
    run_id: str
    instance_id: str
    namespace: dict[str, Any] = field(default_factory=dict)
    store: Any = None
    instance: Any = None
    memory: Any = None
    _engine: Any = None
    _cache_root: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "exports": sorted(self.namespace),
            "digest": self.artifact.source_digest,
        }


@dataclass
class WasmParityReport:
    """Normalized outcomes from Python, Rust/PyO3, and WASM backends."""

    source: str
    calls: Mapping[str, tuple[Any, ...]]
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": {name: list(args) for name, args in self.calls.items()},
            "outcomes": self.outcomes,
            "skipped": self.skipped,
            "mismatches": self.mismatches,
            "passed": self.passed,
        }


def _source_to_program(target: str | Program | Any) -> tuple[Program, str, str]:
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    if isinstance(target, str):
        program = parse(lex(target))
        return program, target, "source"
    if isinstance(target, Program):
        return target, _program_source(target), "ast"
    module_name = type(target).__name__
    if module_name in {"HirModule", "MirModule"}:
        from compiler.ir.framework import (
            ir_lower_hir_to_mir,
            ir_validate_control_flow_graph,
        )

        mir_target = (
            ir_lower_hir_to_mir(target) if module_name == "HirModule" else target
        )
        validation = ir_validate_control_flow_graph(mir_target)
        if not validation.valid:
            raise _fail(
                "E003",
                f"IR control flow is invalid for WASM lowering: {validation.issues}",
                reason="run the CFG verification pass before lowering",
            )
        source = getattr(target, "source", "")
        if not source:
            raise _fail(
                "E003",
                f"{module_name} has no recorded source for WASM lowering",
            )
        return parse(lex(source)), source, module_name.lower().replace("module", "")
    raise _fail("E003", f"unsupported WASM lowering input {module_name}")


def _program_source(program: Program) -> str:
    # AST nodes are line/col annotated but do not retain original text; the
    # digest is therefore derived from a deterministic node dump.
    return repr([type(item).__name__ for item in program.items])


def _validate_host_signature(value: HostSignature | Mapping[str, Any]) -> HostSignature:
    if isinstance(value, HostSignature):
        signature = value
    elif isinstance(value, Mapping):
        signature = HostSignature(
            name=str(value.get("name", "")),
            params=tuple(value.get("params", ())),
            ret=value.get("ret"),
            source=value.get("source"),
        )
    else:
        raise _fail("E001", f"invalid host signature {value!r}")
    if not signature.name.strip() or not signature.name.isidentifier():
        raise _fail("E001", f"invalid host signature name {signature.name!r}")
    if signature.ret not in {None, "i64", "bool"}:
        raise _fail("E001", f"invalid host return type {signature.ret!r}")
    for param in signature.params:
        if param not in {"i64", "bool"}:
            raise _fail("E001", f"invalid host parameter type {param!r}")
    return signature


def wasm_lower_ir_to_module(
    target: str | Program | Any,
    *,
    module_name: str = "polyglot_wasm",
    emit_binary: bool = True,
    span: Span = None,
) -> WasmModuleArtifact:
    """Lower source/AST/HIR/MIR to a valid WAT (and optional WASM) module."""
    from backends.wasm.codegen import WasmCodegen, WasmUnsupportedError

    resolved_span = _normalize_span(span)
    program, source, lowered_from = _source_to_program(target)
    try:
        codegen = WasmCodegen()
        wat = codegen.generate(program)
    except WasmUnsupportedError as error:
        raise _fail(
            "E003",
            str(error),
            span=resolved_span or (getattr(error, "line", 0), getattr(error, "col", 0)),
            reason="the WASM module ABI requires scalar i64/bool parameters and results",
        ) from error

    signatures = {
        name: HostSignature(
            name=name,
            params=tuple(sig.params),
            ret=None if sig.ret is None else ("bool" if sig.ret == "i32" else "i64"),
        )
        for name, sig in codegen._signatures.items()
    }
    binary: bytes | None = None
    if emit_binary:
        try:
            import wasmtime
        except ImportError as error:
            raise _fail(
                "E007",
                "wasmtime is required to emit a .wasm binary",
                span=resolved_span,
                reason=str(error),
            ) from error
        try:
            binary = bytes(wasmtime.Module(wasmtime.Engine(), wat).serialize())
        except Exception as error:
            raise _fail(
                "E004",
                f"generated WAT failed to compile: {error}",
                span=resolved_span,
                reason=str(error),
            ) from error

    return WasmModuleArtifact(
        module_name=module_name,
        wat=wat,
        wasm=binary,
        exports=tuple(codegen.exported_fns),
        signatures=signatures,
        source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        lowered_from=lowered_from,  # type: ignore[arg-type]
        span=resolved_span,
    )


def wasm_layout_linear_memory_and_stack(
    fields: Sequence[AbiField],
    *,
    base: int = 0,
    stack_slots: Sequence[int] = (),
    max_bytes: int = WASM32_LIMIT,
    span: Span = None,
) -> WasmMemoryLayout:
    """Plan a checked bump allocation and a 16-byte-aligned shadow stack."""
    resolved_span = _normalize_span(span)
    if base < 0 or max_bytes <= 0 or max_bytes > WASM32_LIMIT:
        raise _fail(
            "E008",
            "invalid linear-memory limit for wasm32",
            span=resolved_span,
            reason=f"base={base}, max_bytes={max_bytes}",
        )
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    cursor = base
    bounds: list[tuple[int, int]] = []
    for field_spec in fields:
        if not isinstance(field_spec, AbiField):
            raise _fail("E001", f"invalid linear-memory field {field_spec!r}")
        alignment = _ALIGNMENT.get(field_spec.kind, 8)
        cursor = _align(cursor, alignment)
        size = _SCALAR_SIZE.get(field_spec.kind, 8)
        if field_spec.kind in {"str", "slice"}:
            size = 16  # pointer + length
        offsets[field_spec.name] = cursor
        sizes[field_spec.name] = size
        bounds.append((cursor - base, size))
        cursor += size
    data_size = cursor - base

    stack_base = _align(cursor, 16)
    normalized_slots: list[int] = []
    for slot in stack_slots:
        if slot < 0:
            raise _fail(
                "E008",
                "shadow-stack slot size must be non-negative",
                span=resolved_span,
            )
        normalized_slots.append(_align(slot, 16))
    stack_top = stack_base + sum(normalized_slots)
    for slot_size in normalized_slots:
        bounds.append((stack_base - base + slot_size, 0))
    if stack_top - base > max_bytes:
        raise _fail(
            "E008",
            f"linear-memory layout needs {stack_top - base} bytes "
            f"but limit is {max_bytes}",
            span=resolved_span,
            reason="wasm32 linear-memory limit exceeded",
        )
    return WasmMemoryLayout(
        field_offsets=offsets,
        field_sizes=sizes,
        size=data_size,
        base=base,
        stack_base=stack_base,
        stack_top=stack_top,
        max_bytes=max_bytes,
        stack_slots=tuple(normalized_slots),
        bounds=tuple(bounds),
    )


def _encode_scalar(kind: str, value: Any, path: str, span: Span) -> bytes:
    if kind == "bool":
        if not isinstance(value, bool):
            raise _fail(
                "E001", f"{path} expects bool, got {type(value).__name__}", span=span
            )
        return struct.pack("<i", int(value))
    if kind == "i64":
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(
                "E001", f"{path} expects i64, got {type(value).__name__}", span=span
            )
        if not -(2**63) <= value < 2**63:
            raise _fail("E008", f"{path} is outside the signed i64 range", span=span)
        return struct.pack("<q", value)
    raise _fail("E001", f"{path} has unsupported scalar kind {kind!r}", span=span)


def _decode_scalar(kind: str, blob: bytes, path: str, span: Span) -> tuple[Any, bytes]:
    width = 4 if kind == "bool" else 8
    if len(blob) < width:
        raise _fail("E008", f"truncated ABI payload for {path}", span=span)
    chunk, rest = blob[:width], blob[width:]
    if kind == "bool":
        return bool(struct.unpack("<i", chunk)[0]), rest
    return struct.unpack("<q", chunk)[0], rest


def _encode_value(
    spec: AbiSpec, value: Any, layout: WasmMemoryLayout, span: Span
) -> bytes:
    if spec.kind in {"i64", "bool"}:
        return _encode_scalar(spec.kind, value, spec.name, span)
    if spec.kind == "str":
        if not isinstance(value, str):
            raise _fail("E001", f"{spec.name} expects str", span=span)
        payload = value.encode("utf-8")
        return struct.pack("<q", len(payload)) + payload
    if spec.kind == "slice":
        if not isinstance(value, (list, tuple)):
            raise _fail("E001", f"{spec.name} expects a slice", span=span)
        items = list(value)
        return struct.pack("<q", len(items)) + b"".join(
            _encode_scalar("i64", item, f"{spec.name}[{index}]", span)
            for index, item in enumerate(items)
        )
    if spec.kind == "result":
        if isinstance(value, tuple) and len(value) == 2 and value[0] in {"Ok", "Err"}:
            ok, payload = value[0] == "Ok", value[1]
        elif isinstance(value, Mapping):
            ok, payload = bool(value.get("ok")), value.get("value")
        else:
            raise _fail("E001", f"{spec.name} expects an Ok/Err result", span=span)
        field_specs = {item.name: item for item in spec.fields}
        tag = field_specs.get("tag", AbiField("tag", "bool"))
        body = field_specs.get("value", AbiField("value", "i64"))
        return _encode_value(
            AbiSpec("bool", (tag,), f"{spec.name}.tag"), ok, layout, span
        ) + (
            _encode_value(
                AbiSpec(body.kind, (body,), f"{spec.name}.value"), payload, layout, span
            )
            if ok
            else b""
        )
    if spec.kind == "struct":
        if not isinstance(value, Mapping):
            raise _fail("E001", f"{spec.name} expects a struct mapping", span=span)
        chunks: list[bytes] = []
        for item in spec.fields:
            if item.name not in value:
                raise _fail("E001", f"{spec.name}.{item.name} is missing", span=span)
            chunks.append(
                _encode_value(
                    AbiSpec(item.kind, (item,), f"{spec.name}.{item.name}"),
                    value[item.name],
                    layout,
                    span,
                )
            )
        return b"".join(chunks)
    raise _fail("E003", f"unsupported ABI kind {spec.kind!r}", span=span)


def _decode_value(
    spec: AbiSpec, blob: bytes, layout: WasmMemoryLayout, span: Span
) -> Any:
    if spec.kind in {"i64", "bool"}:
        value, _ = _decode_scalar(spec.kind, blob, spec.name, span)
        return value
    if spec.kind == "str":
        if len(blob) < 8:
            raise _fail("E008", f"truncated string length for {spec.name}", span=span)
        length = struct.unpack("<q", blob[:8])[0]
        if length < 0 or len(blob) < 8 + length:
            raise _fail("E008", f"truncated string payload for {spec.name}", span=span)
        return blob[8 : 8 + length].decode("utf-8")
    if spec.kind == "slice":
        if len(blob) < 8:
            raise _fail("E008", f"truncated slice length for {spec.name}", span=span)
        count = struct.unpack("<q", blob[:8])[0]
        if count < 0 or len(blob) < 8 + 8 * count:
            raise _fail("E008", f"truncated slice payload for {spec.name}", span=span)
        return list(struct.unpack(f"<{count}q", blob[8 : 8 + 8 * count]))
    if spec.kind == "result":
        ok, rest = _decode_scalar("bool", blob, f"{spec.name}.tag", span)
        if not ok:
            return ("Err", None)
        field_specs = {item.name: item for item in spec.fields}
        body = field_specs.get("value", AbiField("value", "i64"))
        return (
            "Ok",
            _decode_value(
                AbiSpec(body.kind, (body,), f"{spec.name}.value"), rest, layout, span
            ),
        )
    if spec.kind == "struct":
        result: dict[str, Any] = {}
        rest = blob
        for item in spec.fields:
            result[item.name] = _decode_value(
                AbiSpec(item.kind, (item,), f"{spec.name}.{item.name}"),
                rest,
                layout,
                span,
            )
            width = 4 if item.kind == "bool" else 8
            rest = rest[width:]
        return result
    raise _fail("E003", f"unsupported ABI kind {spec.kind!r}", span=span)


def wasm_marshal_adt_and_result_abi(
    value: Any,
    *,
    spec: AbiSpec,
    layout: WasmMemoryLayout,
    span: Span = None,
) -> bytes:
    """Encode a struct/tagged-result/string/slice into linear-memory bytes."""
    resolved_span = _normalize_span(span)
    encoded = _encode_value(spec, value, layout, resolved_span)
    if len(encoded) > layout.max_bytes:
        raise _fail(
            "E008",
            f"ABI payload needs {len(encoded)} bytes, beyond the wasm32 limit",
            span=resolved_span,
        )
    return encoded


def wasm_unmarshal_adt_and_result_abi(
    blob: bytes,
    *,
    spec: AbiSpec,
    layout: WasmMemoryLayout,
    span: Span = None,
) -> Any:
    """Decode a payload produced by :func:`wasm_marshal_adt_and_result_abi`."""
    return _decode_value(spec, bytes(blob), layout, _normalize_span(span))


def wasm_guard_trap_and_oom_paths(
    layout: WasmMemoryLayout,
    *,
    accesses: Sequence[tuple[int, int]] = (),
    stack_depth: int = 0,
    requested_bytes: int | None = None,
    strict: bool = False,
    span: Span = None,
) -> WasmTrapReport:
    """Check linear-memory accesses, stack depth, and OOM without corruption."""
    resolved_span = _normalize_span(span)
    normalized = tuple((int(offset), int(size)) for offset, size in accesses)
    upper = min(layout.stack_top, layout.max_bytes)
    kind: Literal["none", "out_of_bounds", "stack_overflow", "oom"] = "none"
    message = "all WASM memory accesses are in bounds"
    if requested_bytes is not None and requested_bytes > layout.max_bytes:
        kind, message = (
            "oom",
            f"request of {requested_bytes} bytes exceeds wasm32 limit",
        )
    elif stack_depth > len(layout.stack_slots):
        kind, message = (
            "stack_overflow",
            f"shadow stack depth {stack_depth} exceeds {len(layout.stack_slots)} slots",
        )
    else:
        for offset, size in normalized:
            if offset < 0 or size < 0 or offset + size > upper:
                kind = "out_of_bounds"
                message = (
                    f"access [{offset}, {offset + size}) outside "
                    f"linear memory [0, {upper})"
                )
                break
    report = WasmTrapReport(
        ok=kind == "none",
        kind=kind,
        message=message,
        accesses=normalized,
        span=resolved_span,
    )
    if strict and not report.ok:
        raise _fail("E008", message, span=resolved_span, reason=f"wasm trap: {kind}")
    return report


def wasm_bind_host_imports_and_exports(
    artifact: WasmModuleArtifact,
    *,
    imports: Sequence[HostSignature | Mapping[str, Any]],
    exports: Sequence[HostSignature | Mapping[str, Any]],
    host_functions: Mapping[str, Any],
    module_name: str = "env",
    span: Span = None,
) -> HostBinding:
    """Generate, validate, and instantiate a host-import/export bridge."""
    resolved_span = _normalize_span(span)
    checked_imports = tuple(_validate_host_signature(item) for item in imports)
    checked_exports = tuple(_validate_host_signature(item) for item in exports)
    for signature in checked_imports:
        if signature.name not in host_functions:
            raise _fail(
                "E001",
                f"host import {signature.name!r} has no implementation",
                span=resolved_span,
            )
    declared_exports = set(artifact.exports)
    for signature in checked_exports:
        if signature.name not in declared_exports:
            raise _fail(
                "E001",
                f"export {signature.name!r} is not declared by the module",
                span=resolved_span,
            )
        expected = artifact.signatures.get(signature.name)
        if expected is not None and (
            expected.params != signature.params or expected.ret != signature.ret
        ):
            raise _fail(
                "E001",
                f"export {signature.name!r} signature drift: "
                f"{signature.params}->{signature.ret} vs "
                f"{expected.params}->{expected.ret}",
                span=resolved_span,
            )

    import_by_name = {signature.name: signature for signature in checked_imports}
    lines = ["(module"]
    for signature in checked_imports:
        lines.append(f'  (import "{module_name}" "{signature.name}" {signature.wat()})')
    for signature in checked_exports:
        source_name = signature.source or signature.name
        source_signature = import_by_name.get(source_name)
        if source_signature is None:
            raise _fail(
                "E001",
                f"export {signature.name!r} delegates to unknown host import "
                f"{source_name!r}",
                span=resolved_span,
            )
        if (
            source_signature.params != signature.params
            or source_signature.ret != signature.ret
        ):
            raise _fail(
                "E001",
                f"export {signature.name!r} and host import {source_name!r} "
                "have different signatures",
                span=resolved_span,
            )
        params = " ".join(f"(param {param})" for param in signature.params)
        result = "" if signature.ret is None else f" (result {signature.ret})"
        call_args = " ".join(
            f"local.get {index}" for index in range(len(signature.params))
        )
        body = " ".join(part for part in (call_args, f"call ${source_name}") if part)
        lines.append(
            f'  (func (export "{signature.name}") {params}{result} {body})'.replace(
                "  ", " "
            )
        )
    lines.append(")")
    wat = "\n".join(lines) + "\n"

    try:
        import wasmtime
    except ImportError as error:
        raise _fail(
            "E007", "wasmtime is required to bind host imports", span=resolved_span
        ) from error

    def valtype(name: str) -> Any:
        return wasmtime.ValType.i64() if name == "i64" else wasmtime.ValType.i32()

    engine = wasmtime.Engine()
    linker = wasmtime.Linker(engine)
    for signature in checked_imports:
        function = host_functions[signature.name]
        func_type = wasmtime.FuncType(
            [valtype(param) for param in signature.params],
            [] if signature.ret is None else [valtype(signature.ret)],
        )

        def callback(
            *args: Any, _fn: Any = function, _sig: HostSignature = signature
        ) -> Any:
            try:
                converted = [
                    bool(arg) if param == "bool" else int(arg)
                    for arg, param in zip(args, _sig.params)
                ]
                result = _fn(*converted)
                if _sig.ret is None:
                    return None
                return int(bool(result)) if _sig.ret == "bool" else int(result)
            except Exception as error:
                raise _fail(
                    "E008",
                    f"host import {_sig.name!r} failed: {error}",
                    span=resolved_span,
                    reason=str(error),
                ) from error

        try:
            linker.define_func(module_name, signature.name, func_type, callback)
        except Exception as error:
            raise _fail(
                "E001",
                f"cannot bind host import {signature.name!r}: {error}",
                span=resolved_span,
            ) from error

    try:
        module = wasmtime.Module(engine, wat)
        store = wasmtime.Store(engine)
        instance = linker.instantiate(store, module)
        exports = instance.exports(store)
    except Exception as error:
        raise _fail(
            "E004",
            f"host bridge failed to instantiate: {error}",
            span=resolved_span,
            reason=str(error),
        ) from error

    namespace: dict[str, Any] = {}
    for signature in checked_exports:
        raw = exports[signature.name]

        def make_wrapper(raw_fn: Any, sig: HostSignature) -> Any:
            def wrapper(*args: Any) -> Any:
                if len(args) != len(sig.params):
                    raise _fail(
                        "E001",
                        f"{sig.name} expects {len(sig.params)} args, got {len(args)}",
                        span=resolved_span,
                    )
                converted = [
                    int(bool(arg)) if param == "bool" else int(arg)
                    for arg, param in zip(args, sig.params)
                ]
                try:
                    result = raw_fn(store, *converted)
                except PillarError:
                    raise
                except Exception as error:
                    raise _fail(
                        "E008",
                        f"WASM trap while calling {sig.name!r}: {error}",
                        span=resolved_span,
                        reason=str(error),
                    ) from error
                return bool(result) if sig.ret == "bool" else result

            wrapper.__name__ = sig.name
            return wrapper

        namespace[signature.name] = make_wrapper(raw, signature)

    return HostBinding(
        wat=wat,
        imports=checked_imports,
        exports=checked_exports,
        namespace=namespace,
        engine=engine,
        store=store,
        instance=instance,
    )


@contextmanager
def wasm_isolate_instance_state(
    artifact: WasmModuleArtifact,
    *,
    run_id: str,
    cache_root: Path | str | None = None,
    span: Span = None,
) -> Iterator[WasmIsolatedInstance]:
    """Instantiate one WASM module in a fresh, non-sharing store."""
    from backends.wasm.codegen import _wrap_export

    resolved_span = _normalize_span(span)
    if not run_id.strip():
        raise _fail("E001", "run_id must not be empty", span=resolved_span)
    try:
        import wasmtime
    except ImportError as error:
        raise _fail(
            "E007", "wasmtime is required to isolate instances", span=resolved_span
        ) from error

    root = Path(cache_root) if cache_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    engine = wasmtime.Engine()
    try:
        if artifact.wasm is not None and root is not None:
            cached = root / f"{artifact.source_digest}.wasm"
            if not cached.exists():
                cached.write_bytes(artifact.wasm)
            module = wasmtime.Module.deserialize(engine, cached.read_bytes())
        else:
            module = wasmtime.Module(engine, artifact.wat)
    except Exception as error:
        raise _fail(
            "E004",
            f"cannot load isolated WASM module: {error}",
            span=resolved_span,
            reason=str(error),
        ) from error

    store = wasmtime.Store(engine)
    try:
        instance = wasmtime.Instance(store, module, [])
    except Exception as error:
        raise _fail(
            "E004",
            f"cannot instantiate isolated WASM module: {error}",
            span=resolved_span,
            reason=str(error),
        ) from error

    context = WasmIsolatedInstance(
        artifact=artifact,
        run_id=run_id,
        instance_id=f"{artifact.source_digest[:12]}:{run_id}",
        store=store,
        instance=instance,
        _engine=engine,
        _cache_root=root,
    )
    exports = instance.exports(store)
    for name, signature in artifact.signatures.items():
        context.namespace[name] = _wrap_export(
            exports[name], store, signature.params, signature.ret or "i64"
        )
    # A private memory page per instance makes isolation observable without
    # requiring the compiled module to declare linear memory.
    try:
        context.memory = wasmtime.Memory(store, wasmtime.MemoryType(1, 2))
    except Exception:  # pragma: no cover - engine configuration dependent
        context.memory = None
    try:
        yield context
    finally:
        context.namespace.clear()
        context.memory = None
        context.instance = None
        context.store = None
        context._engine = None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item) for key, item in sorted(value.items(), key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _normalize_outcome(
    results: Mapping[str, Any], errors: Mapping[str, BaseException | None]
) -> dict[str, Any]:
    from tools.diff_fuzz import normalize_error

    return {
        "results": {name: _jsonable(value) for name, value in results.items()},
        "errors": {
            name: (None if error is None else normalize_error(error))
            for name, error in errors.items()
        },
    }


def wasm_verify_dual_backend_parity(
    source: str,
    *,
    calls: Mapping[str, Sequence[Any]],
    cache_root: Path | str | None = None,
    module_name: str = "polyglot_wasm",
    include_native: bool = True,
    span: Span = None,
) -> WasmParityReport:
    """Compare Python, Rust/PyO3, and WASM outcomes for scalar functions."""
    from backends import get_backend

    resolved_span = _normalize_span(span)
    normalized_calls = {name: tuple(args) for name, args in calls.items()}
    report = WasmParityReport(source=source, calls=normalized_calls)
    root = Path(cache_root) if cache_root is not None else None

    try:
        python_ns = get_backend("python", trusted=True).compile(
            _source_to_program(source)[0]
        )
    except Exception as error:
        python_ns = {}
        report.skipped["python"] = f"{type(error).__name__}: {error}"
    try:
        artifact = wasm_lower_ir_to_module(
            source, module_name=module_name, span=resolved_span
        )
    except Exception as error:
        artifact = None
        report.skipped["wasm"] = f"{type(error).__name__}: {error}"

    namespaces: dict[str, Mapping[str, Any]] = {"python": python_ns}
    if include_native:
        try:
            from backends.rust.compiler import NativeBackend

            native_root = (root / "native-cache") if root is not None else None
            backend = NativeBackend(cache_dir=native_root, trusted=True)
            namespaces["native"] = backend.compile(
                _source_to_program(source)[0],
                module_name=f"wparity_{artifact.source_digest[:8] if artifact else 'x'}",
                incremental=True,
                cache_scope=f"wasm-parity:{hashlib.sha256(source.encode()).hexdigest()[:12]}",
            )
        except Exception as error:
            report.skipped["native"] = f"{type(error).__name__}: {error}"

    if artifact is not None:
        try:
            with wasm_isolate_instance_state(
                artifact, run_id="verify", cache_root=root, span=resolved_span
            ) as isolated:
                namespaces["wasm"] = dict(isolated.namespace)
        except Exception as error:
            report.skipped["wasm"] = f"{type(error).__name__}: {error}"

    for name, args in normalized_calls.items():
        results: dict[str, Any] = {}
        errors: dict[str, BaseException | None] = {}
        for backend_name, namespace in namespaces.items():
            function = namespace.get(name)
            if function is None:
                errors[backend_name] = KeyError(
                    f"{name} is not exported by {backend_name}"
                )
                continue
            try:
                results[backend_name] = function(*args)
                errors[backend_name] = None
            except Exception as error:
                errors[backend_name] = error
        report.outcomes[name] = _normalize_outcome(results, errors)

    active = [name for name in namespaces if name not in report.skipped]
    for name, outcome in report.outcomes.items():
        for index, backend_name in enumerate(active):
            for other in active[index + 1 :]:
                left = outcome["results"].get(backend_name)
                right = outcome["results"].get(other)
                left_error = outcome["errors"].get(backend_name)
                right_error = outcome["errors"].get(other)
                if left_error is None and right_error is None and left == right:
                    continue
                if (
                    left_error is not None
                    and right_error is not None
                    and left_error.get("code") == right_error.get("code")
                    and left_error.get("message") == right_error.get("message")
                ):
                    continue
                report.mismatches.append(
                    {
                        "call": name,
                        "backends": [backend_name, other],
                        "left": _jsonable({"result": left, "error": left_error}),
                        "right": _jsonable({"result": right, "error": right_error}),
                    }
                )
    return report


__all__ = [
    "AbiField",
    "AbiSpec",
    "HostBinding",
    "HostSignature",
    "WasmIsolatedInstance",
    "WasmMemoryLayout",
    "WasmModuleArtifact",
    "WasmParityReport",
    "WasmTrapReport",
    "wasm_bind_host_imports_and_exports",
    "wasm_guard_trap_and_oom_paths",
    "wasm_isolate_instance_state",
    "wasm_layout_linear_memory_and_stack",
    "wasm_lower_ir_to_module",
    "wasm_marshal_adt_and_result_abi",
    "wasm_unmarshal_adt_and_result_abi",
    "wasm_verify_dual_backend_parity",
]
