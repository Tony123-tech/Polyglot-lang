"""Python/Rust boundary mapping for generated PyO3 extensions.

These helpers model argument extraction, value coercion, result conversion and
PyO3 reference ownership without invoking Rust or Cargo.  All emitted strings
are stable for a given input, and invalid mappings raise
:class:`PolyglotError` with the framework E001-E008 diagnostic codes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from backends.rust.framework import RsParamSpec
from compiler.frontend.errors import PolyglotError

__all__ = [
    "rsb_arg_contract",
    "rsb_boundary_report",
    "rsb_bytes_decode",
    "rsb_call_signature",
    "rsb_coerce_mapping",
    "rsb_coerce_optional",
    "rsb_coerce_scalar",
    "rsb_coerce_sequence",
    "rsb_default_arg",
    "rsb_error_bridge",
    "rsb_exception_map",
    "rsb_extract_arg",
    "rsb_extract_plan",
    "rsb_lifetime_annotations",
    "rsb_none_marker",
    "rsb_numeric_narrow_check",
    "rsb_ownership_mode",
    "rsb_py_ref_guard",
    "rsb_python_type_map",
    "rsb_result_plan",
    "rsb_return_convert",
    "rsb_rust_type_classify",
    "rsb_string_encode",
    "rsb_type_compat",
    "rsb_validate_values",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TYPE_PATH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*\Z")
_INT_RANGES: dict[str, tuple[int, int]] = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "isize": (-(2**63), 2**63 - 1),
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "usize": (0, 2**64 - 1),
}
_TO_PYTHON = {
    "bool": "bool",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "isize": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "usize": "int",
    "f32": "float",
    "f64": "float",
    "String": "str",
    "char": "str",
    "()": "None",
}
_EXCEPTION_MODULES = {
    "ValueError": "pyo3::exceptions::PyValueError",
    "TypeError": "pyo3::exceptions::PyTypeError",
    "RuntimeError": "pyo3::exceptions::PyRuntimeError",
    "OSError": "pyo3::exceptions::PyOSError",
    "PillarError": "PyPillarError",
    "PillarException": "PyPillarError",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured native-boundary diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-boundary")


def _text(value: object, what: str) -> str:
    """Validate one caller-supplied string."""
    if not isinstance(value, str):
        _fail("E003", f"{what} must be a str, got {type(value).__name__}")
    return value


def _identifier(value: object, what: str) -> str:
    """Validate a Rust or Python-facing identifier."""
    text = _text(value, what)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a valid identifier, got {text!r}")
    return text


def _type_text(value: object, what: str) -> str:
    """Normalize and validate a Rust type expression."""
    text = " ".join(_text(value, what).split())
    if not text:
        _fail("E003", f"{what} must not be empty")
    if ";" in text or "\n" in text:
        _fail("E003", f"{what} must be a type expression, got {text!r}")
    if text.count("<") != text.count(">"):
        _fail("E001", f"{what} has unbalanced generics: {text!r}")
    return text


def _split_top_level(text: str) -> tuple[str, ...]:
    """Split comma-separated generic arguments at top level only."""
    pieces: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">])" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            piece = text[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return tuple(pieces)


def _parts(text: str) -> tuple[str, tuple[str, ...]]:
    """Split a Rust type into its base name and generic arguments."""
    normalized = _type_text(text, "Rust type")
    if "<" not in normalized:
        return normalized, ()
    base, rest = normalized.split("<", 1)
    if not rest.endswith(">") or _TYPE_PATH_RE.fullmatch(base.strip()) is None:
        _fail("E001", f"malformed parameterized type {text!r}")
    return base.strip(), _split_top_level(rest[:-1])


def _string_literal(value: str) -> str:
    """Render a stable escaped Rust string literal for diagnostic messages."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _balanced(value: object, what: str) -> str:
    """Validate balanced Rust delimiters in an expression."""
    text = _text(value, what)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            _fail("E003", f"{what} has unbalanced delimiters")
    if stack:
        _fail("E003", f"{what} has unbalanced delimiters")
    return text


def _param_spec(
    value: RsParamSpec | Mapping[str, Any] | tuple[str, str],
) -> RsParamSpec:
    """Coerce one parameter description into ``RsParamSpec``."""
    if isinstance(value, RsParamSpec):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return RsParamSpec(name=str(value[0]), type=str(value[1]))
    if not isinstance(value, Mapping):
        _fail(
            "E003", f"parameter must be a mapping or tuple, got {type(value).__name__}"
        )
    payload = {str(key): item for key, item in value.items()}
    if "rust_type" in payload and "type" not in payload:
        payload["type"] = payload.pop("rust_type")
    try:
        return RsParamSpec(**payload)
    except TypeError as error:
        _fail("E003", f"invalid parameter spec: {error}")


def _param_specs(params: Sequence[Any]) -> tuple[RsParamSpec, ...]:
    """Coerce a sequence of parameter descriptions."""
    if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
        _fail("E003", "params must be a sequence")
    specs = tuple(_param_spec(item) for item in params)
    names = tuple(_identifier(item.name, "argument name") for item in specs)
    if len(set(names)) != len(names):
        _fail("E003", "duplicate argument names in boundary contract")
    return specs


# ---------------------------------------------------------------------------
# 1. rsb_python_type_map
# ---------------------------------------------------------------------------


def rsb_python_type_map(
    type_name: str, *, direction: str = "to_rust"
) -> dict[str, Any]:
    """Map a Python type name to a Rust type or back again deterministically."""
    name = _text(type_name, "Python type name")
    if direction not in {"to_rust", "to_python"}:
        _fail("E003", f"unknown mapping direction {direction!r}")
    if direction == "to_python":
        rust_type = _type_text(name, "Rust type")
        mapped = _TO_PYTHON.get(rust_type)
        base, args = _parts(rust_type)
        if mapped is None and base == "Vec" and len(args) == 1:
            mapped = f"list[{rsb_python_type_map(args[0], direction='to_python')['python_type']}]"
        if mapped is None and base == "Option" and len(args) == 1:
            mapped = f"{rsb_python_type_map(args[0], direction='to_python')['python_type']} | None"
        if mapped is None:
            _fail("E007", f"unsupported Rust type {rust_type!r} at Python boundary")
        return {
            "source": rust_type,
            "python_type": mapped,
            "direction": direction,
            "exact": rust_type in _TO_PYTHON,
        }
    mapping = {
        "bool": "bool",
        "int": "i64",
        "float": "f64",
        "str": "String",
        "bytes": "Vec<u8>",
        "bytearray": "Vec<u8>",
        "list": "Vec<T>",
        "tuple": "Vec<T>",
        "dict": "HashMap<K, V>",
        "NoneType": "Option<T>",
        "None": "Option<T>",
    }
    if name not in mapping:
        _fail("E007", f"unsupported Python type {name!r} at Rust boundary")
    return {
        "source": name,
        "rust_type": mapping[name],
        "direction": direction,
        "generic": "<" in mapping[name],
    }


# ---------------------------------------------------------------------------
# 2. rsb_rust_type_classify
# ---------------------------------------------------------------------------


def rsb_rust_type_classify(rust_type: str) -> dict[str, Any]:
    """Classify a Rust type into its boundary family and generic arguments."""
    text = _type_text(rust_type, "Rust type")
    base, args = _parts(text)
    if base == "Result":
        family = "result"
    elif base == "Option":
        family = "option"
    elif base in {"Vec", "ArrayVec", "HashSet", "BTreeSet"}:
        family = "sequence"
    elif base in {"HashMap", "BTreeMap"}:
        family = "mapping"
    elif text == "()" or base == "unit":
        family = "unit"
    elif text.startswith("&"):
        family = "reference"
    elif base in {"Py", "Bound"} or text.startswith("Py<"):
        family = "py_handle"
    elif base in _INT_RANGES or base in {"f32", "f64", "bool", "char", "String", "str"}:
        family = "scalar"
    else:
        family = "named"
    return {
        "type": text,
        "base": base,
        "family": family,
        "args": args,
        "is_generic": bool(args),
        "by_reference": text.startswith("&"),
    }


# ---------------------------------------------------------------------------
# 3. rsb_extract_plan
# ---------------------------------------------------------------------------


def rsb_extract_plan(
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    *,
    args_expr: str = "args",
) -> dict[str, Any]:
    """Generate PyO3 extraction statements for an argument tuple."""
    identifier = _identifier(args_expr, "argument expression")
    specs = _param_specs(params)
    lines = [f"let __py_args = {identifier};"]
    extracted: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        rendered = rsb_extract_arg(
            index,
            spec.name,
            spec.type,
            required=spec.required,
            default=spec.default,
        )
        lines.extend(rendered["code"].splitlines())
        extracted.append(rendered)
    return {
        "code": "\n".join(lines),
        "params": tuple(spec.name for spec in specs),
        "rust_types": {
            spec.name: _type_text(spec.type, "argument type") for spec in specs
        },
        "extractions": tuple(extracted),
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 4. rsb_extract_arg
# ---------------------------------------------------------------------------


def rsb_extract_arg(
    index: int,
    name: str,
    rust_type: str,
    *,
    required: bool = True,
    default: str | None = None,
) -> dict[str, Any]:
    """Render the extraction of one tuple item into a typed Rust binding."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        _fail("E003", "argument index must be a non-negative int")
    identifier = _identifier(name, "argument name")
    canonical = _type_text(rust_type, "argument type")
    missing = _string_literal(
        f"missing required argument '{identifier}' at position {index}"
    )
    wrong_type = _string_literal(
        f"argument '{identifier}' at position {index} must be {canonical}"
    )
    if required:
        lines = [
            (
                f"let __arg_{index} = __py_args.get_item({index}).map_err(|_| "
                f"pyo3::exceptions::PyTypeError::new_err({missing}))?;"
            ),
            (
                f"let {identifier}: {canonical} = __arg_{index}.extract::<{canonical}>()"
                f".map_err(|_| pyo3::exceptions::PyTypeError::new_err({wrong_type}))?;"
            ),
        ]
    else:
        default_line = rsb_default_arg(identifier, canonical, default)["code"]
        lines = [
            f"let {identifier}: {canonical} = match __py_args.get_item({index}) {{",
            (
                "    Ok(item) => item.extract::<"
                f"{canonical}>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("
                f"{wrong_type}))?,"
            ),
            f"    Err(_) => {default_line},",
            "};",
        ]
    return {
        "code": "\n".join(lines),
        "index": index,
        "name": identifier,
        "rust_type": canonical,
        "required": required,
        "type_error": wrong_type.strip('"'),
    }


# ---------------------------------------------------------------------------
# 5. rsb_default_arg
# ---------------------------------------------------------------------------


def rsb_default_arg(name: str, rust_type: str, default: str | None) -> dict[str, Any]:
    """Validate and return the fallback expression for an optional argument."""
    identifier = _identifier(name, "argument name")
    canonical = _type_text(rust_type, "argument type")
    if default is None or not default.strip():
        _fail("E003", f"optional argument {identifier!r} needs a Rust default")
    expression = _balanced(default, f"default of {identifier!r}")
    return {
        "code": expression,
        "name": identifier,
        "rust_type": canonical,
        "is_literal": expression.replace("_", "").isdigit()
        or expression in {"true", "false", '""', "None", "0.0"},
    }


# ---------------------------------------------------------------------------
# 6. rsb_coerce_scalar
# ---------------------------------------------------------------------------


def rsb_coerce_scalar(value: Any, rust_type: str) -> dict[str, Any]:
    """Validate one scalar Python value against a Rust scalar type."""
    text = _type_text(rust_type, "Rust scalar type")
    if text in _INT_RANGES:
        checked = rsb_numeric_narrow_check(value, text)
        return {
            "value": checked["value"],
            "rust_type": text,
            "kind": "integer",
            "range": [checked["min"], checked["max"]],
        }
    if text in {"f32", "f64"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("E001", f"value {value!r} must be a number for {text}")
        return {
            "value": float(value),
            "rust_type": text,
            "kind": "float",
            "range": None,
        }
    if text == "bool":
        if not isinstance(value, bool):
            _fail("E001", f"value {value!r} must be a bool")
        return {"value": value, "rust_type": text, "kind": "bool", "range": None}
    if text in {"String", "str", "char"}:
        if not isinstance(value, str):
            _fail("E001", f"value {value!r} must be a str for {text}")
        if text == "char" and len(value) != 1:
            _fail("E001", f"value {value!r} must contain exactly one character")
        return {"value": value, "rust_type": text, "kind": "string", "range": None}
    if text == "()":
        if value is not None:
            _fail("E001", "Rust unit type accepts only None")
        return {"value": None, "rust_type": text, "kind": "unit", "range": None}
    _fail("E007", f"unsupported Rust scalar type {text!r}")


# ---------------------------------------------------------------------------
# 7. rsb_coerce_sequence
# ---------------------------------------------------------------------------


def rsb_coerce_sequence(value: Any, element_type: str) -> dict[str, Any]:
    """Coerce a Python list or tuple into a Rust Vec element sequence."""
    if not isinstance(value, (list, tuple)):
        _fail("E001", f"expected a list or tuple for Vec<{element_type}>")
    items = tuple(rsb_coerce_scalar(item, element_type)["value"] for item in value)
    return {
        "value": list(items),
        "rust_type": f"Vec<{_type_text(element_type, 'element type')}>",
        "element_type": _type_text(element_type, "element type"),
        "length": len(items),
    }


# ---------------------------------------------------------------------------
# 8. rsb_coerce_mapping
# ---------------------------------------------------------------------------


def rsb_coerce_mapping(value: Any, key_type: str, value_type: str) -> dict[str, Any]:
    """Coerce a Python mapping into a Rust HashMap-compatible record."""
    if not isinstance(value, Mapping):
        _fail(
            "E001",
            f"expected a mapping for HashMap<{key_type}, {value_type}>",
        )
    items: dict[Any, Any] = {}
    for key, item in value.items():
        checked_key = rsb_coerce_scalar(key, key_type)["value"]
        checked_value = rsb_coerce_scalar(item, value_type)["value"]
        try:
            items[checked_key] = checked_value
        except TypeError:
            _fail("E001", f"key {key!r} is not hashable after coercion")
    return {
        "value": items,
        "rust_type": f"HashMap<{_type_text(key_type, 'key type')}, {_type_text(value_type, 'value type')}>",
        "key_type": key_type,
        "value_type": value_type,
        "length": len(items),
    }


# ---------------------------------------------------------------------------
# 9. rsb_coerce_optional
# ---------------------------------------------------------------------------


def rsb_coerce_optional(value: Any, inner_type: str) -> dict[str, Any]:
    """Coerce ``None`` or an inner value into a Rust Option representation."""
    if value is None:
        return {
            "value": None,
            "rust_type": f"Option<{_type_text(inner_type, 'inner type')}>",
            "present": False,
            "rust_expr": "None",
        }
    checked = rsb_coerce_scalar(value, inner_type)["value"]
    return {
        "value": checked,
        "rust_type": f"Option<{_type_text(inner_type, 'inner type')}>",
        "present": True,
        "rust_expr": f"Some({checked!r})"
        if isinstance(checked, str)
        else f"Some({checked})",
    }


# ---------------------------------------------------------------------------
# 10. rsb_validate_values
# ---------------------------------------------------------------------------


def rsb_validate_values(
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate supplied Python values against a boundary argument contract."""
    specs = _param_specs(params)
    if not isinstance(values, Mapping):
        _fail("E003", "values must be a mapping")
    checked: dict[str, Any] = {}
    rust_types: dict[str, str] = {}
    diagnostics: list[str] = []
    for spec in specs:
        name = spec.name
        rust_type = _type_text(spec.type, "argument type")
        rust_types[name] = rust_type
        if name not in values:
            if spec.required:
                _fail("E001", f"missing required argument {name!r}")
            continue
        base, args = _parts(rust_type)
        value = values[name]
        if base == "Option" and len(args) == 1:
            checked[name] = rsb_coerce_optional(value, args[0])["value"]
        elif base in {"Vec", "ArrayVec"} and len(args) == 1:
            checked[name] = rsb_coerce_sequence(value, args[0])["value"]
        elif base in {"HashMap", "BTreeMap"} and len(args) == 2:
            checked[name] = rsb_coerce_mapping(value, args[0], args[1])["value"]
        else:
            checked[name] = rsb_coerce_scalar(value, rust_type)["value"]
        if (
            base in {"f32", "f64"}
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            diagnostics.append(f"argument '{name}' widened from int to {rust_type}")
    return {
        "values": checked,
        "rust_types": rust_types,
        "diagnostics": tuple(diagnostics),
        "checked": len(checked),
        "skipped_optional": len(specs) - len(checked),
    }


# ---------------------------------------------------------------------------
# 11. rsb_result_plan
# ---------------------------------------------------------------------------


def rsb_result_plan(
    rust_type: str,
    *,
    result_expr: str = "__result",
    error_class: str = "PyPillarError",
) -> dict[str, Any]:
    """Plan a typed PyResult conversion for a Rust result specification."""
    expression = _identifier(result_expr, "result variable")
    exception = _identifier(error_class, "error class")
    text = _type_text(rust_type, "Rust result type")
    base, args = _parts(text)
    if base == "Result":
        if len(args) != 2:
            _fail("E001", "Result requires an Ok and an Err type")
        ok_type, error_type = args
    else:
        ok_type, error_type = text, None
    bridge = rsb_error_bridge(error_type, error_class=exception) if error_type else None
    code = (
        f"Ok({expression})"
        if bridge is None
        else f"match {expression} {{\n{bridge['code']}\n}}"
    )
    return {
        "code": code,
        "ok_type": ok_type,
        "error_type": error_type,
        "python_exception": bridge["python_exception"] if bridge else "None",
        "rust_exception": exception,
        "signature": f"PyResult<{ok_type}>",
    }


# ---------------------------------------------------------------------------
# 12. rsb_error_bridge
# ---------------------------------------------------------------------------


def rsb_error_bridge(
    error_type: str, *, error_class: str = "PyPillarError"
) -> dict[str, Any]:
    """Emit the ``Err`` arm that converts a Rust error to a Python exception."""
    text = _type_text(error_type, "Rust error type")
    exception = _identifier(error_class, "error class")
    mapping = rsb_exception_map(text)
    if text in {"PillarError", "RsPillarError"}:
        arm = f"    Err(error) => Err({exception}::new_err(error.format_pillar())),"
    elif text in {"PyErr", "pyo3::PyErr"}:
        arm = "    Err(error) => Err(error),"
    elif text in {"String", "str"}:
        arm = (
            "    Err(message) => Err(pyo3::exceptions::PyValueError::new_err(message)),"
        )
    else:
        _fail(
            "E007",
            f"unsupported Rust error type {text!r}; use PillarError, PyErr, or String",
        )
    return {
        "code": "    Ok(value) => Ok(value),\n" + arm,
        "rust_type": text,
        "python_exception": mapping["python_exception"],
        "exception_path": mapping["exception_path"],
        "error_class": exception,
    }


# ---------------------------------------------------------------------------
# 13. rsb_exception_map
# ---------------------------------------------------------------------------


def rsb_exception_map(error_type: str) -> dict[str, Any]:
    """Map a Rust error type name to its PyO3 exception constructor path."""
    text = _type_text(error_type, "Rust error type")
    if text in {"PyErr", "pyo3::PyErr"}:
        return {
            "rust_type": text,
            "python_exception": "PyErr",
            "exception_path": "pyo3::PyErr",
            "requires_message": False,
        }
    if text in {"PillarError", "RsPillarError"}:
        return {
            "rust_type": text,
            "python_exception": "PillarError",
            "exception_path": "PyPillarError",
            "requires_message": True,
        }
    if text not in _EXCEPTION_MODULES:
        _fail("E007", f"no Python exception mapping for Rust error {text!r}")
    return {
        "rust_type": text,
        "python_exception": text,
        "exception_path": _EXCEPTION_MODULES[text],
        "requires_message": True,
    }


# ---------------------------------------------------------------------------
# 14. rsb_return_convert
# ---------------------------------------------------------------------------


def rsb_return_convert(ok_type: str, *, value_expr: str = "value") -> dict[str, Any]:
    """Emit the successful-value conversion used by PyO3 result wrappers."""
    expression = _identifier(value_expr, "value expression")
    text = _type_text(ok_type, "Ok type")
    if text == "()":
        code = "Ok(py.None())"
    elif text in _INT_RANGES or text in {"f32", "f64", "bool"}:
        code = f"Ok({expression})"
    else:
        code = f"Ok({expression}.into_pyobject(py)?.into_any().unbind())"
    return {
        "code": code,
        "ok_type": text,
        "value_expr": expression,
        "needs_py_token": code.endswith("unbind())"),
    }


# ---------------------------------------------------------------------------
# 15. rsb_call_signature
# ---------------------------------------------------------------------------


def rsb_call_signature(
    name: str,
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    *,
    result_type: str = "i64",
    receiver: str | None = None,
    py_result: bool = True,
) -> dict[str, Any]:
    """Render the complete PyO3 function signature and its attributes."""
    function_name = _identifier(name, "function name")
    specs = _param_specs(params)
    if receiver not in {None, "&self", "&mut self", "self"}:
        _fail("E001", f"unsupported receiver {receiver!r}")
    declarations = (
        [rsb_extract_arg(0, "__self", "Py<Self>")["code"]]
        if receiver is not None
        else []
    )
    declarations.extend(
        f"{_identifier(spec.name, 'argument name')}: {_type_text(spec.type, 'argument type')}"
        for spec in specs
    )
    result = _type_text(result_type, "result type")
    return_type = f"PyResult<{result}>" if py_result else result
    signature = f"fn {function_name}(py: Python<'_>, args: &Bound<'_, PyTuple>) -> {return_type}"
    attributes = ("#[pyfunction]",) if receiver is None else ("#[pymethods]",)
    return {
        "name": function_name,
        "signature": signature,
        "attributes": attributes,
        "params": tuple(spec.name for spec in specs),
        "return_type": return_type,
        "receiver": receiver,
        "preamble": "\n".join(declarations),
    }


# ---------------------------------------------------------------------------
# 16. rsb_arg_contract
# ---------------------------------------------------------------------------


def rsb_arg_contract(
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    *,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured call contract with ownership and default metadata."""
    specs = _param_specs(params)
    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        rust_type = _type_text(spec.type, "argument type")
        if not spec.required and not spec.default:
            _fail("E003", f"optional argument {spec.name!r} needs a default")
        entries.append(
            {
                "index": index,
                "name": spec.name,
                "rust_type": rust_type,
                "required": spec.required,
                "default": spec.default,
                "ownership": rsb_ownership_mode(rust_type)["mode"],
                "family": rsb_rust_type_classify(rust_type)["family"],
            }
        )
    supplied = rsb_validate_values(specs, values) if values is not None else None
    return {
        "args": tuple(entries),
        "count": len(entries),
        "rust_types": tuple(item["rust_type"] for item in entries),
        "required": tuple(item["name"] for item in entries if item["required"]),
        "validated": supplied,
    }


# ---------------------------------------------------------------------------
# 17. rsb_ownership_mode
# ---------------------------------------------------------------------------


def rsb_ownership_mode(rust_type: str) -> dict[str, Any]:
    """Decide whether a boundary value is copied, borrowed, owned, or a Py handle."""
    classification = rsb_rust_type_classify(rust_type)
    text, family = classification["type"], classification["family"]
    if text in _INT_RANGES or text in {"f32", "f64", "bool", "char"}:
        mode, reason = "copy", "scalar values cross the boundary by value"
    elif family == "py_handle":
        mode, reason = "py_handle", "Py<T>/Bound<T> must be cloned before GIL release"
    elif family == "reference":
        mode, reason = "borrow", "borrowed references are tied to the GIL token"
    elif text in {"String", "Vec<u8>"} or family in {"sequence", "mapping"}:
        mode, reason = "own", "owned collections survive extraction and GIL release"
    elif family in {"option", "result"}:
        mode, reason = "own", "wrappers own their payload at the boundary"
    else:
        mode, reason = "own", "named types are extracted into owned Rust values"
    return {
        "rust_type": text,
        "mode": mode,
        "family": family,
        "reason": reason,
        "crosses_gil_release": mode in {"copy", "own", "py_handle"},
    }


# ---------------------------------------------------------------------------
# 18. rsb_lifetime_annotations
# ---------------------------------------------------------------------------


def rsb_lifetime_annotations(rust_type: str) -> dict[str, Any]:
    """Annotate borrowed Rust types with an explicit PyO3 GIL lifetime."""
    text = _type_text(rust_type, "Rust type")
    if text.count("&") > 1:
        _fail("E007", f"nested references {text!r} are unsupported at PyO3 boundary")
    if text.startswith("&"):
        rest = text[1:].strip()
        if rest.startswith("'"):
            annotated = f"&{rest}"
        else:
            annotated = f"&'py {rest}"
        lifetimes = ("'py",)
        owned_alternative = "String" if rest == "str" else f"Owned<{rest}>"
    else:
        annotated = text
        lifetimes = ()
        owned_alternative = text
    return {
        "text": annotated,
        "source": text,
        "lifetimes": lifetimes,
        "owned_alternative": owned_alternative,
        "borrowed": bool(lifetimes),
    }


# ---------------------------------------------------------------------------
# 19. rsb_py_ref_guard
# ---------------------------------------------------------------------------


def rsb_py_ref_guard(
    name: str,
    *,
    rust_type: str = "Py<PyAny>",
    owned: bool = True,
) -> dict[str, Any]:
    """Emit the clone/drop pair that keeps a Python reference RAII-safe."""
    identifier = _identifier(name, "reference name")
    text = _type_text(rust_type, "Python reference type")
    if not text.startswith(("Py<", "Bound<")):
        _fail("E001", f"{text!r} is not a PyO3 reference type")
    if owned:
        code = f"let __py_guard_{identifier} = {identifier}.clone_ref(py);"
        drop_line = f"drop(__py_guard_{identifier});"
    else:
        code = f"let __py_guard_{identifier} = {identifier}.clone().unbind();"
        drop_line = f"drop(__py_guard_{identifier});"
    return {
        "code": code,
        "name": identifier,
        "rust_type": text,
        "owned": owned,
        "drop_line": drop_line,
        "drop_on_scope_exit": True,
    }


# ---------------------------------------------------------------------------
# 20. rsb_numeric_narrow_check
# ---------------------------------------------------------------------------


def rsb_numeric_narrow_check(value: Any, rust_type: str) -> dict[str, Any]:
    """Check that a Python integer fits the target Rust integer type."""
    text = _type_text(rust_type, "Rust integer type")
    if text not in _INT_RANGES:
        _fail("E001", f"{text!r} is not a Rust integer type")
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E001", f"value {value!r} must be an int for {text}")
    low, high = _INT_RANGES[text]
    if not low <= value <= high:
        _fail(
            "E001",
            f"value {value} is outside the range of Rust {text} [{low}, {high}]",
        )
    return {
        "value": value,
        "rust_type": text,
        "min": low,
        "max": high,
        "fits": True,
    }


# ---------------------------------------------------------------------------
# 21. rsb_string_encode
# ---------------------------------------------------------------------------


def rsb_string_encode(value: str) -> dict[str, Any]:
    """Encode Python text as an escaped Rust string literal."""
    text = _text(value, "string value")
    literal = _string_literal(text)
    return {
        "literal": literal,
        "value": text,
        "escaped": literal != f'"{text}"',
        "utf8_bytes": len(text.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# 22. rsb_bytes_decode
# ---------------------------------------------------------------------------


def rsb_bytes_decode(value: bytes | bytearray | Sequence[int]) -> dict[str, Any]:
    """Model conversion of Python bytes into a Rust ``Vec<u8>``."""
    if isinstance(value, (bytes, bytearray)):
        values = list(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            _fail("E001", "byte sequence entries must be ints")
        if any(not 0 <= item <= 255 for item in values):
            _fail("E001", "byte sequence entries must be in 0..=255")
    else:
        _fail("E001", "expected bytes, bytearray, or a sequence of byte ints")
    return {
        "values": values,
        "rust_type": "Vec<u8>",
        "length": len(values),
        "rust_expr": f"vec![{', '.join(str(item) for item in values)}]",
    }


# ---------------------------------------------------------------------------
# 23. rsb_none_marker
# ---------------------------------------------------------------------------


def rsb_none_marker(rust_type: str) -> dict[str, Any]:
    """Describe the Python/Rust representations of an Option value."""
    text = _type_text(rust_type, "Rust type")
    base, args = _parts(text)
    is_option = base == "Option" and len(args) == 1
    if is_option:
        return {
            "type": text,
            "is_option": True,
            "inner_type": args[0],
            "none_expr": "None",
            "some_expr": "Some(value)",
            "python_none": "py.None()",
        }
    return {
        "type": text,
        "is_option": False,
        "inner_type": None,
        "none_expr": None,
        "some_expr": None,
        "python_none": "py.None()",
    }


# ---------------------------------------------------------------------------
# 24. rsb_type_compat
# ---------------------------------------------------------------------------


def rsb_type_compat(source_type: str, target_type: str) -> dict[str, Any]:
    """Report whether a source Rust type can cross into a target Rust type."""
    source = _type_text(source_type, "source type")
    target = _type_text(target_type, "target type")
    if source == target:
        return {
            "source": source,
            "target": target,
            "compatible": True,
            "coercion": "identity",
            "reason": "identical Rust types",
        }
    if source in _INT_RANGES and target in _INT_RANGES:
        source_low, source_high = _INT_RANGES[source]
        target_low, target_high = _INT_RANGES[target]
        compatible = target_low <= source_low and source_high <= target_high
        return {
            "source": source,
            "target": target,
            "compatible": compatible,
            "coercion": "checked_cast" if compatible else "narrowing_rejected",
            "reason": "integer width and signedness comparison",
        }
    if source == "str" and target == "String":
        return {
            "source": source,
            "target": target,
            "compatible": True,
            "coercion": "to_owned",
            "reason": "borrowed str is copied into an owned String",
        }
    if source in {"dict", "Generator", "Coroutine"} or target in {
        "Generator",
        "Coroutine",
    }:
        _fail("E007", f"unsupported native boundary type pair {source!r} -> {target!r}")
    return {
        "source": source,
        "target": target,
        "compatible": False,
        "coercion": None,
        "reason": f"no deterministic coercion from {source} to {target}",
    }


# ---------------------------------------------------------------------------
# 25. rsb_boundary_report
# ---------------------------------------------------------------------------


def rsb_boundary_report(
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    *,
    values: Mapping[str, Any] | None = None,
    result_type: str = "i64",
) -> dict[str, Any]:
    """Aggregate extraction, validation and result plans into one report."""
    specs = _param_specs(params)
    extraction = rsb_extract_plan(specs)
    contract = rsb_arg_contract(specs, values=values)
    result = rsb_result_plan(result_type)
    ownership = tuple(
        rsb_ownership_mode(_type_text(spec.type, "argument type")) for spec in specs
    )
    diagnostics: list[str] = []
    if contract["validated"] is not None:
        diagnostics.extend(contract["validated"]["diagnostics"])
    return {
        "extraction": extraction,
        "contract": contract,
        "result": result,
        "ownership": ownership,
        "diagnostics": tuple(diagnostics),
        "safe_for_gil_release": all(item["crosses_gil_release"] for item in ownership),
    }
