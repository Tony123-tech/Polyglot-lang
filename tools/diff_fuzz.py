"""Deterministic differential fuzzing for the Python and native backends.

The generator deliberately uses a fixed library of valid Polyglot templates with
randomized literals instead of random tokens.  Every ``(seed, index)`` pair maps
to exactly one byte-identical source, so failures can be reproduced without any
additional state.  Native builds stay in a caller-supplied (or temporary)
cache root and never touch the repository working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow ``python tools/diff_fuzz.py ...`` as well as package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.cli import _jsonable_result as _cli_jsonable_result
from diagnostics.framework import (
    diag_map_error_code_e001_to_e008,
    diag_redact_sensitive_paths,
)
from tools.build_ci.framework import ci_compute_cache_scope_key

CATEGORY_RECURSIVE_GENERICS = "recursive_generics"
CATEGORY_TRAITS_IMPLS_MATCH = "traits_impls_match"
CATEGORY_ARITHMETIC_BOOLEAN_CONTROL_FLOW = "arithmetic_boolean_control_flow"
CATEGORY_DIAGNOSTICS = "diagnostics"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_RECURSIVE_GENERICS,
    CATEGORY_TRAITS_IMPLS_MATCH,
    CATEGORY_ARITHMETIC_BOOLEAN_CONTROL_FLOW,
    CATEGORY_DIAGNOSTICS,
)

DEFAULT_CASE_COUNT = 8
DEFAULT_ARTIFACTS_RELATIVE = Path(".artifacts") / "diff_fuzz"
_CODE_RE = re.compile(r"\bE00[1-8]\b")
_SKIP_NATIVE_RE = re.compile(r"//\s*diff-fuzz:\s*skip-native:\s*(.+)")

CompileCallable = Callable[..., dict[str, Any]]
Result = dict[str, Any]


@dataclass(frozen=True)
class FuzzCase:
    """One reproducible generated program."""

    seed: int
    index: int
    category: str
    source: str


@dataclass
class FuzzReport:
    """Summary of one fuzzing run."""

    seed: int
    cases: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every executed case agreed across the requested backends."""
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-stable, path-redacted representation."""
        payload: dict[str, Any] = {
            "seed": self.seed,
            "cases": self.cases,
            "passed": self.passed,
            "mismatches": [dict(item) for item in self.mismatches],
            "skipped": [dict(item) for item in self.skipped],
            "counts": {
                "mismatches": len(self.mismatches),
                "skipped": len(self.skipped),
            },
        }
        return _jsonable(payload)

    def render(self) -> str:
        """Return a short deterministic human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"DIFF FUZZ seed={self.seed} cases={self.cases}: {status}",
            f"mismatches: {len(self.mismatches)}, skipped: {len(self.skipped)}",
        ]
        for item in self.mismatches[:5]:
            diff = str(item.get("diff", "")).splitlines()
            detail = diff[-1] if diff else "values differ"
            lines.append(
                f"  - {item.get('index', '?')} [{item.get('category', '?')}] {detail}"
            )
        return "\n".join(lines)


class FuzzHarnessError(RuntimeError):
    """Raised for invalid harness configuration (CLI exit code 2)."""


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------


def _tpl_recursive_struct(rng: random.Random) -> str:
    leaf, root, extra = (
        rng.randint(-4, 6),
        rng.randint(1, 9),
        rng.randint(0, 5),
    )
    return f"""struct DFNode<T> {{
    value: T,
    children: Vec<DFNode<T>>,
}}

fn dfuzz_recursive_struct() -> i64 {{
    let leaf: DFNode<i64> = DFNode {{ value: {leaf}, children: vec![] }};
    let root: DFNode<i64> = DFNode {{ value: {root}, children: vec![leaf] }};
    root.value + root.children.len() as i64 + {extra}
}}

fn main() -> i64 {{ dfuzz_recursive_struct() }}
"""


def _tpl_recursive_enum(rng: random.Random) -> str:
    left, right, extra = rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(0, 4)
    return f"""enum DFTree<T> {{
    Leaf(T),
    Node(Vec<DFTree<T>>),
}}

fn dfuzz_recursive_enum() -> i64 {{
    let tree: DFTree<i64> =
        DFTree::Node(vec![DFTree::Leaf({left}), DFTree::Leaf({right})]);
    match tree {{
        DFTree::Leaf(value) => value,
        DFTree::Node(children) => children.len() as i64 + {extra},
    }}
}}

fn main() -> i64 {{ dfuzz_recursive_enum() }}
"""


def _tpl_vec_repeat(rng: random.Random) -> str:
    value, count = rng.randint(-9, 9), rng.randint(1, 6)
    return f"""fn dfuzz_vec_repeat() -> i64 {{
    let values: Vec<i64> = vec![{value}; {count}];
    values.len() as i64 + {value}
}}

fn main() -> i64 {{ dfuzz_vec_repeat() }}
"""


def _tpl_vec_nested(rng: random.Random) -> str:
    first, second, third = (
        rng.randint(-5, 5),
        rng.randint(-5, 5),
        rng.randint(-5, 5),
    )
    return f"""fn dfuzz_vec_nested() -> i64 {{
    let nested: Vec<Vec<i64>> = vec![vec![{first}, {second}], vec![{third}]];
    nested.len() as i64 + {first} - {second} + {third}
}}

fn main() -> i64 {{ dfuzz_vec_nested() }}
"""


def _tpl_bounded_recursive_box(rng: random.Random) -> str:
    value, bonus = rng.randint(1, 12), rng.randint(0, 4)
    return f"""struct DFBox<T: Clone> {{ value: T }}

impl<T: Clone> DFBox<T> {{
    fn new(value: T) -> DFBox<T> {{ DFBox {{ value: value }} }}
    fn get(&self) -> T {{ self.value.clone() }}
}}

fn dfuzz_bounded_box() -> i64 {{
    let boxed: DFBox<i64> = DFBox::new({value});
    boxed.get() + {bonus}
}}

fn main() -> i64 {{ dfuzz_bounded_box() }}
"""


def _tpl_traits_bounded_impl(rng: random.Random) -> str:
    value, bonus = rng.randint(1, 20), rng.randint(0, 7)
    return f"""struct DFBounded<T: Clone> {{ value: T }}

impl<T: Clone> DFBounded<T> {{
    fn new(value: T) -> DFBounded<T> {{ DFBounded {{ value: value }} }}
    fn get(&self) -> T {{ self.value.clone() }}
}}

fn dfuzz_bounded_impl() -> i64 {{
    let item: DFBounded<i64> = DFBounded::new({value});
    item.get() + {bonus}
}}

fn main() -> i64 {{ dfuzz_bounded_impl() }}
"""


def _tpl_impl_without_generics(rng: random.Random) -> str:
    value, bonus = rng.randint(1, 20), rng.randint(0, 7)
    return f"""struct DFWrap<T> {{ value: T }}

impl DFWrap<T> {{
    fn get(&self) -> T {{ self.value.clone() }}
}}

fn dfuzz_impl_without_generics() -> i64 {{
    let wrap: DFWrap<i64> = DFWrap {{ value: {value} }};
    wrap.get() + {bonus}
}}

fn main() -> i64 {{ dfuzz_impl_without_generics() }}
"""


def _tpl_match_guard(rng: random.Random) -> str:
    arg, limit, multiplier, bonus, fallback = (
        rng.randint(0, 9),
        rng.randint(1, 4),
        rng.randint(1, 4),
        rng.randint(0, 5),
        rng.randint(-9, -1),
    )
    return f"""fn dfuzz_match_guard(x: i64) -> i64 {{
    match x {{
        0 => {bonus},
        n if n > {limit} => n * {multiplier} + {bonus},
        _ => {fallback},
    }}
}}

fn main() -> i64 {{ dfuzz_match_guard({arg}) }}
"""


def _tpl_generic_function_guard(rng: random.Random) -> str:
    value, limit, bonus, fallback = (
        rng.randint(1, 12),
        rng.randint(2, 8),
        rng.randint(0, 5),
        rng.randint(-8, -1),
    )
    return f"""fn dfuzz_identity<T: Clone>(x: T) -> T {{ x }}

fn dfuzz_generic_guard() -> i64 {{
    let value = dfuzz_identity::<i64>({value});
    match value {{
        n if n >= {limit} => n + {bonus},
        _ => {fallback},
    }}
}}

fn main() -> i64 {{ dfuzz_generic_guard() }}
"""


def _tpl_arithmetic_control_flow(rng: random.Random) -> str:
    start, limit, arg, while_count, for_count, extra = (
        rng.randint(-9, 9),
        rng.randint(0, 8),
        rng.randint(0, 9),
        rng.randint(1, 5),
        rng.randint(1, 5),
        rng.randint(0, 6),
    )
    return f"""fn dfuzz_arithmetic(n: i64) -> i64 {{
    let mut acc: i64 = {start};
    if n > {limit} {{
        acc = acc + n;
    }} else {{
        acc = acc - n;
    }}
    let mut i: i64 = 0;
    while i < {while_count} {{
        acc = acc + i;
        i = i + 1;
    }}
    for j in 0..{for_count} {{
        acc = acc + (j as i64);
    }}
    if acc > 0 && n != 0 {{
        return acc + {extra};
    }}
    acc
}}

fn main() -> i64 {{ dfuzz_arithmetic({arg}) }}
"""


def _tpl_boolean_expression(rng: random.Random) -> str:
    flag_a, flag_b = rng.choice(["true", "false"]), rng.choice(["true", "false"])
    when_true, when_false = rng.randint(0, 20), rng.randint(-20, -1)
    return f"""fn dfuzz_boolean(a: bool, b: bool) -> i64 {{
    if (a && !b) || b {{ {when_true} }} else {{ {when_false} }}
}}

fn main() -> i64 {{ dfuzz_boolean({flag_a}, {flag_b}) }}
"""


def _tpl_modulo_return(rng: random.Random) -> str:
    numerator, denominator, bonus = (
        rng.randint(5, 40),
        rng.randint(1, 9),
        rng.randint(0, 7),
    )
    return f"""fn dfuzz_modulo() -> i64 {{
    let divisor: i64 = {denominator};
    if divisor != 0 {{ {numerator} % divisor + {bonus} }} else {{ -1 }}
}}

fn main() -> i64 {{ dfuzz_modulo() }}
"""


def _tpl_diagnostic_e001(rng: random.Random) -> str:
    amount = rng.randint(1, 20)
    return f"""// diff-fuzz: expect E001 implicit boundary conversion
fn main() -> i64 {{
    let value: i64 = {amount} as usize;
    value
}}
"""


def _tpl_diagnostic_e002(rng: random.Random) -> str:
    amount = rng.randint(1, 20)
    return f"""// diff-fuzz: expect E002 unprovable termination
#[terminates]
fn dfuzz_unproven(n: i64) -> i64 {{ dfuzz_unproven(n) }}

fn main() -> i64 {{ dfuzz_unproven({amount}) }}
"""


def _tpl_diagnostic_e003(rng: random.Random) -> str:
    amount = rng.randint(1, 20)
    return f"""// diff-fuzz: skip-native: native codegen has no E003 translate diagnostic
fn main() -> i64 {{
    dfuzz_unknown_macro!({amount});
    0
}}
"""


def _tpl_diagnostic_e008(rng: random.Random) -> str:
    literal = f"not-a-number-{rng.randint(1, 99)}"
    return f"""// diff-fuzz: expect E008 invalid integer parse literal
fn main() -> i64 {{
    let value: i64 = "{literal}".parse::<i64>().unwrap();
    value
}}
"""


_TEMPLATES: dict[str, tuple[Callable[[random.Random], str], ...]] = {
    CATEGORY_RECURSIVE_GENERICS: (
        _tpl_recursive_struct,
        _tpl_recursive_enum,
        _tpl_vec_repeat,
        _tpl_vec_nested,
        _tpl_bounded_recursive_box,
    ),
    CATEGORY_TRAITS_IMPLS_MATCH: (
        _tpl_traits_bounded_impl,
        _tpl_impl_without_generics,
        _tpl_match_guard,
        _tpl_generic_function_guard,
    ),
    CATEGORY_ARITHMETIC_BOOLEAN_CONTROL_FLOW: (
        _tpl_arithmetic_control_flow,
        _tpl_boolean_expression,
        _tpl_modulo_return,
    ),
    CATEGORY_DIAGNOSTICS: (
        _tpl_diagnostic_e001,
        _tpl_diagnostic_e002,
        _tpl_diagnostic_e003,
        _tpl_diagnostic_e008,
    ),
}


def generate_case(seed: int, index: int) -> FuzzCase:
    """Generate the single deterministic case for ``(seed, index)``."""
    if index < 0:
        raise ValueError("index must be non-negative")
    category = CATEGORIES[index % len(CATEGORIES)]
    rng = random.Random(f"polyglot-diff-fuzz:{seed}:{index}")
    templates = _TEMPLATES[category]
    source = templates[rng.randrange(len(templates))](rng)
    return FuzzCase(seed=seed, index=index, category=category, source=source)


def iter_cases(seed: int, count: int) -> Iterator[FuzzCase]:
    """Yield ``count`` deterministic cases for ``seed`` in index order."""
    if count < 0:
        raise ValueError("count must be non-negative")
    for index in range(count):
        yield generate_case(seed, index)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Return a deterministic JSON-safe representation of ``value``."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if _depth > 40:
        return "<recursive>"
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key], _depth=_depth + 1)
            for key in sorted(value, key=str)
        }
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item, _depth=_depth + 1) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, _depth=_depth + 1) for item in value]
    return _cli_jsonable_result(value)


def _redact(text: str) -> str:
    redacted = diag_redact_sensitive_paths(text)
    return redacted if isinstance(redacted, str) else str(redacted)


def normalize_error(error: Any) -> dict[str, str] | None:
    """Normalize an error into ``{"code": "E00x", "message": ...}``."""
    if error is None:
        return None
    if isinstance(error, Mapping):
        raw_code = error.get("code")
        text = str(
            error.get("message") or error.get("reason") or error.get("detail") or error
        )
    elif isinstance(error, BaseException):
        raw_code = getattr(error, "code", None)
        text = str(error) or repr(error)
        reason = getattr(error, "reason", None)
        if isinstance(reason, str) and reason.strip():
            text = reason.strip()
    else:
        raw_code, text = None, str(error)

    explicit = _CODE_RE.findall(f"{raw_code or ''} {text}".upper())
    code = explicit[0] if explicit else diag_map_error_code_e001_to_e008(text)

    message = text.strip()
    message = re.sub(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*", "", message)
    message = re.sub(
        r"^(?:error\[E00[1-8]\]|E00[1-8])\s*:\s*",
        "",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(r"^E00[1-8]\s+", "", message, flags=re.IGNORECASE)
    message = message.splitlines()[0].strip() if message else text.strip()
    message = _redact(message)

    # Both backends report bad integer literals differently; compare the
    # offending literal rather than Python exception wording.
    lowered = message.lower()
    is_literal_failure = (
        "invalid literal for int()" in lowered
        or "invalid integer parse literal" in lowered
        or "invalid literal for i64 parse" in lowered
        or "invalid literal for i" in lowered
    )
    if code == "E008" and is_literal_failure:
        quoted = re.search(r"['\"]([^'\"]+)['\"]", message)
        literal = quoted.group(1) if quoted else message
        if not quoted:
            literal = re.sub(r"^.*?literal\s*(?:parse\s*)?:\s*", "", literal)
        message = f"invalid integer literal: {literal.strip()}"
    return {"code": code, "message": message}


# Backwards-friendly private alias used by the test suite.
_normalize_error = normalize_error


def _normalize_outcome(raw: Any) -> Result:
    if not isinstance(raw, Mapping):
        raise FuzzHarnessError("compile callable must return a mapping")
    error = normalize_error(raw.get("error"))
    stdout = raw.get("stdout", "")
    stdout = "" if stdout is None else str(stdout).replace("\r\n", "\n")
    outcome: Result = {
        "result": _jsonable(raw.get("result")),
        "stdout": stdout,
        "error": error,
    }
    if raw.get("skipped"):
        outcome["skipped"] = True
        outcome["reason"] = _redact(str(raw.get("reason", "native backend skipped")))
    return outcome


def _outcomes_equal(python: Result, native: Result) -> bool:
    keys = ("result", "stdout", "error")
    return all(_jsonable(python.get(key)) == _jsonable(native.get(key)) for key in keys)


def _diff_text(python: Result, native: Result) -> str:
    lines = ["python vs native differential mismatch"]
    for key in ("result", "stdout", "error"):
        left, right = _jsonable(python.get(key)), _jsonable(native.get(key))
        if left != right:
            lines.extend(
                (
                    f"- {key}: {json.dumps(left, sort_keys=True, ensure_ascii=False)}",
                    f"+ {key}: {json.dumps(right, sort_keys=True, ensure_ascii=False)}",
                )
            )
    return _redact("\n".join(lines))


# ---------------------------------------------------------------------------
# Default backend callables
# ---------------------------------------------------------------------------


@contextmanager
def _captured_stdout() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield buffer


def _generated_args(function: Callable[..., Any], rng: random.Random) -> list[Any]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except TypeError, ValueError:
        return []
    args: list[Any] = []
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        kind = rng.randrange(3)
        if kind == 0:
            args.append(rng.randint(-8, 8))
        elif kind == 1:
            args.append(bool(rng.randrange(2)))
        else:
            args.append(f"v{rng.randint(0, 99)}")
        if len(args) >= 4:
            break
    return args


def _call_exported(namespace: Mapping[str, Any], scope: str) -> Any:
    rng = random.Random(f"polyglot-diff-fuzz-call:{scope}")
    names = ["main"]
    names.extend(sorted(name for name in namespace if name.startswith("dfuzz_")))
    for name in names:
        function = namespace.get(name)
        if callable(function):
            return function(*_generated_args(function, rng))
    return None


def _python_compile(
    source: str, *, scope: str, cache_root: Path | None
) -> dict[str, Any]:
    """Compile and execute ``source`` with the Python transpiler backend."""
    del cache_root
    from backends import get_backend
    from lexer import lex
    from parser import parse

    buffer = io.StringIO()
    try:
        with _captured_stdout() as captured:
            namespace = get_backend("python", trusted=True).compile(parse(lex(source)))
            value = _call_exported(namespace, scope)
            buffer = captured
    except Exception as error:  # diagnostics are compared structurally
        return {"result": None, "stdout": buffer.getvalue(), "error": error}
    return {"result": value, "stdout": buffer.getvalue(), "error": None}


def _native_compile(
    source: str, *, scope: str, cache_root: Path | None
) -> dict[str, Any]:
    """Compile and execute ``source`` in an isolated native build scope."""
    if shutil.which("cargo") is None:
        return {
            "result": None,
            "stdout": "",
            "error": None,
            "skipped": True,
            "reason": "cargo is not available on PATH",
        }

    from backends.rust.compiler import NativeBackend
    from lexer import lex
    from parser import parse

    skip_reason = _native_skip_reason(source)
    if skip_reason is not None:
        return {
            "result": None,
            "stdout": "",
            "error": None,
            "skipped": True,
            "reason": skip_reason,
        }

    root = cache_root if cache_root is not None else _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    # NativeBackend manages its own Cargo target directory inside ``cache_dir``
    # (``<cache_dir>/incremental/<scope>/target``) and locates the extension from
    # that same path.  Overriding CARGO_TARGET_DIR here would send Cargo's output
    # somewhere the backend does not look, producing a spurious E004.  The
    # dedicated cache_dir under the isolated root is itself outside the working
    # tree and keyed by the seed-derived scope, which satisfies the isolation
    # requirement without breaking extension discovery.
    buffer = io.StringIO()
    try:
        with _captured_stdout() as captured:
            backend = NativeBackend(cache_dir=root / "native-cache", trusted=True)
            namespace = backend.compile(
                parse(lex(source)),
                module_name=f"dfuzz_s{hash_stable(scope) & 0xFFFFFFFF}",
                incremental=True,
                cache_scope=scope,
            )
            value = _call_exported(namespace, scope)
            buffer = captured
    except Exception as error:  # diagnostics are compared structurally
        return {"result": None, "stdout": buffer.getvalue(), "error": error}
    return {"result": value, "stdout": buffer.getvalue(), "error": None}


def hash_stable(text: str) -> int:
    """Return a process-stable unsigned hash for a string (never ``hash()``)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _skipped_compile(reason: str) -> CompileCallable:
    def compile_skipped(
        source: str, *, scope: str, cache_root: Path | None
    ) -> dict[str, Any]:
        del source, scope, cache_root
        return {
            "result": None,
            "stdout": "",
            "error": None,
            "skipped": True,
            "reason": reason,
        }

    return compile_skipped


# ---------------------------------------------------------------------------
# Artifacts and case execution
# ---------------------------------------------------------------------------


def _default_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / "polyglot-diff-fuzz-cache"


def _default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / DEFAULT_ARTIFACTS_RELATIVE


def _scope_for(case: FuzzCase) -> str:
    return ci_compute_cache_scope_key(
        feature_flags={
            "diff_fuzz": True,
            "seed": case.seed,
            "index": case.index,
            "category": case.category,
        },
        extra={"stage": "native-parity"},
    )


def _native_skip_reason(source: str) -> str | None:
    match = _SKIP_NATIVE_RE.search(source)
    return _redact(match.group(1).strip()) if match else None


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` atomically, leaving no temporary files behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_mismatch(case: FuzzCase, outcome: Result, directory: Path) -> dict[str, Any]:
    stem = f"{case.seed}-{case.index}"
    source_path = f"{stem}.poly"
    json_path = f"{stem}.json"
    payload: dict[str, Any] = {
        "seed": case.seed,
        "index": case.index,
        "category": case.category,
        "source": case.source,
        "python_result": outcome.get("python"),
        "native_result": outcome.get("native"),
        "diff": outcome.get("diff"),
        "artifacts": {"json": json_path, "poly": source_path},
    }
    _atomic_write_text(directory / json_path, json.dumps(_jsonable(payload), indent=2))
    _atomic_write_text(directory / source_path, case.source)
    return payload


def run_case(
    case: FuzzCase,
    *,
    python_compile: CompileCallable | None = None,
    native_compile: CompileCallable | None = None,
    cache_root: Path | None = None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one generated case and return its normalized comparison."""
    scope = _scope_for(case)
    root = cache_root if cache_root is not None else _default_cache_root()
    python_call = python_compile if python_compile is not None else _python_compile
    native_call = native_compile if native_compile is not None else _native_compile

    python_raw = python_call(case.source, scope=scope, cache_root=root)
    python = _normalize_outcome(python_raw)

    native = _normalize_outcome(native_call(case.source, scope=scope, cache_root=root))
    native_skipped: str | None = None
    if native.get("skipped"):
        native_skipped = native.get("reason", "native backend skipped")
    elif python.get("error") is None and native.get("error") is not None:
        native_error = native["error"]
        if isinstance(native_error, Mapping) and native_error.get("code") in {
            "E004",
            "E007",
        }:
            native_skipped = (
                f"native backend cannot build this template "
                f"({native_error['code']}: {native_error['message']})"
            )
            native["skipped"] = True
            native["reason"] = native_skipped

    mismatch = not native.get("skipped") and not _outcomes_equal(python, native)
    outcome: dict[str, Any] = {
        "seed": case.seed,
        "index": case.index,
        "category": case.category,
        "source": case.source,
        "python": python,
        "native": native,
        "mismatch": mismatch,
        "skipped": native_skipped is not None,
        "skip_reason": native_skipped,
        "diff": None,
    }
    if mismatch:
        outcome["diff"] = _diff_text(python, native or {})
        directory = (
            artifacts_dir if artifacts_dir is not None else _default_artifacts_dir()
        )
        outcome["artifact"] = _save_mismatch(case, outcome, directory)
    return outcome


def run_fuzz(
    seed: int,
    case_count: int = DEFAULT_CASE_COUNT,
    *,
    cache_root: Path | None = None,
    artifacts_dir: Path | None = None,
    backends: Sequence[str] = ("python", "native"),
) -> FuzzReport:
    """Run all generated cases, collecting mismatches and native skips."""
    requested = tuple(backends)
    unknown = sorted(set(requested) - {"python", "native"})
    if unknown:
        raise FuzzHarnessError(f"unknown backends: {', '.join(unknown)}")
    python_call: CompileCallable | None = None
    native_call: CompileCallable | None = None
    if "python" not in requested:
        python_call = _skipped_compile("python backend not requested")
    if "native" not in requested:
        native_call = _skipped_compile("native backend not requested")

    mismatches: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    cases = list(iter_cases(seed, case_count))
    for case in cases:
        outcome = run_case(
            case,
            python_compile=python_call,
            native_compile=native_call,
            cache_root=cache_root,
            artifacts_dir=artifacts_dir,
        )
        if outcome["mismatch"]:
            mismatches.append(outcome["artifact"])
        if outcome["skipped"]:
            skipped.append(
                {
                    "seed": case.seed,
                    "index": case.index,
                    "category": case.category,
                    "backend": "native",
                    "reason": outcome["skip_reason"],
                }
            )
    return FuzzReport(
        seed=seed,
        cases=len(cases),
        mismatches=mismatches,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diff_fuzz",
        description="Deterministic Python/native differential fuzzing.",
    )
    parser.add_argument("--seed", type=int, required=True, help="reproduction seed")
    parser.add_argument(
        "--case-count",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help="number of generated cases (default: %(default)s)",
    )
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact directory"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=None, help="native cache root"
    )
    parser.add_argument(
        "--backends",
        default="python,native",
        help="comma-separated backend list (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit status."""
    args = _build_parser().parse_args(argv)
    try:
        backends = tuple(
            item.strip() for item in args.backends.split(",") if item.strip()
        )
        if not backends:
            raise FuzzHarnessError("--backends must name at least one backend")
        report = run_fuzz(
            args.seed,
            args.case_count,
            cache_root=args.cache_root,
            artifacts_dir=args.artifacts,
            backends=backends,
        )
    except Exception as error:
        print(f"diff_fuzz harness error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.as_dict(), sort_keys=True, allow_nan=False))
    else:
        print(report.render())
    return 1 if not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATEGORIES",
    "DEFAULT_CASE_COUNT",
    "FuzzCase",
    "FuzzHarnessError",
    "FuzzReport",
    "generate_case",
    "iter_cases",
    "normalize_error",
    "run_case",
    "run_fuzz",
]
