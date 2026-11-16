"""
Polyglot command-line interface.

Subcommands:
    run   Compile and execute a source file.
    build Compile only and produce an artifact when applicable.
    check Parse, translate, and run termination checks without executing.
    compile  (compatibility) Compile and print exported names.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from ast_nodes import Program
from backends import UntrustedCompileError, get_backend
from include_expander import (
    IncludeCycleError,
    IncludeNotFoundError,
    expand_includes,
    load_program,
)
from lexer import lex
from parser import parse, parse_recovering
from termination import TerminationError


def _read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_program(source: str, path: str | None = None) -> Program:
    program = parse(lex(source))
    if path is not None:
        program = expand_includes(program, path)
    return program


def _format_five_line(error: Exception) -> str:
    """Format an error using the toolchain five-line/six-line format."""
    if isinstance(error, (TerminationError, IncludeCycleError, IncludeNotFoundError)):
        return error.format()

    message = str(error).strip() or error.__class__.__name__
    if message.startswith(("error:", "error[")):
        return message

    for attr in ("message", "msg"):
        value = getattr(error, attr, None)
        if isinstance(value, str):
            value = value.strip()
            if value.startswith(("error:", "error[")):
                return value

    line = getattr(error, "line", None)
    col = getattr(error, "col", None)
    if line is None:
        line = getattr(error, "source_line", None)
    if col is None:
        col = getattr(error, "source_col", None)
    if line is None:
        line = getattr(error, "rust_line", None)
    if col is None:
        col = getattr(error, "rust_col", None)

    reason = message.splitlines()[0] if message else error.__class__.__name__
    help_text = "fix the reported error and try again"
    note = error.__class__.__name__
    if isinstance(error, FileNotFoundError):
        help_text = "check that the path exists and is readable"
    elif isinstance(error, (IncludeCycleError, IncludeNotFoundError)):
        help_text = getattr(error, "help_text", help_text)
    elif isinstance(error, UntrustedCompileError):
        help_text = "run compilation in a trusted context"
    elif "parse" in note.lower() or "syntax" in note.lower():
        help_text = "check the source syntax near the reported location"

    loc = (
        f"at: line {line}, column {col}"
        if line is not None and col is not None
        else "at: <unknown>"
    )
    return (
        f"error: {reason}\n"
        f"  reason: {reason}\n"
        f"  help: {help_text}\n"
        f"  note: {note}\n"
        f"  {loc}"
    )


def _error_payload(error: Exception) -> dict[str, Any]:
    """Build a machine-readable diagnostic payload."""
    text = str(error).strip()
    first_line = text.splitlines()[0] if text else error.__class__.__name__
    code = getattr(error, "code", None)
    if code is None:
        match = re.search(r"error\[(E\d+)\]", text)
        code = match.group(1) if match else None
    message = re.sub(r"^error(?:\[E\d+\])?:\s*", "", first_line)

    line = getattr(error, "line", None)
    col = getattr(error, "col", None)
    if line is None:
        line = getattr(error, "source_line", None)
    if col is None:
        col = getattr(error, "source_col", None)
    if line is None:
        line = getattr(error, "rust_line", None)
    if col is None:
        col = getattr(error, "rust_col", None)

    span = getattr(error, "span", None)
    source = getattr(error, "source_path", None)
    if source is None:
        source = getattr(error, "source_line_text", None)

    return {
        "code": code,
        "message": message,
        "line": line,
        "col": col,
        "span": list(span) if span is not None else None,
        "source": str(source) if source is not None else None,
    }


def _print_error(
    error: Exception, *, verbose: bool, error_format: str = "text"
) -> None:
    if verbose:
        traceback.print_exception(type(error), error, error.__traceback__)
        return
    if error_format == "json":
        print(json.dumps(_error_payload(error)), file=sys.stderr)
        return
    print(_format_five_line(error), file=sys.stderr)


def _run_main(namespace: dict[str, Any]) -> Any:
    main = namespace.get("main")
    if not callable(main):
        return None
    return main()


def cmd_run(args: argparse.Namespace) -> int:
    program = load_program(args.file)
    backend = get_backend(args.backend, trusted=True, target_width=args.target_width)
    namespace = backend.compile(program)
    result = _run_main(namespace)
    if result is not None:
        print(result)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    source_path = Path(args.file)
    program = load_program(source_path)
    backend_name = args.backend

    if backend_name == "python":
        get_backend("python", trusted=True).compile(program)
        print(str(source_path))
        return 0

    if backend_name == "wasm":
        from backends.wasm.codegen import WasmCodegen

        wat = WasmCodegen().generate(program)
        out_dir = Path(args.out_dir) if args.out_dir else source_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{source_path.stem}.wat"
        dest.write_text(wat, encoding="utf-8")
        print(str(dest))
        return 0

    backend = get_backend("native", trusted=True)
    module_name = source_path.stem
    backend.compile(program, module_name=module_name)
    # NativeBackend stores the built extension in its cache; copy it out.
    cache_dir = getattr(backend, "cache_dir", None)
    if cache_dir is None:
        raise RuntimeError("native backend did not expose a build cache")
    # Recompute the cache key the same way NativeBackend does.
    from backends.rust.codegen import RustCodegen

    source_rs = RustCodegen(module_name=module_name).generate(program)
    import hashlib

    key = hashlib.sha256(source_rs.encode()).hexdigest()[:16]
    build_dir = Path(cache_dir) / key
    ext = backend._find_extension(build_dir)
    out_dir = Path(args.out_dir) if args.out_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # On macOS, Python imports extension modules from ``.so`` files. Cargo
    # produces a ``.dylib``; the copied artifact gets the importable suffix.
    suffix = ext.suffix
    if sys.platform == "darwin" and suffix == ".dylib":
        suffix = ".so"
    dest = out_dir / f"{source_path.stem}{suffix}"
    shutil.copy2(ext, dest)
    print(str(dest))
    return 0


def cmd_lsp(args: argparse.Namespace) -> int:
    from api.lsp import main as lsp_main

    return lsp_main()


def cmd_check(args: argparse.Namespace) -> int:
    if getattr(args, "all_errors", False):
        source = _read_source(args.file)
        program, errors = parse_recovering(lex(source))
        if errors:
            for error in errors:
                print(_format_five_line(error), file=sys.stderr)
            return 1
    else:
        program = load_program(args.file)
    # Translation/codegen runs termination checks for #[terminates] functions.
    from backends.python.translator import Translator

    Translator().translate_program(program)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Compatibility command used by api.executor and existing tooling."""
    if args.stdin:
        source = sys.stdin.read()
    else:
        source = _read_source(args.file)
    program = _parse_program(source)
    backend = get_backend(args.backend, trusted=args.trusted)
    namespace = backend.compile(program)
    for name in sorted(namespace):
        if not name.startswith("_"):
            print(name)
    return 0


def _add_error_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--error-format",
        choices=["text", "json"],
        default="text",
        help="Diagnostic format (default: text)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyglot",
        description=(
            "Compile and run a Rust-like language with two backends: "
            "a Python transpiler (no native dependencies) and a native "
            "PyO3/Rust extension backend (requires a Rust toolchain)."
        ),
        epilog=(
            "Examples:\n"
            "  polyglot run hello.poly\n"
            "  polyglot run hello.poly --backend native\n"
            "  polyglot build hello.poly --out-dir out/\n"
            "  polyglot check hello.poly"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Compile and execute a file",
        description=(
            "Compile a .poly file and execute it. Prints the result of "
            "main() when it returns a value."
        ),
        epilog=(
            "Examples:\n"
            "  polyglot run hello.poly\n"
            "  polyglot run hello.poly --backend native"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument("file", help="Path to the .poly file")
    run_p.add_argument(
        "--backend",
        choices=["python", "native", "wasm"],
        default="python",
        help="Backend to use (default: python)",
    )
    run_p.add_argument(
        "--target-width",
        type=int,
        dest="target_width",
        default=None,
        help="Simulate a 32- or 64-bit usize/isize target (Python backend only)",
    )
    run_p.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on error"
    )
    _add_error_format(run_p)
    run_p.set_defaults(func=cmd_run)

    build_p = sub.add_parser(
        "build",
        help="Compile a file",
        description=(
            "Compile a .poly file. The native backend produces a Python "
            "extension module, the wasm backend writes a .wat module, and "
            "both print the artifact path."
        ),
        epilog=(
            "Examples:\n"
            "  polyglot build hello.poly\n"
            "  polyglot build hello.poly --out-dir out/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build_p.add_argument("file", help="Path to the .poly file")
    build_p.add_argument(
        "--backend",
        choices=["python", "native", "wasm"],
        default="native",
        help="Backend to use (default: native)",
    )
    build_p.add_argument(
        "--target",
        dest="target",
        default=None,
        help="Target triple (reserved; not yet implemented)",
    )
    build_p.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory for the built extension (default: source directory)",
    )
    build_p.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on error"
    )
    _add_error_format(build_p)
    build_p.set_defaults(func=cmd_build)

    check_p = sub.add_parser(
        "check",
        help="Parse, translate, and check termination",
        description=(
            "Parse a .poly file, translate it to the Python backend, and "
            "run termination checks. Does not execute the program."
        ),
        epilog=("Examples:\n  polyglot check hello.poly"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check_p.add_argument("file", help="Path to the .poly file")
    check_p.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on error"
    )
    check_p.add_argument(
        "--all-errors",
        action="store_true",
        help="Recover after syntax errors and report every top-level error",
    )
    _add_error_format(check_p)
    check_p.set_defaults(func=cmd_check)

    lsp_p = sub.add_parser(
        "lsp",
        help="Run the language server over stdio",
        description=(
            "Start a JSON-RPC language server on stdin/stdout with document "
            "sync, go-to-definition, and completion."
        ),
    )
    lsp_p.set_defaults(func=cmd_lsp)

    compile_p = sub.add_parser(
        "compile",
        help="Compatibility compile command",
        description="Compile and print the exported names. Used by api.executor.",
    )
    compile_p.add_argument(
        "--backend", choices=["python", "native", "wasm"], required=True
    )
    src = compile_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--file", type=str)
    compile_p.add_argument("--trusted", action="store_true")
    compile_p.add_argument("--verbose", action="store_true")
    _add_error_format(compile_p)
    compile_p.set_defaults(func=cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except UntrustedCompileError as error:
        _print_error(
            error,
            verbose=getattr(args, "verbose", False),
            error_format=getattr(args, "error_format", "text"),
        )
        return 2 if args.command == "compile" else 1
    except Exception as error:  # noqa: BLE001
        _print_error(
            error,
            verbose=getattr(args, "verbose", False),
            error_format=getattr(args, "error_format", "text"),
        )
        if args.command == "compile" and isinstance(error, UntrustedCompileError):
            return 2
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
