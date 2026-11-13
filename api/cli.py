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
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from ast_nodes import Program
from backends import UntrustedCompileError, get_backend
from lexer import lex
from parser import parse
from termination import TerminationError


def _read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_program(source: str) -> Program:
    return parse(lex(source))


def _format_five_line(error: Exception) -> str:
    """Format an error using the toolchain five-line format."""
    if isinstance(error, TerminationError):
        return error.format()

    message = str(error).strip() or error.__class__.__name__
    if message.count("\n") >= 4 and message.startswith("error:"):
        return message

    for attr in ("message", "msg"):
        value = getattr(error, attr, None)
        if isinstance(value, str) and value.count("\n") >= 4 and value.startswith("error:"):
            return value.strip()

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
    elif isinstance(error, UntrustedCompileError):
        help_text = "run compilation in a trusted context"
    elif "parse" in note.lower() or "syntax" in note.lower():
        help_text = "check the source syntax near the reported location"

    loc = f"at: line {line}, column {col}" if line is not None and col is not None else "at: <unknown>"
    return (
        f"error: {reason}\n"
        f"  reason: {reason}\n"
        f"  help: {help_text}\n"
        f"  note: {note}\n"
        f"  {loc}"
    )


def _print_error(error: Exception, *, verbose: bool) -> None:
    if verbose:
        traceback.print_exception(type(error), error, error.__traceback__)
        return
    print(_format_five_line(error), file=sys.stderr)


def _run_main(namespace: dict[str, Any]) -> Any:
    main = namespace.get("main")
    if not callable(main):
        return None
    return main()


def _artifact_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".pyd"
    return ".so"


def _native_artifact_path(source_path: Path) -> Path:
    return source_path.with_suffix(_artifact_extension())


def cmd_run(args: argparse.Namespace) -> int:
    source = _read_source(args.file)
    program = _parse_program(source)
    backend = get_backend(args.backend, trusted=True, target_width=args.target_width)
    namespace = backend.compile(program)
    result = _run_main(namespace)
    if result is not None:
        print(result)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    if args.target is not None:
        raise NotImplementedError(
            "--target is not yet implemented; cross-compilation is deferred. "
            "Remove --target to build for the host."
        )
    source_path = Path(args.file)
    source = source_path.read_text(encoding="utf-8")
    program = _parse_program(source)
    backend_name = args.backend

    if backend_name == "python":
        get_backend("python", trusted=True).compile(program)
        print(str(source_path))
        return 0

    backend = get_backend("native", trusted=True)
    backend.compile(program)
    # NativeBackend stores the built extension in its cache; copy it out.
    cache_dir = getattr(backend, "cache_dir", None)
    if cache_dir is None:
        raise RuntimeError("native backend did not expose a build cache")
    # Recompute the cache key the same way NativeBackend does.
    from backends.rust.codegen import RustCodegen

    source_rs = RustCodegen().generate(program)
    import hashlib

    key = hashlib.sha256(source_rs.encode()).hexdigest()[:16]
    build_dir = Path(cache_dir) / key
    ext = getattr(backend, "_find_extension")(build_dir)
    out_dir = Path(args.out_dir) if args.out_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{source_path.stem}{ext.suffix}"
    shutil.copy2(ext, dest)
    print(str(dest))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    source = _read_source(args.file)
    program = _parse_program(source)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polyglot")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Compile and execute a file")
    run_p.add_argument("file")
    run_p.add_argument("--backend", choices=["python", "native"], default="python")
    run_p.add_argument("--target-width", type=int, dest="target_width", default=None)
    run_p.add_argument("--verbose", action="store_true")
    run_p.set_defaults(func=cmd_run)

    build_p = sub.add_parser("build", help="Compile a file")
    build_p.add_argument("file")
    build_p.add_argument("--backend", choices=["python", "native"], default="native")
    build_p.add_argument("--target", dest="target", default=None)
    build_p.add_argument("--out-dir", dest="out_dir", default=None)
    build_p.add_argument("--verbose", action="store_true")
    build_p.set_defaults(func=cmd_build)

    check_p = sub.add_parser("check", help="Parse, translate, and check termination")
    check_p.add_argument("file")
    check_p.add_argument("--verbose", action="store_true")
    check_p.set_defaults(func=cmd_check)

    compile_p = sub.add_parser("compile", help="Compatibility compile command")
    compile_p.add_argument("--backend", choices=["python", "native"], required=True)
    src = compile_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--file", type=str)
    compile_p.add_argument("--trusted", action="store_true")
    compile_p.add_argument("--verbose", action="store_true")
    compile_p.set_defaults(func=cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except UntrustedCompileError as error:
        _print_error(error, verbose=getattr(args, "verbose", False))
        return 2 if args.command == "compile" else 1
    except Exception as error:  # noqa: BLE001
        _print_error(error, verbose=getattr(args, "verbose", False))
        if args.command == "compile" and isinstance(error, UntrustedCompileError):
            return 2
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
