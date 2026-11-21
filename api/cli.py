"""
Polyglot command-line interface.

Subcommands:
    run   Compile and execute a source file.
    build Compile only and produce an artifact when applicable.
    check Parse, translate, and run termination checks without executing.
    compile  (compatibility) Compile and print exported names.
    exec  Compile and execute with JSON timing/memory metadata.
    visualizer  Serve the interactive AST/monomorphization visualizer.
    telemetry  Show recorded compile/cache/hotpatch telemetry (aliases: timings, cache, hotpatch).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from api.visualizer import fix_hint
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

# The most recently read program text.  Errors raised deep inside a command
# reach ``main``'s generic handler without a source argument; keeping the text
# here lets --error-format=json produce the same exact fix edits as the LSP.
_LAST_SOURCE: str = ""


def _read_source(path: str) -> str:
    global _LAST_SOURCE
    _LAST_SOURCE = Path(path).read_text(encoding="utf-8")
    return _LAST_SOURCE


def _read_stdin_source() -> str:
    global _LAST_SOURCE
    _LAST_SOURCE = sys.stdin.read()
    return _LAST_SOURCE


def _parse_program(source: str, path: str | None = None) -> Program:
    program = parse(lex(source))
    if path is not None:
        program = expand_includes(program, path)
    return program


def _load_program(path: str) -> Program:
    """Load a file while retaining its text for structured diagnostics.

    ``load_program`` keeps the established E005/E006 behaviour for missing or
    cyclic includes; the extra read only feeds the structured fix-hint payload.
    """
    program = load_program(path)
    _read_source(path)
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


def _error_payload(error: Exception, source_text: str | None = None) -> dict[str, Any]:
    """Build a machine-readable diagnostic payload, including any fix hint."""
    if source_text is None:
        source_text = _LAST_SOURCE
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

    hint = fix_hint(error, source_text)
    return {
        "code": code,
        "message": message,
        "line": line,
        "col": col,
        "span": list(span) if span is not None else None,
        "source": str(source) if source is not None else None,
        "fix": hint.to_dict() if hint is not None else None,
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


def _jsonable_result(value: Any) -> Any:
    """Convert a program return value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_result(item) for item in value]
    text = repr(value)
    return text if len(text) <= 4000 else text[:4000] + "..."


def _peak_memory_bytes() -> int | None:
    """Return the runner's peak resident set size in bytes."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = int(usage.ru_maxrss)
    if value <= 0:
        return None
    # macOS reports bytes; Linux and the BSDs report kibibytes.
    if sys.platform != "darwin":
        value *= 1024
    return value


def _write_exec_payload(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    try:
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def cmd_exec(args: argparse.Namespace) -> int:
    """Compile + execute with structured timing, memory, and diagnostics."""
    started = time.perf_counter()
    source = _read_stdin_source() if args.stdin else _read_source(args.file)
    payload: dict[str, Any] = {
        "ok": False,
        "backend": args.backend,
        "compile_ms": None,
        "run_ms": None,
        "total_ms": None,
        "peak_memory_bytes": None,
        "return_value": None,
        "diagnostics": [],
    }
    exit_code = 0
    compile_started: float | None = None
    run_started: float | None = None
    try:
        program = _parse_program(source)
        backend = get_backend(
            args.backend,
            trusted=True,
            target_width=args.target_width,
        )
        compile_started = time.perf_counter()
        namespace = backend.compile(program)
        payload["compile_ms"] = round((time.perf_counter() - compile_started) * 1000, 3)

        run_started = time.perf_counter()
        result = _run_main(namespace)
        payload["run_ms"] = round((time.perf_counter() - run_started) * 1000, 3)
        payload["return_value"] = _jsonable_result(result)
        payload["ok"] = True
    except Exception as error:
        from api.visualizer import diagnostic

        if compile_started is not None and payload["compile_ms"] is None:
            payload["compile_ms"] = round(
                (time.perf_counter() - compile_started) * 1000, 3
            )
        if run_started is not None and payload["run_ms"] is None:
            payload["run_ms"] = round((time.perf_counter() - run_started) * 1000, 3)
        phase = (
            "parse"
            if type(error).__name__ in {"LexError", "ParseError"}
            else args.backend
        )
        payload["diagnostics"] = [diagnostic(error, source, phase=phase)]
        _print_error(
            error,
            verbose=args.verbose,
            error_format=args.error_format,
        )
        exit_code = 1
    finally:
        payload["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload["peak_memory_bytes"] = _peak_memory_bytes()
        _write_exec_payload(args.result_file, payload)
    return exit_code


def cmd_run(args: argparse.Namespace) -> int:
    program = _load_program(args.file)
    backend = get_backend(args.backend, trusted=True, target_width=args.target_width)
    namespace = backend.compile(program)
    result = _run_main(namespace)
    if result is not None:
        print(result)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    source_path = Path(args.file)
    program = _load_program(str(source_path))
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

    backend = get_backend(backend_name, trusted=True)
    module_name = source_path.stem
    extension = backend.build(
        program,
        module_name=module_name,
        incremental=getattr(args, "incremental", False),
        cache_scope=str(source_path.resolve()),
    )
    out_dir = Path(args.out_dir) if args.out_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # On macOS, Python imports extension modules from ``.so`` files. Cargo
    # produces a ``.dylib``; the copied artifact gets the importable suffix.
    suffix = extension.suffix
    if sys.platform == "darwin" and suffix == ".dylib":
        suffix = ".so"
    dest = out_dir / f"{source_path.stem}{suffix}"
    shutil.copy2(extension, dest)
    print(str(dest))
    return 0


def cmd_lsp(args: argparse.Namespace) -> int:
    from api.lsp import main as lsp_main

    return lsp_main()


def cmd_visualizer(args: argparse.Namespace) -> int:
    from api.visualizer_server import serve

    return serve(
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        verbose=args.verbose,
    )


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
        _read_source(args.file)
    # Translation/codegen runs termination checks for #[terminates] functions.
    from backends.python.translator import Translator

    Translator().translate_program(program)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Compatibility command used by api.executor and existing tooling."""
    if args.stdin:
        source = _read_stdin_source()
    else:
        source = _read_source(args.file)
    program = _parse_program(source)
    backend = get_backend(args.backend, trusted=args.trusted)
    namespace = backend.compile(program)
    for name in sorted(namespace):
        if not name.startswith("_"):
            print(name)
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    """Print the session telemetry snapshot as JSON or aligned text."""
    from api.telemetry import (
        compare_with,
        format_comparison,
        format_terminal,
        save_baseline,
        snapshot,
    )

    payload = snapshot()
    saved_as = getattr(args, "save_as", None)
    compare_name = getattr(args, "compare", None)
    if saved_as:
        save_baseline(saved_as)
    comparison = compare_with(compare_name) if compare_name else None

    if args.json:
        output = dict(payload)
        if saved_as:
            output["baseline_saved_as"] = saved_as
        if comparison is not None:
            output["comparison"] = comparison
        print(json.dumps(output, sort_keys=True))
    else:
        print(format_terminal(payload, section=args.section, color=args.color))
        if comparison is not None:
            print()
            print(format_comparison(comparison, color=args.color))
        elif saved_as:
            print()
            print(f"Saved telemetry baseline '{saved_as}'.")
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
        choices=["python", "native", "wasm", "jit", "zig"],
        default="python",
        help=(
            "Backend to use (default: python). `jit` emits AArch64 machine "
            "code directly and requires macOS on Apple Silicon."
        ),
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
        choices=["python", "native", "wasm", "zig"],
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
        "--incremental",
        action="store_true",
        help=(
            "Reuse per-function Rust compilation fragments when rebuilding "
            "the native backend (faster edit/rebuild cycles)"
        ),
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

    exec_p = sub.add_parser(
        "exec",
        help="Compile and execute with machine-readable metrics",
        description=(
            "Compile and execute a program, printing its output. When "
            "--result-file is given, structured timing, peak memory, return "
            "value, and diagnostics are written to that path as JSON. Used by "
            "the visualizer's /api/run endpoint, which never executes user "
            "code inside the HTTP server."
        ),
        epilog=(
            "Examples:\n"
            "  polyglot exec --stdin --backend python --result-file out.json\n"
            "  polyglot exec hello.poly --backend native"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    exec_src = exec_p.add_mutually_exclusive_group(required=True)
    exec_src.add_argument("file", nargs="?", help="Path to the .poly file")
    exec_src.add_argument("--stdin", action="store_true")
    exec_p.add_argument(
        "--backend",
        choices=["python", "native", "wasm", "jit", "zig"],
        default="python",
        help="Backend to use (default: python)",
    )
    exec_p.add_argument(
        "--target-width",
        type=int,
        dest="target_width",
        default=None,
        help="Simulate a 32- or 64-bit usize/isize target (Python backend only)",
    )
    exec_p.add_argument(
        "--result-file",
        dest="result_file",
        default=None,
        help="Write structured execution metadata (JSON) to this path",
    )
    exec_p.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on error"
    )
    _add_error_format(exec_p)
    exec_p.set_defaults(func=cmd_exec)

    visualizer_p = sub.add_parser(
        "visualizer",
        aliases=["gui"],
        help="Serve the interactive AST and monomorphization visualizer",
        description=(
            "Start a local, dependency-free web UI that visualizes the AST, "
            "the monomorphization pass, E001-E008 diagnostics, and side-by-side "
            "Python/native execution. The server binds to 127.0.0.1 by default "
            "and runs submitted programs as trusted local code."
        ),
        epilog=(
            "Examples:\n"
            "  polyglot visualizer\n"
            "  polyglot visualizer --port 9000 --no-open"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    visualizer_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1)",
    )
    visualizer_p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind; use 0 to choose a free port (default: 8765)",
    )
    visualizer_p.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open a web browser automatically",
    )
    visualizer_p.add_argument(
        "--verbose", action="store_true", help="Log every HTTP request"
    )
    visualizer_p.set_defaults(func=cmd_visualizer, open_browser=True)

    compile_p = sub.add_parser(
        "compile",
        help="Compatibility compile command",
        description="Compile and print the exported names. Used by api.executor.",
    )
    compile_p.add_argument(
        "--backend", choices=["python", "native", "wasm", "jit", "zig"], required=True
    )
    src = compile_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--file", type=str)
    compile_p.add_argument("--trusted", action="store_true")
    compile_p.add_argument("--verbose", action="store_true")
    _add_error_format(compile_p)
    compile_p.set_defaults(func=cmd_compile)

    def add_telemetry_arguments(
        command_parser: argparse.ArgumentParser, *, section: str
    ) -> None:
        command_parser.add_argument(
            "--section",
            choices=["all", "telemetry", "timings", "cache", "hotpatch"],
            default=section,
            help="Section to render (default: %(default)s)",
        )
        command_parser.add_argument(
            "--json", action="store_true", help="Print the JSON snapshot"
        )
        command_parser.add_argument(
            "--color", action="store_true", help="Use ANSI colors in text output"
        )
        command_parser.add_argument(
            "--save-as",
            dest="save_as",
            metavar="NAME",
            default=None,
            help="Save the current snapshot as a named baseline",
        )
        command_parser.add_argument(
            "--compare",
            dest="compare",
            metavar="NAME",
            default=None,
            help="Compare the current snapshot with a saved baseline",
        )
        command_parser.set_defaults(func=cmd_telemetry, section=section)

    telemetry_p = sub.add_parser(
        "telemetry",
        help="Show recorded session telemetry",
        description=(
            "Render compile timings, cache metrics, monomorphization counts, "
            "and explicit hotpatch activity from this CLI session."
        ),
    )
    add_telemetry_arguments(telemetry_p, section="all")

    for alias, alias_section in (
        ("timings", "telemetry"),
        ("cache", "cache"),
        ("hotpatch", "hotpatch"),
    ):
        alias_p = sub.add_parser(
            alias,
            help=f"Alias for telemetry --section {alias_section}",
            description=f"Alias for `polyglot telemetry --section {alias_section}`.",
        )
        add_telemetry_arguments(alias_p, section=alias_section)

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
    except Exception as error:
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
