"""Zero-dependency HTTP server for the interactive Polyglot visualizer.

The server is intentionally small and local-only:

* ``GET /`` serves the embedded single-page visualizer.
* ``POST /api/analyze`` parses + monomorphizes source in this process (read-only).
* ``POST /api/run`` shells out to ``python -m api.cli exec`` so user programs
  never execute inside the HTTP server.
* ``GET /api/health`` reports toolchain availability for the UI.

Only the standard library is used, and the default bind address is
``127.0.0.1``. Code executed through ``/api/run`` is trusted local code: the
UI warns about this and ``polyglot visualizer --host`` is the escape hatch for
remote exposure.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from api.visualizer import analyze

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "visualizer.html"

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
DEFAULT_TIMEOUT = 180.0
MIN_TIMEOUT = 1.0
MAX_TIMEOUT = 600.0
ALLOWED_BACKENDS = ("python", "native", "wasm", "jit")
DEFAULT_BACKENDS = ("python", "native")

try:  # pragma: no cover - depends on installation metadata
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _package_version

    try:
        VERSION = _package_version("polyglot-lang")
    except PackageNotFoundError:
        VERSION = "0.3.8"
except Exception:  # pragma: no cover - defensive
    VERSION = "0.3.8"


class BadRequest(ValueError):
    """Raised for malformed visualizer API requests."""


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            proc.kill()
    except ProcessLookupError, PermissionError, OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _read_result_file(path: str) -> dict[str, Any] | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _offset_for_line_col(source: str, line: int | None, col: int | None) -> int | None:
    if line is None or col is None or line < 1:
        return None
    lines = source.splitlines(keepends=True)
    if line > len(lines):
        return None
    return sum(len(text) for text in lines[: line - 1]) + max(0, col - 1)


def _fallback_diagnostic(
    stderr: str,
    source: str,
    backend: str,
    *,
    timed_out: bool,
) -> dict[str, Any]:
    text = stderr.strip()
    match = re.search(r"error\[(E\d+)\]:\s*(.+)", text)
    code = match.group(1) if match else None
    message = (
        match.group(2).strip()
        if match
        else (text.splitlines()[0] if text else "backend process failed")
    )
    location = re.search(r"at: line (\d+), column (\d+)", text)
    line = int(location.group(1)) if location else None
    col = int(location.group(2)) if location else None
    offset = _offset_for_line_col(source, line, col)
    span = None
    if offset is not None:
        token = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S", source[offset:])
        end = offset + (len(token.group(0)) if token else 1)
        span = [offset, min(end, len(source))]
    if timed_out:
        message = "execution timed out"
    return {
        "code": code,
        "phase": backend,
        "message": message,
        "note": "Timeout" if timed_out else "BackendProcess",
        "line": line,
        "col": col,
        "span": span,
        "excerpt": "",
        "detail": text[-8000:],
    }


def run_backend_source(
    source: str,
    backend: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute ``source`` on ``backend`` in an isolated child process."""
    started = time.perf_counter()
    fd, result_path = tempfile.mkstemp(prefix=f"polyglot-{backend}-", suffix=".json")
    os.close(fd)
    result: dict[str, Any] = {
        "backend": backend,
        "ok": False,
        "timed_out": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "compile_ms": None,
        "run_ms": None,
        "peak_memory_bytes": None,
        "return_value": None,
        "diagnostics": [],
        "duration_ms": None,
    }
    command = [
        sys.executable,
        "-m",
        "api.cli",
        "exec",
        "--backend",
        backend,
        "--stdin",
        "--result-file",
        result_path,
    ]
    try:
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=_build_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            result["stderr"] = f"failed to start backend process: {error}"
            result["diagnostics"] = [
                _fallback_diagnostic(result["stderr"], source, backend, timed_out=False)
            ]
            return result

        try:
            stdout, stderr = proc.communicate(source, timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            stderr = (stderr or "") + (
                f"\n[visualizer] execution timed out after {timeout:g}s\n"
            )

        result["exit_code"] = proc.returncode
        result["stdout"] = stdout or ""
        result["stderr"] = stderr or ""

        payload = _read_result_file(result_path)
        if payload is not None:
            for key in (
                "compile_ms",
                "run_ms",
                "peak_memory_bytes",
                "return_value",
                "diagnostics",
            ):
                if key in payload:
                    result[key] = payload[key]
            result["ok"] = bool(
                payload.get("ok") and not result["timed_out"] and proc.returncode == 0
            )
        elif not result["timed_out"]:
            result["diagnostics"] = [
                _fallback_diagnostic(result["stderr"], source, backend, timed_out=False)
            ]
        else:
            result["diagnostics"] = [
                _fallback_diagnostic(result["stderr"], source, backend, timed_out=True)
            ]
        if result["timed_out"]:
            timed_out_diagnostic = _fallback_diagnostic(
                result["stderr"], source, backend, timed_out=True
            )
            if not any(
                isinstance(item, dict)
                and item.get("message") == timed_out_diagnostic["message"]
                for item in result["diagnostics"]
            ):
                result["diagnostics"].append(timed_out_diagnostic)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


def _parse_source(payload: dict[str, Any]) -> str:
    source = payload.get("source")
    if not isinstance(source, str):
        raise BadRequest("`source` must be a string")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise BadRequest(
            f"source is too large ({len(encoded)} bytes; limit {MAX_SOURCE_BYTES})"
        )
    return source


def _parse_backends(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("backends", list(DEFAULT_BACKENDS))
    if not isinstance(raw, list) or not raw:
        raise BadRequest("`backends` must be a non-empty list")
    backends: list[str] = []
    for item in raw:
        if not isinstance(item, str) or item not in ALLOWED_BACKENDS:
            raise BadRequest(
                f"unknown backend {item!r}; expected one of {', '.join(ALLOWED_BACKENDS)}"
            )
        if item not in backends:
            backends.append(item)
    return backends


def _parse_timeout(payload: dict[str, Any]) -> float:
    raw = payload.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BadRequest("`timeout` must be a number of seconds")
    return max(MIN_TIMEOUT, min(float(raw), MAX_TIMEOUT))


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one analysis/execution request payload (used by tests too)."""
    source = _parse_source(payload)
    backends = _parse_backends(payload)
    timeout = _parse_timeout(payload)
    started = time.perf_counter()

    def invoke(backend: str) -> dict[str, Any]:
        return run_backend_source(source, backend, timeout=timeout)

    if len(backends) == 1:
        results = [invoke(backends[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(backends)) as pool:
            results = list(pool.map(invoke, backends))

    return {
        "ok": bool(results) and all(item["ok"] for item in results),
        "source_length": len(source),
        "results": results,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


class VisualizerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the visualizer API and static page."""

    server_version = f"PolyglotVisualizer/{VERSION}"
    protocol_version = "HTTP/1.1"
    timeout = 30  # close idle keep-alive sockets instead of pinning threads

    # -- low-level responses ------------------------------------------------
    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise BadRequest("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise BadRequest("invalid Content-Length") from error
        if length < 0 or length > MAX_BODY_BYTES:
            raise BadRequest(
                f"request body is too large (limit {MAX_BODY_BYTES} bytes)"
            )
        body = self.rfile.read(length)
        if not body:
            raise BadRequest("request body is empty")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BadRequest(f"invalid JSON body: {error}") from error
        if not isinstance(payload, dict):
            raise BadRequest("request body must be a JSON object")
        return payload

    # -- routing ------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/visualizer.html"):
            self._serve_index()
            return
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, health_payload())
            return
        if path == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown path: {path}")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/analyze", "/api/run"):
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown path: {path}")
            return
        try:
            payload = self._read_json()
        except BadRequest as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return

        if path == "/api/analyze":
            try:
                source = _parse_source(payload)
            except BadRequest as error:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
                return
            try:
                result = analyze(source)
            except Exception as error:  # pragma: no cover - hard internal failure
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(error).__name__}: {error}"},
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return

        try:
            result = run_request(payload)
        except BadRequest as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        except Exception as error:  # pragma: no cover - hard internal failure
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"{type(error).__name__}: {error}"},
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _serve_index(self) -> None:
        try:
            body = INDEX_FILE.read_bytes()
        except OSError:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"visualizer frontend is missing: {INDEX_FILE}",
            )
            return
        self._send_bytes(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)


class VisualizerHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with quiet logging by default."""

    daemon_threads = True
    allow_reuse_address = True
    verbose = False


def health_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "name": "polyglot-visualizer",
        "version": VERSION,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cargo": shutil.which("cargo"),
        "native_available": shutil.which("cargo") is not None,
        "backends": list(ALLOWED_BACKENDS),
        "default_backends": list(DEFAULT_BACKENDS),
        "timeout_seconds": DEFAULT_TIMEOUT,
        "trusted_execution": True,
        "note": (
            "Code submitted to /api/run executes as trusted local code on this "
            "machine. Do not expose this server to untrusted networks."
        ),
    }


def make_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    verbose: bool = False,
) -> VisualizerHTTPServer:
    """Create (but do not start) a visualizer HTTP server."""
    server = VisualizerHTTPServer((host, port), VisualizerHandler)
    server.verbose = verbose
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    verbose: bool = False,
) -> int:
    """Run the visualizer until interrupted, returning a process exit code."""
    try:
        server = make_server(host, port, verbose=verbose)
    except OSError as error:
        print(
            f"error: cannot start visualizer on {host}:{port}: {error}",
            file=sys.stderr,
        )
        print("       try a different port with --port", file=sys.stderr)
        return 1

    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"Polyglot visualizer listening on {url}")
    print("Press Ctrl+C to stop.")
    print(
        "Note: code you run is trusted local code; the server binds to "
        f"{bound_host} and is not a security sandbox."
    )

    if open_browser:
        timer = threading.Timer(0.4, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        server.server_close()
    return 0


__all__ = [
    "ALLOWED_BACKENDS",
    "DEFAULT_BACKENDS",
    "DEFAULT_TIMEOUT",
    "VisualizerHTTPServer",
    "health_payload",
    "make_server",
    "run_backend_source",
    "run_request",
    "serve",
]
