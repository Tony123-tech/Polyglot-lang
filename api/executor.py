"""
Executor: run a compile request in isolation.

Two modes:
- "subprocess" (B): spawn `python -m api.cli` with source on stdin.
  Crash-safe, timeout-safe, but not a sandbox.
- "docker" (C): `docker run --rm --network=none --read-only ...`.
  Real sandbox (no network, read-only FS, memory + CPU limits).

Both modes return (returncode, stdout, stderr).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from api.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SubprocessExecutor:
    """B: spawn a child Python process."""

    mode = "subprocess"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def run(
        self, source: str, backend: str, trusted: bool,
    ) -> tuple[int, str, str]:
        args = [
            sys.executable, "-m", "api.cli", "compile",
            "--backend", backend, "--stdin",
        ]
        if trusted:
            args.append("--trusted")
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(source.encode()),
                timeout=self.cfg.compile_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"timeout after {self.cfg.compile_timeout}s"
        if proc.returncode is None:
            return 1, "", "process terminated without return code"
        return proc.returncode, stdout.decode(), stderr.decode()


class DockerExecutor:
    """C: run a container with no network, read-only FS, resource limits."""

    mode = "docker"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def run(
        self, source: str, backend: str, trusted: bool,
    ) -> tuple[int, str, str]:
        args = [
            "docker", "run", "--rm", "-i",
            "--network=none",
            "--read-only",
            f"--memory={self.cfg.docker_memory}",
            f"--cpus={self.cfg.docker_cpus}",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            self.cfg.docker_image,
            "python", "-m", "api.cli", "compile",
            "--backend", backend, "--stdin",
        ]
        if trusted:
            args.append("--trusted")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(source.encode()),
                timeout=self.cfg.compile_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"timeout after {self.cfg.compile_timeout}s"
        if proc.returncode is None:
            return 1, "", "process terminated without return code"
        return proc.returncode, stdout.decode(), stderr.decode()

def make_executor(cfg: Config) -> SubprocessExecutor | DockerExecutor:
    if cfg.executor == "docker":
        return DockerExecutor(cfg)
    return SubprocessExecutor(cfg)
