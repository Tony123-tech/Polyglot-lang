"""
FastAPI server for compile-as-a-service.

Security model:
- All compile requests go through the executor (subprocess B or docker C).
- The server process NEVER calls backend.compile() directly.
- trusted=true requires X-Trusted-Token header matching POLYGLOT_TRUSTED_TOKEN.
- If POLYGLOT_TRUSTED_TOKEN is empty, trusted=true always returns 403.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException

from api.config import get_config
from api.executor import make_executor
from api.models import CompileRequest, CompileResponse
from api.worker import make_queue

cfg = get_config()
executor = make_executor(cfg)
queue = make_queue(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await queue.start()
    try:
        yield
    finally:
        await queue.stop()


app = FastAPI(title="polyglot-lang API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "executor": executor.mode,
        "queue": cfg.queue,
        "trusted_enabled": cfg.has_trusted,
    }


@app.post("/compile", response_model=CompileResponse)
async def compile_endpoint(
    req: CompileRequest,
    x_trusted_token: str | None = Header(default=None),
) -> CompileResponse:
    # Trust gate.
    trusted = False
    if req.trusted:
        if not cfg.has_trusted:
            raise HTTPException(
                status_code=403,
                detail="trusted=true is disabled on this server",
            )
        if x_trusted_token != cfg.trusted_token:
            raise HTTPException(
                status_code=403,
                detail="trusted=true requires valid X-Trusted-Token",
            )
        trusted = True

    code, stdout, stderr = await queue.submit(req.source, req.backend, trusted)

    if code == 0:
        names = [line for line in stdout.splitlines() if line]
        return CompileResponse(ok=True, names=names, executor=executor.mode)
    return CompileResponse(
        ok=False, error=stderr or stdout, executor=executor.mode,
    )
