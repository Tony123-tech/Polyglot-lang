"""Pydantic models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompileRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=1_000_000)
    backend: str = Field(default="python", pattern="^(python|native)$")
    trusted: bool = False


class CompileResponse(BaseModel):
    ok: bool
    names: list[str] = Field(default_factory=list)
    error: str | None = None
    executor: str | None = None
