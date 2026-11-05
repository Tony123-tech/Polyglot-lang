"""
Configuration for the API.

Environment variables:
    POLYGLOT_EXECUTOR         "subprocess" (B, default) or "docker" (C)
    POLYGLOT_QUEUE            "inproc" (A, default) or "redis"
    POLYGLOT_COMPILE_TIMEOUT  seconds, default 30
    POLYGLOT_TRUSTED_TOKEN    secret for trusted=true; empty = disabled
    POLYGLOT_DOCKER_IMAGE     image for docker executor, default "polyglot-compiler"
    POLYGLOT_DOCKER_MEMORY    memory limit, default "512m"
    POLYGLOT_DOCKER_CPUS      cpu limit, default "1.0"
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    executor: str = "subprocess"
    queue: str = "inproc"
    compile_timeout: int = 30
    trusted_token: str = ""
    docker_image: str = "polyglot-compiler"
    docker_memory: str = "512m"
    docker_cpus: str = "1.0"

    @property
    def has_trusted(self) -> bool:
        return bool(self.trusted_token)


def get_config() -> Config:
    return Config(
        executor=os.environ.get("POLYGLOT_EXECUTOR", "subprocess"),
        queue=os.environ.get("POLYGLOT_QUEUE", "inproc"),
        compile_timeout=int(os.environ.get("POLYGLOT_COMPILE_TIMEOUT", "30")),
        trusted_token=os.environ.get("POLYGLOT_TRUSTED_TOKEN", ""),
        docker_image=os.environ.get("POLYGLOT_DOCKER_IMAGE", "polyglot-compiler"),
        docker_memory=os.environ.get("POLYGLOT_DOCKER_MEMORY", "512m"),
        docker_cpus=os.environ.get("POLYGLOT_DOCKER_CPUS", "1.0"),
    )
