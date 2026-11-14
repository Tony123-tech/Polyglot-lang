"""ASPI — Polyglot HTTP layer (compile-as-a-service)."""

from api.config import get_config

__all__ = ["app", "get_config"]


def __getattr__(name: str):
    if name == "app":
        from api.server import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
