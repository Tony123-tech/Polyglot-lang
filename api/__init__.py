"""ASPI — Polyglot HTTP layer (compile-as-a-service)."""

from api.config import get_config
from api.server import app

__all__ = ["app", "get_config"]
