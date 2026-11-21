"""Dependency-free pub/sub for hotpatch telemetry events.

Runtime modules emit small, count-only observations here; the API layer may
subscribe a bridge that forwards them to session telemetry.  This module never
imports ``api.*`` (or anything else from the project), so hotpatch operations
cannot form an import cycle with telemetry and never depend on it being healthy.

Subscribers are isolated: a callback that raises is ignored and can never break
the runtime operation that emitted the event.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

Callback = Callable[[str, dict[str, Any]], None]

_lock = threading.RLock()
_subscribers: list[Callback] = []


def subscribe(callback: Callback) -> Callable[[], None]:
    """Register ``callback`` and return an idempotent unsubscribe function."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    with _lock:
        if callback not in _subscribers:
            _subscribers.append(callback)
    done = False

    def unsubscribe() -> None:
        nonlocal done
        if done:
            return
        with _lock:
            if callback in _subscribers:
                _subscribers.remove(callback)
            done = True

    return unsubscribe


def emit(event: str, **payload: Any) -> None:
    """Publish one event to all current subscribers (never raises)."""
    with _lock:
        callbacks = tuple(_subscribers)
    message = str(event)
    data = dict(payload)
    for callback in callbacks:
        try:
            callback(message, data)
        except Exception:  # noqa: S110 - telemetry must never break runtime
            pass


def reset_subscribers() -> None:
    """Forget all subscribers (primarily useful for isolated tests)."""
    with _lock:
        _subscribers.clear()


__all__ = ["emit", "reset_subscribers", "subscribe"]
