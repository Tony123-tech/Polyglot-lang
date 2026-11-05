"""Runtime helpers used by the Python backend."""

from __future__ import annotations


def panic(msg: str = "") -> None:
    raise RuntimeError(msg or "panic")


def assert_eq(a, b) -> None:
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")


def assert_ne(a, b) -> None:
    if a == b:
        raise AssertionError(f"{a!r} == {b!r}")
