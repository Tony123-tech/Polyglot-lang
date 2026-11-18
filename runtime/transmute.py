"""Speculative dynamic-to-native transmutation foundations.

The layer profiles the exact argument types of a callable, asks a pluggable
specializer to build a faster path in the background, installs that path
atomically, and guards every specialised call with a tuple comparison.  A
guard miss deoptimises to the original fallback and continues correctly.

Safety boundaries are explicit:

* Compilation never happens on the calling thread's critical path.
* An in-flight call keeps using the callable it already resolved; live Python
  frames are never rewritten.
* Deoptimisation is a guard check plus a fallback call.  It is fast, but it is
  not "nanoseconds"; no cross-language guard can promise that on CPython.
* A specializer failure is contained.  The fallback remains authoritative and
  the failure is reported through :class:`PillarError` (``E004``) when the
  caller opts into strict error propagation.

The ``PolyglotSpecializer`` adapter drives the project's own JIT or native
backend from generated ``.poly`` source, so the same layer can target either
runtime.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError

# Honest label for the deopt cost model.  The guard itself is one tuple
# construction plus equality; the fallback call dominates.
DEOPT_COST_MODEL = "guard (tuple compare) + Python fallback call"


# ---------------------------------------------------------------------------
# Signatures and specialisation results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeSignature:
    """The exact runtime classes observed for positional arguments."""

    types: tuple[type, ...]

    def render(self) -> str:
        return "(" + ", ".join(ty.__name__ for ty in self.types) + ")"


def signature_of(args: tuple[Any, ...]) -> TypeSignature:
    return TypeSignature(tuple(type(value) for value in args))


@dataclass(frozen=True)
class Specialization:
    """A compiled callable plus the backend that produced it."""

    fn: Callable[..., Any]
    backend: str = "python"


@runtime_checkable
class Specializer(Protocol):
    """Builds a specialised callable for one exact signature."""

    def compile(
        self,
        name: str,
        signature: TypeSignature,
        fallback: Callable[..., Any],
    ) -> Specialization | Callable[..., Any]: ...


@dataclass(frozen=True)
class CompiledPath:
    signature: TypeSignature
    fn: Callable[..., Any]
    backend: str
    generation: int
    installed_at: float


@dataclass(frozen=True)
class TransmuteStats:
    name: str
    calls: int
    fallbacks: int
    specialized_calls: int
    deopts: int
    compilations: int
    failures: int
    disabled: bool
    active_signature: TypeSignature | None
    active_backend: str | None
    generation: int
    signatures: tuple[tuple[TypeSignature, int], ...]


# ---------------------------------------------------------------------------
# Transmuter
# ---------------------------------------------------------------------------


class SpeculativeTransmuter:
    """A guarded, profiled, background-specialising callable wrapper."""

    def __init__(
        self,
        name: str,
        fallback: Callable[..., Any],
        specializer: Specializer,
        *,
        threshold: int = 3,
        max_deopts: int = 3,
        max_signatures: int = 64,
        executor: concurrent.futures.Executor | None = None,
        raise_on_compile_error: bool = False,
    ) -> None:
        if threshold < 1:
            raise PillarError("E003", "threshold must be at least 1")
        if max_deopts < 0:
            raise PillarError("E003", "max_deopts must be non-negative")
        self.name = name
        self.fallback = fallback
        self.specializer = specializer
        self.threshold = threshold
        self.max_deopts = max_deopts
        self.max_signatures = max_signatures
        self.raise_on_compile_error = raise_on_compile_error

        self._owns_executor = executor is None
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"transmute-{name}",
        )
        self._lock = threading.RLock()
        self._counts: Counter[TypeSignature] = Counter()
        self._pending: set[TypeSignature] = set()
        self._failed: set[TypeSignature] = set()
        self._futures: list[concurrent.futures.Future[None]] = []
        self._path: CompiledPath | None = None
        self._generation = 0
        self._calls = 0
        self._fallbacks = 0
        self._specialized_calls = 0
        self._deopts = 0
        self._compilations = 0
        self._failures = 0
        self._disabled = False
        self._last_error: PillarError | None = None

    # -- calling -----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            with self._lock:
                self._calls += 1
                self._fallbacks += 1
            return self.fallback(*args, **kwargs)

        signature = signature_of(args)
        with self._lock:
            self._calls += 1
            path = self._path
            if path is not None and path.signature == signature:
                self._specialized_calls += 1
                fn = path.fn
            else:
                if path is not None:
                    self._deopt_locked()
                self._fallbacks += 1
                self._profile_locked(signature)
                fn = None
        if fn is not None:
            return fn(*args)
        return self.fallback(*args)

    # -- profiling ---------------------------------------------------------

    def _profile_locked(self, signature: TypeSignature) -> None:
        if self._disabled:
            return
        if signature not in self._counts and len(self._counts) >= self.max_signatures:
            return
        self._counts[signature] += 1
        if (
            self._counts[signature] < self.threshold
            or self._path is not None
            or signature in self._pending
            or signature in self._failed
        ):
            return
        self._pending.add(signature)
        future = self._executor.submit(self._compile, signature)
        self._futures.append(future)

    def _deopt_locked(self) -> None:
        path = self._path
        if path is None:
            return
        self._deopts += 1
        self._path = None
        self._generation += 1
        if self.max_deopts and self._deopts > self.max_deopts:
            self._disabled = True

    # -- compilation -------------------------------------------------------

    def _compile(self, signature: TypeSignature) -> None:
        try:
            result = self.specializer.compile(self.name, signature, self.fallback)
        except PillarError as error:
            self._record_failure(signature, error)
            return
        except Exception as error:  # pragma: no cover - defensive boundary
            self._record_failure(
                signature,
                PillarError(
                    "E004",
                    f"specializer failed for {self.name}",
                    reason=f"{type(error).__name__}: {error}",
                    help_text="inspect the specializer and keep the fallback",
                ),
            )
            return

        if isinstance(result, Specialization):
            fn, backend = result.fn, result.backend
        elif callable(result):
            fn, backend = result, "python"
        else:
            self._record_failure(
                signature,
                PillarError(
                    "E004",
                    "specializer returned a non-callable result",
                    reason=f"got {type(result).__name__}",
                ),
            )
            return
        if not callable(fn):
            self._record_failure(
                signature,
                PillarError("E004", "specialized path is not callable"),
            )
            return

        with self._lock:
            self._pending.discard(signature)
            self._failed.discard(signature)
            self._generation += 1
            self._compilations += 1
            self._last_error = None
            self._path = CompiledPath(
                signature=signature,
                fn=fn,
                backend=backend,
                generation=self._generation,
                installed_at=time.monotonic(),
            )

    def _record_failure(self, signature: TypeSignature, error: PillarError) -> None:
        with self._lock:
            self._pending.discard(signature)
            self._failed.add(signature)
            self._failures += 1
            self._last_error = error

    # -- control -----------------------------------------------------------

    def wait(self, timeout: float | None = 10.0) -> bool:
        """Wait for outstanding compilations and return whether a path is live."""
        with self._lock:
            futures = list(self._futures)
        if futures:
            concurrent.futures.wait(futures, timeout=timeout)
        with self._lock:
            if self.raise_on_compile_error and self._last_error is not None:
                raise self._last_error
            return self._path is not None

    def force_specialize(
        self, signature: TypeSignature | None = None
    ) -> CompiledPath | None:
        """Synchronously compile a signature (mainly for startup/warm-up)."""
        with self._lock:
            if signature is None:
                if not self._counts:
                    return None
                signature = self._counts.most_common(1)[0][0]
            if signature in self._pending:
                return None
            self._failed.discard(signature)
            self._pending.add(signature)
        self._compile(signature)
        with self._lock:
            return self._path

    def deoptimize(self) -> None:
        """Drop the active path immediately; the fallback stays authoritative."""
        with self._lock:
            self._deopt_locked()

    def path(self) -> CompiledPath | None:
        with self._lock:
            return self._path

    def profile(self) -> tuple[tuple[TypeSignature, int], ...]:
        with self._lock:
            return tuple(self._counts.most_common())

    def stats(self) -> TransmuteStats:
        with self._lock:
            path = self._path
            return TransmuteStats(
                name=self.name,
                calls=self._calls,
                fallbacks=self._fallbacks,
                specialized_calls=self._specialized_calls,
                deopts=self._deopts,
                compilations=self._compilations,
                failures=self._failures,
                disabled=self._disabled,
                active_signature=path.signature if path is not None else None,
                active_backend=path.backend if path is not None else None,
                generation=self._generation,
                signatures=tuple(self._counts.most_common()),
            )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def transmute(
    fallback: Callable[..., Any],
    specializer: Specializer,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> SpeculativeTransmuter:
    """Wrap an untyped callable in a speculative transmutation layer."""
    return SpeculativeTransmuter(
        name or getattr(fallback, "__name__", "anonymous"),
        fallback,
        specializer,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dual-backend adapter
# ---------------------------------------------------------------------------

_TYPE_MAP: Mapping[type, str] = {
    int: "i64",
    float: "f64",
    bool: "bool",
}


def polyglot_type(ty: type) -> str:
    mapped = _TYPE_MAP.get(ty)
    if mapped is None:
        raise PillarError(
            "E007",
            f"cannot transmute argument type `{ty.__name__}` to a native path",
            reason="the scalar native subset only accepts int, float, and bool",
            help_text="keep using the Python fallback for this signature",
        )
    return mapped


@dataclass
class PolyglotSpecializer:
    """Compile a generated ``.poly`` function with the JIT or native backend.

    ``source_for(name, signature)`` must return a complete program that
    defines ``entry`` (default: ``name``).  Use :func:`polyglot_type` to spell
    the observed argument types.
    """

    source_for: Callable[[str, TypeSignature], str]
    entry: str | None = None
    backend: str = "jit"

    def compile(
        self,
        name: str,
        signature: TypeSignature,
        fallback: Callable[..., Any],
    ) -> Specialization:
        del fallback
        source = self.source_for(name, signature)
        program = parse(lex(source))
        from backends import get_backend

        try:
            namespace = get_backend(self.backend, trusted=True).compile(program)
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E004",
                f"{self.backend} specialization failed for `{name}`",
                reason=f"{type(error).__name__}: {error}",
                help_text="keep the Python fallback or fix the generated program",
            ) from error
        entry = self.entry or name
        if entry not in namespace:
            raise PillarError(
                "E006",
                f"specialized program does not export `{entry}`",
                help_text="check source_for() and the backend namespace",
            )
        fn = namespace[entry]
        if not callable(fn):
            raise PillarError("E004", f"specialized export `{entry}` is not callable")
        return Specialization(fn=fn, backend=self.backend)


def jit_inline_cache_polymorphic(
    profile: Sequence[tuple[TypeSignature, int]],
    *,
    ways: int = 4,
) -> tuple[tuple[TypeSignature, int], ...]:
    """Return the hottest signatures that fit a modelled inline cache.

    This is a profile selection helper, not a real PIC: the dispatch still
    uses a single guarded path.  ``ways`` must be at least one.
    """
    if ways < 1:
        raise PillarError("E003", "ways must be at least 1")
    return tuple(list(profile)[:ways])


__all__ = [
    "DEOPT_COST_MODEL",
    "CompiledPath",
    "PolyglotSpecializer",
    "Specialization",
    "Specializer",
    "SpeculativeTransmuter",
    "TransmuteStats",
    "TypeSignature",
    "jit_inline_cache_polymorphic",
    "polyglot_type",
    "signature_of",
    "transmute",
]
