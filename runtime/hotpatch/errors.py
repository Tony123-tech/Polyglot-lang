"""Structured diagnostics for the quantum hot-patching subsystem.

The pillar-wide error surface uses ``E001``-``E008`` (see
``runtime.diagnostics``).  Hot-patch relocation, ABI, and journal-corruption
failures are reported as ``E008`` with the caller's character span and, when
the source text is available, the exact byte span.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

HOTPATCH_ERROR_CODE = "E008"


@dataclass(frozen=True)
class ByteSpan:
    """A character span paired with its UTF-8 byte offsets."""

    start: int
    end: int
    byte_start: int
    byte_end: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


def byte_offsets(source: str | None, span: tuple[int, int] | None) -> ByteSpan | None:
    """Translate a character span into byte offsets for ``source``."""
    if span is None:
        return None
    start, end = span
    if source is None:
        return ByteSpan(start, end, start, end)
    prefix = source[:start].encode("utf-8")
    body = source[start:end].encode("utf-8")
    return ByteSpan(start, end, len(prefix), len(prefix) + len(body))


def span_from_line_col(
    source: str | None, line: int, col: int, width: int = 1
) -> tuple[int, int] | None:
    """Convert a 1-based ``line``/``col`` pair into a character span."""
    if source is None or line <= 0:
        return None
    offset = 0
    for _ in range(line - 1):
        newline = source.find("\n", offset)
        if newline == -1:
            return None
        offset = newline + 1
    start = offset + max(col - 1, 0)
    return (start, min(start + width, len(source)))


class HotpatchError(PillarError):
    """A hot-patch failure carrying character and byte spans.

    ``code`` defaults to ``E008`` (relocation/ABI/journal failure).  Platform
    or unsupported-mechanism failures use ``E007`` so callers can distinguish
    "cannot do this here" from "the patch you asked for is malformed".
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = HOTPATCH_ERROR_CODE,
        span: tuple[int, int] | None = None,
        source: str | None = None,
        line: int = 0,
        col: int = 0,
        reason: str | None = None,
        help_text: str | None = None,
        note: str | None = None,
    ) -> None:
        if span is None:
            span = span_from_line_col(source, line, col)
        self.byte_span = byte_offsets(source, span)
        self.source = source
        super().__init__(
            code,
            message,
            reason=reason,
            help_text=help_text,
            note=note or "HotpatchError",
            line=line,
            col=col,
            span=span,
        )

    @property
    def bytes(self) -> tuple[int, int] | None:
        if self.byte_span is None:
            return None
        return (self.byte_span.byte_start, self.byte_span.byte_end)

    def format(self) -> str:
        text = super().format()
        if (
            self.byte_span is not None
            and self.byte_span.byte_start != self.byte_span.start
        ):
            text += f"\n  bytes: {self.byte_span.byte_start}..{self.byte_span.byte_end}"
        return text

    def __str__(self) -> str:
        return self.format()


def relocation_error(
    message: str,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
    detail: str | None = None,
) -> HotpatchError:
    return HotpatchError(
        message,
        span=span,
        source=source,
        reason=detail or "a relocation target was missing, ambiguous, or not writable",
        help_text=(
            "verify the symbol is bound, the page is writable, and the "
            "replacement preserves the ABI signature"
        ),
    )


def abi_error(
    message: str,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
    detail: str | None = None,
) -> HotpatchError:
    return HotpatchError(
        message,
        span=span,
        source=source,
        reason=detail or "the old and new ABI signatures differ",
        help_text="keep argument and return layouts identical, or ship a new process",
    )


def unsupported(feature: str, *, span: tuple[int, int] | None = None) -> HotpatchError:
    return HotpatchError(
        f"{feature} is not available on this platform",
        code="E007",
        span=span,
        reason="the required kernel/CPU primitive is missing or refused by the loader",
        help_text="use the Python host bridge or an equivalent supported capability",
    )


__all__ = [
    "HOTPATCH_ERROR_CODE",
    "ByteSpan",
    "HotpatchError",
    "abi_error",
    "byte_offsets",
    "relocation_error",
    "span_from_line_col",
    "unsupported",
]
