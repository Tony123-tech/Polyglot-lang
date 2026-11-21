"""Frontend frame implementations for the 1,000-function framework rollout.

Batch 1 (ids ``0001``–``0100``) lives in this package.  The original ten
functions remain in :mod:`compiler.frontend.framework`; the remaining ninety
are split across :mod:`lexing`, :mod:`parsing`, :mod:`spans`, and
:mod:`ast_utils`.  All frontend failures use :class:`PolyglotError` from
:mod:`compiler.frontend.errors`.
"""

from __future__ import annotations

from compiler.frontend.errors import PolyglotError

__all__ = [
    "PolyglotError",
    "ast_utils",
    "errors",
    "framework",
    "lexing",
    "parsing",
    "spans",
]
