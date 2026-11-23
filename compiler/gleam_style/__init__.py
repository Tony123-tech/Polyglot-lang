"""Gleam-inspired static ADTs, exhaustive matching, and bit-array parsing.

Three pure-Python submodules bring Gleam's signature strengths to
``polyglot-lang`` while staying backend-neutral:

* :mod:`compiler.gleam_style.adt` — Hindley-Milner-style unification over
  algebraic data types and exhaustiveness checking of ``match`` expressions
  with concrete missing-pattern witnesses.
* :mod:`compiler.gleam_style.bitarray` — structured bit-segment pattern
  matching (``<<header:8, len:16, payload:binary>>``) with endianness and
  checked bounds validation instead of unchecked pointer casts.
* :mod:`compiler.gleam_style.contracts` — cross-target capability checks so
  pure functional data transformations stay portable across Python, Rust/PyO3,
  Zig, and optional BEAM/JS targets.

Diagnostics follow the shared ``E001``-``E008`` surface.  Everything here is
pure Python and never touches native toolchains, raw addresses, ``MAP_JIT``,
or PEP 667 frame locals.
"""

from __future__ import annotations

from compiler.gleam_style.adts import (
    AdtDef,
    AdtRegistry,
    ExhaustivenessResult,
    MissingPatterns,
    Variant,
    check_exhaustive,
    check_total_adt,
)
from compiler.gleam_style.bitarray import (
    BitarrayPattern,
    BitSegment,
    BitSegmentKind,
    MatchBitResult,
    match_bit_array,
    parse_bit_pattern,
)
from compiler.gleam_style.contracts import (
    Capability,
    Target,
    TargetContract,
    TargetDescriptor,
    emit_contract,
)

__all__ = [
    "AdtDef",
    "AdtRegistry",
    "BitSegment",
    "BitSegmentKind",
    "BitarrayPattern",
    "Capability",
    "ExhaustivenessResult",
    "MatchBitResult",
    "MissingPatterns",
    "Target",
    "TargetContract",
    "TargetDescriptor",
    "Variant",
    "check_exhaustive",
    "check_total_adt",
    "emit_contract",
    "match_bit_array",
    "parse_bit_pattern",
]
