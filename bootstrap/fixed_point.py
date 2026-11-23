"""Stage 2: re-read the IR and prove the digest is a fixed point.

The check compiles the printed IR again — IR in, IR out — and requires the
digest to be unchanged. That is the property a self-hosting compiler needs from
its own intermediate form: printing and re-reading must not change meaning.

Bounded model (SPEC 8.21, 10.51): stage 2 runs the same Python implementation,
so this is IR canonicality rather than a second-generation binary.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .ir import parse_ir
from .stage1 import Artifact
from .subset import digest, emit_ir


@dataclass(frozen=True)
class FixedPointReport:
    """How many rounds stage 2 needed, and whether the digest held."""

    stable: bool
    rounds: int
    stage1_digest: str
    stage2_digest: str
    ir_lines: int


def bootstrap_verify_stage2_fixed_point(
    artifact: Artifact, *, max_rounds: int = 4
) -> FixedPointReport:
    """Re-read and re-emit the IR until the digest repeats."""
    if max_rounds <= 0:
        raise PillarError("E003", "fixed-point round budget must be positive")
    if not artifact.ir:
        raise PillarError("E006", "artifact contains no IR to verify")

    text = artifact.ir
    rounds = 0
    stage2 = artifact.digest
    for rounds in range(1, max_rounds + 1):
        program = parse_ir(text)
        text = emit_ir(program)
        stage2 = digest(text)
        if stage2 == artifact.digest:
            return FixedPointReport(
                stable=True,
                rounds=rounds,
                stage1_digest=artifact.digest,
                stage2_digest=stage2,
                ir_lines=len(text.splitlines()),
            )
    raise PillarError(
        "E004",
        f"stage 2 did not reach a fixed point after {max_rounds} round(s)",
        reason=f"stage 1 digest {artifact.digest[:12]} never repeated",
        help_text="the IR printer and reader disagree; fix the emitter before claiming self-hosting",
    )
