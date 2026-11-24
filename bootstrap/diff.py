"""Compare two stage artifacts by IR digest."""

from __future__ import annotations

from dataclasses import dataclass

from .stage1 import Artifact


@dataclass(frozen=True)
class DiffReport:
    """Whether two artifacts agree, and where they first disagree."""

    equal: bool
    left_digest: str
    right_digest: str
    first_diff_line: int | None = None

    @property
    def short(self) -> str:
        if self.equal:
            return f"{self.left_digest[:12]} = {self.right_digest[:12]}"
        return f"{self.left_digest[:12]} != {self.right_digest[:12]} (line {self.first_diff_line})"


def bootstrap_diff_stage_ir_hashes(left: Artifact, right: Artifact) -> DiffReport:
    """Return whether the two IR digests match, with the first differing line."""
    if left.digest == right.digest:
        return DiffReport(True, left.digest, right.digest)
    left_lines = left.ir.splitlines()
    right_lines = right.ir.splitlines()
    for index, (a, b) in enumerate(zip(left_lines, right_lines, strict=False)):
        if a != b:
            return DiffReport(False, left.digest, right.digest, index + 1)
    return DiffReport(
        False, left.digest, right.digest, min(len(left_lines), len(right_lines)) + 1
    )
