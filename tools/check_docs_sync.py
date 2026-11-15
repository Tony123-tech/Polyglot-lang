"""Assert that README.md and SPECIFICATION.md 12.1 record the same baseline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAST_RE = re.compile(
    r"(?P<passed>\d+) passed,\s*"
    r"(?P<skipped>\d+) skipped,\s*"
    r"(?P<deselected>\d+) deselected"
)
SLOW_RE = re.compile(r"(?P<passed>\d+) passed,\s*(?P<deselected>\d+) deselected")


def _baseline(text: str) -> tuple[int, int, int, int, int]:
    """Return (fast_passed, skipped, fast_deselected, slow_passed, slow_deselected)."""
    fast = FAST_RE.search(text)
    if fast is None:
        raise ValueError("fast baseline not found")
    slow = SLOW_RE.search(text)
    if slow is None:
        raise ValueError("slow baseline not found")
    return (
        int(fast.group("passed")),
        int(fast.group("skipped")),
        int(fast.group("deselected")),
        int(slow.group("passed")),
        int(slow.group("deselected")),
    )


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec = (ROOT / "SPECIFICATION.md").read_text(encoding="utf-8")

    try:
        readme_baseline = _baseline(readme)
        spec_baseline = _baseline(spec)
    except ValueError as error:
        print(f"docs-sync: {error}", file=sys.stderr)
        return 1

    if readme_baseline != spec_baseline:
        print(
            "docs-sync: README and SPEC 12.1 baselines disagree\n"
            f"  README: {readme_baseline}\n"
            f"  SPEC:   {spec_baseline}",
            file=sys.stderr,
        )
        return 1

    passed, skipped, deselected, slow_passed, slow_deselected = readme_baseline
    print(
        "docs-sync: baselines agree "
        f"(pytest -q: {passed} passed, {skipped} skipped, "
        f"{deselected} deselected; "
        f"pytest -m slow: {slow_passed} passed, {slow_deselected} deselected)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
