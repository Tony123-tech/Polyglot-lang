"""Check documentation consistency."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors = []

def fail(msg):
    errors.append(msg)

# 1. README and SPEC test baseline must match
readme = (ROOT / "README.md").read_text()
spec = (ROOT / "SPECIFICATION.md").read_text()
pattern = r"(\d+) passed, (\d+) skipped, (\d+) deselected"
readme_match = re.search(pattern, readme)
spec_match = re.search(pattern, spec)
if readme_match and spec_match and readme_match.groups() != spec_match.groups():
    fail(f"README and SPEC test baseline mismatch: {readme_match.groups()} vs {spec_match.groups()}")

# 2. SPEC section numbers must be unique and sequential
section_nums = re.findall(r"^- \*\*(\d+\.\d+)\*\*", spec, re.MULTILINE)
seen = set()
for num in section_nums:
    if num in seen:
        fail(f"duplicate SPEC section number: {num}")
    seen.add(num)

# 3. Every Known Limitation (10.x) must have a cross-reference
limitations = re.findall(r"^- \*\*(10\.\d+)\*\*", spec, re.MULTILINE)
for lim in limitations:
    if f"See {lim}" not in spec and f"see {lim}" not in spec:
        fail(f"Known Limitation {lim} has no cross-reference")

# 4. All error codes E001–E008 must appear in README
readme_codes = set(re.findall(r"`E(\d{3})`", readme))
for i in range(1, 9):
    code = f"{i:03d}"
    if code not in readme_codes:
        fail(f"error code E{code} missing from README")

if errors:
    print(f"{len(errors)} issue(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("all documentation invariants hold")
