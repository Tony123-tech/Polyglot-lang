"""Analysis stages: type-level and proof-level passes over the core AST.

These modules were previously top-level modules at the repository root
(``monomorph.py``, ``termination.py``, ``struct_order.py``,
``match_diagnostics.py``). They moved in so the root directory holds only
entry points and packaging shims. Root-level shims re-export them for
backward compatibility; new code should import from ``compiler.analysis``.
"""

from __future__ import annotations
