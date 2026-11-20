"""Documentation-sync helpers that are part of the framework registry.

The Markdown files in this directory are documentation assets; the Python
modules are importable helpers used by framework functions 91-93.
"""

from __future__ import annotations

from docs.framework import (
    docs_record_platform_constraints,
    docs_sync_capability_matrix,
    docs_sync_cli_and_backend_flags,
    union_backend_choices,
)

__all__ = [
    "docs_record_platform_constraints",
    "docs_sync_capability_matrix",
    "docs_sync_cli_and_backend_flags",
    "union_backend_choices",
]
