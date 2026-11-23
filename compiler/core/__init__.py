"""Core compiler stages: the real lexer, parser, AST, and expansion layer.

The modules here were previously top-level modules at the repository root
(``lexer.py``, ``parser.py``, ``ast_nodes.py``, ``include_expander.py``,
``inline_lang.py``). They moved in so the root directory holds only entry
points and packaging shims. Root-level shims re-export these modules for
backward compatibility; new code should import from ``compiler.core``.
"""

from __future__ import annotations
