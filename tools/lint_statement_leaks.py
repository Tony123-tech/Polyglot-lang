#!/usr/bin/env python3
"""Regression guard for commit 5d9d9d9."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors = []

def fail(msg): errors.append(msg)

p = (ROOT / "parser.py").read_text()
for node in ast.walk(ast.parse(p)):
    if isinstance(node, ast.FunctionDef) and node.name == "parse_block":
        body = ast.get_source_segment(p, node) or ""
        if "tail = e" not in body: fail("parse_block: tail = e missing")
        if "isinstance(e, (For, While, Loop))" not in body: fail("parse_block: For/While/Loop guard missing")
        if "_ifexpr_has_stmts" not in body: fail("parse_block: _ifexpr_has_stmts missing")
        if "_iflet_has_tail" not in body: fail("parse_block: _iflet_has_tail missing")
    if isinstance(node, ast.FunctionDef) and node.name == "parse_expr":
        body = ast.get_source_segment(p, node) or ""
        if "Statement-only nodes" not in body: fail("parse_expr: statement-only guard missing")

c = (ROOT / "backends/rust/codegen.py").read_text()
if "as_fn_body and isinstance(block.tail, IfLet)" not in c: fail("codegen: IfLet tail dispatch missing")

rt = (ROOT / "backends/python/runtime.py").read_text()
if "class _Option" in rt:
    tr = (ROOT / "backends/python/translator.py").read_text()
    if "class _Option" in tr: print("WARNING: _Option duplicated (technical debt from 5d9d9d9)")

if errors:
    print(str(len(errors)) + " issue(s):")
    for e in errors: print("  - " + e)
    sys.exit(1)
print("all statement-leak invariants hold")
