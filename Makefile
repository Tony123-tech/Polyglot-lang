.PHONY: core test test-slow lint format format-check docs-sync check

# Front end, analysis, diagnostics and CLI: no Rust or Zig toolchain needed.
core:
	pytest -q -m core_only

test:
	pytest -q

test-slow:
	pytest -q -m slow

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

docs-sync:
	python3 tools/check_docs_sync.py

check: lint format-check docs-sync test
