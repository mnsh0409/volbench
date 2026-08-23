.PHONY: test lint type check reproduce

test:
	uv run pytest

lint:
	uv run ruff check .

type:
	uv run mypy

check: lint type test

# Placeholder: will rebuild every paper number from raw inputs in a pinned
# container. Grows with the pipeline; must stay green (CLAUDE.md rule 3).
reproduce: check
	@echo "reproduce: nothing to rebuild yet (v0.0.1 scaffold)"
