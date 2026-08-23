.PHONY: test lint type check reproduce benchmark clean-results

TOY_FIXTURE := src/volbench/benchmarks/data/toy_asset_daily.csv
TOY_OUT     := data/toy_benchmark

test:
	uv run pytest

lint:
	uv run ruff check .

type:
	uv run mypy

check: lint type test

clean-results:
	rm -rf $(TOY_OUT)

# Rebuild the M1 toy benchmark from scratch. "From scratch" is literal and
# matters (CLAUDE.md rule 3): the results directory is deleted first, so the
# ResultsStore cache cannot short-circuit the run and quietly re-publish
# yesterday's numbers, and the synthetic input series is regenerated from its
# committed generator rather than trusted as an opaque file.
#
# Regenerating the fixture must be a no-op against what is committed. If it is
# not, the benchmark's input is not reproducible and every number downstream
# of it is unanchored, so this stops rather than carrying on.
benchmark: clean-results
	uv run python -m volbench.benchmarks.make_toy_asset
	@git diff --quiet -- $(TOY_FIXTURE) || { \
	  echo ""; \
	  echo "ERROR: regenerating $(TOY_FIXTURE) changed it."; \
	  echo "The committed fixture and its generator disagree, so the toy"; \
	  echo "benchmark is not reproducible from scratch. Inspect with:"; \
	  echo "    git diff -- $(TOY_FIXTURE)"; \
	  exit 1; \
	}
	uv run python -m volbench.benchmarks.toy --out-dir $(TOY_OUT)
	@echo ""
	@echo "reproduce: rebuilt $(TOY_OUT)/ (summary.csv, summary.md, one parquet per model)"

# The paper's numbers will grow into this target. Today it rebuilds the M1
# toy benchmark, behind the full check suite.
reproduce: check benchmark
