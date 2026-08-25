.PHONY: test lint type check reproduce benchmark clean-results smoke-tsfm

# Which optional backends the gate runs against. `EXTRAS` is a variable
# (override from the environment or the command line) because the two torch
# builds cannot coexist and `uv run --extra ...` syncs the environment to
# exactly the extras named — adding `--extra torch-cpu` here would silently
# swap a GPU box's cu121 torch out on every `make test`.
#
#   default    --extra classical            statsforecast + lightgbm; the
#                                           development gate is only a gate on
#                                           models/sf.py and models/lgbm.py if
#                                           their backends are importable —
#                                           without them their tests
#                                           `importorskip` and the suite goes
#                                           green having checked nothing.
#   CI         --extra classical --extra torch-cpu   (.github/workflows/ci.yml)
#   GPU box    EXTRAS="--extra classical --extra tsfm"   then the opt-in sets
#              run with VOLBENCH_RUN_TSFM=1 / VOLBENCH_RUN_GPU=1 (tests/conftest.py)
#
# Without torch the PatchTST/TSFM tests importorskip.
EXTRAS ?= --extra classical
UV_RUN := uv run $(EXTRAS)

TOY_FIXTURE := src/volbench/benchmarks/data/toy_asset_daily.csv
TOY_OUT     := data/toy_benchmark

test:
	$(UV_RUN) pytest

lint:
	$(UV_RUN) ruff check .

type:
	$(UV_RUN) mypy

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
	$(UV_RUN) python -m volbench.benchmarks.make_toy_asset
	@git diff --quiet -- $(TOY_FIXTURE) || { \
	  echo ""; \
	  echo "ERROR: regenerating $(TOY_FIXTURE) changed it."; \
	  echo "The committed fixture and its generator disagree, so the toy"; \
	  echo "benchmark is not reproducible from scratch. Inspect with:"; \
	  echo "    git diff -- $(TOY_FIXTURE)"; \
	  exit 1; \
	}
	$(UV_RUN) python -m volbench.benchmarks.toy --out-dir $(TOY_OUT)
	@echo ""
	@echo "reproduce: rebuilt $(TOY_OUT)/ (summary.csv, summary.md, one parquet per model)"

# The paper's numbers will grow into this target. Today it rebuilds the toy
# benchmark (the cheap models only), behind the full check suite.
reproduce: check benchmark

# Local only, never part of `reproduce`: the zero-shot foundation models and
# PatchTST over the same toy series, into their own ResultsStore. Needs the
# `tsfm` extra (CUDA torch + backends), cached HF weights, and ideally a GPU
# (docs/P2_INTEGRATION.md §7). Named explicitly rather than through $(EXTRAS)
# because the extras it needs are not negotiable.
SMOKE_TSFM_OUT := data/smoke_tsfm
smoke-tsfm:
	uv run --extra classical --extra tsfm python -m volbench.benchmarks.smoke_tsfm --out-dir $(SMOKE_TSFM_OUT)
