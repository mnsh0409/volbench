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

# Byte-identity of `reproduce` is a claim within one numpy SIMD kernel family.
# numpy's AVX-512-only float64 log/exp kernels differ from the x86-v3 ones in
# the last bit for some inputs, which moves the content digest of a computed
# proxy and with it every config hash (docs/P2_INTEGRATION.md §3.6). Pinning
# to x86-v3-or-lower here and in CI makes an AVX-512 machine compute what the
# committed identities were computed with; on a machine without AVX-512 the
# setting is a silent no-op. Override with NPY_DISABLE_CPU_FEATURES= to measure
# the difference on purpose.
export NPY_DISABLE_CPU_FEATURES ?= X86_V4 AVX512_ICL AVX512_SPR

# Pin the BLAS to one thread (D-032). Not a performance setting — a
# correctness one. Threaded OpenBLAS reorders a reduction by an ulp and
# arch's SLSQP turns that into a *different local optimum* of the GARCH
# likelihood: measured at 5.5e-1 relative on garch11_t and 9.2e-5 on
# garch11 (docs/P3_DETERMINISM.md §2). Unlike the kernel pin above, the
# thread count moves no content digest, so before D-032 the two answers
# shared one config hash. `blas_threads` is now hashed, so an unpinned run
# misses the cache instead of being served the wrong fragment — and pinning
# to 1 here is what keeps every machine on the same side of that split.
#
# Exported for the WHOLE target, not just the workers: a pool inherits the
# parent environment, and pinning only the workers is exactly how a pooled
# run stops reproducing the serial one.
#
# `?=` so a deliberate measurement can override it
# (OPENBLAS_NUM_THREADS=8 make benchmark); the run then records 8 in every
# config hash and lands in its own fragments rather than overwriting these.
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

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
