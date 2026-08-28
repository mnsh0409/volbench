# P3 — the study driver is now in version control

**The problem.** `data/grid_primary/run_grid.py` (393 lines) produced every
number in the primary grid, and it sat under `data/`, which is gitignored — so
it was never committed. The committed artifacts were the package,
`docs/P3_GRID.md` and `docs/P3_GRID_manifest.json`; `make reproduce` covers the
cheap models only. There was therefore no committed path from a clean checkout
to the headline results.

For this project that is not tidiness. Reproducibility is the paper's claim,
and a reader who can install the package but cannot run the study cannot check
it.

**The location rule is unchanged.** `tests/test_licensing_guard.py::TestNoDataIsTracked`
runs `git ls-files -- data/` and requires an empty answer; nothing under `data/`
is ever tracked, and there is no per-file judgement call. The error was putting
the driver under `data/` at all.

---

## 1. The new path

```
src/volbench/benchmarks/grid_primary.py
```

**This is the repository's existing convention, not a new directory.**
`src/volbench/benchmarks/` already holds the committed, runnable study drivers:

| module | invoked as | writes to |
|---|---|---|
| `benchmarks.make_toy_asset` | `python -m volbench.benchmarks.make_toy_asset` | the committed fixture |
| `benchmarks.toy` | `python -m volbench.benchmarks.toy --out-dir data/toy_benchmark` | `data/toy_benchmark/` |
| `benchmarks.smoke_tsfm` | `python -m volbench.benchmarks.smoke_tsfm --out-dir data/smoke_tsfm` | `data/smoke_tsfm/` |
| **`benchmarks.grid_primary`** | `python -m volbench.benchmarks.grid_primary` | `data/grid_primary/` |

Both existing drivers take `--out-dir` with a **working-directory-relative
default under `data/`** (`toy.py`: `default=Path("data/toy_benchmark")`), and
both are driven from the Makefile that way. `grid_primary` now matches. `scripts/`
and `studies/` do not exist in this repository and were not invented.

Run:

```
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --extra classical --extra tsfm python -m volbench.benchmarks.grid_primary
```

The driver still refuses to start unless both determinism pins are in force —
neither can be repaired after numpy is imported.

## 2. What changed in the file

Three edits, and nothing else. The full diff against the original is 4 hunks:

| # | change | why it was unavoidable |
|---|---|---|
| 1 | module docstring: the "not committed" paragraph replaced by the provenance note | it was a statement about the file's location, and the location changed |
| 2 | docstring invocation line: `python data/grid_primary/run_grid.py` -> `python -m volbench.benchmarks.grid_primary` | same |
| 3 | `HERE = Path(__file__).resolve().parent` -> `DEFAULT_OUT_DIR = Path("data/grid_primary")`, and `--out-dir` default follows it | **this is the one that had to change.** `--out-dir` defaulted to the driver's own directory. Inside the package that would write a study's store, manifest and reports into `src/volbench/benchmarks/` |

Not touched: the pin check, `model_configs()` (all 13 configs and every
hyperparameter), `asset_data()` and its leading trim, `ARM`, `SEED`,
`missing_reason_counts()`, `report()`, the `run_grid` call, the executors, the
worker counts. No orchestration, no configuration, no model spec.

## 3. It contains no data

Checked mechanically rather than by reading: every `ast.Constant` in the file
was enumerated. **Every numeric literal in the driver is configuration, a
formatting constant, or an index.**

| literal | line | what it is |
|---|---|---|
| `20260825` | 109 | the grid seed |
| `500`, `21`, `1` | 110 | window, refit cadence, step — the protocol arm |
| `0.94` | 134 | EWMA's lambda — a model hyperparameter |
| `0` | 135, 136 | GARCH's asymmetry order `o` |
| `12`, `1` | 305, 370 | CPU worker count, GPU worker count |
| `1024.0`, `60`, `2` | 253, 264, 278, 320, 386 | GiB conversion, seconds-to-minutes, `round`/`indent` arguments |
| `1`, `0`, `2` | 240, 389 | `str.split` maxsplit arguments and the exit code |
| `1` | 345 | the horizon tuple `(1,)` |
| `0`, `1`, `-1` | 210, 211 | first/last element of the returns index, for `panel_start`/`panel_end` |

**No series value, no price, no variance number, no digest.** Every string
literal is a label, a column name, an error message or a format string. The
only data that reaches the file is read at run time through
`volbench.data.panel.build_panel()`, and the only data-derived values it emits
are `panel_start` / `panel_end` — two dates, computed from the loaded index, not
literals.

Nothing had to be moved to a file under `data/` and no path reference was
needed.

## 4. Guard result

```
$ uv run pytest tests/test_licensing_guard.py -q
13 passed

$ git ls-files -- data/
(empty)
```

`TestNoDataIsTracked` is green. `git ls-files -- data/` returns nothing: the
store, the manifests, the reports and the raw archives all stay untracked,
exactly as before. `.gitignore`'s root-anchored `/data/` is unchanged, and
`src/volbench/data/` (source code) is unaffected by it, which is why the anchor
is there.

The original file was **not** deleted from `data/grid_primary/`. It is
gitignored either way, so leaving it costs the repository nothing, and the run
it produced is legible beside its own store. `src/volbench/benchmarks/grid_primary.py`
is the committed copy and the one to run.

## 5. It still runs, and still resumes 143 cache hits

The failure mode a move invites is a changed relative path, so this was
verified rather than assumed. Run from the repository root, from the new
location, against the existing store, with `--tag resume_after_move` so the
committed manifest and report are not overwritten:

```
$ NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --extra classical --extra tsfm \
  python -m volbench.benchmarks.grid_primary --tag resume_after_move

cells attempted 143  computed 0  cached 143  failed 0
wall clock 0.3 min   peak RSS 1.00 GiB
```

**143 cached, 0 computed, 0 failed.** Every cell's config hash resolved to a
fragment already in `data/grid_primary/store/`, so the driver from its new home
addresses exactly the same 143 experiments as the driver from its old one — the
strongest available statement that nothing about the run's identity moved with
the file.

Store integrity, checked two ways over all 286 files:

| check | result |
|---|---|
| SHA-256 of every fragment and sidecar, before vs after | **identical**, 286/286 |
| size and mtime of every file, before vs after | **identical** — not merely rewritten to the same bytes, but **not rewritten at all** |

The per-model report the resumed run printed reproduces the original run's
`missing_reason` counts and fallback counts cell for cell (e.g. `SPY/garch11`
234 fits, 1 fallback, 1 non-converged; `TWSE/*` 80 missing each).

## 6. Two more things now committed for the same reason

The same argument applied to two scripts written for this phase, so both went
to the same place rather than into a scratch directory:

| module | what it produces |
|---|---|
| `src/volbench/benchmarks/leakage_canary.py` | docs/P3_LEAKAGE_CANARY_EXT.md — the extended leakage canary |
| `src/volbench/benchmarks/fit_diagnostics_probe.py` | docs/P3_INSTRUMENTATION_GAP.md — what each backend exposes about a fit |

Both take `--out-dir`/`--out` defaulting under `data/`, neither writes to the
primary store, and neither is imported by `volbench.analysis` (which is
forbidden from importing `volbench.benchmarks` at all —
`tests/test_analysis.py::TestBoundary`).

## 7. Drift flagged, not edited

`docs/P3_GRID.md` names the old path in two places — its header table
(`Driver | data/grid_primary/run_grid.py (gitignored with the results)`) and
§4 ("the only code written for this run is the driver, which is gitignored with
the results it produced"). Both were true when that document was written and
are now stale.

They are **not** edited here. P3_GRID.md is the run report for a specific run
on a specific commit, and silently rewriting its account of the tree it ran in
would make it a worse record, not a better one. The correction lives in this
document, which is the one about the move.

## 8. What this does not fix

`make reproduce` still covers the cheap models only, and adding the primary
grid to it would put a 70-minute GPU run behind `make check`. The committed
path from a clean checkout to the headline numbers now exists and is one
command; wiring it into a Make target, and what that target should require of
the machine, is a separate decision.

The **data digests** remain uncommitted: `series_sha256`, `fit_series_sha256`,
`proxy.sha256` and `raw_sha256` live only in the 143 JSON sidecars under
`data/grid_primary/store/`, and `docs/P3_GRID_manifest.json` carries no digest
field. So a clean checkout can now *run* the study, but cannot check that the
data it downloads is the data the published numbers came from. Flagged in
docs/P3_ANALYSIS_ASSUMPTIONS.md §4, not fixed here: what a manifest publishes
is a format decision.
