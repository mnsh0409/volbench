# Prompt J1 — Evidence gate (run this before any analysis)

**Model/effort:** any capable model. Run `/effort high` before starting. This part is mechanical. Nothing here computes a result; it establishes whether the results are worth computing.

**Why this is a gate and not a warm-up.** Every item below can invalidate work done after it. If the leakage canary comes back inert or leaking on `patchtst`, the entire foundation-model column is suspect and there is no point computing model confidence sets over it. Do these first, report, and stop.
**Terminal:** the main integration checkout (the one that ran the primary grid), on branch `feat/p3-analysis`, branched from an up-to-date `main`.

**Session:** start this as a **fresh, separately named** Claude Code session (`claude -n vb-<this-prompt>`). Do not continue an earlier session — this prompt is written to be self-contained, and a session carrying prior conclusions is the specific failure mode it is designed to avoid.

---

## Your role and the one hard constraint

You are working on the **analysis layer** of volbench: it reads the completed primary grid out of the `ResultsStore` and produces tables, tests, and diagnostics.

**HARD CONSTRAINT — report, do not interpret.** Compute the numbers, write them to files, and report what you computed plus anything that looks *mechanically* wrong (a NaN where none should be, a p-value of exactly 0 or 1, a bootstrap that did not converge, a block length larger than n/4, a statistic that is not finite). **Do NOT rank models, do NOT say which model wins, do NOT call a result strong/weak/surprising/expected, do NOT write conclusions or an "interpretation" section.** The results review happens on the planning machine. If you find yourself typing "outperforms", "best", "as expected", or "notably", delete the sentence.

## Engineering rules (unchanged from the rest of the project)

- Python 3.12+, fully typed, small composable functions, `ruff`-clean, `pytest` for anything with a decidable answer.
- No notebooks as deliverables.
- **All outputs go to `docs/` — NEVER anywhere under `data/`.** `tests/test_licensing_guard.py::TestNoDataIsTracked` runs `git ls-files -- data/` and requires an empty answer. The guard tests **location, not content**, so there is never a per-file judgement call: if it is an output, it goes in `docs/`.
- Export the pinned environment for every run: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`.
- **The analysis layer must not be able to re-run a model.** Keep/add the structural test — the same shape as `econ.py`'s — asserting the analysis module does not import the model package. Analysis reads the store; it never fits.
- Push the branch **before** merging so CI sees it, and commit after each numbered section so partial completion is still useful.
- **Two traps this box has already hit — do not rediscover them.** (i) `gh` here is 2.4.0 and has neither `--branch` nor `--json`, so status polls built on those **fail silently and look exactly like "no problems"** — use the API poll. (ii) `run_grid`'s `on_cell` hook reports a whole lane at once, so a per-cell progress counter sits at zero while work proceeds — count store fragments, not log lines.

---

## 1. Verify these assumptions and write them down

I am writing this without the repo in front of me. **Each of these is an assumption. Check it, and if it is wrong, report the actual state and adapt rather than forcing my wording.** Do not silently "fix" a mismatch by changing the meaning of a metric.

1. The `ResultsStore` reads back into a long/tidy frame keyed by `(asset, config, horizon, protocol arm, origin/target_index, date)`, one row per scored day, including NaN-score rows carrying a `missing_reason`.
2. Per-row loss columns exist or are recomputable from stored forecasts + realized target: **CRPS, log score, pinball at the evaluated levels, QLIKE, FZ0 joint (VaR, ES)**. Report which are stored versus recomputed.
3. VaR and ES at the evaluated levels are recoverable per row from the stored predictive `Distribution`, or already stored. Report the levels actually evaluated.
4. `docs/` holds the primary-grid manifest with config hashes, thread pins and data digests.
5. Convergence and fallback status are recorded per **fit** (not per origin), with the fit's origin recoverable.
6. `econ.py` exists as a volatility-targeting backtest importing nothing from the package.
7. Crisis windows: check whether the codebase defines named ones. If it does, use exactly those and list them. If not, use only **GFC 2007-07-01 → 2009-06-30** and **COVID 2020-02-01 → 2020-04-30**, and say so.
8. **Which Python interpreter ran the primary grid.** Record it. The 3.12 `getattr_static` protocol defect was test-double-only and production `fit_status` was verified unaffected — but `fit_status` is the exact column that produced the 38 fallbacks, so the interpreter belongs in the record rather than in memory.

**Write the result to `docs/P3_ANALYSIS_ASSUMPTIONS.md`** as a table: assumption → confirmed / actual state. Later prompts read that file instead of re-deriving it, so it must be complete enough to stand alone.

---

## 2. The leakage canary covers 4 of 13 models

The grid's leakage audit corrupted SPY's raw OHLC strictly after target row 560 and reported future-corruption-identical / past-corruption-differs — a correct canary with a live inert-proof. But it ran over **`naive`, `ewma`, `garch11`, `har` only**. The nine it did not cover include every model where leakage is hardest to reason about from source alone.

Extend the same canary — through the driver's own bridge, not a stand-in — to at least these five, reporting the same two-line verdict for each:

- **`patchtst`** — highest priority. It *trains* per origin (that is why device class had to enter the config hash), so there is an optimizer, a data loader and dropout RNG in the path. "The driver adds no batching" is an argument; the canary is evidence.
- **`lgbm`** — trains, and has a known open issue with out-of-fold smearing. If anything leaks, this and `patchtst` are where.
- **`chronos`** — covers the zero-shot TSFM context-window path, which none of the four tested models exercises. Cheapest GPU cell at ~61 s.
- **`autoarima`** — covers the statsforecast path including its model selection.
- **`garch11_t`** — covers the Student-t branch and the `ν ≤ 50` bound.

Budget roughly 10 minutes on SPY, both corruption directions. **If any model reports "past-corruption: identical", stop and report immediately** — that means the canary is inert there and the audit proves nothing about that model, which is a different and worse finding than a leak. If a model cannot be canaried through the bridge for a structural reason, say what the reason is rather than skipping it silently.

Output: `docs/P3_LEAKAGE_CANARY_EXT.md`.

---

## 3. Ten of thirteen models have no convergence instrumentation

The run report shows `n_fits = 0` and a fallback rate of `nan` for the ten non-GARCH configs. For `naive` and `ewma` that is correct — nothing is estimated. For **`autoarima`, `autoets`, `lgbm`, `har` and the four TSFMs** it is not: several of those estimate, select, or optimize, and a reader cannot distinguish "never failed" from "never measured". A table printing a number for three models and `nan` for ten reads as if the ten were perfect.

Two steps, in order, and **do not re-run the grid**:

1. **Report the gap.** For each of the ten, state whether its backend exposes any convergence, selection or fit-status signal at all, and whether that signal is already in the stored fragments or sidecars. Where it is already stored, extract and report it post-hoc.
2. **Only if** it can be done with no config-hash change, no fragment rewrite, and a resumability re-run still showing 143 cached — wire the available signals into the same status column. **If it would require re-running any cell, do not do it**: write down what the wiring would be, and stop. Re-running a clean, gate-passed grid to add a diagnostic is a bad trade.

In every fallback-rate table anyone produces from here on, the ten uninstrumented configs must read **`not instrumented`**, never `nan` and never `0`.

Output: `docs/P3_INSTRUMENTATION_GAP.md`.

---

## 4. The study driver is not in version control

`data/grid_primary/run_grid.py` (393 lines) produced every number in the primary grid, and it sits under `data/`, which is gitignored — so it is **not committed**. The committed artifacts are the package, `docs/P3_GRID.md` and `docs/P3_GRID_manifest.json`; `make reproduce` covers cheap models only.

There is therefore **no committed path from a clean checkout to the headline results**. For this project that is not tidiness: reproducibility is the paper's claim, and a reader who can install the package but cannot run the study cannot check it.

The location rule is right and stays exactly as it is — *nothing under `data/` is ever tracked*, no per-file judgement. The error was putting the driver under `data/` at all.

1. Move it out of `data/` to a committed location for study drivers (`scripts/`, `studies/`, or the repo's existing convention — look for one before inventing a directory).
2. Read it and confirm explicitly that it contains **no series values, no prices, no derived variance numbers** — only configuration, paths and orchestration. If it holds any data value, move that value to a file under `data/` and leave a path reference.
3. Commit. Verify the licensing guard is green and `git ls-files -- data/` is still empty.
4. It must still run unchanged and still resume the existing store as 143 cache hits. Verify that — a moved file with changed relative paths is the obvious way to break it.

Output: `docs/P3_DRIVER_PROVENANCE.md` with the new path, the guard result and the resumability re-run.

---

## 5. Data validity checks (these can invalidate every table computed later)

Output: `docs/P3_ANALYSIS_VALIDITY.md`.

- **QLIKE positivity.** QLIKE needs strictly positive forecasts *and* strictly positive realized targets. Per asset: the minimum realized target, the count of target values at zero or at any floor, the minimum forecast across all 13 models, and how zeros and floors are currently handled. One silently floored zero-variance day can move an entire column.
- **Missing-row accounting.** Per asset × model: origins, scored (non-NaN) rows, and NaN rows broken down by `missing_reason`. Expected: NKX loses its first refit block (21 origins) on the eight variance-fed configs only, so NKX's five return-fed configs score 4,794 rows and its eight variance-fed score 4,773. Confirm that, and confirm no other asset has an unexpected gap.
- **Score finiteness.** Count non-finite values in every loss column per asset × model, and list any offending rows.
- **Alignment canary.** Pick one row per asset and independently recompute its CRPS (or QLIKE) by hand from the stored predictive distribution and the stored realized target; assert it matches the stored column. This catches an off-by-one between forecast and realization, the single most expensive error possible here. Known hazard: **results-frame `target_index` = raw-frame position minus one** (the driver's leading trim), which has already produced one false leak report.

---

## 6. Report

In chat: the assumptions table, the five canary verdicts, the instrumentation findings, the driver's new path and guard result, and the validity numbers. Flag anything mechanically wrong. **Then stop** — do not begin the forensics or the loss tables; those are separate prompts. No interpretation, no rankings, no conclusions.
