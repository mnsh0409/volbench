# Prompt J2 — Convergence forensics and loss tables

**Model/effort:** any capable model. Run `/effort high` before starting. This is careful extraction and tabulation, not inference.

**Prerequisite:** Prompt J1 has run and `docs/P3_ANALYSIS_ASSUMPTIONS.md` exists. **Read it first** rather than re-deriving the assumptions, and say so. If J1 reported an inert or leaking canary on any model, stop and say so instead of proceeding.
**Terminal:** the main integration checkout (the one that ran the primary grid), on branch `feat/p3-analysis`, branched from an up-to-date `main`.

---

## STOP — read this before anything else

Prompts K and L ran after this file was written. Three things changed, and the first can silently invalidate everything you compute.

### 1. The store is append-only, and 44 cells were replaced

Prompt L fixed two model defects and re-ran **44 cells** — 11 `lgbm` (CPU) and 33 TSFM (GPU) — under **new config hashes**. The superseded fragments are **still on disk beside the corrected ones**, because the store never deletes.

So before computing anything:

- Assert that **every fragment you read is the one the current manifest names.** Report the manifest's path and the git SHA of the commit you read it at.
- Report how many of the 143 cells resolve to hashes changed by L. **Expect exactly 44.** If it is 0, you are reading the pre-fix grid and everything downstream would be wrong with no error raised — stop and report. If it is neither 0 nor 44, stop and report that too.

This check exists because a stale manifest here produces entirely plausible tables and no exception. Treat "I read the right fragments" as something to prove, not assume.

### 2. Some J1 artifacts are superseded

`docs/P3_LEAKAGE_CANARY_EXT.md`'s rows for `lgbm`, `chronos`, `timesfm` and `moirai`, and `docs/P3_GRID*`'s records for those 44 cells, are superseded by **`docs/P3_MODEL_DEFECT_FIXES.md`**. Prefer the latter wherever they disagree, and say which you used.

`docs/P3_ANALYSIS_ASSUMPTIONS.md` remains valid — it records schema facts, not results — and you should still read it first rather than re-deriving it.

### 3. Two additions to the work below

- **§2a, QLIKE leverage.** J1 found **five target days within 1e-8 of zero that are scored**, contributing QLIKE terms of 8.6–13.9 and **up to 1.01% of a cell's QLIKE sum from five observations**. Report every QLIKE figure **twice — with and without those five rows** — and report which asset and date each belongs to. A ranking that turns on five observations needs to be visible as such. (Separately, 13 further days are exactly zero and lose QLIKE entirely; they are monotone bars with stale opens, the largest carrying a +4.13% close-to-close return. Report whether the invalid-target policy tests `< 0` or `<= 0`, since that decides whether these are handled or falling through.)
- **§1 and §2b, AutoARIMA.** J1 measured **non-zero optimizer status on 2,334 of 2,366 `autoarima` fits (98.6%)** — scipy BFGS status 2, never `maxiter`, never NaN — and 40 distinct ARMA orders selected. Carry that rate in the loss tables the same way the GARCH fallback rates are carried, and report the per-asset breakdown. Do not interpret it.

---
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

## 1. Convergence forensics

The primary grid produced **38 EWMA fallbacks in 7,101 GARCH-family fits (0.54%)**: `garch11` 4, `garch11_t` 18, `gjr` 16. Every one carries **SLSQP exit mode 8, "positive directional derivative for line search"** — the signature of a flat or ill-conditioned likelihood surface rather than a bad start value or an iteration cap.

The complete per-cell list (7 non-clean cells of 33, fully attributing all 38):

| asset | config | fits | fallback | rate |
|---|---|---|---|---|
| BTC-USD | garch11_t | 133 | 15 | 11.28% |
| HSI | gjr | 230 | 14 | 6.09% |
| DIA | garch11 | 234 | 3 | 1.28% |
| DIA | garch11_t | 234 | 3 | 1.28% |
| DIA | gjr | 234 | 1 | 0.43% |
| ETH-USD | gjr | 133 | 1 | 0.75% |
| SPY | garch11 | 234 | 1 | 0.43% |

Produce `docs/P3_CONVERGENCE_FORENSICS.md`:

1. **One row per fallback fit** (38 rows): asset, config, fit origin index, calendar date, the parameter vector *at the point the optimizer stopped*, exit flag, final log-likelihood. If failed-fit parameters are not retained, say so and add the retention rather than guessing.
2. **Boundary table.** For each of the 38, which parameter sat at or within tolerance of which bound, and its value. Answer explicitly: is `α+β` at or near 1 (the IGARCH boundary)? Is `ν` at either end of `[2.1, 50]`? Is `gjr`'s γ at a bound? State the tolerance and show the answer is not sensitive to it.
3. **The DIA question — the most informative thing here.** DIA is the only asset failing on **all three** specifications. Put the origin indices for all three configs side by side and report the intersection explicitly. Overlapping origins across three different specifications point at the data; disjoint origins point at the specifications. State which, with dates.
4. **The HSI question.** HSI fails only on `gjr` (14/230) while `garch11` and `garch11_t` are clean, and `gjr` ran clean on its first real-panel exposure (SPY pre-flight, 234 fits, 0 fallback). Report where HSI's 14 sit in γ / α+β space relative to its clean `gjr` fits.
5. **The BTC/ETH question.** BTC-USD `garch11_t` is 15/133; ETH-USD `garch11_t` is 0/133 on an identical calendar and span (ETH has 1/133 on `gjr` and nothing else). Report any *measurable* difference between the two series' fit windows at BTC's 15 failing origins — sample kurtosis, max absolute return, and the fitted `α+β` of the nearest converged fit, for both assets at those same dates. **Do not propose an explanation. Report the numbers.**
6. **Calendar placement.** The 38 dates, each marked for whether it falls inside a crisis window.
7. **Per-cell rate table** for all 33 GARCH-family cells, as `k/n` and percent, cells above 5% marked.

End with one sentence, no interpretation: does any of the 38 show a sign of a **coding defect** — a bound set wrong, a scale mismatch, an obviously bad starting value — as distinct from a flat surface?

---

## 2. Loss tables

`docs/P3_LOSS_TABLES.md`: per asset, a 13-row table with mean CRPS, mean log score, mean pinball (per level and averaged), mean QLIKE and mean FZ0 — each with a **standard error that accounts for serial dependence** (HAC / Newey–West; report the bandwidth rule and the chosen bandwidth), and the **n used**.

**Do not aggregate any loss across assets.** Equities score against an overnight-plus-range variance target and crypto against 5-minute realized variance; the levels are not comparable, so a pooled or averaged loss across the 11 assets is meaningless. Cross-asset summaries are rank-based and belong to a later prompt.

---

## 3. Pairwise-complete accounting

Every model-vs-model comparison must run on the **intersection of origins where both models have a finite score**. Per asset and per loss, produce the 13×13 matrix of **n used** and the matrix of **rows dropped** relative to the asset's origin count. Report the largest drop per asset.

This is not bookkeeping for its own sake: on NKX the variance-fed and return-fed models are scored on different samples (4,773 vs 4,794 rows), so a comparison quietly using different samples on the two sides would be invalid.

Write the matrices to `docs/` in machine-readable form (CSV or parquet) as well as markdown — the next prompt consumes them rather than recomputing them.

---

## 4. Report

In chat: the boundary table, the DIA intersection answer, the HSI and BTC/ETH numbers, the defect-or-flat-surface sentence, the largest pairwise drop per asset, and the file list with row counts. Do not paste full 13×13 matrices — they go in the files. **No interpretation, no rankings, no conclusions.**
