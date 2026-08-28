# Prompt J2 — Convergence forensics and loss tables

**Model/effort:** any capable model. Run `/effort high` before starting. This is careful extraction and tabulation, not inference.

**Prerequisite:** Prompt J1 has run and `docs/P3_ANALYSIS_ASSUMPTIONS.md` exists. **Read it first** rather than re-deriving the assumptions, and say so. If J1 reported an inert or leaking canary on any model, stop and say so instead of proceeding.
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
