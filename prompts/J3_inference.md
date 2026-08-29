# Prompt J3 — Comparison inference, backtests, economic value

**Model/effort:** the strongest model available. Run `/effort max` before starting — it is session-only and resets, so set it every time you open this prompt. This is inference code where a subtle error produces plausible-looking tables that are wrong — a bandwidth rule quietly defaulting to zero lags, a block bootstrap resampling i.i.d., a pairwise comparison run on two different samples. None of those raise an exception.

**Prerequisite:** J1 and J2 have run. **Read `docs/P3_ANALYSIS_ASSUMPTIONS.md` and J2's pairwise-complete matrices** rather than re-deriving either, and say so. If J1 reported an inert or leaking canary on any model, or J2 found a coding defect behind the fallbacks, stop and say so instead of proceeding.
**Terminal:** the main integration checkout (the one that ran the primary grid), on branch `feat/p3-analysis`, branched from an up-to-date `main`.

---

## STOP — four amendments from J2, one of which threatens every p-value below

### 1. The HAC standard errors are lower bounds, and DM inherits that

J2 used the fixed rule `floor(4(n/100)^(2/9))` — bandwidth 9 on equities, 8 on crypto — and then did the right thing: it validated the estimator against a known AR(1) long-run variance. It recovers **99% / 94% / 60% of the truth at ρ = 0 / 0.5 / 0.9**. J2's own conclusion, correct: *these SEs are lower bounds.*

Volatility loss differentials are strongly persistent, so on the comparisons this paper is about the estimator is nearer the 60% end than the 99% end. **A DM denominator that is 60% of the truth inflates every statistic and makes every p-value too small.** Nothing downstream catches this — the matrices look entirely normal. Treat it as the single largest threat to §1's validity.

Required:

- **Pre-whiten before applying the HAC.** Fit an AR(1) to each loss differential, apply the kernel estimator to the residuals, then recolour (the Andrews–Monahan construction). This is the standard remedy for exactly this failure and it is not optional here.
- **Use a data-driven bandwidth** — Andrews (1991) or the Newey–West (1994) plug-in — not the fixed n-only rule. Report the chosen bandwidth per pair, or its distribution.
- **Report the first-order autocorrelation ρ̂₁ of each pair's loss differential.** It is the diagnostic that says how much the correction is doing.
- **Report an effective sample size alongside n.** At ρ = 0.9 the effective n for a mean is roughly `n(1−ρ)/(1+ρ)` — about 260 for a 4,900-origin asset. Printing `n = 4,904` alone, when the differential carries that much dependence, overstates the evidence by an order of magnitude. Print both.
- **Sensitivity ladder, reported:** re-run the DM matrices at the fixed rule, at the automatic bandwidth, and at twice the automatic bandwidth. Report **how many pairs change significance at 5%** between them. If that count is large, say so plainly — it is a finding about the inference, not a nuisance.
- **Cross-check the MCS.** The Politis–White automatic block length is data-driven and should handle persistence better than the fixed rule did, but verify it: compare each chosen block length against that series' measured ρ̂₁. A short block on a highly persistent loss series means the rule has failed, not answered.

### 2. Crisis windows: do not move them

Two edge effects surfaced in J2, and both are traps.

Thirteen of the 38 GARCH fallbacks sit inside the span `stress_2025_26` names and **deliberately leaves undated**; DIA's fallback cluster sits at 2008-07-23 and 2008-08-21, just *outside* a GFC window that opens 2008-09-01.

**Do not date `stress_2025_26` and do not widen the GFC window.** Choosing crisis boundaries after seeing where results fall is selection on the outcome, and it would contaminate the headline. Instead:

- Headline crisis tables use the **pre-registered windows exactly as the codebase defines them.**
- `stress_2025_26` stays undated and is **excluded from the headline**, labelled as an unset window rather than an absence of stress.
- Report a **robustness check** with a wider GFC definition (e.g. opening 2007-07-01) alongside, clearly marked as sensitivity, never as the headline.
- State the window definitions and their source in the output, so a reader can see they were fixed in advance.

### 3. One measurement that could shrink a paper-level caveat — and it needs no model

At BTC-USD's fifteen `garch11_t` fallback origins, **the nearest converged fit has α+β = 1.000000000 at 14 of 15** — nine zeros, so the persistence constraint is binding on the *converged* fits too, not only the failures. If those boundary fits are already numerically near-EWMA (a GARCH pinned at α+β = 1 with small ω essentially is), then falling back to EWMA on 11% of that cell changes far less than the rate suggests, and D-033's heaviest caveat softens.

**Both series are already in the store**, so this is a comparison of stored forecasts, not a model run:

- Compare BTC-USD `garch11_t` forecasts against BTC-USD `ewma` forecasts at the same origins. Report the distribution of the ratio — median, IQR, and the tails — split by whether the governing fit was boundary-pinned.
- Report **how many of BTC's 133 `garch11_t` fits have α+β within 1e-6 of 1**, not just the 15 fallbacks.
- Do the same for BTC `garch11`, which had zero fallbacks. If it is also boundary-pinned, the phenomenon is the asset, not the specification.

Report the numbers. Do not draw the conclusion.

### 4. Verify two things before you build on J2's tables

The J2 amendments asked for both, and neither is visible in its chat report — they may be in the files. Confirm, and compute them yourself if they are missing:

- **Every QLIKE figure reported twice, with and without the five near-zero target rows** that contribute up to 1.01% of a cell's QLIKE sum from five observations.
- **AutoARIMA's non-convergence rate (2,334 / 2,366 = 98.6%) carried in the loss tables**, the way the GARCH fallback rates are.

Also note: the authoritative grid manifest is **`data/grid_primary/manifest_fix.json`**, not the stale `docs/P3_GRID_manifest.json`. It is under `data/` and therefore uncommitted — use it, and say in your report that you did, because a clean checkout currently cannot tell which fragment set is current.

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

## 1. Diebold–Mariano

Per asset, per loss: the 13×13 matrix of DM statistics and p-values on the loss differential `d_t = L_i,t − L_j,t`.

- **Harvey–Leybourne–Newbold small-sample correction** applied, compared against `t_{n−1}` rather than the standard normal. State the correction factor used.
- HAC variance for `d_t` with a stated automatic bandwidth rule; **report the chosen bandwidth per pair**, or its distribution. Do not assume the h=1 loss differential is serially uncorrelated — volatility loss differentials are persistent in practice, and a bandwidth of zero here is a silent error, not a simplification.
- Pairwise-complete samples from J2, with the n printed **inside the matrix**.
- **Report p-values uncorrected and label them uncorrected.** Also report, per asset × loss, how many of the 78 pairs are significant at 5% against the 3.9 expected by chance. MCS is the multiple-comparison-correct instrument; the DM matrix is descriptive. Do not Bonferroni the matrix and do not present it as if it controlled anything.
- Note which pairs are **not independent by construction**: where the `ν ≤ 50` bound binds, `garch11_t` collapses toward `garch11`. Report the per-asset share of fits where the bound binds, so closeness between those two can be read correctly.

Output: `docs/P3_DM.md` plus machine-readable matrices.

---

## 2. Model Confidence Set

Per asset, per loss: MCS (Hansen, Lunde & Nason 2011) over the 13 models.

- Moving-block bootstrap, **B = 10,000**, block length from the Politis–White automatic rule computed on the loss series. **Report the chosen block length per asset × loss**, and flag any that is 1 or exceeds n/4 — both indicate the rule has failed rather than answered.
- Report both the range statistic and the semi-quadratic statistic, and the surviving set at **α = 0.10 and α = 0.25**.
- Report the **elimination order and each model's MCS p-value**, not just the surviving set. The ordering carries more information than the boundary.
- **Sensitivity runs, required, reported alongside the headline and never instead of it:**
  - BTC-USD MCS **with `garch11_t` removed** — that cell is ~11% EWMA fallback by construction.
  - HSI MCS **with `gjr` removed** — 6.1% fallback.
  Report whether the surviving set changes. Do not drop these from the headline.

Output: `docs/P3_MCS.md`.

---

## 3. VaR / ES backtests

Per asset × model, at the evaluated levels: Kupiec unconditional coverage, Christoffersen independence and conditional coverage, observed vs expected exceedance counts, and mean FZ0 joint loss. Same "uncorrected across 13 models × 11 assets" label on the p-values. Report exceedance clustering visibly enough to be checked — at minimum the longest run of consecutive exceedances.

Output: `docs/P3_BACKTESTS.md`.

---

## 4. Economic value

Using the existing volatility-targeting backtest: per asset × model, annualized return, volatility, Sharpe, max drawdown and turnover, at transaction costs of **0, 1, 5 and 10 bps**, with annualization stated per asset class (252 equities / 365 crypto).

Add what a point estimate cannot give: a **bootstrap confidence interval on the Sharpe difference** between each model and the `garch11` baseline, using the same moving-block scheme and a pinned seed. A Sharpe table without uncertainty invites exactly one referee comment.

Output: `docs/P3_ECON.md`.

---

## 5. Cross-asset aggregation — ranks only

`docs/P3_CROSS_ASSET.md`. Targets differ by asset class, so cross-asset summaries are **rank-based only**:

- Mean rank per model across the 11 assets, per loss, with the rank distribution — not just the mean.
- MCS membership counts: for each model, in how many of the 11 assets it survives at α = 0.10 and α = 0.25.
- **Kendall's τ between the CRPS ranking and the QLIKE ranking, per asset.** Report all 11 values and the minimum. This one is load-bearing for a framing decision on the planning machine — report it even if it looks boring.
- Equity block and crypto block reported separately as well as together.

**No pooled loss, no averaged loss, no "overall best" line anywhere in this file.**

---

## 6. Determinism

**Every bootstrap seed pinned and written into the output manifest.** Re-running the analysis must reproduce every number bit-for-bit; add a test that runs one small MCS twice and asserts identical output. Write `docs/P3_ANALYSIS_manifest.json` with seeds, bootstrap B, block lengths, bandwidths, config hashes, data digests, package versions, thread pins and git SHA.

## 7. Report

In chat: what ran and what failed, wall-clock per section, any mechanical anomaly, the block-length and bandwidth summaries, the MCS surviving sets, the Kendall's τ list, and the file list with row counts. Do not paste full 13×13 matrices. **No interpretation, no rankings, no conclusions** — the results review happens on the planning machine.
