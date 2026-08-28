# Prompt K — Measure two known model-side defects on the real panel

**Terminal:** the main integration checkout, on branch `feat/p3-analysis`, in a **fresh, separately named** Claude Code session (`claude -n vb-k-defects`). This prompt only *measures* — it writes documents and does not change model code — so it shares the analysis branch. If a fix is decided afterwards, that fix goes on its own branch, because it changes numbers.

Run this **after** the J1 gate and **before** J2: if a fix is needed, it must land before any loss table is computed, or those tables are built from forecasts you are about to replace.

**Model/effort:** any capable model. Run `/effort high` before starting.

---

## Why this runs before the analysis, not after

Two defects are known, quantified only on toy fixtures, and never measured on the real panel. Both sit inside a model's predictive distribution. If either is fixed later, the affected cells must be re-run and **every table computed from them recomputed** — so the expensive mistake is not the defect, it is analyzing twice.

This prompt **measures and reports. It does not fix.** Do not change any config hash, do not rewrite any fragment, do not re-run any grid cell. The decision to fix or disclose is made on the planning machine from your numbers.

---

## Defect 1 — TSFM variance derived from the quantile-grid mean

Recorded as an addendum to D-014: the TSFM adapters derive the predictive variance from the quantile grid's **mean**, which understated variance by roughly **8% at ν=5** on a toy fixture. It was never revisited.

This matters more than its size suggests. The paper's central empirical question is foundation models versus GARCH. If the TSFM variance is systematically low by construction, then a TSFM failing a VaR or ES backtest is a finding about our adapter, not about Chronos — and that is a fatal-in-review flaw if it is unfixed and undisclosed.

Report, in `docs/P3_TSFM_VARIANCE_AUDIT.md`:

1. **What the adapter actually does today**, per TSFM (`chronos`, `timesfm`, `moirai`, `patchtst`). What does the backend return — a quantile grid, samples, or parameters? How is the location derived, and how is the variance derived? Quote the code. Do not assume all four are the same; report each.
2. **Is the correct variance recoverable from the stored fragments post-hoc?** This single question decides everything downstream: if the full grid or the samples are stored, a fix costs nothing but recomputation; if only derived moments are stored, a fix costs a 68-minute GPU re-run of 44 cells. Answer it explicitly and show what the store actually holds.
3. **Measure the discrepancy on the real panel.** Per TSFM per asset, the distribution of `correct_variance / current_variance` — median, IQR, min, max. If it is not recoverable post-hoc, measure it on a bounded sample of re-run origins (one asset, a few hundred origins) rather than the whole grid, and say that is what you did.
4. **Translate it into what the paper reports.** The implied shift in VaR at the evaluated levels, per TSFM, as a percentage of the current level. A variance understatement propagates to VaR as roughly its square root — state the actual number rather than that approximation.
5. **Check the location parameter too.** Chronos's "mean" head is documented to be a median, and the adapter already works around that. Verify the workaround is in force on the real panel, and check whether the other three adapters have an analogous location issue that was never caught.
6. **Cost of the fix**: which cells change, how long they take, whether the config hash changes.

---

## Defect 2 — LightGBM out-of-fold smearing

The Duan smearing factor is computed from **in-sample** log residuals. LightGBM collapses those (measured 0.015 against a realized 0.42), which drives the smearing factor toward 1 and effectively removes the retransformation correction. Out-of-fold residuals are the known fix. Retuned defaults improved it to 0.28 against 0.38, with a capacity regression guard, but the underlying construction is unchanged.

Report, in `docs/P3_LGBM_SMEARING_AUDIT.md`:

1. **In-sample versus out-of-fold residual scale on the real panel**, per asset. The toy numbers are not evidence about 21 years of index data.
2. **The resulting bias in `lgbm`'s variance forecasts** — the distribution of the current smearing factor against what an out-of-fold factor would give, per asset.
3. **Whether the bias is regime-dependent.** If the gap is larger in high-volatility windows, the defect interacts with the crisis sub-samples, which are a headline result. Report the factor split by crisis window versus the rest.
4. **Cost of the fix**: 11 `lgbm` cells, CPU lane, ~0.7 minutes of cell time total — but confirm that rather than trusting my arithmetic, and confirm whether the fix changes the config hash.

---

## Output

One decision table at the end, and nothing else in chat beyond the numbers:

| defect | size on real panel | recoverable post-hoc? | re-run cost | changes config hash? |
|---|---|---|---|---|

**Do not recommend fix or disclose.** Report the measurements and the costs. The trade-off is a paper-framing decision, not a code decision, and it is made on the planning machine.

---

## Engineering rules

- Python 3.12+, typed, `ruff`-clean, tests for anything decidable.
- Outputs to `docs/`, never under `data/` — `tests/test_licensing_guard.py::TestNoDataIsTracked` requires `git ls-files -- data/` to be empty, and the guard tests location, not content.
- Export `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`.
- **The primary store at `data/grid_primary/store/` is read-only for this prompt.** If a measurement needs new forecasts, write them to a scratch store elsewhere under `data/`, never into the primary store. Verify at the end that a resumability re-run of the primary grid still reports 143 cached, 0 computed.
- `gh` on this box is 2.4.0 with neither `--branch` nor `--json`; status polls built on those fail silently and look exactly like "no problems". Use the API poll.
