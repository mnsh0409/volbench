# Prompt K — addendum from J1 (read this BEFORE `K_defects.md`, then do both)

J1 established something that was not previously written down anywhere, and it reframes the whole prompt.

> **12 of the 13 configs emit `Normal(0, sqrt(v))`** — stored quantiles match that form to 1.7e-16 on all 595,524 rows. Only `garch11_t` emits a Student-t. **That includes all four foundation models.**

Chronos, TimesFM and Moirai emit quantile grids natively. Something in the pipeline reduces that grid to a single variance and re-expresses it as a Gaussian. If so, the framework is scoring Gaussian approximations of the foundation models rather than the foundation models' own predictive distributions — and the distributional shape is exactly what makes them interesting under CRPS and VaR/ES.

**Do this as PART 0, before the two defects in `K_defects.md`.** The answers may make Defect 1 a sub-question rather than a separate issue.

---

## Part 0 — What is actually being predicted, and what is actually being scored

Write `docs/P3_METRIC_TARGETS.md`. Quote code for every answer; do not infer from naming.

1. **Per metric — CRPS, log score, pinball, QLIKE, FZ0 — what is the predictive object and what is the realization it is scored against?**
   Specifically: is CRPS computed against `realized_return` or against `proxy_var`? Is the predictive distribution over next-day **returns** or over next-day **variance**? The mean-zero `Normal(0, sqrt(v))` form points at returns; the QLIKE column points at variance; both columns exist in the store. State which metric uses which, per metric, with the line of code. **This is the single most important answer in the prompt** — it determines whether a volatility-proxy objection applies to the primary metric at all, and a framing decision on the planning machine is currently blocked on it.

2. **Where does each TSFM's native output get reduced to one variance?** Find and quote that step for `chronos`, `timesfm`, `moirai` and `patchtst` separately. Do not assume the four share a path.

3. **Is the pre-reduction quantile grid available anywhere on disk?** J1 found only three VaR/ES levels plus `v` persisted. Confirm whether the native grid survives anywhere, or whether recovering it requires re-running the GPU lane. Answer explicitly — the cost of everything downstream depends on it.

4. **What is lost.** For one asset and a bounded sample of origins (say 200), compare each TSFM's native quantile grid against the `Normal(0, sqrt(v))` that was actually scored. Report the maximum quantile discrepancy, the native grid's implied skew and excess kurtosis, and the VaR difference at 0.01 / 0.025 / 0.05.

5. Related, from J1's instrumentation pass: **`chronos` clipped a negative RV quantile at 14 of 241 sampled origins.** Report what the clip does, at which quantile levels it fires, and whether it fires before or after the reduction in (2).

**Do not fix anything and do not argue for a fix.** Report what the pipeline does and what the difference is. Say plainly if a question cannot be answered without re-running cells.

---

## One amendment to Defect 2 (LightGBM)

J1 measured that `lgbm` runs **100/100 boosting rounds at every one of the grid's refit origins** — early stopping never fires. Add: report whether the round cap is binding on training loss, i.e. whether more rounds would keep improving it. A capacity-capped model and a collapsed in-sample residual may be the same story told twice, and the smearing measurement should not be read without that.

---

## Decision table

Replace the table at the end of `K_defects.md` with:

| item | what the pipeline does | size of the effect | recoverable post-hoc? | re-run cost | changes config hash? |
|---|---|---|---|---|---|

One row for the distributional reduction (Part 0), one for the variance derivation (Defect 1), one for LightGBM smearing (Defect 2). Still no recommendation — report measurements and costs only.
