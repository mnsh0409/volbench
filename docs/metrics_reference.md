# 05 · Metrics reference (single source of truth for definitions & notation)

> Status: DRAFT v0.1 — written 18 Aug 2026. Verify every formula against the cited source before freezing (Phase 1). Items marked [TODO-VERIFY] have sign/convention traps. Once frozen, paper and code must match this file exactly.

## Notation
- `y_t` realized target at day t (realized variance RV_t, or return r_t for VaR/ES)
- `F̂_{t+h|t}` predictive CDF issued at t for t+h; `f̂` its density; `q̂_τ` its τ-quantile
- `α` tail level for risk measures (default 1%, 2.5%, 5%); losses are averaged over the evaluation set

## Volatility targets & proxies
- Realized variance: `RV_t = Σ_i r_{t,i}²` over intraday returns (5-min default). Sparse-sampling noise and microstructure caveats.
- If only daily data: squared return `r_t²` is an unbiased but very noisy proxy.
- **Patton (2011):** with noisy proxies, only certain loss functions rank models consistently — MSE and QLIKE are robust; many others are not. Use QLIKE as the point-forecast workhorse. [TODO-VERIFY exact class conditions]
- QLIKE (one common form): `L(σ̂², y) = y/σ̂² − ln(y/σ̂²) − 1`. [TODO-VERIFY orientation vs. the form used in `arch`/literature]

## Probabilistic scores (all negatively oriented: smaller = better)
- CRPS: `CRPS(F̂, y) = ∫ (F̂(z) − 1{z ≥ y})² dz`; sample form via E|X−y| − ½E|X−X′|. Closed forms exist for Gaussian/lognormal.
- Log score: `−ln f̂(y)`. Unbounded penalty for tail misses; report alongside CRPS, not instead.
- Quantile (pinball) loss at τ: `ρ_τ(y − q̂_τ)` with `ρ_τ(u) = u·(τ − 1{u<0})`.

## VaR / ES backtesting
- VaR_α at t: `q̂_α` of the return distribution (left tail). Hit: `H_t = 1{r_t < VaR_α,t}`.
- **Kupiec (1995) POF:** LR test of E[H] = α (unconditional coverage). Low power at α=1% with short samples — report sample sizes.
- **Christoffersen (1998):** independence + conditional coverage LR tests (hit clustering).
- **Fissler–Ziegel (2016):** (VaR, ES) jointly elicitable; use an FZ loss (e.g., FZ0) to *score and rank* ES forecasts, not just test them. [TODO-VERIFY FZ0 formula and sign convention before implementing — common source of bugs]
- Traffic-light style summaries are secondary; scores + tests are primary.

## Comparison inference
- **Diebold–Mariano (1995):** test on loss differentials `d_t = L(A) − L(B)`; HAC variance; Harvey–Leybourne–Newbold small-sample correction. Pairwise only — beware multiplicity across many pairs.
- **Model Confidence Set (Hansen, Lunde & Nason 2011):** sequential elimination yielding the set containing the best model(s) at confidence 1−α; default α = 0.10; bootstrap: block bootstrap, B = [10k]. Primary tool for "who wins" claims.

## Reporting rules
- Never rank on a single metric; show CRPS + a tail-focused score + QLIKE, with MCS membership.
- Report per-regime (crisis sub-samples) alongside full-sample.
- Every table states: n forecasts, refit schedule, and which proxy was the target.
