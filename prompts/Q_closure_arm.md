# Prompt Q — The return-law closure: a Mixture distribution, the native-mixture arm, and the attribution test

**Branch:** after Prompt P's branch is merged, `git switch feat/p3-analysis && git pull`, then `git switch -c feat/closure-axis`.

**Session:** fresh — `claude -n vb-q-closure`. **Run `/effort max`** (session-only; set it every time this prompt is reopened). This prompt contains numerics where a subtle error produces plausible wrong numbers; that is what the effort is for.

---

## 0. Why this arm exists — read before coding

The primary grid's VaR backtests fail almost everywhere: mean observed/expected exceedances at α=0.01 is **2.42× for the twelve Gaussian-law configs against 1.505 for `garch11_t`** — the only fat-tailed config — and the gap **vanishes by α=0.05** (1.19 vs 1.18). `garch11_t` is the best-covering model on all eleven assets. If the volatility forecasts were simply too low, the miss would be proportional at every level; it is not. The working hypothesis: **the failure is the return-law closure, not the variance forecasts.**

The framework currently hardcodes that closure: every variance forecast becomes `Normal(0, √v̂)`. For the three zero-shot foundation models this discards their own distributional output — a 9-level quantile grid over next-day RV — whose measured cost is excess kurtosis 1.09–1.27 and VaR +10.0/+11.7% at α=0.01 (docs/P3_TSFM_VARIANCE_AUDIT.md).

This prompt makes the closure an **explicit, hashed configuration axis** and runs one arm: `native_mixture` for chronos, timesfm and moirai — `r | σ² ~ N(0, σ²)` with σ² drawn from the model's own grid. Combined with the existing `garch11` → `garch11_t` contrast (same dynamics, closure changed, coverage 2.10 → 1.51), this is a two-leg attribution experiment. **You compute; interpretation happens on the planning machine.**

Prediction on record, to be tested not assumed: the audit's +10–12% VaR widening at α=0.01 should reduce exceedances substantially. Report what happens.

---

## 1. `Mixture` in `dist.py` — route (b), closed forms, no sampling

A frozen dataclass like the others: `weights` (sum to 1, all ≥ 0, ≥ 2 components) and `sigmas` (all > 0), representing `Σₖ wₖ · N(0, σₖ²)`.

- `cdf(x) = Σₖ wₖ Φ(x/σₖ)` — closed form.
- `quantile(tau)`: deterministic bisection on the cdf. Bracket from the extreme components' quantiles. The tolerance is a fixed constant that **enters `spec()`/the hash** — it is a parameter that touches results (D-032's lesson).
- `crps(y)` closed form: `Σₖ wₖ E|Xₖ − y| − ½ ΣᵢΣⱼ wᵢwⱼ E|Xᵢ − Xⱼ|` with `E|X − y| = y(2Φ(y/σ) − 1) + 2σφ(y/σ)` for `X ~ N(0, σ²)` and `E|Xᵢ − Xⱼ| = √(2(σᵢ² + σⱼ²)/π)` for independent components. **Derive and verify these yourself rather than trusting this prompt**: the acceptance standard is D-014's — agreement with **two independent quadratures** to ~1e-12, plus the reduction test below.
- `log_score(y) = −log Σₖ wₖ φ(y; 0, σₖ)` — match the sign convention `dist.py` already uses; do not assume mine.
- `expected_shortfall(level)`: closed form given the quantile — `E[X·1{X≤q}] = −σφ(q/σ)` per component, so `ES = (1/α) Σₖ wₖ(−σₖφ(q/σₖ))`, with sign convention matched to the existing implementations, and cross-checked against the base class's generic quadrature.
- `mean() = 0`, `variance() = Σₖ wₖσₖ²` — closed form.
- `sample(n, seed)`: seeded component choice then normal draw; deterministic given seed.

**Required tests**, beyond the suite's conventions:

1. **Reduction:** a single-component `Mixture` must reproduce `Normal` exactly — every method, machine precision.
2. **Two-quadrature CRPS check** at multiple y values including deep tails, ~1e-12.
3. **Quantile/cdf round-trip** and monotonicity; ES closed form vs base quadrature.
4. **Order-statistic policy:** nothing here emits data values, but the new class must pass the existing guards untouched.

## 2. The closure as a hashed config field

TSFM configs gain `return_law: "gaussian" | "native_mixture"`, default `"gaussian"` (the headline — nothing already computed moves). It enters `spec()` and therefore the config hash. Tests both ways: the hash moves when the field changes, and does not move at the default.

The `native_mixture` path builds the mixing law from the per-origin RV grid using **exactly the same discretization and D-038 lognormal tail closure the mean reduction uses** — same code path, not a reimplementation. This is the load-bearing design constraint, because it yields the invariant that makes the experiment clean:

**Second-moment equivalence, enforced by a test:** for every origin, `Mixture.variance()` must equal the collapsed v̂ of the Gaussian twin to ~1e-12. Same variance forecast, same tail closure, same origins — the two configs then differ **only** in shape, and any coverage difference is attributable to the closure alone. If the discretization cannot deliver this exactly, stop and report why before proceeding.

Where a grid triggers the flat-tail fallback (chronos 119, timesfm 215, moirai 12 of 2,199 — K's counts), the mixture inherits the same fallback the mean does; count these, do not hide them.

`patchtst` is **excluded** — D-037: no grid, nothing to integrate; requesting `native_mixture` on it must raise with that reason.

**Persist the native grid** in the arm's fragments or sidecars (under `data/`, untracked) so future closure work is post-hoc rather than another GPU re-run — this closes the gap K found.

## 3. Run the arm, score it, backtest it

- 33 cells — chronos/timesfm/moirai × 11 assets — under `--tag closure_mixture`, own sibling manifest under `docs/`, headline untouched. Expected GPU cost ~41 min (K's measurement); confirm against your smoke run first.
- Verify the arm's origin sets are **identical** to the Gaussian twins' (same forecasts' calendar); report any difference as a stop.
- Score with the existing evaluation machinery. **Count and report the diff**: the closure should require zero changes to `evaluate.py`/`backtests.py` — a new return law is one `Distribution` subclass plus one config value. That count is a §4 number; report it precisely.
- Produce `docs/P3_CLOSURE_ARM.md` + CSVs: per asset × level, exceedance counts and observed/expected ratio for the mixture cell **beside its Gaussian twin**; Kupiec/Christoffersen/CC p-values; the realized VaR widening at each level against the audit's predicted +10.0/+11.7/+4.4/+5.9/−0.4/+0.5; CRPS/log-score/pinball for the mixture cells; per-origin excess kurtosis summary (aggregates only — O's column policy applies). **Report, do not interpret. No "fixes", no "confirms", no rankings.**

## 4. Re-certify and gate

- **Leakage canary on the new path** — the mixture uses only the model's own forecast grid at each origin, but the evidence follows the code (L's lesson): run the canary on the three mixture configs, both directions, determinism leg included.
- Two full runs of the arm bit-identical; resumability of the **primary** grid still 143 cached / 0 computed, store untouched (SHA-256, size, mtime).
- Full gate on all three interpreters; identity-leakage, licensing, manifest-counterpart and column-policy guards all green. Push the branch (CI covers `feat/**`); do not merge.

## 5. Report

The §1 verification results (quadrature agreement, reduction test); the second-moment-equivalence result; hash-move tests; the diff count from §3; the per-asset coverage table beside the Gaussian twins; predicted-vs-realized VaR widening; canary and determinism results; wall-clock; anything that surprised you, stated as an observation.
