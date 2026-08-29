# P3 — convergence forensics: the 38 EWMA fallbacks

**What this is.** Every one of the primary grid's 38 fallback fits, described
from the optimizer's own terminal state rather than from its exit flag. Eight
tables (T1–T8), and the five questions J2 asks of them.

**Reported, not interpreted.** No model is ranked and no fallback is read as
evidence about a forecast. Where this document says "at a bound" it means a
measured distance below a stated tolerance, and the tolerance's ladder is
shown so the claim can be checked rather than believed.

| | |
|---|---|
| Grid | `data/grid_primary/manifest_fix.json` — the post-fix manifest, 143 cells, 645,151 rows |
| GARCH family | 33 cells, **7,101** scheduled fits, **38** fallbacks, all `flag=8` |
| Driver | `src/volbench/benchmarks/convergence_forensics.py`; tests `tests/test_convergence_forensics.py` |
| Per-fit output | `docs/P3_CONVERGENCE_FITS.parquet` — 7,101 rows, one per scheduled fit |
| Interpreter | CPython 3.11.5, `.venv` (docs/P3_ANALYSIS_ASSUMPTIONS.md §8) |
| Environment | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` |
| Read first | `docs/P3_ANALYSIS_ASSUMPTIONS.md` (J1) — §5 for the per-fit view of `fit_status`, §7 for the crisis windows |

`manifest_fix.json` is the current manifest: `docs/P3_MODEL_DEFECT_FIXES.md` §3
re-ran 44 cells after the LightGBM and TSFM fixes, and
`docs/P3_GRID_manifest.json` describes the grid before them. **It makes no
difference to this document**: `garch11`, `garch11_t` and `gjr` are among the
99 configs-by-asset whose hashes did not move, so their fragments are
byte-identical between the two manifests and the 38 are the same 38 either way.

---

## 0. The retention this needed, and the check that makes it usable

**The failed-fit parameters were not retained.** `FittedGARCH` set
`result=None` on a fit that did not converge, so a fallback's parameter vector,
its final log-likelihood and its optimizer message were discarded at the moment
they became interesting. `fit_status` kept the exit flag and nothing else. The
question "which parameter sat at which bound when SLSQP stopped?" was therefore
not answerable from any artifact this project holds — only guessable.

So it was added rather than guessed at. `models/garch.py` now carries
`TerminalFit`: the parameter vector, the log-likelihood, the exit flag and
message, SLSQP's iteration and function-evaluation counts, `arch`'s rescale
factor, and **the box bounds the optimizer was run inside**, retained for every
scheduled fit, converged or not. It is not in `spec()`, so it is in no config
hash; it is read by no code path that produces a number; `update` carries it
forward unchanged, exactly as it does `detail`, because it describes the
*scheduled* fit. `tests/test_models_garch.py::TestTerminalFit` pins all of
that, `fit_status`'s exact spelling included; that it is byte-identical over
the real grid is §0's 7,101-of-7,101 check below.

**Then the grid's GARCH-family fits were re-run to read it back** — all 7,101,
at the grid's own scheduled origins read out of the store's `fit_origin`
column, through the committed copy of the grid driver
(`benchmarks.grid_primary`), so the windows are the runner's own and not a
reconstruction of them. Nothing was written to the store; no fragment was
touched.

The re-fit is only worth reading if it is the same experiment, so that is
checked rather than asserted: **all 7,101 re-fitted `fit_status` strings equal
the stored ones**, fit for fit, including all 38 `fallback=ewma|flag=8`.
`refit_all` raises instead of returning if a single one differs. The 38 are the
same 38, at the same origins, with the same exit flag.

Two things the store already fixes and this does not move: the 38 attribute to
the 7 cells the manifest names, and the per-cell counts of T8 reproduce the
manifest exactly.

---

## 1. One row per fallback fit

All 38 stopped at SLSQP `status = 8`, message **"Positive directional
derivative for linesearch"** — the same message on every one, with no other
exit flag anywhere in the 38. Iteration counts run 18–43 and function
evaluations 89–341: none of the 38 stopped at its starting value.

`arch` is called with `rescale=True`, so ω is estimated on a series multiplied
by `scale` and is in units of `scale²`; α, β, γ and ν are scale-free. Both
columns are given. `predict` divides the forecast variance by `scale²`, which
is what keeps the daily-units rule (CLAUDE.md rule 2); the J1 validity report
independently confirmed the stored quantiles reproduce from the stored variance
to 1.7e-16, so the scale is undone correctly on the way out.

### T1 — the 38 fallback fits

| asset | config | origin | date | ω (rescaled) | ω (return scale) | α | γ | β | ν | log-lik | flag | nit | nfev | scale |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | `garch11_t` | 499 | 2018-12-30 | 2.327e-09 | 2.327e-11 | 0.05976 | — | 0.9402404 | 3.994 | -268.0794 | 8 | 24 | 154 | 10 |
| BTC-USD | `garch11_t` | 520 | 2019-01-20 | 2.304e-09 | 2.304e-11 | 0.060101 | — | 0.9398991 | 3.8033 | -261.4170 | 8 | 26 | 160 | 10 |
| BTC-USD | `garch11_t` | 541 | 2019-02-10 | 2.107e-09 | 2.107e-11 | 0.070738 | — | 0.9292624 | 3.7959 | -232.8954 | 8 | 24 | 130 | 10 |
| BTC-USD | `garch11_t` | 562 | 2019-03-03 | 2.079e-09 | 2.079e-11 | 0.06641 | — | 0.9335902 | 3.5852 | -223.2424 | 8 | 30 | 217 | 10 |
| BTC-USD | `garch11_t` | 583 | 2019-03-24 | 2.71e-09 | 2.71e-11 | 0.070833 | — | 0.9291736 | 3.3778 | -196.5662 | 8 | 26 | 163 | 10 |
| BTC-USD | `garch11_t` | 604 | 2019-04-14 | 2.092e-09 | 2.092e-11 | 0.059543 | — | 0.940457 | 3.1856 | -181.1228 | 8 | 27 | 161 | 10 |
| BTC-USD | `garch11_t` | 625 | 2019-05-05 | 1.753e-09 | 1.753e-11 | 0.058965 | — | 0.9410352 | 3.1404 | -152.8995 | 8 | 26 | 177 | 10 |
| BTC-USD | `garch11_t` | 646 | 2019-05-26 | 1.635e-09 | 1.635e-11 | 0.062042 | — | 0.9379588 | 3.1814 | -145.9271 | 8 | 21 | 102 | 10 |
| BTC-USD | `garch11_t` | 667 | 2019-06-16 | 1.423e-09 | 1.423e-11 | 0.063177 | — | 0.9368229 | 3.1887 | -129.6172 | 8 | 25 | 145 | 10 |
| BTC-USD | `garch11_t` | 688 | 2019-07-07 | 0.0003343 | 3.343e-06 | 0.075741 | — | 0.9242594 | 3.2422 | -129.6443 | 8 | 32 | 242 | 10 |
| BTC-USD | `garch11_t` | 730 | 2019-08-18 | 0.0005554 | 5.554e-06 | 0.080764 | — | 0.9192362 | 3.1265 | -123.2314 | 8 | 32 | 168 | 10 |
| BTC-USD | `garch11_t` | 772 | 2019-09-29 | 0.0009859 | 9.859e-06 | 0.089525 | — | 0.9104747 | 2.9752 | -105.6133 | 8 | 40 | 312 | 10 |
| BTC-USD | `garch11_t` | 940 | 2020-03-15 | 0.01034 | 0.0001034 | 0.14272 | — | 0.8572825 | 2.5859 | -142.0250 | 8 | 43 | 341 | 10 |
| BTC-USD | `garch11_t` | 2494 | 2024-06-16 | 0.06672 | 6.672e-06 | 0.051589 | — | 0.948411 | 2.921 | -1112.0180 | 8 | 36 | 216 | 100 |
| BTC-USD | `garch11_t` | 2578 | 2024-09-08 | 0.06366 | 6.366e-06 | 0.04306 | — | 0.956941 | 3.0652 | -1115.2177 | 8 | 32 | 193 | 100 |
| DIA | `garch11` | 856 | 2008-07-23 | 0.002306 | 2.306e-07 | 0.0085629 | — | 0.9914372 | — | -616.9308 | 8 | 23 | 107 | 100 |
| DIA | `garch11` | 898 | 2008-09-22 | 0.003497 | 3.497e-07 | 2.2092e-06 | — | 1 | — | -670.0194 | 8 | 22 | 113 | 100 |
| DIA | `garch11` | 5056 | 2025-04-03 | 0.0008915 | 8.915e-08 | 0.0093773 | — | 0.9906227 | — | -548.2048 | 8 | 25 | 155 | 100 |
| DIA | `garch11_t` | 856 | 2008-07-23 | 0.001747 | 1.747e-07 | 0.030382 | — | 0.9696185 | 5.7943 | -596.3324 | 8 | 27 | 230 | 100 |
| DIA | `garch11_t` | 877 | 2008-08-21 | 0.002072 | 2.072e-07 | 0.024132 | — | 0.975868 | 6.239 | -618.9586 | 8 | 27 | 225 | 100 |
| DIA | `garch11_t` | 898 | 2008-09-22 | 0.004342 | 4.342e-07 | 0.050043 | — | 0.9499567 | 5.9965 | -654.3315 | 8 | 19 | 89 | 100 |
| DIA | `gjr` | 877 | 2008-08-21 | 0.002583 | 2.583e-07 | 7.2472e-15 | 0.0116 | 0.9941984 | — | -632.2016 | 8 | 37 | 265 | 100 |
| ETH-USD | `gjr` | 2662 | 2024-12-01 | 0.0578 | 5.78e-06 | 4.9914e-16 | -4.497e-08 | 0.9961547 | — | -1237.1883 | 8 | 24 | 189 | 100 |
| HSI | `gjr` | 3460 | 2019-01-16 | 0.002735 | 2.735e-07 | 1.8142e-12 | 0.007069 | 0.9964653 | — | -692.8522 | 8 | 25 | 207 | 100 |
| HSI | `gjr` | 5014 | 2025-05-16 | 0.0328 | 3.28e-06 | 0.062182 | -0.06218 | 0.9593583 | — | -944.5162 | 8 | 23 | 151 | 100 |
| HSI | `gjr` | 5035 | 2025-06-16 | 0.02726 | 2.726e-06 | 0.06728 | -0.06729 | 0.957448 | — | -941.9741 | 8 | 23 | 148 | 100 |
| HSI | `gjr` | 5077 | 2025-08-14 | 0.02587 | 2.587e-06 | 0.077366 | -0.07737 | 0.9530611 | — | -926.0703 | 8 | 22 | 136 | 100 |
| HSI | `gjr` | 5098 | 2025-09-12 | 0.01158 | 1.158e-06 | 0.081603 | -0.0816 | 0.9560584 | — | -921.4921 | 8 | 23 | 156 | 100 |
| HSI | `gjr` | 5119 | 2025-10-15 | 0.009915 | 9.915e-07 | 0.08084 | -0.08084 | 0.9568254 | — | -917.3720 | 8 | 19 | 106 | 100 |
| HSI | `gjr` | 5140 | 2025-11-14 | 0.005776 | 5.776e-07 | 0.077265 | -0.07726 | 0.9593099 | — | -916.9114 | 8 | 33 | 289 | 100 |
| HSI | `gjr` | 5161 | 2025-12-15 | 0.008407 | 8.407e-07 | 0.07943 | -0.07943 | 0.957589 | — | -909.0745 | 8 | 31 | 263 | 100 |
| HSI | `gjr` | 5182 | 2026-01-16 | 0.002773 | 2.773e-07 | 0.076961 | -0.07696 | 0.9599647 | — | -903.3274 | 8 | 26 | 182 | 100 |
| HSI | `gjr` | 5203 | 2026-02-16 | 0.001576 | 1.576e-07 | 0.081095 | -0.0811 | 0.9588306 | — | -893.1153 | 8 | 22 | 145 | 100 |
| HSI | `gjr` | 5245 | 2026-04-23 | 0.007641 | 7.641e-07 | 0.084946 | -0.08495 | 0.956449 | — | -896.8431 | 8 | 29 | 225 | 100 |
| HSI | `gjr` | 5266 | 2026-05-26 | 0.006212 | 6.212e-07 | 0.086569 | -0.08657 | 0.9564733 | — | -888.9198 | 8 | 20 | 136 | 100 |
| HSI | `gjr` | 5287 | 2026-06-25 | 0.006847 | 6.847e-07 | 0.086423 | -0.08642 | 0.9565043 | — | -889.8615 | 8 | 23 | 150 | 100 |
| HSI | `gjr` | 5308 | 2026-07-27 | 0.008448 | 8.448e-07 | 0.084143 | -0.08414 | 0.9566755 | — | -892.7589 | 8 | 30 | 225 | 100 |
| SPY | `garch11` | 898 | 2008-09-22 | 0.004197 | 4.197e-07 | 1.1573e-09 | — | 1 | — | -711.7796 | 8 | 18 | 119 | 100 |


---

## 2. The boundary table

Two different objects are involved and this document keeps them apart, because
a fit can sit on one while the other is slack:

- a **box bound**, handed to SLSQP per parameter — `arch`'s own
  `ω ∈ [1e-8·v̄, v̄]` (with `v̄` the mean squared rescaled residual),
  `α ∈ [0,1]`, `β ∈ [0,1]`, `γ ∈ (−1,2)`, and this project's
  `ν ∈ [2.1, 50]` (D-032, `NU_BOUNDS`);
- a **linear inequality constraint**, handed to SLSQP as a constraint —
  `ω ≥ 0`, `α ≥ 0`, `α+γ ≥ 0` (GJR only), `β ≥ 0`, and stationarity
  `α + γ/2 + β ≤ 1`.

Both are recorded per fit: the box bounds travel on `TerminalFit` as the
optimizer received them, and the constraints are arithmetic on the retained
parameters (`constraint_slack`). Slack is signed, and a small **negative**
value means the optimizer stopped a hair outside — which SLSQP permits within
its own constraint tolerance, and which happens here at the 1e-9…1e-6 scale.

T2 lists, per fit, every bound and constraint active at 1e-5. T3 shows how the
counts move across five decades of tolerance.

### The three questions, answered

**Is α+β at or near 1 — the IGARCH boundary?** For the **22** GARCH(1,1)
fallbacks (`garch11` 4, `garch11_t` 18), α+β *is* the stationarity quantity,
and **all 22 sit on it**: `|1 − (α+β)| ≤ 6.24e-6`, 20 of the 22 within 1e-6,
16 within 1e-7. All 22 stopped marginally *past* the boundary — α+β > 1 in
every one of the 22, by between 1.16e-9 and 6.24e-6.

For the **16** `gjr` fallbacks the stationarity quantity is `α + γ/2 + β`, and
α+β is not it: α+β there runs 0.9942 → 1.0430 while `α+γ/2+β` runs 0.990449 →
1.000001, within 9.55e-3 of 1 for all 16 and within 5.4e-7 for 2 of them
(DIA 877, HSI 3460).

**Is ν at either end of [2.1, 50]?** **No — not one of the 18 `garch11_t`
fallbacks, at any tolerance.** Fitted ν runs 2.586 → 6.239. The closest
approach to the lower end is 0.486 (BTC-USD origin 940, ν = 2.586) and to the
upper end 43.76. Both are four to ten orders of magnitude outside every
tolerance on the ladder, so no tolerance choice can move this answer.

**Is `gjr`'s γ at a bound?** **Not at a box bound** — nearest approach 0.913 to
the lower and 1.988 to the upper, again untouchable by any tolerance. But 14 of
the 16 `gjr` fallbacks sit on the **constraint** `α + γ = 0`: γ is negative and
equal to −α, so the coefficient on a negative shock is driven to zero.
`|α+γ| ≤ 5.06e-6` on all 14 and ≤ 5e-8 on 13 of them. The remaining two
(DIA 877, HSI 3460) sit on `α = 0` instead — α at 7.2e-15 and 1.8e-12 against a
box bound of exactly 0, with γ > 0 — and both of those also sit on
stationarity. One of the 14 (ETH-USD 2662) sits on `α = 0` as well as
`α + γ = 0`, both parameters being zero to 5e-16 and 4.5e-8.

### Tolerance sensitivity

The two negative answers — ν at a bound, γ at a box bound — read **0 at every
tolerance from 1e-12 to 1e-4** and are not sensitive to the choice at all. The
positive answers are sensitive, in the expected direction and only in it:
looser tolerances find more, monotonically (pinned by
`tests/test_convergence_forensics.py`). The count of fits with **at least one**
active bound or constraint is 9 at 1e-12, 12 at 1e-10, 22 at 1e-8, 37 at 1e-6,
**38 at 1e-5**, and 38 at 1e-4. The one fit that needs 1e-5 rather than 1e-6 is
HSI `gjr` at origin 5035, whose `α+γ` is −5.06e-6.

### T2 — active bounds and constraints, per fit (tolerance 1e-5)

| asset | config | origin | active at 1e-5 | 1 − (α+γ/2+β) | α+γ | α |
|---|---|---:|---|---:|---:|---:|
| BTC-USD | `garch11_t` | 499 | `omega` at lower box bound; **α+γ/2+β = 1** | -2.79e-08 | +5.98e-02 | 5.98e-02 |
| BTC-USD | `garch11_t` | 520 | `omega` at lower box bound; **α+γ/2+β = 1** | -2.71e-08 | +6.01e-02 | 6.01e-02 |
| BTC-USD | `garch11_t` | 541 | `omega` at lower box bound; **α+γ/2+β = 1** | -4.70e-09 | +7.07e-02 | 7.07e-02 |
| BTC-USD | `garch11_t` | 562 | `omega` at lower box bound; **α+γ/2+β = 1** | -1.42e-09 | +6.64e-02 | 6.64e-02 |
| BTC-USD | `garch11_t` | 583 | `omega` at lower box bound; **α+γ/2+β = 1** | -6.24e-06 | +7.08e-02 | 7.08e-02 |
| BTC-USD | `garch11_t` | 604 | `omega` at lower box bound; **α+γ/2+β = 1** | -1.07e-07 | +5.95e-02 | 5.95e-02 |
| BTC-USD | `garch11_t` | 625 | `omega` at lower box bound; **α+γ/2+β = 1** | -7.66e-09 | +5.90e-02 | 5.90e-02 |
| BTC-USD | `garch11_t` | 646 | `omega` at lower box bound; **α+γ/2+β = 1** | -5.19e-07 | +6.20e-02 | 6.20e-02 |
| BTC-USD | `garch11_t` | 667 | `omega` at lower box bound; **α+γ/2+β = 1** | -5.15e-09 | +6.32e-02 | 6.32e-02 |
| BTC-USD | `garch11_t` | 688 | **α+γ/2+β = 1** | -2.24e-07 | +7.57e-02 | 7.57e-02 |
| BTC-USD | `garch11_t` | 730 | **α+γ/2+β = 1** | -1.26e-09 | +8.08e-02 | 8.08e-02 |
| BTC-USD | `garch11_t` | 772 | **α+γ/2+β = 1** | -6.85e-09 | +8.95e-02 | 8.95e-02 |
| BTC-USD | `garch11_t` | 940 | **α+γ/2+β = 1** | -3.34e-08 | +1.43e-01 | 1.43e-01 |
| BTC-USD | `garch11_t` | 2494 | **α+γ/2+β = 1** | -7.94e-09 | +5.16e-02 | 5.16e-02 |
| BTC-USD | `garch11_t` | 2578 | **α+γ/2+β = 1** | -7.22e-07 | +4.31e-02 | 4.31e-02 |
| DIA | `garch11` | 856 | **α+γ/2+β = 1** | -3.54e-08 | +8.56e-03 | 8.56e-03 |
| DIA | `garch11` | 898 | `alpha[1]` at lower box bound; `beta[1]` at upper box bound; **α+γ/2+β = 1**; **α = 0** | -2.21e-06 | +2.21e-06 | 2.21e-06 |
| DIA | `garch11` | 5056 | **α+γ/2+β = 1** | -1.42e-08 | +9.38e-03 | 9.38e-03 |
| DIA | `garch11_t` | 856 | **α+γ/2+β = 1** | -5.19e-08 | +3.04e-02 | 3.04e-02 |
| DIA | `garch11_t` | 877 | **α+γ/2+β = 1** | -1.17e-08 | +2.41e-02 | 2.41e-02 |
| DIA | `garch11_t` | 898 | **α+γ/2+β = 1** | -2.70e-08 | +5.00e-02 | 5.00e-02 |
| DIA | `gjr` | 877 | `alpha[1]` at lower box bound; **α+γ/2+β = 1**; **α = 0** | -5.37e-07 | +1.16e-02 | 7.25e-15 |
| ETH-USD | `gjr` | 2662 | `alpha[1]` at lower box bound; **α+γ = 0**; **α = 0** | +3.85e-03 | -4.50e-08 | 4.99e-16 |
| HSI | `gjr` | 3460 | `alpha[1]` at lower box bound; **α+γ/2+β = 1**; **α = 0** | -1.65e-08 | +7.07e-03 | 1.81e-12 |
| HSI | `gjr` | 5014 | **α+γ = 0** | +9.55e-03 | -2.89e-09 | 6.22e-02 |
| HSI | `gjr` | 5035 | **α+γ = 0** | +8.91e-03 | -5.06e-06 | 6.73e-02 |
| HSI | `gjr` | 5077 | **α+γ = 0** | +8.26e-03 | -1.31e-09 | 7.74e-02 |
| HSI | `gjr` | 5098 | **α+γ = 0** | +3.14e-03 | -2.00e-08 | 8.16e-02 |
| HSI | `gjr` | 5119 | **α+γ = 0** | +2.75e-03 | -1.33e-09 | 8.08e-02 |
| HSI | `gjr` | 5140 | **α+γ = 0** | +2.06e-03 | -3.81e-08 | 7.73e-02 |
| HSI | `gjr` | 5161 | **α+γ = 0** | +2.70e-03 | -8.37e-09 | 7.94e-02 |
| HSI | `gjr` | 5182 | **α+γ = 0** | +1.55e-03 | -4.38e-08 | 7.70e-02 |
| HSI | `gjr` | 5203 | **α+γ = 0** | +6.22e-04 | -2.35e-08 | 8.11e-02 |
| HSI | `gjr` | 5245 | **α+γ = 0** | +1.08e-03 | -1.83e-08 | 8.49e-02 |
| HSI | `gjr` | 5266 | **α+γ = 0** | +2.42e-04 | -4.83e-08 | 8.66e-02 |
| HSI | `gjr` | 5287 | **α+γ = 0** | +2.84e-04 | -1.15e-09 | 8.64e-02 |
| HSI | `gjr` | 5308 | **α+γ = 0** | +1.25e-03 | -2.53e-08 | 8.41e-02 |
| SPY | `garch11` | 898 | `alpha[1]` at lower box bound; `beta[1]` at upper box bound; **α+γ/2+β = 1**; **α = 0** | -1.16e-09 | +1.16e-09 | 1.16e-09 |

### T3 — the tolerance ladder

| tolerance | ω@low | α@low | β@high | γ@any box | ν@either end | α+γ/2+β = 1 | α+γ = 0 (gjr) | α = 0 | ≥1 active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-12 | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 9 |
| 1e-10 | 7 | 3 | 2 | 0 | 0 | 0 | 0 | 3 | 12 |
| 1e-08 | 9 | 4 | 2 | 0 | 0 | 8 | 5 | 4 | 22 |
| 1e-06 | 9 | 4 | 2 | 0 | 0 | 22 | 13 | 4 | 37 |
| 1e-04 | 9 | 5 | 2 | 0 | 0 | 24 | 14 | 5 | 38 |


---

## 3. The DIA question

DIA is the only asset failing on all three specifications: `garch11` 3/234,
`garch11_t` 3/234, `gjr` 1/234. The origins, side by side:

| config | fallback origins | dates |
|---|---|---|
| `garch11` | 856, 898, **5056** | 2008-07-23, 2008-09-22, 2025-04-03 |
| `garch11_t` | 856, **877**, 898 | 2008-07-23, 2008-08-21, 2008-09-22 |
| `gjr` | **877** | 2008-08-21 |

**The three-way intersection is empty.** No single origin fails under all
three specifications.

Every pairwise intersection, stated rather than summarised:

| pair | intersection | dates |
|---|---|---|
| `garch11` ∩ `garch11_t` | {856, 898} | 2008-07-23, 2008-09-22 |
| `garch11_t` ∩ `gjr` | {877} | 2008-08-21 |
| `garch11` ∩ `gjr` | ∅ | — |

The union is 4 origins: {856, 877, 898, 5056}. **Three of them — 856, 877, 898
— are consecutive scheduled refit origins**, exactly 21 apart, which is the
grid's refit cadence: they are DIA refits number 18, 19 and 20 of 234, and they
span 2008-07-23 → 2008-09-22, 61 calendar days. Six of DIA's seven fallbacks
fall inside that span. Each of the three origins has exactly two of the three
specifications failing on it, and each of the three specifications fails
somewhere inside it. The seventh fallback (`garch11` at 5056, 2025-04-03)
is 16.5 years later and is the only DIA fallback outside the 2008 span.

**Which way this points, under the rule the question sets.** Overlapping
origins across specifications point at the data; disjoint origins point at the
specifications. At the strict three-way level the origins are **disjoint**; at
the pairwise level and in calendar terms they **overlap heavily** — three
adjacent refits, one 61-day span, all three specifications represented, two of
three failing at each origin. The arithmetic supports the *overlapping*
reading: the failures are concentrated on shared windows rather than spread
across each specification's own origins. What separates a two-of-three from a
three-of-three at each origin is not measured here.

For completeness, the fit windows behind those origins are 500 trading days
each, beginning 2006-07-28 (origin 856), 2006-08-28 (877) and 2006-09-27
(898) — so the three windows share 458 of their 500 days.


---

## 4. The HSI question

HSI fails only on `gjr` — 14 of 230 — while `garch11` and `garch11_t` are
clean at 0/230, and `gjr` itself ran 234 fits with 0 fallbacks on its first
real-panel exposure (SPY pre-flight). T4 places the 14 against the 216 clean
`gjr` fits of the same cell; T5 lists every fit in the two runs the 14 fall in.

Where the 14 sit in γ / α+β space:

- **γ.** All but one of the 14 have **γ < 0** (median −0.0801, range −0.0866 →
  +0.0071). Among the 216 clean fits γ is negative in **14** of them (median
  +0.0847, range −0.1006 → +0.2546). So both the failing and the clean fits
  reach negative γ, and the failing ones are almost entirely inside that
  region.
- **α+β.** The 14 run 0.9965 → 1.0430, median 1.0373. The 216 clean fits run
  0.7589 → 1.0428, median 0.9271, 95th percentile 0.9915. Every one of the 14
  is above the clean 95th percentile.
- **α+γ/2+β** (the constraint that actually binds). The 14 run 0.9904 →
  1.000001, median 0.9982. The 216 clean fits run 0.8464 → 1.0000, median
  0.9742, with 27 of them above 0.99.
- **α+γ.** Thirteen of the 14 are at 0 to within 5.06e-6 — γ = −α. Among the
  216 clean fits, 11 also sit within 1e-4 of `α+γ = 0`, and the clean minimum
  is −6.94e-10.

The 14 are two runs, not one (T5):

- **Origin 3460 alone** (2019-01-16), between two converged fits at 3439 and
  3481. It is the one of the 14 with γ > 0; it stops with α at its lower box
  bound (1.8e-12) and stationarity at 1.000000.
- **Thirteen of the last fifteen refits**, origins 5014 → 5308
  (2025-05-16 → 2026-07-27). The two converged fits inside that run (5056 and
  5224) have the *same* structure as the thirteen that failed — α+γ at
  6.2e-9 and 4.3e-10, α+β at 1.0309 and 1.0428, stationarity at 0.9930 and
  1.000000 — so within this run a converged and a fallback fit are not
  separated by the parameter vector.
  Every fit from 4993 onward sees the **same** window maximum — the ratio
  column of T5 does not move across the run, so one observation is being
  carried by all seventeen windows — and window kurtosis runs 15.72 (origin
  5014) to 20.65 (origin 5266) across the run, rising but not monotonically,
  against a clean-fit median of 5.29. The maximum's own value is not reported;
  it is an order statistic, and the constancy is the whole of the point.

### T4 — HSI `gjr`: the 14 against the 216

| quantity | clean (n=216) min | 5% | median | 95% | max | fallback (n=14) min | median | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| γ | -0.1006 | -0.0065 | +0.0847 | +0.1892 | +0.2546 | -0.0866 | -0.0801 | +0.0071 |
| α+β | +0.7589 | +0.8368 | +0.9271 | +0.9915 | +1.0428 | +0.9965 | +1.0373 | +1.0430 |
| α+γ/2+β | +0.8464 | +0.8942 | +0.9742 | +0.9950 | +1.0000 | +0.9904 | +0.9982 | +1.0000 |
| α | +0.0000 | +0.0000 | +0.0000 | +0.0500 | +0.1095 | +0.0000 | +0.0801 | +0.0866 |
| β | +0.7589 | +0.8046 | +0.9242 | +0.9774 | +0.9958 | +0.9531 | +0.9571 | +0.9965 |
| α+γ | -0.0000 | +0.0016 | +0.0880 | +0.2209 | +0.2922 | -0.0000 | -0.0000 | +0.0071 |
| window kurtosis | +3.3167 | +3.7365 | +5.2869 | +8.1992 | +20.3871 | +4.9912 | +18.5939 | +20.6508 |
| window max \|r\| / clean median | +0.5039 | +0.5428 | +1.0000 | +2.3310 | +2.4340 | +0.9013 | +2.4340 | +2.4340 |

### T5 — HSI `gjr`, every fit in the two affected runs

| origin | date | outcome | α | γ | β | α+β | α+γ/2+β | α+γ | window kurtosis | window max \|r\| / clean median |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3418 | 2018-11-14 | converged | 0.028714 | +0.015873 | 0.958221 | 0.986935 | 0.994872 | +4.46e-02 | 5.30 | 0.901 |
| 3439 | 2018-12-13 | converged | 0.018350 | +0.005411 | 0.978945 | 0.997294 | 1.000000 | +2.38e-02 | 5.14 | 0.901 |
| 3460 | 2019-01-16 | **fallback** | 0.000000 | +0.007069 | 0.996465 | 0.996465 | 1.000000 | +7.07e-03 | 4.99 | 0.901 |
| 3481 | 2019-02-19 | converged | 0.006088 | +0.031888 | 0.967631 | 0.973719 | 0.989663 | +3.80e-02 | 4.92 | 0.901 |
| 3502 | 2019-03-20 | converged | 0.000000 | +0.075525 | 0.935973 | 0.935973 | 0.973736 | +7.55e-02 | 4.84 | 0.901 |
| 4951 | 2025-02-11 | converged | 0.106552 | -0.097675 | 0.817974 | 0.924526 | 0.875689 | +8.88e-03 | 6.68 | 1.695 |
| 4972 | 2025-03-12 | converged | 0.102681 | -0.100575 | 0.849178 | 0.951859 | 0.901572 | +2.11e-03 | 6.39 | 1.695 |
| 4993 | 2025-04-11 | converged | 0.061769 | -0.061769 | 0.959620 | 1.021389 | 0.990505 | -6.94e-10 | 16.10 | 2.434 |
| 5014 | 2025-05-16 | **fallback** | 0.062182 | -0.062182 | 0.959358 | 1.021540 | 0.990449 | -2.89e-09 | 15.72 | 2.434 |
| 5035 | 2025-06-16 | **fallback** | 0.067280 | -0.067285 | 0.957448 | 1.024728 | 0.991086 | -5.06e-06 | 16.14 | 2.434 |
| 5056 | 2025-07-16 | converged | 0.075673 | -0.075673 | 0.955192 | 1.030864 | 0.993028 | +6.18e-09 | 16.49 | 2.434 |
| 5077 | 2025-08-14 | **fallback** | 0.077366 | -0.077366 | 0.953061 | 1.030427 | 0.991744 | -1.31e-09 | 17.26 | 2.434 |
| 5098 | 2025-09-12 | **fallback** | 0.081603 | -0.081603 | 0.956058 | 1.037662 | 0.996860 | -2.00e-08 | 17.72 | 2.434 |
| 5119 | 2025-10-15 | **fallback** | 0.080840 | -0.080840 | 0.956825 | 1.037666 | 0.997246 | -1.33e-09 | 18.16 | 2.434 |
| 5140 | 2025-11-14 | **fallback** | 0.077265 | -0.077265 | 0.959310 | 1.036574 | 0.997942 | -3.81e-08 | 18.22 | 2.434 |
| 5161 | 2025-12-15 | **fallback** | 0.079430 | -0.079430 | 0.957589 | 1.037019 | 0.997304 | -8.37e-09 | 18.97 | 2.434 |
| 5182 | 2026-01-16 | **fallback** | 0.076961 | -0.076961 | 0.959965 | 1.036925 | 0.998445 | -4.38e-08 | 19.40 | 2.434 |
| 5203 | 2026-02-16 | **fallback** | 0.081095 | -0.081095 | 0.958831 | 1.039926 | 0.999378 | -2.35e-08 | 20.56 | 2.434 |
| 5224 | 2026-03-20 | converged | 0.085512 | -0.085512 | 0.957244 | 1.042756 | 1.000000 | +4.33e-10 | 20.39 | 2.434 |
| 5245 | 2026-04-23 | **fallback** | 0.084946 | -0.084946 | 0.956449 | 1.041395 | 0.998922 | -1.83e-08 | 20.03 | 2.434 |
| 5266 | 2026-05-26 | **fallback** | 0.086569 | -0.086569 | 0.956473 | 1.043042 | 0.999758 | -4.83e-08 | 20.65 | 2.434 |
| 5287 | 2026-06-25 | **fallback** | 0.086423 | -0.086423 | 0.956504 | 1.042927 | 0.999716 | -1.15e-09 | 20.44 | 2.434 |
| 5308 | 2026-07-27 | **fallback** | 0.084143 | -0.084143 | 0.956676 | 1.040819 | 0.998747 | -2.53e-08 | 20.33 | 2.434 |


---

## 5. The BTC / ETH question

BTC-USD `garch11_t` is 15/133; ETH-USD `garch11_t` is 0/133. The two series
were checked to be on **one calendar and one origin grid** before anything was
compared — 3,291 bars each, 2017-08-18 → 2026-08-21, identical dates at
identical positions, identical scheduled refit origins — and `paired_windows`
raises rather than returning if that fails. So "the same dates" means the same
dates and the same 500-day windows, positionally.

T6 gives, at each of BTC's 15 fallback origins: both series' fit-window sample
kurtosis (Pearson, bias-corrected; a normal reads 3), the **ratio** of the two
series' window maximum absolute returns (ETH over BTC — the maxima themselves
are order statistics and are not reported), and the fitted α+β of the nearest
converged fit in each series' own `garch11_t` cell.

The measurable differences, counted:

| measure | BTC greater | ETH greater |
|---|---:|---:|
| window kurtosis | 13 of 15 | 2 of 15 |
| window max \|r\| | 0 of 15 | 15 of 15 |
| window standard deviation | 0 of 15 | 15 of 15 |

- **Kurtosis.** BTC 5.50 → 39.77 (median 5.74); ETH at the same origins 4.92 →
  31.86 (median 5.10). The two exceptions are origins 2494 and 2578 (2024),
  where ETH reads 7.80 and 8.12 against BTC's 5.50 and 5.55.
- **Maximum absolute return.** ETH's window maximum is larger at every one of
  the 15, by a factor between **1.106×** (origin 541) and **1.801×** (origin
  2494). Reported as the ratio: each operand is a single realised return of a
  licensed-derived series, and the comparison is what the argument needs.
- **Window standard deviation.** ETH's is larger at all 15, by a factor
  between 1.101 (origin 2494) and 1.374 (origin 667). A window standard
  deviation is an aggregate over 500 observations and would be publishable as a
  level; it is given as a ratio here only to read alongside the row above.
- **α+β of the nearest converged fit.** For BTC, 12 of the 15 fallbacks have
  their nearest converged fit at origin 709 or 751 — the run 499 → 772 is 12
  fallbacks out of 14 consecutive refits — and that fit's α+β is 1.000000000.
  Fourteen of the 15 nearest-converged BTC fits have α+β = 1.000000000; the
  exception is origin 2473 at 0.720283893. Across the cell, 49 of BTC's 118
  converged `garch11_t` fits have α+β within 1e-6 of 1.
- For ETH the nearest converged fit is at **the same origin** in all 15 cases,
  because ETH converged everywhere. Its α+β at those origins runs 0.972383 →
  1.000000, and is within 1e-6 of 1 at 7 of the 15. Across the cell, 23 of
  ETH's 133 fits are within 1e-6 of 1.
- **ν.** BTC's 15 fallbacks fit ν 2.586 → 3.994; ETH's fits at the same origins
  read ν 2.499 → 3.242. Neither is near either end of [2.1, 50].

No explanation is proposed for any of the above.

### T6 — BTC-USD vs ETH-USD `garch11_t` at BTC's 15 fallback origins

| origin | date | BTC kurtosis | ETH kurtosis | ETH/BTC max \|r\| | BTC nearest converged (gap) | its α+β | ETH nearest converged (gap) | its α+β |
|---:|---|---:|---:|---:|---|---:|---|---:|
| 499 | 2018-12-30 | 5.650 | 5.108 | 1.162 | 709 (210) | 1.000000000 | 499 (0) | 0.995491699 |
| 520 | 2019-01-20 | 5.744 | 5.095 | 1.162 | 709 (189) | 1.000000000 | 520 (0) | 0.982951038 |
| 541 | 2019-02-10 | 5.661 | 4.943 | 1.106 | 709 (168) | 1.000000000 | 541 (0) | 0.997040428 |
| 562 | 2019-03-03 | 5.725 | 4.923 | 1.106 | 709 (147) | 1.000000000 | 562 (0) | 0.990968708 |
| 583 | 2019-03-24 | 5.980 | 4.967 | 1.106 | 709 (126) | 1.000000000 | 583 (0) | 1.000000000 |
| 604 | 2019-04-14 | 6.411 | 5.058 | 1.106 | 709 (105) | 1.000000000 | 604 (0) | 1.000000000 |
| 625 | 2019-05-05 | 6.422 | 5.124 | 1.106 | 709 (84) | 1.000000000 | 625 (0) | 1.000000000 |
| 646 | 2019-05-26 | 6.589 | 5.095 | 1.106 | 709 (63) | 1.000000000 | 646 (0) | 1.000000000 |
| 667 | 2019-06-16 | 5.730 | 4.972 | 1.345 | 709 (42) | 1.000000000 | 667 (0) | 0.977162994 |
| 688 | 2019-07-07 | 5.658 | 5.101 | 1.398 | 709 (21) | 1.000000000 | 688 (0) | 0.988932128 |
| 730 | 2019-08-18 | 5.928 | 5.307 | 1.398 | 709 (21) | 1.000000000 | 730 (0) | 0.972383184 |
| 772 | 2019-09-29 | 6.255 | 5.787 | 1.398 | 751 (21) | 1.000000000 | 772 (0) | 0.990379192 |
| 940 | 2020-03-15 | 39.768 | 31.857 | 1.175 | 919 (21) | 1.000000000 | 940 (0) | 1.000000000 |
| 2494 | 2024-06-16 | 5.497 | 7.797 | 1.801 | 2473 (21) | 0.720283893 | 2494 (0) | 1.000000000 |
| 2578 | 2024-09-08 | 5.554 | 8.125 | 1.566 | 2557 (21) | 1.000000000 | 2578 (0) | 1.000000000 |


---

## 6. Calendar placement

Tagged with the windows the codebase defines (`src/volbench/data/crisis.py`),
not the fallback windows — per J1 §7 those are different, and these are the
ones used. T7 gives all 38 with the origin date, the fit window's first day and
the tag.

| tag | window | fallback origins in it |
|---|---|---:|
| `gfc` | 2008-09-01 → 2009-03-31 | **3** |
| `covid` | 2020-02-01 → 2020-04-30 | **1** |
| `tightening_2022` | 2022-01-01 → 2022-10-31 | 0 |
| `spike_2024_08` | 2024-08-01 → 2024-08-31 | 0 |
| `calm` | everything else | **34** |

The four in a named window are DIA `garch11` and `garch11_t` and SPY `garch11`,
all at origin 898 (2008-09-22, `gfc`), and BTC-USD `garch11_t` at origin 940
(2020-03-15, `covid`).

**One caveat, mechanical rather than interpretive.** Thirteen of the 38 —
HSI `gjr`, origins 5014 → 5308 — fall between 2025-05-16 and 2026-07-27. That
is inside the span the codebase names `stress_2025_26` and **deliberately
leaves undated** (D-004 fixes its dates at grid freeze; `window_by_tag` raises
rather than guessing). Those 13 read `calm` because the window carrying that
period has no dates yet, not because the period was dated and found calm. A
reader taking "34 calm" at face value would be reading an unset window as a
measurement.

The origin date is the information cutoff; the fit window is the 500 trading
days ending on it. Nothing here tags windows — only origins — so a fallback at
a `calm` origin can still rest on a window containing a crisis. DIA at origin
898 illustrates the reverse: its origin is three weeks after `gfc` opens while
its window opens in 2006.

---

### T7 — the 38 dates against the codebase's crisis windows

| asset | config | origin | date | fit-window start | crisis tag |
|---|---|---:|---|---|---|
| BTC-USD | `garch11_t` | 499 | 2018-12-30 | 2017-08-18 | `calm` |
| BTC-USD | `garch11_t` | 520 | 2019-01-20 | 2017-09-08 | `calm` |
| BTC-USD | `garch11_t` | 541 | 2019-02-10 | 2017-09-29 | `calm` |
| BTC-USD | `garch11_t` | 562 | 2019-03-03 | 2017-10-20 | `calm` |
| BTC-USD | `garch11_t` | 583 | 2019-03-24 | 2017-11-10 | `calm` |
| BTC-USD | `garch11_t` | 604 | 2019-04-14 | 2017-12-01 | `calm` |
| BTC-USD | `garch11_t` | 625 | 2019-05-05 | 2017-12-22 | `calm` |
| BTC-USD | `garch11_t` | 646 | 2019-05-26 | 2018-01-12 | `calm` |
| BTC-USD | `garch11_t` | 667 | 2019-06-16 | 2018-02-02 | `calm` |
| BTC-USD | `garch11_t` | 688 | 2019-07-07 | 2018-02-23 | `calm` |
| BTC-USD | `garch11_t` | 730 | 2019-08-18 | 2018-04-06 | `calm` |
| BTC-USD | `garch11_t` | 772 | 2019-09-29 | 2018-05-18 | `calm` |
| BTC-USD | `garch11_t` | 940 | 2020-03-15 | 2018-11-02 | `covid` |
| BTC-USD | `garch11_t` | 2494 | 2024-06-16 | 2023-02-03 | `calm` |
| BTC-USD | `garch11_t` | 2578 | 2024-09-08 | 2023-04-28 | `calm` |
| DIA | `garch11` | 856 | 2008-07-23 | 2006-07-28 | `calm` |
| DIA | `garch11` | 898 | 2008-09-22 | 2006-09-27 | `gfc` |
| DIA | `garch11` | 5056 | 2025-04-03 | 2023-04-06 | `calm` |
| DIA | `garch11_t` | 856 | 2008-07-23 | 2006-07-28 | `calm` |
| DIA | `garch11_t` | 877 | 2008-08-21 | 2006-08-28 | `calm` |
| DIA | `garch11_t` | 898 | 2008-09-22 | 2006-09-27 | `gfc` |
| DIA | `gjr` | 877 | 2008-08-21 | 2006-08-28 | `calm` |
| ETH-USD | `gjr` | 2662 | 2024-12-01 | 2023-07-21 | `calm` |
| HSI | `gjr` | 3460 | 2019-01-16 | 2017-01-06 | `calm` |
| HSI | `gjr` | 5014 | 2025-05-16 | 2023-05-02 | `calm` |
| HSI | `gjr` | 5035 | 2025-06-16 | 2023-06-01 | `calm` |
| HSI | `gjr` | 5077 | 2025-08-14 | 2023-08-02 | `calm` |
| HSI | `gjr` | 5098 | 2025-09-12 | 2023-08-31 | `calm` |
| HSI | `gjr` | 5119 | 2025-10-15 | 2023-10-04 | `calm` |
| HSI | `gjr` | 5140 | 2025-11-14 | 2023-11-03 | `calm` |
| HSI | `gjr` | 5161 | 2025-12-15 | 2023-12-04 | `calm` |
| HSI | `gjr` | 5182 | 2026-01-16 | 2024-01-05 | `calm` |
| HSI | `gjr` | 5203 | 2026-02-16 | 2024-02-05 | `calm` |
| HSI | `gjr` | 5245 | 2026-04-23 | 2024-04-10 | `calm` |
| HSI | `gjr` | 5266 | 2026-05-26 | 2024-05-10 | `calm` |
| HSI | `gjr` | 5287 | 2026-06-25 | 2024-06-12 | `calm` |
| HSI | `gjr` | 5308 | 2026-07-27 | 2024-07-12 | `calm` |
| SPY | `garch11` | 898 | 2008-09-22 | 2006-09-27 | `gfc` |

---

## 7. Per-cell fallback rate, all 33 GARCH-family cells

T8. Totals: **38 / 7,101 = 0.535 %**. Twenty-six of the 33 cells are clean;
**two are above 5 %**: BTC-USD `garch11_t` at 15/133 = 11.278 % and HSI `gjr`
at 14/230 = 6.087 %. The remaining five non-clean cells are DIA `garch11`
(1.282 %), DIA `garch11_t` (1.282 %), ETH-USD `gjr` (0.752 %), DIA `gjr`
(0.427 %) and SPY `garch11` (0.427 %). This reproduces the manifest cell for
cell.

The other ten configs of the grid are absent from this table by design: they
implement no `fit_diagnostics`, so their rate is **not instrumented**, never 0
(docs/P3_INSTRUMENTATION_GAP.md). `analysis.fallback_rates` returns `<NA>` for
them rather than a number.

### T8 — fallback rate, all 33 GARCH-family cells

| asset | config | k/n | percent | above 5% |
|---|---|---:|---:|:--:|
| BTC-USD | `garch11` | 0/133 | 0.000% |  |
| BTC-USD | `garch11_t` | 15/133 | 11.278% | **yes** |
| BTC-USD | `gjr` | 0/133 | 0.000% |  |
| CAC | `garch11` | 0/240 | 0.000% |  |
| CAC | `garch11_t` | 0/240 | 0.000% |  |
| CAC | `gjr` | 0/240 | 0.000% |  |
| DAX | `garch11` | 0/238 | 0.000% |  |
| DAX | `garch11_t` | 0/238 | 0.000% |  |
| DAX | `gjr` | 0/238 | 0.000% |  |
| DIA | `garch11` | 3/234 | 1.282% |  |
| DIA | `garch11_t` | 3/234 | 1.282% |  |
| DIA | `gjr` | 1/234 | 0.427% |  |
| ETH-USD | `garch11` | 0/133 | 0.000% |  |
| ETH-USD | `garch11_t` | 0/133 | 0.000% |  |
| ETH-USD | `gjr` | 1/133 | 0.752% |  |
| HSI | `garch11` | 0/230 | 0.000% |  |
| HSI | `garch11_t` | 0/230 | 0.000% |  |
| HSI | `gjr` | 14/230 | 6.087% | **yes** |
| KOSPI | `garch11` | 0/231 | 0.000% |  |
| KOSPI | `garch11_t` | 0/231 | 0.000% |  |
| KOSPI | `gjr` | 0/231 | 0.000% |  |
| NDX | `garch11` | 0/236 | 0.000% |  |
| NDX | `garch11_t` | 0/236 | 0.000% |  |
| NDX | `gjr` | 0/236 | 0.000% |  |
| NKX | `garch11` | 0/229 | 0.000% |  |
| NKX | `garch11_t` | 0/229 | 0.000% |  |
| NKX | `gjr` | 0/229 | 0.000% |  |
| SPY | `garch11` | 1/234 | 0.427% |  |
| SPY | `garch11_t` | 0/234 | 0.000% |  |
| SPY | `gjr` | 0/234 | 0.000% |  |
| TWSE | `garch11` | 0/229 | 0.000% |  |
| TWSE | `garch11_t` | 0/229 | 0.000% |  |
| TWSE | `gjr` | 0/229 | 0.000% |  |
| **total** | | **38/7101** | **0.535%** | 2 of 33 |

---

## 8. Coding defect, or flat surface?

The three defect signatures the question names, each checked against the
retained record rather than argued:

**A bound set wrong.** The box bounds are retained per fit as SLSQP received
them, and across all 38 they are `arch`'s own and identical everywhere they
apply: `α ∈ [0,1]`, `β ∈ [0,1]`, `γ ∈ (−1,2)`, `ω ∈ [1e-9·v̄, v̄]`. The only
bound this project sets is `ν ∈ [2.1, 50]` (D-032), recorded as exactly that on
all 18 `garch11_t` fallbacks, and it is active on none of them — nearest
approach 0.486. That bound is not unreachable, which is what makes its absence
here informative: **61 of the 2,349 converged `garch11_t` fits in the grid do
sit at ν = 50** (at 1e-4; 57 at 1e-9 — the count is tolerance-dependent, and 61
reconciles with J1 §3's independent recovery from the stored quantiles). So the
ν bound binds in this grid, just never on a fit that fell back.

*One inconsistency found and measured, which reached no number.*
`_BoundedStudentsT` overrides `arch`'s `bounds()` to `[2.1, 50]` but inherits
its `constraints()`, which are `ν ≥ 2.05` and `ν ≤ 500`. SLSQP is therefore
handed a box and a strictly looser redundant linear constraint. The box is the
tighter of the two everywhere, SLSQP enforces box bounds directly, and the
constraint is slack at every one of the 2,367 `garch11_t` fits — the largest
fitted ν in the grid is 50.000000 against a constraint ceiling of 500. It is
reported here because it is a real disagreement between two descriptions of one
model, not because it moved anything.

**A scale mismatch.** `rescale=True` is deliberate and documented; the factor
is retained per fit (100 on 25 of the 38, 10 on the other 13), ω is recorded in
both the rescaled and the return-scale units, and `predict` divides the
forecast variance by `scale²`. Three independent checks say the units that
leave the model are return units: the stored VaR quantiles reproduce the
Gaussian quantiles of the stored `(mean, variance)` to 1.7e-16 over 595,524
rows and the stored CRPS and QLIKE reproduce from independent closed forms
(J1 §2, `docs/P3_ANALYSIS_VALIDITY.md` §4.2); `tests/test_models_update.py`
pins that re-conditioning at fixed parameters reproduces the fitted forecast to
the bit; and the new `TestTerminalFit` checks that the same returns presented
in different units give the same fit and the same return-scale ω.

**An obviously bad starting value.** None of the 38 stopped where it started:
SLSQP ran 18–43 iterations and 89–341 function evaluations on every one, and
all 38 terminal parameter vectors and all 38 terminal log-likelihoods are
distinct. `arch`'s starting values are its own, chosen by its internal grid
search on the rescaled window, and are interior wherever measured
(e.g. `(ω, α, β) = (0.00465, 0.05, 0.93)` for BTC-USD `garch11`, `(ω, α, γ, β)
= (0.0219, 0.01, 0.10, 0.92)` for DIA `gjr`). They are **not** retained per fit
— `arch` does not expose the composed starting vector on its result, and
reconstructing it would mean restating `arch`'s internals rather than calling
them — so the iteration counts, not a stored start, are the evidence here.

**What the surface looks like instead.** One exit flag on all 38 and no other
anywhere: `status = 8`, "positive directional derivative for linesearch", which
SLSQP returns when its line search cannot find a descent direction. Every one
of the 38 stops with at least one bound or constraint active at 1e-5: 24 on
stationarity `α+γ/2+β = 1` (the 22 GARCH(1,1) fits, where that quantity is
α+β, plus 2 `gjr`), 14 on `α+γ = 0`, 9 with ω at its lower box bound, 5 with α
at 0, and 2 with β at 1. And the same optimizer, the same bounds and the same
starting-value machinery converged on the other **7,063** fits of these 33
cells.

**None of the 38 shows a sign of a coding defect — no bound is set wrong, no
scale is mismatched, no starting value is obviously bad — and every one of them
carries the signature of a flat or ill-conditioned likelihood surface: a single
line-search exit flag and at least one active bound or constraint, on all 38.**
