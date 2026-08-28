# P3 — Defect 2: LightGBM's out-of-fold smearing

**Scope.** Measured, not fixed. Nothing here changes a model, a config hash or a
fragment. §5 records the resumability check on the primary store. No
recommendation to fix or disclose is made.

---

## Conclusion up front

- **The toy figures do not transfer, and they overstate the problem.** The
  fixture measured an in-sample log-residual variance of 0.015 against a
  realized 0.42 — a 28× understatement. On 21 years of index data the shipped
  configuration lands at a median **0.61 in-sample against 0.90 realized**: an
  understatement of **1.5×**, not 28×. The deliberately small ensemble is doing
  what `docs/P2_INTEGRATION.md` §3.2 said it would.

- **The bias that survives is still material and is one-directional.** The
  shipped smearing factor is **1.37** at the panel median where the realized
  one-step forecast errors imply **1.68**: the correction the theory asks for is
  **21.8 % larger** than the one applied, so every `lgbm` variance forecast in
  the store sits **17.9 % below** it — on every one of the 11 assets. (§2)

- **It is strongly regime-dependent, and worst exactly where it matters.** In
  the crisis sub-samples the implied factor is **2.09–3.32** against **1.53–1.69**
  in calm windows on the nine equity series — a 1.35× to 2.07× widening. In the
  COVID window alone the implied factor is **5.57**. The crisis sub-samples are a
  headline result and this defect interacts with them directly. (§3)

- **An out-of-fold factor would essentially close it.** Expanding temporal folds
  inside the training window give a median factor of **1.70** against the
  realized **1.68** — within 1 % of the target, on data the model is allowed to
  see. (§2)

- **The round cap is binding, and the two stories are one.** Training MSE falls
  monotonically from 0.78 at 25 rounds to 0.24 at 800, and the smearing factor
  falls with it from 1.49 to 1.13. `lgbm` running 100/100 rounds at every origin
  is not evidence that the budget was sufficient; it is the reason the residuals
  are as small as they are, and more rounds would keep shrinking them. (§4 —
  the addendum's amendment)

- **Cost of a fix: 11 CPU cells, ~0.77 min of cell time.** The hash does not
  move for the change itself, but does move for the release that carries it.
  (§5)

---

## 0. Provenance

`src/volbench/benchmarks/lgbm_smearing_probe.py`, at **every one of the grid's
2,366 scheduled refit origins** (the count matches
`docs/P3_INSTRUMENTATION_GAP.md` §3 exactly), all 11 assets. Zero probe errors.
Windows come from the same `AssetData.fit_series(policy)` the runner hands a
cell, so D-018 compaction applies identically; origins are read out of each
fragment's own `refit` / `fit_origin` columns rather than re-derived.

```
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 uv run --extra classical python -m \
  volbench.benchmarks.lgbm_smearing_probe --ladder-every 20
```

### 0.1 Three residual scales, and why all three are needed

| name | definition | implementable as a factor? |
|---|---|---|
| **in-sample** | `y − f(x)` on the 478 rows the booster trained on. What ships. | yes — it is what ships |
| **out-of-fold** | expanding **chronological** folds inside the window: the 478 rows are cut into 5 contiguous blocks and block *k* is predicted by a booster trained on blocks `0..k−1` only. 383 residuals per fit. Never a random split — a random fold of a time series lets tomorrow's neighbours predict today (`docs/design.md`). | **yes** — this is the candidate fix |
| **realized** | `log RV_{t+1} − mu_hat_t` at the grid's own 49,484 scored origins: the genuine one-step forecast error. | **no** — it reads the future. It is the quantity the factor is *trying to estimate*, so it is the target the other two are judged against. |

`mu_hat_t` is not re-run. It is recovered exactly from the stored fragment as
`log(forecast_var_t) − log(smear)`, with `smear` the factor of the refit block
that row rests on (`fit_origin`). `update` re-conditions without re-estimating
(`lgbm.py:269`), so the factor is constant within a block by construction and
the inversion is exact.

The out-of-fold arm trains on 20–80 % of the window rather than 100 %, so it is
a mildly **pessimistic** estimate of what the fix would deliver — it should
overshoot the realized figure slightly, and §2 shows it does.

---

## 1. In-sample versus out-of-fold residual scale, per asset

Log-space residual variance. Median over each asset's refit origins for the two
in-window scales; the variance of the realized one-step errors for the third.

| asset | fits | in-sample | out-of-fold | realized | OOF / in | realized / in |
|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 133 | 0.385 | 0.796 | 0.509 | 2.07 | 1.32 |
| CAC | 240 | 0.597 | 0.964 | 0.913 | 1.62 | 1.53 |
| DAX | 238 | 0.580 | 0.976 | 0.887 | 1.68 | 1.53 |
| DIA | 234 | 0.587 | 1.017 | 0.897 | 1.73 | 1.53 |
| ETH-USD | 133 | 0.343 | 0.650 | 0.468 | 1.90 | 1.36 |
| HSI | 230 | 0.637 | 0.986 | 0.899 | 1.55 | 1.41 |
| KOSPI | 231 | 0.621 | 0.934 | 0.904 | 1.50 | 1.45 |
| NDX | 236 | 0.633 | 1.065 | 0.938 | 1.68 | 1.48 |
| NKX | 228 | 0.711 | 1.060 | 0.983 | 1.49 | 1.38 |
| SPY | 234 | 0.615 | 1.100 | 0.952 | 1.79 | 1.55 |
| TWSE | 229 | 0.811 | 1.241 | 1.194 | 1.53 | 1.47 |
| **panel median** | | **0.615** | **0.986** | **0.904** | **1.68** | **1.47** |

**The toy number was not evidence about this panel, and it was pessimistic.**
The fixture's 0.015-against-0.42 (28×) came from LightGBM's *stock* shape — 300
rounds, 31/15 leaves, `min_data_in_leaf=20`. What ships is 100 rounds, 4 leaves,
`min_data_in_leaf=60`, `lambda_l2=5`, and on real data it understates the
residual scale by **1.3× to 1.6×**, uniformly.

The out-of-fold scale sits **above** the realized one on all 11 assets
(0.99 against 0.90 at the median), which is the expected direction: those folds
train on less data. It is a bound, not a bias.

---

## 2. The resulting bias in `lgbm`'s variance forecasts

Duan's factor is `mean(exp(e))`, so it is not a function of the residual
variance alone; it is reported directly.

| asset | shipped (in-sample) | out-of-fold | realized | OOF / shipped | **realized / shipped** |
|---|---:|---:|---:|---:|---:|
| BTC-USD | 1.300 | 1.347 | 1.391 | 1.037 | **1.070** |
| CAC | 1.362 | 1.719 | 1.678 | 1.262 | **1.232** |
| DAX | 1.322 | 1.703 | 1.686 | 1.288 | **1.275** |
| DIA | 1.367 | 1.846 | 1.724 | 1.351 | **1.261** |
| ETH-USD | 1.246 | 1.389 | 1.328 | 1.115 | **1.066** |
| HSI | 1.376 | 1.724 | 1.620 | 1.254 | **1.178** |
| KOSPI | 1.372 | 1.675 | 1.671 | 1.220 | **1.218** |
| NDX | 1.371 | 1.703 | 1.681 | 1.242 | **1.227** |
| NKX | 1.386 | 1.644 | 1.598 | 1.185 | **1.153** |
| SPY | 1.390 | 1.923 | 1.777 | 1.384 | **1.279** |
| TWSE | 1.466 | 1.824 | 1.765 | 1.245 | **1.204** |
| **panel median** | **1.371** | **1.703** | **1.678** | **1.245** | **1.218** |

**Read the last column as the size of the defect.** `lgbm`'s stored
`forecast_var` is `exp(mu_hat) · shipped_factor`; the retransformation the
theory asks for would multiply by the realized factor instead. That correction
is **6.6 % (ETH-USD) to 27.9 % (SPY) larger**, median **21.8 %** — equivalently,
the stored variance sits 6.2 % to 21.8 % below it, median 17.9 %. The sign is
the same on all eleven assets.

**The out-of-fold factor lands on the target.** Panel median 1.703 against a
realized 1.678 — a 1.5 % overshoot, in the direction §1 predicted. Per asset it
overshoots on 9 of 11 and undershoots on 2. Whatever else is true, the known fix
is not approximately right; it is right to within its own sampling noise.

**Crypto is the exception, and for a legible reason.** BTC-USD and ETH-USD show
the smallest gap (1.07). Their in-sample residual scale is also the smallest
(0.34–0.39) and their refit counts the lowest (133 against ~230), because their
panel is shorter. A smaller factor on a shorter series is not the same defect
behaving differently; it is less of it.

### 2.1 Propagation, for the same reason as Defect 1

`lgbm` emits `Normal(0, sqrt(v))` (verified row-by-row,
`docs/P3_METRIC_TARGETS.md` §1.2), so VaR and ES are homogeneous of degree 1 in
`sqrt(v)` and a factor ratio `ρ` moves both by exactly `sqrt(ρ)` at every level:
**+3.3 % (ETH-USD) to +13.1 % (SPY), median +10.4 %**, identical across
α = 0.01 / 0.025 / 0.05.

---

## 3. Is the bias regime-dependent?

**Yes, strongly, and it is worst in the windows that carry a headline result.**
Rows are tagged by their **target** date through `volbench.data.crisis.tag_dates`
— the four settled windows of `CRISIS_WINDOWS`; `stress_2025_26` remains
undated per D-004 and is therefore in `calm` here, which any reader of a crisis
table needs to know.

### 3.1 Per asset

| asset | n calm | n crisis | factor, calm | factor, crisis | **crisis / calm** |
|---|---:|---:|---:|---:|---:|
| CAC | 4,563 | 447 | 1.532 | **3.172** | **2.07** |
| SPY | 4,465 | 439 | 1.625 | **3.321** | **2.04** |
| DIA | 4,465 | 439 | 1.583 | **3.152** | **1.99** |
| DAX | 4,551 | 445 | 1.559 | **2.986** | **1.92** |
| KOSPI | 4,406 | 431 | 1.579 | **2.609** | **1.65** |
| NDX | 4,503 | 439 | 1.596 | **2.557** | **1.60** |
| HSI | 4,385 | 430 | 1.541 | **2.431** | **1.58** |
| TWSE | 4,309 | 412 | 1.690 | **2.549** | **1.51** |
| NKX | 4,350 | 423 | 1.550 | **2.088** | **1.35** |
| BTC-USD | 2,366 | 425 | 1.391 | 1.394 | 1.00 |
| ETH-USD | 2,366 | 425 | 1.329 | 1.318 | 0.99 |

### 3.2 Per window, pooled

| window | rows | realized residual variance | implied factor |
|---|---:|---:|---:|
| `calm` | 44,729 | 0.871 | 1.560 |
| `covid` (Feb–Apr 2020) | 734 | **1.789** | **5.568** |
| `gfc` (Sep 2008–Mar 2009) | 1,303 | 1.330 | **2.686** |
| `spike_2024_08` | 258 | 1.205 | **2.396** |
| `tightening_2022` (Jan–Oct 2022) | 2,460 | 0.693 | 1.529 |

Against a **shipped factor of ~1.37**, the COVID window's realized retransformation
is **5.57** — a 4.1× understatement in exactly the sub-sample where a
volatility model is being asked its hardest question. GFC is 2.0× and the
August-2024 spike 1.75×.

`tightening_2022` is the counter-example and is worth naming: its realized
residual variance (0.693) is *below* the calm average and its factor (1.53) is
indistinguishable from calm. A slow tightening cycle is a high-level, low-shock
regime, and the smearing defect keys on shock, not on level. "Crisis" is not one
thing in this table.

The two crypto series show **no** regime effect (ratios 1.00 and 0.99), and the
composition of their crisis sample is why: of 425 tagged rows each, **304 (72 %)
are `tightening_2022`** — the one window that shows no effect on any asset —
against 90 `covid` and 31 `spike_2024_08`. Neither carries a GFC (both panels
start in 2017). Their crisis aggregate is therefore dominated by the mild
window, and it should not be read as "crypto is immune".

---

## 4. Is the 100-round cap binding? (the addendum's amendment)

J1 measured that `lgbm` builds 100 of 100 rounds at all 2,366 refit origins.
That is consistent with two very different readings, and the ladder separates
them: the same design matrix, retrained at 25 / 50 / 100 / 200 / 400 / 800
rounds, at 122 origins spread across all 11 assets.

| rounds | trees built | training MSE (median) | smearing factor (median) |
|---:|---:|---:|---:|
| 25 | 25 | 0.7777 | 1.4861 |
| 50 | 50 | 0.6933 | 1.4225 |
| **100 (shipped)** | **100** | **0.6123** | **1.3660** |
| 200 | 200 | 0.5148 | 1.3011 |
| 400 | 400 | 0.3829 | 1.2182 |
| 800 | 800 | 0.2423 | 1.1293 |

**The cap is binding on the training loss.** MSE falls monotonically and by
another 60 % between 100 and 800 rounds; there is no plateau anywhere on the
ladder. So "every fit built every round" says nothing about sufficiency — the
budget stops the boosting, the boosting does not stop itself.

**And the two stories are one story.** The factor tracks the training loss down
the whole ladder: every additional unit of capacity shrinks the in-sample
residuals and moves the retransformation closer to 1, which is the mechanism
`docs/P2_INTEGRATION.md` §3.2 describes. At 800 rounds the factor is 1.13
against a realized 1.68 — the retransformation would be **two-thirds gone**.
Conversely, at 25 rounds it is 1.486, closer to the realized 1.678 than the
shipped 1.366 is.

**So the smearing measurement in §2 must not be read as a property of the
defect alone.** It is a joint property of the defect and of the capacity choice
made to bound it, and the 100-round cap is the load-bearing part of that bound.
`tests/test_models_lgbm.py::TestRetransformation::
test_the_ensemble_does_not_memorize_its_own_residuals` is the guard that keeps
it there, and this ladder is what it is guarding against.

The corollary the ladder does *not* settle: whether more rounds would improve
the *forecast*. Training MSE is not validation MSE, and this probe measures only
the former, because only the former is what the smearing factor reads. A
capacity increase that improved the point forecast and destroyed the
retransformation would show up here as an improvement.

---

## 5. Cost of a fix

**Which cells.** The 11 `lgbm` cells. `patchtst` carries the same construction
(`docs/P3_TSFM_VARIANCE_AUDIT.md` §1.4) and is **not** counted here; it was not
measured, and a fix scoped to `lgbm` leaves it in place.

**How long.** Confirmed against `docs/P3_GRID_manifest.json`'s own
`wall_clock_s` rather than taken on trust:

| asset | s | asset | s |
|---|---:|---|---:|
| BTC-USD | 3.494 | KOSPI | 3.950 |
| CAC | 4.726 | NDX | 4.319 |
| DAX | 4.258 | NKX | 4.765 |
| DIA | 3.829 | SPY | **0.006** (cache hit) |
| ETH-USD | 4.235 | TWSE | 4.211 |
| HSI | 4.120 | | |

Recorded total **0.699 min**. SPY's 0.006 s is a cache hit from an earlier
partial run, so a true 11-cell recompute is ~46 s — **0.77 min of cell time**,
CPU lane, 12 workers. The prompt's arithmetic is confirmed.

**Does it change the config hash?** Demonstrated on the stored artifact, SPY /
`lgbm`, hash `65876078…71b782f`:

| change | resulting hash | moved? |
|---|---|---|
| as stored (v0.6.0) | `65876078…71b782f` | — |
| a fix inside `fit()` that leaves `spec()` alone | `65876078…71b782f` | **no** |
| `spec()`'s `retransform` → `"smearing_oof"` | `3e2821e7…5cff938` | **yes** |
| `package_version` → `0.7.0` | `f634b86c…ad90806` | **yes** |

Two readings, and they differ from Defect 1's:

1. **The change itself need not move the hash.** The smearing factor is computed
   in `fit` (`lgbm.py:390`) and is not a field of `spec()`. Recomputing it from
   out-of-fold residuals is invisible to the hash — which is a *hazard*, not a
   convenience: the same `config_hash` would then name two numerically different
   fragments, and `ResultsStore.has()` short-circuits on file existence before
   any fitting (`docs/P3_INSTRUMENTATION_GAP.md` §4.2), so a re-run would report
   `cached 11, computed 0` and silently keep the old numbers.

2. **The release that carries it does move every hash.** `retransform` is
   already a documented, hashed option whose whole point is that "which one is
   used is a documented, config-hashed option (never a silent default)"
   (`models/_rv.py:22`). A new arm belongs in that vocabulary, which moves the
   11 `lgbm` hashes; and shipping a behaviour change to `volbench.models` is a
   release, which moves `package_version` and therefore all 143.

So the honest cost is not 0.77 minutes. It is 0.77 minutes of `lgbm` plus
whatever the study decides about the other 132 cells — the same
two-hash-spaces-or-re-run-everything question Defect 1 raises, and the reason
both are being measured before J2 rather than after.

---

## 6. Resumability of the primary store

Run after both audits, with all the new probe code in the tree and both probes
having been executed against it.

```
$ uv run --extra classical --extra tsfm python -m volbench.benchmarks.grid_primary \
      --tag resume_after_k --device cuda

cells attempted 143  computed 0  cached 143  failed 0
wall clock 0.3 min   peak RSS 1.01 GiB
```

**143 cached, 0 computed, 0 failed** — the required result. Checked further,
because "cached" only proves the short-circuit fired:

| check | before | after |
|---|---|---|
| files in `data/grid_primary/store` | 286 | 286 |
| `md5sum data/grid_primary/store/* \| md5sum` | `6532f1cd…f664d` | `6532f1cd…f664d` |
| files modified since the audits began | — | **0** |
| `git status --short data/` | empty | empty |

Every fragment and every sidecar is byte-identical and **unrewritten** (no mtime
moved). `data/` remains untracked, so
`tests/test_licensing_guard.py::TestNoDataIsTracked` is unaffected: both probes
write under `data/tsfm_dist_probe/` and `data/lgbm_smear_probe/`, and every
document produced by this work is under `docs/`.

The re-run's own manifest and report are at
`data/grid_primary/manifest_resume_after_k.json` and
`data/grid_primary/report_resume_after_k.txt`.

The `missing_reason` census it printed is unchanged from the primary run —
NKX's 21 `InsufficientHistoryError` rows on the seven variance-fed configs and
one `proxy_nonpositive`, TWSE's 80 `proxy_nan` on all 13 — matching
`docs/P3_ANALYSIS_VALIDITY.md` §2.

