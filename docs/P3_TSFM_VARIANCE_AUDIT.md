# P3 — Defect 1: the TSFM variance derived from the quantile grid's mean

**Scope.** Measured, not fixed. Nothing here changes a model, a config hash or a
fragment. The primary store was opened read-only and the resumability check in
§7 confirms it is unchanged. No recommendation to fix or disclose is made.

**Which fragment set this was computed against — the *pre-fix* grid.** Every
number below reads the 143 cells of run digest `cb28a214`, store digest
`8f1f83db`, now archived at `docs/archive/P3_GRID_manifest.91ba622a8e50.json`.
That matters because the fix this document motivated has since landed: L's
lognormal tail closure moved the 33 TSFM cells, and
docs/P3_MANIFEST_INVENTORY.md then promoted the post-fix manifest into
`docs/P3_GRID_manifest.json`. `benchmarks.tsfm_distribution_probe` defaults
`--manifest` to that path and always has, so **its default changed meaning
without changing text**: re-running it today measures the post-fix cells, whose
closure is the one this document argued for, and comparing that output against
the numbers below would be comparing two different grids. To reproduce these,
pass `--manifest docs/archive/P3_GRID_manifest.91ba622a8e50.json`.

**Read `docs/P3_METRIC_TARGETS.md` first**, particularly §2: the quantile grid
is over **realized variance**, not returns, and `patchtst` has no grid at all.
Both change what "the defect" is.

---

## Conclusion up front

**The defect as recorded is misstated in one respect and larger than recorded in
another.**

- **Misstated.** The addendum to D-014 reads as though taking the *mean* of the
  grid were the error. It is not: the target functional is a conditional
  variance, QLIKE and MSE are both minimized at the conditional mean, and the
  mean is the right estimator. The error is in *how the mean is computed* —
  with **flat tails outside the outermost quantile level**, which on a 9-level
  0.1…0.9 grid places **20 % of the probability mass in two point atoms**. That
  is the D-014 truncation bias exactly, in the same family and for the same
  reason, and `docs/design.md` already names it as such.

- **Larger.** The "~8 % at ν = 5" figure is a *borrowed* number: it is the M1
  Student-t GARCH measurement, never measured on a TSFM. On the real panel the
  tail-closed mean is **11 % to 21 % above** the variance that was scored
  (equivalently, the scored variance sits 10 % to 17 % below it), and three
  independent tail closures — one of them assumption-free — agree on that band.
  (§3)

- **Not recoverable post-hoc**, but exactly re-derivable: no grid is on disk
  (`docs/P3_METRIC_TARGETS.md` §3), and re-running reproduces the stored
  variance to 2.2e-16 relative. A fix costs a **41-minute GPU re-run of 33
  cells**, not 68 minutes of 44 — `patchtst` is not affected by this defect.
  (§2, §6)

- **VaR consequence is exact, not approximate.** Because the scored object is
  `Normal(0, sqrt(v))`, `VaR_α = sqrt(v)·Φ⁻¹(α)` and the shift is **exactly**
  `sqrt(ratio)` at every level, identically across the three: **+5.4 %
  (`moirai`), +6.5 % (`chronos`), +9.6 % (`timesfm`)** at the median. (§4)

- **The location parameter is clean on all four.** Chronos's median-labelled-as-
  mean head is never bound by the adapter; TimesFM's point head *is* recorded
  and *is* discarded, and it sits 9.4 % **above** the scored `v̂` — pointing the
  same way as the tail-closure understatement and roughly the same size. (§5)

The three sizes measured here — the understatement of the level (§3), the shape
loss at fixed variance (`docs/P3_METRIC_TARGETS.md` §4.3), and the clip (§5.4
there) — are arithmetically separate and do **not** cancel.

---

## 1. What each adapter does today

Per config, from the code. The four are not the same and the differences matter.

### 1.1 `chronos` — `amazon/chronos-bolt-small`

| | |
|---|---|
| backend returns | a **quantile grid**, 9 levels 0.1…0.9, direct multi-step |
| how location is derived | it is not — the return law's location is fixed at `mu = 0.0` |
| how variance is derived | `quantile_grid_mean` over the repaired grid |
| native point head | **none recorded** (`native_mean=None`, verified on 2,199 origins) |

```python
# src/volbench/models/tsfm_chronos.py:118
with torch.inference_mode():
    quantiles, _median = self._pipeline.predict_quantiles(
        [ctx], prediction_length=h, quantile_levels=list(self._taus))
first = quantiles[0]
arr = first.detach().to(torch.float32).cpu().numpy().astype(np.float64)
return RVQuantileForecast(taus=self._taus, values=arr.reshape(-1, len(self._taus)))
```

The pipeline's second return value — the one it calls `mean` and its own source
documents as the 0.5 quantile — is bound to `_median` and never used. See §5.

### 1.2 `timesfm` — `google/timesfm-2.5-200m-pytorch`

| | |
|---|---|
| backend returns | `(h, 10)`: **index 0 the point head, 1..9 the quantiles** at 0.1…0.9 |
| how location is derived | `mu = 0.0`, as above |
| how variance is derived | `quantile_grid_mean` over `arr[:, 1:]` |
| native point head | **recorded as `native_mean`, not scored** |

```python
# src/volbench/models/tsfm_timesfm.py:169
arr = np.asarray(quantiles, dtype=np.float64)[0]  # (h, 10): point head, then q0.1..q0.9
...
return RVQuantileForecast(taus=self._taus, values=arr[:, 1:], native_mean=arr[:, 0])
```

Uniquely among the three, TimesFM applies `infer_is_positive` and
`fix_quantile_crossing` **inside the package**, before the adapter sees anything
(`tsfm_timesfm.py:75-77`).

### 1.3 `moirai` — `Salesforce/moirai-2.0-R-small`

| | |
|---|---|
| backend returns | a **quantile grid**, `(n_quantiles, h)`, transposed to `(h, 9)` |
| how location is derived | `mu = 0.0`, as above |
| how variance is derived | `quantile_grid_mean` over the repaired grid |
| native point head | **none** — Moirai 2.0 is quantile-loss trained and has no point head |

```python
# src/volbench/models/tsfm_moirai.py:123
out = self._forecaster.predict([ctx.astype(np.float32)])
arr = np.asarray(out, dtype=np.float64)[0]  # (n_quantiles, h)
return RVQuantileForecast(taus=self._taus, values=arr.T.copy())
```

### 1.4 `patchtst` — **not affected by this defect**

| | |
|---|---|
| backend returns | `max_horizon` **direct point outputs** under an MSE objective |
| how location is derived | `mu = 0.0`; the log-space point forecast is the *scale* input |
| how variance is derived | `exp(mu_hat) · smearing_factor` — **no grid anywhere** |
| native point head | the whole model is a point head |

```python
# src/volbench/models/patchtst.py:197
mu = float(self.log_forecast[h - 1])
factor = float(self.smearing[h - 1])
return Normal(mu=0.0, sigma=math.sqrt(variance_from_log(mu, factor)))
```

`patchtst` has no quantile grid, so it has no grid-mean truncation bias. It has
a **different** bias in the same place — the Duan smearing factor is computed
from in-sample training residuals, which is Defect 2's construction. Its
mitigation (early stopping on a chronological hold-out) is partial and is
already flagged in `docs/P2_INTEGRATION.md` §3.2. It is not measured here.

**Consequence for the fix's scope: 33 cells, not 44.**

---

## 2. Is the correct variance recoverable post-hoc?

**No, and the store is the reason.** The full answer, with the artifact checked
rather than assumed, is `docs/P3_METRIC_TARGETS.md` §3. In summary:

| what the store holds | where | grid? |
|---|---|---|
| 32 columns, identical on all 143 fragments: `forecast_var`, `forecast_mean`, `var_/es_/pinball_/hit_` at 3 levels | every `.parquet` | **no** |
| `quantile_levels: [0.1 … 0.9]` and `variance_from: "mean_of_rv_quantile_grid"` | the 33 TSFM `.json` sidecars | **levels only, no values** |
| the fitted `spec()`'s `rv_forecasts` block (taus, values, crossings, clips) | **nowhere** | — |

The cause is that `run_backtest` hashes and records the *unfitted* probe's
`spec()` (`evaluate.py:867`), so nothing a fitted object learns reaches disk.

**But re-derivation is exact.** The probe reproduced the stored `forecast_var`
to 2.2e-16 relative on 600 TSFM origins (`docs/P3_METRIC_TARGETS.md` §3.1). So
the choice is not "recompute or lose it": it is "spend GPU time", and the
measurement below cost 3.8 minutes of it.

**A second, deeper sense in which it is not recoverable.** Even with every grid
in hand, "the correct variance" is not identified. The grid *is* the
checkpoint's entire output; nothing in it determines the law beyond
q<sub>0.1</sub> and q<sub>0.9</sub>, which is where the missing mass lives. Any
number is a number under a stated tail assumption. §3 therefore reports three
closures and their spread rather than one figure.

---

## 3. The discrepancy on the real panel

**Method.** `src/volbench/benchmarks/tsfm_distribution_probe.py`, 200 origins
per asset evenly spaced across each cell's whole evaluation span, all 11 assets,
all 3 quantile configs: **6,597 origins** (6,600 minus NKX's three known
`InsufficientHistoryError` origins). The interior of each grid is left exactly
as the checkpoint emitted it; only the two flat closures are re-expressed.

**Three closures, deliberately.**

| closure | what it assumes | direction of its own error |
|---|---|---|
| `lognormal` | a lognormal fitted by OLS to the whole grid in log-z space; its partial expectations replace the two atoms | mild; also inherits the interior trapezoid's small upward bias |
| `loglinear` | the same shape fitted to the **outermost pair** of levels at each end, so the tail follows the grid's own edge spacing | heavier when the grid fans out; more local, noisier |
| `empirical` | **no distributional assumption**: the atoms are replaced by the realized `proxy_var`'s own conditional means over exceedances of q<sub>0.1</sub> / q<sub>0.9</sub> | valid only if the grid is calibrated; absorbs any upper-tail miscalibration, so it is an **upper** reading |

**The empirical closure is a diagnostic, not a candidate fix.** It pools realized
`proxy_var` over the whole evaluation sample, so it reads data after the origin
it would correct. Nothing it produces is fed to a model or to a forecast — it
exists only to bound the other two closures without assuming a shape — and it
could not be implemented at forecast time. The lognormal and log-linear closures
use nothing but the origin's own grid and are implementable; those are the two
in the per-asset table.

### 3.1 `correct_variance / current_variance`, per TSFM per asset

Lognormal closure. Median, IQR, min, max over the 200 origins of each cell.

| asset | `chronos` med (IQR) | `timesfm` med (IQR) | `moirai` med (IQR) |
|---|---|---|---|
| BTC-USD | 1.052 (1.031–1.100) | 1.064 (1.041–1.103) | 1.038 (1.022–1.073) |
| CAC | 1.129 (1.096–1.177) | 1.205 (1.138–1.309) | 1.104 (1.072–1.136) |
| DAX | 1.149 (1.102–1.217) | 1.218 (1.128–1.351) | 1.111 (1.088–1.155) |
| DIA | 1.146 (1.103–1.226) | 1.226 (1.144–1.352) | 1.107 (1.072–1.153) |
| ETH-USD | 1.053 (1.031–1.094) | 1.057 (1.038–1.100) | 1.034 (1.021–1.060) |
| HSI | 1.131 (1.101–1.188) | 1.224 (1.163–1.298) | 1.129 (1.103–1.160) |
| KOSPI | 1.138 (1.101–1.183) | 1.194 (1.153–1.308) | 1.124 (1.097–1.156) |
| NDX | 1.147 (1.116–1.209) | 1.216 (1.154–1.335) | 1.126 (1.091–1.161) |
| NKX | 1.185 (1.144–1.263) | 1.283 (1.181–1.430) | 1.163 (1.129–1.211) |
| SPY | 1.153 (1.107–1.220) | 1.234 (1.159–1.337) | 1.102 (1.070–1.138) |
| TWSE | 1.162 (1.109–1.220) | 1.283 (1.196–1.420) | 1.140 (1.106–1.190) |
| **panel** | **1.135** (1.091–1.199) | **1.201** (1.112–1.317) | **1.111** (1.071–1.156) |
| panel min / max | 0.974 / 22.09 | 0.983 / 9.82 | 0.982 / 5.78 |
| origins the closure fits | 2,080 / 2,199 | 1,984 / 2,199 | 2,187 / 2,199 |

The closure is undefined where the grid contains a zero — a clipped
`chronos`/`moirai` quantile or a package-floored `timesfm` one (see
`docs/P3_METRIC_TARGETS.md` §5) — since a lognormal cannot describe a zero
quantile. Those origins are excluded from the ratio rather than imputed; there
are 119, 215 and 12 of them respectively.

The heavy maxima (22.09 on NDX/`chronos`) are single origins where the grid is
nearly flat and the fitted `sigma` is unstable. The median and IQR are the
figures to read.

**Two clean regularities.** The two crypto series show a much smaller gap
(1.03–1.06) than the nine equity series (1.10–1.28), and the mechanism is
visible in the grids: their median coefficient of variation is the lowest in the
panel (ETH-USD 0.505, BTC-USD 0.527, against 0.597–0.678 on the nine equities,
`chronos`). A narrower grid strands less distance in the two atoms, so the flat
closure costs less. And `timesfm` is worst on every asset, `moirai` best on nine
of eleven.

### 3.2 The three closures agree on the band

Panel medians of `correct / current`:

| config | `lognormal` | `loglinear` | `empirical` (assumption-free) | `mean(realized RV / v̂)` |
|---|---:|---:|---:|---:|
| `chronos` | 1.135 | 1.102 | **1.212** | 1.259 |
| `timesfm` | 1.201 | 1.091 | **1.205** | 1.302 |
| `moirai` | 1.111 | 1.097 | **1.206** | 1.251 |

The last column is the crudest possible check and needs no closure at all: the
realized `proxy_var` averages 25–30 % above the variance that was scored. It is
an **upper** bound on this defect because it also contains any genuine forecast
bias, but it is the wrong side of 1.0 by an amount consistent with everything
else in the table.

**So: the correct variance is between roughly 9 % and 21 % above what was
scored, and the direction is downward on every closure, every config and every
asset.**

### 3.3 The grid's calibration, since the empirical closure rests on it

Median over the 11 assets, `n` = 2,193 scorable origins per config:

| config | P(RV < q<sub>0.1</sub>) | P(RV > q<sub>0.9</sub>) | E[RV \| RV > q<sub>0.9</sub>] / v̂ |
|---|---:|---:|---:|
| nominal | 0.100 | 0.100 | — |
| `chronos` | 0.090 | **0.130** | 4.34 |
| `timesfm` | 0.045 | **0.147** | 4.19 |
| `moirai` | 0.106 | **0.135** | 4.27 |

The upper tail is under-covered on all three — realizations land above
q<sub>0.9</sub> 30–47 % more often than nominal. This is why the empirical
closure reads highest of the three: part of that 1.21 is closure error and part
is the grid's own upper-tail miscalibration, and the two are not separable from
this data. `timesfm`'s lower-tail coverage (0.045 against 0.100) is the
package's positivity flooring showing up as a calibration statistic.

---

## 4. What it means for what the paper reports

**The propagation is exact, not "roughly the square root".** The scored object
is `Normal(0, sqrt(v))` on all three configs (verified row-by-row,
`docs/P3_METRIC_TARGETS.md` §1.2), so

    VaR_α = sqrt(v) · Φ⁻¹(α)     and     ES_α = −sqrt(v) · φ(Φ⁻¹(α)) / α

are both **homogeneous of degree 1 in `sqrt(v)`**. Multiplying the variance by
`ρ` multiplies every VaR and every ES by exactly `sqrt(ρ)`, at every level, with
no approximation and no level-dependence.

| config | variance ratio (median) | **VaR/ES shift, all three levels** | IQR of that shift |
|---|---:|---:|---|
| `chronos` | 1.135 | **+6.53 %** | +4.45 % to +9.51 % |
| `timesfm` | 1.201 | **+9.57 %** | +5.46 % to +14.76 % |
| `moirai` | 1.111 | **+5.42 %** | +3.50 % to +7.52 % |

Read against the empirical closure instead (1.205–1.212) the shift is +9.8 % to
+10.1 % on all three.

Every reported TSFM VaR and ES is therefore shallower than the tail-closed
variance would give, by 5–10 % of its own magnitude, uniformly across
α = 0.01 / 0.025 / 0.05. That is a **level** effect. It is separate from and
additive to the **shape** effect of §4.3 of `docs/P3_METRIC_TARGETS.md`, which
is +10–12 % at α = 0.01, +4–6 % at α = 0.025 and ~0 % at α = 0.05. At α = 0.01
the two compound to roughly +16 % to +23 %; at α = 0.05 only the level effect
survives.

QLIKE moves too, and not proportionally: `qlike(v, p) = p/v − ln(p/v) − 1` is
convex in `v` with its minimum at `v = p`, so raising an under-stated `v` toward
`p` lowers QLIKE. The direction is unambiguous given §3.3's finding that
realized RV averages 25–30 % above `v̂`; the magnitude per cell is not computed
here because it depends on the realized `proxy_var` row by row and is a J2
table, not a defect measurement.

---

## 5. The location parameter

### 5.1 On the scored axis there is no location to get wrong

All three TSFMs emit `Normal(mu=0.0, ...)` (`tsfm_common.py:384`). The return
law's location is fixed at zero by the package convention, not derived from the
model, so a location bug of the Chronos kind cannot reach the scored object
through the mean. It could only reach it through `v̂`.

### 5.2 Chronos's median-labelled-as-mean: the workaround is in force

The `chronos` adapter calls `predict_quantiles`, which returns a pair, and binds
the second element to `_median` — an underscore-prefixed name that is never read
(`tsfm_chronos.py:120`). Verified on the panel: `native_mean` is `None` on
**2,199 of 2,199** `chronos` origins, so nothing that could be the pipeline's
mislabelled "mean" enters any computation. The workaround is structural — there
is no configuration under which it can lapse — rather than conditional.

### 5.3 The other three, checked for an analogous issue

| config | native point head? | is it used? | how it compares to the scored `v̂` |
|---|---|---|---|
| `chronos` | no (`native_mean=None`, 2,199/2,199) | n/a | n/a |
| `moirai` | no (`native_mean=None`, 2,199/2,199) | n/a | n/a |
| `timesfm` | **yes**, on 2,199/2,199 origins | **no** | median **head / v̂ = 1.094** |
| `patchtst` | the model is a point head | yes, it is the forecast | see Defect 2's construction |

**TimesFM's own point head sits 9.4 % above the variance that was scored** — and
25 % above the grid's median (`head / q₀.₅ = 1.248`). Per asset the median
`head / v̂` runs from 1.074 (DAX) to 1.124 (TWSE); it is above 1 on every asset.

This is corroboration from an independent direction: the model's own trained
point estimator says its predictive mean is higher than the flat-tailed grid
mean says it is, by very nearly the amount §3's closures independently estimate
(1.09 against 1.09–1.21). It is not proof — TimesFM's point head is trained
under its own objective and is not guaranteed to be a mean — but it is the only
number in this audit that comes from the checkpoint rather than from an
assumption imposed on it.

Recorded and not acted on, per the module's own note: the head is "recorded next
to the grid in the fitted `spec()` but is not used, so that every adapter is
scored on one estimator" (`tsfm_common.py:27`). That is a defensible reason;
it is not a reason the number is uninformative.

### 5.4 No config has a *median-for-mean* substitution

The check that would catch one: the grid's own median against the scored
estimator.

| config | median of `q₀.₅ / v̂` | share of origins with `q₀.₅ < v̂` |
|---|---:|---:|
| `chronos` | 0.845 | **100.0 %** |
| `timesfm` | 0.881 | **100.0 %** |
| `moirai` | 0.838 | **100.0 %** |

The scored number is strictly above the grid's median at every one of 6,597
origins, which is what a mean of a right-skewed law must be. Nothing here is
silently a median.

---

## 6. Cost of a fix

**Which cells change.** The 33 quantile-TSFM cells: `chronos`, `timesfm`,
`moirai` × 11 assets. **Not** the 11 `patchtst` cells (§1.4) — so 33, against
the 44 the prompt anticipated.

**How long.** From `docs/P3_GRID_manifest.json`'s own `wall_clock_s`:

| config | cells | lane | cell time |
|---|---:|---|---:|
| `chronos` | 11 | gpu | 8.55 min |
| `timesfm` | 11 | gpu | 26.79 min |
| `moirai` | 11 | gpu | 5.71 min |
| **the 33 affected** | 33 | gpu | **41.05 min** |
| (`patchtst`, unaffected) | 11 | gpu | 27.12 min |
| (the whole GPU lane, for reference) | 44 | gpu | 68.17 min |

The GPU lane runs single-worker (`gpu_executor=ProcessExecutor(workers=1)`), so
cell time is wall clock for that lane: **~41 minutes**, not 68.

**Does it change the config hash?** Demonstrated on the stored artifact, SPY /
`chronos`, hash `75b969df…1c6f0d6`:

| change | resulting hash | moved? |
|---|---|---|
| as stored (v0.6.0) | `75b969df…1c6f0d6` | — |
| `package_version` → `0.7.0` | `744e4590…f9652fb` | **yes** |
| `spec()`'s `variance_from` → `tail_closed_mean_of_rv_quantile_grid` | `1758b8a3…12746ee` | **yes** |

**Both, and unavoidably.** Unlike Defect 2 (§4 of the LGBM audit), this fix
cannot be hidden inside `fit`: `variance_from` is a declared field of
`ZeroShotRVModel.spec()` (`tsfm_common.py:299`) whose whole purpose is to name
the estimator, and changing the estimator while leaving the label would make the
sidecar a false statement about how a number was produced. Shipping it is also a
public behaviour change to `volbench.models`, which under this project's
conventions is a release — and `package_version` is one of the eight hash blocks
(`docs/P3_INSTRUMENTATION_GAP.md` §4.1), so **all 143 hashes move, not just the
33**.

The 110 unaffected cells would then need recomputation to restore a
single-hash-space grid: the 99 CPU cells are **12.44 min of cell time** (~1–2
min wall at 12 workers, floored by `autoarima`'s longest single cell) plus
`patchtst`'s **27.12 min** on the GPU. Total for a clean single-hash re-run:
**41.05 + 27.12 GPU-minutes and 12.44 CPU-cell-minutes**, i.e. the whole
80.6-minute grid over again, since every hash moved. Or the grid keeps two hash
spaces and every table states which. That is a study-management decision, not a
code one.

---

## 7. The store is unchanged

Verified after every measurement in this document:

```
$ ls data/grid_primary/store | wc -l        # 286 files, 143 pairs
$ git status --short data/                  # (empty: data/ is gitignored and untracked)
```

Nothing in this document wrote to `data/grid_primary/`. The probe writes only to
`data/tsfm_dist_probe/`. The resumability re-run of the primary grid is recorded
at the end of `docs/P3_LGBM_SMEARING_AUDIT.md` §6, run once after both audits.
