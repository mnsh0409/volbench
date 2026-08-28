# P3 — what is actually predicted, and what it is actually scored against

**Why this document exists.** J1 established that 12 of the 13 configs emit
`Normal(0, sqrt(v))` and read that as a sign that the foundation models'
predictive distributions were being flattened into Gaussians before scoring.
That reading is half right, and the half it gets wrong changes the framing
decision that is blocked on it. This document answers the five questions
literally, from code, with the measurements behind each.

---

## Conclusion up front

1. **CRPS, log score, pinball, VaR/ES and FZ0 are all scored against the
   realized daily `realized_return`. QLIKE is the only metric on the variance
   axis, and it is the only one that touches `proxy_var` at all.** The
   predictive distribution is over the next-day **return**, never over
   variance. *A volatility-proxy objection therefore does not reach the primary
   metric.* It reaches QLIKE, and QLIKE alone. (§1)

2. **The TSFM quantile grids are not grids over returns.** Chronos, TimesFM and
   Moirai are fed a realized-variance series and emit a quantile grid over
   next-day **RV**. There is no way to score an RV grid against a realized
   return, and no Gaussian approximation of a return distribution is being
   taken anywhere: the reduction maps a distribution over RV to a *point*
   variance forecast, which then becomes the scale of a return distribution.
   The framing "the framework scores Gaussian approximations of the foundation
   models" is not what the code does. (§2)

3. **`patchtst` is not a fourth quantile model and must not be grouped with the
   other three.** It has no quantile head at all: it trains under MSE on log RV
   and emits `max_horizon` direct point outputs, retransformed by a Duan
   smearing factor. There is no grid to reduce. Its distributional assumption
   is *more* imposed than the TSFMs', not less. (§2.4)

4. **The pre-reduction grid is nowhere on disk.** Only the nine *levels* survive
   (in the sidecar's model spec); no `values` do. Recovering a grid requires
   re-running inference — but re-running reproduces the stored variance to
   2.2e-16 relative, so re-derivation is exact and does not require re-running
   the scored grid. (§3)

5. **What is lost is measurable and it is not small.** The RV grid the reduction
   collapses runs from ~0.25× to ~2.2× the variance that was scored (median
   over 6,597 origins). On the return axis the loss is exactly one number —
   excess kurtosis ~1.2 — and the VaR it implies differs from what was scored
   by **+10% to +12% at α=0.01, +4% to +6% at α=0.025, and ~0% at α=0.05**.
   The sign flips between 0.025 and 0.05, i.e. *inside* the three evaluated
   levels. (§4)

**Reported, not interpreted.** Nothing below says whether any of it should
change a result, a model, or a number. No fix is proposed and none is argued
for.

---

## 0. Provenance of every number here

| what | how |
|---|---|
| §1 | read off `src/volbench/evaluate.py` and `src/volbench/backtests.py`; no measurement |
| §2 | read off the four adapters; no measurement |
| §3 | all 143 sidecars and all 143 fragments scanned |
| §4, §5 | `src/volbench/benchmarks/tsfm_distribution_probe.py`, 200 origins × 11 assets × 3 configs = 6,600 origins (6,597 after NKX's three known `InsufficientHistoryError` origins) |

The probe re-runs inference at origins read out of each fragment's own
`origin_index` column, writes to no `ResultsStore`, moves no config hash and
rewrites no fragment — the same contract
`benchmarks/fit_diagnostics_probe.py` runs under. Origins are taken evenly
spaced across each cell's whole evaluation span rather than as a prefix, so the
sample is not one volatility regime.

```
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 uv run --extra classical --extra tsfm python -m \
  volbench.benchmarks.tsfm_distribution_probe --models chronos timesfm moirai \
  --n-origins 200 --out data/tsfm_dist_probe/panel.parquet
```

---

## 1. Per metric: the predictive object and the realization

**This is the answer the framing decision is blocked on.** Every score in the
store is produced by `evaluate._score`, which is handed exactly three things:
the `Distribution`, `realized_return`, and `proxy_var`.

```python
# src/volbench/evaluate.py:228
def _score(dist, realized_return, proxy_var, levels) -> dict[str, Any]:
    mean, variance = forecast_moments(dist)
    ...
    "crps": dist.crps(realized_return) if target_ok else math.nan,     # :249
    out["log_score"] = dist.log_score(realized_return)                 # :256
    out["qlike"] = qlike(variance, proxy_var)                          # :274
    var_quantile = dist.quantile(level)                                # :278
    out[f"pinball_{tag}"] = dist.pinball(realized_return, level)       # :289
    out[f"hit_{tag}"] = float(realized_return < var_quantile)          # :290
```

| metric | predictive object | realization it is scored against | axis | line |
|---|---|---|---|---|
| **CRPS** | the whole predictive law of the next-day return | `realized_return` | return | `evaluate.py:249` |
| **log score** | the same law's density | `realized_return` | return | `evaluate.py:256` |
| **pinball** at 0.01/0.025/0.05 | the same law's `quantile(level)` | `realized_return` | return | `evaluate.py:289` |
| **VaR hit** | the same law's `quantile(level)` | `realized_return` | return | `evaluate.py:290` |
| **QLIKE** | the law's **variance** only | `proxy_var` | **variance** | `evaluate.py:274` |
| **FZ0** | stored `var_<level>` and `es_<level>` | `realized_return` | return | `backtests.py:357`, `:545` |

FZ0 is computed downstream, in `backtests.var_backtest`, and reads only
`realized_return`, `var_<level>` and `es_<level>` — the docstring is explicit
that an ES is never approximated from `forecast_var`, "because that would
presume a distributional family the row does not record". So FZ0 never touches
the variance axis either.

The convention is not incidental; it is stated as a module invariant:

```
# src/volbench/evaluate.py:11
- ``FittedModel.predict(h)`` returns a Distribution over the **next-period
  return**, never over variance. The variance forecast is a property of that
  distribution.
- CRPS, log score, pinball and VaR hits are therefore computed against the
  realized *return*; QLIKE compares the distribution's variance against a
  realized-variance *proxy*.
```

### 1.1 The consequence, stated plainly

`proxy_var` enters exactly one column of 32. A criticism of the realized-variance
proxy — sparse-sampling noise, the overnight term, D-016's estimator choice — is
a criticism of the **QLIKE column and nothing else**. CRPS, the log score, the
three pinball columns, the three hit columns and FZ0 are scored against a
realized close-to-close log return, which is observed rather than estimated.

The two columns are on the same rows and are easy to conflate in a table; they
should never be footnoted together.

### 1.2 The Normal claim, re-verified independently

J1's finding is confirmed here from the stored columns alone, without re-running
anything: for a `Normal(0, sqrt(v))` the stored `var_<α>` must equal
`sqrt(v)·Φ⁻¹(α)` and `es_<α>` must equal `−sqrt(v)·φ(Φ⁻¹(α))/α`. Checked on
every row of every cell:

| config | max relative deviation from the Normal closed forms | rows | verdict |
|---|---:|---:|---|
| `naive`, `ewma`, `garch11`, `gjr`, `har`, `autoets`, `autoarima`, `lgbm`, `chronos`, `timesfm`, `moirai`, `patchtst` | ≤ 5.1e-16 | 49,606–49,627 each | `Normal(0, sqrt(v))` |
| `garch11_t` | **5.4e-01** | 49,627 | not Normal (Student-t, D-014) |

12 of 13, as J1 reported. §2 is why that is true of the TSFMs.

---

## 2. Where each model's output is reduced to one variance

Quoted per adapter, because the four do **not** share a path.

### 2.1 The shared reduction — `chronos`, `timesfm`, `moirai`

All three subclass `ZeroShotRVModel` and share one `predict`. The reduction is
these four lines, and it is the only place it happens:

```python
# src/volbench/models/tsfm_common.py:364
def predict(self, h: int) -> Distribution:
    """``Normal(0, sqrt(vhat))`` over the return at ``t+h``; ``vhat`` = mean of the RV grid."""
    fc = self.rv_forecast(h)
    taus = np.asarray(fc.taus, dtype=np.float64)
    sorted_values, crossings = rearrange_quantiles(fc.values[h - 1])   # (i) repair crossings
    clipped = int(np.sum(sorted_values < 0.0))
    grid = np.maximum(sorted_values, 0.0)                              # (ii) clip negatives
    vhat = quantile_grid_mean(taus, grid)                              # (iii) THE REDUCTION
    ...
    return Normal(mu=0.0, sigma=math.sqrt(vhat))                       # (iv) the scored object
```

`quantile_grid_mean` (`tsfm_common.py:124`) is the **first moment of the law
whose quantile function linearly interpolates the grid, flat outside it**:
probability mass `taus[0]` is placed at `values[0]` and `1 - taus[-1]` at
`values[-1]`. Nine levels 0.1…0.9 means **20 % of the probability mass sits in
those two flat closures** for all three configs on this grid (confirmed from the
sidecars: `quantile_levels` is `[0.1, …, 0.9]` on all 33 TSFM cells).

The input contract is the other half of the answer, and it is why "reduced to a
Gaussian" is the wrong description:

```
# src/volbench/models/tsfm_common.py:12
1. ``fit(train)`` takes a 1-D **realized-variance** series in daily units —
   the same input contract as HAR, never returns
2. ``predict(h)`` ... takes the model's own predictive distribution of RV at
   ``t+h`` ... and uses its **MEAN** as the variance forecast ``vhat``.
3. The scored object is ``Normal(mu=0, sigma=sqrt(vhat))`` over the
   next-period **return**
```

So: grid over RV → one number → the *scale* of a return law. The grid and the
scored object never lived on the same axis, and no step in between discards a
return distribution the model produced, because no model here produces one.

### 2.2 Where the three differ

The reduction is shared; what reaches it is not.

| | `chronos` | `timesfm` | `moirai` |
|---|---|---|---|
| checkpoint (from the sidecars) | `amazon/chronos-bolt-small` | `google/timesfm-2.5-200m-pytorch` | `Salesforce/moirai-2.0-R-small` |
| backend returns | `(quantiles, median)`; **the second element is discarded** (`tsfm_chronos.py:120`) | `(points, quantiles)`, shape `(h, 10)`: index 0 the point head, 1..9 the quantiles (`tsfm_timesfm.py:169`) | `(batch, n_quantiles, h)`, transposed (`tsfm_moirai.py:124`) |
| `native_mean` recorded | `None` | `arr[:, 0]` — the model's own point head | `None` |
| point head used for `vhat`? | n/a | **no** | n/a |
| in-package positivity repair | none | `infer_is_positive=True` clamps **inside the package**, before the adapter sees the grid | none |
| in-package crossing repair | none | `fix_quantile_crossing=True` | none |

`chronos`'s pipeline returns a `mean` that its own source says is the 0.5
quantile. The adapter never binds it (`quantiles, _median = ...`), so the
documented workaround is structural rather than conditional — see §2 of
`docs/P3_TSFM_VARIANCE_AUDIT.md` for the verification on the panel.

### 2.3 `input_scale` sits between the context and the backend

Not part of the reduction, but part of the path: the context is multiplied by
`1e4` before it reaches the model and the returned quantiles are divided by it
(`tsfm_common.py:351-360`). It is a fixed, data-independent unit convention in
`spec()`, forced by Moirai-2's scaler epsilon.

### 2.4 `patchtst` — there is no grid, and therefore no reduction

`patchtst` is a **trained** model, not a foundation model, and its predictive
object was never a distribution:

```python
# src/volbench/models/patchtst.py:197
def predict(self, h: int) -> Distribution:
    mu = float(self.log_forecast[h - 1])
    factor = float(self.smearing[h - 1])
    ...
    return Normal(mu=0.0, sigma=math.sqrt(variance_from_log(mu, factor)))
```

`spec()` says `"loss": "mse"` and the architecture is "flattened and projected
to `max_horizon` **direct outputs**" — a point head, no quantile head anywhere.
The Gaussian is therefore not an approximation of anything the net emitted; it
is the only distributional statement in the model, and it is imposed by the
adapter. The retransformation from `E[log RV]` to a variance is Duan's smearing
factor over **in-sample** residuals — the same construction
`docs/P3_LGBM_SMEARING_AUDIT.md` measures for `lgbm`, and `patchtst` carries it
too.

Grouping `patchtst` with Chronos/TimesFM/Moirai as "the four foundation models"
is wrong on both counts: it is not zero-shot (D-005 covers the other three only)
and it has no native predictive distribution to lose.

---

## 3. Is the pre-reduction grid anywhere on disk?

**No. Not for any origin of any cell.** Answered by scanning the artifact, not
by inference.

**Fragments.** All 143 parquet files share one 32-column schema, listed in
`_result_dtypes` (`evaluate.py:296`). It holds `forecast_mean`, `forecast_var`,
and `var_/es_/pinball_/hit_` at three levels. There is no column that could
hold nine RV quantiles, and no cell has a wider schema than any other.

**Sidecars.** 33 of the 143 mention "quantile" — the 3 TSFM configs × 11 assets.
What they carry is the *levels*, never the values:

```json
"quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
"variance_from": "mean_of_rv_quantile_grid"
```

The reason is structural and already documented: `run_backtest` hashes and
records the **probe's** spec — `probe = model_factory()`, an *unfitted*
instance (`evaluate.py:867`) — so the `rv_forecasts` block that
`FittedTSFM.spec()` builds per origin never reaches disk
(`docs/P3_INSTRUMENTATION_GAP.md` §2.1).

### 3.1 But re-derivation is exact, and that is the operative distinction

"Not on disk" is not the same as "gone". The grid is a deterministic function of
the context, and re-running it reproduces what was scored:

| config | origins compared | max relative deviation of the re-run `vhat` from the stored `forecast_var` |
|---|---:|---:|
| `chronos` | 200 (SPY) | **2.21e-16** |
| `timesfm` | 200 (SPY) | **2.21e-16** |
| `moirai` | 200 (SPY) | **2.20e-16** |
| `patchtst` | 29 (SPY, refit origins only) | **2.01e-16** |

One unit in the last place. So the cost of recovering a grid is GPU time, not a
lost experiment — and, importantly, **measuring the grid does not require
re-running the scored grid**, because the re-run lands on the same numbers.
Cost is in §6 of `docs/P3_TSFM_VARIANCE_AUDIT.md`.

`patchtst` is compared at refit origins only, for a reason worth recording: it
implements no `update`, so between refits the evaluator **holds** the forecast
issued at the last scheduled fit. At the 171 sampled non-refit origins a fresh
fit differs from the held value by a median of **40 %** and a maximum of 11×.
That is the documented frozen-between-refits behaviour (`patchtst.py:69`), not a
defect, and it is out of scope here — but any table that reads `patchtst`'s
forecasts as daily-conditioned would be wrong, and `conditioned_through` is the
column that says so.

---

## 4. What is lost

Measured on 6,597 origins (200 per asset × 11 assets × 3 configs, minus NKX's
three known `InsufficientHistoryError` origins at 499).

### 4.1 On the RV axis: a whole law, replaced by its own mean

The scored object records `vhat` and nothing else. What the grid said, relative
to that one number:

| config | max<sub>τ</sub> \|q<sub>τ</sub> − v̂\| / v̂, median | p95 | max | q<sub>0.1</sub>/v̂ median | q<sub>0.9</sub>/v̂ median |
|---|---:|---:|---:|---:|---:|
| `chronos` | 1.198 | 1.655 | 2.817 | **0.247** | **2.198** |
| `timesfm` | 1.191 | 1.656 | 2.496 | **0.178** | **2.191** |
| `moirai` | 1.149 | 1.504 | 3.598 | **0.294** | **2.149** |

The model's own 10th and 90th percentiles for tomorrow's variance sit at roughly
**a quarter of** and **twice** the number that was scored. That entire interval
is discarded at `tsfm_common.py:371`.

Shape of the discarded law (panel medians):

| config | coefficient of variation | skewness | excess kurtosis |
|---|---:|---:|---:|
| `chronos` | 0.630 | +0.659 | −0.751 |
| `timesfm` | 0.651 | +0.508 | −0.902 |
| `moirai` | 0.602 | +0.682 | −0.755 |

All three grids are **right-skewed** — as an RV law should be — and platykurtic,
which is what a nine-level grid with 20 % of its mass in two point atoms looks
like when its moments are taken literally.

### 4.2 On the return axis: exactly one number

The comparison the paper's metrics actually see. If the model's RV uncertainty
were carried into the return distribution rather than collapsed, the predictive
law would be the scale mixture `r = sqrt(V)·Z` with `V` the grid law and `Z`
standard normal. That mixture and the scored `Normal(0, sqrt(v̂))`:

- have the **same mean** (zero — both symmetric);
- have the **same variance** (`E[V] = v̂`, by construction of the reduction);
- have the **same skewness** (zero, both);
- differ in **excess kurtosis**, which is `3·Var(V)/E[V]²` in closed form.

So on the return axis the reduction is exactly *one* lost number, and its
measured value is:

| config | implied excess kurtosis of the return law, median | IQR |
|---|---:|---|
| `chronos` | **1.189** | 1.05–1.53 |
| `timesfm` | **1.270** | 1.11–1.68 |
| `moirai` | **1.088** | 0.94–1.30 |

For scale: an excess kurtosis of 1.2 is what a Student-t with about ν = 9 has
(`6/(ν−4) = 1.2`).
The scored object has 0.

### 4.3 VaR, at the three levels that are actually evaluated

The mixture's quantile against the `Normal`'s, as a percentage change. Positive
means the mixture's VaR sits **further into the loss tail** than what was scored.

| config | α = 0.01 | α = 0.025 | α = 0.05 |
|---|---:|---:|---:|
| `chronos` | **+10.90 %** (p5–p95 +5.2 to +17.3) | **+4.94 %** (+1.9 to +8.9) | −0.30 % (−1.0 to +1.9) |
| `timesfm` | **+11.68 %** (+4.8 to +18.4) | **+5.90 %** (+1.9 to +9.9) | +0.50 % (−0.5 to +2.0) |
| `moirai` | **+10.02 %** (+4.6 to +15.0) | **+4.40 %** (+1.6 to +7.2) | −0.42 % (−1.1 to +0.5) |

Share of origins where the mixture is deeper: **100.0 %** at α = 0.01 and at
α = 0.025, for all three configs. At α = 0.05 it is 33 % (`chronos`), 71 %
(`timesfm`), 16 % (`moirai`).

**The sign changes inside the evaluated levels.** This is a property of matching
the variance of a symmetric fat-tailed law with a Normal — mass moves from the
shoulders to the far tail, so the two quantile functions cross — and the
crossing falls between α = 0.025 and α = 0.05 on every config. Any sentence of
the form "the reduction understates the TSFMs' tail risk" is therefore true at
1 % and 2.5 % and false at 5 %, and `tests/test_tsfm_distribution_probe.py::
TestReturnAxis::test_the_var_error_changes_sign_inside_the_evaluated_levels`
pins that so it cannot be quietly written the other way.

Note this is a **shape** effect at fixed variance, and is arithmetically
separate from the **level** effect measured in
`docs/P3_TSFM_VARIANCE_AUDIT.md` §3, which moves every level in the same
direction. The two do not cancel; at α = 0.01 they compound.

---

## 5. The `chronos` zero-clip

J1 sampled every tenth refit origin and found `chronos` clipping at 14 of 241.
On the denser sample here (200 origins per asset, all 11 assets):

### 5.1 What it does, and when

From `tsfm_common.py:364-371`, in order: (i) `rearrange_quantiles` **sorts** the
grid and counts crossings; (ii) `clipped = sum(sorted < 0)` counts the negatives
and `grid = maximum(sorted, 0.0)` replaces each with **exactly zero**; (iii)
`vhat = quantile_grid_mean(taus, grid)`.

**The clip fires strictly before the reduction.** The mean is taken from the
clipped grid, never the raw one, and the count is written to the fitted
`spec()` — which never reaches disk (§3).

### 5.2 Which levels fire, on 2,199 origins per config

| config | origins with ≥1 clip | negative raw quantiles, by level | crossings rearranged |
|---|---:|---|---:|
| `chronos` | **119 (5.4 %)** | q<sub>0.1</sub>: 119 · q<sub>0.2</sub>: 27 · q<sub>0.3</sub>: 9 · q<sub>0.4</sub>: 2 | 0 |
| `moirai` | **12 (0.5 %)** | q<sub>0.1</sub>: 12 · q<sub>0.2</sub>: 7 · q<sub>0.3</sub>: 4 · q<sub>0.4</sub>: 1 | 0 |
| `timesfm` | 0 | none | 0 |

It is a **lower-tail** phenomenon: never above q<sub>0.4</sub>, on any config, at
any origin. No model emitted a crossed grid anywhere. `chronos`'s clipped
origins are spread over all 11 assets (BTC-USD 37, ETH-USD 22, DIA 13, SPY 11,
DAX 9, CAC 7, NKX 7, NDX 5, HSI 3, TWSE 3, KOSPI 2) — J1's "none on HSI or TWSE"
was a small-sample artefact of the every-tenth stride.

**`moirai` clips too**, on 12 origins across BTC-USD, ETH-USD, DAX and NDX. J1's
"0 on all" is correct for its 241-origin sample and wrong for the panel; the
instrumentation-gap table should be read as a sample, not a census.

### 5.3 Size of the clip's effect on the scored variance

Recomputing `quantile_grid_mean` from the sorted-but-unclipped grid on the 119
affected `chronos` origins:

| | value |
|---|---:|
| unclipped mean / scored v̂, median | 0.980 |
| min | 0.189 |
| max | 0.9999 |

So the clip **raises** the scored variance, by 2.0 % at the median of the
affected origins and by up to 5.3× in the worst case. Direction is always the
same (a negative quantile replaced by zero can only raise the mean). Against the
whole panel it moves 119 of 2,199 `chronos` origins and 12 of 2,199 `moirai`
ones; it moves no `timesfm` origin.

### 5.4 A related finding the adapter's counter cannot see

`timesfm` reports `clipped_at_zero = 0` at every origin — and that is true of the
*adapter's* clip. But its grid contains an **exactly zero** quantile at
q<sub>0.1</sub> on **215 of 2,199 origins (9.8 %)**, at q<sub>0.2</sub> on 8 and
at q<sub>0.3</sub> on 1, spread across all 11 assets (SPY 31, TWSE 26, DAX 23,
NDX 23, DIA 22, BTC-USD 19, CAC 17, NKX 17, KOSPI 15, HSI 12, ETH-USD 10).

The reason is §2.2's `infer_is_positive=True`: the TimesFM package clamps the
non-negativity **inside itself**, so the value the adapter receives is already
zero rather than negative and `sum(sorted < 0)` is zero. The repair happened; the
counter cannot report it. `timesfm` is therefore the config most affected by
lower-tail flooring and the one whose instrumentation shows it least.

A zero RV quantile means the model puts at least 10 % of its predictive mass on
"tomorrow's variance is at most zero", which is worth recording separately from
what the clip does to `vhat`.

---

## 6. What could not be answered without re-running cells

Nothing in this document required re-running a scored cell, and none was
re-run. The primary store was opened read-only throughout; §7 of
`docs/P3_TSFM_VARIANCE_AUDIT.md` records the resumability check.

Two things are **not** answerable from what exists, and are named rather than
estimated:

1. **What the checkpoints' true predictive laws are outside their outermost
   quantile levels.** The grid is the model's entire output; nothing identifies
   the tail beyond q<sub>0.1</sub> and q<sub>0.9</sub>. Any number for "the
   correct variance" is a number under a tail assumption, and
   `docs/P3_TSFM_VARIANCE_AUDIT.md` §3 reports three closures and their range
   rather than one figure.

2. **Whether the per-origin grid at a *specific* stored origin was what the
   probe reproduced.** §3.1 establishes agreement to 1 ulp on 800 origins, which
   is as close to a proof as re-running can give, but it is a re-derivation and
   not a read of the original.

---

## 7. Decision table

The three items measured across this document and its two companions. Sizes are
panel medians; ranges are across the three affected configs. **No
recommendation is made — these are measurements and costs.**

| item | what the pipeline does | size of the effect | recoverable post-hoc? | re-run cost | changes config hash? |
|---|---|---|---|---|---|
| **Distributional reduction** (Part 0) | Chronos / TimesFM / Moirai emit a 9-level quantile grid over next-day **RV**; `tsfm_common.py:371` takes its mean `v̂` and `:384` scores `Normal(0, sqrt(v̂))` over the **return**. The grid never reaches the store. `patchtst` is unaffected — it has no grid. | On the RV axis: the discarded law runs from **0.18–0.29× to 2.15–2.20× v̂** (median q₀.₁, q₀.₉). On the return axis it is exactly one number — **excess kurtosis 1.09–1.27**, against 0 for what was scored — and the VaR it implies differs by **+10.0 to +11.7 % at α=0.01, +4.4 to +5.9 % at α=0.025, −0.4 to +0.5 % at α=0.05**. The sign changes inside the evaluated levels. | **No.** Fragments carry 32 columns and no grid; sidecars carry the *levels* only. But re-running reproduces the stored variance to **2.2e-16** relative, so the grid is exactly re-derivable — measuring it cost 3.8 GPU-min and required no re-run of a scored cell. | **33 cells, 41.05 GPU-min** (`chronos` 8.55 + `timesfm` 26.79 + `moirai` 5.71), single-worker GPU lane, so that is wall clock. Not 44 cells — `patchtst` is out. | **Yes, unavoidably.** `variance_from` is a declared field of `spec()` whose purpose is to name the estimator (`75b969df…` → `1758b8a3…` on SPY/`chronos`). Shipping it is also a release, and `package_version` moves **all 143** (`744e4590…`). |
| **Variance derivation** (Defect 1) | The mean is taken from a linear interpolant with **flat tails**, placing **20 % of the mass** in two point atoms at q₀.₁ and q₀.₉ — the D-014 truncation bias, not a wrong choice of functional. The mean is the right estimator for a variance target. | Correct / current, three closures: **lognormal 1.11–1.20**, **loglinear 1.09–1.10**, **empirical (assumption-free) 1.205–1.212**. Crude check: realized RV averages **1.25–1.30×** `v̂`. VaR/ES shift is **exactly** `sqrt(ratio)` — **+5.4 % (`moirai`), +6.5 % (`chronos`), +9.6 % (`timesfm`)** at every level. The recorded "~8 % at ν=5" was borrowed from M1's Student-t GARCH and never measured on a TSFM. | **No**, twice over: no grid is stored, *and* the grid is the checkpoint's whole output, so the law beyond q₀.₁/q₀.₉ is not identified by anything. Every figure above is a figure under a stated closure; the spread between closures is reported instead of one number. | Same 33 cells, **41.05 GPU-min**. Would be fixed in the same pass as the row above. | **Yes**, for the same two reasons and the same hashes. |
| **LightGBM smearing** (Defect 2) | Duan's factor is `mean(exp(e))` over the fit window's **in-sample** residuals (`lgbm.py:390`), reused unchanged at every origin in the refit block. | Shipped factor **1.371** against a realized **1.678** (panel medians) — the correction the theory asks for is **21.8 % larger**, so stored variances sit **17.9 % below** it; VaR/ES **+10.4 %** at every level. Per asset the gap runs 1.07× (ETH-USD) to 1.28× (SPY), same sign on all 11. **Regime-dependent**: crisis/calm factor ratio **1.35–2.07** on the nine equities, and **5.57 vs 1.56** in the COVID window. An out-of-fold factor gives **1.703** — within 1.5 % of the realized target. The 100-round cap **is binding**: training MSE 0.78→0.24 and the factor 1.49→1.13 from 25 to 800 rounds, so the capacity cap and the collapsed residual are one story. | **Yes, partly.** The factor is recoverable from the store: `mu_hat = log(forecast_var) − log(smear)` inverts exactly because `update` never re-estimates, and 2,366 probe fits supply `smear`. What is *not* recoverable is a corrected fragment — the stored scores were computed with the old factor. | **11 cells, 0.77 min of cell time**, CPU lane, 12 workers. (Recorded 0.699 min; SPY's 0.006 s was a cache hit.) The prompt's arithmetic is confirmed. | **Not for the change itself** — the factor lives in `fit`, not `spec()`, so the hash is unmoved (`65876078…` unchanged). That is a hazard: `ResultsStore.has()` short-circuits on file existence, so a re-run would report `cached 11, computed 0` and keep the old numbers. **Yes** if the arm is named in `spec()` (`3e2821e7…`, correct per `_rv.py:22`), and **yes for all 143** via `package_version`. |
