# P3 — ten of thirteen configs have no convergence instrumentation

**The problem.** `docs/P3_GRID_manifest.json` reports `n_fits = 0` and a
fallback rate of `nan` for ten of the thirteen configs. For `naive` and `ewma`
that is correct: nothing is estimated, so nothing can fail. For the other
eight it is not. `autoarima` and `autoets` **select** a model and run an
optimizer; `har` solves a least-squares problem; `lgbm` boosts; `patchtst`
trains a network per origin with Adam and early stopping; the three zero-shot
TSFMs repair their own quantile output before it is scored. A table printing a
number for three configs and `nan` for ten reads as if the ten were perfect,
and **a reader cannot distinguish "never failed" from "never measured".**

**Reported, not interpreted.** Nothing below says whether any signal, or its
value, should change a result.

**Conclusion up front.** Step 1 (report the gap) is done. **Step 2 was not
done, deliberately**: wiring the signals into `fit_status` cannot reach the
143 stored fragments without recomputing every one of them, and — separately —
cannot ship at all without moving all 143 config hashes. Both reasons are
demonstrated in §4. The wiring is written down there instead.

**The one number a reader should not skip.** Exactly one of the eight has a
signal shaped like a *failure*: `autoarima`'s optimizer returns a **non-zero
status on 2,334 of its 2,366 fits (98.6 %)** — the status `statsforecast`
itself warns about — and nothing in the store records it (§3.1). The other
seven report health or selection values, and none of those shows a fit that did
not do what it was asked (§3.2–§3.6). Whether the `autoarima` figure matters is
a question for the results review. That the question was not previously
*askable* from the committed artifacts is the point of this document.

**The reporting rule from here on.** In every fallback-rate table anyone
produces, the ten uninstrumented configs read **`not instrumented`** — never
`nan`, never `0`.

---

## 1. The gap, stated exactly

`fit_status` is a per-row string that describes the *scheduled fit* the row
rests on (docs/P3_ANALYSIS_ASSUMPTIONS.md §5). Its vocabulary is fixed by
`FitDiagnostics.status()`: `ok`, `nonconverged`, `fallback=<name>`, each
optionally with `|<detail>`. The empty string is **reserved** for a model that
reports nothing at all, and `status()` can never return it — so an empty
`fit_status` cannot be mistaken for a clean fit by anything that reads the
docstring. The manifest's `nan` fallback rate is the same reservation surfaced
one level up.

Across the whole grid:

| `fit_status` head | rows | configs |
|---|---:|---|
| `""` | 496,270 | the ten |
| `ok` | 148,085 | `garch11`, `garch11_t`, `gjr` |
| `fallback=ewma` | 796 | `garch11`, `garch11_t`, `gjr` |

Only `volbench/models/garch.py` implements `fit_diagnostics()`. Grepped: it is
the only file in `src/volbench/models/` that mentions `FitDiagnostics` outside
`base.py`.

## 2. Does each backend expose a signal at all?

Measured, not inferred: each adapter was fitted and the fitted object inspected.

| config | estimates? | signal the backend exposes | in the fragments? | in the sidecars? |
|---|---|---|---|---|
| `naive` | no | none — it *is* its window statistic | n/a | n/a |
| `ewma` | no | none — `lambda` is fixed at 0.94 | n/a | n/a |
| `har` | yes (OLS) | `np.linalg.lstsq` returns the design matrix's **rank** and **singular values**; `har.py` keeps only `[0]`, the coefficients | **no** | **no** |
| `autoets` | yes (selection + optimizer) | `backend.model_`: `method` (the **selected ETS form**, e.g. `ETS(A,N,N)`), `aic`, `aicc`, `bic`, `loglik`, `sigma2`, `n_params` | **no** | **no** |
| `autoarima` | yes (selection + optimizer) | `backend.model_`: **`code`** — `scipy.optimize.minimize`'s own `status` for the selected model — plus `arma` (the **selected order**), `aic`, `aicc`, `loglik`, `sigma2` | **no** | **no** |
| `lgbm` | yes (boosting) | `booster.num_trees()`, `booster.current_iteration()` against the requested `num_boost_round` | **no** | **no** |
| `chronos` | **no** (zero-shot, D-005) | not convergence — a per-origin **forecast-repair** record: `crossings_rearranged`, `clipped_at_zero`, already on `FittedTSFM.spec()` | **no** | **no** |
| `timesfm` | **no** (zero-shot) | same | **no** | **no** |
| `moirai` | **no** (zero-shot) | same | **no** | **no** |
| `patchtst` | **yes (trains per origin)** | a full training record already on `FittedPatchTST.training`: `epochs_run`, `best_epoch`, `stopped_early`, `best_val_mse`, `final_train_mse`, `n_train_windows`, `n_val_windows` | **no** | **no** |

### 2.1 Nothing is already stored — checked, not assumed

**No per-fit diagnostic from any of the ten is anywhere in the store**, so
there is nothing to extract post-hoc. The reason is structural rather than an
oversight:

`evaluate.run_backtest` builds the config from a **probe** — `probe =
model_factory()`, an *unfitted* instance — and hashes `probe.spec()`. The
sidecar records that same unfitted spec. So even for the four adapters whose
*fitted* `spec()` already carries the diagnostic — `patchtst` and the three
TSFMs — the sidecar carries the hyperparameters and nothing else. Verified on the
stored artifact:

```
patchtst sidecar spec has 'max_epochs':   True      (a hyperparameter)
patchtst sidecar spec has 'epochs_run':   False     (the diagnostic)
patchtst sidecar spec has 'best_val_mse': False     (the diagnostic)
```

The same probe-not-fit path is why the TSFMs' `rv_forecasts` block — which
holds `crossings_rearranged` and `clipped_at_zero` per horizon — never reaches
disk either.

### 2.2 `autoarima`'s `code` is a real convergence flag

Worth naming precisely, because it is the one signal in the list that is
already a convergence verdict rather than something a verdict could be built
from.

`statsforecast/arima.py` line 923 writes `"code": res.status` into the fitted
model dict, where `res` is the result of `scipy.optimize.minimize` with
`optim_method="BFGS"`. The same value drives statsforecast's own warning at
line 673: *"possible convergence problem: minimize gave code {res.status}"* —
issued whenever `res.status > 0`. scipy's BFGS statuses are `0` (converged),
`1` (max iterations), `2` (precision loss), `3` (NaN).

`autoets` has no equivalent flag; its selection outcome (`method`) and its
`loglik` are what it offers.

Confirmed against scipy itself rather than read off documentation: BFGS returns
status 0 for `Optimization terminated successfully.` and status 2 for
`Desired error not necessarily achieved due to precision loss.` (`scipy.optimize._optimize._status_message['pr_loss']`).

## 3. Measured over the grid's own refit origins

`src/volbench/benchmarks/fit_diagnostics_probe.py` refits each adapter at the
**origins the grid actually fitted at** — read out of each fragment's `refit` /
`fit_origin` columns, not re-derived — using the same
`AssetData.fit_series(policy)` the runner hands a cell, so D-018 compaction
applies identically. It writes to no `ResultsStore`, moves no hash, and
rewrites no fragment.

`naive` and `ewma` are excluded — there is nothing to read. The four CPU
adapters were probed at **every** scheduled refit; the four torch-backed ones at
**every tenth**, which is what keeps them affordable.

| config | fits probed | of | wall clock |
|---|---:|---|---:|
| `har` | 2,366 | all | 8.5 s |
| `autoets` | 2,366 | all | 16.0 s |
| `autoarima` | 2,366 | all | 370.5 s |
| `lgbm` | 2,366 | all | 25.5 s |
| `chronos` | 241 | every 10th | 6.8 s |
| `timesfm` | 241 | every 10th | 10.4 s |
| `moirai` | 241 | every 10th | 2.8 s |
| `patchtst` | 241 | every 10th | 102.7 s |

**No probed fit raised**, on any config. The 2,366 is not a typo against the
GARCH family's 2,367: the variance-fed configs lose NKX's first refit block to
`InsufficientHistoryError` (docs/P3_ANALYSIS_VALIDITY.md §2), so NKX has 228
scheduled fits on them against 229 on the return-fed ones. The probe's per-asset
counts match the store's own `refit`/`fit_origin` columns exactly, 11 of 11.

### 3.1 `autoarima` — the optimizer flag is non-zero on 98.6 % of fits

| `code` | fits | scipy's own message for that status |
|---:|---:|---|
| 0 | **32** | Optimization terminated successfully. |
| 2 | **2,334** | Desired error not necessarily achieved due to precision loss. |

Never `1` (max iterations) and never `3` (NaN). Per asset, the share with
`code != 0` runs from 0.974 (DIA) to **1.000** (BTC-USD, 133 of 133).

`statsforecast` warns on exactly this condition in its own code path
(`arima.py:673`, `if res.status > 0`), so the value is the backend's own view of
its fit, not a reading imposed here. It is recorded nowhere in the store.

Model selection also varied: **40 distinct** `(p,d,q,P,D,Q,s)` orders were
selected across the 2,366 fits, the most common being `(0,1,0,0,1,1,0)` (766
fits), then `(1,1,0,...)` (291) and `(2,1,0,...)` (269). Which order a cell used
at a given origin is not recoverable from anything committed.

`loglik` and `aicc` are finite on all 2,366.

**Stated and left there.** Whether a `pr_loss` termination of the AutoARIMA
likelihood should change how its column is read is a question for the results
review. What this section establishes is only that the question is *askable* —
it was not, from the manifest.

### 3.2 `autoets` — two selected forms, no convergence flag

| selected form | fits |
|---|---:|
| `ETS(A,N,N)` | 2,304 |
| `ETS(A,Ad,N)` | 62 |

`backend.model_` carries no `code`, so AutoETS offers a *selection* signal and
not a convergence one. `loglik` and `aicc` are finite on all 2,366. Which of the
two forms a cell used at a given origin is not recoverable from the store.

### 3.3 `har` — nothing to report, and now that is a measurement

| | |
|---|---:|
| full-rank design matrices | **2,366 / 2,366** (rank 4 of 4, always) |
| condition number, median | 228 |
| condition number, max | 589 |
| fits with condition number > 1e8 | **0** |

The OLS solve is well posed at every one of the grid's refit origins. That is
now a checked statement rather than an absent one — which is the whole
distinction this document is about.

### 3.4 `lgbm` — every fit built every round

100 of 100 boosting rounds on **2,366 / 2,366** fits. LightGBM has no
convergence criterion to fail, and nothing here was configured to stop early, so
this is the only sense in which a `lgbm` fit can be incomplete, and none was.

### 3.5 The three zero-shot TSFMs — a repair count, not a convergence flag

| config | fits probed | quantile crossings rearranged | quantiles clipped at zero | fits with any repair |
|---|---:|---:|---:|---:|
| `chronos` | 241 | 0 on all | 1 on 9 fits, 2 on 3, 3 on 2 | **14 (5.8 %)** |
| `timesfm` | 241 | 0 on all | 0 on all | 0 |
| `moirai` | 241 | 0 on all | 0 on all | 0 |

No model emitted a crossed quantile grid at any sampled origin. `chronos`
emitted at least one negative RV quantile at 14 of 241 sampled origins, spread
across 9 of the 11 assets (none on HSI or TWSE); those quantiles were clipped at
zero before the variance was taken, and the clip count is discarded. `timesfm`
and `moirai` never needed either repair.

### 3.6 `patchtst` — early stopping fired on every sampled fit

| | |
|---|---:|
| `stopped_early` | **True on 241 / 241** |
| `epochs_run` | min 11, median 16, max 44 (against `max_epochs = 100`) |
| `best_epoch` | min 1, median 6, max 34 |
| fits reaching `max_epochs` | **0 / 241** |
| non-finite `best_val_mse` | **0** |

Per-asset median `epochs_run` runs from 14 (HSI) to 27.5 (ETH-USD). No fit ran
out its epoch budget; every one stopped on the validation criterion and restored
its best weights. `best_epoch = 1` occurs, meaning the first epoch was the best
one on that origin's validation windows.

All of it already exists on `FittedPatchTST.training`, and none of it reaches
disk.

## 4. Step 2: the wiring is written down, and not applied

The instruction is to wire the signals in **only if** it costs no config-hash
change, no fragment rewrite, and a resumability re-run still reporting 143
cached. Both conditions were checked against the artifact, and the honest
answer is that the wiring cannot help this grid.

### 4.1 A fitted object's diagnostic cannot move a hash — but a release can

The config hash is computed over eight blocks: `model` (`name` + the
**unfitted** `spec()`), `data`, `splitter`, `scoring`, `protocol`, `seed`,
`environment`, and **`package_version`**. All 143 stored sidecars re-hash to
their own filename under `results.config_hash` (143/143, re-checked here), so
that list is the whole of it.

Adding `fit_diagnostics()` to a *fitted* class therefore moves nothing: it is
not in `spec()`, and `spec()` is read off the probe.

`package_version` is another matter. Changing it moves **every** hash:

```
SPY / autoarima, as stored          716738f4...4ec54c3
the same config at version 0.6.1    2ea6d752...0678f3c   moved
the same config at version 0.7.0    64561bf1...bc3d87c   moved
```

Instrumenting eight adapters is a public behaviour change to
`volbench.models`, which under this project's conventions ships as a release.
That release invalidates all 143 cells at once. The condition "no config-hash
change" is satisfiable only by shipping the change without a version bump,
which would make `package_version` a lie about which code produced a number —
the exact property D-032 was written to protect.

### 4.2 Even at a frozen version, the existing fragments never gain the column

The store is append-only and `run_grid` short-circuits on
`ResultsStore.has(config_hash)` — a file-existence check, before any fitting.
So a re-run after wiring would report `computed 0, cached 143` and leave every
fragment byte-identical, `fit_status = ""` included. This is not hypothetical:
the resumability re-run in docs/P3_DRIVER_PROVENANCE.md §5 did exactly that,
with all the new code in the tree, and all 286 files came back byte-identical
**and unrewritten**.

The instrumented column would reach the store only by recomputing the cells —
which is the trade the instruction rules out, and rightly: re-running a clean,
gate-passed 70-minute grid to add a diagnostic buys a column and risks the
numbers.

**So: not wired.** What follows is what the wiring would be.

### 4.3 What the wiring would be

Each adapter's *fitted* class grows a `fit_diagnostics()` returning
`FitDiagnostics(converged=..., fallback=..., detail=...)`; nothing else changes,
because `evaluate._fit_status` already calls it through the
`SupportsFitDiagnostics` protocol on every scheduled fit and writes the result
to the existing column.

| config | `converged` | `fallback` | `detail` | change needed beyond the method |
|---|---|---|---|---|
| `har` | `rank == 4` | `""` | `rank=<r> cond=<c:.3g>` | `har.fit` must keep `rank`/`singular` from `np.linalg.lstsq` instead of `[0]` |
| `autoets` | `isfinite(model_["loglik"])` | `""` | `method=<ETS form> aicc=<a:.6g>` | none — read from `backend.model_` |
| `autoarima` | `model_["code"] == 0` | `""` | `code=<c> order=<arma>` | none — read from `backend.model_` |
| `lgbm` | `num_trees() == num_boost_round` | `""` | `trees=<n>` | none — read from `booster` |
| `patchtst` | `stopped_early or epochs_run < max_epochs` | `""` | `epochs=<n> best=<b> val=<v:.4g>` | none — already on `FittedPatchTST.training` |
| `chronos`, `timesfm`, `moirai` | **does not fit this column** — see below | | | |
| `naive`, `ewma` | **do not implement it** — `""` stays correct | | | |

§3's measurements change what each of those columns would be worth, and that is
worth saying before anyone builds them:

- `autoarima`'s would carry information immediately — `code != 0` on 98.6 % of
  fits, varying by asset from 97.4 % to 100 %.
- `har`'s and `lgbm`'s would be constant (`ok` everywhere) on this grid. Still
  worth having: a constant that has been checked is not the same artifact as a
  blank.
- `patchtst`'s `converged` as written above would be `True` on every fit, since
  early stopping fired on all 241 sampled and none reached `max_epochs`. Its
  information is entirely in `detail` — the epoch counts — not in the boolean.
- `autoets` has no convergence axis at all; only `method` would be recorded.

Two things the wiring would force a decision on, and neither is this stream's
to make:

1. **The vocabulary would have to widen.** `FitDiagnostics.status()` offers
   `ok` / `nonconverged` / `fallback=<name>`. A rank-deficient OLS did not
   "fail to converge" — it has no optimizer — and neither did a booster that
   built 100 of 100 trees. Forcing them into `nonconverged` would make the
   token mean two different things in one column. `status()`'s own docstring
   says the vocabulary is small and stable *on purpose* and that widening it is
   a schema change.
2. **The three zero-shot TSFMs do not belong in this column at all.** Nothing
   is estimated, so `""` is as correct for them as it is for `naive`. Their
   signal — quantile crossings rearranged, quantiles clipped at zero — is a
   property of a *forecast*, produced once per origin, not once per scheduled
   fit, and `fit_status` is documented as a per-fit property that `update`
   carries forward unchanged. Recording it in `fit_status` would break that
   invariant. It needs its own per-row column, and §3.5 shows it would not be
   empty: `chronos` clipped a negative RV quantile at 14 of 241 sampled
   origins.

## 5. The rule for every table from here on

The manifest's `fallback_rate` is already `nan` rather than `0` for the ten
(`CellOutcome.fallback_rate` divides by `n_fits` and returns `nan` at zero
denominator — deliberate, and the right choice). What is missing is that `nan`
and `0` look alike in a rendered table and `nan` reads as "missing data" rather
than "not measured".

So, for every fallback-rate or convergence table produced from this grid:

| configs | reads |
|---|---|
| `garch11`, `garch11_t`, `gjr` | the measured rate |
| `naive`, `ewma` | `not instrumented` (nothing is estimated) |
| `har`, `autoets`, `autoarima`, `lgbm`, `chronos`, `timesfm`, `moirai`, `patchtst` | `not instrumented` |

Never `nan`. Never `0`. The two groups of "not instrumented" are footnoted
differently — for `naive`/`ewma` there is nothing to instrument; for the other
eight there is, and §2 says what.
