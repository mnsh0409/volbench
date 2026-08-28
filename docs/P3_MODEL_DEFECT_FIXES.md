# P3 — the two model-side defects K measured, fixed

> Branch `fix/p3-model-defects`, off `feat/p3-analysis` at `1b1b89f`. It fixes
> exactly the two defects `docs/P3_TSFM_VARIANCE_AUDIT.md` and
> `docs/P3_LGBM_SMEARING_AUDIT.md` measured, re-runs exactly the cells those
> fixes invalidate, and re-certifies the evidence that stood on the code they
> replaced. **It changes numbers**, which is why it is on its own branch.
>
> No score is interpreted here. Whether any model now looks better or worse is
> J2's and J3's question, and this document deliberately does not answer it.

---

## 0. What shipped

| | |
|---|---|
| `src/volbench/models/lgbm.py` | `out_of_fold_residuals`; `smearing_residuals` (default `"out_of_fold"`) and `oof_folds` as hashed `spec()` fields |
| `src/volbench/models/tsfm_common.py` | `tail_closed_grid_mean`, `grid_mean_under_closures`, `CLOSURES`, `VARIANCE_FROM`; `predict` closes the grid's tails lognormally; `spec()`'s `variance_from` renamed to match |
| `src/volbench/benchmarks/defect_tables.py` (new) | K's per-asset tables as committed code, plus the acceptance test for Fix 1 |
| `src/volbench/benchmarks/data_digests.py` (new) | writes and `--check`s `docs/P3_DATA_DIGESTS.json` |
| `src/volbench/determinism.py` | `interpreter_info()`; the run manifest now records the interpreter (reported, never hashed) |
| `src/volbench/benchmarks/lgbm_smearing_probe.py` | imports the adapter's out-of-fold construction instead of restating it; new `smear_shipped` column |
| `src/volbench/benchmarks/tsfm_distribution_probe.py` | imports the adapter's tail closure instead of restating it; new `flat_tail_mean` / `tail_closure` columns |
| tests | `TestOutOfFoldFoldsAreCausal` (the canary, ported into the adapter's suite), `TestTailClosure`, `tests/test_defect_tables.py`, `tests/test_data_digests.py` |
| docs | `docs/P3_DATA_DIGESTS.json` (new), this file |

**`package_version` was deliberately not bumped.** Bumping it would move all 143
config hashes to fix 44, and the 99 unaffected cells' numbers did not change.
Both fixes are named in the `spec()` of the models they change instead, so
exactly the cells whose numbers moved lost their cache entries. §3 measures
that rather than asserting it.

---

## 1. Fix 1 — LightGBM's smearing factor is now out-of-fold

**What was wrong.** `models/lgbm.py` estimated Duan's (1983) retransformation
factor `mean(exp(e))` from the booster's own **in-sample** residuals. A boosted
ensemble shrinks those, which drives the factor toward 1 and quietly turns the
variance forecast back into a median forecast. On the panel the shipped factor
was **1.371** at the median where the grid's own realized one-step errors imply
**1.678** (`docs/P3_LGBM_SMEARING_AUDIT.md` §2) — a one-directional
understatement on all eleven assets, 5.57 against 1.37 inside the COVID window.

**What changed.** `LightGBMRV.smearing_residuals`, a hashed `spec()` field,
defaults to `"out_of_fold"`. The construction is K's, moved out of the probe
and into the adapter (`out_of_fold_residuals`): the window's design rows are
cut into `oof_folds = 5` contiguous **chronological** blocks and block *k* is
predicted by a booster trained on blocks `0..k-1` only. On a 500-observation
window that is 478 design rows, fold edges `[0, 95, 191, 286, 382, 478]`, and
**383 out-of-fold residuals per fit** — the same geometry §0.1 of the audit
describes. The old construction remains as `smearing_residuals="in_sample"`,
because it is what the audit measured and what the capacity guard guards.

Three things were deliberately **not** changed:

- **The round cap stays at 100.** The audit's ladder shows it is binding on the
  training loss (MSE 0.78 → 0.24 from 25 to 800 rounds, no plateau) and that
  the in-sample factor falls with it. Raising it would change the model rather
  than fix the estimator. It is recorded as a known limitation in the module
  docstring and in `tests/test_lgbm_smearing_probe.py::TestRoundLadder`.
- **`resid_var` stays in-sample.** It feeds the `gaussian` arm, whose whole
  purpose is to be HAR's estimator on this model's residuals; making it
  out-of-fold would end the like-for-like comparison.
- **`patchtst` is untouched**, though it carries the same construction
  (`docs/P3_TSFM_VARIANCE_AUDIT.md` §1.4). It was never measured, and a fix
  scoped to `lgbm` leaves it in place.

### 1.1 The canary, ported into the adapter's own suite

K's probe carried a corrupt-the-future canary for the folds. A probe being
leakage-clean is not evidence that the adapter is, so it moved with the
construction: `tests/test_models_lgbm.py::TestOutOfFoldFoldsAreCausal` now runs
it against `models/lgbm.py` itself — replace every design row from the second
fold boundary onward with noise and the residuals of the fold *before* it must
be bit-identical, while the later folds must react. Four more claims travel with
it: the same statement through `fit` (a window's factor cannot depend on what
follows the window), the causal-boundary argument stated arithmetically, that a
too-short window raises rather than falling through to a factor `spec()` does
not name, and that the folds are deterministic.

The full raw-CSV canary is §5.

### 1.2 Acceptance: did the fix land on what it estimates?

Recomputed after the re-run by `volbench.benchmarks.defect_tables`, from the
re-run store and a re-run of the probe at all **2,366** refit origins (0 probe
errors). `mu_hat` is inverted out of the stored variance exactly —
`log(forecast_var) - log(smear_shipped[fit_origin])`, constant within a refit
block because `update` re-conditions without re-estimating — and the realized
factor is Duan's formula over `log(proxy_var) - mu_hat` at the cell's own
scored origins.

| asset | fits | in-sample | out-of-fold | **shipped** | realized | **shipped / realized** |
|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 133 | 1.300 | 1.347 | **1.347** | 1.391 | 0.969 |
| CAC | 240 | 1.362 | 1.719 | **1.719** | 1.678 | 1.024 |
| DAX | 238 | 1.322 | 1.702 | **1.702** | 1.686 | 1.010 |
| DIA | 234 | 1.367 | 1.846 | **1.846** | 1.724 | 1.071 |
| ETH-USD | 133 | 1.246 | 1.389 | **1.389** | 1.328 | 1.046 |
| HSI | 230 | 1.375 | 1.724 | **1.724** | 1.620 | 1.064 |
| KOSPI | 231 | 1.372 | 1.675 | **1.675** | 1.671 | 1.002 |
| NDX | 236 | 1.371 | 1.703 | **1.703** | 1.681 | 1.013 |
| NKX | 228 | 1.386 | 1.643 | **1.643** | 1.598 | 1.028 |
| SPY | 234 | 1.390 | 1.923 | **1.923** | 1.777 | 1.082 |
| TWSE | 229 | 1.466 | 1.824 | **1.824** | 1.765 | 1.033 |
| **panel median** | | **1.371** | **1.703** | **1.703** | **1.678** | **1.028** |

**`shipped` equals `out_of_fold` on every asset**, which is the check that the
adapter is running the construction the probe measures rather than something
adjacent to it. The panel medians reproduce the audit's table exactly (1.371 /
1.703 / 1.678).

**Verdict: PASS.** The stated tolerances are **5 % at the panel median** and
**15 % per asset** (`defect_tables.PANEL_TOLERANCE` / `PER_ASSET_TOLERANCE`).
The panel median of the per-asset ratios is **1.0284**; the ratio of the panel
medians is 1.703 / 1.678 = **1.0149**. The worst asset is SPY at **1.082**. For
contrast, the pre-fix state fails the same test on both legs, which
`tests/test_defect_tables.py::TestAcceptanceCanFail` pins with the audit's own
in-sample numbers — an acceptance test that passed on those would be testing
nothing.

The residual overshoot is expected and has a known sign: the folds train on
20–80 % of the window rather than all of it, so the estimate is mildly
pessimistic. §1 of the audit predicted the direction and it holds on 9 of 11
assets.

---

## 2. Fix 2 — the TSFM grid mean is closed lognormally

**What was wrong.** The three quantile TSFMs emit a 9-level 0.1…0.9 grid over
realized variance, and the adapter took its mean with **flat tails** — 20 % of
the probability mass in two point atoms at q₀.₁ and q₀.₉. The audit's diagnosis
is that the *mean* is the right functional (QLIKE and MSE are both minimized at
the conditional mean) and the *closure* is the bug: on a right-skewed RV law
flat tails understate, by 11–21 % at the panel median, one-directional on every
asset and every config, moving VaR and ES by exactly the square root of that.

**What changed.** `FittedTSFM.predict` now takes the mean under a lognormal
tail closure (`tail_closed_grid_mean`). The grid's interior is left exactly as
the checkpoint emitted it; only the two atoms are re-expressed, by fitting
`log q_tau = mu + sigma Phi^{-1}(tau)` to the whole grid by OLS and using the
lognormal's closed-form partial expectations.

**Why lognormal, and not because it is the middle number.** Realized volatility
is approximately lognormally distributed — one of the better-established
stylized facts in the realized-volatility literature (Andersen, Bollerslev,
Diebold & Labys 2001, *JASA* 96(453) 42-55; 2003, *Econometrica* 71(2) 579-625).
It is the same fact every log-RV model in this package already rests on:
`models/har.py`, `models/lgbm.py` and `models/sf.py` all work in logs for it. A
lognormal tail on an RV quantile grid is therefore the closure with literature
behind it. The citation is in the module docstring so the choice stays legible.

**Where the closure cannot be fitted, the flat tails stand.** A lognormal
cannot describe a grid holding a zero — a clipped `chronos`/`moirai` quantile or
a package-floored `timesfm` one, which the audit counted at 119 / 215 / 12 of
2,199 sampled origins. Those origins keep the flat-tailed mean and record
`tail_closure: "flat"` in the fitted `spec()`, so the fallback is countable
rather than invisible. They retain the understatement; imputing a shape onto a
grid that contradicts it would be worse than reporting how often it happens.
§2.2 reports the realized rate.

**Out of scope, deliberately.** The collapse of the RV distribution to a point
variance is *not* changed here. It is a disclosed design decision with K's
measurements behind it (`docs/P3_METRIC_TARGETS.md` §4.3), and a mixture return
law may be added later as a separate arm.

### 2.1 What the closure moved, per config per asset

`volbench.benchmarks.defect_tables.tsfm_closure_table`, over the re-run
distribution probe: 200 origins per cell evenly spaced across each cell's whole
evaluation span, 11 assets × 3 configs = **6,597 scorable origins** (6,600 minus
NKX's three known `InsufficientHistoryError` origins). Medians of each closure's
mean over the flat-tailed reading that shipped before.

| config | `lognormal` | `loglinear` | `empirical` | **scored / flat** | flat fallbacks | **VaR & ES shift** |
|---|---:|---:|---:|---:|---:|---:|
| `chronos` | 1.135 | 1.102 | 1.174 | **1.129** | 119 / 2,199 | **+6.3 %** |
| `timesfm` | 1.201 | 1.090 | 1.173 | **1.181** | 215 / 2,199 | **+8.7 %** |
| `moirai` | 1.111 | 1.097 | 1.175 | **1.111** | 12 / 2,199 | **+5.4 %** |

The `lognormal` and `loglinear` columns reproduce
`docs/P3_TSFM_VARIANCE_AUDIT.md` §3.2 to the third decimal (1.135 / 1.102,
1.201 / 1.091, 1.111 / 1.097), and the fallback counts are exactly the origins
that audit had to exclude — which is the check that this is the same closure it
measured, now running inside the adapter.

**`scored / flat` is the column that describes the store**, and it is below the
`lognormal` column for `chronos` and `timesfm` because the origins that kept
their flat tails are *included* in it rather than dropped. The VaR/ES shift is
exactly its square root: the scored object is `Normal(0, sqrt(v))`, so VaR and
ES are homogeneous of degree 1 in `sqrt(v)` and every level moves by the same
factor with no approximation.

Per asset the two crypto series move least (1.03–1.06) and NKX/TWSE most
(1.14–1.26), the same ordering and the same mechanism the audit reports: a
narrower grid strands less distance in the two atoms.

### 2.2 The empirical closure, re-derived — and it does not match to three decimals

The audit's third closure replaces the atoms with what the realizations actually
did beyond q₀.₁/q₀.₉. It is a **diagnostic and never a candidate**: it reads
realizations after the origin it would correct, so no forecast-time code could
compute it. It is kept because the tail is genuinely unidentified and a single
number would overstate what the grid supports.

Its construction was in a scratchpad script and is now committed
(`defect_tables._empirical_ratio`), and the two do not agree exactly: the
committed version reads **1.173–1.175** where the audit reported **1.205–1.212**.
The difference is the construction, not the data. Each origin's grid has its own
scale, so the committed version pools the exceedances **as ratios** —
`mean(RV / q)` over the origins where `RV` fell outside `q`, applied back to
each origin's own outer quantile — where the audit pooled conditional means
across origins of differing scale. Both land in the same place relative to the
question being asked (above the lognormal reading, below the crudest
`mean(realized RV / v̂)` check at 1.25–1.30), and only one of them can be re-run.
The committed number is the one this branch reports.

**The range the paper can state**, from
`volbench.models.tsfm_common.grid_mean_under_closures` on demand, is therefore
**1.09× (log-linear) to 1.20× (lognormal, `timesfm`)** among the implementable
closures, with an assumption-free diagnostic at ~1.17×. The shipped closure is
not the middle of that range by construction — it is the one the RV literature
supports (§2).

---

## 3. The re-run, and the cell arithmetic

Before running anything, every one of the 143 stored sidecars was re-hashed with
the *current* model spec substituted in — nothing fitted, nothing written — to
establish which identities move and to catch the cache trap
`docs/P3_LGBM_SMEARING_AUDIT.md` §5 flagged. All 143 sidecars re-hash to their
own filenames, so the substitution is measuring what it claims to.

| | cells | which |
|---|---:|---|
| hashes that **move** | **44** | `lgbm` × 11 (CPU), `chronos` / `timesfm` / `moirai` × 11 each (GPU) |
| hashes that **stay** | **99** | `naive`, `ewma`, `garch11`, `garch11_t`, `gjr`, `har`, `autoets`, `autoarima`, `patchtst` × 11 each |

Then the run itself:

```
$ NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 uv run --extra classical --extra tsfm python -m \
      volbench.benchmarks.grid_primary --tag fix --device cuda

cells attempted 143  computed 44  cached 99  failed 0
wall clock 51.2 min   peak RSS 1.00 GiB
```

**44 computed, 99 cached, 0 failed** — the required arithmetic, and it matches
the pre-flight cell for cell.

Cell time, against the audits' estimates: `lgbm` **1.36 min** for 11 cells
(against 0.77 min estimated — the out-of-fold folds train four extra boosters
per refit, on 20–80 % of the window each, so roughly doubling a cell rather than
quintupling it); `chronos` 9.39, `moirai` 7.22, `timesfm` 34.43 = **51.0
GPU-minutes** against the 41.05 estimated.

**The missing-reason census is byte-identical to the primary run's** — NKX's 21
`InsufficientHistoryError` rows on the seven variance-fed configs plus one
`proxy_nonpositive`, TWSE's 80 `proxy_nan` on all 13, CAC's 28, HSI's 13 —
diffed line for line against `data/grid_primary/report_primary.txt`. The fixes
changed what a forecast *is*, not which origins are scorable.

The manifest is at `data/grid_primary/manifest_fix.json`, the report at
`data/grid_primary/report_fix.txt`. The 44 superseded fragments are still in the
store under their old hashes: the store is append-only and nothing was
overwritten, which is what makes the two hash spaces distinguishable rather than
merged.

---

## 4. Re-certifying what the fixes invalidated

`docs/P3_LEAKAGE_CANARY_EXT.md` certified `lgbm`, `chronos`, `timesfm` and
`moirai` against the code this branch replaced. All four code paths changed, so
the evidence had to follow the code: the same harness, the same three legs, the
same cutoff, re-run against the fixed adapters.

```
$ uv run --extra classical --extra tsfm python -m volbench.benchmarks.leakage_canary \
      --models lgbm chronos timesfm moirai --work-root data/leakage_canary_fix --device cuda

SPY: 5405 windowed bars, 5404 after the driver's leading trim
  cutoff target_index 560 -> windowed position 561 -> 2007-05-21
  rows compared: target_index in [500, 560]     (61 rows)
  corrupted bars: future leg 4843, past leg 22

  lgbm       future-corruption: identical    past-corruption: differs (canary alive)
  chronos    future-corruption: identical    past-corruption: differs (canary alive)
  timesfm    future-corruption: identical    past-corruption: differs (canary alive)
  moirai     future-corruption: identical    past-corruption: differs (canary alive)

VERDICT: PASS
```

**No config reported "past-corruption: identical"**, so every "identical" above
is a test that could have failed and did not. The determinism leg — clean
against clean, into two different stores so neither read is a cache hit — was
bit-identical on all four; the verdict line reports it inline only on failure,
and none was reported.

The out-of-fold folds are the new thing in that verdict, and they are canaried
twice over: once here through the raw CSV and the whole production path, and
once directly against `out_of_fold_residuals` in the adapter's unit suite
(§1.1). The first proves the folds cannot reach data outside the training
window; the second proves they cannot reach data outside their own block
*inside* it, which the raw-CSV canary cannot see.

---

## 5. The determinism gates, re-run

### 5.1 Serial versus parallel, byte-identical

`docs/P3_RUNNER.md` §4's gate against the fixed adapters: SPY × the four changed
configs, one grid through two execution backends into two separate stores,
compared on the parquet **bytes**. One arm per process, as §4 did, so the
serial arm's in-process CUDA context cannot reach the pool.

| config | fragment | serial vs parallel |
|---|---|---|
| `chronos` | `669c2da451e9` · 812,995 bytes | **identical** |
| `lgbm` | `328a8de6c093` · 809,889 bytes | **identical** |
| `moirai` | `68184c010a9b` · 813,066 bytes | **identical** |
| `timesfm` | `b2fd22d3386e` · 812,975 bytes | **identical** |

```
serial   (SerialExecutor  / SerialExecutor)          252.3 s   fingerprint 61b1270685200732
parallel (ProcessExecutor(4) / ProcessExecutor(1))   256.8 s   fingerprint 61b1270685200732
```

**One fingerprint across both backends.** The out-of-fold folds train four extra
boosters per refit, which is four more chances for a thread count or an RNG to
reorder a reduction; under the D-032 pin they do not.
`tests/test_runner.py::TestSerialParallelIdentity` — the same claim on a toy
grid, with its inert-proof companion — is green in the suite.

### 5.2 Resumability: nothing recomputed, nothing rewritten

```
$ uv run --extra classical --extra tsfm python -m volbench.benchmarks.grid_primary \
      --tag resume_after_fix --device cuda

cells attempted 143  computed 0  cached 143  failed 0
wall clock 0.1 min   peak RSS 1.01 GiB
```

**143 cached, 0 computed, 0 failed**, and "cached" alone only proves the
short-circuit fired, so:

| check | before the re-run | after |
|---|---|---|
| files in `data/grid_primary/store` | 374 | 374 |
| `md5sum data/grid_primary/store/* \| md5sum` | `40453fb39abc878455dc7f3d237208dc` | `40453fb39abc878455dc7f3d237208dc` |
| files with an mtime inside the re-run | — | **0** |
| `git status --short data/` | empty | empty |

Every fragment and every sidecar is byte-identical **and unrewritten**. 374
files is 187 pairs: the 143 current cells plus the 44 superseded ones, which the
append-only store keeps under their old hashes.

### 5.3 The interpreter is now in the manifest

`RunManifest.environment` records it, reported and never hashed (`interpreter`
is in `environment_report`, not `environment_spec` — the version is not known to
move a number here, and hashing it would split the cache three ways for a claim
nothing has measured):

```json
"interpreter": {
  "python": "3.11.5",
  "implementation": "CPython",
  "executable": ".../volbench/.venv/bin/python"
}
```

`docs/P3_ANALYSIS_VALIDITY.md` had to establish 3.11.5 three indirect ways
because the run recorded it nowhere. It now records it directly.

---

## 6. Folded in while here

Three things that change no number.

**The data digests are committed.** `docs/P3_DATA_DIGESTS.json`, written and
checked by `volbench.benchmarks.data_digests`. Every config hash in the study is
built over `series_sha256`, `fit_series_sha256`, `proxy.sha256` and
`raw_sha256`, and those existed only in the store's gitignored sidecars — so a
clean checkout could *run* the study but could not check it had rebuilt the
right inputs, and a silently different archive would look like an empty store
rather than an error. The manifest was verified against **all 143 stored
sidecars**: 0 mismatches on `series_sha256`, `raw_sha256`, `n`, `proxy`, and on
`fit_series_sha256` for every variance-fed cell. It records digests, never data,
so `tests/test_licensing_guard.py::TestNoDataIsTracked` is untouched.
`--check` rebuilds the panel and names the asset and the digest that moved.

**The per-asset aggregation is committed.** `volbench.benchmarks.defect_tables`
produces §1.2's and §2.1's tables and the acceptance verdict from the store and
the two probe parquets. K's tables came from scratchpad scripts: re-derivable,
but not re-runnable, which is exactly the property that matters once the code
under them moves — as it just did.

**The interpreter is in the run manifest** (§5.3).

---

## 7. Known limitations, unchanged and recorded rather than fixed

1. **`lgbm`'s 100-round cap is binding on the training loss** and was not
   raised. Training MSE falls monotonically 0.78 → 0.24 from 25 to 800 rounds
   with no plateau, and the *in-sample* factor falls with it, 1.49 → 1.13
   (`docs/P3_LGBM_SMEARING_AUDIT.md` §4). Raising it would change the model
   rather than fix the estimator, and the out-of-fold factor lands within 1.5 %
   of the realized one at the current capacity, so the defect is closed without
   touching capacity. Whether more rounds would improve the *forecast* is a
   different question this branch does not open: the ladder measures training
   loss, not validation loss.
2. **`patchtst` still computes its smearing factor from in-sample training
   residuals** (`docs/P3_TSFM_VARIANCE_AUDIT.md` §1.4). It was never measured,
   and this branch's scope is the two defects that were.
3. **~5–10 % of `chronos` / `timesfm` origins keep flat tails**, because their
   grid holds a zero the lognormal cannot describe (§2.1). Those forecasts
   retain the understatement, and the count is now recorded per origin in the
   fitted `spec()` rather than inferred.
4. **The RV distribution is still collapsed to a point variance.** Out of scope
   by instruction; a mixture return law would be a separate arm.
5. **The `empirical` closure's committed re-derivation reads 1.17 where the
   audit's scratchpad script read 1.21** (§2.2). Different pooling of the
   exceedances, same conclusion; the committed one is the re-runnable one.

---

## 8. Drift to reconcile on the planning machine

`docs/design.md` and `docs/decisions.md` are read-only mirrors here (CLAUDE.md),
and this branch changes things both of them describe. Flagged, not edited:

- **`docs/design.md`** describes `LightGBMRV`'s and the TSFM adapters' public
  surface. Three new hashed `spec()` fields (`smearing_residuals`, `oof_folds`,
  and `variance_from`'s new value), three new public functions
  (`out_of_fold_residuals`, `tail_closed_grid_mean`, `grid_mean_under_closures`)
  and two new benchmark modules are as-built and not yet in that document.
- **`docs/decisions.md`** — the retransformation arm (D-014's family) and the
  TSFM variance derivation are settled decisions whose implementations moved.
  Both fixes were instructed by prompt rather than by an appended decision, so a
  decision record for each is owed: which residuals the smearing factor reads,
  and which tail closure the grid mean uses.
- **`docs/P3_LEAKAGE_CANARY_EXT.md`** records verdicts for thirteen configs
  against code four of which no longer exists. §4 above supersedes those four
  rows; the other nine stand.
- **`docs/P3_GRID.md` and `docs/P3_GRID_manifest.json`** describe the pre-fix
  grid. 44 of their 143 cells now have different hashes and different numbers;
  `data/grid_primary/manifest_fix.json` is the current one.
