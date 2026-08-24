# 04 · volbench API design

> Status: **AS-BUILT at M1 (v0.1.0-m1)**. This file described the plan until
> the three Phase 1 streams landed; it now describes what exists, with every
> place the build diverged from the plan called out inline under
> **Diverged:**. The planning-folder copy is now behind this one and needs
> re-syncing from here, not the other way round.

## Components

### Data — `volbench.data`

- **`TimeSeriesFrame`** (`data/types.py`) — frozen container: a UTC
  `DatetimeIndex`, `close` or the full `open/high/low/close` set, `asset_id`,
  `source`. Validates tz-awareness, strict monotonicity, no duplicate
  timestamps, no NaN in required columns; copies its input defensively.
  Deliberately never reindexes two assets onto a shared calendar — that
  belongs to whatever consumes several frames.

  **Diverged:** the plan said "values + trading-calendar-aware index + asset
  metadata". As built it is a *price* container (OHLC), not a general values
  container, and it holds no trading-calendar object — only the timestamps the
  source actually delivered. Nothing in the library knows a market's session
  schedule yet.

- **`DataAdapter`** — **not built as an abstraction.** The plan called for a
  `fetch(spec) -> TimeSeriesFrame` protocol with per-source modules behind it.
  As built there are per-source *modules* with source-shaped signatures and no
  common protocol:
  - `data/stooq.py` — `download_index(asset_id, ...)`, `ingest_manual_csv(...)`,
    `fetch_stooq_csv`, `parse_stooq_csv`, `STOOQ_INDEX_SYMBOLS`.
  - `data/crypto.py` — `daily_realized_variance(asset_id, start, end, ...)`,
    `load_minute_bars`, `fetch_and_cache_day`, `CRYPTO_SYMBOLS`.
  - `data/byo.py` — `load_ohlc_csv`, `load_ohlc_parquet`.

  **Diverged:** no `DataAdapter` protocol exists, and the licence flag the plan
  wanted "enforced at packaging time" is documented prose in
  `docs/data_licenses.md`, not machine-readable metadata. Nothing enforces it
  in code. A `spec`-driven `fetch` is still the right shape for Phase 2 — the
  three signatures have little in common today.

- **Proxies** (`data/proxies.py`) — pure functions, daily units, no hidden
  state: `squared_return`, `parkinson`, `garman_klass`,
  `realized_variance_from_bars`, `log_returns`, and
  `overnight_plus_range_variance`.

  **Diverged (added at M1):** `log_returns` was not in the plan. The data layer
  exposed only `r^2`, but the models and the evaluator both speak in *signed*
  returns, so every caller was re-deriving them by hand. It returns a leading
  NaN so the output stays index-aligned with its input.

  **Added at M2 (report §4.4, D-016):** `overnight_plus_range_variance` =
  `(ln(O_t/C_{t-1}))^2 + RS_t`, the per-day CLOSE-TO-CLOSE variance estimator
  (Rogers & Satchell 1991 range term plus the squared overnight jump).
  Deliberately not Yang-Zhang, which is windowed and would reach past day `t`.
  It is HAR's scoring target, since a range proxy alone omits the overnight
  variance HAR's forecast is scored against. First observation NaN (no
  `C_{t-1}`).

### Forecasting — `volbench.dist`, `volbench.models`

- **`Distribution`** (`dist.py`) — the only forecast currency. Constructors
  `from_normal`, `from_student_t`, `from_samples`, `from_quantiles`; concrete
  `Normal`, `StudentT`, `Empirical`, `QuantileGrid`. Methods: `quantile`,
  `cdf`, `crps`, `log_score`, `pinball`, `sample`, and — parametric families
  only — `mean`/`variance` in closed form.

  **Added on `m2/evaluator-hardening`:** `StudentT(loc, scale, df)`, the
  location-scale t with closed-form moments and CRPS (Jordan, Krüger & Lerch
  2019), cdf/quantile via `scipy.stats.t`. Requires `df > 1`; `variance()`
  raises for `df <= 2`. `StudentT.from_variance(loc, v, df)` builds it from a
  target variance so `variance()` round-trips exactly. `mean()`/`variance()`
  joined the base interface (default `NotImplementedError`, like
  `log_score`) so the evaluator can ask the object before estimating.

  **Diverged:** the plan named the constructor `from_params` and the method
  `logscore`. As built they are `from_normal` and `log_score`. `log_score`
  raises `NotImplementedError` for distributions with no tractable density
  (`Empirical`, `QuantileGrid`) rather than returning a sentinel; the evaluator
  catches that and records `log_score_undefined`.

- **`ForecastModel` / `FittedModel`** (`models/base.py`) — **the single
  definition of the model interface.** Both are `@runtime_checkable` Protocols;
  concrete models are plain frozen dataclasses that satisfy them structurally,
  with no shared base class.

  ```python
  class ForecastModel(Protocol):
      @property
      def name(self) -> str: ...
      def spec(self) -> dict[str, Any]: ...
      def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedModel: ...

  class FittedModel(Protocol):
      @property
      def name(self) -> str: ...
      def spec(self) -> dict[str, Any]: ...
      def predict(self, h: int) -> Distribution: ...
  ```

  `fit` takes a plain 1-D array, never a `TimeSeriesFrame`: `volbench.models`
  has no dependency on `volbench.data`. `predict(h)` returns a `Distribution`
  over the **next-period return**; its variance is the variance forecast.

  **Diverged:** the plan wrote `fit(train: TimeSeriesFrame) -> FittedModel`. As
  built the adapter boundary is a bare array, which is why the evaluator can
  hand a model a *variance* series (HAR) through the same interface it hands
  another a *return* series.

  **Diverged:** during Phase 1 `evaluate.py` carried a second, local copy of
  these Protocols because `volbench.models` was being built in parallel. That
  copy is gone. `volbench.evaluate.ForecastModel` is now the same object as
  `volbench.models.base.ForecastModel`, and
  `tests/test_model_interface.py` fails — statically under mypy and at runtime
  — if the two ever diverge again.

- **Baselines** (`models/`): `NaiveVol`, `EWMA`, `GARCH` (with `gjr_garch`),
  `HAR`. Every one returns `Normal(mu=0, sigma=...)` except Student-t GARCH,
  which returns a parametric `StudentT` built with `from_variance` from arch's
  conditional variance (no RNG, so still bit-identical across runs). Until
  `m2/evaluator-hardening` it returned a 199-point `QuantileGrid`, which is
  what produced the QLIKE floor in M1 report §4.2 — see the resolved open
  question below.

  **Diverged:** `HAR.fit` takes a realized-**variance** series, not returns.
  This is documented in its module and handled by `run_backtest(fit_series=...)`,
  but it means "the model interface" is uniform in *type* and not in *meaning*
  — nothing in the type system distinguishes a returns array from a variance
  array. See `docs/M1_REPORT.md` risk 2.

  **Open (found at M2, docs/M2_NOTES.md):** HAR's lognormal retransformation
  `E[RV]=exp(ŷ+½·resid_var)` is sensitive to the target's log-space noise. On
  the toy fixture, feeding HAR the (correct, noisier) overnight-plus-range
  target inflates its forecast ~13% above the true variance, where the
  intraday Parkinson target happened to leave it well-calibrated. A bias-
  corrected or component overnight+intraday HAR is the Phase-2 fix.

  **Diverged:** `GARCH.fit` never raises on optimizer failure; it falls back to
  EWMA on the same window and records `fallback=True`. HAR, by contrast, *does*
  raise on a degenerate input (non-positive RV, short window). The two models
  disagree about whether a bad origin should be survivable.

- **Zero-shot foundation models** (`models/tsfm_*.py`, added on
  `feat/p2-models-tsfm`): `Chronos` (Chronos-Bolt by default, Chronos-2 by
  checkpoint), `TimesFM` (2.5, 200M), `Moirai` (2.0-R-small) and `TimeGPT`
  (Nixtla's hosted API, opt-in). All four share one contract in
  `models/tsfm_common.py`, and the contract is the design choice to review:

  - Like HAR, `fit` takes a **realized-variance** series — the trailing
    `context_length` observations become the model's context; nothing is
    estimated (D-005: zero-shot only, no fine-tuning path exists).
  - `predict(h)` takes the model's own predictive distribution of RV at
    `t+h` — a quantile grid at the levels the checkpoint was trained on —
    and uses its **mean** (the mean of the interpolated grid, flat tails —
    the same estimator `forecast_moments` applies to a `QuantileGrid`, pinned
    equal by test) as the variance forecast, emitted as `Normal(0, sqrt(vhat))`
    over the return, the shape HAR emits. The grid itself, the model's native
    point head where one exists, and the crossing/clipping counts are kept in
    the fitted `spec()` under `rv_forecasts`; none of it is scored.
  - `update` is context extension, exact by construction; `refit_every` is
    irrelevant to these models (every origin is one forward pass either way).
  - `input_scale` (default `1e4`, variance in percent-squared) is a fixed unit
    convention in `spec()`, forced by Moirai-2's scaler epsilon (`sqrt(var +
    1e-5)`), which flattens a ~1e-4-level series into a constant.
  - `spec()` carries checkpoint id, the resolved commit hash of the weights,
    dtype, and package versions, so the config hash moves with the weights.
    Chronos-Bolt/2, TimesFM 2.5 and Moirai-2 emit quantiles directly — no
    sampling — and the `tsfm`-marked tests pin bit-identity on the GPU.
    `device` is not in `spec()`.
  - TimeGPT is triple-gated (constructor `enabled=True`, `NIXTLA_API_KEY`,
    `@pytest.mark.timegpt`) and cannot pin remote weights; it stays out of
    the headline as the research design says.

- **PatchTST** (`models/patchtst.py`, added on `feat/p2-models-tsfm`): the
  one trained deep-learning baseline — chosen over N-BEATS on the flagged
  assumption that architectural proximity to the patch-based TSFMs is the
  more useful comparison. Small fixed channel-independent PatchTST (64-step
  lookback, 16/8 patches, d_model 32, 2 layers, ~20k parameters) on
  instance-normalized log RV; Adam/MSE under a bounded, hashed budget (max
  100 epochs, early stop on the chronologically-last 20% of windows,
  patience 10, best weights restored); Duan smearing retransformation with a
  per-horizon factor from the training residuals; output
  `Normal(0, sqrt(vhat))`. Deterministic by construction (explicit-matmul
  attention, `use_deterministic_algorithms`, seeded batches/dropout) and
  pinned bit-identical twice on CPU and GPU; across devices, dropout draws
  from each device's own RNG stream, so results reproduce per device class
  (with `dropout=0` CPU and GPU agree to ~1e-8), and `device` is not hashed. **No `update`:** re-conditioning
  a trained net without re-estimation is not well defined, so it runs frozen
  between refits and `conditioned_through == fit_origin` on every row.
  Training windows are cut from the fit array only (last target = the
  origin), which is the leakage-check focus for a dataloader.

  **Diverged:** the plan's model list names these as adapters of a generic
  kind; as built they are RV-fed, like HAR, not return-fed — the same
  type-uniform / meaning-divergent interface noted for HAR above, now shared
  by five models. Optional deps live under the `tsfm` extra and are never
  installed in CI: `tests/conftest.py` skips `tsfm`/`timegpt`-marked tests
  by default and unconditionally under `CI`, while each adapter keeps a
  mocked-backend test in the default suite.

### Splitting — `volbench.splitter`

- **`RollingOriginSplitter`** — the ONLY sanctioned producer of train/test
  indices. Parameters `window`, `horizon`, `step`, `refit_every`; yields
  `Origin(train, origin, test, refit)`, guaranteeing `max(train) == origin <
  min(test)` by construction. `n_splits(n)` reports the count without
  materializing.

  **Diverged:** the plan said "window, step, refit schedule". As built there is
  also `horizon`, and the refit schedule is the integer `refit_every` rather
  than a schedule object. Only rolling (fixed-length) windows exist; expanding
  windows are not implemented.

### Refit protocol — what "refit every N days" means

Settled after M1 report §4.3 (open at M1, implemented on
`m2/evaluator-hardening`):

- **Re-estimate every `refit_every` origins.** `fit` runs only at origins the
  splitter marks `refit=True`; the number of `fit` calls equals the number of
  refit origins, and `fit_origin` records on every row which one served it.
- **Re-condition daily in between** (`recondition="daily"`, the default). At
  every other origin the backtest calls `FittedModel.update(train)` with that
  origin's own splitter window — observations dated ≤ the origin, the exact
  array `fit` would have been handed — and the model re-filters its
  conditional state at the parameters of the last scheduled fit. `update`
  never re-estimates: GARCH/GJR re-filter through `arch`'s `ARCHModel.fix`
  (no optimizer runs; the fit's `scale` is reapplied so the parameters keep
  their units); EWMA re-runs its recursion (λ is a fixed hyperparameter); HAR
  refreshes its 22 trailing RV lags under the fitted coefficients; naive
  slides its window. `conditioned_through` records the origin on every row.
- **Frozen** (`recondition="none"`): the forecast issued at the refit origin
  is held until the next refit — exactly what every baseline did before
  `update` existed — kept as an explicit ablation arm, not a default anyone
  can fall into. `conditioned_through == fit_origin` on every row.
- **Identity.** `recondition` enters the config hash (under `protocol`)
  whenever it can change a number, i.e. whenever `refit_every > 1`. At
  `refit_every == 1` every origin refits, `update` is unreachable, and the two
  settings are the same experiment, so nothing is recorded and every hash
  computed before the key existed — the toy benchmark's included — is
  unchanged.
- **Invariant.** `update` on the fit window reproduces the fit exactly, for
  every model: re-conditioning is a no-op precisely when nothing new has been
  observed. That is what makes (b) above hold (`tests/test_model_interface.py`,
  `tests/test_models_update.py`, `tests/test_recondition.py`).
- **Frozen by design.** `PatchTST` implements no `update`; between refits
  the evaluator holds its forecast and records `conditioned_through ==
  fit_origin`. This is the documented exception to the invariant above.
- **Zero-shot models.** For the TSFM adapters `fit` and `update` are the same
  operation (record the trailing context), so `refit_every` changes no number
  — pinned in `tests/test_models_tsfm_common.py` by running the same cell at
  `refit_every=1` and `21` and comparing scores byte for byte.

### Evaluation — `volbench.evaluate`, `volbench.results`, `volbench.execute`

- **`run_backtest(model_factory, series, proxy, splitter, seed, *, asset,
  proxy_name, data_spec=None, fit_series=None, levels=DEFAULT_LEVELS,
  executor=None, store=None, overwrite=False) -> pd.DataFrame`** — scores one
  cell. Returns one tidy row per `(origin, horizon)` with the forecast's mean
  and variance, the realized return, the proxy, CRPS, log score, QLIKE, and
  pinball/VaR-quantile/hit at each level, plus `config_hash`, `seed`,
  `fit_origin`, `conditioned_through`, `refit` and `missing_reason`.
  `frame.attrs` carries `config_hash`, `config` and `cached`.

  **Diverged:** the plan had an **`Evaluator`** class consuming
  `(Distribution, realized target)` streams. As built it is a function over a
  whole cell, and DM/MCS on the score matrix is **not implemented** — the
  comparison-inference half of the plan is Phase 2.

  **Diverged:** nothing is ever dropped. An unscorable row is emitted with NaN
  and a `missing_reason` naming every cause, so a model cannot look good by
  averaging over the origins that happened to work.

  **Hardened since `m2/evaluator-hardening` (was open at M1, report §4.5):**
  that contract now covers exceptions too. A `fit`, `update`, `predict` or
  scoring exception becomes a NaN row whose `missing_reason` is
  `<stage>[@origin]: <ExceptionType>: <message>` — e.g. `fit_error@499:
  ValueError: realized-variance series must be finite and strictly positive`
  — instead of aborting the cell. A failed *scheduled* fit fails its whole
  block (there is no off-schedule refit, so the cadence the config hash
  describes stays true); a failed `update` costs one origin. Rows with no
  fitted model carry `fit_origin = conditioned_through = -1`. Only
  `Exception` is caught — `KeyboardInterrupt`/`SystemExit` propagate. Every
  failure is logged at WARNING on `volbench.evaluate`.

  **Added beyond the plan:** `SupportsUpdate`, an optional Protocol letting a
  model re-condition on newer data between scheduled refits without
  re-estimating. At M1 no model implemented it, so `refit_every > 1` froze
  every forecast between refits (M1 report §4.3, risk 1). Since
  `m2/evaluator-hardening` all four baselines implement it and the backtest
  takes `recondition="daily" | "none"` — see "Refit protocol" above.

  **Added beyond the plan:** `forecast_moments(dist) -> (mean, variance)`.
  Asks the object for `mean()`/`variance()` first (closed form: `Normal`,
  `StudentT`); only genuinely non-parametric objects fall through — plug-in
  for `Empirical`, exact-for-the-interpolant for `QuantileGrid`, quadrature
  otherwise.

- **`ResultsStore`** (`results.py`) — append-only parquet, one fragment per
  `config_hash` plus a JSON config sidecar, written through a temp file and
  `os.replace`. `has()` is a file-existence check, so a cache hit skips the
  whole run. Distinct cells write distinct paths, so backends merge by landing
  in one directory (D-011).

  **Diverged:** the config hash covers the data's *content* digest
  (`array_digest` of series, `fit_series` and proxy), not just a data label.
  That is a leakage control: a cached artifact computed from a different — for
  example later, longer, or revised — series can never be served for this one.
  Supporting API: `config_hash`, `build_config`, `canonical_repr`,
  `array_digest`, `normalize_frame`, `package_version`, `KEY_COLUMNS`,
  `REQUIRED_COLUMNS`.

- **`Executor` / `SerialExecutor`** (`execute.py`) — the execution seam. A
  cell is one `(asset, model, splitter, seed)` unit; within a cell,
  `run_backtest` further splits origins into *refit blocks* and maps those
  through the same seam. Serial only at M1; process- and Slurm-backed
  implementations are Phase 2 and belong here and nowhere else.

  **Diverged:** the plan's **`Runner`** — grid orchestration across
  (series × model × window), GPU batching — **does not exist**. `execute.py` is
  the seam it will plug into, not the orchestrator.

### Benchmarks — `volbench.benchmarks`

**Added beyond the plan.** `benchmarks/toy.py` composes all three streams over
a synthetic series: at M2, **5** baselines (naive, EWMA, GARCH, GARCH-t, HAR) ×
200 rolling origins, ~5s, byte-identical across runs. The scoring target is a
property of the run, never of a model: every cell scores QLIKE against
`overnight_plus_range_variance` (D-016), with Parkinson available as a labeled
robustness arm behind the `target` flag; HAR's *fit input* is always the
overnight-plus-range series regardless of the flag. The GARCH-t config
exercises the parametric `StudentT` path (D-014) under `make reproduce`.
`benchmarks/make_toy_asset.py` generates the input as independent overnight and
intraday components summing to a recorded `true_variance` (M2), so estimators
can be validated against the truth. `make reproduce` rebuilds both from
scratch. The series is synthetic because no licence in `docs/data_licenses.md`
permits vendoring a real one — see `docs/M1_REPORT.md`. The M1 byte-identity
baseline was superseded by the M2 fixture; old `ResultsStore` fragments keep
their M1 hashes and are never overwritten.

## Public API surface

`volbench` (root) exports the shared vocabulary and the entry points:
`Distribution`/`Normal`/`StudentT`/`Empirical`/`QuantileGrid`, `TimeSeriesFrame`, the
proxies and `log_returns`, `Origin`/`RollingOriginSplitter`, the four baseline
model classes and their fitted types, the four zero-shot adapters
(`Chronos`/`TimesFM`/`Moirai`/`TimeGPT`, fitted type `FittedTSFM`; their
shared base `ZeroShotRVModel`, `TSFMBackend` and `TimesFMForecastOptions`
stay in `volbench.models`), `PatchTST`/`FittedPatchTST`, plus
`ForecastModel`/`FittedModel`,
`run_backtest`/`forecast_moments`/`DEFAULT_LEVELS`/`SupportsUpdate`/
`ModelFactory`, `ResultsStore` and the config-hash helpers,
`Executor`/`SerialExecutor`, and `mse`/`qlike`/`pinball`.

Per-source ingestion — the Stooq and Binance downloaders, their error types and
symbol maps, and the bring-your-own-data loaders — stays in `volbench.data`.
Which source a series came from is a provenance question
(`docs/data_licenses.md`), not part of the library's vocabulary.

`__version__` is read from installed package metadata, so it cannot drift from
the `package_version()` that enters every config hash.

## Invariants (violations are bugs, not choices)
1. No code path lets information from t' > t influence a forecast for t.
   Structurally enforced by `RollingOriginSplitter`; checked end-to-end by the
   corruption canary in `tests/test_m1_smoke.py`.
2. Every model output is a `Distribution`; no bare point arrays cross module
   boundaries. Pinned by `tests/test_model_interface.py`.
3. Every result row carries seed + config hash; `make reproduce` regenerates
   the toy benchmark from scratch.
4. Scalers/feature transforms fit on train windows only, inside the splitter's
   contract.

**Enforced since `m2/evaluator-hardening` (was open at M1, report §4.6):**
`run_backtest` aligns `series`, `proxy` and `fit_series` *positionally*, so it
now requires them to be on one calendar. Pandas inputs must carry identical
indexes — values and order, checked with `Index.equals`, the first mismatching
position named in the `ValueError` — and the index must be in ascending time
order (`is_monotonic_increasing`; added on `m2/cleanup` to close the M2
leakage-audit gap, since a backwards calendar would make the positional
splitter run against time); mixing an indexed input with a bare array is
refused. Bare arrays for *every* input remain accepted as an explicit
positional opt-in (only lengths are checked), because they carry no calendar
to compare. `benchmarks/toy.py` passes indexed Series so its path exercises
the guard, and keeps its own index assertion as a redundant belt.

## Open questions (carried into Phase 2)
- [x] Closed-form vs. sample-based CRPS — settled: closed form for `Normal`,
      exact ensemble form for `Empirical`, trapezoidal pinball integral for
      `QuantileGrid`.
- [x] `forecast_moments` on a `QuantileGrid` treats the grid as the whole law
      (flat tails), so it understates the variance of a heavy-tailed forecast —
      ~8% at nu=5, ~24% at nu=3. QLIKE for Student-t GARCH was biased upward as
      a result. **Resolved on `m2/evaluator-hardening`, in `dist.py`:** a
      parametric `StudentT`; GARCH emits it; `forecast_moments` uses its
      closed-form moments. `tests/test_qlike_student_t.py` reproduces the M1
      report's floors on the old path and pins the new path below 1e-6. The
      grid's understatement is unchanged and still documented — it is now only
      reachable by objects that really are quantile grids.
- [x] `SupportsUpdate` on the econometric models so `refit_every > 1` means
      "re-estimate every 21 days, re-filter daily" rather than "freeze the
      forecast for 21 days" — **resolved on `m2/evaluator-hardening`**, see
      "Refit protocol". Still open: per-model refit-schedule overrides.
- [ ] Multi-horizon: separate `Distribution` per h, or joint object? (`horizon`
      exists in the splitter and in result rows; only h=1 is exercised.)
- [x] Should a range/RV proxy feeding HAR be reconciled with the close-to-close
      return target it is scored against? **Resolved (D-016):** HAR is fed and
      scored on `overnight_plus_range_variance`, the close-to-close estimator.
      Two follow-ups remain open (docs/M2_NOTES.md): HAR's retransformation
      sensitivity to the target's noise, and whether the return-fed models
      should also score QLIKE against the close-to-close proxy rather than
      Parkinson.
- [x] TSFM context construction — **settled on `feat/p2-models-tsfm`**: the
      context is the trailing window of the splitter's `train` indices (ends
      at the origin inclusive), capped by `context_length` and the checkpoint's
      maximum; one series per forward pass, so no cross-series padding or
      calendar alignment exists to leak through. Still open: batching several
      assets per forward pass for the Phase 3 grid (H4), which would
      reintroduce padding/alignment and must be leakage-checked when built.
- [ ] R interop (rugarch) — subprocess adapter or drop?
- [ ] A `DataAdapter` protocol, and machine-readable licence metadata that
      packaging can actually enforce.
