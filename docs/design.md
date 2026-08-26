# 04 · volbench API design

> Status: **AS-BUILT at the determinism build (v0.6.0-determinism)**. This
> file described the plan until the three Phase 1 streams landed; it now
> describes what exists, with every place the build diverged from the plan
> called out inline under **Diverged:**. Updated at M1, at M2
> (`m2/evaluator-hardening`, `m2/cleanup`), at the Phase-2 integration of
> `feat/p2-models-classical`, `feat/p2-models-tsfm`, `feat/p2-inference` and
> `feat/p2-data-panel` (docs/P2_INTEGRATION.md — all four streams flagged
> this file's drift; it is reconciled there in one pass), on
> `feat/p2-protocol`, which took the three protocol decisions that
> integration deferred (D-018 invalid targets, D-019 fit window, D-020 the
> panel list) and closed the ES gap, and on `feat/p2-runner`
> (docs/P3_RUNNER.md), which built the grid runner, the local multiprocessing
> executor and the economic-value metric, and closed two model-protocol open
> questions (D-030 HAR's retransformation, D-031 PatchTST's device class),
> and on `feat/p3-determinism` (docs/P3_DETERMINISM.md), which closed the
> BLAS-thread-count reproducibility defect `feat/p2-runner` had reported
> open — pinning the thread count, putting it in every config hash, bounding
> the Student-t `nu` that made GARCH sensitive to it in the first place, and
> making a fit's fallback/convergence state visible per row and per cell
> (D-032).
> The planning-folder copy is behind this one and needs re-syncing from here,
> not the other way round.

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
    **Since `feat/p2-data-panel`:** `parse_stooq_csv` reads both the
    per-symbol CSV export and the hand-downloaded bulk `d_*_txt` archive
    layout (`<TICKER>,<PER>,<DATE>,...`, `YYYYMMDD` dates parsed with an
    explicit format — as a bare integer `20050225` is a valid nanosecond
    epoch); `ingest_manual_csv(expect_ticker=...)` refuses a file whose
    declared `<TICKER>` is not the one asked for, and `ManualIngestResult`
    carries `ticker`. **Since the Phase-2 integration:** `STOOQ_INDEX_SYMBOLS`
    holds only the seven indices Stooq still serves as indices — the
    SPX/DJI/FTSE entries that pointed at unlicensed CFD proxies are gone; the
    asset list is `data/panel.py`'s, below.
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
  `realized_variance_from_bars`, `log_returns`, `overnight_plus_range_variance`
  and — public since `feat/p2-data-panel` — its two pieces `overnight_variance`
  (`(ln(O_t/C_{t-1}))^2`) and `rogers_satchell`. The D-016 target is literally
  their sum, so the panel report's overnight-share decomposition cannot drift
  from the estimator it decomposes.

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

- **The panel** (`data/panel.py`, `data/crisis.py`, `data/diagnostics.py`,
  `data/build_panel.py`; added on `feat/p2-data-panel`) — the study's actual
  asset list and how its targets are built, composed out of the adapters
  above; nothing here re-implements parsing or an estimator.
  - `EQUITY_PANEL` / `CRYPTO_PANEL` (`EquitySpec`, `CryptoSpec`): seven
    indices Stooq still serves (NDX, DAX, CAC, NKX, HSI, TWSE, KOSPI) plus
    the D-012 ETF stand-ins SPY/DIA for the SPX/DJI slots, and BTC/ETH from
    Binance minute bars — **11 assets** (D-020). `PANEL_START`/`PANEL_END`
    bound the window; `build_equity_series`/`build_crypto_series`/
    `build_panel` produce `PanelSeries`; `build_targets` the four daily
    variance targets (`TARGET_NAMES`). Stooq is never fetched
    programmatically — the equity arm reads hand-downloaded bulk archives
    under a `raw_root` outside the repo (`tests/test_licensing_guard.py`
    asks git that the data trees stay untracked). Targets are built on each
    file's full history and trimmed to the window afterwards, so the first
    in-window day keeps a genuine previous close (a backward-looking read of
    data that already existed).
  - `RETIRED_EQUITY` / `equity_spec` (D-020): the FTSE-100 slot's ISF is
    ingestable but not in the panel. It starts 2015-03-04 and so holds no GFC
    observations at any window, which no protocol choice can fix, so it was
    dropped from the study — but not from the code: `equity_spec` resolves
    both maps and `build_equity_series("ISF")` still works. "The panel" is
    exactly `EQUITY_PANEL`, and `build_panel` iterates only it.
  - `FIT_WINDOW_DEFAULT = 500` / `FIT_WINDOW_ROBUSTNESS = 1000` (D-019): the
    study's rolling window and its robustness arm, named here because they
    are a property of the panel runs. Both are ordinary
    `RollingOriginSplitter(window=...)` values and therefore already in every
    `config_hash`; no separate key exists or is needed. At 1000 the GFC arm
    was mostly warm-up (31-86 of 140-149 days scored per equity series, 0 of
    90 crypto COVID days); at 500 both are scored in full.
  - `PanelSeries.fit_input(target=None, *, policy=...)` (D-018): **the seam
    where the invalid-target policy is enforced**, returning a
    `volbench.compaction.FitSeries` over the series' primary target (or a
    named other one), on the panel's own calendar and index. Every adapter is
    covered by this one implementation; none sanitizes its own input.
    `PanelSeries.invalid_target_days` counts what it will drop.
  - `repair_bars` / `BarQuality`: a bar must satisfy `low <= min(O,C) <=
    max(O,C) <= high`; sub-1e-5 relative violations (decimal rounding) are
    clamped and counted, larger ones (a close printed outside its own
    session — two feeds disagreeing) are left as they are, flagged, and their
    range-based targets set NaN. Nothing is dropped: the row survives so
    `run_backtest` records a `missing_reason`.
  - `crisis.py`: the D-004 sub-sample windows (`CRISIS_WINDOWS`,
    `PENDING_WINDOWS`, `tag_dates`, `crisis_mask`, `crisis_table`) as
    metadata about *dates*, applied to result rows after scoring. Its whole
    public API takes only a `DatetimeIndex` and the module's AST is checked
    for imports of the forecasting stack — a tag can never reach a model
    (every window is defined by dates only knowable after the episode). The
    2025-26 window is undated on purpose (D-004 fixes it at grid freeze).
  - `diagnostics.py`: full-sample measurements of a built panel — correct for
    a *report*, leakage in a *feature*; nothing consumes them. `crisis_coverage`
    takes the union of `RollingOriginSplitter`'s own `test` indices rather than
    re-deriving the arithmetic (the audit found the re-derivation off by one).
  - `build_panel.py` regenerates `docs/PANEL_REPORT.md` (`--window` /
    `--robustness-window` select which fit windows §8.1 reports); no figure in
    it is hand-entered. Three of its findings became decisions on
    `feat/p2-protocol` — ISF starts 2015 (D-020), the GFC arm is mostly inside
    the warm-up (D-019), HSI's zero-variance days (D-018) — and the report is
    regenerated under them. The overnight share being 33-51% rather than the
    ~9-15% the toy generator suggested is a correction to the *planning*
    documents and is still outstanding there.

  **Diverged:** the plan's `DataAdapter` protocol is still absent; the panel
  is source-shaped composition over source-shaped adapters, which is the
  honest description of what a `fetch(spec)` would have to hide.

### Forecasting — `volbench.dist`, `volbench.models`

- **`Distribution`** (`dist.py`) — the only forecast currency. Constructors
  `from_normal`, `from_student_t`, `from_samples`, `from_quantiles`; concrete
  `Normal`, `StudentT`, `Empirical`, `QuantileGrid`. Methods: `quantile`,
  `cdf`, `crps`, `log_score`, `pinball`, `sample`, `expected_shortfall`, and
  — parametric families only — `mean`/`variance` in closed form.

  **Added on `feat/p2-protocol`:** `expected_shortfall(level)`, the lower-tail
  mean `level^-1 ∫_0^level Q(u) du`, in the same return-side sign convention
  as `quantile` (negative in the lower tail). Closed forms on `Normal` and
  `StudentT`, the exact integral of the piecewise-linear quantile function on
  `Empirical` and `QuantileGrid`, 128-node Gauss-Legendre quadrature in the
  base class for anything else. It lives on the distribution, next to
  `variance`, because two modules need the same number and neither may import
  the other: `evaluate.py` writes the `es_<level>` columns at scoring time and
  `backtests.py` scores FZ0 from them, while the dependency direction is
  evaluation → results → distributions. `backtests.expected_shortfall` remains
  as a public wrapper over it.

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

  **Resolved on `feat/p2-runner` (D-030), was open since M2.** HAR's lognormal
  retransformation `E[RV]=exp(ŷ+½·resid_var)` is sensitive to the target's
  log-space noise: on the toy fixture the correct, noisier
  overnight-plus-range target inflated HAR's forecast ~13% above the *known*
  true variance, where the intraday Parkinson target happened to leave it
  well-calibrated. HAR now retransforms through `models/_rv` like every other
  log-RV model, Duan smearing by default (`retransform="smearing"` in `spec()`
  and in `name`, so `har_rv-smearing`), with `retransform="gaussian"` kept as
  the arm that reproduces the pre-0.5.0 numbers exactly. Measured at the
  switch on the toy fixture: forecast/true **1.1320 → 1.1102**,
  QLIKE-against-the-truth **0.0263 → 0.0242**, QLIKE against the scored proxy
  0.1823 → 0.1806, the 5% hit rate unchanged. About a sixth of the overshoot,
  so an improvement and not a cure — the residual is HAR's own one-step
  in-sample factor, and a component overnight+intraday HAR remains the deeper
  fix. Every HAR hash moved; so did the version (0.5.0).

  **Diverged:** `GARCH.fit` never raises on optimizer failure; it falls back to
  EWMA on the same window and records `fallback=True`. HAR, by contrast, *does*
  raise on a degenerate input (non-positive RV, short window). The two models
  disagree about whether a bad origin should be survivable. Every Phase-2
  model sides with HAR (raise; the evaluator records the origin as a
  `fit_error@` row).

  **Since `feat/p3-determinism` (D-032)** that fallback is no longer silent:
  `FittedGARCH.fit_diagnostics()` reports it, `run_backtest` records it per
  row in `fit_status`, and `run_grid` counts it per cell. Measured where it
  matters: **0 of 1410** scheduled fits on three real panel assets, and 0 of
  200 on the toy fixture — the fallback is a guard that does not fire, not a
  hidden estimator swap, which is what the diagnosis had to establish before
  the real cause could be found.

  **Also D-032:** the Student-t degrees of freedom are bounded to
  `NU_BOUNDS = (2.1, 50)` and SLSQP runs at `FIT_TOL = 1e-10`, both in
  `spec()` and therefore in the hash. `arch`'s own upper bound is 500, and a
  500-observation window carries no information about tail thickness up there
  — the likelihood is flat along `nu`, and that flat direction is what let a
  last-ulp BLAS difference move the optimizer into a different local optimum.
  On the Gaussian toy fixture the bound binds on every fit (correctly: the
  true `nu` is infinite); on the real panel it binds on 18 of 705 GARCH-t fits
  (2.6%) with `nu` interior at a median around 7, so estimable tail thickness
  is untouched. A multi-start with best-likelihood selection was measured as
  an alternative and rejected — it made the sensitivity worse.

- **Classical log-RV models** (`models/sf.py`, `models/lgbm.py`, added on
  `feat/p2-models-classical`): `AutoETSRV` (statsforecast `AutoETS`, pinned
  `model="AZN"`, `season_length=1`), `AutoARIMARV` (Hyndman-Khandakar
  stepwise, `aicc`/`kpss`, no approximation — every search setting in
  `spec()`) and `LightGBMRV` (L2 boosting of `log RV_{t+1}` on lags 1..22 of
  log RV plus HAR's weekly and monthly aggregates — 24 features, so a
  HAR-equivalent linear function exists inside the feature set). All three
  fit on the realized-variance series like HAR and emit `Normal(0,
  sqrt(vhat))`.
  - **One retransformation, shared** (`models/_rv.py`): a model fit in logs
    forecasts `E[log RV]`, and `exp(·)` of that is a median. `_rv` implements
    Duan's (1983) smearing factor `mean(exp(e_i))` over the fit window's
    in-sample residuals (the DEFAULT, `retransform="smearing"`) and the
    Gaussian `exp(σ²/2)` (`retransform="gaussian"`, the like-for-like arm
    against HAR), both config-hashed and both in the model `name`. Smearing
    is the default because docs/M2_NOTES.md measured the Gaussian correction
    over-inflating on the noisy overnight-plus-range target. Since the
    Phase-2 integration `PatchTST` retransforms through the same module (it
    had carried a local copy), and since `feat/p2-runner` (D-030) so does
    `HAR` — the last hold-out, and the model the M2 measurement was taken on.
    Every log-RV model in the package now shares one correction and one
    default.
  - **`SupportsUpdate` is implemented, exactly.** statsforecast's `forward`
    re-filters at fixed parameters (`ets_f(y, model=fitted)` /
    `Arima(x, model=fitted)`; the ETS h-step variance is read from the
    scheduled fit's own `predict`, never from `forward`, which would
    re-estimate the innovation variance). Note the modelling property: the
    filter re-runs over the whole new window from the initial state of the
    last fit — not carried forward — which is what R does and is immaterial
    at 500-observation windows. LightGBM's `update` moves the RV buffer under
    fixed trees, which *is* re-conditioning for a deterministic feature map.
  - **LightGBM temporal integrity:** every feature row is a function of
    `rv[t-21..t]` only, built from the window handed to `fit`/`update`; there
    is no scaler, no early stopping and no validation split — each of those
    is a documented leak in a boosted-tree pipeline, and their absence is
    asserted (a window's design matrix is bit-identical whether or not the
    array continues afterwards). `deterministic=True`, `force_row_wise=True`,
    `num_threads=1`, one seed: bit-identical forecast and byte-identical
    serialized model, pinned.
  - **Open — in-sample smearing optimism (LightGBM):** an ensemble shrinks
    its own residuals, and a shrunken residual set drives Duan's factor to 1,
    turning the variance forecast back into a median. At LightGBM's stock
    shape the in-sample log-space residual variance was 0.015 against a
    realized one-step forecast-error variance of 0.42 (factor 1.008 vs
    HAR's 1.207). The shipped defaults (100 rounds, 4 leaves,
    `min_data_in_leaf=60`, `lambda_l2=5`) land at **0.28 vs 0.38** — an
    optimism of HAR's order, bounded by a regression test that fails if the
    capacity is raised. Not eliminated: an **out-of-fold factor** is the
    honest fix and a Phase-2 modelling decision (docs/P2_INTEGRATION.md).

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
    1e-5)`), which flattens a ~1e-4-level series into a constant (measured:
    at raw units Moirai's q10-q90 spread collapses below 5% of the level; at
    1e4 it exceeds 50% and is stable from there upward). Chronos and TimesFM
    are indifferent to it; it is applied to all four so the contract has one
    unit.
  - **Known, bounded bias — the grid mean.** The mean of a quantile grid with
    flat tails *truncates* the tails of the model's own predictive law: the
    estimator is the same one that understated a Student-t GARCH's variance
    by ~8% at nu=5 / ~24% at nu=3 (M1 report §4.2, D-014), and here the grid
    is what the checkpoint emits (9 levels for Chronos-Bolt/Moirai, 21 for
    Chronos-2, 0.1..0.9 for TimesFM), so no parametric object exists to read
    a closed-form mean from. The bias is downward and monotone in the tail
    mass outside the outer quantiles. **Revisit if TSFM QLIKE looks odd**
    relative to the econometric models (docs/P2_INTEGRATION.md §3).
  - `spec()` carries checkpoint id, the resolved commit hash of the weights,
    dtype, and package versions, so the config hash moves with the weights.
    Chronos-Bolt/2, TimesFM 2.5 and Moirai-2 emit quantiles directly — no
    sampling — and the `tsfm`-marked tests pin bit-identity on the GPU.
    `device` is not in `spec()`: these adapters estimate nothing, so no RNG
    stream is drawn from and D-031's argument for hashing the device class
    does not apply to them (a forward pass differs across devices only by
    float accumulation order, which is the same tolerance question D-026
    already scopes).
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
  from each device's own RNG stream, so results reproduce **per device
  class** (with `dropout=0` CPU and GPU agree to ~1e-8; with dropout on, a CPU
  and a CUDA fit of one seed are two training realizations, ~1.6% apart on a
  300-point window).

  **The device class is hashed (D-031, resolved on `feat/p2-runner`).** Until
  v0.4.0 `device` was outside `spec()` entirely, so two fragments with one
  `config_hash` — one computed on a CPU, one on a GPU — could legitimately
  hold different numbers, and the content-addressed store could serve either
  for the other. `spec()` now carries `device_class` (`"cuda"` / `"cpu"`, the
  torch device *type*), so those are two cells and neither can be served for
  the other. The ordinal stays out: `"cuda:0"` and `"cuda:1"` hash the same,
  which is what lets a multi-GPU grid place a cell on whichever card is free;
  two different GPU *models* are not distinguished either, the same
  qualification D-026 puts on the numpy kernel family. `device="auto"`
  resolves against the machine and therefore needs torch importable — pin
  `"cpu"` or `"cuda"` to fix a configuration's identity in advance; an
  explicit device keeps `spec()` a torch-free call. **No `update`:** re-conditioning
  a trained net without re-estimation is not well defined, so it runs frozen
  between refits and `conditioned_through == fit_origin` on every row.
  Training windows are cut from the fit array only (last target = the
  origin), which is the leakage-check focus for a dataloader.

  **Diverged:** the plan's model list names these as adapters of a generic
  kind; as built they are RV-fed, like HAR, not return-fed — the same
  type-uniform / meaning-divergent interface noted for HAR above, now shared
  by eight models. Optional deps live under the `tsfm` extra and are never
  installed in CI: `tests/conftest.py` skips `tsfm`/`timegpt`/`gpu`-marked
  tests by default and unconditionally under `CI`, while each adapter keeps a
  mocked-backend test in the default suite.

- **Optional backends — one rule** (Phase-2 integration; the two model
  streams had chosen differently): every adapter is re-exported from
  `volbench.models` and `volbench`, and every optional backend
  (statsforecast, lightgbm, torch, chronos, timesfm, uni2ts, nixtla) is
  imported *inside* `fit`, so `import volbench` needs no extra
  (`tests/test_optional_backends.py` runs the import with the backends
  blocked). Three extras: `classical` (statsforecast + lightgbm; cheap, on
  every CI leg), `tsfm` (CUDA torch 2.5.1+cu121 pinned for the 4090 box's
  driver, plus the foundation-model packages; never in CI) and `torch-cpu`
  (the same torch, CPU wheels, for CI's 2-epoch PatchTST smoke test;
  declared to conflict with `tsfm`). The Makefile's `EXTRAS` variable
  selects them because `uv run --extra` syncs the environment to exactly the
  extras named. `pyproject.toml`'s `[tool.uv]` block documents the three
  resolver mechanisms (override-dependencies for stale upper bounds,
  dependency-metadata for uni2ts's torch cap that an override cannot lift,
  conflicts/sources/indexes for the two torch builds).

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

### Invalid targets — `volbench.compaction`

Added on `feat/p2-protocol` (D-018). An **invalid target day** is a day whose
primary variance target is NaN or `<= 0`. The panel has 125 of them: 109 from
bars whose close printed outside their own session range (TWSE 80, CAC 28,
HSI 1), 14 where a monotone bar met a stale open so that Rogers-Satchell and
the overnight term are *both* exactly zero (HSI 12, NKX 2), and 2 first-in-
window days with no previous close (SPY and DIA). Every log-RV
model takes `log(RV)`, so before this a single such day failed every training
window containing it — measurably, at window 500 and `refit_every=21`, 36% of
HSI's origins and 52% of TWSE's.

- **`FitSeries(values, policy, index)`** — the series a model is fitted on,
  kept on the **full calendar**, plus the policy for its unusable days.
  `FitSeries.compact(...)` / `.raw(...)` / `.of(..., policy=...)`;
  `window(train)` materializes one origin's fit input, `window_positions` and
  `dropped_positions` expose what it did, `n_invalid` counts.
- **The rule.** Under `policy="compact"`, the window for a splitter `train`
  array of `N` positions ending at `origin` is the last `N` **valid**
  observations at positions `<= origin`. Where invalid days sit inside the
  span, the window's calendar extent stretches further into the past; its last
  observation never moves past the origin. Under `policy="none"` it is
  `values[train]`, the pre-D-018 behaviour, kept as an explicit arm.
- **The splitter is untouched.** Compaction happens when a window is
  *materialized*, never by reshaping the series the splitter sees, so origins
  and targets remain calendar positions and *which days are scored does not
  change*. An invalid day is still a perfectly good origin: its own target is
  unmeasurable, but its history is intact, so the forecast issued at it is a
  normal forecast — only the row whose *target* is that day carries a
  `missing_reason`.
- **Temporal integrity.** Validity at a position depends on that position's own
  value alone, and only positions `<= origin` are ever selected, so no future
  observation — and no future day's *validity* — can change an earlier window.
  `tests/test_compaction.py` asserts both, with an inert-proof companion.
- **Too little history is explicit.** Fewer than `N` valid observations at or
  before an origin raises `InsufficientHistoryError`, which the evaluator turns
  into the standard NaN-plus-`missing_reason` row. A short window is never
  silently fitted: the run would then report a window length it did not use.
  Structurally this can only affect a prefix of a series' origins.
- **Identity.** The policy enters the config hash under
  `protocol.invalid_target_policy` whenever it is not `"none"`, so the two arms
  can never share a cache entry. A bare array means `"none"` and hashes exactly
  as it did before the key existed.

**The cost, stated because it is real.** For models that read *positional* lags
of the fit series — HAR's daily/weekly/monthly components, LightGBM's 22 lags —
"yesterday" now means *the previous measured day*, so on the six affected
series a lag-1 regressor can span two or more calendar days and the 22-lag
window more than 22. Imputing the missing variance instead would put a number
nobody measured into the regressors, which is worse; but any statement about
these models' memory in calendar time has to say so. Both adapters' docstrings
carry the caveat, and `tests/test_compaction.py` pins an instance of it.

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

### Evaluation — `volbench.evaluate`, `volbench.results`, `volbench.execute`, `volbench.runner`, `volbench.econ`

- **`run_backtest(model_factory, series, proxy, splitter, seed, *, asset,
  proxy_name, data_spec=None, fit_series=None, levels=DEFAULT_LEVELS,
  executor=None, store=None, overwrite=False, recondition="daily") ->
  pd.DataFrame`** — scores one cell. Returns one tidy row per
  `(origin, horizon)` with the forecast's mean and variance, the realized
  return, the proxy, CRPS, log score, QLIKE, and
  pinball/VaR-quantile/ES/hit at each level, plus `config_hash`, `seed`,
  `fit_origin`, `conditioned_through`, `refit` and `missing_reason`.
  `frame.attrs` carries `config_hash`, `config` and `cached`.

  **Added on `feat/p2-protocol`:** `es_<level>` beside every `var_<level>` —
  the expected shortfall the same predictive law implies, from
  `Distribution.expected_shortfall`. Like the VaR quantile it describes the
  *forecast*, so it is written even where the target is unscorable, and NaN
  only where no forecast was made at all. `var_backtest` reads it with no
  argument, which is what makes the FZ0 loss available for every cell rather
  than only for callers willing to re-assert a distributional family the row
  does not record. A schema change, hence the 0.4.0 version bump: every
  `config_hash` moves and no pre-0.4.0 fragment is served again.

  **Also on `feat/p2-protocol`:** `fit_series` accepts a
  `volbench.compaction.FitSeries` as well as an array or Series, which is how
  D-018's invalid-target policy reaches the model. A bare input still means
  `policy="none"` and hashes as it always did.

  **Diverged:** the plan had an **`Evaluator`** class consuming
  `(Distribution, realized target)` streams. As built it is a function over a
  whole cell; the comparison-inference half lives in two separate modules
  (below) that only *consume* its rows.

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

- **`volbench.inference`** (added on `feat/p2-inference`) — "who wins?" over
  loss arrays / `ResultsStore` rows; touches nothing on the scored path.
  `diebold_mariano` (rectangular or Bartlett window truncated at `h-1`, the
  Harvey-Leybourne-Newbold small-sample factor, `t_{n-1}`; DM's own rule for
  a non-positive variance estimate, flagged; at `h=1` exactly the one-sample
  t, pinned; size checked by simulation). `model_confidence_set` (Hansen,
  Lunde & Nason 2011; `T_R` default per their corrigendum or `T_max`;
  moving-block bootstrap by hand with contiguous forward-running blocks and
  **no wrap-around**; Politis-White automatic block length mirrored from and
  pinned against `arch.bootstrap.optimal_block_length`; `B=10 000`,
  `alpha=0.10`; MCS p-values as cumulative maxima). `loss_matrix` /
  `dm_matrix` / `compare_models` return the MCS together with the pairwise
  DM matrix, whose p-values are *not* multiplicity-corrected — the MCS is
  the primary tool. Rows with a `missing_reason` are dropped pairwise-
  complete (DM) / listwise (MCS) and `n_dropped` is recorded; from a store,
  cells must share the same series bytes before their origins are aligned.
  Every result carries a `config_hash` over its inputs and settings and the
  bootstrap takes a mandatory `seed`.

- **`volbench.backtests`** (added on `feat/p2-inference`) — VaR/ES backtests
  on the `hit_<level>` / `var_<level>` / `realized_return` columns:
  `kupiec_pof` (with the `0·log 0` convention), `christoffersen`
  (independence and conditional coverage from first-order Markov transition
  counts, conditional on the first observation so `LR_cc = LR_uc + LR_ind`
  holds exactly; a NaN hit removes its neighbouring transitions rather than
  splicing across the gap), `fz0_loss` (Patton, Ziegel & Chen 2019 eq. 6,
  domain enforced, pinned against their Figure 1/2 and the `L(kY,kv,ke) =
  L + log k` identity), `expected_shortfall` (closed forms for `Normal` /
  `StudentT`, exact for `Empirical`/`QuantileGrid`, quadrature otherwise)
  and `var_backtest` for one cell at one level. Below 10 expected
  exceedances a `SmallSampleWarning` names both `n` and `expected_hits`.

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

  **Added on `feat/p3-determinism` (D-032):** `build_config` records an
  `environment` block — today one key, `blas_threads`, from
  `volbench.determinism.thread_pin()`. It is written **unconditionally**,
  which is the opposite of the `protocol` block's rule and deliberate: the
  BLAS thread count moves `arch`'s optimizer between local optima while moving
  no content digest, so before D-032 a 32-thread fragment and a 1-thread
  fragment of one cell shared a hash and the store served either for the
  other. A setting that changes a number always binds. The pin itself
  (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) is exported by the Makefile
  and CI, so the sanctioned path records `1` on every machine and
  cross-machine cache sharing survives; it is the *unpinned* path that stops
  sharing, which is the point.

- **`Executor` / `SerialExecutor` / `ProcessExecutor`** (`execute.py`) — the
  execution seam. A cell is one `(asset, model, splitter, seed)` unit; within a
  cell, `run_backtest` further splits origins into *refit blocks* and maps
  those through the same seam. Serial only at M1; `ProcessExecutor`, the local
  multiprocessing backend, landed on `feat/p2-runner` (D-028). A Slurm-array
  backend belongs here too and nowhere else.

  **`ProcessExecutor`** — `concurrent.futures.ProcessPoolExecutor` over a
  configurable worker count, **fork** start method by default, one pool per
  `map` call (so the object holds no OS resources between calls and is safe in
  a config). Three things it does beyond mapping:

  - **Byte-identity is the gate, not a hope.**
    `tests/test_runner.py::TestSerialParallelIdentity` runs one grid through
    both backends into two stores and compares the *parquet bytes* fragment by
    fragment. That is D-011's H4 claim ("same forecasts, three execution
    paths, verified identity"); it has an inert-proof companion that perturbs
    one input by one ulp and asserts the comparison notices.
  - **The numpy kernel family travels with the work (D-026).** Fork means a
    worker inherits the parent's already-initialized numpy — the dispatch
    decision itself, not merely the environment that produced it. On top of
    that the parent's `NPY_DISABLE_CPU_FEATURES` is propagated verbatim
    (including "unset", because inventing a value the parent did not have
    would make the pool disagree with a serial run in the same shell), and
    every worker compares its own enabled dispatch targets against the
    parent's (`kernel_signature()`) and refuses the task if they differ. A
    mismatch is then a loud error naming both signatures rather than a
    silently orphaned fragment.
  - **The BLAS thread count travels with it too (D-032).** Same two
    mechanisms as the kernel family: `OMP_NUM_THREADS` and
    `OPENBLAS_NUM_THREADS` are propagated verbatim (including "unset") and
    every worker checks its resolved count against the parent's. It needs the
    guard *more* than the kernel family does — the thread count moves no
    content digest, so a worker at a different count produced numbers that
    were filed under the parent's hash rather than orphaned. Both pins and
    both checks now live in `volbench.determinism`, which
    `execute.py` re-exports `kernel_signature`/`KERNEL_PIN_VAR` from.
  - **Rule 1 ("no nesting") is enforced.** `map` raises inside a worker
    process, because a pool-backed executor nested in a bounded pool deadlocks
    — and a deadlock is indistinguishable from a slow grid until someone kills
    it. The runner hands every cell a `SerialExecutor` for its refit blocks.
  - **The start method is `forkserver`, on measurement** (D-028): a parent that
    has trained a LightGBM model and then plain-`fork`s produces workers that
    deadlock on their first LightGBM call, because OpenMP's locks are copied
    mid-flight. `fork` stays available, faster to start, and is *refused* in a
    process that has already used a backend in `FORK_UNSAFE_MODULES`. The
    spawn-family methods are likewise refused where `__main__` is not
    importable (a REPL, `python -`, a heredoc) — both failure modes are
    messages rather than a hang or a `BrokenProcessPool`.

  `ProcessExecutor` knows nothing about GPUs. Serializing GPU-bound cells is
  the runner's job, from explicit configuration (below).

- **`volbench.runner`** (added on `feat/p2-runner`, D-027) — grid
  orchestration. `run_grid(grid, source, store, *, cpu_executor, gpu_executor,
  overwrite, manifest_path, on_cell) -> RunManifest` runs the full cross of
  `asset × model-config × horizon × protocol-arm`. It is deliberately thin:
  every number is still decided by `RollingOriginSplitter`, `config_hash`,
  `ResultsStore` and `run_backtest`, and what the runner adds is four
  properties a bare loop would not have.

  - **`GridSpec`** (`assets`, `models`, `horizons`, `arms`, `seed`, `levels`)
    expands to `Cell`s in a **total order** over `(asset, model label, horizon,
    arm label)` — sorted, never set/dict iteration order — so two descriptions
    of one grid produce one manifest ordering. `ModelConfig` carries the
    factory, `fits_on_variance` and `lane`; `ProtocolArm` carries `window`,
    `refit_every`, `step`, `recondition` and `invalid_target_policy`, i.e.
    exactly the settings that already reach the config hash, so two arms can
    never share a cached cell. The `horizon` dimension is the *splitter's*
    horizon: a cell at `H` scores every `h ≤ H` and has its own origin set, so
    horizons 1 and 5 are different experiments, not a subset relation.
  - **Resumable by construction.** A cell whose `config_hash` is in the store
    is skipped (`status="cached"`) by `run_backtest`'s own short-circuit —
    before any fit. There is no run state on disk beyond the results
    themselves, so "resume after an interruption" and "extend the grid" are
    the same operation, and a re-run leaves existing fragments byte-identical
    *and unrewritten* (the tests check mtimes as well as bytes).
  - **Per-cell fault isolation**, the evaluator's per-origin philosophy one
    level up: a cell that raises is recorded `status="failed"` with the
    exception type and message, logged at WARNING, and the grid continues.
    Only `Exception` is caught. A failure *inside* a cell's origins never
    reaches the runner at all — `run_backtest` has already made it a NaN row —
    so `n_missing` is carried on every outcome to keep those visible from the
    manifest. A failure of the *data source* is not isolated and propagates:
    a wrong path is a wasted grid, not a result.
  - **Lanes.** `ModelConfig.lane` is `"cpu"` (fan out) or `"gpu"` (serialized,
    its own executor, `SerialExecutor` by default). It is **declared, never
    inferred from a model's name** — a name-based guess misroutes the first
    adapter that breaks the pattern, and the failure mode is a CUDA OOM
    halfway through a grid. The lanes run **CPU first, then GPU**
    (`LANE_ORDER`), because the CPU lane forks and forking a process that has
    already initialized a CUDA context is undefined behaviour; running them
    concurrently needs `gpu_executor=ProcessExecutor(workers=1)` and is not
    the default.

  **`AssetData`** (returns, proxy, `proxy_name`, optional variance series,
  `data_spec`) is what a `DataSource.load(asset)` returns, and it is loaded
  **once per asset in the parent process** before any cell runs — one read,
  one content digest, every cell of that asset provably scored against the
  same bytes. `MappingDataSource` wraps an in-memory dict; a plain mapping is
  accepted directly.

  **`RunManifest`** is a tuple of `CellOutcome`s in grid order — index, asset,
  model, horizon, arm, lane, status, `config_hash`, `n_rows`, `n_missing`,
  `n_fits`, `n_fits_fallback`, `n_fits_nonconverged`, `wall_clock_s`, `error`
  — plus an `environment` block, with `to_frame()`, `as_json()` and
  `write(path)` (temp file + `os.replace`, the store's discipline).

  The three fit counts and `environment` are D-032. The counts are per
  *scheduled fit*, not per row: a block of 21 origins rests on one fit and a
  cell at horizon 5 writes five rows per origin, so a row-weighted rate would
  really be a statistic about the refit cadence. `CellOutcome.fallback_rate`
  is `nan`, never `0.0`, when nothing reported a status — "no fit fell back"
  and "no fit said" are different claims and only one is evidence. A cached
  cell is counted from the fragment it read, so resuming a grid does not reset
  its fallback rate to zero. `environment` says more than the hash does
  (which carries only `blas_threads`): the BLAS build and version, the kernel
  signature, the pins as they stood and the observed thread pools — what a
  reader diagnosing two runs that disagree actually needs.

  It is a *record*, never
  an input: resuming reads the store, so a stale manifest cannot corrupt a
  grid. Two manifests of one grid differ only in their timings; there is
  deliberately no reader. `read_grid_results(store, manifest)` returns this
  grid's rows only, so a store shared with an earlier study does not silently
  widen the table an analysis runs on.

  **Diverged:** the plan's **`Runner`** was a class; as built it is a function
  over dataclasses, matching `run_backtest`. The plan's **GPU batching**
  (several assets per forward pass) is still not built — the runner serializes
  the GPU lane rather than batching within it, and batching remains the open
  question the TSFM stream flagged, because it would reintroduce cross-series
  padding and alignment and needs its own leakage check.

- **`volbench.econ`** (added on `feat/p2-runner`, D-029) — economic value, the
  last metric in docs/research_design.md. `volatility_target_backtest(rows, *,
  horizon, annual_target_vol, leverage_cap, cost_bps, periods_per_year,
  risk_free_annual) -> VolTargetBacktest` sizes
  `position_t = min(target_vol / forecast_vol_t, leverage_cap)` from each
  stored forecast and reports annualized return, annualized vol, Sharpe net of
  costs (and gross, so the cost assumption's worth is visible), maximum
  drawdown, average leverage, turnover and cost drag.

  - **The alignment.** A stored row is already `(forecast issued at origin,
    realization at origin + h)`, so the position and the return it earns are
    staggered by construction and the backtest is one element-wise product.
    The forecast issued AT t sizes the position held INTO t+1, never the other
    way round. `tests/test_econ.py::TestAlignment` pins it three ways: the
    arithmetic by hand, a one-row shift of the realized-return column that
    must move the Sharpe by more than 25%, and a structural check that the
    module contains no `shift`/`reindex`/`roll` at all.
  - **It cannot run a model.** `econ.py` imports nothing from `volbench` —
    not `evaluate`, not a splitter, not an adapter — and `TestBoundary`
    asserts that from the module's AST and its bound globals. Economic value
    is computed after the fact from a table whose temporal integrity is
    already established and hashed.
  - **Conventions, stated because a Sharpe means nothing without them.**
    `realized_return` is a log return; sizing uses it as-is (consistent units
    with the forecast variance) and the P&L converts with `expm1` before
    levering, so the equity curve compounds like capital and the drawdown is a
    drawdown of capital. Annualization is **per asset** — 252 for equities,
    365 for crypto (`periods_per_year_for`, whose crypto list is pinned
    against `CRYPTO_PANEL` by test, D-024's discipline). The vol target is
    quoted annualized (10% default) and converted to daily units exactly once,
    so one target means one risk on either calendar. Costs are `cost_bps`
    (default **10**, docs/research_design.md) on `|position_t -
    position_{t-1}|`, with the first day charged against a flat book. The
    risk-free rate defaults to 0 and no financing spread on leverage is
    modelled. An unusable forecast sizes to **zero** rather than being
    dropped — dropping would splice non-adjacent days into one turnover step
    and under-charge costs — and `n_flat` reports how often.
  - A levered book that is wiped out is reported as such (`ruined`, a
    `RuntimeWarning`, equity floored at zero) rather than compounded through a
    negative wealth.

### Benchmarks — `volbench.benchmarks`

**Added beyond the plan.** `benchmarks/toy.py` composes all three streams over
a synthetic series: since the Phase-2 integration, **8** models — the five
of M2 (naive, EWMA, GARCH, GARCH-t, HAR — the last as `har_rv-smearing`
since D-030) plus AutoETS, AutoARIMA and
LightGBM on the log-RV series — × 200 rolling origins, ~1 minute, byte-
identical across runs; the three classical models are the *cheap* Phase-2
additions and the only ones in `make reproduce`. `benchmarks/smoke_tsfm.py`
(`make smoke-tsfm`) runs Chronos, TimesFM, Moirai and PatchTST over the same
fixture, splitter and target into their own `ResultsStore`
(`data/smoke_tsfm/`), local-only and never in CI or `reproduce`: it needs the
`tsfm` extra, cached weights and a GPU. Both runs fail by name if an extra is
missing rather than recording 200 `fit_error` rows. The scoring target is a
property of the run, never of a model: every cell scores QLIKE against
`overnight_plus_range_variance` (D-016), with Parkinson available as a labeled
robustness arm behind the `target` flag; HAR's *fit input* is always the
overnight-plus-range series regardless of the flag. Since `feat/p2-protocol`
the variance-fed cells take that input as a `FitSeries` under D-018's default
policy, so `make reproduce` exercises the protocol the study runs rather than a
simpler path beside it; on this fixture the policy is provably a no-op
(`load_series` refuses a target that is not finite and strictly positive
throughout), so it moved the four cells' hashes and none of their numbers. The
GARCH-t config exercises the parametric `StudentT` path (D-014) under
`make reproduce`.
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
proxies (`log_returns`, `overnight_variance` and `rogers_satchell` included),
`Origin`/`RollingOriginSplitter`, every model adapter and its fitted type —
the four baselines, `AutoETSRV`/`AutoARIMARV` (`FittedStatsForecastRV`),
`LightGBMRV` (`FittedLightGBMRV`), the four zero-shot adapters
(`Chronos`/`TimesFM`/`Moirai`/`TimeGPT`, fitted type `FittedTSFM`; their
shared base `ZeroShotRVModel`, `TSFMBackend` and `TimesFMForecastOptions`
stay in `volbench.models`, as does the private `_rv`), `PatchTST`/
`FittedPatchTST` — plus `ForecastModel`/`FittedModel`,
`run_backtest`/`forecast_moments`/`DEFAULT_LEVELS`/`SupportsUpdate`/
`ModelFactory`, `ResultsStore` and the config-hash helpers,
`Executor`/`SerialExecutor`/`ProcessExecutor`, `mse`/`qlike`/`pinball`, the
inference entry points (`diebold_mariano`, `model_confidence_set`,
`loss_matrix`, `dm_matrix`, `compare_models` and their result types), the
VaR-backtest ones (`kupiec_pof`, `christoffersen`, `fz0_loss`,
`expected_shortfall`, `var_backtest` and their result types), since
`feat/p2-protocol` the invalid-target policy (`FitSeries`,
`InvalidTargetPolicy`, `DEFAULT_INVALID_TARGET_POLICY`,
`InsufficientHistoryError`, `valid_target_mask`, `invalid_target_mask`), and
since `feat/p2-runner` the grid runner (`run_grid`, `GridSpec`,
`ModelConfig`, `ProtocolArm`, `AssetData`, `DataSource`,
`MappingDataSource`, `CellOutcome`, `RunManifest`, `read_grid_results`) and
economic value (`volatility_target_backtest`, `VolTargetBacktest`,
`periods_per_year_for`). `volbench.models.base` additionally publishes
`FitDiagnostics` and `SupportsFitDiagnostics` (D-032), the optional protocol
by which a fitted model says how its fit went.

Kept out of the root and reachable at their modules: `volbench.determinism`
entire (`kernel_signature`, `thread_pin`, `environment_spec`,
`environment_report`, `KERNEL_PIN_VAR`, `THREAD_PIN_VARS` — D-026/D-032
machinery, not vocabulary; `execute.py` re-exports the first two names it
already published),
`volbench.runner`'s `Cell`/`Lane`/`CellStatus`/`LANE_ORDER` (grid internals
a caller describes a grid without), and `volbench.models.patchtst`'s
`resolve_device_class`.

Per-source ingestion — the Stooq and Binance downloaders, their error types and
symbol maps, the bring-your-own-data loaders — and the study's own panel
assembly (`panel`, `crisis`, `diagnostics`, `build_panel`) stay in
`volbench.data`. Which source a series came from is a provenance question
(`docs/data_licenses.md`), not part of the library's vocabulary.

`__version__` is read from installed package metadata, so it cannot drift from
the `package_version()` that enters every config hash.

## Invariants (violations are bugs, not choices)
1. No code path lets information from t' > t influence a forecast for t.
   Structurally enforced by `RollingOriginSplitter`; checked end-to-end by the
   corruption canary in `tests/test_m1_smoke.py`. Since D-018 one thing
   *rewrites* which past observations a window holds — compaction — and it is
   bounded by the same rule: it selects only positions `<= origin`, so it
   reaches backwards and never forwards (`tests/test_compaction.py`, which
   corrupts both later *values* and later *validity*).
2. Every model output is a `Distribution`; no bare point arrays cross module
   boundaries. Pinned by `tests/test_model_interface.py`.
3. Every result row carries seed + config hash; `make reproduce` regenerates
   the toy benchmark from scratch. **Byte-identity is a claim within one
   numpy SIMD kernel family** (found at the Phase-2 integration,
   docs/P2_INTEGRATION.md §3.6): numpy's AVX-512-only float64 `log`/`exp`
   kernels differ from the x86-v3 ones in the last ulp for some inputs, which
   moves the content digest of a computed proxy and every hash built on it.
   CI and `make reproduce` pin the v3-or-lower family
   (`NPY_DISABLE_CPU_FEATURES`); across families the store misses (recomputes)
   rather than serving the wrong artefact. **It is also a claim across
   execution backends** (D-011): `ProcessExecutor` produces fragments byte-
   identical to `SerialExecutor` over a whole grid, checked on the parquet
   bytes with an inert-proof companion
   (`tests/test_runner.py::TestSerialParallelIdentity`), and the pool carries
   the parent's kernel family into every worker so the two claims cannot come
   apart. **And a claim per device class** for the one model that estimates
   under an RNG: PatchTST's `device_class` is hashed (D-031), so a CPU and a
   GPU fragment are two cells rather than one hash with two answers.

   **And a claim within one BLAS thread count (D-032)**, closed on
   `feat/p3-determinism` after `feat/p2-runner` reported it open
   (docs/P3_RUNNER.md §7, docs/P3_DETERMINISM.md). Threaded OpenBLAS reorders
   a reduction by an ulp and arch's SLSQP amplifies it into a *different local
   optimum*: `garch11_t` moved by 5.5e-1 relative on the toy fixture and
   `garch11` by 9.2e-5, the other six models bit-identical. Unlike D-026's
   case this moved **results without moving the config hash**, so a store
   filled on a 32-thread box and topped up on a 2-core runner held two answers
   under one name. Three things close it: the pin
   (`OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` in the Makefile and CI, and
   propagated into every worker), `blas_threads` in every config hash — so an
   unpinned run now misses the cache instead of being served the wrong
   fragment, exactly D-026's failure mode rather than the worse one — and the
   `nu` bound at the source, which takes `garch11_t`'s residual sensitivity to
   2.9e-6, below `garch11`'s own. The determinism rule therefore reads: **same
   seed, same code, same data, same kernel family, same BLAS thread count.**

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
      exists in the splitter, in result rows and — since `feat/p2-runner` — as
      a grid dimension; only h=1 is exercised.)
- [x] Should a range/RV proxy feeding HAR be reconciled with the close-to-close
      return target it is scored against? **Resolved (D-016):** HAR is fed and
      scored on `overnight_plus_range_variance`, the close-to-close estimator.
      Both follow-ups (docs/M2_NOTES.md) are now closed: the shared scoring
      target is D-017, and HAR's retransformation sensitivity is D-030 (below).
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
- [ ] **Out-of-fold smearing factor for LightGBM** (and, in principle, any
      high-capacity log-RV model): the in-sample factor is optimistic by a
      bounded, measured amount (0.28 vs 0.38 log-space variance). An
      out-of-fold estimate inside the training window is the fix; it must be
      a *temporal* fold, never a random one.
- [x] **HAR's retransformation** — **resolved on `feat/p2-runner` (D-030)**:
      HAR goes through `_rv` with Duan smearing as the default, like every
      other log-RV model, with `retransform="gaussian"` kept as the arm that
      reproduces the pre-0.5.0 numbers. Measured on the toy fixture:
      forecast/true 1.1320 → 1.1102, QLIKE-vs-truth 0.0263 → 0.0242. Its hash
      and its numbers moved, deliberately and before any grid freeze. Still
      open, and now the only open half: the residual over-inflation is HAR's
      own *one-step in-sample* factor, so a component overnight+intraday HAR —
      or an out-of-fold factor, the item above — remains the deeper fix.
- [ ] **TSFM grid-mean truncation** — the same family as D-014's bug, now on
      the checkpoint's own grid where no parametric object exists. Revisit if
      TSFM QLIKE looks odd; tail-extrapolation of the grid would be a patch on
      a lossy representation, the same objection D-014 recorded.
- [x] **PatchTST per-device-class reproducibility** — **resolved on
      `feat/p2-runner` (D-031)**: `spec()` carries `device_class`, so a CPU
      fragment and a GPU fragment of one configuration are two cells and the
      store can never serve one for the other. The ordinal is not hashed
      (placement, not identity), and two different GPU models are still not
      distinguished — the same qualification D-026 puts on the numpy kernel
      family, so the paper states its hardware.
- [x] **Pending protocol decisions the panel report raised** (deliberately not
      made at integration) — **all three resolved on `feat/p2-protocol`**:
      the invalid-target policy is D-018 (drop unusable days from fit windows,
      keep them as scored NaN rows — see "Invalid targets" above), the
      rolling-window length is D-019 (500, with 1000 as the robustness arm),
      and the FTSE-100 slot is D-020 (dropped; the panel is 11 assets, the
      ingestion code is kept).
- [ ] **Lag semantics under compaction**, new with D-018: for HAR and LightGBM
      a positional lag can now span more than one calendar day on a series
      with invalid days. Documented everywhere it applies; what remains open
      is a presentation question — whether the paper reports those models'
      memory in calendar days or in observations.
- [x] **The OpenBLAS thread count changes GARCH's numbers** (found on
      `feat/p2-runner`, docs/P3_RUNNER.md §7) — **resolved on
      `feat/p3-determinism` (D-032)**, and the diagnosis moved where the fix
      had to go. The hypothesis on the table was that the 5.5e-1 was the EWMA
      fallback firing on some origins under one thread count and not the
      other; it is not — the fallback fired on **0 of 200** toy origins and
      **0 of 1410** real panel fits at both thread counts, with
      `convergence_flag == 0` on every one. The mechanism is that `nu` is
      unidentified on a 500-observation window (estimated anywhere in
      [109, 500]), the likelihood is flat along it, and SLSQP's wandering
      there couples into (omega, alpha, beta) hard enough for a last-ulp BLAS
      difference to land it in a different local optimum. Three things
      shipped: the thread pin in the Makefile and CI, `blas_threads` in every
      config hash (so an unpinned run misses the cache rather than being
      served the wrong fragment), and `nu` bounded to (2.1, 50) with a tighter
      SLSQP tolerance — which takes `garch11_t`'s thread sensitivity from
      5.5e-1 to **2.9e-6**, below `garch11`'s own 9.2e-5. Still open, and now
      the only open half: **the estimator's conditioning is a property of the
      window, not of this project.** The pin makes the answer reproducible; it
      does not make the likelihood surface less multimodal, and a different
      BLAS build or scipy version can still reshuffle which optimum is found
      on the handful of origins where two are nearly tied. What bounds that
      risk today is the measurement above (2.9e-6 max relative, no optimum
      switching) and the `fit_status` column, which makes a fit that did not
      go cleanly visible instead of silent.
- [ ] **Fallback and convergence are visible but not yet actionable.** D-032
      records `fit_status` per row and `n_fits_fallback` per cell, so a cell
      that ran a degraded estimator can be *seen*. Nothing yet decides what a
      study does about one — whether a cell above some fallback rate is
      reported with a caveat, excluded, or refitted under a different
      configuration. That is a protocol question for the grid freeze, and it
      needs a rate measured on the full panel rather than on the three assets
      §5 of docs/P3_DETERMINISM.md uses.
- [ ] **Concurrent lanes.** The runner runs the CPU lane to completion before
      the GPU lane, because the CPU lane forks and forking after a CUDA
      context exists is undefined behaviour. The GPU therefore idles while the
      CPU lane runs. Running both at once needs the GPU lane in its own
      process from the start (`gpu_executor=ProcessExecutor(workers=1)`) and a
      thread in the parent; whether the wall-clock is worth that is an H4
      measurement nobody has taken yet.
- [ ] **Slurm-array executor.** D-011's third execution path is still
      unbuilt. `ProcessExecutor` is the second; the byte-identity gate is
      written so a third backend joins it by parametrization.
- [ ] **Economic value beyond one cell.** `econ.py` backtests one cell at a
      time by design. Portfolio-level questions — several assets sized
      together, a cross-sectional vol target — are out of scope for v1
      (docs/research_design.md's guard rail excludes multivariate work), but
      the aggregation layer over per-asset results is not designed yet.
- [ ] **Financing the leverage.** The economic-value Sharpe nets transaction
      costs but not a borrowing spread on `position > 1`, and
      `risk_free_annual` defaults to 0. Whether the paper reports a financed
      variant is a presentation decision.
- [ ] **The 1e-5 bar-repair threshold** still has no decision entry
      (docs/PANEL_REPORT.md §9). D-018 lowered the stakes — a NaN'd day now
      costs its own scored row and nothing else — but the split between
      "rounding" and "real error" is still a judgement calibrated on one
      archive.
