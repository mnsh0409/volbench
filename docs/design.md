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
  `realized_variance_from_bars`, and `log_returns`.

  **Diverged (added at M1):** `log_returns` was not in the plan. The data layer
  exposed only `r^2`, but the models and the evaluator both speak in *signed*
  returns, so every caller was re-deriving them by hand. It returns a leading
  NaN so the output stays index-aligned with its input.

### Forecasting — `volbench.dist`, `volbench.models`

- **`Distribution`** (`dist.py`) — the only forecast currency. Constructors
  `from_normal`, `from_samples`, `from_quantiles`; concrete `Normal`,
  `Empirical`, `QuantileGrid`. Methods: `quantile`, `cdf`, `crps`, `log_score`,
  `pinball`, `sample`.

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
  which returns a 199-point `QuantileGrid` (chosen over `from_samples` because
  it needs no RNG and so scores bit-identically across runs).

  **Diverged:** `HAR.fit` takes a realized-**variance** series, not returns.
  This is documented in its module and handled by `run_backtest(fit_series=...)`,
  but it means "the model interface" is uniform in *type* and not in *meaning*
  — nothing in the type system distinguishes a returns array from a variance
  array. See `docs/M1_REPORT.md` risk 2.

  **Diverged:** `GARCH.fit` never raises on optimizer failure; it falls back to
  EWMA on the same window and records `fallback=True`. HAR, by contrast, *does*
  raise on a degenerate input (non-positive RV, short window). The two models
  disagree about whether a bad origin should be survivable.

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

  **Added beyond the plan:** `SupportsUpdate`, an optional Protocol letting a
  model re-condition on newer data between scheduled refits without
  re-estimating. **No Phase 1 model implements it**, so at `refit_every > 1`
  every baseline currently holds a stale forecast between refits, recorded per
  row in `conditioned_through`. See `docs/M1_REPORT.md` risk 1.

  **Added beyond the plan:** `forecast_moments(dist) -> (mean, variance)`,
  closed-form for `Normal`, plug-in for `Empirical`, exact-for-the-interpolant
  for `QuantileGrid`, quadrature otherwise.

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
a synthetic series: 4 baselines × 200 rolling origins, ~2.3s, byte-identical
across runs. `benchmarks/make_toy_asset.py` generates its input. `make
reproduce` rebuilds both from scratch. The series is synthetic because no
licence in `docs/data_licenses.md` permits vendoring a real one — see
`docs/M1_REPORT.md`.

## Public API surface

`volbench` (root) exports the shared vocabulary and the entry points:
`Distribution`/`Normal`/`Empirical`/`QuantileGrid`, `TimeSeriesFrame`, the
proxies and `log_returns`, `Origin`/`RollingOriginSplitter`, the four model
classes and their fitted types plus `ForecastModel`/`FittedModel`,
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
position named in the `ValueError`; mixing an indexed input with a bare array
is refused. Bare arrays for *every* input remain accepted as an explicit
positional opt-in (only lengths are checked), because they carry no calendar
to compare. `benchmarks/toy.py` passes indexed Series so its path exercises
the guard, and keeps its own index assertion as a redundant belt.

## Open questions (carried into Phase 2)
- [x] Closed-form vs. sample-based CRPS — settled: closed form for `Normal`,
      exact ensemble form for `Empirical`, trapezoidal pinball integral for
      `QuantileGrid`.
- [ ] `forecast_moments` on a `QuantileGrid` treats the grid as the whole law
      (flat tails), so it understates the variance of a heavy-tailed forecast —
      ~8% at nu=5, ~24% at nu=3. QLIKE for Student-t GARCH is biased upward as
      a result. Fix in `dist.py` (a parametric Student-t) or in
      `forecast_moments` (tail extrapolation)?
- [ ] Refit schedule API: per-model overrides, and `SupportsUpdate` on the
      econometric models so `refit_every > 1` means "re-estimate every 21 days,
      re-filter daily" rather than "freeze the forecast for 21 days".
- [ ] Multi-horizon: separate `Distribution` per h, or joint object? (`horizon`
      exists in the splitter and in result rows; only h=1 is exercised.)
- [ ] Should a range/RV proxy feeding HAR be reconciled with the close-to-close
      return target it is scored against? They are different quantities.
- [ ] TSFM context construction; alignment across calendars.
- [ ] R interop (rugarch) — subprocess adapter or drop?
- [ ] A `DataAdapter` protocol, and machine-readable licence metadata that
      packaging can actually enforce.
