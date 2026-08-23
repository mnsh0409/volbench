# 04 · volbench API design

> Status: SKELETON v0.1 — evolves via P3 (design red-team). Mirror of the repo's `docs/design.md`; keep the two identical.

## Components

- **`TimeSeriesFrame`** — canonical container: values + trading-calendar-aware index + asset metadata. All data enters through adapters that emit this.
- **`DataAdapter`** — `fetch(spec) -> TimeSeriesFrame`; per-source modules (stooq, fred, crypto, m6, byo). Each declares license + redistributable flag (enforced at packaging time).
- **`Distribution`** — unified probabilistic forecast object. Constructors: `from_params` (e.g., N(μ, σ²)), `from_quantiles`, `from_samples`. Methods: `quantile(τ)`, `cdf(y)`, `crps(y)`, `logscore(y)`, `sample(n, seed)`. This is the only forecast currency in the system.
- **`ForecastModel` protocol** — `fit(train: TimeSeriesFrame) -> FittedModel`; `FittedModel.predict(h) -> Distribution`. Zero-shot models implement `fit` as a no-op that records context. Adapters: arch/GARCH, HAR, statsforecast, [DL route], chronos, timesfm, moirai, timegpt.
- **`RollingOriginSplitter`** — the ONLY sanctioned producer of train/test index pairs. Parameters: window, step, refit schedule. Structural leakage guarantee lives here.
- **`Evaluator`** — consumes (Distribution, realized target) streams; computes metrics per 05_metrics_reference; DM/MCS on the score matrix.
- **`ResultsStore`** — append-only parquet keyed by config hash (model, data, splitter, seed, code version). Cache hit = skip recompute.
- **`Runner`** — orchestrates grid execution: process-parallel across (series × model × window), GPU batching for TSFM inference.

## Invariants (violations are bugs, not choices)
1. No code path lets information from t' > t influence a forecast for t.
2. Every model output is a `Distribution`; no bare point arrays cross module boundaries.
3. Every result row carries seed + config hash; `make reproduce` regenerates all paper numbers.
4. Scalers/feature transforms fit on train windows only, inside the splitter's contract.

## Open questions (feed to P3)
- [ ] Distribution: closed-form CRPS for parametric families vs. sample-based fallback — accuracy/speed tradeoff
- [ ] Refit schedule API: per-model overrides?
- [ ] Multi-horizon: separate Distribution per h, or joint object?
- [ ] TSFM context construction: how much history; alignment across calendars
- [ ] R interop (rugarch) — subprocess adapter or drop?
