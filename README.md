# volbench

**Leakage-safe, reproducible evaluation of probabilistic volatility and tail-risk forecasts — from GARCH to time-series foundation models.**

> Status: pre-alpha (v0.1.0-m1 — Phase 1 complete). Built toward a submission to the *International Journal of Forecasting* special section on Open-Source Forecasting. API will change without notice until v1.0.

## Why

Volatility and tail-risk forecasts are the workhorses of risk management, yet open evaluation practice is fragmented and leakage-prone, and 2026's zero-shot time-series foundation models are untested on finance in the open. volbench provides a uniform adapter API and an evaluation engine where **temporal integrity is structural, not conventional**: all train/test indices come from one splitter whose guarantees are enforced by tests.

## Design invariants

1. **Temporal integrity.** No code path lets information from `t' > t` influence a forecast for `t`. `RollingOriginSplitter` is the only sanctioned producer of train/test indices (`tests/test_splitter.py` is the contract).
2. **Probabilistic outputs are first-class.** Every model adapter returns a `Distribution` (parametric, ensemble, or quantile-grid) — never a bare point array. Scores: CRPS (closed-form where available), log score, pinball; proxy-robust point losses (QLIKE, MSE) per Patton (2011).
3. **Determinism.** Every entry point takes a seed; every result row carries a config hash over the model spec, the data's *content* digest, the splitter, the seed and the package version; `make reproduce` must stay green.

## Quickstart

```python
import numpy as np
from volbench import GARCH, RollingOriginSplitter, run_backtest

returns = np.random.default_rng(0).normal(0, 0.01, size=1500)   # daily log returns
proxy = returns**2                                              # daily variance proxy

results = run_backtest(
    GARCH,                                                      # a zero-arg model factory
    returns,
    proxy,
    RollingOriginSplitter(window=1000, horizon=1, refit_every=21),
    seed=0,
    asset="DEMO",
    proxy_name="squared_return",
)
print(results[["origin_index", "forecast_var", "crps", "qlike", "hit_0p01"]].head())
```

A model is anything with `name`, `spec()` and `fit(train) -> FittedModel`, where
`FittedModel.predict(h)` returns a `Distribution` over the **next-period return**
— its variance is the variance forecast, always in daily units.

To see the whole pipeline run end to end:

```bash
make reproduce      # checks, then rebuilds the toy benchmark from scratch
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

**Built (Phase 1):** data adapters with explicit licensing (`volbench.data`),
daily variance proxies, baseline models (naive, EWMA, GARCH/GJR-GARCH, HAR-RV),
rolling-origin backtesting with CRPS / log score / QLIKE / pinball / VaR hits,
a content-addressed results store, and a serial execution seam.

**Roadmap (paper §3–§5):** more model adapters (statsforecast, LightGBM,
PatchTST, Chronos/TimesFM/Moirai/TimeGPT); the comparison-inference suite
(Fissler–Ziegel, Kupiec, Christoffersen, Diebold–Mariano, Model Confidence Set,
economic value); and the scalable runner (process-parallel and Slurm-array
executors, GPU batching). See `docs/M1_REPORT.md` for what is and is not built,
and `docs/design.md` for the as-built API.

## License

Apache-2.0. Data adapters ship only redistributable sources; licensed data (CRSP/Bloomberg/…) enters through a bring-your-own adapter.
