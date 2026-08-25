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
uv sync --dev --extra classical          # + --extra torch-cpu (CI) or --extra tsfm (GPU box)
make check                               # ruff, mypy --strict, pytest; EXTRAS="..." to change the extras
make smoke-tsfm                          # local only: foundation models + PatchTST on the toy series
```

**Built (Phase 1 + Phase 2 core, v0.3.0):** data adapters with explicit
licensing and the D-004/D-012 evaluation panel (`volbench.data`), daily
variance proxies, thirteen model adapters — naive, EWMA, GARCH/GJR-GARCH,
HAR-RV, AutoETS/AutoARIMA (statsforecast), LightGBM, PatchTST, and the
zero-shot foundation models Chronos, TimesFM, Moirai and TimeGPT —
rolling-origin backtesting with CRPS / log score / QLIKE / pinball / VaR hits,
the comparison-inference suite (Diebold–Mariano, Model Confidence Set) and
VaR/ES backtests (Kupiec, Christoffersen, FZ0 loss), a content-addressed
results store, and a serial execution seam. Optional backends live in the
`classical`, `tsfm` and `torch-cpu` extras (see the `Makefile`).

**Roadmap (paper §3–§5):** the scalable runner (process-parallel and
Slurm-array executors, GPU batching), the full-panel grid, and economic-value
evaluation. See `docs/P2_INTEGRATION.md` for what is and is not built after
Phase 2, `docs/M1_REPORT.md` for Phase 1, and `docs/design.md` for the
as-built API.

## License

Apache-2.0. Data adapters ship only redistributable sources; licensed data (CRSP/Bloomberg/…) enters through a bring-your-own adapter.
