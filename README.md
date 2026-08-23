# volbench

**Leakage-safe, reproducible evaluation of probabilistic volatility and tail-risk forecasts — from GARCH to time-series foundation models.**

> Status: pre-alpha scaffold (v0.0.1). Built toward a submission to the *International Journal of Forecasting* special section on Open-Source Forecasting. API will change without notice until v0.1.

## Why

Volatility and tail-risk forecasts are the workhorses of risk management, yet open evaluation practice is fragmented and leakage-prone, and 2026's zero-shot time-series foundation models are untested on finance in the open. volbench provides a uniform adapter API and an evaluation engine where **temporal integrity is structural, not conventional**: all train/test indices come from one splitter whose guarantees are enforced by tests.

## Design invariants

1. **Temporal integrity.** No code path lets information from `t' > t` influence a forecast for `t`. `RollingOriginSplitter` is the only sanctioned producer of train/test indices (`tests/test_splitter.py` is the contract).
2. **Probabilistic outputs are first-class.** Every model adapter returns a `Distribution` (parametric, ensemble, or quantile-grid) — never a bare point array. Scores: CRPS (closed-form where available), log score, pinball; proxy-robust point losses (QLIKE, MSE) per Patton (2011).
3. **Determinism.** Every entry point takes a seed; results will carry config hashes; `make reproduce` must stay green.

## Quickstart

```python
import numpy as np
from volbench import Distribution, RollingOriginSplitter

splitter = RollingOriginSplitter(window=1000, horizon=1, refit_every=21)
y = np.random.default_rng(0).normal(size=3000)

for origin in splitter.split(len(y)):
    train = y[origin.train]              # everything a model may see (<= origin)
    forecast = Distribution.from_normal(mu=0.0, sigma=float(train.std()))
    score = forecast.crps(float(y[origin.test[0]]))
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

Roadmap (paper §3–§5): model adapters (`arch`/GARCH, HAR-RV, statsforecast, deep-learning route, Chronos/TimesFM/Moirai/TimeGPT), evaluation suite (VaR/ES backtests, Fissler–Ziegel, Diebold–Mariano, Model Confidence Set, economic value), scalable runner (config-hash caching, process-parallel origins, GPU batching), data adapters with explicit licensing.

## License

Apache-2.0. Data adapters ship only redistributable sources; licensed data (CRSP/Bloomberg/…) enters through a bring-your-own adapter.
