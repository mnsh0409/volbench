"""volbench — leakage-safe evaluation of probabilistic volatility & tail-risk forecasts.

What this root namespace exports, and what it deliberately does not:

- **exported**: the shared vocabulary every layer speaks in (``Distribution``
  and friends, ``TimeSeriesFrame``, ``Origin``), the sanctioned splitter, the
  baseline models, the pure variance proxies, and the evaluation/results/
  execution entry points. These are the names a user or a Phase 2 stream
  composes a benchmark out of.
- **not exported**: per-source ingestion — the Stooq and Binance downloaders,
  their error types and symbol maps, and the bring-your-own-data loaders.
  Those live in :mod:`volbench.data`, because which *source* a series came
  from is a provenance question (docs/data_licenses.md), not part of the
  library's core vocabulary, and pinning them at the root would suggest a
  stability promise the licensing situation does not support.

``__version__`` is read from the installed package metadata rather than
written here as a literal, so it cannot drift from the ``package_version()``
that goes into every ``config_hash`` (CLAUDE.md rule 3).
"""

from volbench.data import (
    TimeSeriesFrame,
    garman_klass,
    log_returns,
    overnight_plus_range_variance,
    parkinson,
    realized_variance_from_bars,
    squared_return,
)
from volbench.dist import Distribution, Empirical, Normal, QuantileGrid, StudentT
from volbench.evaluate import (
    DEFAULT_LEVELS,
    ModelFactory,
    Recondition,
    SupportsUpdate,
    forecast_moments,
    run_backtest,
)
from volbench.execute import Executor, SerialExecutor
from volbench.metrics import mse, pinball, qlike
from volbench.models import (
    EWMA,
    GARCH,
    HAR,
    Chronos,
    FittedEWMA,
    FittedGARCH,
    FittedHAR,
    FittedModel,
    FittedNaiveVol,
    FittedPatchTST,
    FittedTSFM,
    ForecastModel,
    Moirai,
    NaiveVol,
    PatchTST,
    TimeGPT,
    TimesFM,
    gjr_garch,
)
from volbench.results import (
    KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ResultsStore,
    array_digest,
    build_config,
    canonical_repr,
    config_hash,
    normalize_frame,
    package_version,
)
from volbench.splitter import Origin, RollingOriginSplitter

__version__ = package_version()

__all__ = [
    "DEFAULT_LEVELS",
    "EWMA",
    "GARCH",
    "HAR",
    "KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "Chronos",
    "Distribution",
    "Empirical",
    "Executor",
    "FittedEWMA",
    "FittedGARCH",
    "FittedHAR",
    "FittedModel",
    "FittedNaiveVol",
    "FittedPatchTST",
    "FittedTSFM",
    "ForecastModel",
    "ModelFactory",
    "Moirai",
    "NaiveVol",
    "Normal",
    "Origin",
    "PatchTST",
    "QuantileGrid",
    "Recondition",
    "ResultsStore",
    "RollingOriginSplitter",
    "SerialExecutor",
    "StudentT",
    "SupportsUpdate",
    "TimeGPT",
    "TimeSeriesFrame",
    "TimesFM",
    "__version__",
    "array_digest",
    "build_config",
    "canonical_repr",
    "config_hash",
    "forecast_moments",
    "garman_klass",
    "gjr_garch",
    "log_returns",
    "mse",
    "normalize_frame",
    "overnight_plus_range_variance",
    "package_version",
    "parkinson",
    "pinball",
    "qlike",
    "realized_variance_from_bars",
    "run_backtest",
    "squared_return",
]
