"""volbench — leakage-safe evaluation of probabilistic volatility & tail-risk forecasts.

What this root namespace exports, and what it deliberately does not:

- **exported**: the shared vocabulary every layer speaks in (``Distribution``
  and friends, ``TimeSeriesFrame``, ``Origin``), the sanctioned splitter,
  every model adapter and its fitted type, the pure variance proxies, the
  evaluation/results/execution entry points, the invalid-target policy
  (``FitSeries`` and friends) that says what an unusable variance day does to
  a fit window, the grid runner (``run_grid``, ``GridSpec``, ``ModelConfig``,
  ``ProtocolArm``, ``AssetData``, ``RunManifest``) and the local
  multiprocessing backend (``ProcessExecutor``), and — since Phase 2 — the
  inference (``diebold_mariano``, ``model_confidence_set``,
  ``compare_models``), VaR-backtest (``kupiec_pof``, ``christoffersen``,
  ``fz0_loss``, ``var_backtest``) and economic-value
  (``volatility_target_backtest``) entry points that consume the evaluation's
  rows. These are the names a user or a Phase 2 stream composes a benchmark
  out of. Adapters whose backend is optional import it lazily, so none of
  these imports needs an extra installed.
- **not exported**: per-source ingestion — the Stooq and Binance downloaders,
  their error types and symbol maps, the bring-your-own-data loaders, and the
  study's own panel assembly (``volbench.data.panel`` / ``crisis`` /
  ``diagnostics``). Those live in :mod:`volbench.data`, because which
  *source* a series came from is a provenance question
  (docs/data_licenses.md), not part of the library's core vocabulary, and
  pinning them at the root would suggest a stability promise the licensing
  situation does not support.

``__version__`` is read from the installed package metadata rather than
written here as a literal, so it cannot drift from the ``package_version()``
that goes into every ``config_hash`` (CLAUDE.md rule 3).
"""

from volbench.backtests import (
    ChristoffersenResult,
    KupiecResult,
    VaRBacktest,
    christoffersen,
    expected_shortfall,
    fz0_loss,
    kupiec_pof,
    var_backtest,
)
from volbench.compaction import (
    DEFAULT_INVALID_TARGET_POLICY,
    FitSeries,
    InsufficientHistoryError,
    InvalidTargetPolicy,
    invalid_target_mask,
    valid_target_mask,
)
from volbench.data import (
    TimeSeriesFrame,
    garman_klass,
    log_returns,
    overnight_plus_range_variance,
    overnight_variance,
    parkinson,
    realized_variance_from_bars,
    rogers_satchell,
    squared_return,
)
from volbench.dist import Distribution, Empirical, Normal, QuantileGrid, StudentT
from volbench.econ import VolTargetBacktest, periods_per_year_for, volatility_target_backtest
from volbench.evaluate import (
    DEFAULT_LEVELS,
    ModelFactory,
    Recondition,
    SupportsUpdate,
    forecast_moments,
    run_backtest,
)
from volbench.execute import Executor, ProcessExecutor, SerialExecutor
from volbench.inference import (
    DMMatrix,
    DMResult,
    LossMatrix,
    MCSResult,
    ModelComparison,
    compare_models,
    diebold_mariano,
    dm_matrix,
    loss_matrix,
    model_confidence_set,
)
from volbench.metrics import mse, pinball, qlike
from volbench.models import (
    EWMA,
    GARCH,
    HAR,
    AutoARIMARV,
    AutoETSRV,
    Chronos,
    FittedEWMA,
    FittedGARCH,
    FittedHAR,
    FittedLightGBMRV,
    FittedModel,
    FittedNaiveVol,
    FittedPatchTST,
    FittedStatsForecastRV,
    FittedTSFM,
    ForecastModel,
    LightGBMRV,
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
from volbench.runner import (
    AssetData,
    CellOutcome,
    DataSource,
    GridSpec,
    MappingDataSource,
    ModelConfig,
    ProtocolArm,
    RunManifest,
    read_grid_results,
    run_grid,
)
from volbench.splitter import Origin, RollingOriginSplitter

__version__ = package_version()

__all__ = [
    "DEFAULT_INVALID_TARGET_POLICY",
    "DEFAULT_LEVELS",
    "EWMA",
    "GARCH",
    "HAR",
    "KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "AssetData",
    "AutoARIMARV",
    "AutoETSRV",
    "CellOutcome",
    "ChristoffersenResult",
    "Chronos",
    "DMMatrix",
    "DMResult",
    "DataSource",
    "Distribution",
    "Empirical",
    "Executor",
    "FitSeries",
    "FittedEWMA",
    "FittedGARCH",
    "FittedHAR",
    "FittedLightGBMRV",
    "FittedModel",
    "FittedNaiveVol",
    "FittedPatchTST",
    "FittedStatsForecastRV",
    "FittedTSFM",
    "ForecastModel",
    "GridSpec",
    "InsufficientHistoryError",
    "InvalidTargetPolicy",
    "KupiecResult",
    "LightGBMRV",
    "LossMatrix",
    "MCSResult",
    "MappingDataSource",
    "ModelComparison",
    "ModelConfig",
    "ModelFactory",
    "Moirai",
    "NaiveVol",
    "Normal",
    "Origin",
    "PatchTST",
    "ProcessExecutor",
    "ProtocolArm",
    "QuantileGrid",
    "Recondition",
    "ResultsStore",
    "RollingOriginSplitter",
    "RunManifest",
    "SerialExecutor",
    "StudentT",
    "SupportsUpdate",
    "TimeGPT",
    "TimeSeriesFrame",
    "TimesFM",
    "VaRBacktest",
    "VolTargetBacktest",
    "__version__",
    "array_digest",
    "build_config",
    "canonical_repr",
    "christoffersen",
    "compare_models",
    "config_hash",
    "diebold_mariano",
    "dm_matrix",
    "expected_shortfall",
    "forecast_moments",
    "fz0_loss",
    "garman_klass",
    "gjr_garch",
    "invalid_target_mask",
    "kupiec_pof",
    "log_returns",
    "loss_matrix",
    "model_confidence_set",
    "mse",
    "normalize_frame",
    "overnight_plus_range_variance",
    "overnight_variance",
    "package_version",
    "parkinson",
    "periods_per_year_for",
    "pinball",
    "qlike",
    "read_grid_results",
    "realized_variance_from_bars",
    "rogers_satchell",
    "run_backtest",
    "run_grid",
    "squared_return",
    "valid_target_mask",
    "var_backtest",
    "volatility_target_backtest",
]
