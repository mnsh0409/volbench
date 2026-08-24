"""volbench.data — leakage-safe ingestion of raw market data into daily variance targets.

Every adapter here emits :class:`TimeSeriesFrame` objects on the asset's own
calendar (see ``types.py``); proxy construction (``proxies.py``) never reads
across the train/test boundary. See docs/data_licenses.md for what may be
downloaded/cached vs. must go through :mod:`volbench.data.byo`.

``panel.py`` assembles the study's actual asset list (D-004/D-012) out of those
adapters, ``diagnostics.py`` measures what it built, and ``crisis.py`` carries
the crisis-window labels — metadata about dates, never an input to fitting.
"""

from volbench.data.byo import load_ohlc_csv, load_ohlc_parquet
from volbench.data.crisis import (
    CALM_TAG,
    CRISIS_WINDOWS,
    PENDING_WINDOWS,
    CrisisWindow,
    PendingWindow,
    crisis_mask,
    crisis_table,
    tag_dates,
    window_by_tag,
)
from volbench.data.crypto import (
    CRYPTO_SYMBOLS,
    BinanceDownloadError,
    BinanceUnavailableError,
    daily_realized_variance,
    load_minute_bars,
)
from volbench.data.diagnostics import (
    SeriesDiagnostics,
    diagnose,
    diagnose_panel,
    diagnostics_frame,
)
from volbench.data.panel import (
    CRYPTO_PANEL,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RAW_ROOT,
    EQUITY_PANEL,
    PANEL_END,
    PANEL_START,
    TARGET_NAMES,
    BarQuality,
    CryptoSpec,
    EquitySpec,
    PanelSeries,
    build_crypto_series,
    build_equity_series,
    build_panel,
    build_targets,
    repair_bars,
)
from volbench.data.proxies import (
    garman_klass,
    log_returns,
    overnight_plus_range_variance,
    overnight_variance,
    parkinson,
    realized_variance_from_bars,
    rogers_satchell,
    squared_return,
)
from volbench.data.stooq import (
    STOOQ_INDEX_SYMBOLS,
    StooqBlockedError,
    StooqDownloadError,
    download_index,
    ingest_manual_csv,
)
from volbench.data.types import CLOSE_COLUMN, OHLC_COLUMNS, TimeSeriesFrame

__all__ = [
    "CALM_TAG",
    "CLOSE_COLUMN",
    "CRISIS_WINDOWS",
    "CRYPTO_PANEL",
    "CRYPTO_SYMBOLS",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_RAW_ROOT",
    "EQUITY_PANEL",
    "OHLC_COLUMNS",
    "PANEL_END",
    "PANEL_START",
    "PENDING_WINDOWS",
    "STOOQ_INDEX_SYMBOLS",
    "TARGET_NAMES",
    "BarQuality",
    "BinanceDownloadError",
    "BinanceUnavailableError",
    "CrisisWindow",
    "CryptoSpec",
    "EquitySpec",
    "PanelSeries",
    "PendingWindow",
    "SeriesDiagnostics",
    "StooqBlockedError",
    "StooqDownloadError",
    "TimeSeriesFrame",
    "build_crypto_series",
    "build_equity_series",
    "build_panel",
    "build_targets",
    "crisis_mask",
    "crisis_table",
    "daily_realized_variance",
    "diagnose",
    "diagnose_panel",
    "diagnostics_frame",
    "download_index",
    "garman_klass",
    "ingest_manual_csv",
    "load_minute_bars",
    "load_ohlc_csv",
    "load_ohlc_parquet",
    "log_returns",
    "overnight_plus_range_variance",
    "overnight_variance",
    "parkinson",
    "realized_variance_from_bars",
    "repair_bars",
    "rogers_satchell",
    "squared_return",
    "tag_dates",
    "window_by_tag",
]
