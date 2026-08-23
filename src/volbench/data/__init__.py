"""volbench.data — leakage-safe ingestion of raw market data into daily variance targets.

Every adapter here emits :class:`TimeSeriesFrame` objects on the asset's own
calendar (see ``types.py``); proxy construction (``proxies.py``) never reads
across the train/test boundary. See docs/data_licenses.md for what may be
downloaded/cached vs. must go through :mod:`volbench.data.byo`.
"""

from volbench.data.byo import load_ohlc_csv, load_ohlc_parquet
from volbench.data.crypto import (
    CRYPTO_SYMBOLS,
    BinanceDownloadError,
    BinanceUnavailableError,
    daily_realized_variance,
    load_minute_bars,
)
from volbench.data.proxies import (
    garman_klass,
    parkinson,
    realized_variance_from_bars,
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
    "CLOSE_COLUMN",
    "CRYPTO_SYMBOLS",
    "OHLC_COLUMNS",
    "STOOQ_INDEX_SYMBOLS",
    "BinanceDownloadError",
    "BinanceUnavailableError",
    "StooqBlockedError",
    "StooqDownloadError",
    "TimeSeriesFrame",
    "daily_realized_variance",
    "download_index",
    "garman_klass",
    "ingest_manual_csv",
    "load_minute_bars",
    "load_ohlc_csv",
    "load_ohlc_parquet",
    "parkinson",
    "realized_variance_from_bars",
    "squared_return",
]
