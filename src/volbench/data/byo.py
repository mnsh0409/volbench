"""Bring-your-own-data adapter.

Use this for any source whose terms do not clearly permit programmatic
download and redistribution (docs/data_licenses.md's rule): the user
supplies their own OHLC/close data — CRSP, Bloomberg, Refinitiv, a
manually-downloaded CSV, or anything else they are personally licensed to
use — and volbench never downloads or vendors it. This is also the fallback
path noted in stooq.py while that source's anti-bot gate blocks automated
access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from volbench.data.types import TimeSeriesFrame

__all__ = ["load_ohlc_csv", "load_ohlc_parquet"]

_KNOWN_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlc_csv(
    path: Path | str,
    *,
    asset_id: str,
    source: str = "byo",
    timestamp_column: str = "date",
    tz: str | None = None,
) -> TimeSeriesFrame:
    """Load a user-supplied OHLC/close CSV into a :class:`TimeSeriesFrame`.

    Columns are matched case-insensitively; only recognized OHLCV columns are
    kept. If ``tz`` is given, timestamps are assumed naive-local in that zone
    and converted to UTC; otherwise they are parsed as already tz-aware.
    The caller is responsible for the data's license — this function only
    parses and validates, it never fetches anything.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if timestamp_column not in df.columns:
        raise ValueError(
            f"missing timestamp column {timestamp_column!r}; found {list(df.columns)}"
        )

    if tz is not None:
        naive = pd.to_datetime(df[timestamp_column])
        timestamps = naive.dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        timestamps = pd.to_datetime(df[timestamp_column], utc=True)

    keep = [c for c in _KNOWN_PRICE_COLUMNS if c in df.columns]
    if not keep:
        raise ValueError(
            f"no recognized OHLCV columns found among {list(df.columns)}; "
            f"expected one of {_KNOWN_PRICE_COLUMNS}"
        )
    out = df[keep].copy()
    out.index = pd.DatetimeIndex(timestamps)
    out.index.name = "timestamp"
    return TimeSeriesFrame(data=out, asset_id=asset_id, source=source)


def load_ohlc_parquet(
    path: Path | str, *, asset_id: str, source: str = "byo"
) -> TimeSeriesFrame:
    """Load a user-supplied OHLC/close parquet file into a :class:`TimeSeriesFrame`.

    The file must already have a tz-aware DatetimeIndex and OHLC/close
    columns — this is a thin, validating wrapper, not a format converter.
    """
    df = pd.read_parquet(path)
    return TimeSeriesFrame(data=df, asset_id=asset_id, source=source)
