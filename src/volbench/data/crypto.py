"""Binance 1-minute klines -> daily realized variance for the crypto RV arm.

STATUS AS VERIFIED 2026-08-23 (see docs/data_licenses.md): both the bulk
archive at ``data.binance.vision`` and the public ``api.binance.com``
klines endpoint answered plain HTTP requests normally (checked with
``curl``) — unlike Stooq, Binance actively documents this bulk-download
path (github.com/binance/binance-public-data, MIT-licensed *client code*).
However, no explicit statement about redistributing the *data itself* (or
series derived from it) could be found in the portions of Binance's Terms
of Use retrieved during this review. Treat the redistribution/vendoring
question as unresolved pending a human legal read — this module only ever
caches raw bars locally (gitignored, never committed, see CLAUDE.md) and
hands callers back a heavily aggregated derived series (daily realized
variance), never the raw bars themselves as a shipped artifact.

USD proxy note: Binance has no comparably liquid direct fiat-USD BTC/ETH
pair, so BTCUSDT/ETHUSDT (quoted in the USDT stablecoin) stand in for the
USD-denominated series, consistent with common practice in the realized-
volatility literature. This is a modeling approximation, not an exact
substitution, and should be stated as such wherever results are reported.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from volbench.data.proxies import realized_variance_from_bars

__all__ = [
    "CRYPTO_SYMBOLS",
    "BinanceDownloadError",
    "BinanceUnavailableError",
    "daily_realized_variance",
    "fetch_and_cache_day",
    "fetch_daily_klines_zip",
    "load_minute_bars",
    "parse_klines_zip",
]

BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/spot/daily/klines"

#: canonical asset id -> Binance spot symbol.
CRYPTO_SYMBOLS: dict[str, str] = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

#: A contact address in a scraper's User-Agent is courtesy to the operator of
#: the endpoint, and this one used to be hardcoded. It is read from the
#: environment now: IJF review is double-blind, this file ships in the
#: reproducibility package, and an email address in it identifies the author
#: (``tests/test_identity_leakage.py``). Unset, the header still names the tool
#: and its version, which is the part the endpoint actually needs.
_CONTACT = os.environ.get("VOLBENCH_CONTACT", "").strip()
_REQUEST_HEADERS = {
    "User-Agent": "volbench-research/0.1" + (f" (+mailto:{_CONTACT})" if _CONTACT else "")
}


class BinanceDownloadError(RuntimeError):
    """Raised when a Binance archive file is unreachable or unparseable."""


class BinanceUnavailableError(BinanceDownloadError):
    """Raised when a given (symbol, day) has no archive file (pre-listing, future, or gap)."""


def _daily_zip_url(symbol: str, day: date) -> str:
    return f"{BINANCE_ARCHIVE_BASE}/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"


def fetch_daily_klines_zip(
    symbol: str, day: date, *, session: requests.Session | None = None, timeout: float = 30.0
) -> bytes:
    """Download one day's 1-minute klines archive (zip) for ``symbol``."""
    http = session or requests
    resp = http.get(_daily_zip_url(symbol, day), headers=_REQUEST_HEADERS, timeout=timeout)
    if resp.status_code == 404:
        raise BinanceUnavailableError(
            f"no {symbol} 1m archive for {day.isoformat()} "
            "(before listing date, in the future, or a data gap)"
        )
    resp.raise_for_status()
    if not resp.content:
        raise BinanceDownloadError(f"empty archive for {symbol} on {day.isoformat()}")
    return resp.content


def _epoch_to_utc_index(epoch: NDArray[np.int64]) -> pd.DatetimeIndex:
    # Binance klines timestamps are milliseconds by convention; some newer
    # endpoints emit microseconds. Millisecond epochs for 2005-2100 are
    # O(1e12-1e13); microsecond epochs are O(1e15-1e16) - detect by magnitude.
    unit: Literal["us", "ms"] = "us" if epoch.size and epoch[0] > 10**14 else "ms"
    idx: pd.DatetimeIndex = pd.to_datetime(epoch, unit=unit, utc=True)
    return idx


def parse_klines_zip(raw: bytes) -> pd.DataFrame:
    """Parse a Binance daily klines zip archive into an OHLCV DataFrame.

    Returns a DataFrame indexed by each bar's UTC open-time, sorted ascending,
    with float ``open``/``high``/``low``/``close``/``volume`` columns.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise BinanceDownloadError(f"expected exactly one member in klines zip, got {names}")
        csv_bytes = zf.read(names[0])

    df = pd.read_csv(io.BytesIO(csv_bytes), header=None, names=_KLINE_COLUMNS)
    if df.empty:
        raise BinanceDownloadError("klines archive contained no rows")

    out = df[["open", "high", "low", "close", "volume"]].astype(np.float64)
    out.index = _epoch_to_utc_index(df["open_time"].to_numpy())
    out.index.name = "timestamp"
    return out.sort_index()


def _cache_paths(cache_dir: Path, symbol: str, day: date) -> tuple[Path, Path]:
    stem = f"binance_{symbol.lower()}_1m_{day.isoformat()}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def fetch_and_cache_day(
    symbol: str,
    day: date,
    *,
    cache_dir: Path | str = Path("data/cache/crypto"),
    session: requests.Session | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch (or load from cache) one day's 1-minute OHLCV bars for ``symbol``.

    Cached under ``cache_dir`` keyed by symbol + calendar day (the archive
    file for a past day is immutable, unlike Stooq's always-current export),
    with a JSON sidecar carrying the SHA256 of the raw zip. Never committed.
    """
    cache_dir_path = Path(cache_dir)
    parquet_path, meta_path = _cache_paths(cache_dir_path, symbol, day)
    if not force and parquet_path.exists() and meta_path.exists():
        df: pd.DataFrame = pd.read_parquet(parquet_path)
        return df

    raw = fetch_daily_klines_zip(symbol, day, session=session)
    sha256 = hashlib.sha256(raw).hexdigest()
    parsed = parse_klines_zip(raw)

    cache_dir_path.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "day": day.isoformat(),
                "downloaded_at": datetime.now(tz=UTC).isoformat(),
                "sha256": sha256,
                "url": _daily_zip_url(symbol, day),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return parsed


def load_minute_bars(
    symbol: str,
    start: date,
    end: date,
    *,
    cache_dir: Path | str = Path("data/cache/crypto"),
    session: requests.Session | None = None,
) -> pd.Series:
    """Concatenate cached/fetched 1-minute close prices for ``symbol`` over ``[start, end]``."""
    if end < start:
        raise ValueError("end must be >= start")

    frames = []
    day = start
    while day <= end:
        frames.append(fetch_and_cache_day(symbol, day, cache_dir=cache_dir, session=session))
        day += timedelta(days=1)

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    series: pd.Series = combined["close"].rename(f"{symbol}_close")
    return series


def daily_realized_variance(
    asset_id: str,
    start: date,
    end: date,
    *,
    sample_interval: str = "5min",
    min_bars: int = 2,
    cache_dir: Path | str = Path("data/cache/crypto"),
    session: requests.Session | None = None,
) -> pd.Series:
    """Daily realized variance for ``asset_id`` (``"BTC-USD"``/``"ETH-USD"``) over ``[start, end]``.

    1-minute bars are resampled to ``sample_interval`` (last price in each
    bucket) before computing RV, trading estimator noise for robustness to
    microstructure effects at the finest granularity — the standard sparse-
    sampling tradeoff for realized variance (see docs/metrics_reference.md).

    ``sample_interval`` must evenly divide one day. Otherwise a resample
    bucket can straddle midnight (pandas anchors bins to a fixed origin and
    steps by a constant frequency, so a non-divisor interval eventually
    produces a bin like ``[23:58, 00:05)``), and ``.last()`` on that bucket
    could pick a bar from day t+1 while ``realized_variance_from_bars``
    labels it day t via the bucket's own timestamp — a real, if narrow,
    leakage path (caught by leakage-check, not merely a style nit).
    """
    if asset_id not in CRYPTO_SYMBOLS:
        raise KeyError(f"unknown asset_id {asset_id!r}; known assets: {sorted(CRYPTO_SYMBOLS)}")
    interval = pd.Timedelta(sample_interval)
    if interval <= pd.Timedelta(0) or pd.Timedelta(days=1) % interval != pd.Timedelta(0):
        raise ValueError(
            f"sample_interval={sample_interval!r} must evenly divide one day "
            "(e.g. '1min', '5min', '15min', '1h') so no resample bucket can "
            "straddle midnight and leak a day t+1 bar into day t's RV"
        )
    symbol = CRYPTO_SYMBOLS[asset_id]
    bars = load_minute_bars(symbol, start, end, cache_dir=cache_dir, session=session)
    sampled = bars.resample(sample_interval).last().dropna()
    return realized_variance_from_bars(sampled, min_bars=min_bars)
