"""Binance klines -> daily RV: parsing, caching, and the resample-then-RV pipeline.

No network access here: a small fake "session" object serves canned zip bytes
keyed by the exact URL the code would request, built from a committed 1-minute
klines fixture (never a live download).
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volbench.data.crypto import (
    CRYPTO_SYMBOLS,
    BinanceUnavailableError,
    _daily_zip_url,
    _epoch_to_utc_index,
    daily_realized_variance,
    fetch_and_cache_day,
    load_minute_bars,
    parse_klines_zip,
)

FIXTURE = Path(__file__).parent / "fixtures" / "binance_btcusdt_1m_sample.csv"
DAY1 = date(2024, 1, 2)
DAY2 = date(2024, 1, 3)
_ONE_DAY_MS = 86_400_000


def _zip_bytes(csv_text: str, member_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member_name, csv_text)
    return buf.getvalue()


def _shift_epoch_columns(csv_text: str, offset_ms: int) -> str:
    lines = []
    for line in csv_text.strip().splitlines():
        parts = line.split(",")
        parts[0] = str(int(parts[0]) + offset_ms)  # open_time
        parts[6] = str(int(parts[6]) + offset_ms)  # close_time
        lines.append(",".join(parts))
    return "\n".join(lines) + "\n"


def _day1_zip() -> bytes:
    return _zip_bytes(FIXTURE.read_text(), "BTCUSDT-1m-2024-01-02.csv")


def _day2_zip() -> bytes:
    shifted = _shift_epoch_columns(FIXTURE.read_text(), _ONE_DAY_MS)
    return _zip_bytes(shifted, "BTCUSDT-1m-2024-01-03.csv")


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 404:
            raise RuntimeError(f"fake http error {self.status_code}")


class _FakeSession:
    """Stands in for requests.Session: serves canned bytes keyed by URL, no network."""

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str] | None = None, timeout: float | None = None):
        self.calls.append(url)
        if url not in self._responses:
            return _FakeResponse(status_code=404, content=b"")
        return _FakeResponse(status_code=200, content=self._responses[url])


class TestSymbolMap:
    def test_expected_assets(self) -> None:
        assert CRYPTO_SYMBOLS == {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}


class TestEpochUnitDetection:
    def test_milliseconds(self) -> None:
        idx = _epoch_to_utc_index(np.array([1704153600000, 1704153660000]))
        assert idx[0] == pd.Timestamp("2024-01-02T00:00:00Z")
        assert idx[1] == pd.Timestamp("2024-01-02T00:01:00Z")

    def test_microseconds(self) -> None:
        idx = _epoch_to_utc_index(np.array([1704153600000000, 1704153660000000]))
        assert idx[0] == pd.Timestamp("2024-01-02T00:00:00Z")
        assert idx[1] == pd.Timestamp("2024-01-02T00:01:00Z")


class TestParseKlinesZip:
    def test_parses_fixture(self) -> None:
        df = parse_klines_zip(_day1_zip())
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 10
        assert df.index.is_monotonic_increasing
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2024-01-02T00:00:00Z")
        assert df["close"].iloc[0] == pytest.approx(42005.00)


class TestFetchAndCacheDay:
    def test_fetches_parses_and_caches(self, tmp_path: Path) -> None:
        url = _daily_zip_url("BTCUSDT", DAY1)
        session = _FakeSession({url: _day1_zip()})

        df = fetch_and_cache_day("BTCUSDT", DAY1, cache_dir=tmp_path, session=session)
        assert len(df) == 10
        assert len(session.calls) == 1

        parquet_files = list(tmp_path.glob("*.parquet"))
        meta_files = list(tmp_path.glob("*.json"))
        assert len(parquet_files) == 1
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["symbol"] == "BTCUSDT"
        assert meta["day"] == DAY1.isoformat()

    def test_second_call_hits_cache_not_network(self, tmp_path: Path) -> None:
        url = _daily_zip_url("BTCUSDT", DAY1)
        session = _FakeSession({url: _day1_zip()})

        fetch_and_cache_day("BTCUSDT", DAY1, cache_dir=tmp_path, session=session)
        fetch_and_cache_day("BTCUSDT", DAY1, cache_dir=tmp_path, session=session)
        assert len(session.calls) == 1

    def test_missing_day_raises_unavailable(self, tmp_path: Path) -> None:
        session = _FakeSession({})
        with pytest.raises(BinanceUnavailableError):
            fetch_and_cache_day("BTCUSDT", DAY1, cache_dir=tmp_path, session=session)


class TestLoadMinuteBars:
    def test_concatenates_across_days(self, tmp_path: Path) -> None:
        session = _FakeSession(
            {
                _daily_zip_url("BTCUSDT", DAY1): _day1_zip(),
                _daily_zip_url("BTCUSDT", DAY2): _day2_zip(),
            }
        )
        bars = load_minute_bars("BTCUSDT", DAY1, DAY2, cache_dir=tmp_path, session=session)
        assert len(bars) == 20
        assert bars.index.is_monotonic_increasing
        assert not bars.index.has_duplicates

    def test_end_before_start_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="end must be >= start"):
            load_minute_bars("BTCUSDT", DAY2, DAY1, cache_dir=tmp_path, session=_FakeSession({}))


class TestDailyRealizedVariance:
    def test_end_to_end_pipeline(self, tmp_path: Path) -> None:
        session = _FakeSession(
            {
                _daily_zip_url("BTCUSDT", DAY1): _day1_zip(),
                _daily_zip_url("BTCUSDT", DAY2): _day2_zip(),
            }
        )
        rv = daily_realized_variance(
            "BTC-USD", DAY1, DAY2, sample_interval="5min", min_bars=2,
            cache_dir=tmp_path, session=session,
        )
        assert len(rv) == 2
        assert (rv.dropna() >= 0.0).all()
        assert rv.notna().all()

    def test_unknown_asset_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            daily_realized_variance(
                "DOGE-USD", DAY1, DAY2, cache_dir=tmp_path, session=_FakeSession({})
            )

    def test_non_day_dividing_interval_rejected(self, tmp_path: Path) -> None:
        # A "7min" grid does not evenly divide 24h, so pandas' fixed-origin
        # resample bins can straddle midnight; that could pick a day t+1 bar
        # into a bucket realized_variance_from_bars labels as day t. Reject
        # up front rather than silently leaking (see leakage-check).
        with pytest.raises(ValueError, match="evenly divide one day"):
            daily_realized_variance(
                "BTC-USD", DAY1, DAY2, sample_interval="7min",
                cache_dir=tmp_path, session=_FakeSession({}),
            )

    def test_zero_interval_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="evenly divide one day"):
            daily_realized_variance(
                "BTC-USD", DAY1, DAY2, sample_interval="0min",
                cache_dir=tmp_path, session=_FakeSession({}),
            )
