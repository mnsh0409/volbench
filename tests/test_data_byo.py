"""Bring-your-own-data adapter: the fallback path for sources of ambiguous license."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volbench.data import TimeSeriesFrame, load_ohlc_csv, load_ohlc_parquet

FIXTURE = Path(__file__).parent / "fixtures" / "stooq_sample.csv"


class TestLoadOhlcCsv:
    def test_loads_and_validates(self) -> None:
        frame = load_ohlc_csv(FIXTURE, asset_id="SPX", timestamp_column="date")
        assert isinstance(frame, TimeSeriesFrame)
        assert frame.source == "byo"
        assert frame.has_ohlc
        assert len(frame) == 5
        assert str(frame.index.tz) == "UTC"

    def test_custom_source_tag(self) -> None:
        frame = load_ohlc_csv(FIXTURE, asset_id="SPX", source="crsp", timestamp_column="date")
        assert frame.source == "crsp"

    def test_missing_timestamp_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing timestamp column"):
            load_ohlc_csv(FIXTURE, asset_id="SPX", timestamp_column="not_a_column")

    def test_no_recognized_price_columns_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.csv"
        path.write_text("date,foo\n2024-01-01,1.0\n2024-01-02,2.0\n")
        with pytest.raises(ValueError, match="no recognized OHLCV columns"):
            load_ohlc_csv(path, asset_id="X")

    def test_naive_local_timestamps_converted_with_tz(self, tmp_path: Path) -> None:
        path = tmp_path / "naive.csv"
        path.write_text("date,close\n2024-01-02 09:30,100.0\n2024-01-02 16:00,101.0\n")
        frame = load_ohlc_csv(path, asset_id="X", tz="America/New_York")
        assert str(frame.index.tz) == "UTC"
        # 09:30 EST (UTC-5 in January) -> 14:30 UTC
        assert frame.index[0] == pd.Timestamp("2024-01-02T14:30:00Z")


class TestLoadOhlcParquet:
    def test_roundtrip(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        path = tmp_path / "prices.parquet"
        df.to_parquet(path)

        frame = load_ohlc_parquet(path, asset_id="X", source="crsp")
        assert len(frame) == 3
        assert frame.source == "crsp"
