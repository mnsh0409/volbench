"""TimeSeriesFrame: validation is the leakage/data-quality contract for the data layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volbench.data import TimeSeriesFrame


def _close_only(n: int = 5, tz: str = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz=tz)
    return pd.DataFrame({"close": np.linspace(100.0, 100.0 + n, n)}, index=idx)


def _ohlc(n: int = 5, tz: str = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz=tz)
    base = np.linspace(100.0, 100.0 + n, n)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.arange(n, dtype=np.float64),
        },
        index=idx,
    )


class TestConstruction:
    def test_close_only_frame_constructs(self) -> None:
        frame = TimeSeriesFrame(data=_close_only(), asset_id="SPX", source="stooq")
        assert len(frame) == 5
        assert not frame.has_ohlc
        pd.testing.assert_series_equal(frame.close, frame.data["close"], check_names=False)

    def test_ohlc_frame_constructs(self) -> None:
        frame = TimeSeriesFrame(data=_ohlc(), asset_id="SPX", source="stooq")
        assert frame.has_ohlc
        for prop, col in (
            (frame.open, "open"),
            (frame.high, "high"),
            (frame.low, "low"),
            (frame.close, "close"),
        ):
            pd.testing.assert_series_equal(prop, frame.data[col], check_names=False)

    def test_accessing_missing_ohlc_column_raises(self) -> None:
        frame = TimeSeriesFrame(data=_close_only(), asset_id="SPX", source="stooq")
        with pytest.raises(ValueError, match="no 'open' column"):
            _ = frame.open


class TestValidationRejectsBadInput:
    def test_missing_close_and_ohlc_rejected(self) -> None:
        df = pd.DataFrame(
            {"foo": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        )
        with pytest.raises(ValueError, match="requires a 'close' column"):
            TimeSeriesFrame(data=df, asset_id="X", source="byo")

    def test_naive_index_rejected(self) -> None:
        df = _close_only()
        df.index = df.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")

    def test_non_datetime_index_rejected(self) -> None:
        df = _close_only()
        df.index = pd.RangeIndex(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")

    def test_empty_frame_rejected(self) -> None:
        df = _close_only(n=0)
        with pytest.raises(ValueError, match="at least one observation"):
            TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")

    def test_unsorted_index_rejected(self) -> None:
        df = _close_only()
        shuffled = df.iloc[[0, 2, 1, 3, 4]]
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeSeriesFrame(data=shuffled, asset_id="SPX", source="stooq")

    def test_duplicate_timestamps_rejected(self) -> None:
        df = _close_only()
        dup = pd.concat([df, df.iloc[[0]]]).sort_index()
        with pytest.raises(ValueError, match="duplicate"):
            TimeSeriesFrame(data=dup, asset_id="SPX", source="stooq")

    def test_nan_in_close_rejected(self) -> None:
        df = _close_only()
        df.iloc[2, df.columns.get_loc("close")] = np.nan
        with pytest.raises(ValueError, match="contain NaN"):
            TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")

    def test_nan_in_ohlc_rejected(self) -> None:
        df = _ohlc()
        df.iloc[1, df.columns.get_loc("high")] = np.nan
        with pytest.raises(ValueError, match="contain NaN"):
            TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")

    def test_nan_in_non_required_extra_column_is_allowed(self) -> None:
        df = _ohlc()
        df.iloc[1, df.columns.get_loc("volume")] = np.nan
        frame = TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")
        assert len(frame) == len(df)

    def test_empty_asset_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            TimeSeriesFrame(data=_close_only(), asset_id="", source="stooq")

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source"):
            TimeSeriesFrame(data=_close_only(), asset_id="SPX", source="")


class TestImmutabilityAndCalendars:
    def test_frame_is_a_defensive_copy(self) -> None:
        df = _close_only()
        frame = TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")
        df.iloc[0, df.columns.get_loc("close")] = -999.0
        assert frame.data["close"].iloc[0] != -999.0

    def test_dataclass_field_reassignment_rejected(self) -> None:
        frame = TimeSeriesFrame(data=_close_only(), asset_id="SPX", source="stooq")
        with pytest.raises(AttributeError):
            frame.asset_id = "NDX"  # type: ignore[misc]

    def test_non_utc_tz_normalized_to_utc(self) -> None:
        df = _close_only(tz="America/New_York")
        frame = TimeSeriesFrame(data=df, asset_id="SPX", source="stooq")
        assert str(frame.index.tz) == "UTC"
        assert frame.index[0] == df.index[0].tz_convert("UTC")

    def test_never_reindexes_across_assets(self) -> None:
        # Two assets on genuinely different calendars (weekday-only vs. every day)
        # must be constructible independently, on their own index, with no
        # cross-asset alignment performed by this module.
        weekday_idx = pd.bdate_range("2024-01-04", periods=5, tz="UTC")  # Thu-Fri, Mon-Wed
        daily_idx = pd.date_range("2024-01-04", periods=5, freq="D", tz="UTC")  # every calendar day
        a = TimeSeriesFrame(
            data=pd.DataFrame({"close": [1.0] * 5}, index=weekday_idx), asset_id="A", source="byo"
        )
        b = TimeSeriesFrame(
            data=pd.DataFrame({"close": [1.0] * 5}, index=daily_idx), asset_id="B", source="byo"
        )
        assert not a.index.equals(b.index)
