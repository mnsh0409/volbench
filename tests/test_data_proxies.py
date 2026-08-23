"""Daily variance proxies: hand-computed values, non-negativity, and the RV min_bars guard.

Leakage note: every proxy here is contemporaneous (day t's proxy uses only day
t's own OHLC/bars), so there is nothing to check against RollingOriginSplitter
directly — the property under test is that the *formulas* are correct and
that realized_variance_from_bars never lets a return span two days.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from volbench.data import (
    garman_klass,
    log_returns,
    parkinson,
    realized_variance_from_bars,
    squared_return,
)

_LN2 = math.log(2.0)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")


class TestLogReturns:
    """The A-to-C seam added at M1 integration: models and scoring speak in
    returns, and everything downstream aligns positionally, so the shape and
    the leading NaN matter as much as the arithmetic."""

    def test_hand_computed_values(self) -> None:
        close = pd.Series([100.0, 105.0, 103.0], index=_idx(3))
        out = log_returns(close)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == pytest.approx(math.log(1.05), rel=1e-12)
        assert out.iloc[2] == pytest.approx(math.log(103.0 / 105.0), rel=1e-12)

    def test_keeps_sign_unlike_squared_return(self) -> None:
        close = pd.Series([100.0, 95.0, 99.0], index=_idx(3))
        out = log_returns(close)
        assert out.iloc[1] < 0.0
        assert out.iloc[2] > 0.0

    def test_stays_index_aligned_with_its_input(self) -> None:
        # run_backtest matches returns to proxies positionally; a helper that
        # dropped the leading gap would offset every forecast by one day.
        close = pd.Series([100.0, 105.0, 103.0, 107.0], index=_idx(4))
        out = log_returns(close)
        assert len(out) == len(close)
        assert out.index.equals(close.index)

    def test_squared_return_is_exactly_its_square(self) -> None:
        rng = np.random.default_rng(3)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=_idx(200))
        expected = log_returns(close) ** 2
        pd.testing.assert_series_equal(
            squared_return(close), expected.rename("squared_return"), check_names=True
        )


class TestSquaredReturn:
    def test_hand_computed_values(self) -> None:
        close = pd.Series([100.0, 105.0, 103.0], index=_idx(3))
        out = squared_return(close)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == pytest.approx(0.0023804801196801307, rel=1e-12)
        assert out.iloc[2] == pytest.approx(0.00036984528160140636, rel=1e-12)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(1)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=_idx(500))
        out = squared_return(close)
        assert (out.dropna() >= 0.0).all()


class TestParkinson:
    def test_hand_computed_value(self) -> None:
        high = pd.Series([110.0])
        low = pd.Series([90.0])
        out = parkinson(high, low)
        assert out.iloc[0] == pytest.approx(0.014523873553353111, rel=1e-12)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        n = 500
        low = pd.Series(90.0 + rng.random(n) * 5.0)
        high = low + rng.random(n) * 10.0
        out = parkinson(high, low)
        assert (out >= 0.0).all()

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(ValueError, match="high must be >= low"):
            parkinson(pd.Series([90.0]), pd.Series([110.0]))


class TestGarmanKlass:
    def test_hand_computed_value(self) -> None:
        out = garman_klass(
            open_=pd.Series([95.0]), high=pd.Series([110.0]), low=pd.Series([90.0]),
            close=pd.Series([108.0]),
        )
        assert out.iloc[0] == pytest.approx(0.013780140623061429, rel=1e-12)

    def test_non_negative_for_valid_ohlc_bars(self) -> None:
        # For genuine OHLC bars (H = max(O,H,L,C), L = min(O,H,L,C)) GK is provably
        # non-negative: L <= O,C <= H forces (ln H/L)^2 >= (ln C/O)^2, and the
        # 0.5 vs. (2 ln2 - 1 ~= 0.386) coefficients keep the difference >= 0.
        rng = np.random.default_rng(3)
        n = 2000
        o = 100.0 * np.exp(rng.normal(0, 0.01, n))
        c = 100.0 * np.exp(rng.normal(0, 0.01, n))
        wick = rng.random(n) * 2.0
        h = np.maximum(o, c) + wick
        low_wick = rng.random(n) * 2.0
        low = np.minimum(o, c) - low_wick
        out = garman_klass(pd.Series(o), pd.Series(h), pd.Series(low), pd.Series(c))
        assert (out >= -1e-15).all()

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(ValueError, match="high must be >= low"):
            garman_klass(
                pd.Series([100.0]), pd.Series([90.0]), pd.Series([110.0]), pd.Series([100.0])
            )


class TestParkinsonMatchesSquaredReturnInExpectation:
    def test_matches_on_simulated_gbm(self) -> None:
        rng = np.random.default_rng(0)
        n_days, steps_per_day = 3000, 1000
        daily_var = 0.0004
        step_var = daily_var / steps_per_day
        increments = rng.normal(
            loc=-0.5 * step_var, scale=math.sqrt(step_var), size=n_days * steps_per_day
        )
        log_price = np.cumsum(increments) + math.log(100.0)
        price = np.exp(log_price).reshape(n_days, steps_per_day)

        high = pd.Series(price.max(axis=1), index=_idx(n_days))
        low = pd.Series(price.min(axis=1), index=_idx(n_days))
        close = pd.Series(price[:, -1], index=_idx(n_days))

        pk = parkinson(high, low)
        sq = squared_return(close)

        pk_mean = float(pk.mean())
        sq_mean = float(sq.dropna().mean())

        assert pk_mean == pytest.approx(daily_var, rel=0.2)
        assert sq_mean == pytest.approx(daily_var, rel=0.2)
        assert pk_mean == pytest.approx(sq_mean, rel=0.25)


class TestRealizedVarianceFromBars:
    def test_hand_computed_single_day(self) -> None:
        prices = [100.0, 100.5, 100.2, 100.8, 100.6]
        idx = pd.date_range("2024-01-02", periods=5, freq="1min", tz="UTC")
        rv = realized_variance_from_bars(pd.Series(prices, index=idx), min_bars=2)
        assert rv.size == 1
        assert rv.iloc[0] == pytest.approx(7.340039184987576e-05, rel=1e-10)

    def test_never_spans_two_days(self) -> None:
        day1 = pd.date_range("2024-01-02 23:57", periods=3, freq="1min", tz="UTC")
        day2 = pd.date_range("2024-01-03 00:00", periods=3, freq="1min", tz="UTC")
        idx = day1.append(day2)
        prices = pd.Series([100.0, 200.0, 100.0, 50.0, 100.0, 50.0], index=idx)
        rv = realized_variance_from_bars(prices, min_bars=2)
        assert rv.size == 2
        # A day-boundary return (100.0 -> 50.0 crossing midnight) must be excluded
        # from both days' RV; each day only sums its own within-day returns.
        expected_day1 = math.log(200.0 / 100.0) ** 2 + math.log(100.0 / 200.0) ** 2
        expected_day2 = math.log(50.0 / 100.0) ** 2 + math.log(100.0 / 50.0) ** 2
        assert rv.iloc[0] == pytest.approx(expected_day1, rel=1e-12)
        assert rv.iloc[1] == pytest.approx(expected_day2, rel=1e-12)

    def test_min_bars_guard_returns_nan_for_thin_days(self) -> None:
        idx = pd.to_datetime(
            ["2024-01-02 00:00", "2024-01-03 00:00", "2024-01-03 00:01", "2024-01-03 00:02"],
            utc=True,
        )
        prices = pd.Series([100.0, 100.0, 100.5, 100.2], index=idx)
        rv = realized_variance_from_bars(prices, min_bars=3)
        assert np.isnan(rv.iloc[0])  # day 1 has a single bar: below min_bars
        assert not np.isnan(rv.iloc[1])  # day 2 has 3 bars: meets min_bars

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(4)
        idx = pd.date_range("2024-01-02", periods=2000, freq="1min", tz="UTC")
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 2000))), index=idx)
        rv = realized_variance_from_bars(prices, min_bars=2)
        assert (rv.dropna() >= 0.0).all()

    def test_min_bars_below_two_rejected(self) -> None:
        idx = pd.date_range("2024-01-02", periods=3, freq="1min", tz="UTC")
        with pytest.raises(ValueError, match="min_bars must be >= 2"):
            realized_variance_from_bars(pd.Series([1.0, 2.0, 3.0], index=idx), min_bars=1)

    def test_naive_index_rejected(self) -> None:
        idx = pd.date_range("2024-01-02", periods=3, freq="1min")
        with pytest.raises(ValueError, match="tz-aware"):
            realized_variance_from_bars(pd.Series([1.0, 2.0, 3.0], index=idx))

    def test_nonpositive_prices_rejected(self) -> None:
        idx = pd.date_range("2024-01-02", periods=3, freq="1min", tz="UTC")
        with pytest.raises(ValueError, match="strictly positive"):
            realized_variance_from_bars(pd.Series([1.0, 0.0, 3.0], index=idx))
