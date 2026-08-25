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
    overnight_plus_range_variance,
    overnight_variance,
    parkinson,
    realized_variance_from_bars,
    rogers_satchell,
    squared_return,
)

_LN2 = math.log(2.0)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")


def _consistent_bars(
    rng: np.random.Generator, n: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Random OHLC bars satisfying low <= min(O, C) <= max(O, C) <= high."""
    o = 100.0 * np.exp(rng.normal(0, 0.01, n))
    c = o * np.exp(rng.normal(0, 0.01, n))
    hi = np.maximum(o, c) + rng.random(n)
    lo = np.minimum(o, c) - rng.random(n)
    idx = _idx(n)
    return (
        pd.Series(o, index=idx),
        pd.Series(hi, index=idx),
        pd.Series(lo, index=idx),
        pd.Series(c, index=idx),
    )


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


class TestOvernightPlusRangeVariance:
    """The M2 close-to-close target: squared overnight jump + Rogers-Satchell.

    ``(ln(O_t/C_{t-1}))^2 + ln(H/O)ln(H/C) + ln(L/O)ln(L/C)``. Formulas
    corroborated 2026-08-24 across CRAN TTR, arXiv:1803.07152 and
    portfoliooptimizer.io (see the function docstring); primary papers
    (Rogers & Satchell 1991; Yang & Zhang 2000) are paywalled.
    """

    def test_hand_computed_value(self) -> None:
        o = pd.Series([100.0, 103.0], index=_idx(2))
        h = pd.Series([105.0, 108.0], index=_idx(2))
        low = pd.Series([98.0, 101.0], index=_idx(2))
        c = pd.Series([102.0, 104.0], index=_idx(2))
        out = overnight_plus_range_variance(o, h, low, c)
        assert np.isnan(out.iloc[0])  # no C_{-1}
        overnight = math.log(103.0 / 102.0) ** 2  # O_1 vs C_0
        rs = math.log(108.0 / 103.0) * math.log(108.0 / 104.0) + math.log(
            101.0 / 103.0
        ) * math.log(101.0 / 104.0)
        assert out.iloc[1] == pytest.approx(overnight + rs, rel=1e-12)
        assert out.name == "overnight_plus_range"

    def test_first_observation_is_nan_and_index_is_preserved(self) -> None:
        n = 6
        o = pd.Series(100.0 + np.arange(n), index=_idx(n))
        h = pd.Series(101.0 + np.arange(n), index=_idx(n))
        low = pd.Series(99.0 + np.arange(n), index=_idx(n))
        c = pd.Series(100.5 + np.arange(n), index=_idx(n))
        out = overnight_plus_range_variance(o, h, low, c)
        assert out.index.equals(o.index)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1:].notna().all()

    def test_it_is_causal_no_forward_reach(self) -> None:
        """Row t must be computable from data at or before t. Truncating the
        series after t therefore cannot change any row <= t — which it would if
        the overnight term used ``C_{t+1}`` (a ``shift(-1)`` slip)."""
        n = 12
        rng = np.random.default_rng(0)
        o, h, low, c = _consistent_bars(rng, n)
        full = overnight_plus_range_variance(o, h, low, c)
        for t in (3, 6, 9):
            trunc = overnight_plus_range_variance(
                o.iloc[: t + 1], h.iloc[: t + 1], low.iloc[: t + 1], c.iloc[: t + 1]
            )
            pd.testing.assert_series_equal(full.iloc[: t + 1], trunc)

    def test_overnight_anchor_is_the_previous_close_only(self) -> None:
        """Row t's overnight term is ``ln(O_t / C_{t-1})``: moving ``C_{t-1}``
        moves it, moving a strictly-future close (kept within its own bar so
        the estimator still accepts it) does not."""
        n = 8
        rng = np.random.default_rng(1)
        o, h, low, c = _consistent_bars(rng, n)
        base = overnight_plus_range_variance(o, h, low, c)
        t = 4

        future = c.copy()  # a valid within-bar move of a strictly-future close
        future.iloc[t + 1] = float(low.iloc[t + 1])
        after_future = overnight_plus_range_variance(o, h, low, future)
        pd.testing.assert_series_equal(base.iloc[: t + 1], after_future.iloc[: t + 1])

        prev = c.copy()  # move C_{t-1} within its bar; only the overnight anchor of row t
        prev.iloc[t - 1] = float(low.iloc[t - 1])
        after_prev = overnight_plus_range_variance(o, h, low, prev)
        assert after_prev.iloc[t] != base.iloc[t]

    def test_rogers_satchell_term_is_drift_independent(self) -> None:
        """RS's whole reason for existing (Rogers & Satchell 1991): its
        expectation is the diffusion variance whatever the drift. Build
        continuous-open days (each opens at the prior close, so the overnight
        term is exactly zero and this isolates RS), then re-run with a large
        drift added to the *same* Brownian increments. RS barely moves;
        Parkinson, which is drift-blind, inflates by orders more.

        The comparison is deliberately relative — RS-with-drift against
        RS-without — so the fixed discretization bias at finite steps cancels
        and only the drift sensitivity is measured. Absolute accuracy against a
        known variance is covered by tests/test_target_estimators.py, where the
        generator supplies the truth.
        """
        rng = np.random.default_rng(7)
        n_days, steps = 3000, 5000
        true_var = 4e-4
        vol = math.sqrt(true_var)
        base_increments = [
            vol / math.sqrt(steps) * rng.standard_normal(steps) for _ in range(n_days)
        ]

        def build(drift: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            price = 100.0
            o, h, low, c = (np.empty(n_days) for _ in range(4))
            for d in range(n_days):
                path = np.concatenate([[0.0], np.cumsum(base_increments[d] + drift / steps)])
                o[d] = price
                prices = price * np.exp(path)
                h[d], low[d], c[d] = prices.max(), prices.min(), prices[-1]
                price = c[d]  # next day opens here -> overnight term is exactly 0
            idx = _idx(n_days)
            o_s, h_s, l_s, c_s = (pd.Series(a, index=idx) for a in (o, h, low, c))
            return o_s, h_s, l_s, c_s

        o0, h0, l0, c0 = build(drift=0.0)
        od, hd, ld, cd = build(drift=2.0 * vol)  # a huge daily drift: 2x the daily vol

        rs_sensitivity = abs(
            float(overnight_plus_range_variance(od, hd, ld, cd).iloc[1:].mean())
            / float(overnight_plus_range_variance(o0, h0, l0, c0).iloc[1:].mean())
            - 1.0
        )
        park_sensitivity = abs(
            float(parkinson(hd, ld).iloc[1:].mean())
            / float(parkinson(h0, l0).iloc[1:].mean())
            - 1.0
        )
        assert rs_sensitivity < 0.05  # RS is nearly drift-invariant
        assert park_sensitivity > 1.0  # Parkinson more than doubles under the same drift
        assert park_sensitivity > 20.0 * rs_sensitivity  # and is orders more drift-sensitive

    def test_inconsistent_bars_are_rejected(self) -> None:
        good = dict(
            open_=pd.Series([100.0, 100.0]),
            high=pd.Series([101.0, 101.0]),
            low=pd.Series([99.0, 99.0]),
            close=pd.Series([100.5, 100.5]),
        )
        overnight_plus_range_variance(**good)  # baseline is fine
        with pytest.raises(ValueError, match="high must be >= low"):
            overnight_plus_range_variance(
                open_=pd.Series([100.0]), high=pd.Series([98.0]),
                low=pd.Series([99.0]), close=pd.Series([100.0]),
            )
        with pytest.raises(ValueError, match="high must be >= open"):
            overnight_plus_range_variance(
                open_=pd.Series([110.0]), high=pd.Series([105.0]),
                low=pd.Series([99.0]), close=pd.Series([100.0]),
            )
        with pytest.raises(ValueError, match="low must be <= open"):
            overnight_plus_range_variance(
                open_=pd.Series([100.0]), high=pd.Series([105.0]),
                low=pd.Series([101.0]), close=pd.Series([104.0]),
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


class TestOvernightIntradayDecomposition:
    """The D-016 target is exactly its two published pieces (D-016, panel report §5)."""

    def _bars(
        self, n: int = 300, steps_per_day: int = 120
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bars from a simulated driftless intraday path of known variance 0.008**2."""
        rng = np.random.default_rng(4242)
        index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        log_open = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
        scale = 0.008 / np.sqrt(steps_per_day)
        steps = np.cumsum(rng.normal(0.0, scale, (n, steps_per_day)), axis=1)
        open_ = pd.Series(np.exp(log_open), index=index)
        close = pd.Series(np.exp(log_open + steps[:, -1]), index=index)
        high = pd.Series(np.exp(log_open + np.maximum(steps.max(axis=1), 0.0)), index=index)
        low = pd.Series(np.exp(log_open + np.minimum(steps.min(axis=1), 0.0)), index=index)
        return open_, high, low, close

    def test_target_is_the_sum_of_its_parts(self) -> None:
        o, h, low, c = self._bars()
        total = overnight_variance(o, c) + rogers_satchell(o, h, low, c)
        pd.testing.assert_series_equal(
            total, overnight_plus_range_variance(o, h, low, c), check_names=False
        )

    def test_overnight_term_reads_only_the_previous_close(self) -> None:
        # Backward-looking by construction: changing a *later* close must not
        # move an earlier day's overnight variance.
        o, _h, _low, c = self._bars(50)
        before = overnight_variance(o, c)
        tampered = c.copy()
        tampered.iloc[30:] *= 1.5
        after = overnight_variance(o, tampered)
        pd.testing.assert_series_equal(before.iloc[:30], after.iloc[:30])

    def test_first_overnight_observation_is_nan(self) -> None:
        o, _, _, c = self._bars(10)
        assert np.isnan(overnight_variance(o, c).iloc[0])

    def test_overnight_variance_is_non_negative(self) -> None:
        o, _, _, c = self._bars()
        assert (overnight_variance(o, c).dropna() >= 0).all()

    def test_rogers_satchell_is_non_negative_on_consistent_bars(self) -> None:
        o, h, low, c = self._bars()
        assert (rogers_satchell(o, h, low, c) >= 0).all()

    def test_rogers_satchell_is_zero_on_a_monotone_bar(self) -> None:
        # A day that opens at its high and closes at its low has RS == 0
        # exactly. Combined with a stale open this is how a real series gets a
        # zero variance target — see docs/PANEL_REPORT.md §4.
        index = pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")])
        o = pd.Series([10.0], index=index)
        h = pd.Series([10.0], index=index)
        low = pd.Series([9.0], index=index)
        c = pd.Series([9.0], index=index)
        assert rogers_satchell(o, h, low, c).iloc[0] == 0.0

    def test_rogers_satchell_recovers_the_intraday_variance(self) -> None:
        """RS is unbiased for the intraday variance of a driftless path.

        Asserted as a *limit*, because the only bias present is discretization:
        a bar's high/low are the extremes of the sampled path, and a discretely
        sampled maximum is below the continuous one. Refining the path from 120
        to 2000 steps per day moves E[RS] from ~84% to ~96% of the truth, so the
        test pins both the level at 2000 steps and the direction of the
        convergence — a genuinely biased estimator would fail the second part
        even if the tolerance on the first were loosened.
        """
        truth = 0.008**2
        coarse = float(rogers_satchell(*self._bars(4000, steps_per_day=120)).mean())
        fine = float(rogers_satchell(*self._bars(4000, steps_per_day=2000)).mean())
        assert fine == pytest.approx(truth, rel=0.06)
        assert coarse < fine < truth
