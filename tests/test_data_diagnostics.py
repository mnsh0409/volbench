"""Panel diagnostics: the numbers docs/PANEL_REPORT.md quotes must be the right ones.

Built on synthetic series with known answers, so a change in the measurement
shows up here rather than as a quietly different number in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volbench.data.diagnostics import GAP_ALERT_DAYS, diagnose, diagnose_panel, diagnostics_frame
from volbench.data.panel import BarQuality, PanelSeries, build_targets
from volbench.data.types import TimeSeriesFrame

#: Intraday steps simulated per day. The range estimators are defined for a
#: continuous path, so the high/low must come from an actual simulated path
#: rather than being drawn independently — bars whose extremes are unrelated to
#: their own open-to-close move would make Rogers-Satchell measure noise, and
#: the overnight-share test below would then pass or fail for the wrong reason.
_INTRADAY_STEPS = 240


def _series(
    index: pd.DatetimeIndex,
    *,
    overnight_scale: float = 0.004,
    intraday_scale: float = 0.008,
    seed: int = 7,
    asset_id: str = "TEST",
) -> PanelSeries:
    """A synthetic asset with a known overnight/intraday variance split.

    Each day is an overnight jump of standard deviation ``overnight_scale``
    followed by a driftless Brownian intraday path of total standard deviation
    ``intraday_scale``, so the true overnight share of close-to-close variance
    is ``overnight_scale**2 / (overnight_scale**2 + intraday_scale**2)``.
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    step = intraday_scale / np.sqrt(_INTRADAY_STEPS)

    jumps = rng.normal(0.0, overnight_scale, size=n)
    path = np.cumsum(rng.normal(0.0, step, size=(n, _INTRADAY_STEPS)), axis=1)

    log_open = np.empty(n)
    log_price = np.log(100.0)
    for i in range(n):
        log_open[i] = log_price + jumps[i]
        log_price = log_open[i] + path[i, -1]

    open_ = np.exp(log_open)
    close = np.exp(log_open + path[:, -1])
    high = np.exp(log_open + np.maximum(path.max(axis=1), 0.0))
    low = np.exp(log_open + np.minimum(path.min(axis=1), 0.0))

    data = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )
    targets, components = build_targets(data)
    return PanelSeries(
        asset_id=asset_id,
        source="test",
        role="index",
        description="synthetic",
        frame=TimeSeriesFrame(data=data, asset_id=asset_id, source="test"),
        targets=targets,
        components=components,
        quality=BarQuality(n_bars=n, repaired=1, inconsistent=2, zero_range=3, non_positive=0),
        primary_target="overnight_plus_range",
        archive_start=index[0],
        archive_end=index[-1],
    )


class TestSpanAndCounts:
    def test_reports_the_panel_span_and_size(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=300, tz="UTC")
        diag = diagnose(_series(index))
        assert diag.n_obs == 300
        assert diag.panel_start == index[0]
        assert diag.panel_end == index[-1]
        assert diag.obs_per_year == pytest.approx(261, abs=6)

    def test_bar_quality_counts_are_passed_through_untouched(self) -> None:
        diag = diagnose(_series(pd.bdate_range("2020-01-01", periods=60, tz="UTC")))
        assert (diag.repaired_bars, diag.inconsistent_bars, diag.zero_range_days) == (1, 2, 3)

    def test_nan_is_counted_per_target(self) -> None:
        diag = diagnose(_series(pd.bdate_range("2020-01-01", periods=60, tz="UTC")))
        # Only the first day, which has no previous close for its overnight term.
        assert diag.nan_by_target["overnight_plus_range"] == 1
        assert diag.nan_by_target["squared_return"] == 1
        assert diag.nan_by_target["parkinson"] == 0


class TestCalendarGaps:
    def test_business_day_index_has_a_weekend_max_gap(self) -> None:
        diag = diagnose(_series(pd.bdate_range("2020-01-06", periods=40, tz="UTC")))
        assert diag.max_gap_days == 3
        assert diag.n_gaps_over_alert == 0

    def test_a_long_closure_is_reported_with_its_bracketing_bars(self) -> None:
        days = list(pd.bdate_range("2020-01-06", periods=20, tz="UTC"))
        # Drop two weeks in the middle: a Lunar-New-Year-shaped closure.
        index = pd.DatetimeIndex(days[:10] + days[20:] + list(
            pd.bdate_range("2020-02-17", periods=10, tz="UTC")
        ))
        diag = diagnose(_series(index))
        assert diag.max_gap_days > GAP_ALERT_DAYS
        assert diag.n_gaps_over_alert >= 1
        before, after, length = diag.longest_gaps[0]
        assert before in index and after in index
        assert (after - before).days == length
        assert diag.max_gap_at == after

    def test_single_observation_has_no_gaps(self) -> None:
        diag = diagnose(_series(pd.DatetimeIndex([pd.Timestamp("2020-01-06", tz="UTC")])))
        assert (diag.max_gap_days, diag.n_gaps_over_alert, diag.longest_gaps) == (0, 0, ())
        assert diag.max_gap_at is None


class TestOvernightShare:
    def test_recovers_a_known_overnight_share(self) -> None:
        # Overnight sd 0.004, intraday sd 0.008 => share = 16/(16+64) = 0.2.
        index = pd.bdate_range("2000-01-03", periods=6000, tz="UTC")
        diag = diagnose(_series(index, overnight_scale=0.004, intraday_scale=0.008))
        assert diag.overnight_share == pytest.approx(0.2, abs=0.03)

    def test_a_market_with_no_overnight_session_reports_a_share_near_zero(self) -> None:
        index = pd.bdate_range("2010-01-04", periods=2000, tz="UTC")
        diag = diagnose(_series(index, overnight_scale=1e-9, intraday_scale=0.01))
        assert diag.overnight_share < 1e-6

    def test_opr_exceeds_parkinson_when_there_is_an_overnight_gap(self) -> None:
        index = pd.bdate_range("2010-01-04", periods=2000, tz="UTC")
        diag = diagnose(_series(index, overnight_scale=0.006, intraday_scale=0.008))
        assert diag.opr_over_parkinson_quantiles["q50"] > 1.0

    def test_quantiles_are_ordered(self) -> None:
        index = pd.bdate_range("2010-01-04", periods=1000, tz="UTC")
        diag = diagnose(_series(index))
        for quantiles in (diag.overnight_share_quantiles, diag.opr_over_parkinson_quantiles):
            values = [quantiles[k] for k in ("q10", "q25", "q50", "q75", "q90")]
            assert values == sorted(values)

    def test_annualized_vol_is_a_faithful_restatement_of_the_daily_mean(self) -> None:
        index = pd.bdate_range("2010-01-04", periods=1500, tz="UTC")
        diag = diagnose(_series(index))
        assert diag.annualized_vol_pct == pytest.approx(
            float(np.sqrt(diag.mean_primary * 252.0) * 100.0)
        )


class TestPanelLevel:
    def test_diagnose_panel_preserves_order(self) -> None:
        index = pd.bdate_range("2015-01-05", periods=100, tz="UTC")
        panel = {name: _series(index, asset_id=name, seed=i) for i, name in enumerate("CAB")}
        assert list(diagnose_panel(panel)) == ["C", "A", "B"]

    def test_frame_has_one_row_per_series_keyed_by_asset(self) -> None:
        index = pd.bdate_range("2015-01-05", periods=100, tz="UTC")
        panel = {name: _series(index, asset_id=name, seed=i) for i, name in enumerate("XY")}
        frame = diagnostics_frame(diagnose_panel(panel))
        assert list(frame.index) == ["X", "Y"]
        for column in ("n_obs", "max_gap_days", "overnight_share", "nan_parkinson"):
            assert column in frame.columns
