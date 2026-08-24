"""Per-origin fault isolation (M1 report §4.5): one bad origin costs one row.

The results contract says nothing is ever dropped — an unscorable origin
yields a row with NaN scores and a ``missing_reason``. Until this was fixed,
an *exception* from a model was the one thing that contract did not cover:
it propagated out of ``_run_block`` and took the whole cell down. HAR raises
on a non-positive realized variance, so a single limit-locked day (high ==
low, Parkinson proxy exactly 0) on a real index would have crashed that
asset's entire backtest instead of costing it one row.

Every train/test index here comes from ``RollingOriginSplitter``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from volbench.benchmarks.make_toy_asset import DEFAULT_PATH
from volbench.benchmarks.toy import ASSET_ID, WINDOW, build_summary
from volbench.data import load_ohlc_csv, log_returns, parkinson
from volbench.dist import Distribution, Normal
from volbench.evaluate import run_backtest
from volbench.models import HAR
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

SIGMA = 0.012

#: This file's HAR scenario deliberately uses the intraday Parkinson series —
#: it needs a proxy that a limit-locked day sends to exactly zero.
PROXY_NAME = "parkinson"


# --------------------------------------------------------------------------
# test doubles — each fails in exactly one, chosen way
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    sigma: float

    def predict(self, h: int) -> Distribution:
        return Normal(0.0, max(self.sigma, 1e-12))


class FitSpy:
    """Refuses a window containing NaN — HAR's shape of strictness — and
    counts every attempt, so the refit cadence can be checked under failure."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "fit_spy"

    def spec(self) -> dict[str, Any]:
        return {"kind": "fit_spy"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _Fit:
        self.attempts += 1
        if np.isnan(train).any():
            raise ValueError("planted: window contains NaN")
        return _Fit(float(np.std(train)))


@dataclass(frozen=True)
class _ExplodesAt:
    sigma: float
    bad_horizon: int

    def predict(self, h: int) -> Distribution:
        if h == self.bad_horizon:
            raise RuntimeError(f"planted: no forecast at h={h}")
        return Normal(0.0, max(self.sigma, 1e-12))


@dataclass(frozen=True)
class ExplodesAtHorizon:
    bad_horizon: int = 2

    @property
    def name(self) -> str:
        return "explodes_at_horizon"

    def spec(self) -> dict[str, Any]:
        return {"kind": "explodes", "bad_horizon": self.bad_horizon}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _ExplodesAt:
        return _ExplodesAt(float(np.std(train)), self.bad_horizon)


class UpdateSpy:
    """Re-conditions between refits; the ``fail_on`` -th update raises."""

    def __init__(self, fail_on: int) -> None:
        self.fail_on = fail_on
        self.fit_calls = 0
        self.update_calls = 0

    @property
    def name(self) -> str:
        return "update_spy"

    def spec(self) -> dict[str, Any]:
        return {"kind": "update_spy"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _UpdateFit:
        self.fit_calls += 1
        return _UpdateFit(self, float(np.std(train)), abs(float(train[-1])))


@dataclass(frozen=True, eq=False)
class _UpdateFit:
    owner: UpdateSpy
    scale: float
    last_abs: float

    def predict(self, h: int) -> Distribution:
        return Normal(0.0, max(0.5 * self.scale + 0.5 * self.last_abs, 1e-12))

    def update(self, train: NDArray[np.float64]) -> _UpdateFit:
        self.owner.update_calls += 1
        if self.owner.update_calls == self.owner.fail_on:
            raise RuntimeError("planted: update refused")
        return _UpdateFit(self.owner, self.scale, abs(float(train[-1])))


@dataclass(frozen=True, eq=False)
class _Cursed(Distribution):
    """A forecast whose scoring blows up, not its construction."""

    def quantile(self, tau: float) -> float:
        return 0.0

    def crps(self, y: float) -> float:
        raise ArithmeticError("planted: crps exploded")


@dataclass(frozen=True)
class _CursedFit:
    def predict(self, h: int) -> Distribution:
        return _Cursed()


@dataclass(frozen=True)
class CursedModel:
    @property
    def name(self) -> str:
        return "cursed"

    def spec(self) -> dict[str, Any]:
        return {"kind": "cursed"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _CursedFit:
        return _CursedFit()


class Abort(BaseException):
    """Not an Exception: stands in for KeyboardInterrupt / SystemExit."""


@dataclass(frozen=True)
class AbortingModel:
    @property
    def name(self) -> str:
        return "aborting"

    def spec(self) -> dict[str, Any]:
        return {"kind": "aborting"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _Fit:
        raise Abort()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def panel(n: int = 40, seed: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, SIGMA, size=n)
    return returns, returns**2


def splitter(refit_every: int = 1, horizon: int = 1) -> RollingOriginSplitter:
    return RollingOriginSplitter(window=5, horizon=horizon, step=1, refit_every=refit_every)


def backtest(model: Any, returns: Any, proxy: Any, split: RollingOriginSplitter, **kw: Any) -> Any:
    return run_backtest(
        lambda: model, returns, proxy, split, seed=1, asset="SIM", proxy_name="squared_return", **kw
    )


def flagged(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["missing_reason"] != ""]


# --------------------------------------------------------------------------
# a failed fit
# --------------------------------------------------------------------------


class TestFailedFit:
    def test_costs_one_row_when_every_origin_refits(self) -> None:
        returns, proxy = panel()
        fit_series = returns.copy()
        fit_series[0] = np.nan  # in the first origin's window and no other
        split = splitter(refit_every=1)
        spy = FitSpy()

        frame = backtest(spy, returns, proxy, split, fit_series=fit_series)

        assert len(frame) == split.n_splits(returns.size)  # nothing dropped
        bad = flagged(frame)
        assert len(bad) == 1
        row = bad.iloc[0]
        assert row["origin_index"] == 4
        assert row["missing_reason"] == "fit_error@4: ValueError: planted: window contains NaN"
        assert row["fit_origin"] == -1 and row["conditioned_through"] == -1
        for column in ("forecast_mean", "forecast_var", "crps", "log_score", "qlike", "var_0p01"):
            assert math.isnan(row[column]), column
        # The data columns are data, not forecasts: they are still recorded.
        assert row["realized_return"] == returns[row["target_index"]]
        assert row["proxy_var"] == proxy[row["target_index"]]
        # Everything else is fully scored.
        good = frame[frame["missing_reason"] == ""]
        assert good["crps"].notna().all() and good["qlike"].notna().all()
        assert spy.attempts == split.n_splits(returns.size)

    def test_fails_its_whole_block_and_never_refits_off_schedule(self) -> None:
        """The refit cadence is part of the config hash; a failure must not
        quietly change it. Origins 4, 7, 10, 13, 16 are the scheduled fits;
        the NaN at position 5 sits in origin 7's window and no other
        scheduled one, so exactly block [7, 8, 9] has no model."""
        returns, proxy = panel(n=20)
        fit_series = returns.copy()
        fit_series[5] = np.nan
        split = splitter(refit_every=3)
        spy = FitSpy()

        frame = backtest(spy, returns, proxy, split, fit_series=fit_series)

        origins = list(split.split(returns.size))
        assert spy.attempts == sum(1 for o in origins if o.refit) == 5
        bad = flagged(frame)
        assert bad["origin_index"].tolist() == [7, 8, 9]
        assert set(bad["missing_reason"]) == {
            "fit_error@7: ValueError: planted: window contains NaN"
        }
        assert (bad["fit_origin"] == -1).all() and (bad["conditioned_through"] == -1).all()
        # The next block fits on schedule, at 10, and is unaffected.
        after = frame[frame["origin_index"] >= 10]
        assert (after["missing_reason"] == "").all()
        assert after["fit_origin"].tolist() == [10, 10, 10, 13, 13, 13, 16, 16, 16]
        # And the block before it was never touched.
        before = frame[frame["origin_index"] < 7]
        assert (before["missing_reason"] == "").all()
        assert (before["fit_origin"] == 4).all()


# --------------------------------------------------------------------------
# other stages
# --------------------------------------------------------------------------


class TestOtherStages:
    def test_a_failed_predict_costs_only_that_horizon(self) -> None:
        returns, proxy = panel()
        split = splitter(horizon=3)
        frame = backtest(ExplodesAtHorizon(bad_horizon=2), returns, proxy, split)

        assert len(frame) == split.n_splits(returns.size) * 3
        bad = flagged(frame)
        assert set(bad["horizon"]) == {2}
        assert len(bad) == split.n_splits(returns.size)
        assert set(bad["missing_reason"]) == {
            "predict_error: RuntimeError: planted: no forecast at h=2"
        }
        assert frame.loc[frame["horizon"] != 2, "crps"].notna().all()
        # The model itself was fine, and its provenance says so.
        assert (bad["fit_origin"] == bad["origin_index"]).all()

    def test_a_failed_update_costs_one_origin_and_later_ones_recover(self) -> None:
        returns, proxy = panel(n=20)
        split = splitter(refit_every=4)
        model = UpdateSpy(fail_on=2)
        frame = backtest(model, returns, proxy, split)

        origins = list(split.split(returns.size))
        assert model.fit_calls == sum(1 for o in origins if o.refit)  # cadence untouched
        bad = flagged(frame)
        assert len(bad) == 1
        row = bad.iloc[0]
        assert row["origin_index"] == 6  # second update: refit at 4, updates at 5, 6
        assert row["missing_reason"] == "update_error@6: RuntimeError: planted: update refused"
        assert row["fit_origin"] == 4
        assert row["conditioned_through"] == 5  # the last conditioning that succeeded
        # The very next origin re-conditions on its own window and is scored.
        nxt = frame[frame["origin_index"] == 7].iloc[0]
        assert nxt["missing_reason"] == "" and nxt["conditioned_through"] == 7

    def test_a_failed_score_is_recorded_as_such(self) -> None:
        returns, proxy = panel()
        frame = backtest(CursedModel(), returns, proxy, splitter())
        reason = "score_error: ArithmeticError: planted: crps exploded"
        assert (frame["missing_reason"] == reason).all()
        assert frame["crps"].isna().all()
        assert len(frame) == splitter().n_splits(returns.size)

    def test_base_exceptions_are_not_swallowed(self) -> None:
        returns, proxy = panel()
        with pytest.raises(Abort):
            backtest(AbortingModel(), returns, proxy, splitter())

    def test_each_failure_is_logged_once_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        returns, proxy = panel()
        fit_series = returns.copy()
        fit_series[0] = np.nan
        with caplog.at_level(logging.WARNING, logger="volbench.evaluate"):
            backtest(FitSpy(), returns, proxy, splitter(), fit_series=fit_series)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "fit_error@4: ValueError: planted: window contains NaN" in warnings[0].getMessage()


# --------------------------------------------------------------------------
# the frame is still a results frame
# --------------------------------------------------------------------------


class TestFrameContract:
    def test_failure_rows_keep_the_pinned_dtypes_and_round_trip_the_store(
        self, tmp_path: Path
    ) -> None:
        returns, proxy = panel()
        fit_series = returns.copy()
        fit_series[0] = np.nan
        store = ResultsStore(tmp_path)
        frame = backtest(FitSpy(), returns, proxy, splitter(), fit_series=fit_series, store=store)

        assert str(frame["fit_origin"].dtype) == "int64"
        assert str(frame["conditioned_through"].dtype) == "int64"
        assert str(frame["crps"].dtype) == "float64"
        stored = store.read(frame.attrs["config_hash"])
        pd.testing.assert_frame_equal(stored, frame)

    def test_a_clean_run_is_unchanged(self) -> None:
        """No failure means no failure column, sentinel or reason anywhere."""
        returns, proxy = panel()
        frame = backtest(FitSpy(), returns, proxy, splitter(refit_every=3))
        assert (frame["missing_reason"] == "").all()
        assert (frame["fit_origin"] >= 0).all()
        assert (frame["conditioned_through"] == frame["fit_origin"]).all()


# --------------------------------------------------------------------------
# the case from the report: one limit-locked day, HAR, the toy series
# --------------------------------------------------------------------------


def _toy_series_with_a_limit_locked_day(bar: int) -> tuple[pd.Series, pd.Series]:
    """The committed toy fixture with bar ``bar`` locked limit-up at +5%.

    A limit-locked session opens at the limit and never leaves it, so open,
    high, low and close are all that one price: the Parkinson proxy is
    exactly 0. HAR refuses a non-positive realized variance *before* it
    takes any logs — the effect is the same, its ``fit`` raises.
    """
    frame = load_ohlc_csv(DEFAULT_PATH, asset_id=ASSET_ID, source="synthetic")
    prices = frame.data.copy()
    limit = float(prices["close"].iloc[bar - 1]) * 1.05
    prices.loc[prices.index[bar], ["open", "high", "low", "close"]] = limit
    returns = log_returns(prices["close"]).iloc[1:]
    proxy = parkinson(prices["high"], prices["low"]).iloc[1:]
    return returns, proxy


class TestLimitLockedDayWithHAR:
    def test_one_bad_day_costs_har_exactly_one_row(self) -> None:
        # Bar 1 becomes position 0 after the leading trim: inside the first
        # origin's window and no other, and never a forecast target.
        returns, proxy = _toy_series_with_a_limit_locked_day(bar=1)
        assert float(proxy.iloc[0]) == 0.0
        split = RollingOriginSplitter(window=WINDOW, horizon=1, step=1, refit_every=1)

        frame = run_backtest(
            HAR,
            returns,
            proxy,
            split,
            0,
            asset=ASSET_ID,
            proxy_name=PROXY_NAME,
            fit_series=proxy,
        )

        assert len(frame) == split.n_splits(returns.size) == 200
        bad = flagged(frame)
        assert len(bad) == 1
        row = bad.iloc[0]
        assert row["origin_index"] == WINDOW - 1
        assert row["missing_reason"].startswith(f"fit_error@{WINDOW - 1}: ValueError: ")
        assert "strictly positive" in row["missing_reason"]  # HAR's own words, preserved
        assert row["fit_origin"] == -1
        assert math.isnan(row["forecast_var"]) and math.isnan(row["crps"])
        good = frame[frame["missing_reason"] == ""]
        assert len(good) == 199
        assert good["crps"].notna().all() and good["qlike"].notna().all()
        assert (good["fit_origin"] == good["origin_index"]).all()

    def test_the_summary_counts_only_what_was_scored(self) -> None:
        returns, proxy = _toy_series_with_a_limit_locked_day(bar=1)
        split = RollingOriginSplitter(window=WINDOW, horizon=1, step=1, refit_every=1)
        frame = run_backtest(
            HAR, returns, proxy, split, 0, asset=ASSET_ID, proxy_name=PROXY_NAME, fit_series=proxy
        )
        summary = build_summary(frame.assign(label="har"))
        assert summary["n"].item() == 200
        assert summary["n_scored"].item() == 199
