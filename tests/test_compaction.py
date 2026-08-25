"""D-018, the invalid-target policy: what an unusable variance day does.

The decision has two halves that pull in opposite directions, and most of this
file exists to pin the seam between them:

- **as a target**, an invalid day (primary target NaN or ``<= 0``) keeps its
  place. The row is produced, the scores are NaN, ``missing_reason`` says why.
  Nothing is ever dropped from the scored table.
- **as a fit input**, it is dropped, and the window reaches further back so the
  model still gets exactly ``window`` observations.

Three properties follow, and are asserted here rather than assumed:

1. the splitter still runs on the **full calendar** — compaction happens when a
   window is materialized, never by reshaping the series the splitter sees, so
   which days are scored is untouched;
2. an invalid day is still a valid **origin** — its history is intact, so the
   forecast issued at it is a normal forecast, and only the row whose *target*
   is that day is unscorable;
3. compaction reaches **backwards only**. Every day it drops from a window lies
   in the past of that window's origin, and the window's last observation never
   moves past the origin. That is the leakage claim, and
   :class:`TestCompactionCannotReachForward` is where it is tested.

The two end-to-end cases carry the panel's own defect counts (docs/PANEL_REPORT.md
§3, §4): HSI's 12 exactly-zero targets and TWSE's 80 NaN'd inconsistent bars,
reproduced here by their real mechanisms — a monotone bar whose open equals the
previous close, and a close printed outside its own session range — through the
panel's own :func:`~volbench.data.panel.repair_bars` and
:func:`~volbench.data.panel.build_targets`. The series are shorter than the
panel's so the suite stays quick; the defects are not.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from volbench.compaction import (
    DEFAULT_INVALID_TARGET_POLICY,
    FitSeries,
    InsufficientHistoryError,
    invalid_target_mask,
    valid_target_mask,
)
from volbench.data.panel import build_targets, repair_bars
from volbench.data.proxies import log_returns
from volbench.dist import Distribution, Normal
from volbench.evaluate import run_backtest
from volbench.models.har import HAR
from volbench.splitter import RollingOriginSplitter

# The panel's own counts (docs/PANEL_REPORT.md §3-§4), reproduced by mechanism.
HSI_ZERO_DAYS = 12
TWSE_INCONSISTENT_BARS = 80

N_DAYS = 2000
WINDOW = 500
REFIT_EVERY = 21
SEED = 20260825


# --------------------------------------------------------------------------
# fixtures: series that carry the panel's defects, built by their mechanism
# --------------------------------------------------------------------------


def _well_formed_bars(n: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Ordinary OHLC bars: every one satisfies ``low <= o, c <= high``."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2005-01-03", periods=n, tz="UTC", name="timestamp")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.011, size=n)))
    prev = np.concatenate(([100.0], close[:-1]))
    open_ = prev * np.exp(rng.normal(0.0, 0.004, size=n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.005, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.005, size=n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )


def _stale_open_monotone(bars: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Make each named day a zero-target day, HSI's way.

    Rogers-Satchell is ``ln(H/O)ln(H/C) + ln(L/O)ln(L/C)``, so a *monotone*
    bar — open at the high, close at the low — has a zero factor in each
    product and RS is exactly 0. The overnight term is exactly 0 when the open
    equals the previous close, which is what a stale or synthetic open in the
    source file looks like. Where both happen, ``overnight_plus_range`` is
    exactly 0.0 — not small, not negative: zero, where ``log RV`` diverges.
    """
    frame = bars.copy()
    values = frame.to_numpy(dtype=np.float64, copy=True)
    for position in positions:
        assert position >= 1, "a stale open needs a previous close"
        previous_close = float(values[position - 1, 3])
        low = previous_close * 0.992
        # open == high == previous close; low == close: a monotone down day
        # that opened exactly where the last one closed.
        values[position, 0] = previous_close
        values[position, 1] = previous_close
        values[position, 2] = low
        values[position, 3] = low
    return pd.DataFrame(values, index=frame.index, columns=list(frame.columns))


def _inconsistent(bars: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Print each named day's close above its own session high, TWSE's way.

    1% above, far beyond ``repair_bars``' 1e-5 rounding tolerance, so the bar
    is flagged rather than clamped and its three range-based targets become
    NaN. The close itself is left readable, so returns and ``squared_return``
    are unaffected — exactly as in the archives.
    """
    frame = bars.copy()
    values = frame.to_numpy(dtype=np.float64, copy=True)
    for position in positions:
        values[position, 3] = float(values[position, 1]) * 1.01
    return pd.DataFrame(values, index=frame.index, columns=list(frame.columns))


def _panel_targets(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``(returns, primary target)`` through the panel's own two functions.

    The first row is dropped from both: neither a return nor an overnight term
    exists for it. A leading-edge trim applied identically to every series
    moves no information backwards.
    """
    repaired, inconsistent, _ = repair_bars(bars)
    targets, _ = build_targets(repaired, inconsistent=inconsistent)
    returns = log_returns(repaired["close"]).iloc[1:]
    primary = targets["overnight_plus_range"].iloc[1:]
    assert returns.index.equals(primary.index)
    return returns, primary


@pytest.fixture(scope="module")
def hsi_like() -> tuple[pd.Series, pd.Series]:
    """A series with HSI's 12 exactly-zero primary targets."""
    positions = list(range(WINDOW + 40, N_DAYS - 40, 97))[:HSI_ZERO_DAYS]
    assert len(positions) == HSI_ZERO_DAYS
    returns, primary = _panel_targets(_stale_open_monotone(_well_formed_bars(), positions))
    assert int((primary.to_numpy() == 0.0).sum()) == HSI_ZERO_DAYS
    assert not primary.isna().any()
    return returns, primary


@pytest.fixture(scope="module")
def twse_like() -> tuple[pd.Series, pd.Series]:
    """A series with TWSE's 80 NaN'd inconsistent bars."""
    positions = list(range(WINDOW + 30, N_DAYS - 30, 17))[:TWSE_INCONSISTENT_BARS]
    assert len(positions) == TWSE_INCONSISTENT_BARS
    returns, primary = _panel_targets(_inconsistent(_well_formed_bars(), positions))
    assert int(primary.isna().sum()) == TWSE_INCONSISTENT_BARS
    return returns, primary


# --------------------------------------------------------------------------
# a model that validates the window it is handed
# --------------------------------------------------------------------------


class _FittedWindowValidator:
    """Records the window it was given; forecasts its mean as a variance."""

    def __init__(self, window: NDArray[np.float64], log: list[NDArray[np.float64]]) -> None:
        self.window = window
        self.log = log

    @property
    def name(self) -> str:
        return "window_validator"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name, "context": int(self.window.size)}

    def predict(self, h: int) -> Distribution:
        return Normal(mu=0.0, sigma=math.sqrt(float(np.mean(self.window))))

    def update(self, train: NDArray[np.float64]) -> _FittedWindowValidator:
        return WindowValidator().fit(train, log=self.log)


class WindowValidator:
    """A TSFM-style adapter: it validates its context and refuses a bad one.

    Every foundation-model adapter in volbench checks the window it is handed
    (a NaN or a non-positive variance would propagate silently through a
    scaler or a log), so this stands in for all of them without needing
    weights. It also *records* every window, which is what lets the tests below
    assert what a model actually saw rather than what it was meant to see.
    """

    @property
    def name(self) -> str:
        return "window_validator"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name}

    def fit(
        self, train: NDArray[np.float64], log: list[NDArray[np.float64]] | None = None, **ctx: Any
    ) -> _FittedWindowValidator:
        window = np.asarray(train, dtype=np.float64)
        if not np.isfinite(window).all() or (window <= 0.0).any():
            raise ValueError("context must be finite and strictly positive")
        if log is not None:
            log.append(window.copy())
        return _FittedWindowValidator(window, log if log is not None else [])


def _factory(log: list[NDArray[np.float64]]) -> Any:
    class _Recording(WindowValidator):
        def fit(  # type: ignore[override]
            self, train: NDArray[np.float64], **ctx: Any
        ) -> _FittedWindowValidator:
            return WindowValidator.fit(self, train, log=log)

    return _Recording


# --------------------------------------------------------------------------
# the definition
# --------------------------------------------------------------------------


class TestWhatCountsAsInvalid:
    def test_nan_and_non_positive_are_invalid_everything_else_is_not(self) -> None:
        values = np.array([1e-8, 1.0, 0.0, -1.0, np.nan, np.inf, -np.inf, 4e-4])
        expected = np.array([True, True, False, False, False, False, False, True])
        np.testing.assert_array_equal(valid_target_mask(values), expected)
        np.testing.assert_array_equal(invalid_target_mask(values), ~expected)

    def test_zero_is_invalid_because_that_is_where_the_estimators_break(self) -> None:
        """Not a threshold anyone chose: ``log 0`` is ``-inf`` and QLIKE is
        ``v/y - log(v/y) - 1``, undefined at ``y = 0``. A tiny positive
        variance is unusual but perfectly scorable, so it stays."""
        assert bool(invalid_target_mask(np.array([0.0]))[0])
        assert not bool(invalid_target_mask(np.array([1e-300]))[0])

    def test_a_series_with_no_defects_has_nothing_to_drop(self) -> None:
        fit = FitSeries.compact(np.linspace(1.0, 2.0, 100))
        assert fit.n_invalid == 0 and fit.n_valid == fit.size == 100


# --------------------------------------------------------------------------
# window semantics
# --------------------------------------------------------------------------


class TestWindowIsTheLastNValidObservations:
    def test_the_window_reaches_back_past_the_days_it_drops(self) -> None:
        values = np.arange(1.0, 21.0)
        values[15] = 0.0  # invalid, inside the window below
        values[12] = np.nan  # invalid, inside it too
        fit = FitSeries.compact(values)
        train = np.arange(15, 20, dtype=np.int64)  # 5 positions, origin 19

        kept = fit.window_positions(train)
        assert kept.size == train.size == 5, "a compacted window is never short"
        np.testing.assert_array_equal(kept, [14, 17, 18, 19, 16][:0] or [14, 16, 17, 18, 19])
        np.testing.assert_array_equal(fit.window(train), values[[14, 16, 17, 18, 19]])
        # The span stretched from 5 calendar days to 6; the end did not move.
        assert kept[-1] == train[-1] == 19
        assert kept[0] == 14 < train[0] == 15

    def test_only_the_length_and_the_origin_of_the_splitters_window_are_read(self) -> None:
        """The pair the splitter guarantees is "N observations, ending at t";
        compaction consumes exactly that pair and invents no index of its own."""
        values = np.arange(1.0, 21.0)
        values[15] = 0.0
        fit = FitSeries.compact(values)
        assert fit.window(np.arange(15, 20, dtype=np.int64)).size == 5
        assert fit.window(np.arange(10, 20, dtype=np.int64)).size == 10

    def test_policy_none_is_the_splitters_own_positions_untouched(self) -> None:
        values = np.arange(1.0, 21.0)
        values[15] = 0.0
        fit = FitSeries.raw(values)
        train = np.arange(15, 20, dtype=np.int64)
        np.testing.assert_array_equal(fit.window_positions(train), train)
        np.testing.assert_array_equal(fit.window(train), values[train])
        assert fit.dropped_positions(train).size == 0

    def test_compact_equals_none_when_there_is_nothing_to_drop(self) -> None:
        values = np.linspace(1.0, 2.0, 50)
        train = np.arange(10, 30, dtype=np.int64)
        np.testing.assert_array_equal(
            FitSeries.compact(values).window(train), FitSeries.raw(values).window(train)
        )

    def test_a_window_is_a_copy_so_a_model_cannot_write_through_it(self) -> None:
        values = np.linspace(1.0, 2.0, 50)
        fit = FitSeries.compact(values)
        window = fit.window(np.arange(10, 30, dtype=np.int64))
        window[:] = -1.0
        assert float(fit.values[10]) == pytest.approx(values[10])

    def test_default_policy_is_compaction(self) -> None:
        assert DEFAULT_INVALID_TARGET_POLICY == "compact"
        assert FitSeries.of(np.ones(5)).policy == "compact"
        assert FitSeries.raw(np.ones(5)).policy == "none"

    def test_the_series_cannot_move_under_the_positions_derived_from_it(self) -> None:
        """``valid_positions`` is derived once, so ``values`` must be frozen.

        ``np.asarray`` on a float64 pandas Series returns a *view* of that
        Series' buffer — the trap that made ``panel.repair_bars`` describe
        repaired bars rather than the ones the file contained. Here it would
        be worse: the cached positions would say a day is valid that no longer
        is, and a model would be handed it.
        """
        source = pd.Series([1.0, 2.0, 3.0, 4.0])
        fit = FitSeries.compact(source)
        assert fit.n_invalid == 0
        source.iloc[1] = 0.0  # the caller mutates their own series afterwards
        assert fit.n_invalid == 0, "the wrapper must hold its own copy"
        assert float(fit.values[1]) == 2.0
        with pytest.raises(ValueError):
            fit.values[0] = -1.0  # ... and nothing can write through it either
        # A window is still writable, because model backends may want that.
        window = fit.window(np.arange(0, 3, dtype=np.int64))
        window[0] = 99.0
        assert float(fit.values[0]) == 1.0

    def test_it_survives_a_process_boundary(self) -> None:
        """``_BlockTask`` holds one of these and the Phase-3 executors pickle
        it across processes and Slurm array tasks (D-011). A frozen dataclass
        with a derived ``init=False`` field is exactly the shape that can come
        back half-built, so the round trip is pinned rather than assumed."""
        values = np.array([1.0, 0.0, 3.0, np.nan, 5.0, 6.0])
        original = FitSeries.compact(values)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.policy == original.policy
        np.testing.assert_array_equal(restored.valid_positions, original.valid_positions)
        train = np.arange(3, 6, dtype=np.int64)
        np.testing.assert_array_equal(restored.window(train), original.window(train))

    def test_a_bad_policy_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 'compact' or 'none'"):
            FitSeries(values=np.ones(5), policy="drop")  # type: ignore[arg-type]


class TestTooLittleHistoryIsExplicit:
    """A short window is a reported failure, never a silent short fit."""

    def test_it_raises_rather_than_handing_over_a_short_window(self) -> None:
        values = np.arange(1.0, 11.0)
        values[2] = 0.0
        fit = FitSeries.compact(values)
        train = np.arange(0, 5, dtype=np.int64)  # wants 5 valid, only 4 exist by position 4
        with pytest.raises(InsufficientHistoryError, match="only 4 valid observations"):
            fit.window(train)

    def test_the_message_says_what_was_asked_for_and_what_exists(self) -> None:
        values = np.arange(1.0, 11.0)
        values[2] = np.nan
        with pytest.raises(InsufficientHistoryError) as excinfo:
            FitSeries.compact(values).window(np.arange(0, 5, dtype=np.int64))
        message = str(excinfo.value)
        assert "origin 4" in message and "5-observation fit window" in message
        assert "'compact'" in message

    def test_the_shortfall_is_a_prefix_and_clears_once_history_accumulates(self) -> None:
        values = np.arange(1.0, 31.0)
        values[2] = 0.0
        fit = FitSeries.compact(values)
        with pytest.raises(InsufficientHistoryError):
            fit.window(np.arange(0, 5, dtype=np.int64))
        # One day later there are five valid observations behind the origin.
        assert fit.window(np.arange(1, 6, dtype=np.int64)).size == 5

    def test_run_backtest_turns_it_into_the_standard_missing_reason_row(self) -> None:
        n, window = 40, 10
        rv = np.full(n, 4e-4)
        rv[3] = 0.0  # inside the very first window, so origin 9 is short
        index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        returns = pd.Series(np.full(n, 0.001), index=index)
        proxy = pd.Series(rv, index=index)
        frame = run_backtest(
            lambda: WindowValidator(),
            returns,
            proxy,
            RollingOriginSplitter(window=window, horizon=1),
            seed=1,
            asset="SHORT",
            proxy_name="rv",
            fit_series=FitSeries.compact(pd.Series(rv, index=index)),
        )
        first = frame.sort_values("origin_index").iloc[0]
        assert first["origin_index"] == window - 1
        assert "InsufficientHistoryError" in str(first["missing_reason"])
        assert "fit_error@9" in str(first["missing_reason"])
        assert math.isnan(float(first["crps"])) and math.isnan(float(first["forecast_var"]))
        assert int(first["fit_origin"]) == -1
        # And it is exactly one origin: the next one has ten valid days behind it.
        assert (
            frame["missing_reason"].astype(str).str.contains("InsufficientHistory").sum() == 1
        )

    def test_a_short_scheduled_fit_fails_its_whole_block_like_any_other(self) -> None:
        """Not a new rule — the existing one, applied to a new failure.

        A failed *scheduled* fit costs its whole refit block, because the
        alternative is refitting off-schedule and reporting a cadence the run
        did not use. Compaction can only trigger that at a series' start, so
        the cost is bounded by one block; pinned here so the interaction is
        explicit rather than discovered later in a results table.
        """
        n, window, refit_every = 60, 10, 21
        rv = np.full(n, 4e-4)
        rv[3] = 0.0
        index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        frame = run_backtest(
            lambda: WindowValidator(),
            pd.Series(np.full(n, 0.001), index=index),
            pd.Series(rv, index=index),
            RollingOriginSplitter(window=window, horizon=1, refit_every=refit_every),
            seed=1,
            asset="BLOCK",
            proxy_name="rv",
            fit_series=FitSeries.compact(pd.Series(rv, index=index)),
        ).sort_values("origin_index")
        short = frame["missing_reason"].astype(str).str.contains("InsufficientHistory")
        # The whole first block, and every row of it names the origin that failed.
        assert int(short.sum()) == refit_every
        assert frame.loc[short, "origin_index"].tolist() == list(range(9, 9 + refit_every))
        assert frame.loc[short, "missing_reason"].astype(str).str.contains("@9").all()
        # The next scheduled fit has enough history and the cell recovers.
        assert not bool(short.iloc[refit_every:].any())


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def defective() -> FitSeries:
    """A 600-day series peppered with both kinds of defect."""
    rng = np.random.default_rng(7)
    values = rng.uniform(1e-4, 9e-4, size=600)
    values[rng.choice(600, size=60, replace=False)] = 0.0
    values[rng.choice(600, size=30, replace=False)] = np.nan
    return FitSeries.compact(values)


class TestCompactionCannotReachForward:
    """The temporal-integrity claim, stated three ways.

    ``.claude/skills/leakage-check`` items 1 (index arithmetic at boundaries)
    and 2 (splitter monopoly): compaction rewrites *which* past observations a
    window contains, so the thing to prove is that it only ever reaches
    backwards and that the origin still bounds it.
    """

    def test_every_window_ends_at_or_before_its_origin(self, defective: FitSeries) -> None:
        splitter = RollingOriginSplitter(window=100, horizon=1)
        checked = 0
        for origin in splitter.split(defective.size):
            try:
                kept = defective.window_positions(origin.train)
            except InsufficientHistoryError:
                continue
            assert int(kept[-1]) <= origin.origin
            assert int(kept.max()) <= origin.origin
            checked += 1
        assert checked > 400, "the sweep must actually cover the series"

    def test_every_dropped_day_lies_strictly_in_the_past_of_its_origin(
        self, defective: FitSeries
    ) -> None:
        splitter = RollingOriginSplitter(window=100, horizon=1)
        dropped_anywhere = 0
        for origin in splitter.split(defective.size):
            try:
                dropped = defective.dropped_positions(origin.train)
            except InsufficientHistoryError:
                continue
            assert np.all(dropped < origin.origin), (
                "compaction dropped a day at or after its own origin, which means the "
                "window was shifted forward rather than extended backwards"
            )
            dropped_anywhere += int(dropped.size)
        assert dropped_anywhere > 0, "a series with no dropped day proves nothing here"

    def test_a_window_never_contains_a_value_from_after_the_origin(
        self, defective: FitSeries
    ) -> None:
        """The same claim by value rather than by index: corrupt everything
        after the origin and the window is unchanged."""
        splitter = RollingOriginSplitter(window=100, horizon=1)
        checked = 0
        for origin in list(splitter.split(defective.size))[::37]:
            try:
                before = defective.window(origin.train)
            except InsufficientHistoryError:
                continue
            corrupted = defective.values.copy()
            corrupted[origin.origin + 1 :] = 1e6
            np.testing.assert_array_equal(
                before, FitSeries.compact(corrupted).window(origin.train)
            )
            checked += 1
        assert checked >= 10

    def test_the_canary_is_not_inert(self, defective: FitSeries) -> None:
        """Corrupting from *before* the origin must change the window, or the
        test above would pass on an implementation that ignores its input."""
        splitter = RollingOriginSplitter(window=100, horizon=1)
        origin = list(splitter.split(defective.size))[200]
        corrupted = defective.values.copy()
        corrupted[origin.origin - 10 :] = 1e6
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                defective.window(origin.train),
                FitSeries.compact(corrupted).window(origin.train),
            )

    def test_validity_of_a_later_day_cannot_change_an_earlier_window(
        self, defective: FitSeries
    ) -> None:
        """The subtler version: not "a later *value*" but "a later day's
        *validity*". Making a future day invalid changes the set of valid
        positions, and must still leave earlier windows alone."""
        splitter = RollingOriginSplitter(window=100, horizon=1)
        origin = list(splitter.split(defective.size))[150]
        corrupted = defective.values.copy()
        corrupted[origin.origin + 1 :] = 0.0  # every later day becomes invalid
        np.testing.assert_array_equal(
            defective.window(origin.train), FitSeries.compact(corrupted).window(origin.train)
        )

    def test_the_splitter_still_runs_on_the_full_calendar(self) -> None:
        """Compaction must not change *which* days are scored: origins and
        targets are calendar positions, and the splitter never sees the
        policy."""
        values = np.full(50, 4e-4)
        values[[10, 20, 30]] = 0.0
        splitter = RollingOriginSplitter(window=10, horizon=1)
        origins = list(splitter.split(50))
        assert [o.origin for o in origins] == list(range(9, 49))
        assert [int(o.test[0]) for o in origins] == list(range(10, 50))
        # The invalid days are ordinary origins and ordinary targets.
        assert {10, 20, 30} <= {o.origin for o in origins}
        assert {10, 20, 30} <= {int(o.test[0]) for o in origins}


# --------------------------------------------------------------------------
# an invalid day is still an origin
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored() -> tuple[pd.DataFrame, int]:
    """One cell over a 60-day series whose day 25 has an exactly-zero target."""
    n, window, bad = 60, 10, 25
    rv = np.full(n, 4e-4)
    rv[bad] = 0.0
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    returns = pd.Series(np.linspace(-0.01, 0.01, n), index=index)
    frame = run_backtest(
        lambda: WindowValidator(),
        returns,
        pd.Series(rv, index=index),
        RollingOriginSplitter(window=window, horizon=1),
        seed=1,
        asset="ONE",
        proxy_name="rv",
        fit_series=FitSeries.compact(pd.Series(rv, index=index)),
    )
    return frame.set_index("origin_index").sort_index(), bad


class TestAnInvalidDayIsStillAnOrigin:
    def test_the_forecast_issued_at_the_invalid_day_is_a_normal_forecast(
        self, scored: tuple[pd.DataFrame, int]
    ) -> None:
        frame, bad = scored
        row = frame.loc[bad]
        assert str(row["missing_reason"]) == ""
        assert math.isfinite(float(row["forecast_var"])) and float(row["forecast_var"]) > 0
        assert math.isfinite(float(row["crps"])) and math.isfinite(float(row["qlike"]))
        assert int(row["fit_origin"]) == bad  # it fitted, at that origin

    def test_only_the_row_whose_target_is_that_day_is_unscorable(
        self, scored: tuple[pd.DataFrame, int]
    ) -> None:
        frame, bad = scored
        flagged = frame.index[frame["missing_reason"].astype(str) != ""]
        assert list(flagged) == [bad - 1], "exactly the origin whose target is the invalid day"
        row = frame.loc[bad - 1]
        assert str(row["missing_reason"]) == "proxy_nonpositive"
        assert math.isnan(float(row["qlike"]))
        # The forecast itself was made and is recorded — only QLIKE is missing.
        assert math.isfinite(float(row["forecast_var"]))
        assert math.isfinite(float(row["crps"]))

    def test_a_zero_target_is_routed_to_the_missing_reason_row_not_to_a_crash(
        self, scored: tuple[pd.DataFrame, int]
    ) -> None:
        """D-018(a). Before the policy a zero reached ``HAR.fit`` and crashed;
        as a *target* it has always belonged on the NaN row, and that is now
        the stated policy rather than an accident of the QLIKE guard."""
        frame, bad = scored
        assert float(frame.loc[bad - 1, "proxy_var"]) == 0.0
        assert "proxy_nonpositive" in set(frame["missing_reason"].astype(str))


# --------------------------------------------------------------------------
# end to end, on the panel's own defects
# --------------------------------------------------------------------------


def _run(
    factory: Any,
    returns: pd.Series,
    primary: pd.Series,
    *,
    policy: str,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    return run_backtest(
        factory,
        returns,
        primary,
        RollingOriginSplitter(window=WINDOW, horizon=1, refit_every=refit_every),
        seed=SEED,
        asset="PANELLIKE",
        proxy_name="overnight_plus_range",
        fit_series=FitSeries.of(primary, policy=policy),  # type: ignore[arg-type]
    )


def _failed(frame: pd.DataFrame) -> int:
    """Origins with no forecast at all — the thing compaction exists to prevent."""
    return int(frame["forecast_var"].isna().sum())


class TestPanelDefectsEndToEnd:
    """HSI's 12 zeros and TWSE's 80 NaN'd bars, through ``run_backtest``."""

    @pytest.mark.parametrize("model", ["har", "tsfm_style"])
    @pytest.mark.parametrize("series", ["hsi_like", "twse_like"])
    def test_the_overwhelming_majority_of_origins_survive(
        self, model: str, series: str, request: pytest.FixtureRequest
    ) -> None:
        returns, primary = request.getfixturevalue(series)
        factory: Any = HAR if model == "har" else WindowValidator
        frame = _run(factory, returns, primary, policy="compact")

        n_origins = len(frame)
        assert n_origins > 1400
        assert _failed(frame) == 0, (
            "compaction should leave every origin fittable: "
            f"{sorted(set(frame.loc[frame['forecast_var'].isna(), 'missing_reason']))[:3]}"
        )
        # The only unscorable rows are the ones whose *target* is unusable, and
        # there are exactly as many as the series has invalid days in range.
        invalid = invalid_target_mask(primary.to_numpy(dtype=np.float64))
        expected = int(invalid[frame["target_index"].to_numpy()].sum())
        flagged = frame["missing_reason"].astype(str) != ""
        assert int(flagged.sum()) == expected
        assert expected > 0, "this fixture is meant to carry defects"
        assert int(flagged.sum()) / n_origins < 0.06

    @pytest.mark.parametrize("series", ["hsi_like", "twse_like"])
    def test_without_the_policy_the_same_series_loses_most_of_its_column(
        self, series: str, request: pytest.FixtureRequest
    ) -> None:
        """The contrast D-018 was taken on. Uncompacted, one unusable day fails
        every window that contains it, so a handful of defects removes most of
        a model's column — silently, as correctly-recorded NaN rows."""
        returns, primary = request.getfixturevalue(series)
        compacted = _run(HAR, returns, primary, policy="compact")
        uncompacted = _run(HAR, returns, primary, policy="none")
        assert _failed(compacted) == 0
        assert _failed(uncompacted) > 0.5 * len(uncompacted)
        assert "fit_error" in " ".join(
            sorted(set(uncompacted["missing_reason"].astype(str)))
        )

    def test_the_model_never_sees_an_invalid_value(
        self, hsi_like: tuple[pd.Series, pd.Series]
    ) -> None:
        returns, primary = hsi_like
        seen: list[NDArray[np.float64]] = []
        frame = _run(_factory(seen), returns, primary, policy="compact")
        assert _failed(frame) == 0
        assert seen, "the recording factory saw no fit at all"
        for window in seen:
            assert window.size == WINDOW
            assert np.isfinite(window).all() and (window > 0.0).all()

    def test_a_dropped_day_makes_the_lag_span_two_calendar_days(
        self, hsi_like: tuple[pd.Series, pd.Series]
    ) -> None:
        """The documented cost of the policy (docs/design.md): for a model that
        reads positional lags, "yesterday" becomes the previous *measured* day.
        Pinned so the caveat cannot quietly stop being true."""
        _, primary = hsi_like
        values = primary.to_numpy(dtype=np.float64)
        bad = int(np.flatnonzero(invalid_target_mask(values))[0])
        fit = FitSeries.compact(primary)
        train = np.arange(bad - WINDOW + 1, bad + 1, dtype=np.int64)
        kept = fit.window_positions(train)
        # The most recent observation the model sees is the day before the
        # invalid one: one position back, two calendar days.
        assert int(kept[-1]) == bad - 1
        assert primary.index[bad] - primary.index[int(kept[-1])] >= pd.Timedelta(days=1)
        assert kept.size == WINDOW

    def test_compaction_changes_nothing_on_a_clean_series(self) -> None:
        returns, primary = _panel_targets(_well_formed_bars())
        assert not invalid_target_mask(primary.to_numpy(dtype=np.float64)).any()
        compacted = _run(HAR, returns, primary, policy="compact")
        uncompacted = _run(HAR, returns, primary, policy="none")
        scores = ["forecast_var", "crps", "qlike", "log_score", "es_0p01"]
        pd.testing.assert_frame_equal(compacted[scores], uncompacted[scores])
        # ... except the identity, which records the policy on purpose.
        assert compacted.attrs["config_hash"] != uncompacted.attrs["config_hash"]


# --------------------------------------------------------------------------
# the real archives, where they exist
# --------------------------------------------------------------------------

RAW_ROOT = Path(__file__).parents[1] / "data" / "raw"
CACHE_ROOT = Path(__file__).parents[1] / "data" / "cache"

#: The real HSI and TWSE series are the two D-018 was decided on, and the
#: numbers in docs/decisions.md and docs/P2_INTEGRATION.md §11.3 come from
#: them. The archives are hand-downloaded and never committed
#: (docs/data_licenses.md), so this can only run where a human has unpacked
#: them — never in CI, which is also where it must never run: the fixtures
#: above reproduce the same two defects by mechanism and carry the contract.
_HAS_ARCHIVES = RAW_ROOT.is_dir() and not os.environ.get("CI")


@pytest.mark.skipif(not _HAS_ARCHIVES, reason=f"needs the Stooq archives under {RAW_ROOT}")
@pytest.mark.parametrize(
    ("asset", "expected_invalid"), [("HSI", 13), ("TWSE", 80)]
)
def test_the_real_panel_series_keep_every_origin(asset: str, expected_invalid: int) -> None:
    """D-018 on the data it was decided on, not on a reconstruction.

    HSI's 13 invalid days (12 exactly-zero targets plus one NaN'd bar) and
    TWSE's 80 are the counts docs/PANEL_REPORT.md §3-§4 measures. Compacted,
    HAR forecasts at every origin of both and the only flagged rows are the
    ones whose own target is unmeasurable.
    """
    from volbench.data.panel import FIT_WINDOW_DEFAULT, build_equity_series

    series = build_equity_series(asset, raw_root=RAW_ROOT, cache_root=CACHE_ROOT)
    primary = series.primary.iloc[1:]
    returns = log_returns(series.frame.close).iloc[1:]
    assert series.invalid_target_days == expected_invalid

    frame = run_backtest(
        HAR,
        returns,
        primary,
        RollingOriginSplitter(window=FIT_WINDOW_DEFAULT, horizon=1, refit_every=REFIT_EVERY),
        seed=SEED,
        asset=asset,
        proxy_name="overnight_plus_range",
        fit_series=FitSeries.compact(primary),
    )
    assert len(frame) > 4000
    assert _failed(frame) == 0, "every origin must be fittable under compaction"

    invalid = invalid_target_mask(primary.to_numpy(dtype=np.float64))
    flagged = frame["missing_reason"].astype(str) != ""
    assert int(flagged.sum()) == int(invalid[frame["target_index"].to_numpy()].sum())


# --------------------------------------------------------------------------
# the config hash
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inputs() -> tuple[pd.Series, pd.Series]:
    """``(returns, rv)`` on one calendar, with one invalid day in the rv."""
    n = 60
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    rv = np.full(n, 4e-4)
    rv[40] = 0.0
    return pd.Series(np.linspace(-0.01, 0.01, n), index=index), pd.Series(rv, index=index)


class TestThePolicyIsPartOfTheExperiment:
    def _hash(self, inputs: tuple[pd.Series, pd.Series], fit_series: Any) -> str:
        returns, rv = inputs
        frame = run_backtest(
            lambda: WindowValidator(),
            returns,
            rv,
            RollingOriginSplitter(window=10, horizon=1),
            seed=1,
            asset="HASH",
            proxy_name="rv",
            fit_series=fit_series,
        )
        return str(frame.attrs["config_hash"])

    def test_the_two_arms_can_never_share_a_cache_entry(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        _, rv = inputs
        assert self._hash(inputs, FitSeries.compact(rv)) != self._hash(inputs, FitSeries.raw(rv))

    def test_a_bare_array_means_no_policy_and_hashes_as_it_always_did(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        """A caller who passes a plain series gets the pre-D-018 behaviour and
        the pre-D-018 identity: adding the policy must not silently move the
        hash of a run that never opted into it."""
        _, rv = inputs
        assert self._hash(inputs, rv) == self._hash(inputs, FitSeries.raw(rv))

    def test_the_policy_is_recorded_under_protocol(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        returns, rv = inputs
        frame = run_backtest(
            lambda: WindowValidator(),
            returns,
            rv,
            RollingOriginSplitter(window=10, horizon=1),
            seed=1,
            asset="HASH",
            proxy_name="rv",
            fit_series=FitSeries.compact(rv),
        )
        assert frame.attrs["config"]["protocol"] == {"invalid_target_policy": "compact"}

    def test_a_fit_series_of_the_wrong_length_is_refused(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        """Bare arrays throughout, so the length guard is what answers: on
        indexed inputs the calendar guard catches it first, which the next two
        tests cover."""
        returns, rv = inputs
        with pytest.raises(ValueError, match="expected 60 to match series"):
            run_backtest(
                lambda: WindowValidator(),
                returns.to_numpy(),
                rv.to_numpy(),
                RollingOriginSplitter(window=10, horizon=1),
                seed=1,
                asset="HASH",
                proxy_name="rv",
                fit_series=FitSeries.compact(rv.to_numpy()[:-1]),
            )

    def test_a_bare_fit_series_beside_indexed_inputs_is_refused(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        """The wrapper must not become a way round the calendar guard: a
        ``FitSeries`` built from a bare array carries no index, and mixing one
        with indexed inputs is exactly what that guard refuses."""
        returns, rv = inputs
        with pytest.raises(ValueError, match="but fit_series is a bare array"):
            run_backtest(
                lambda: WindowValidator(),
                returns,
                rv,
                RollingOriginSplitter(window=10, horizon=1),
                seed=1,
                asset="HASH",
                proxy_name="rv",
                fit_series=FitSeries.compact(rv.to_numpy()),
            )

    def test_a_fit_series_off_calendar_is_still_refused(
        self, inputs: tuple[pd.Series, pd.Series]
    ) -> None:
        """The calendar guard must see through the wrapper: a ``FitSeries``
        carries the index it was built from precisely so this check survives."""
        returns, rv = inputs
        shifted = pd.Series(rv.to_numpy(), index=rv.index + pd.Timedelta(days=1))
        with pytest.raises(ValueError, match="not on the same calendar"):
            run_backtest(
                lambda: WindowValidator(),
                returns,
                rv,
                RollingOriginSplitter(window=10, horizon=1),
                seed=1,
                asset="HASH",
                proxy_name="rv",
                fit_series=FitSeries.compact(shifted),
            )
