"""Economic value: the volatility-targeting backtest (`volbench.econ`, D-029).

``TestAlignment`` is the reason this file exists. The leakage-critical line in
``econ.py`` is which return a position earns, and a wrong answer there does not
raise, does not produce NaNs, and does not look wrong — it produces a better
Sharpe. So it is pinned by a test that shifts the realized-return column by one
row and asserts the result moves materially, with an inert-proof companion
showing the shift is detectable at all.

Everything else here is arithmetic that has to be exactly right (positions,
turnover, costs, drawdown) plus the conventions the paper will have to state
(annualization per asset, the 10 bps default).
"""

from __future__ import annotations

import ast
import functools
import inspect
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from volbench import econ
from volbench.econ import (
    CRYPTO_ASSETS,
    CRYPTO_PERIODS_PER_YEAR,
    DEFAULT_COST_BPS,
    EQUITY_PERIODS_PER_YEAR,
    periods_per_year_for,
    volatility_target_backtest,
)

HASH = "a" * 64


def rows(
    forecast_var: list[float] | np.ndarray,
    realized_return: list[float] | np.ndarray,
    *,
    asset: str = "SPY",
    horizon: int = 1,
    config_hash: str = HASH,
    model: str = "toy",
) -> pd.DataFrame:
    """A minimal stored-results frame: exactly the columns the backtest needs."""
    n = len(forecast_var)
    return pd.DataFrame(
        {
            "config_hash": [config_hash] * n,
            "asset": [asset] * n,
            "model": [model] * n,
            "origin_index": np.arange(n, dtype=np.int64),
            "horizon": np.full(n, horizon, dtype=np.int64),
            "target_index": np.arange(n, dtype=np.int64) + horizon,
            "forecast_var": np.asarray(forecast_var, dtype=np.float64),
            "realized_return": np.asarray(realized_return, dtype=np.float64),
        }
    )


def oracle_rows(n: int = 3000, seed: int = 0, c: float = 0.15, sd: float = 1.0) -> pd.DataFrame:
    """Rows whose forecast is the *true* variance of that row's own return.

    The drift is proportional to the day's volatility (``mu_t = c * sigma_t``),
    so a correctly aligned vol-targeting strategy earns a constant risk premium
    at a constant volatility — its Sharpe is ``c * sqrt(252)`` by construction
    and its realized vol is the target. That makes both quantities sharp enough
    to detect a one-day misalignment against.

    ``sd`` is deliberately large (log-vol dispersion 1.0, far above a real
    index's): the damage a one-day shift does is governed by the dispersion of
    ``sigma_{t+1} / sigma_t``, so a persistent-volatility series would hide the
    bug this test exists to catch. That is itself the finding — see
    ``test_the_shift_is_hardest_to_see_exactly_where_volatility_is_persistent``.
    """
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.normal(np.log(0.01), sd, n))
    returns = c * sigma + sigma * rng.standard_normal(n)
    return rows(sigma**2, returns)


def shifted(frame: pd.DataFrame, by: int) -> pd.DataFrame:
    """The same rows with the realized-return column moved ``by`` rows.

    This is the mistake being guarded against, made concrete: every forecast
    keeps its place and every position is sized identically, but each position
    now earns a *neighbouring* day's return.
    """
    out = frame.copy()
    out["realized_return"] = out["realized_return"].shift(by).fillna(0.0)
    return out


# --------------------------------------------------------------------------
# THE ALIGNMENT
# --------------------------------------------------------------------------


class TestAlignment:
    """The forecast issued AT t sizes the position held INTO t+1."""

    #: No costs and a generous cap: this class is about *which return a
    #: position earns*, and turnover charges would only add noise to that.
    KW: ClassVar[dict[str, Any]] = {"leverage_cap": 3.0, "cost_bps": 0.0}

    def test_the_arithmetic_is_position_from_this_row_times_this_rows_return(self) -> None:
        """Pinned by hand, with no statistics in the way. Row ``i``'s position
        comes from row ``i``'s forecast and earns row ``i``'s realized return —
        and row ``i``'s realized return is dated ``origin + horizon``, which the
        splitter guarantees is strictly after the origin."""
        variances = np.array([1e-4, 4e-4, 1e-4])
        returns = np.array([0.01, -0.02, 0.005])
        result = volatility_target_backtest(
            rows(variances, returns), leverage_cap=10.0, cost_bps=0.0
        )

        daily_target = 0.10 / math.sqrt(EQUITY_PERIODS_PER_YEAR)
        expected_positions = daily_target / np.sqrt(variances)
        assert result.positions == pytest.approx(expected_positions)
        assert result.net_returns == pytest.approx(expected_positions * np.expm1(returns))

    def test_shifting_the_returns_by_one_day_changes_the_sharpe_materially(self) -> None:
        """The gate. If the position earned the wrong day's return, this is the
        number that would move — so it has to move here."""
        for seed in range(4):
            frame = oracle_rows(seed=seed)
            aligned = volatility_target_backtest(frame, **self.KW)
            forward = volatility_target_backtest(shifted(frame, -1), **self.KW)
            backward = volatility_target_backtest(shifted(frame, 1), **self.KW)

            assert aligned.sharpe > 1.5, seed
            assert forward.sharpe < 0.75 * aligned.sharpe, (seed, forward.sharpe, aligned.sharpe)
            assert backward.sharpe < 0.75 * aligned.sharpe, (seed, backward.sharpe, aligned.sharpe)

    def test_only_the_aligned_run_actually_hits_its_volatility_target(self) -> None:
        """The second, sharper symptom, and the one a practitioner would notice
        first: vol targeting that lands 5x off its target is not targeting
        anything."""
        frame = oracle_rows(seed=1)
        aligned = volatility_target_backtest(frame, **self.KW)
        misaligned = volatility_target_backtest(shifted(frame, -1), **self.KW)

        assert aligned.annual_vol == pytest.approx(0.10, rel=0.05)
        assert misaligned.annual_vol > 3.0 * aligned.annual_vol

    def test_the_shift_is_hardest_to_see_exactly_where_volatility_is_persistent(self) -> None:
        """Reported because it bounds what the test above proves.

        The damage a one-day shift does is governed by the dispersion of
        ``sigma_{t+1}/sigma_t``. On a strongly persistent series that ratio is
        near 1, so the misaligned Sharpe is only a few percent off and no
        statistical check on real data would catch the bug. That is precisely
        why the alignment is pinned structurally (the hand-computed test above)
        and not left to a plausibility check on a real backtest.
        """
        rng = np.random.default_rng(7)
        n = 4000
        log_sigma = np.empty(n)
        log_sigma[0] = np.log(0.01)
        for t in range(1, n):
            log_sigma[t] = np.log(0.01) + 0.97 * (log_sigma[t - 1] - np.log(0.01)) + 0.25 * (
                rng.standard_normal()
            )
        sigma = np.exp(log_sigma)
        frame = rows(sigma**2, 0.15 * sigma + sigma * rng.standard_normal(n))

        aligned = volatility_target_backtest(frame, **self.KW)
        misaligned = volatility_target_backtest(shifted(frame, -1), **self.KW)
        assert misaligned.sharpe > 0.85 * aligned.sharpe  # nearly invisible

    def test_no_column_is_shifted_lagged_or_reindexed_inside_the_module(self) -> None:
        """A structural belt: the alignment is "the row", so any ``shift`` or
        ``reindex`` in this module would be a second, competing opinion about
        which return a position earns."""
        source = inspect.getsource(econ)
        for forbidden in (".shift(", ".reindex(", ".tshift(", "np.roll("):
            assert forbidden not in source, forbidden


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------


class TestPositionSizing:
    def test_the_cap_binds_exactly_where_it_should(self) -> None:
        daily_target = 0.10 / math.sqrt(EQUITY_PERIODS_PER_YEAR)
        cap = 2.0
        binding = (daily_target / cap) ** 2  # forecast variance at which cap == ratio
        frame = rows([binding * 4.0, binding, binding / 4.0], [0.0, 0.0, 0.0])
        result = volatility_target_backtest(frame, leverage_cap=cap, cost_bps=0.0)
        assert result.positions == pytest.approx([cap / 2.0, cap, cap])

    def test_an_unusable_forecast_means_no_position_not_a_dropped_row(self) -> None:
        """Dropping would splice two non-adjacent days into one turnover step
        and under-charge the costs of getting out."""
        frame = rows([1e-4, np.nan, 0.0, -1e-4, 1e-4], [0.01, 0.01, 0.01, 0.01, 0.01])
        result = volatility_target_backtest(frame, cost_bps=0.0)
        assert result.n_periods == 5
        assert result.n_flat == 3
        assert list(result.positions[1:4]) == [0.0, 0.0, 0.0]
        assert result.net_returns[1:4] == pytest.approx([0.0, 0.0, 0.0])

    def test_a_flat_day_costs_the_round_trip_out_of_the_position(self) -> None:
        frame = rows([1e-4, np.nan, 1e-4], [0.0, 0.0, 0.0])
        result = volatility_target_backtest(frame, cost_bps=100.0)
        size = result.positions[0]
        # in (0 -> size), out (size -> 0), back in (0 -> size)
        assert result.avg_daily_turnover == pytest.approx(size)
        assert result.net_returns == pytest.approx(-np.array([size, size, size]) * 0.01)

    def test_a_nonfinite_return_under_a_live_position_is_refused(self) -> None:
        frame = rows([1e-4, 1e-4], [0.01, np.nan])
        with pytest.raises(ValueError, match="must be finite"):
            volatility_target_backtest(frame)

    def test_a_nonfinite_return_on_a_flat_day_is_harmless(self) -> None:
        """No position, no P&L: the day's return cannot matter, and refusing it
        would throw away a cell for a day it never traded."""
        frame = rows([1e-4, np.nan], [0.01, np.nan])
        result = volatility_target_backtest(frame, cost_bps=0.0)
        assert result.net_returns[1] == 0.0

    def test_sizes_are_refused_before_they_can_be_silly(self) -> None:
        frame = rows([1e-4], [0.0])
        with pytest.raises(ValueError, match="leverage_cap"):
            volatility_target_backtest(frame, leverage_cap=0.0)
        with pytest.raises(ValueError, match="annual_target_vol"):
            volatility_target_backtest(frame, annual_target_vol=0.0)
        with pytest.raises(ValueError, match="cost_bps"):
            volatility_target_backtest(frame, cost_bps=-1.0)


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


class TestTransactionCosts:
    def test_the_default_is_the_research_designs_ten_basis_points(self) -> None:
        assert DEFAULT_COST_BPS == 10.0
        frame = oracle_rows(n=500, seed=2)
        assert volatility_target_backtest(frame).cost_bps == 10.0

    def test_turnover_is_the_absolute_change_in_position_from_a_flat_start(self) -> None:
        frame = rows([1e-4, 4e-4, 4e-4], [0.0, 0.0, 0.0])
        result = volatility_target_backtest(frame, cost_bps=0.0, leverage_cap=10.0)
        p = result.positions
        expected = np.abs(np.diff(p, prepend=0.0))
        assert result.avg_daily_turnover == pytest.approx(float(np.mean(expected)))
        assert expected[0] == pytest.approx(p[0])  # entering the book is a real trade
        assert expected[2] == pytest.approx(0.0)  # an unchanged position trades nothing

    def test_zero_cost_makes_net_and_gross_agree(self) -> None:
        frame = oracle_rows(n=800, seed=3)
        free = volatility_target_backtest(frame, cost_bps=0.0, leverage_cap=3.0)
        assert free.sharpe == pytest.approx(free.gross_sharpe)
        assert free.annual_cost_drag == 0.0

    def test_the_drag_scales_linearly_with_the_rate(self) -> None:
        frame = oracle_rows(n=800, seed=4)
        cheap = volatility_target_backtest(frame, cost_bps=10.0, leverage_cap=3.0)
        dear = volatility_target_backtest(frame, cost_bps=30.0, leverage_cap=3.0)
        assert dear.annual_cost_drag == pytest.approx(3.0 * cheap.annual_cost_drag)
        assert dear.sharpe < cheap.sharpe < dear.gross_sharpe
        assert cheap.gross_sharpe == pytest.approx(dear.gross_sharpe)

    def test_annual_turnover_is_the_daily_figure_annualized(self) -> None:
        frame = oracle_rows(n=600, seed=5)
        result = volatility_target_backtest(frame, leverage_cap=3.0)
        assert result.annual_turnover == pytest.approx(
            result.avg_daily_turnover * EQUITY_PERIODS_PER_YEAR
        )


# --------------------------------------------------------------------------
# annualization
# --------------------------------------------------------------------------


class TestAnnualization:
    def test_equities_get_252_and_crypto_gets_365(self) -> None:
        assert periods_per_year_for("SPY") == EQUITY_PERIODS_PER_YEAR == 252.0
        assert periods_per_year_for("BTC-USD") == CRYPTO_PERIODS_PER_YEAR == 365.0
        assert periods_per_year_for("ETH-USD") == 365.0

    def test_the_crypto_list_cannot_drift_from_the_panels(self) -> None:
        """D-024's discipline: two lists that can disagree are a trap. econ.py
        keeps its own copy so it depends on no ingestion code; this is what
        stops the copy going stale."""
        from volbench.data.panel import CRYPTO_PANEL

        assert frozenset(CRYPTO_PANEL) == CRYPTO_ASSETS

    def test_the_calendar_is_taken_from_the_asset_and_can_be_overridden(self) -> None:
        variances = np.full(400, 1e-4)
        returns = np.full(400, 0.001)
        equity = volatility_target_backtest(rows(variances, returns, asset="SPY"), cost_bps=0.0)
        crypto = volatility_target_backtest(
            rows(variances, returns, asset="BTC-USD"), cost_bps=0.0
        )
        override = volatility_target_backtest(
            rows(variances, returns, asset="BTC-USD"), cost_bps=0.0, periods_per_year=252.0
        )
        assert (equity.periods_per_year, crypto.periods_per_year) == (252.0, 365.0)
        assert override.periods_per_year == 252.0
        assert override.positions == pytest.approx(equity.positions)

    def test_one_annual_target_means_one_risk_on_either_calendar(self) -> None:
        """The vol target is quoted annualized, so a 10% target has to mean the
        same daily risk whether the year has 252 days or 365 — otherwise a
        crypto and an equity Sharpe are not comparable."""
        variances = np.full(300, 1e-4)
        equity = volatility_target_backtest(
            rows(variances, np.zeros(300), asset="SPY"), cost_bps=0.0
        )
        crypto = volatility_target_backtest(
            rows(variances, np.zeros(300), asset="BTC-USD"), cost_bps=0.0
        )
        assert equity.positions[0] * math.sqrt(252.0) == pytest.approx(
            crypto.positions[0] * math.sqrt(365.0)
        )

    def test_a_nonsense_calendar_is_refused(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            volatility_target_backtest(rows([1e-4], [0.0]), periods_per_year=0.0)


# --------------------------------------------------------------------------
# reported quantities
# --------------------------------------------------------------------------


class TestReportedQuantities:
    def test_a_constant_forecast_and_return_gives_the_textbook_numbers(self) -> None:
        """No costs, no dispersion: every number is checkable by hand."""
        n = 252
        daily = 0.0004
        result = volatility_target_backtest(
            rows(np.full(n, 1e-4), np.full(n, daily)), cost_bps=0.0
        )
        position = (0.10 / math.sqrt(252.0)) / 0.01
        per_day = position * math.expm1(daily)

        assert result.avg_leverage == pytest.approx(position)
        assert result.annual_return == pytest.approx((1.0 + per_day) ** 252 - 1.0)
        assert result.annual_vol == pytest.approx(0.0, abs=1e-12)
        # Zero dispersion: the ratio is undefined, and must come back as NaN
        # rather than as the ~1e16 that dividing by the float noise floor gives.
        assert math.isnan(result.sharpe)
        assert result.max_drawdown == pytest.approx(0.0)

    def test_max_drawdown_is_the_worst_peak_to_trough_fall(self) -> None:
        # +10%, -20%, +5% on a unit position (leverage forced to 1 via the cap).
        returns = np.log1p(np.array([0.10, -0.20, 0.05]))
        result = volatility_target_backtest(
            rows(np.full(3, 1e-4), returns), leverage_cap=1.0, cost_bps=0.0, annual_target_vol=10.0
        )
        assert result.positions == pytest.approx([1.0, 1.0, 1.0])
        assert result.equity == pytest.approx([1.10, 0.88, 0.924])
        assert result.max_drawdown == pytest.approx(1.0 - 0.88 / 1.10)

    def test_a_drawdown_from_the_starting_point_counts(self) -> None:
        """The peak starts at 1.0, so a strategy that only ever loses still has
        a drawdown — a curve that never exceeds its start is not flat."""
        returns = np.log1p(np.array([-0.10, -0.10]))
        result = volatility_target_backtest(
            rows(np.full(2, 1e-4), returns), leverage_cap=1.0, cost_bps=0.0, annual_target_vol=10.0
        )
        assert result.max_drawdown == pytest.approx(1.0 - 0.81)

    def test_a_wipeout_is_reported_not_compounded_through(self) -> None:
        returns = np.log1p(np.array([-0.60, 0.50]))
        with pytest.warns(RuntimeWarning, match="wiped out"):
            result = volatility_target_backtest(
                rows(np.full(2, 1e-4), returns),
                leverage_cap=2.0,
                cost_bps=0.0,
                annual_target_vol=10.0,
            )
        assert result.ruined is True
        assert result.equity[-1] == 0.0
        assert result.annual_return == -1.0
        assert result.max_drawdown == pytest.approx(1.0)

    def test_the_summary_dict_is_json_ready_and_holds_no_arrays(self) -> None:
        import json

        result = volatility_target_backtest(oracle_rows(n=400, seed=6), leverage_cap=3.0)
        summary = result.as_dict()
        json.dumps(summary)
        assert "positions" not in summary and "equity" not in summary
        assert summary["n_periods"] == 400
        assert set(summary) >= {
            "annual_return",
            "annual_vol",
            "sharpe",
            "gross_sharpe",
            "max_drawdown",
            "avg_leverage",
            "annual_turnover",
        }

    def test_the_per_period_series_line_up_with_the_rows(self) -> None:
        result = volatility_target_backtest(oracle_rows(n=120, seed=8), leverage_cap=3.0)
        assert (
            result.positions.size
            == result.net_returns.size
            == result.equity.size
            == result.n_periods
            == 120
        )


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


class TestInputContract:
    def test_two_cells_at_once_are_refused(self) -> None:
        both = pd.concat(
            [rows([1e-4], [0.01]), rows([1e-4], [0.01], config_hash="b" * 64)],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="exactly one cell"):
            volatility_target_backtest(both)

    def test_two_assets_at_once_are_refused(self) -> None:
        both = pd.concat(
            [rows([1e-4], [0.01]), rows([1e-4], [0.01], asset="DIA")], ignore_index=True
        )
        with pytest.raises(ValueError, match="exactly one asset"):
            volatility_target_backtest(both)

    def test_missing_columns_are_named(self) -> None:
        frame = rows([1e-4], [0.01]).drop(columns=["forecast_var"])
        with pytest.raises(ValueError, match="forecast_var"):
            volatility_target_backtest(frame)

    def test_the_horizon_is_selected_never_inferred(self) -> None:
        """A cell run at h=5 holds five strategies, not one."""
        multi = pd.concat(
            [rows([1e-4] * 3, [0.01] * 3, horizon=h) for h in (1, 2)], ignore_index=True
        )
        one = volatility_target_backtest(multi, horizon=1)
        two = volatility_target_backtest(multi, horizon=2)
        assert one.n_periods == two.n_periods == 3
        assert (one.horizon, two.horizon) == (1, 2)
        with pytest.raises(ValueError, match="no rows at horizon 5"):
            volatility_target_backtest(multi, horizon=5)

    def test_duplicate_origins_are_refused(self) -> None:
        frame = rows([1e-4] * 2, [0.01] * 2)
        frame["origin_index"] = [3, 3]
        with pytest.raises(ValueError, match="strictly increasing"):
            volatility_target_backtest(frame)

    def test_rows_are_sorted_by_origin_before_anything_is_computed(self) -> None:
        """Turnover is a difference between neighbours, so row order is not
        cosmetic — a store read that came back shuffled must not change the
        cost."""
        frame = rows([1e-4, 9e-4, 4e-4], [0.01, -0.01, 0.02])
        shuffled = frame.iloc[[2, 0, 1]].reset_index(drop=True)
        assert volatility_target_backtest(frame, cost_bps=7.0).as_dict() == (
            volatility_target_backtest(shuffled, cost_bps=7.0).as_dict()
        )

    def test_a_non_frame_is_refused(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            volatility_target_backtest({"forecast_var": [1e-4]})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the module boundary
# --------------------------------------------------------------------------


class TestBoundary:
    """econ.py consumes stored rows. It must not be able to run a model."""

    def test_it_does_not_import_the_evaluator_or_any_model(self) -> None:
        source = Path(inspect.getfile(econ)).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        volbench_imports = {name for name in imported if name.startswith("volbench")}
        assert volbench_imports == set(), volbench_imports
        # And nothing reached it by another route: the only volbench objects
        # bound in the module are its own, so there is no model, splitter or
        # evaluator here to call even by accident.
        foreign = {
            name
            for name, value in vars(econ).items()
            if str(getattr(value, "__module__", "")).startswith("volbench")
            and getattr(value, "__module__", "") != "volbench.econ"
        }
        assert foreign == set(), foreign

    def test_it_reads_the_columns_the_store_actually_writes(self) -> None:
        from volbench.results import REQUIRED_COLUMNS as STORED

        assert set(econ.REQUIRED_COLUMNS) - {"horizon"} <= set(STORED) | {"horizon"}
        assert set(econ.REQUIRED_COLUMNS) <= set(STORED)


# --------------------------------------------------------------------------
# end to end, off a real store
# --------------------------------------------------------------------------


def _asset_data(asset: str, seed: int, n: int = 320) -> Any:
    """A synthetic asset for the end-to-end check. Local rather than imported
    from ``test_runner`` because ``tests/`` is not an importable package."""
    from volbench.runner import AssetData

    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    variance = np.exp(rng.normal(np.log(1e-4), 0.35, n))
    returns = rng.normal(0.0, np.sqrt(variance))
    return AssetData(
        asset=asset,
        returns=pd.Series(returns, index=index),
        proxy=pd.Series(variance, index=index),
        proxy_name="synthetic_variance",
        variance=pd.Series(variance, index=index),
    )


class TestFromAStoredCell:
    def test_a_real_backtest_row_set_goes_straight_in(self, tmp_path: Path) -> None:
        """The interface claim: whatever the runner stored is what this reads,
        with no reshaping in between."""
        from volbench.models import EWMA
        from volbench.results import ResultsStore
        from volbench.runner import (
            GridSpec,
            ModelConfig,
            ProtocolArm,
            read_grid_results,
            run_grid,
        )
        store = ResultsStore(tmp_path / "store")
        grid = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("ewma", functools.partial(EWMA, lambda_=0.94)),),
            arms=(ProtocolArm("w120", window=120),),
            seed=7,
        )
        manifest = run_grid(grid, {"AAA": _asset_data("AAA", seed=3)}, store)
        stored = read_grid_results(store, manifest)

        result = volatility_target_backtest(stored)

        assert result.config_hash == manifest.cells[0].config_hash
        assert result.n_periods == manifest.cells[0].n_rows
        assert result.model == "ewma"
        assert result.asset == "AAA"
        assert np.isfinite(result.sharpe)
        assert 0.0 <= result.max_drawdown <= 1.0
        assert result.avg_leverage > 0.0
