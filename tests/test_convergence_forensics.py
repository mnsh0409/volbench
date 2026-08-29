"""Tests for the convergence forensics driver.

Everything here has a decidable answer: a boundary count against a frame built
to contain a known number of them, a constraint against arithmetic written out
by hand, an intersection against sets, and a tie-break rule against a case
designed to tie. The re-fit itself is not exercised here — it needs the panel
and the store — but the check that makes it trustworthy is: ``refit_all``
refuses to return a frame whose ``fit_status`` disagrees with the store's, and
that refusal is what the driver's report rests on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volbench.benchmarks.convergence_forensics import (
    BOUNDARY_TOLERANCES,
    at_bound_table,
    boundary_flags,
    constraint_slack,
    dia_intersection,
    gamma_persistence_table,
    paired_windows,
    window_stats,
)


def _fit(**overrides: object) -> dict[str, object]:
    """One row of the forensics frame, strictly inside every bound."""
    row: dict[str, object] = {
        "asset": "AAA",
        "config": "gjr",
        "fit_origin": 0,
        "date": pd.Timestamp("2020-01-01", tz="UTC"),
        "fallback": False,
        "omega": 0.5,
        "alpha[1]": 0.1,
        "gamma[1]": 0.1,
        "beta[1]": 0.5,
        "slack_low_omega": 0.5,
        "slack_high_omega": 0.5,
        "slack_low_alpha[1]": 0.1,
        "slack_high_alpha[1]": 0.9,
        "slack_low_gamma[1]": 1.1,
        "slack_high_gamma[1]": 1.9,
        "slack_low_beta[1]": 0.5,
        "slack_high_beta[1]": 0.5,
        "alpha_plus_beta": 0.6,
        "persistence": 0.65,
        "kurtosis": 3.0,
        "max_abs_return": 0.02,
        "std": 0.01,
    }
    row.update(overrides)
    return row


class TestWindowStats:
    def test_kurtosis_is_pearsons_so_a_normal_reads_three(self) -> None:
        """Fisher's convention would read 0 here. A three-unit difference in a
        published column is a finding-shaped artifact of a default."""
        draw = np.random.default_rng(11).standard_normal(20_000)
        assert window_stats(draw)["kurtosis"] == pytest.approx(3.0, abs=0.15)

    def test_the_rest_are_the_obvious_statistics_of_the_window(self) -> None:
        window = np.array([-0.03, 0.0, 0.01, 0.02])
        stats = window_stats(window)
        assert stats["n"] == 4.0
        assert stats["max_abs_return"] == pytest.approx(0.03)
        assert stats["n_zero_returns"] == 1.0
        assert stats["std"] == pytest.approx(float(np.std(window, ddof=1)))


class TestConstraintSlack:
    def test_gjr_weights_the_leverage_term_by_a_half(self) -> None:
        """``alpha + gamma/2 + beta <= 1`` is the constraint ``arch`` hands
        SLSQP; ``alpha + beta`` is a different number and is not it."""
        frame = pd.DataFrame([_fit(**{"alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.7})])
        slack = constraint_slack(frame)
        assert slack["stationarity"].iloc[0] == pytest.approx(1.0 - (0.1 + 0.1 + 0.7))
        assert slack["alpha_plus_gamma"].iloc[0] == pytest.approx(0.3)

    def test_a_config_without_gamma_is_read_as_gamma_zero(self) -> None:
        frame = pd.DataFrame([_fit(config="garch11", **{"gamma[1]": np.nan})])
        slack = constraint_slack(frame)
        assert slack["stationarity"].iloc[0] == pytest.approx(1.0 - (0.1 + 0.5))
        assert slack["alpha_plus_gamma"].iloc[0] == pytest.approx(0.1)


class TestBoundaryFlags:
    def test_a_parameter_at_a_bound_is_flagged_and_one_inside_is_not(self) -> None:
        frame = pd.DataFrame(
            [_fit(), _fit(slack_low_omega=0.0), _fit(**{"slack_high_beta[1]": 1e-13})]
        )
        flags = boundary_flags(frame, 1e-12)
        assert flags["omega@low"].tolist() == [False, True, False]
        assert flags["beta[1]@high"].tolist() == [False, False, True]
        assert not flags["alpha[1]@low"].any()

    def test_the_ladder_is_monotone_in_the_tolerance(self) -> None:
        """A looser tolerance can only find more boundaries. A table that broke
        this would be measuring something other than distance to a bound."""
        frame = pd.DataFrame(
            [_fit(slack_low_omega=slack) for slack in (0.0, 1e-11, 1e-9, 1e-7, 1e-5, 1.0)]
        )
        table = at_bound_table(frame, BOUNDARY_TOLERANCES)
        counts = table["omega@low"].tolist()
        assert counts == sorted(counts)
        assert counts[0] == 1 and counts[-1] == 5

    def test_persistence_at_one_counts_a_fit_that_stopped_just_past_it(self) -> None:
        """SLSQP is allowed to stop marginally outside a constraint. Treating
        those as "inside" would drop exactly the fits at the IGARCH edge."""
        frame = pd.DataFrame([_fit(persistence=1.0 + 1e-9), _fit(persistence=0.5)])
        assert boundary_flags(frame, 1e-12)["persistence@1"].tolist() == [True, False]


class TestDiaIntersection:
    def test_it_separates_the_three_way_intersection_from_the_pairwise_ones(self) -> None:
        frame = pd.DataFrame(
            [
                _fit(asset="DIA", config="garch11", fit_origin=10, fallback=True),
                _fit(asset="DIA", config="garch11", fit_origin=20, fallback=True),
                _fit(asset="DIA", config="garch11_t", fit_origin=10, fallback=True),
                _fit(asset="DIA", config="gjr", fit_origin=30, fallback=True),
                _fit(asset="DIA", config="gjr", fit_origin=40, fallback=False),
                _fit(asset="SPY", config="gjr", fit_origin=10, fallback=True),
            ]
        )
        answer = dia_intersection(frame)
        assert answer["per_config"] == {"garch11": [10, 20], "garch11_t": [10], "gjr": [30]}
        assert answer["three_way_intersection"] == []
        assert answer["pairwise_intersections"]["garch11&garch11_t"] == [10]
        assert answer["pairwise_intersections"]["garch11&gjr"] == []
        assert answer["union"] == [10, 20, 30]


class TestPairedWindows:
    @staticmethod
    def _two_assets(*, converged_at: tuple[int, ...], dates_differ: bool = False) -> pd.DataFrame:
        rows = []
        for asset in ("BTC-USD", "ETH-USD"):
            for k, origin in enumerate((0, 21, 42, 63)):
                shift = 1 if (dates_differ and asset == "ETH-USD") else 0
                rows.append(
                    _fit(
                        asset=asset,
                        config="garch11_t",
                        fit_origin=origin,
                        date=pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=k + shift),
                        fallback=(asset == "BTC-USD" and origin not in converged_at),
                        alpha_plus_beta=0.5 + origin / 100.0,
                        kurtosis=3.0 + k,
                    )
                )
        return pd.DataFrame(rows)

    def test_it_reads_both_sides_at_the_same_origins(self) -> None:
        frame = self._two_assets(converged_at=(0, 21, 42, 63))
        table = paired_windows(frame, "garch11_t", "BTC-USD", "ETH-USD", [21])
        assert table.loc[0, "BTC-USD_nearest_converged"] == 21
        assert table.loc[0, "BTC-USD_nearest_gap"] == 0
        assert table.loc[0, "ETH-USD_kurtosis"] == pytest.approx(4.0)

    def test_the_nearest_converged_fit_breaks_ties_to_the_earlier_origin(self) -> None:
        """21 is equidistant from 0 and 42. Left unpinned, the answer would
        depend on a sort order and change between pandas versions."""
        frame = self._two_assets(converged_at=(0, 42))
        table = paired_windows(frame, "garch11_t", "BTC-USD", "ETH-USD", [21])
        assert table.loc[0, "BTC-USD_fallback"]
        assert table.loc[0, "BTC-USD_nearest_converged"] == 0
        assert table.loc[0, "BTC-USD_nearest_alpha_plus_beta"] == pytest.approx(0.5)

    def test_it_refuses_two_series_that_are_not_on_one_calendar(self) -> None:
        """The whole comparison is "the same dates"; two drifted calendars
        would compare different windows and say nothing."""
        with pytest.raises(ValueError, match="one calendar"):
            paired_windows(
                self._two_assets(converged_at=(0,), dates_differ=True),
                "garch11_t",
                "BTC-USD",
                "ETH-USD",
                [21],
            )

    def test_it_refuses_two_series_fitted_at_different_origins(self) -> None:
        frame = self._two_assets(converged_at=(0, 21, 42, 63))
        frame = frame.drop(frame[(frame["asset"] == "ETH-USD") & (frame["fit_origin"] == 63)].index)
        with pytest.raises(ValueError, match="same origins"):
            paired_windows(frame, "garch11_t", "BTC-USD", "ETH-USD", [21])


class TestGammaPersistenceTable:
    def test_it_describes_the_fallbacks_and_the_clean_fits_side_by_side(self) -> None:
        frame = pd.DataFrame(
            [_fit(asset="HSI", fallback=False, **{"gamma[1]": 0.2})] * 3
            + [_fit(asset="HSI", fallback=True, **{"gamma[1]": -0.1})] * 2
        )
        table = gamma_persistence_table(frame, "HSI", "gjr")
        assert list(table.columns) == ["clean", "fallback"]
        assert table.loc[("gamma[1]", "count"), "clean"] == 3
        assert table.loc[("gamma[1]", "count"), "fallback"] == 2
        assert table.loc[("gamma[1]", "50%"), "fallback"] == pytest.approx(-0.1)
