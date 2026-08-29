"""Tests for the J3 comparison-inference driver.

Structural first: the module that assembles every DM, MCS, backtest and
economic-value table must not be able to reach a model. Then each piece the
driver owns has a decidable answer on a synthetic frame — a recursion whose
parameters are known, a calendar that is right and one that is off by one, a
run of hits counted by hand, a Sharpe interval that must be identical twice
under one seed, and the one the brief asks for by name: an MCS run twice,
identical.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volbench import inference
from volbench.benchmarks import inference_tables as it

# --------------------------------------------------------------------------
# structural boundary
# --------------------------------------------------------------------------

#: Everything this driver may name from the package. The analysis layer, the
#: inference and backtest modules, the economic-value module, the store, J2's
#: renderer, and the data package's calendar pieces (crisis tags; the panel
#: and log returns, to rebuild the study's calendar) — nothing that fits.
ALLOWED = {
    "volbench",
    "volbench.analysis",
    "volbench.backtests",
    "volbench.econ",
    "volbench.inference",
    "volbench.results",
    "volbench.benchmarks.loss_tables",
    "volbench.data.crisis",
    "volbench.data.panel",
    "volbench.data.proxies",
}
FORBIDDEN = ("volbench.models", "volbench.evaluate", "volbench.runner", "volbench.execute")


def _volbench_imports(module: object) -> set[str]:
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")  # type: ignore[arg-type]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return {name for name in imported if name.startswith("volbench")}


class TestBoundary:
    def test_its_imports_are_the_declared_ones(self) -> None:
        assert _volbench_imports(it) <= ALLOWED

    def test_it_names_no_model_evaluator_runner_or_executor(self) -> None:
        for name in _volbench_imports(it):
            for forbidden in FORBIDDEN:
                assert name != forbidden and not name.startswith(forbidden + "."), name

    def test_nothing_it_leans_on_names_a_model_either(self) -> None:
        """The allow-list is worth something only if what it allows is clean.
        The package root is the namespace and is excluded: importing anything
        under ``volbench`` runs it, which is why the check is on names."""
        for name in sorted(ALLOWED - {"volbench"}):
            module = importlib.import_module(name)
            for imported in _volbench_imports(module):
                for forbidden in FORBIDDEN:
                    assert not imported.startswith(forbidden), (name, imported)


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------


class TestSeeds:
    def test_seeds_are_stable_distinct_and_31_bit(self) -> None:
        a = it.seed_for("mcs", "SPY", "crps", "range", "headline")
        assert a == it.seed_for("mcs", "SPY", "crps", "range", "headline")
        assert a != it.seed_for("mcs", "SPY", "crps", "semi_quadratic", "headline")
        assert a != it.seed_for("mcs", "DIA", "crps", "range", "headline")
        assert 0 <= a < 2**31


# --------------------------------------------------------------------------
# the near-zero days and the excluded QLIKE
# --------------------------------------------------------------------------


def _grid(n: int = 30, models: tuple[str, ...] = ("a", "b", "c")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for model in models:
        for origin in range(n):
            proxy = 1e-4 * (1.0 + 0.5 * rng.random())
            rows.append(
                {
                    "asset": "AAA",
                    "model_label": model,
                    "model": model,
                    "config_hash": f"{model * 64}"[:64],
                    "origin_index": origin,
                    "target_index": origin + 1,
                    "horizon": 1,
                    "missing_reason": "",
                    "proxy_var": proxy,
                    "qlike": 0.5 + 0.1 * rng.random(),
                    "crps": 0.01 + 0.001 * rng.random(),
                    "realized_return": 0.0,
                }
            )
    return pd.DataFrame(rows)


class TestNearZero:
    def test_only_scored_targets_below_the_threshold_are_excluded(self) -> None:
        grid = _grid()
        grid.loc[grid["target_index"] == 5, "proxy_var"] = 5e-9  # near zero, scored
        grid.loc[grid["target_index"] == 6, "proxy_var"] = 0.0  # exactly zero: already NaN QLIKE
        grid.loc[grid["target_index"] == 6, "qlike"] = np.nan
        out = it.with_qlike_excluding_near_zero(grid)
        assert out.loc[out["target_index"] == 5, it.QLIKE_EX].isna().all()
        assert out.loc[out["target_index"] != 5, it.QLIKE_EX].equals(
            out.loc[out["target_index"] != 5, "qlike"]
        )
        listed = it.near_zero_targets(grid)
        assert listed["target_index"].tolist() == [5]
        assert listed["n_models_scored"].tolist() == [3]
        assert 0.0 < float(listed["target_over_asset_median"].iloc[0]) < 1e-3


# --------------------------------------------------------------------------
# the calendar
# --------------------------------------------------------------------------


class TestCalendar:
    def _calendar(self, n: int = 40) -> tuple[pd.DatetimeIndex, np.ndarray]:
        index = pd.date_range("2008-08-20", periods=n, freq="B", tz="UTC")
        return index, np.random.default_rng(1).normal(size=n)

    def test_dates_follow_the_target_index_when_the_series_reproduces(self) -> None:
        index, returns = self._calendar()
        grid = _grid(n=30)
        grid["realized_return"] = returns[grid["target_index"].to_numpy()]
        dated = it.attach_calendar(grid, {"AAA": (index, returns)})
        assert (pd.DatetimeIndex(dated["date"]) == index[dated["target_index"].to_numpy()]).all()

    def test_a_one_day_shift_is_refused(self) -> None:
        index, returns = self._calendar()
        grid = _grid(n=30)
        grid["realized_return"] = returns[grid["target_index"].to_numpy() + 1]
        with pytest.raises(ValueError, match="cannot be trusted"):
            it.attach_calendar(grid, {"AAA": (index, returns)})

    def test_regimes_come_from_the_codebase_and_the_wide_window_only_moves_the_gfc(self) -> None:
        dates = pd.DatetimeIndex(
            ["2008-08-21", "2008-09-01", "2009-03-31", "2009-04-01", "2020-03-16", "2024-08-05"],
            tz="UTC",
        )
        headline, wide = it.regime_tags(dates)
        assert headline.tolist() == ["calm", "gfc", "gfc", "calm", "covid", "spike_2024_08"]
        assert wide.tolist() == [
            "gfc_wide",
            "gfc_wide",
            "gfc_wide",
            "gfc_wide",
            "covid",
            "spike_2024_08",
        ]


# --------------------------------------------------------------------------
# runs of exceedances
# --------------------------------------------------------------------------


class TestRuns:
    def test_runs_counted_by_hand(self) -> None:
        hits = np.array([0, 1, 1, 0, 1, np.nan, 1, 1, 1, 0])
        assert it.runs_of_hits(hits) == [2, 1, 3]
        assert it.longest_run(hits) == 3
        assert it.longest_run(np.zeros(5)) == 0
        assert it.runs_of_hits(np.array([1.0, 1.0])) == [2]


# --------------------------------------------------------------------------
# the GARCH recursion, recovered from forecasts
# --------------------------------------------------------------------------


def _garch_cell(
    params: dict[int, tuple[float, float, float, str]], length: int = 21
) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []
    for fit_origin, (omega, alpha, beta, status) in params.items():
        h = 1e-4
        for k in range(length):
            origin = fit_origin + k
            r = math.sqrt(h) * rng.standard_normal()
            rows.append(
                {
                    "origin_index": origin,
                    "fit_origin": fit_origin,
                    "fit_status": status,
                    "forecast_var": h,
                    "realized_return": r,
                }
            )
            h = omega + alpha * r * r + beta * h
    return pd.DataFrame(rows)


class TestGarchRecursion:
    def test_known_parameters_are_recovered_and_the_residual_is_noise(self) -> None:
        cell = _garch_cell({0: (2e-6, 0.08, 0.90, "ok"), 21: (1e-7, 0.05, 0.95, "ok")})
        blocks = it.garch_recursion_by_block(cell)
        assert blocks["fit_origin"].tolist() == [0, 21]
        assert blocks["alpha_plus_beta"].tolist() == pytest.approx([0.98, 1.00], abs=1e-9)
        assert blocks["omega"].tolist() == pytest.approx([2e-6, 1e-7], rel=1e-6)
        assert (blocks["max_rel_residual"] < 1e-9).all()
        assert not blocks["fallback"].any()
        assert blocks["n_equations"].tolist() == [20, 20]

    def test_an_ewma_block_reads_alpha_plus_beta_one_and_omega_zero(self) -> None:
        cell = _garch_cell({0: (0.0, 0.06, 0.94, "fallback=ewma|flag=8")})
        blocks = it.garch_recursion_by_block(cell)
        assert blocks["fallback"].tolist() == [True]
        assert blocks["alpha_plus_beta"].iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert abs(blocks["omega"].iloc[0]) < 1e-15

    def test_a_block_too_short_to_solve_reads_nan(self) -> None:
        cell = _garch_cell({0: (2e-6, 0.08, 0.90, "ok")}, length=3)
        blocks = it.garch_recursion_by_block(cell)
        assert math.isnan(blocks["alpha_plus_beta"].iloc[0])
        assert blocks["n_equations"].iloc[0] == 2


# --------------------------------------------------------------------------
# Sharpe difference interval
# --------------------------------------------------------------------------


class TestSharpeCI:
    def test_identical_series_give_a_zero_interval(self) -> None:
        r = np.random.default_rng(2).normal(0.0002, 0.01, 500)
        out = it.sharpe_difference_ci(r, r.copy(), periods_per_year=252.0, seed=7, n_boot=50)
        assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0
        assert out["block_length"] == 1

    def test_it_is_seeded_and_reports_its_block(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0005, 0.01, 800)
        b = rng.normal(0.0002, 0.01, 800)
        first = it.sharpe_difference_ci(a, b, periods_per_year=252.0, seed=11, n_boot=300)
        second = it.sharpe_difference_ci(a, b, periods_per_year=252.0, seed=11, n_boot=300)
        other = it.sharpe_difference_ci(a, b, periods_per_year=252.0, seed=12, n_boot=300)
        assert first == second
        assert first["ci_low"] != other["ci_low"]
        assert first["ci_low"] < first["ci_high"]
        assert first["n_boot_finite"] == 300 and first["block_length"] >= 1
        with pytest.raises(ValueError, match="same length"):
            it.sharpe_difference_ci(a, b[:-1], periods_per_year=252.0, seed=1, n_boot=10)


# --------------------------------------------------------------------------
# ranks
# --------------------------------------------------------------------------


class TestRanks:
    def test_rank_table_and_kendall_by_hand(self) -> None:
        means = pd.DataFrame({"a": [0.1, 0.3], "b": [0.2, 0.2], "c": [0.3, 0.1]}, index=["X", "Y"])
        ranks = it.rank_table(means)
        assert ranks.loc["X"].tolist() == [1.0, 2.0, 3.0]
        assert ranks.loc["Y"].tolist() == [3.0, 2.0, 1.0]
        assert it.kendall_between(ranks.loc["X"], ranks.loc["X"]) == pytest.approx(1.0)
        assert it.kendall_between(ranks.loc["X"], ranks.loc["Y"]) == pytest.approx(-1.0)
        summary = it.rank_summary(ranks, "all")
        assert summary.loc[summary["model"] == "b", "mean_rank"].iloc[0] == pytest.approx(2.0)
        assert summary.loc[summary["model"] == "a", "rank_max"].iloc[0] == 3.0
        assert summary["assets"].tolist() == [2, 2, 2]


# --------------------------------------------------------------------------
# MCS: twice, identical — and the pieces around it
# --------------------------------------------------------------------------


def _matrix(n: int = 120, m: int = 4, seed: int = 9) -> inference.LossMatrix:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, m)) ** 2
    values[:, 1] += 0.5
    frame = pd.DataFrame(values, columns=[f"m{k}" for k in range(m)])
    frame.index.name = "origin_index"
    return inference.LossMatrix(
        values=frame,
        score="crps",
        asset="AAA",
        horizon=1,
        n_flagged={c: 0 for c in frame.columns},
        config_hashes={c: "0" * 64 for c in frame.columns},
    )


class TestMCSDeterminism:
    def test_one_small_mcs_run_twice_is_identical(self) -> None:
        """The brief's own test: same inputs, same seed, bit-identical output."""
        matrix = _matrix()
        seed = it.seed_for("mcs", "AAA", "crps", "range", "headline")
        first = it.mcs_run(matrix, "range", seed, n_boot=200)
        second = it.mcs_run(matrix, "range", seed, n_boot=200)
        assert first.p_values == second.p_values
        assert first.elimination_order == second.elimination_order
        assert first.step_p_values == second.step_p_values
        assert first.block_length == second.block_length
        assert first.config_hash == second.config_hash
        pd.testing.assert_frame_equal(
            it.mcs_records(first, "AAA", "crps", "headline"),
            it.mcs_records(second, "AAA", "crps", "headline"),
        )

    def test_records_carry_steps_membership_and_the_dominated_model_first(self) -> None:
        matrix = _matrix()
        result = it.mcs_run(matrix, "semi_quadratic", 3, n_boot=200)
        records = it.mcs_records(result, "AAA", "crps", "headline")
        assert set(records["model"]) == set(matrix.models)
        assert records.loc[records["model"] == "m1", "eliminated_at_step"].iloc[0] == 1
        assert sorted(records["eliminated_at_step"]) == [1, 2, 3, 4]
        assert records["in_mcs_0.1"].dtype == bool and records["in_mcs_0.25"].dtype == bool
        assert (records["in_mcs_0.25"] <= records["in_mcs_0.1"]).all()
        with pytest.raises(ValueError, match="unknown statistic"):
            it.mcs_run(matrix, "median", 1, n_boot=10)

    def test_block_diagnostics_are_deterministic_and_flag_what_they_say(self) -> None:
        matrix = _matrix(n=24)
        first, second = it.block_diagnostics(matrix), it.block_diagnostics(matrix)
        assert first == second
        assert first["n"] == 24
        assert first["block_exceeds_n_over_4"] == (first["block_length"] > 6.0)
        assert first["block_is_1"] == (first["block_length"] == 1)
        assert "/" in first["rho1_pairs_max_pair"]

    def test_drop_model_removes_exactly_one_column(self) -> None:
        matrix = _matrix()
        reduced = it.drop_model(matrix, "m2")
        assert reduced.models == ("m0", "m1", "m3")
        assert set(reduced.config_hashes) == {"m0", "m1", "m3"}
        with pytest.raises(KeyError):
            it.drop_model(matrix, "zz")


# --------------------------------------------------------------------------
# DM tables
# --------------------------------------------------------------------------


class TestDM:
    def test_long_form_has_every_pair_at_every_rung_and_checks_j2s_n(self) -> None:
        matrix = _matrix(n=200)
        long = it.dm_long(matrix)
        assert len(long) == 6 * len(it.HAC_LADDER)
        assert set(long["rung"]) == {name for name, _ in it.HAC_LADDER}
        fixed = long.loc[long["rung"] == "fixed"]
        assert not fixed["prewhiten"].any()
        assert (fixed["bandwidth"] == inference.rule_of_thumb_bandwidth(200)).all()
        auto = long.loc[long["rung"] == "auto"].set_index(["model_a", "model_b"])
        twice = long.loc[long["rung"] == "twice_auto"].set_index(["model_a", "model_b"])
        assert np.allclose(twice["bandwidth"], 2.0 * auto["bandwidth"])
        expected = pd.DataFrame(200, index=list(matrix.models), columns=list(matrix.models))
        assert len(it.dm_long(matrix, expected_n=expected)) == 18
        with pytest.raises(ValueError, match="pairwise-complete matrix says"):
            it.dm_long(matrix, expected_n=expected - 1)

    def test_summary_changes_and_markdown(self) -> None:
        long = it.dm_long(_matrix(n=200))
        summary = it.dm_summary(long)
        assert summary["pairs"].tolist() == [6, 6, 6]
        assert (summary["expected_by_chance"] == 6 * it.SIGNIFICANCE).all()
        assert (summary["significant_5pct"] <= 6).all()
        changes = it.dm_significance_changes(long)
        assert changes.loc[0, "changed_fixed_vs_auto"] <= 6
        table = it.dm_matrix_markdown(long, "AAA", "crps", "auto")
        lines = table.splitlines()
        assert len(lines) == 2 + 4
        assert lines[2].count("200") == 3  # n inside every off-diagonal cell of the row
        assert "—" in lines[2]  # the diagonal


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


class TestRendering:
    def test_table_formats_floats_and_booleans(self) -> None:
        frame = pd.DataFrame({"a": [1.23456789, np.nan], "b": [True, False], "c": ["x", "y"]})
        text = it._table(frame)
        assert "1.235" in text and "nan" in text and "yes" in text and "no" in text

    def test_fit_diagnostics_markdown_pivots_by_asset(self) -> None:
        frame = pd.DataFrame(
            {
                "asset": ["A", "B", "A", "B"],
                "model": ["garch11", "garch11", "har", "har"],
                "rate": ["1/10 = 10.00%", "0/10 = 0.00%", "not instrumented", "not instrumented"],
            }
        )
        text = it.fit_diagnostics_markdown(frame)
        assert text.splitlines()[0] == "| model | A | B |"
        assert "| `har` | not instrumented | not instrumented |" in text
