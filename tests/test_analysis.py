"""Tests for the analysis layer.

Two kinds live here. ``TestBoundary`` is structural: it asserts that the module
which reads the grid cannot reach the machinery that produced it — the same
belt ``tests/test_econ.py::TestBoundary`` puts on the economic backtest, and
for the same reason. Everything else checks a decidable answer: a closed form
against numerical integration, a parser against the strings the evaluator
actually writes, an off-by-one against a frame built to contain one.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats  # type: ignore[import-untyped]

from volbench import analysis
from volbench.analysis import (
    AlignmentCheck,
    alignment_check,
    alignment_table,
    load_grid,
    load_manifest,
    loss_columns,
    missing_accounting,
    nonfinite_report,
    normal_crps,
    normal_quantile,
    qlike_loss,
    qlike_positivity,
    reason_kind,
    reason_kinds,
    recover_predictive_law,
    student_t_crps,
    student_t_df_from_quantile_ratio,
    student_t_scale_from_variance,
)

# --------------------------------------------------------------------------
# structural boundary
# --------------------------------------------------------------------------


#: What the analysis layer is allowed to import from the package. The store is
#: how a fragment is addressed and read; nothing else here can fit anything.
ALLOWED_VOLBENCH_IMPORTS = {"volbench.results"}

#: Naming any of these would give the analysis layer a way to fit a model, cut
#: a window, or run a cell — which is the thing it must not be able to do.
FORBIDDEN_VOLBENCH_IMPORTS = (
    "volbench.models",
    "volbench.evaluate",
    "volbench.runner",
    "volbench.execute",
    "volbench.splitter",
    "volbench.benchmarks",
    "volbench.compaction",
)


def _volbench_imports(module: Any) -> set[str]:
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return {name for name in imported if name.startswith("volbench")}


class TestBoundary:
    """The analysis layer reads the store; it never fits."""

    def test_it_does_not_import_the_model_package(self) -> None:
        for name in _volbench_imports(analysis):
            for forbidden in FORBIDDEN_VOLBENCH_IMPORTS:
                assert name != forbidden and not name.startswith(forbidden + "."), name

    def test_its_volbench_imports_are_the_declared_ones(self) -> None:
        """An allow-list, not just a deny-list: a new import of, say,
        ``volbench.inference`` is a decision to make deliberately rather than
        one that slides in because it was not on a list of banned names."""
        assert _volbench_imports(analysis) <= ALLOWED_VOLBENCH_IMPORTS

    def test_the_store_reader_it_leans_on_imports_no_model_either(self) -> None:
        """The allow-list is only worth something if what it allows is clean."""
        from volbench import results

        for name in _volbench_imports(results):
            assert not name.startswith("volbench.models"), name

    def test_the_losses_are_written_out_here_not_imported(self) -> None:
        """The recomputation is evidence only if it is independent: a check that
        calls the function under test checks that the function is a function."""
        imported = _volbench_imports(analysis)
        assert "volbench.dist" not in imported
        assert "volbench.metrics" not in imported


# --------------------------------------------------------------------------
# loss closed forms, against numerical integration
# --------------------------------------------------------------------------


def _crps_by_quadrature(cdf: Any, y: float, lo: float, hi: float) -> float:
    """``∫ (F(x) - 1{x >= y})^2 dx`` — the definition, integrated."""
    below = integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=400)[0]
    above = integrate.quad(lambda x: (cdf(x) - 1.0) ** 2, y, hi, limit=400)[0]
    return float(below + above)


class TestClosedForms:
    @pytest.mark.parametrize("y", [-3.0, -0.5, 0.0, 0.25, 2.0])
    def test_normal_crps_matches_its_definition(self, y: float) -> None:
        mu, sigma = 0.1, 0.7
        expected = _crps_by_quadrature(
            lambda x: float(stats.norm.cdf(x, loc=mu, scale=sigma)), y, mu - 40.0, mu + 40.0
        )
        assert normal_crps(mu, sigma, y) == pytest.approx(expected, rel=1e-8, abs=1e-12)

    @pytest.mark.parametrize("df", [2.5, 4.0, 12.0])
    def test_student_t_crps_matches_its_definition(self, df: float) -> None:
        loc, scale, y = 0.0, 0.3, -0.45
        expected = _crps_by_quadrature(
            lambda x: float(stats.t.cdf((x - loc) / scale, df=df)), y, loc - 400.0, loc + 400.0
        )
        assert student_t_crps(loc, scale, df, y) == pytest.approx(expected, rel=1e-6, abs=1e-12)

    def test_the_two_families_agree_as_df_grows(self) -> None:
        """A t at large df is a normal; if the two closed forms did not meet
        there, one of them would be wrong in a way no single case shows."""
        df, variance, y = 5_000.0, 4e-4, -0.01
        scale = student_t_scale_from_variance(variance, df)
        assert student_t_crps(0.0, scale, df, y) == pytest.approx(
            normal_crps(0.0, math.sqrt(variance), y), rel=2e-3
        )

    def test_qlike_is_zero_only_at_a_perfect_forecast(self) -> None:
        assert qlike_loss(1e-4, 1e-4) == pytest.approx(0.0, abs=1e-15)
        assert qlike_loss(1e-4, 4e-4) > 0.0
        assert qlike_loss(4e-4, 1e-4) > 0.0

    def test_qlike_refuses_the_inputs_that_make_it_undefined(self) -> None:
        with pytest.raises(ValueError):
            qlike_loss(0.0, 1e-4)
        with pytest.raises(ValueError):
            qlike_loss(1e-4, 0.0)
        with pytest.raises(ValueError):
            qlike_loss(1e-4, -1e-4)


# --------------------------------------------------------------------------
# recovering the predictive law from stored columns
# --------------------------------------------------------------------------


def _row(**overrides: Any) -> dict[str, Any]:
    variance = overrides.pop("forecast_var", 4e-4)
    base: dict[str, Any] = {
        "asset": "TEST",
        "model": "m",
        "model_label": "m",
        "origin_index": 499,
        "horizon": 1,
        "target_index": 500,
        "forecast_mean": 0.0,
        "forecast_var": variance,
        "realized_return": -0.012,
        "proxy_var": 3e-4,
        "crps": math.nan,
        "qlike": math.nan,
        "missing_reason": "",
        "var_0p01": normal_quantile(0.0, math.sqrt(variance), 0.01),
        "var_0p05": normal_quantile(0.0, math.sqrt(variance), 0.05),
    }
    base.update(overrides)
    return base


class TestPredictiveLawRecovery:
    def test_a_gaussian_row_is_recovered_as_gaussian(self) -> None:
        law = recover_predictive_law(_row(), levels=(0.01, 0.05))
        assert law is not None
        assert law.family == "normal"
        assert law.scale == pytest.approx(math.sqrt(4e-4))

    def test_a_student_t_row_recovers_its_degrees_of_freedom(self) -> None:
        variance, df = 4e-4, 6.5
        scale = student_t_scale_from_variance(variance, df)
        row = _row(
            forecast_var=variance,
            var_0p01=float(scale * stats.t.ppf(0.01, df=df)),
            var_0p05=float(scale * stats.t.ppf(0.05, df=df)),
        )
        law = recover_predictive_law(row, levels=(0.01, 0.05))
        assert law is not None
        assert law.family == "student_t"
        assert law.df == pytest.approx(df, rel=1e-6)
        assert law.scale == pytest.approx(scale, rel=1e-6)

    def test_recovery_is_exact_enough_to_reproduce_the_crps(self) -> None:
        variance, df, y = 9e-4, 3.2, -0.031
        scale = student_t_scale_from_variance(variance, df)
        row = _row(
            forecast_var=variance,
            realized_return=y,
            var_0p01=float(scale * stats.t.ppf(0.01, df=df)),
            var_0p05=float(scale * stats.t.ppf(0.05, df=df)),
        )
        law = recover_predictive_law(row, levels=(0.01, 0.05))
        assert law is not None
        assert law.crps(y) == pytest.approx(student_t_crps(0.0, scale, df, y), rel=1e-8)

    def test_an_unusable_row_recovers_nothing_rather_than_guessing(self) -> None:
        assert recover_predictive_law(_row(forecast_var=math.nan)) is None
        assert recover_predictive_law(_row(var_0p01=math.nan)) is None

    def test_a_law_whose_variance_does_not_match_is_rejected(self) -> None:
        """The falsifiable half: a ratio alone can always be fitted by some
        ``nu``, so the recovered law must also carry the row's own variance."""
        variance, df = 4e-4, 6.5
        scale = student_t_scale_from_variance(variance, df)
        row = _row(
            forecast_var=variance * 4.0,  # the quantiles below belong to `variance`
            var_0p01=float(scale * stats.t.ppf(0.01, df=df)),
            var_0p05=float(scale * stats.t.ppf(0.05, df=df)),
        )
        assert recover_predictive_law(row) is None

    def test_a_nu_estimated_at_its_bound_still_recovers(self) -> None:
        """D-032 bounds the GARCH-t optimizer at nu = 50, so nu *at* the bound
        is a normal outcome; the ratio a stored row then carries misses the
        bound's own ratio by floating-point noise, and a plain sign test
        rejects a root that is really there."""
        df = 50.0
        scale = student_t_scale_from_variance(2.5e-4, df)
        q_low = float(scale * stats.t.ppf(0.01, df=df))
        q_high = float(scale * stats.t.ppf(0.05, df=df))
        nudged = q_low * (1.0 + 4e-15)  # the noise a stored quantile actually carries
        recovered = student_t_df_from_quantile_ratio(nudged, q_high, 0.01, 0.05)
        assert recovered == pytest.approx(50.0)

    def test_the_edge_tolerance_is_noise_wide_and_no_wider(self) -> None:
        """It exists to absorb a last-bit ratio, not to admit a law outside the
        bracket: a ratio a whole percent past the bound must still be None."""
        df = 50.0
        scale = student_t_scale_from_variance(2.5e-4, df)
        q_low = float(scale * stats.t.ppf(0.01, df=df))
        q_high = float(scale * stats.t.ppf(0.05, df=df))
        assert student_t_df_from_quantile_ratio(q_low * 0.99, q_high, 0.01, 0.05) is None

    def test_df_inversion_returns_none_outside_its_bracket(self) -> None:
        """A ratio no ``nu`` in the bound reproduces is not a t at all — which
        is what a ``garch11_t`` origin that fell back to EWMA looks like, and
        it must read as unrecovered rather than as some ``nu``."""
        assert student_t_df_from_quantile_ratio(-10.0, -1.0, 0.01, 0.05) is None
        assert student_t_df_from_quantile_ratio(-1.0, -1.0, 0.01, 0.05) is None

    def test_df_inversion_is_free_of_the_variance(self) -> None:
        """The ratio divides the scale out, which is the whole reason it is the
        invertible quantity: the same ``nu`` comes back at any variance."""
        df = 5.0
        pair = (float(stats.t.ppf(0.01, df=df)), float(stats.t.ppf(0.05, df=df)))
        for variance in (1e-8, 1.0, 1e4):
            scale = student_t_scale_from_variance(variance, df)
            recovered = student_t_df_from_quantile_ratio(
                scale * pair[0], scale * pair[1], 0.01, 0.05
            )
            assert recovered is not None
            assert recovered == pytest.approx(df, rel=1e-8)

    @pytest.mark.parametrize("df", [2.2, 3.0, 7.5, 30.0, 49.0])
    def test_df_inversion_round_trips(self, df: float) -> None:
        variance = 2.5e-4
        scale = student_t_scale_from_variance(variance, df)
        recovered = student_t_df_from_quantile_ratio(
            float(scale * stats.t.ppf(0.01, df=df)),
            float(scale * stats.t.ppf(0.05, df=df)),
            0.01,
            0.05,
        )
        assert recovered is not None
        assert recovered == pytest.approx(df, rel=1e-6)


# --------------------------------------------------------------------------
# missing_reason parsing
# --------------------------------------------------------------------------


class TestReasonParsing:
    def test_a_scored_row_names_no_reason(self) -> None:
        assert reason_kinds("") == ()
        assert reason_kind("") == ""

    def test_score_reasons_split_on_the_pipe_the_evaluator_joins_with(self) -> None:
        assert reason_kinds("proxy_nan") == ("proxy_nan",)
        assert reason_kinds("es_undefined|target_nan") == ("es_undefined", "target_nan")

    def test_an_exception_reason_reduces_to_stage_and_type(self) -> None:
        """The message quotes an origin and a count, so keeping it whole would
        fragment one cause into as many groups as it has origins."""
        reason = (
            "fit_error@499: InsufficientHistoryError: only 499 valid observations "
            "at or before origin 499, need 500"
        )
        assert reason_kinds(reason) == ("fit_error/InsufficientHistoryError",)
        other = "fit_error@520: InsufficientHistoryError: only 498 valid observations"
        assert reason_kinds(other) == reason_kinds(reason)

    def test_a_stageless_exception_reason_still_parses(self) -> None:
        assert reason_kinds("predict_error: ValueError: bad") == ("predict_error/ValueError",)


# --------------------------------------------------------------------------
# tables over a small synthetic grid
# --------------------------------------------------------------------------


def _grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(4):
        rows.append(
            _row(
                asset="A",
                model_label="naive",
                origin_index=499 + i,
                target_index=500 + i,
                crps=0.01 * (i + 1),
                qlike=0.1 * (i + 1),
                log_score=-1.0,
                pinball_0p01=0.001,
                proxy_var=3e-4,
            )
        )
    rows[2]["missing_reason"] = "proxy_nonpositive"
    rows[2]["qlike"] = math.nan
    rows[2]["proxy_var"] = 0.0
    rows[3]["missing_reason"] = "fit_error@502: InsufficientHistoryError: only 499"
    rows[3]["crps"] = math.nan
    rows[3]["qlike"] = math.nan
    rows[3]["forecast_var"] = math.nan
    return pd.DataFrame(rows)


class TestTables:
    def test_loss_columns_exclude_the_forecast_descriptions(self) -> None:
        frame = _grid()
        columns = loss_columns(frame)
        assert "crps" in columns and "qlike" in columns and "pinball_0p01" in columns
        assert not any(c.startswith(("var_", "es_", "hit_")) for c in columns)

    def test_missing_accounting_separates_the_three_scored_counts(self) -> None:
        """A proxy-non-positive row keeps a CRPS and loses only QLIKE, so the
        strict ``scored`` count and the per-metric counts must differ — a table
        reporting one of them as all three would be wrong about 1 row in 4."""
        table = missing_accounting(_grid())
        assert len(table) == 1
        row = table.iloc[0]
        assert row["rows"] == 4
        assert row["scored"] == 2
        assert row["crps_scored"] == 3
        assert row["qlike_scored"] == 2
        assert row["reasons"] == {
            "fit_error/InsufficientHistoryError": 1,
            "proxy_nonpositive": 1,
        }

    def test_nonfinite_report_separates_nan_from_infinity(self) -> None:
        frame = _grid()
        frame.loc[0, "crps"] = math.inf
        table = nonfinite_report(frame, columns=["crps", "qlike"])
        crps = table.loc[table["column"] == "crps"].iloc[0]
        assert crps["nan"] == 1
        assert crps["pos_inf"] == 1
        assert crps["nonfinite_not_nan"] == 1

    def test_qlike_positivity_counts_the_zero_target(self) -> None:
        table = qlike_positivity(_grid())
        row = table.iloc[0]
        assert row["proxy_zero"] == 1
        assert row["proxy_min"] == pytest.approx(3e-4)
        assert row["proxy_min_incl_zero"] == pytest.approx(0.0)
        assert row["forecast_nan"] == 1


# --------------------------------------------------------------------------
# the alignment canary
# --------------------------------------------------------------------------


class TestAlignment:
    def test_a_consistent_row_reproduces_its_own_losses(self) -> None:
        variance, y, proxy = 4e-4, -0.012, 3e-4
        row = _row(forecast_var=variance, realized_return=y, proxy_var=proxy)
        row["crps"] = normal_crps(0.0, math.sqrt(variance), y)
        row["qlike"] = qlike_loss(variance, proxy)
        returns = np.zeros(600)
        returns[500] = y
        proxies = np.full(600, proxy)
        check = alignment_check(row, levels=(0.01, 0.05), returns=returns, proxy=proxies)
        assert check.crps_abs_error < 1e-15
        assert check.qlike_abs_error < 1e-15
        assert check.return_abs_error == 0.0
        assert check.proxy_abs_error == 0.0
        assert check.target_index_is_consistent

    def test_a_one_row_shift_in_the_series_is_caught(self) -> None:
        """The failure this exists for: the losses stay self-consistent when the
        realization comes from the wrong day, so only the series comparison
        can see it."""
        variance, y, proxy = 4e-4, -0.012, 3e-4
        row = _row(forecast_var=variance, realized_return=y, proxy_var=proxy)
        row["crps"] = normal_crps(0.0, math.sqrt(variance), y)
        returns = np.zeros(600)
        returns[501] = y  # the target is at 500; this is the off-by-one
        check = alignment_check(row, levels=(0.01, 0.05), returns=returns)
        assert check.crps_abs_error < 1e-15  # self-consistent...
        assert check.return_abs_error > 0.0  # ...and still misaligned

    def test_no_series_reads_as_not_checked_not_as_a_disagreement(self) -> None:
        """``nan`` (not checked) and ``inf`` (checked and disagreed) must not
        look alike to a caller gating on ``max() < tol``."""
        check = alignment_check(_row(), levels=(0.01, 0.05))
        assert not check.series_checked
        assert math.isnan(check.return_abs_error)
        assert math.isnan(check.proxy_abs_error)

    def test_a_wrong_stored_loss_is_caught(self) -> None:
        row = _row()
        row["crps"] = 0.5  # not what this law and this target give
        check = alignment_check(row, levels=(0.01, 0.05))
        assert check.crps_abs_error > 0.1

    def test_one_sided_nan_never_passes_as_agreement(self) -> None:
        """Two NaNs are the contract agreeing a row is unscorable; one NaN is a
        disagreement, and it must not slip through a ``max() < tol`` gate."""
        row = _row(realized_return=math.nan)
        row["crps"] = 0.01
        check = alignment_check(row, levels=(0.01, 0.05))
        assert math.isinf(check.crps_abs_error)

    def test_alignment_table_renders_the_checks(self) -> None:
        row = _row()
        row["crps"] = normal_crps(0.0, math.sqrt(4e-4), -0.012)
        table = alignment_table([alignment_check(row, levels=(0.01, 0.05))])
        assert list(table["asset"]) == ["TEST"]
        assert bool(table["target_ok"].all())
        assert table["crps_abs_err"].iloc[0] < 1e-15


# --------------------------------------------------------------------------
# store / manifest loading
# --------------------------------------------------------------------------


class TestLoading:
    def test_load_manifest_rejects_a_file_that_is_not_one(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text('{"n_cells": 0}', encoding="utf-8")
        with pytest.raises(ValueError, match="cells"):
            load_manifest(path)

    def test_load_grid_refuses_a_partial_read_by_default(self, tmp_path: Path) -> None:
        """A grid read over 2 of 3 cells that does not say so is how a missing
        column becomes a published number."""
        from volbench.results import ResultsStore

        store = ResultsStore(tmp_path / "store")
        manifest = pd.DataFrame(
            [
                {
                    "asset": "A",
                    "model": "naive",
                    "horizon": 1,
                    "arm": "headline",
                    "lane": "cpu",
                    "status": "computed",
                    "config_hash": "0" * 64,
                }
            ]
        )
        with pytest.raises(ValueError, match="no fragment"):
            load_grid(store, manifest)
        assert load_grid(store, manifest, require_all=False).empty


def test_alignment_check_is_a_frozen_record() -> None:
    """It travels into reports; a mutable one would let a caller edit a result
    after the comparison that produced it."""
    row = _row()
    check = alignment_check(row, levels=(0.01, 0.05))
    assert isinstance(check, AlignmentCheck)
    with pytest.raises(AttributeError):
        check.stored_crps = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# FZ0, HAC standard errors, loss tables, pairwise-complete accounting
# --------------------------------------------------------------------------


class TestFZ0:
    def test_it_matches_the_formula_worked_by_hand(self) -> None:
        """PZC (2019) eq. 6 at a point where every term is nonzero, so an error
        in any one of the four cannot cancel against another."""
        # y <= v: -0.5/(0.05 * -2) + (-1.5)/(-2) + log(2) - 1
        assert analysis.fz0_loss(-2.0, -1.5, -2.0, 0.05) == pytest.approx(
            5.0 + 0.75 + math.log(2.0) - 1.0
        )
        # y > v: the shortfall term drops out entirely.
        assert analysis.fz0_loss(0.0, -1.5, -2.0, 0.05) == pytest.approx(
            0.75 + math.log(2.0) - 1.0
        )

    def test_it_is_minimized_at_the_true_var_and_es(self) -> None:
        """The property the loss exists for (PZC Fig. 2): if this failed, the
        column would not rank ES forecasts and the table would mean nothing."""
        level = 0.05
        draw = np.random.default_rng(20260829).standard_normal(400_000)
        var = float(stats.norm.ppf(level))
        es = float(-stats.norm.pdf(stats.norm.ppf(level)) / level)

        def average(v: float, e: float) -> float:
            return float(np.mean([analysis.fz0_loss(y, v, e, level) for y in draw[:20_000]]))

        truth = average(var, es)
        assert truth < average(var - 0.25, es - 0.25)
        assert truth < average(var + 0.25, es + 0.25)
        assert truth < average(var, es - 0.4)

    def test_the_vectorized_path_cannot_drift_from_the_scalar_one(self) -> None:
        frame = pd.DataFrame(
            {
                "realized_return": [-0.05, 0.01, -0.2, np.nan],
                "var_0p05": [-0.03, -0.03, -0.04, -0.03],
                "es_0p05": [-0.05, -0.05, -0.06, -0.05],
            }
        )
        column = analysis.fz0_column(frame, 0.05)
        for i in range(3):
            assert column[i] == pytest.approx(
                analysis.fz0_loss(
                    float(frame["realized_return"][i]),
                    float(frame["var_0p05"][i]),
                    float(frame["es_0p05"][i]),
                    0.05,
                )
            )
        assert math.isnan(column[3])

    def test_it_agrees_with_the_packages_own_implementation(self) -> None:
        """Two implementations written independently from the same paper. The
        analysis layer is forbidden from importing ``volbench.backtests``, which
        is what makes this an agreement rather than a tautology — and a test may
        import both, which is what makes it checkable."""
        from volbench.backtests import fz0_loss as shipped

        rng = np.random.default_rng(41)
        returns = rng.standard_normal(200) * 0.02
        for level in (0.01, 0.025, 0.05):
            var = float(stats.norm.ppf(level)) * 0.02
            es = float(-stats.norm.pdf(stats.norm.ppf(level)) / level) * 0.02
            theirs = shipped(returns, var, es, level)
            for i, y in enumerate(returns):
                assert analysis.fz0_loss(float(y), var, es, level) == pytest.approx(
                    float(theirs[i]), rel=1e-12, abs=1e-15
                )

    def test_it_refuses_the_sign_convention_bug_rather_than_scoring_it(self) -> None:
        """A positive ES is what a loss-side sign convention looks like, and it
        would score — wrongly, and silently — if the domain were not checked."""
        with pytest.raises(ValueError, match="below zero"):
            analysis.fz0_loss(-2.0, 1.5, 2.0, 0.05)
        with pytest.raises(ValueError, match="ES <= VaR"):
            analysis.fz0_loss(-2.0, -2.0, -1.5, 0.05)
        with pytest.raises(ValueError):
            analysis.fz0_loss(-2.0, -1.5, -2.0, 0.0)


class TestHAC:
    def test_the_bandwidth_rule_is_the_one_the_docstring_states(self) -> None:
        assert analysis.hac_bandwidth(100) == 4
        assert analysis.hac_bandwidth(4904) == 9
        assert analysis.hac_bandwidth(2791) == 8
        assert analysis.hac_bandwidth(1) == 0

    def test_the_long_run_variance_matches_the_formula_worked_by_hand(self) -> None:
        """gamma_0 = 3.5, gamma_1 = -0.75, Bartlett weight 1/2 at L = 1."""
        answer = analysis.hac_mean_se(np.array([1.0, 3.0, 2.0, 6.0]), bandwidth=1)
        assert answer["mean"] == pytest.approx(3.0)
        assert answer["se"] == pytest.approx(math.sqrt(2.75 / 4.0))
        assert answer["se_iid"] == pytest.approx(math.sqrt(3.5 / 3.0))
        assert answer["bandwidth"] == 1

    def test_serial_dependence_inflates_the_standard_error(self) -> None:
        """The whole reason for the column. A positively autocorrelated series
        carries less information per observation than an iid one, and an iid
        standard error would say otherwise."""
        rng = np.random.default_rng(7)
        shocks = rng.standard_normal(4000)
        ar1 = np.zeros(4000)
        for t in range(1, 4000):
            ar1[t] = 0.85 * ar1[t - 1] + shocks[t]
        dependent = analysis.hac_mean_se(ar1)
        independent = analysis.hac_mean_se(shocks)
        assert dependent["se"] > 2.5 * dependent["se_iid"]
        assert independent["se"] == pytest.approx(independent["se_iid"], rel=0.25)

    def test_it_recovers_a_known_long_run_variance(self) -> None:
        """An AR(1)'s long-run variance is ``sigma_e^2 / (1 - rho)^2``, so the
        estimator can be checked against a truth rather than against itself.

        The rho = 0.9 leg is the point: at a fixed rule-of-thumb bandwidth the
        Bartlett kernel recovers only about 60 % of a highly persistent
        series' long-run variance, so every standard error built on this rule
        errs *low*. That is a property of the rule, not a defect, and it is
        pinned here so it stays stated rather than discovered.
        """
        rng = np.random.default_rng(2026)
        recovered = {}
        for rho in (0.0, 0.5, 0.9):
            n = 200_000
            shocks = rng.standard_normal(n)
            series = np.zeros(n)
            for t in range(1, n):
                series[t] = rho * series[t - 1] + shocks[t]
            answer = analysis.hac_mean_se(series)
            long_run = answer["se"] ** 2 * answer["n"]
            recovered[rho] = long_run * (1.0 - rho) ** 2

        assert recovered[0.0] == pytest.approx(1.0, rel=0.05)
        assert recovered[0.5] == pytest.approx(1.0, rel=0.10)
        assert 0.5 < recovered[0.9] < 0.75
        assert recovered[0.0] > recovered[0.5] > recovered[0.9]

    def test_it_is_never_negative_under_the_bartlett_kernel(self) -> None:
        """A truncated (unweighted) estimator can return a negative long-run
        variance and then a NaN standard error. Bartlett's weights cannot."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            series = rng.standard_normal(60)
            series[1::2] *= -1.0  # strong negative autocorrelation
            assert analysis.hac_mean_se(series)["se"] >= 0.0

    def test_holes_are_dropped_and_counted_rather_than_filled(self) -> None:
        with_hole = np.array([1.0, np.nan, 3.0, 2.0, 6.0])
        answer = analysis.hac_mean_se(with_hole, bandwidth=1)
        closed = analysis.hac_mean_se(np.array([1.0, 3.0, 2.0, 6.0]), bandwidth=1)
        assert answer["n"] == 4
        assert answer["n_dropped"] == 1
        assert answer["se"] == pytest.approx(closed["se"])

    def test_a_constant_series_has_no_standard_error_to_report(self) -> None:
        answer = analysis.hac_mean_se(np.full(50, 2.5))
        assert answer["mean"] == pytest.approx(2.5)
        assert answer["se"] == pytest.approx(0.0)


def _loss_grid() -> pd.DataFrame:
    """Two models over five origins of one asset, with one hole in each loss."""
    rows = []
    for model, offset in (("alpha", 0.0), ("beta", 1.0)):
        for origin in range(5):
            rows.append(
                {
                    "asset": "AAA",
                    "model_label": model,
                    "origin_index": origin,
                    "crps": np.nan if (model == "alpha" and origin == 1) else origin + offset,
                    "qlike": np.nan if origin == 4 else origin + offset,
                    "realized_return": -0.01 * (origin + 1),
                    "var_0p05": -0.02,
                    "es_0p05": -0.03,
                    "pinball_0p05": 0.1 * origin,
                }
            )
    return pd.DataFrame(rows)


class TestLossTable:
    def test_the_mean_is_the_mean_and_n_is_what_it_was_taken_over(self) -> None:
        table = analysis.loss_table(_loss_grid(), "AAA", losses=["crps", "qlike"])
        alpha = table[(table["model"] == "alpha") & (table["loss"] == "crps")].iloc[0]
        assert alpha["n"] == 4
        assert alpha["n_dropped"] == 1
        assert alpha["mean"] == pytest.approx(np.mean([0.0, 2.0, 3.0, 4.0]))
        assert alpha["origins"] == 5

    def test_the_answer_does_not_depend_on_the_frames_row_order(self) -> None:
        """A long-run variance read off a shuffled frame is a number about the
        shuffle. ``loss_table`` sorts by origin before the estimator sees it."""
        grid = _loss_grid()
        shuffled = grid.sample(frac=1.0, random_state=1).reset_index(drop=True)
        ordered = analysis.loss_table(grid, "AAA", losses=["crps"])
        assert analysis.loss_table(shuffled, "AAA", losses=["crps"]).equals(ordered)

    def test_it_refuses_an_asset_it_has_no_rows_for(self) -> None:
        with pytest.raises(ValueError, match="no rows for asset"):
            analysis.loss_table(_loss_grid(), "ZZZ")


class TestPairwiseComplete:
    def test_the_intersection_is_where_both_models_score(self) -> None:
        result = analysis.pairwise_complete(_loss_grid(), "AAA", "crps")
        assert result.origins == 5
        assert result.n_used.loc["alpha", "alpha"] == 4
        assert result.n_used.loc["beta", "beta"] == 5
        assert result.n_used.loc["alpha", "beta"] == 4
        assert result.dropped.loc["alpha", "beta"] == 1
        assert result.dropped.loc["beta", "beta"] == 0

    def test_both_matrices_are_symmetric(self) -> None:
        result = analysis.pairwise_complete(_loss_grid(), "AAA", "qlike")
        assert result.n_used.equals(result.n_used.T)
        assert result.dropped.equals(result.dropped.T)

    def test_it_aligns_on_the_origin_and_not_on_row_position(self) -> None:
        """Two models whose rows arrive in different orders are still compared
        day against day. Aligning positionally would silently pair 2020-01-02's
        score with 2020-03-11's and report a full sample."""
        grid = _loss_grid()
        flipped = pd.concat(
            [
                grid[grid["model_label"] == "alpha"],
                grid[grid["model_label"] == "beta"].iloc[::-1],
            ],
            ignore_index=True,
        )
        assert analysis.pairwise_complete(flipped, "AAA", "crps").n_used.equals(
            analysis.pairwise_complete(grid, "AAA", "crps").n_used
        )

    def test_the_largest_drop_names_the_pair_it_belongs_to(self) -> None:
        answer = analysis.pairwise_complete(_loss_grid(), "AAA", "crps").largest_drop()
        assert answer["dropped"] == 1
        assert "alpha" in (answer["model_a"], answer["model_b"])

    def test_the_long_form_carries_every_ordered_pair(self) -> None:
        long = analysis.pairwise_complete_long(_loss_grid(), losses=["crps", "qlike"])
        assert len(long) == 2 * 2 * 2
        assert set(long["loss"]) == {"crps", "qlike"}
        assert (long["dropped"] == long["origins"] - long["n_used"]).all()


class TestMissingnessPatterns:
    def test_losses_nan_on_the_same_rows_are_one_pattern(self) -> None:
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [4.0, np.nan, 6.0], "c": [7.0, 8.0, np.nan]}
        )
        patterns = analysis.missingness_patterns(frame, ["a", "b", "c"])
        assert len(patterns) == 2
        assert patterns.loc[0, "losses"] == "a, b"
        assert patterns.loc[0, "n_finite"] == 2
        assert patterns.loc[1, "losses"] == "c"

    def test_a_single_differing_row_splits_a_pattern(self) -> None:
        """The check is element-wise, not a count: two losses with the same
        number of NaNs in different places are not the same sample."""
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        assert len(analysis.missingness_patterns(frame, ["a", "b"])) == 2
