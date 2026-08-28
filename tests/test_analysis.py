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
