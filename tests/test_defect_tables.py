"""The committed aggregation behind K's tables, and the fixes' acceptance test.

The tables themselves need the primary store and the two probe parquets, which
live under gitignored ``data/`` and are not in CI. What is decidable without
them — and what actually has to be right — is the arithmetic: that the
acceptance verdict *can* fail, that the realized factor inverts the stored
variance the way the audit says it does, and that the closure table reduces the
right columns.

Every fixture below is synthetic and built so the expected answer is known in
closed form.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volbench.benchmarks.defect_tables import (
    PANEL_TOLERANCE,
    lgbm_acceptance,
    lgbm_factor_table,
    realized_lgbm_factors,
    tsfm_closure_table,
)
from volbench.results import ResultsStore


def _acceptance_table(ratios: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": list(ratios),
            "smear_shipped": [1.7 * r for r in ratios.values()],
            "smear_realized": [1.7] * len(ratios),
            "shipped_over_realized": list(ratios.values()),
        }
    )


class TestAcceptanceCanFail:
    def test_the_audits_own_out_of_fold_numbers_pass(self) -> None:
        """docs/P3_LGBM_SMEARING_AUDIT.md §2, out-of-fold over realized, per
        asset. If the fix works this is roughly what it must produce."""
        table = _acceptance_table(
            {
                "BTC-USD": 0.968, "CAC": 1.024, "DAX": 1.010, "DIA": 1.071,
                "ETH-USD": 1.046, "HSI": 1.064, "KOSPI": 1.002, "NDX": 1.013,
                "NKX": 1.029, "SPY": 1.082, "TWSE": 1.033,
            }
        )
        verdict = lgbm_acceptance(table)
        assert verdict.passed and verdict.panel_ok and verdict.per_asset_ok
        assert verdict.panel_median == pytest.approx(1.029)
        assert verdict.worst_asset == "SPY"

    def test_the_defect_this_replaced_fails_it(self) -> None:
        """The same table with the *in-sample* ratios the audit measured — the
        state of the store before the fix. An acceptance test that passed on
        these would be testing nothing."""
        table = _acceptance_table(
            {
                "BTC-USD": 1 / 1.070, "CAC": 1 / 1.232, "DAX": 1 / 1.275,
                "DIA": 1 / 1.261, "ETH-USD": 1 / 1.066, "HSI": 1 / 1.178,
                "KOSPI": 1 / 1.218, "NDX": 1 / 1.227, "NKX": 1 / 1.153,
                "SPY": 1 / 1.279, "TWSE": 1 / 1.204,
            }
        )
        verdict = lgbm_acceptance(table)
        assert not verdict.passed
        assert not verdict.panel_ok
        assert "FAIL" in str(verdict)

    def test_one_wild_asset_fails_it_even_with_a_clean_panel_median(self) -> None:
        ratios = dict.fromkeys("ABCDEFGHIJ", 1.0)
        ratios["K"] = 1.8
        verdict = lgbm_acceptance(_acceptance_table(ratios))
        assert verdict.panel_ok and not verdict.per_asset_ok and not verdict.passed
        assert verdict.worst_asset == "K"

    def test_an_empty_table_is_not_a_pass(self) -> None:
        verdict = lgbm_acceptance(_acceptance_table({}))
        assert not verdict.passed and verdict.n_assets == 0

    def test_the_tolerances_are_stated_and_carried(self) -> None:
        verdict = lgbm_acceptance(_acceptance_table({"A": 1.0}), panel=0.01, per_asset=0.02)
        assert verdict.panel_tolerance == 0.01 and verdict.per_asset_tolerance == 0.02
        assert PANEL_TOLERANCE == 0.05


class TestRealizedFactorInvertsTheStoredVariance:
    """``mu_hat = log(forecast_var) - log(smear)`` is exact within a refit
    block because ``update`` re-conditions at fixed parameters. Built here with
    a known ``mu_hat`` and a known residual set, so the recovered factor is
    known in closed form."""

    def _store(self, tmp_path: Path) -> tuple[ResultsStore, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        n = 400
        mu_hat = np.log(1e-4) + 0.1 * rng.standard_normal(n)
        smear = 1.5
        resid = 0.4 * rng.standard_normal(n)
        frame = pd.DataFrame(
            {
                "asset": "TOY",
                "origin_index": np.arange(n) + 499,
                "target_index": np.arange(n) + 500,
                "fit_origin": 499,
                "refit": [True] + [False] * (n - 1),
                "forecast_var": np.exp(mu_hat) * smear,
                "proxy_var": np.exp(mu_hat + resid),
                "horizon": 1,
                "forecast_mean": 0.0,
                "realized_return": 0.0,
                "proxy_name": "toy",
            }
        )
        store = ResultsStore(tmp_path / "store")
        digest = "a" * 64
        store.write(frame.assign(config_hash=digest), config={"toy": True})
        manifest = pd.DataFrame([{"asset": "TOY", "model": "lgbm", "config_hash": digest}])
        probe = pd.DataFrame(
            [{"asset": "TOY", "origin": 499, "smear_shipped": smear, "probe_error": None}]
        )
        return store, manifest, probe

    def test_it_recovers_duans_factor_over_the_realized_errors(self, tmp_path: Path) -> None:
        store, manifest, probe = self._store(tmp_path)
        out = realized_lgbm_factors(store, manifest, probe)
        frame = store.read("a" * 64)
        expected_resid = np.log(frame["proxy_var"]) - (
            np.log(frame["forecast_var"]) - math.log(1.5)
        )
        assert out.loc[0, "n_scored"] == len(frame)
        assert out.loc[0, "smear_realized"] == pytest.approx(
            float(np.mean(np.exp(expected_resid))), rel=1e-12
        )

    def test_a_wrong_shipped_factor_moves_the_realized_one(self, tmp_path: Path) -> None:
        """The inversion has to actually use the factor, or the comparison it
        feeds would be self-fulfilling."""
        store, manifest, probe = self._store(tmp_path)
        doubled = realized_lgbm_factors(store, manifest, probe.assign(smear_shipped=3.0))
        base = realized_lgbm_factors(store, manifest, probe)
        assert doubled.loc[0, "smear_realized"] == pytest.approx(
            base.loc[0, "smear_realized"] * 2.0, rel=1e-12
        )

    def test_the_table_joins_the_probe_to_the_store(self, tmp_path: Path) -> None:
        store, manifest, probe = self._store(tmp_path)
        probe = probe.assign(
            resid_var_in_sample=0.2,
            resid_var_out_of_fold=0.4,
            smear_in_sample=1.2,
            smear_out_of_fold=1.5,
        )
        table = lgbm_factor_table(probe, store, manifest)
        assert list(table["asset"]) == ["TOY"]
        assert table.loc[0, "shipped_over_realized"] == pytest.approx(
            1.5 / table.loc[0, "smear_realized"], rel=1e-12
        )
        assert table.loc[0, "in_sample_over_realized"] < table.loc[0, "shipped_over_realized"]


class TestClosureTable:
    def _probe(self) -> pd.DataFrame:
        rows = []
        for model in ("chronos", "moirai"):
            for asset in ("A", "B"):
                for i in range(5):
                    flat = 1e-4 * (1 + 0.01 * i)
                    rows.append(
                        {
                            "asset": asset,
                            "model": model,
                            "origin": 500 + i,
                            "target_index": 501 + i,
                            "probe_error": None,
                            "flat_tail_mean": flat,
                            "vhat": flat * 1.15,
                            "tail_closure": "lognormal" if i else "flat",
                            "mean_lognormal_tail": flat * 1.15,
                            "mean_loglinear_tail": flat * 1.09,
                            "q_0.1": flat * 0.5,
                            "q_0.9": flat * 1.8,
                        }
                    )
        return pd.DataFrame(rows)

    def test_it_reports_a_row_per_cell_and_a_panel_row_per_config(self) -> None:
        table = tsfm_closure_table(self._probe())
        assert list(table["model"]) == ["chronos"] * 3 + ["moirai"] * 3
        assert list(table["asset"]) == ["A", "B", "panel"] * 2

    def test_the_ratios_are_to_the_flat_reading_and_the_fallbacks_are_counted(self) -> None:
        table = tsfm_closure_table(self._probe()).set_index(["model", "asset"])
        assert table.loc[("chronos", "panel"), "lognormal"] == pytest.approx(1.15)
        assert table.loc[("chronos", "panel"), "loglinear"] == pytest.approx(1.09)
        assert table.loc[("chronos", "panel"), "scored_over_flat"] == pytest.approx(1.15)
        # one origin per cell kept its flat tails, and that is reported rather
        # than silently dropped
        assert table.loc[("chronos", "panel"), "n_origins"] == 10
        assert table.loc[("chronos", "panel"), "n_flat_fallback"] == 2

    def test_the_empirical_closure_is_reported_only_when_realizations_are_given(
        self, tmp_path: Path
    ) -> None:
        assert "empirical" not in tsfm_closure_table(self._probe()).columns
        probe = self._probe()
        store = ResultsStore(tmp_path / "store")
        manifest_rows = []
        for model in ("chronos", "moirai"):
            for asset in ("A", "B"):
                digest = f"{abs(hash((model, asset))):064x}"[:64]
                block = probe[(probe["model"] == model) & (probe["asset"] == asset)]
                # realizations deliberately in both tails, so both ratios exist
                realized = np.where(
                    np.arange(len(block)) % 2 == 0, block["q_0.1"] * 0.5, block["q_0.9"] * 3.0
                )
                store.write(
                    pd.DataFrame(
                        {
                            "config_hash": digest,
                            "asset": asset,
                            "origin_index": block["origin"].to_numpy(),
                            "horizon": 1,
                            "target_index": block["target_index"].to_numpy(),
                            "forecast_mean": 0.0,
                            "forecast_var": block["vhat"].to_numpy(),
                            "realized_return": 0.0,
                            "proxy_name": "toy",
                            "proxy_var": realized,
                        }
                    ),
                    config={"toy": True},
                )
                manifest_rows.append({"asset": asset, "model": model, "config_hash": digest})
        table = tsfm_closure_table(probe, store, pd.DataFrame(manifest_rows))
        assert "empirical" in table.columns
        # a fat upper tail must read above the flat closure it replaces
        assert (table["empirical"] > 1.0).all()
