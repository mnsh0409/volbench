"""No report may carry an order statistic of a licensed-derived series.

An **aggregate** — a mean, a variance, a kurtosis, a count, a fitted parameter,
a ratio of two of these — describes a sample without returning any member of
it, and publishing one redistributes nothing. An **order statistic** returns
one realised observation: a max, a min, a quoted return or target value. A
sequence of them over overlapping windows discloses actual return magnitudes
and brackets their dates, which is why
``docs/P3_CONVERGENCE_FITS.parquet`` was withheld (docs/P3_REPO_HYGIENE.md §3)
— and then why three committed tables had to be rewritten, because the same
values had already reached them in markdown under a different label
(docs/P3_ORDER_STATISTICS.md).

The rule lived in a document. Now it lives in ``COLUMN_POLICY``, and every path
that emits goes through the same gate, so one declaration protects the report
tables and a future committed parquet alike.

These tests need no ``data/``: the policy, the gate and the builders are all
exercised on synthetic frames, and the point of the last class is that the gate
is proved able to **fail** rather than trusted because a clean tree passes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volbench.benchmarks import convergence_forensics as cf
from volbench.benchmarks.convergence_forensics import (
    COLUMN_POLICY,
    OrderStatisticError,
    gamma_persistence_table,
    paired_windows,
    publishable,
    refuse_unpublishable,
    unpublishable_columns,
    window_stats,
    with_publishable_ratios,
    write_fits,
)

#: The one column the study computes that returns a realised observation.
THE_ORDER_STATISTIC = "max_abs_return"


def _cell(asset: str, config: str, maxima: list[float], fallback: list[bool]) -> pd.DataFrame:
    """A minimal fits frame: enough columns for the builders, no more."""
    n = len(maxima)
    return pd.DataFrame(
        {
            "asset": [asset] * n,
            "config": [config] * n,
            "fit_origin": list(range(500, 500 + 21 * n, 21)),
            "date": pd.date_range("2020-01-01", periods=n, freq="21D", tz="UTC"),
            "fallback": fallback,
            "max_abs_return": maxima,
            "kurtosis": [5.0 + i for i in range(n)],
            "std": [0.01] * n,
            "gamma[1]": [0.01] * n,
            "alpha[1]": [0.05] * n,
            "beta[1]": [0.9] * n,
            "alpha_plus_beta": [0.95] * n,
            "persistence": [0.95] * n,
            "omega": [1e-6] * n,
            "nu": [4.0] * n,
        }
    )


class TestThePolicyIsComplete:
    def test_every_window_statistic_is_tagged(self) -> None:
        """``window_stats`` is where the fit window becomes numbers, so an
        untagged key there is a column nobody classified."""
        import numpy as np

        for key in window_stats(np.array([0.01, -0.02, 0.005, -0.001])):
            assert key in COLUMN_POLICY, f"{key} is not in COLUMN_POLICY"

    def test_the_window_maximum_is_the_order_statistic(self) -> None:
        assert COLUMN_POLICY[THE_ORDER_STATISTIC] == "order_statistic"

    def test_the_moments_and_counts_are_aggregates(self) -> None:
        for key in ("std", "kurtosis", "skew", "n", "n_zero_returns"):
            assert COLUMN_POLICY[key] == "aggregate"

    def test_a_ratio_of_two_order_statistics_is_an_aggregate(self) -> None:
        """It discloses neither operand, and every argument the raw column
        supported was comparative."""
        for key in ("max_abs_over_clean_median", "max_abs_ratio"):
            assert COLUMN_POLICY[key] == "aggregate"

    def test_the_committed_artifact_is_covered_if_it_is_here(self) -> None:
        """When a local run has left the fits table on disk, every one of its
        columns must be classified — the file is what a future commit would
        publish."""
        path = Path(__file__).parents[1] / "docs" / "P3_CONVERGENCE_FITS.parquet"
        if not path.is_file():
            pytest.skip("no local fits table: it is gitignored and absent in CI")
        columns = pd.read_parquet(path).columns
        untagged = [c for c in columns if cf._policy_kind(str(c)) is None]
        assert untagged == [], f"columns nobody classified: {untagged}"


class TestTheGateRefuses:
    def test_it_names_the_order_statistic(self) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2], [False, True])
        assert unpublishable_columns(frame.columns) == [THE_ORDER_STATISTIC]

    def test_refuse_raises_rather_than_dropping_silently(self) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2], [False, True])
        with pytest.raises(OrderStatisticError, match=THE_ORDER_STATISTIC):
            refuse_unpublishable(frame, "a report")

    def test_an_untagged_column_is_refused_too(self) -> None:
        """Fail closed. A column added later must not reach a report by nobody
        having thought about it — which is exactly how this one got there."""
        frame = _cell("HSI", "gjr", [0.1], [False]).drop(columns=[THE_ORDER_STATISTIC])
        frame["some_new_diagnostic"] = [1.0]
        assert unpublishable_columns(frame.columns) == ["some_new_diagnostic"]
        with pytest.raises(OrderStatisticError, match="some_new_diagnostic"):
            refuse_unpublishable(frame, "a report")

    def test_the_prefixed_form_is_resolved(self) -> None:
        """``paired_windows`` prefixes with the asset, so the policy has to see
        through ``BTC-USD_max_abs_return`` to the column it is."""
        assert cf._policy_kind("BTC-USD_max_abs_return") == "order_statistic"
        assert cf._policy_kind("BTC-USD_kurtosis") == "aggregate"

    def test_publishable_drops_it_and_keeps_the_rest(self) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2], [False, True])
        kept = publishable(frame)
        assert THE_ORDER_STATISTIC not in kept.columns
        assert {"kurtosis", "std", "alpha_plus_beta"} <= set(kept.columns)
        assert len(kept) == len(frame)


class TestTheWriterRefuses:
    """The inert-proof. A gate over a clean frame passes whether or not it
    works, so a column is *tagged* an order statistic and the writer is
    required to refuse it — and to leave no file behind when it does."""

    def test_it_refuses_the_real_order_statistic(self, tmp_path: Path) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2], [False, True])
        destination = tmp_path / "fits.parquet"

        with pytest.raises(OrderStatisticError, match=THE_ORDER_STATISTIC):
            write_fits(frame, destination)
        assert not destination.exists(), "refused, but a file was written anyway"

    def test_it_refuses_a_column_tagged_at_the_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tag an ordinary column ``order_statistic`` and the writer must stop
        on it. Nothing about the refusal is special to ``max_abs_return``: it
        is the tag that is load-bearing, which is what makes the policy a
        policy rather than one hard-coded column name."""
        # Built and cleared *before* the tag moves, so the frame reaching the
        # writer is one the writer would otherwise have accepted.
        frame = publishable(_cell("HSI", "gjr", [0.1], [False]))
        destination = tmp_path / "fits.parquet"
        assert "kurtosis" in frame.columns
        assert cf.write_fits(frame, tmp_path / "before.parquet").exists()

        monkeypatch.setattr(cf, "COLUMN_POLICY", {**COLUMN_POLICY, "kurtosis": "order_statistic"})

        with pytest.raises(OrderStatisticError, match="kurtosis"):
            cf.write_fits(frame, destination)
        assert not destination.exists(), "refused, but a file was written anyway"

    def test_it_writes_a_clean_frame(self, tmp_path: Path) -> None:
        """The other half: a gate that refuses everything gets switched off."""
        frame = publishable(_cell("HSI", "gjr", [0.1, 0.2], [False, True]))
        destination = tmp_path / "fits.parquet"

        assert write_fits(frame, destination) == destination
        assert pd.read_parquet(destination).shape == frame.shape


class TestTheBuildersEmitRatios:
    def test_the_ratio_is_the_maximum_over_the_clean_median(self) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2, 0.3, 0.9], [False, False, False, True])
        ratios = with_publishable_ratios(frame)["max_abs_over_clean_median"].tolist()
        assert ratios == pytest.approx([0.5, 1.0, 1.5, 4.5])  # clean median 0.2

    def test_gamma_persistence_table_carries_no_order_statistic(self) -> None:
        frame = _cell("HSI", "gjr", [0.1, 0.2, 0.3, 0.9], [False, False, False, True])
        table = gamma_persistence_table(frame, "HSI", "gjr")
        described = {str(i) for i in table.index.get_level_values(0)}
        assert THE_ORDER_STATISTIC not in described
        assert "max_abs_over_clean_median" in described

    def test_paired_windows_emits_the_ratio_and_neither_operand(self) -> None:
        left = _cell("BTC-USD", "garch11_t", [0.10, 0.20], [True, False])
        right = _cell("ETH-USD", "garch11_t", [0.15, 0.50], [False, False])
        paired = paired_windows(
            pd.concat([left, right], ignore_index=True),
            "garch11_t",
            "BTC-USD",
            "ETH-USD",
            [500],
        )

        assert unpublishable_columns(paired.columns) == []
        assert not [c for c in paired.columns if c.endswith(THE_ORDER_STATISTIC)]
        assert paired.loc[0, "max_abs_ratio"] == pytest.approx(1.5)  # 0.15 / 0.10
