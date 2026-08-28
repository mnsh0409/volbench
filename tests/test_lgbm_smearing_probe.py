"""The probe behind docs/P3_LGBM_SMEARING_AUDIT.md, after its fix landed.

The out-of-fold construction this probe introduced now ships inside
``models/lgbm.py``, and its temporal-integrity canary went with it to
``tests/test_models_lgbm.py::TestOutOfFoldFoldsAreCausal`` — a probe being
leakage-clean was never evidence that the adapter is. What is left here is
what is still the probe's own: that it measures *the adapter's* construction
rather than a restatement of it, and the boosting-round ladder, which no
shipped code path uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.benchmarks import lgbm_smearing_probe
from volbench.benchmarks.lgbm_smearing_probe import (
    DEFAULT_FOLDS,
    out_of_fold_residuals,
    round_ladder,
)
from volbench.models import lgbm
from volbench.models.lgbm import DEFAULT_OOF_FOLDS, LightGBMRV, _design_matrix

pytest.importorskip("lightgbm")


def _rv(n: int, seed: int = 0) -> np.ndarray:
    """A positive, persistent RV-like series — enough structure for trees to split on."""
    rng = np.random.default_rng(seed)
    log_rv = np.zeros(n)
    for t in range(1, n):
        log_rv[t] = -9.0 + 0.95 * (log_rv[t - 1] + 9.0) + 0.3 * rng.standard_normal()
    return np.exp(log_rv)


class TestItMeasuresTheShippedConstruction:
    def test_the_probes_out_of_fold_arm_is_the_adapters_own_function(self) -> None:
        """Not a copy of it. A second implementation could drift from the one
        that ships, and then the audit would be measuring the probe."""
        assert out_of_fold_residuals is lgbm.out_of_fold_residuals
        assert lgbm_smearing_probe.DEFAULT_FOLDS == DEFAULT_OOF_FOLDS

    def test_the_shipped_factor_is_the_out_of_fold_column_under_the_default(self) -> None:
        """``smear_shipped`` and ``smear_out_of_fold`` are the same number
        under the default arm and different numbers under the other one, which
        is what makes recording both worth the column."""
        config = LightGBMRV()
        rv = _rv(600)
        x, y = _design_matrix(rv)
        oof = out_of_fold_residuals(config, x, y, DEFAULT_FOLDS)
        assert config.fit(rv).smear == pytest.approx(
            float(np.mean(np.exp(oof))), rel=1e-15
        )
        in_sample = LightGBMRV(smearing_residuals="in_sample").fit(rv)
        assert in_sample.smear != config.fit(rv).smear


class TestRoundLadder:
    def test_more_rounds_shrink_the_training_loss_and_the_factor_together(self) -> None:
        """The addendum's amendment, as a property rather than a table: the
        capacity cap and the collapsing in-sample smearing factor are one
        mechanism. Recorded as a known limitation of the ``in_sample`` arm —
        the cap was deliberately not raised when the default moved off it."""
        config = LightGBMRV()
        x, y = _design_matrix(_rv(600))
        rows = round_ladder(config, x, y)
        mse = [r["train_mse"] for r in rows]
        smear = [r["smear"] for r in rows]
        assert mse == sorted(mse, reverse=True)
        assert smear == sorted(smear, reverse=True)
        assert all(r["trees"] == r["rounds"] for r in rows)
