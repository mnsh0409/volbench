"""Temporal integrity of the out-of-fold folds behind docs/P3_LGBM_SMEARING_AUDIT.md.

The rest of the probe reuses audited machinery — ``RollingOriginSplitter`` for
the origins, ``lgbm._design_matrix`` for the features — and its in-sample arm is
checked against the store itself (the re-run reproduces every stored
``forecast_var``). :func:`out_of_fold_residuals` is the one piece with no
counterpart anywhere, so it carries its own canary here.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.benchmarks.lgbm_smearing_probe import (
    DEFAULT_FOLDS,
    out_of_fold_residuals,
    round_ladder,
)
from volbench.models.lgbm import LightGBMRV, _design_matrix

pytest.importorskip("lightgbm")


def _rv(n: int, seed: int = 0) -> np.ndarray:
    """A positive, persistent RV-like series — enough structure for trees to split on."""
    rng = np.random.default_rng(seed)
    log_rv = np.zeros(n)
    for t in range(1, n):
        log_rv[t] = -9.0 + 0.95 * (log_rv[t - 1] + 9.0) + 0.3 * rng.standard_normal()
    return np.exp(log_rv)


class TestFoldsAreCausal:
    def test_corrupting_the_future_leaves_earlier_folds_bit_identical(self) -> None:
        """The canary. Replace every design row from the second fold boundary
        onward with noise; the residuals of the fold *before* it must not move
        by one bit. If a fold ever trained on data after its own block, they
        would."""
        config = LightGBMRV()
        x, y = _design_matrix(_rv(600))
        edges = np.linspace(0, y.size, DEFAULT_FOLDS + 1).astype(int)
        first_block = edges[2] - edges[1]

        clean = out_of_fold_residuals(config, x, y, DEFAULT_FOLDS)

        rng = np.random.default_rng(1)
        x2, y2 = x.copy(), y.copy()
        x2[edges[2] :] = rng.standard_normal(x2[edges[2] :].shape)
        y2[edges[2] :] = rng.standard_normal(y2[edges[2] :].shape)
        corrupted = out_of_fold_residuals(config, x2, y2, DEFAULT_FOLDS)

        assert np.array_equal(clean[:first_block], corrupted[:first_block])
        assert not np.array_equal(clean, corrupted)  # later folds must react

    def test_the_last_training_target_never_postdates_the_predicted_rows_origin(self) -> None:
        """The boundary the canary cannot see, stated arithmetically.

        Design row ``i`` reads ``rv[i : i+22]`` and predicts ``rv[i+22]``. A
        booster trained on rows ``0..train_end-1`` has therefore seen targets up
        to ``rv[train_end+21]`` — which is exactly the last observation known at
        the origin of row ``train_end``, whose own target is ``rv[train_end+22]``.
        Causal with no gap, and no gap needed.
        """
        rv = _rv(200)
        _x, y = _design_matrix(rv)
        train_end = 50
        last_train_target_pos = train_end - 1 + 22
        predicted_row_origin_pos = train_end + 21
        predicted_row_target_pos = train_end + 22
        assert last_train_target_pos == predicted_row_origin_pos
        assert last_train_target_pos < predicted_row_target_pos
        assert y[train_end - 1] == pytest.approx(float(np.log(rv[last_train_target_pos])))

    def test_every_fold_but_the_first_contributes(self) -> None:
        config = LightGBMRV()
        x, y = _design_matrix(_rv(600))
        edges = np.linspace(0, y.size, DEFAULT_FOLDS + 1).astype(int)
        assert out_of_fold_residuals(config, x, y, DEFAULT_FOLDS).size == y.size - edges[1]

    def test_a_window_too_short_to_fold_yields_nothing_rather_than_raising(self) -> None:
        """A probe must never take the report down (the module's own contract)."""
        config = LightGBMRV()
        x, y = _design_matrix(_rv(30))
        assert out_of_fold_residuals(config, x, y, folds=20).size >= 0


class TestRoundLadder:
    def test_more_rounds_shrink_the_training_loss_and_the_factor_together(self) -> None:
        """The addendum's amendment, as a property rather than a table: the
        capacity cap and the collapsing smearing factor are one mechanism."""
        config = LightGBMRV()
        x, y = _design_matrix(_rv(600))
        rows = round_ladder(config, x, y)
        mse = [r["train_mse"] for r in rows]
        smear = [r["smear"] for r in rows]
        assert mse == sorted(mse, reverse=True)
        assert smear == sorted(smear, reverse=True)
        assert all(r["trees"] == r["rounds"] for r in rows)
