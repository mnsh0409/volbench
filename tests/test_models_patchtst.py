"""PatchTST baseline.

- ``TestSpec``: the torch-free surface (runs everywhere).
- ``TestWindows`` / ``TestCpuSmoke`` / ``TestThroughTheEvaluator``: need torch
  (``importorskip``); the CI legs install the CPU build. A 2-epoch fit of a
  tiny net on 50 points is the smoke test; the leakage canary runs the
  frozen-between-refits protocol through ``run_backtest``.
- ``TestGpu`` (``@pytest.mark.gpu``): the default architecture and budget on
  a CUDA device — bit-identity twice is the gate.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from volbench.dist import Normal
from volbench.evaluate import SupportsUpdate, run_backtest
from volbench.models import FittedPatchTST, PatchTST
from volbench.models.base import FittedModel, ForecastModel
from volbench.splitter import RollingOriginSplitter

SMOKE = dict(
    lookback=16,
    patch_len=8,
    stride=4,
    d_model=8,
    n_heads=2,
    n_layers=1,
    d_ff=16,
    max_epochs=2,
    patience=2,
    batch_size=16,
    device="cpu",
)


def realized_variance(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(1e-4), 0.4, size=n))


class TestSpec:
    def test_name_and_hashed_hyperparameters(self) -> None:
        model = PatchTST()
        assert model.name == "patchtst"
        spec = model.spec()
        json.dumps(spec, sort_keys=True)
        assert spec == PatchTST().spec()
        for key in (
            "lookback",
            "patch_len",
            "stride",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "max_horizon",
            "max_epochs",
            "patience",
            "val_fraction",
            "batch_size",
            "lr",
            "weight_decay",
            "seed",
            "retransform",
            "early_stopping",
            "torch",
        ):
            assert key in spec, key
        assert spec["retransform"] == "smearing"
        assert "device" not in spec
        assert PatchTST(device="cpu").spec() == PatchTST(device="cuda").spec()
        assert PatchTST(seed=1).spec() != spec
        assert PatchTST(max_epochs=5).spec() != spec
        assert PatchTST(patience=3).spec() != spec

    def test_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="patch_len"):
            PatchTST(lookback=8, patch_len=16)
        with pytest.raises(ValueError, match="divisible"):
            PatchTST(d_model=30, n_heads=4)
        with pytest.raises(ValueError, match="dropout"):
            PatchTST(dropout=1.0)
        with pytest.raises(ValueError, match="val_fraction"):
            PatchTST(val_fraction=0.0)
        with pytest.raises(ValueError, match="max_epochs"):
            PatchTST(max_epochs=0)
        with pytest.raises(ValueError, match="lr"):
            PatchTST(lr=0.0)

    def test_min_train_and_input_contract_are_checked_before_torch(self) -> None:
        model = PatchTST(**SMOKE)
        assert model.min_train == 16 + 1 + 2
        with pytest.raises(ValueError, match="at least"):
            model.fit(realized_variance(model.min_train - 1))
        bad = realized_variance(50)
        bad[7] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(bad)
        bad[7] = np.nan
        with pytest.raises(ValueError, match="finite"):
            model.fit(bad)

    def test_no_update_by_design(self) -> None:
        assert not hasattr(FittedPatchTST, "update")
        assert isinstance(PatchTST(), ForecastModel)


class TestWindows:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_pairs_are_cut_from_the_array_only_and_end_at_its_last_point(self) -> None:
        from volbench.models._patchtst_net import _windows

        y = np.arange(12, dtype=np.float64)
        x, t = _windows(y, lookback=4, horizon=2)
        assert x.shape == (7, 4) and t.shape == (7, 2)
        assert x[0].tolist() == [0, 1, 2, 3] and t[0].tolist() == [4, 5]
        assert x[-1].tolist() == [6, 7, 8, 9] and t[-1].tolist() == [10, 11]
        assert float(t.max()) == y[-1]  # the last target is the origin itself
        for i in range(7):
            assert x[i].tolist() == y[i : i + 4].tolist()
            assert t[i].tolist() == y[i + 4 : i + 6].tolist()


class TestCpuSmoke:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_two_epochs_on_fifty_points(self) -> None:
        rv = realized_variance(50)
        fitted = PatchTST(**SMOKE).fit(rv)
        assert isinstance(fitted, FittedModel)
        assert not isinstance(fitted, SupportsUpdate)
        spec = fitted.spec()
        json.dumps(spec, sort_keys=True)
        assert spec["epochs_run"] == 2 and 1 <= spec["best_epoch"] <= 2
        assert spec["n_train_windows"] + spec["n_val_windows"] == 50 - 16 - 1 + 1
        assert spec["n_val_windows"] == math.ceil(0.2 * 34)
        assert spec["stopped_early"] is False
        assert len(spec["smearing_factor"]) == 1 and spec["smearing_factor"][0] > 0.0
        dist = fitted.predict(1)
        assert isinstance(dist, Normal) and dist.mu == 0.0
        level = float(np.mean(rv[-22:]))
        assert 0.1 * level < dist.variance() < 10.0 * level
        assert dist.variance() == pytest.approx(
            math.exp(spec["log_forecast"][0]) * spec["smearing_factor"][0], rel=1e-12
        )
        with pytest.raises(ValueError, match="h must be >= 1"):
            fitted.predict(0)
        with pytest.raises(ValueError, match="max_horizon"):
            fitted.predict(2)

    def test_bit_identical_twice_and_seed_sensitive(self) -> None:
        rv = realized_variance(50)
        a, b = PatchTST(**SMOKE).fit(rv), PatchTST(**SMOKE).fit(rv)
        assert np.array_equal(a.log_forecast, b.log_forecast)
        assert np.array_equal(a.smearing, b.smearing)
        assert a.predict(1) == b.predict(1)
        assert a.spec() == b.spec()
        c = PatchTST(**{**SMOKE, "seed": 1}).fit(rv)
        assert not np.array_equal(a.log_forecast, c.log_forecast)

    def test_multi_horizon_head_and_per_horizon_smearing(self) -> None:
        rv = realized_variance(60)
        fitted = PatchTST(**{**SMOKE, "max_horizon": 3}).fit(rv)
        assert fitted.log_forecast.shape == (3,) and fitted.smearing.shape == (3,)
        variances = [fitted.predict(h).variance() for h in (1, 2, 3)]
        assert all(v > 0.0 for v in variances)
        with pytest.raises(ValueError, match="max_horizon"):
            fitted.predict(4)

    def test_early_stopping_is_bounded_and_recorded(self) -> None:
        rv = realized_variance(80)
        fitted = PatchTST(**{**SMOKE, "max_epochs": 40, "patience": 1}).fit(rv)
        spec = fitted.spec()
        assert spec["epochs_run"] <= 40
        assert spec["best_epoch"] <= spec["epochs_run"]
        assert spec["stopped_early"] == (spec["epochs_run"] < 40)
        assert spec["best_val_mse"] >= 0.0

    def test_deterministic_mode_is_restored_after_fit(self) -> None:
        import torch

        before = torch.are_deterministic_algorithms_enabled()
        PatchTST(**SMOKE).fit(realized_variance(50))
        assert torch.are_deterministic_algorithms_enabled() == before

    def test_fitted_object_is_torch_free_and_picklable(self) -> None:
        import pickle

        fitted = PatchTST(**SMOKE).fit(realized_variance(50))
        again = pickle.loads(pickle.dumps(fitted))
        assert again.predict(1) == fitted.predict(1)


def _panel(n: int, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(np.log(1e-4), 0.4, size=n))
    r = np.sqrt(rv) * rng.standard_normal(n)
    return pd.Series(r, index=pd.RangeIndex(n)), pd.Series(rv, index=pd.RangeIndex(n))


def _run(returns: pd.Series, rv: pd.Series, *, refit_every: int) -> pd.DataFrame:
    splitter = RollingOriginSplitter(window=40, horizon=1, step=1, refit_every=refit_every)
    return run_backtest(
        lambda: PatchTST(**SMOKE),
        returns,
        rv,
        splitter,
        0,
        asset="SIM",
        proxy_name="rv",
        fit_series=rv,
    )


class TestThroughTheEvaluator:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_frozen_between_refits_and_conditioned_through_says_so(self) -> None:
        returns, rv = _panel(100)
        out = _run(returns, rv, refit_every=10)
        assert (out["missing_reason"] == "").all()
        assert (out["conditioned_through"] == out["fit_origin"]).all()
        held = ~out["refit"]
        assert held.any()
        assert (out.loc[held, "fit_origin"] < out.loc[held, "origin_index"]).all()
        # within a block the forecast is literally the one issued at the refit
        for _, block in out.groupby("fit_origin"):
            assert block["forecast_var"].nunique() == 1

    def test_leakage_canary_future_corruption_cannot_change_past_forecasts(self) -> None:
        returns, rv = _panel(100)
        cutoff = 40 + 30
        clean = _run(returns, rv, refit_every=5)
        dirty_rv = rv.copy()
        rng = np.random.default_rng(999)
        dirty_rv.iloc[cutoff + 1 :] = np.exp(rng.normal(-5.0, 1.0, size=rv.size - cutoff - 1))
        dirty = _run(returns, dirty_rv, refit_every=5)
        scores = [c for c in clean.columns if c != "config_hash"]
        pd.testing.assert_frame_equal(
            clean.loc[clean["target_index"] <= cutoff, scores].reset_index(drop=True),
            dirty.loc[dirty["target_index"] <= cutoff, scores].reset_index(drop=True),
            check_exact=True,
        )
        future = clean["target_index"] > cutoff + 10
        assert not np.array_equal(
            clean.loc[future, "crps"].to_numpy(), dirty.loc[future, "crps"].to_numpy()
        )


@pytest.mark.gpu
class TestGpu:
    """The real architecture and budget, on the CUDA device."""

    @pytest.fixture(autouse=True)
    def _cuda(self) -> None:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

    def test_bit_identical_twice_at_full_size(self) -> None:
        rv = realized_variance(1000)
        model = PatchTST(device="cuda")
        a, b = model.fit(rv), model.fit(rv)
        assert np.array_equal(a.log_forecast, b.log_forecast)
        assert np.array_equal(a.smearing, b.smearing)
        assert a.predict(1) == b.predict(1)
        assert a.spec() == b.spec()

    def test_budget_is_bounded_and_the_forecast_is_a_sane_daily_variance(self) -> None:
        rv = realized_variance(1000)
        fitted = PatchTST(device="cuda").fit(rv)
        spec = fitted.spec()
        assert spec["epochs_run"] <= 100
        assert spec["n_train_windows"] + spec["n_val_windows"] == 1000 - 64
        level = float(np.mean(rv[-22:]))
        assert 0.3 * level < fitted.predict(1).variance() < 3.0 * level
        assert 0.5 < spec["smearing_factor"][0] < 2.0

    def test_without_dropout_cpu_and_gpu_agree_to_rounding(self) -> None:
        """Same init, same batch order, no stochastic layer: the only
        difference left is float accumulation order (measured ~1e-8)."""
        rv = realized_variance(300)
        cfg = dict(dropout=0.0, max_epochs=5, patience=5)
        gpu = PatchTST(device="cuda", **cfg).fit(rv)
        cpu = PatchTST(device="cpu", **cfg).fit(rv)
        assert gpu.predict(1).variance() == pytest.approx(cpu.predict(1).variance(), rel=1e-6)

    def test_with_dropout_cpu_and_gpu_are_different_realisations(self) -> None:
        """Dropout masks come from the device's own RNG stream, so a CPU fit
        and a CUDA fit of the same seed are two draws, not one — which is why
        results are reproducible per device and `device` is not hashed."""
        rv = realized_variance(300)
        cfg = dict(dropout=0.1, max_epochs=5, patience=5)
        gpu = PatchTST(device="cuda", **cfg).fit(rv)
        cpu = PatchTST(device="cpu", **cfg).fit(rv)
        assert gpu.predict(1).variance() != pytest.approx(cpu.predict(1).variance(), rel=1e-6)
