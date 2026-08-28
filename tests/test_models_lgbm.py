"""LightGBM on lagged log-RV: features, determinism, update, leakage.

Four things this file exists to pin:

1. **Feature construction is strictly backward-looking and window-local.**
   The design matrix for a window must be bit-identical whether or not the
   array it was sliced from continues afterwards, and every row must be a
   function of ``rv[t-21..t]`` alone. This is the focus of the leakage audit
   for this adapter, so it is tested three independent ways: by construction
   (compare against a hand-built row), by truncation (the window cannot tell
   whether a future exists), and by poisoning (through the real evaluator).
2. **No scaler, no early stopping, no validation split.** Their *absence* is
   the invariant — each is a standard way for a boosted-tree forecasting
   pipeline to leak — so it is asserted rather than left to code review.
3. **Determinism is bit-level.** Same window in, byte-identical model and
   bit-identical forecast out, twice.
4. **The retransformation is still doing something.** A high-capacity
   ensemble shrinks its own residuals until Duan's factor collapses to 1 and
   the "variance" forecast is really a median forecast. The ``in_sample`` arm
   is guarded against that, and the default arm does not read those residuals
   at all.
5. **The out-of-fold folds are causal.** The default smearing factor is read
   off expanding chronological folds inside the training window, which is a
   second place a time-series model can leak. It carries the same corruption
   canary the rest of this file applies to the feature path
   (``TestOutOfFoldFoldsAreCausal``) — ported here from
   ``tests/test_lgbm_smearing_probe.py`` when the construction moved out of
   the probe and into the adapter, because a probe being leakage-clean is not
   evidence that the adapter is.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

pytest.importorskip("lightgbm", reason="the `classical` extra is not installed")

from volbench.dist import Distribution, Normal
from volbench.evaluate import (
    FittedModel,
    ForecastModel,
    SupportsUpdate,
    run_backtest,
)
from volbench.models._rv import smearing_factor
from volbench.models.lgbm import (
    _M_WINDOW,
    _MAX_LAG,
    _MIN_TRAIN,
    _N_FEATURES,
    _W_WINDOW,
    DEFAULT_OOF_FOLDS,
    FEATURE_NAMES,
    FittedLightGBMRV,
    LightGBMRV,
    _design_matrix,
    _feature_row,
    out_of_fold_residuals,
)
from volbench.splitter import RollingOriginSplitter

ARMS = [LightGBMRV(), LightGBMRV(retransform="gaussian")]


def log_ar1_rv(
    n: int = 600, phi: float = 0.9, sigma: float = 0.35, level: float = -9.0, seed: int = 0
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    y = np.empty(n, dtype=np.float64)
    y[0] = level
    for t in range(1, n):
        y[t] = level * (1.0 - phi) + phi * y[t - 1] + sigma * rng.standard_normal()
    return np.exp(y)


def toy_rv() -> NDArray[np.float64]:
    from volbench.benchmarks.toy import SCORING_TARGET, load_series

    return load_series().targets[SCORING_TARGET].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# features: the leakage-critical half of this adapter
# --------------------------------------------------------------------------


class TestFeatures:
    def test_a_feature_row_is_exactly_the_documented_har_information_set(self) -> None:
        """Lags 1..22 of log RV, then log of the 5- and 22-day RV means —
        mean of RV then logged, matching ``models/har.py``'s RV_w / RV_m, not
        mean of log RV."""
        rv = log_ar1_rv(n=60, seed=1)
        t = 40
        row = _feature_row(rv, t)
        assert row.shape == (_N_FEATURES,)
        assert len(FEATURE_NAMES) == _N_FEATURES
        for k in range(1, _MAX_LAG + 1):
            assert row[k - 1] == pytest.approx(math.log(rv[t - k + 1]), rel=1e-15)
        assert row[_MAX_LAG] == pytest.approx(
            math.log(float(np.mean(rv[t - _W_WINDOW + 1 : t + 1]))), rel=1e-15
        )
        assert row[_MAX_LAG + 1] == pytest.approx(
            math.log(float(np.mean(rv[t - _M_WINDOW + 1 : t + 1]))), rel=1e-15
        )

    def test_a_feature_row_reads_nothing_after_t(self) -> None:
        """The direct statement of CLAUDE.md rule 1 at the feature level:
        overwrite everything strictly after ``t`` with nonsense and the row
        must not move."""
        rv = log_ar1_rv(n=200, seed=2)
        t = 100
        poisoned = rv.copy()
        poisoned[t + 1 :] = 1e9
        assert np.array_equal(_feature_row(rv, t), _feature_row(poisoned, t))

    def test_the_first_row_has_a_full_lag_window_and_the_last_has_a_target(self) -> None:
        """Strict windows only — no truncated or back-filled rows (HAR's rule).
        ``n - _M_WINDOW`` rows: one per ``t`` from 21 to ``n - 2``."""
        rv = log_ar1_rv(n=200, seed=3)
        x, y = _design_matrix(rv)
        assert x.shape == (rv.size - _M_WINDOW, _N_FEATURES)
        assert y.shape == (rv.size - _M_WINDOW,)
        assert np.array_equal(x[0], _feature_row(rv, _M_WINDOW - 1))
        assert y[0] == pytest.approx(math.log(rv[_M_WINDOW]), rel=1e-15)
        assert np.array_equal(x[-1], _feature_row(rv, rv.size - 2))
        assert y[-1] == pytest.approx(math.log(rv[-1]), rel=1e-15)
        assert np.isfinite(x).all() and np.isfinite(y).all()

    def test_every_row_pairs_features_dated_t_with_the_target_at_t_plus_one(self) -> None:
        """An off-by-one here would train the model to predict *today* from
        today — invisible in every score, and the model would look excellent."""
        rv = log_ar1_rv(n=120, seed=4)
        x, y = _design_matrix(rv)
        for i in range(x.shape[0]):
            t = _M_WINDOW - 1 + i
            assert x[i, 0] == pytest.approx(math.log(rv[t]), rel=1e-15)
            assert y[i] == pytest.approx(math.log(rv[t + 1]), rel=1e-15)

    def test_a_feature_row_refuses_a_truncated_history(self) -> None:
        """Found by the leakage-check audit. A negative slice start is read by
        numpy as ``len(rv) + start``, so ``t < 21`` used to return a silently
        short row assembled from a slice chosen by the array's total length
        rather than by ``t`` — the exact shape of a leak, even though no
        caller could reach it. It raises now."""
        rv = log_ar1_rv(n=200, seed=8)
        assert _feature_row(rv, _M_WINDOW - 1).shape == (_N_FEATURES,)
        for bad in (0, 5, _M_WINDOW - 2):
            with pytest.raises(ValueError, match=f"full {_M_WINDOW}-day history"):
                _feature_row(rv, bad)
        with pytest.raises(ValueError, match=f"full {_M_WINDOW}-day history"):
            _feature_row(rv, rv.size)

    def test_the_design_matrix_is_window_local(self) -> None:
        """THE feature-construction leakage check: build a window's design
        matrix from a short array and from a long array that merely *starts*
        with it. Any full-series statistic — a mean, a scaler, a quantile
        binning fitted outside the window — makes these differ."""
        long = log_ar1_rv(n=600, seed=5)
        window = long[:300].copy()
        x_short, y_short = _design_matrix(window)
        x_long, y_long = _design_matrix(long[:300])
        assert np.array_equal(x_short, x_long)
        assert np.array_equal(y_short, y_long)

    def test_a_fit_on_a_window_ignores_everything_after_it(self) -> None:
        """The same claim one level up, through the trained model: two fits on
        the same 300 observations must forecast identically, whatever follows
        those 300 observations in the array they were sliced out of."""
        long = log_ar1_rv(n=600, seed=6)
        poisoned = long.copy()
        poisoned[300:] = 1e3
        a = LightGBMRV().fit(long[:300]).predict(1)
        b = LightGBMRV().fit(poisoned[:300]).predict(1)
        assert isinstance(a, Normal) and isinstance(b, Normal)
        assert a.sigma == b.sigma  # bit-identical

    def test_no_scaling_or_normalization_is_applied(self) -> None:
        """The absence is the invariant: a scaler is the standard way this
        pipeline leaks, and trees would gain nothing from one anyway.

        Multiply the whole RV series by a constant: in log space that is an
        additive shift of every feature and every target, so an unscaled
        pipeline's *residuals* — and therefore its smearing factor — are
        invariant, while any fitted scaler would perturb them.

        The tolerance is 1e-6 rather than exact because ``log(7 * rv)`` and
        ``log(rv) + log(7)`` differ in the last few bits and histogram bin
        edges can land either side of that; a scaler would move these numbers
        by orders of magnitude, not by 2e-8.
        """
        rv = toy_rv()[:400]
        base = LightGBMRV().fit(rv)
        shifted = LightGBMRV().fit(rv * 7.0)
        assert shifted.smear == pytest.approx(base.smear, rel=1e-6)
        assert shifted.resid_var == pytest.approx(base.resid_var, rel=1e-6)
        assert shifted.predict(1).variance() == pytest.approx(
            7.0 * base.predict(1).variance(), rel=1e-6
        )

    def test_no_early_stopping_and_no_validation_split(self) -> None:
        """Early stopping needs held-out data; the only data available to hold
        out is inside the training window, and a random split of a time series
        is itself a leak. Fixed rounds instead — pinned so nobody adds one."""
        config = LightGBMRV(num_boost_round=37)
        fitted = config.fit(log_ar1_rv(n=400, seed=7))
        assert fitted.booster.num_trees() == 37
        assert config.spec()["num_boost_round"] == 37
        assert "early_stopping_round" not in config.spec()
        assert "valid_sets" not in config.spec()


# --------------------------------------------------------------------------
# interface and input contract
# --------------------------------------------------------------------------


class TestInterface:
    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_satisfies_the_one_model_protocol(self, config: LightGBMRV) -> None:
        assert isinstance(config, ForecastModel)
        fitted = config.fit(log_ar1_rv(n=300))
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted, SupportsUpdate)

    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_predict_returns_a_zero_mean_return_distribution(
        self, config: LightGBMRV
    ) -> None:
        rv = log_ar1_rv(n=400)
        dist = config.fit(rv).predict(1)
        assert isinstance(dist, Normal)
        assert dist.mu == 0.0
        assert dist.sigma > 0.0
        assert 0.1 < dist.variance() / float(np.median(rv)) < 10.0  # daily units

    def test_predict_rejects_a_nonpositive_horizon(self) -> None:
        fitted = LightGBMRV().fit(log_ar1_rv(n=300))
        with pytest.raises(ValueError, match="h must be >= 1"):
            fitted.predict(0)

    def test_multistep_predict_iterates_the_buffer(self) -> None:
        """h > 1 feeds each retransformed point forecast back in as if
        realized, exactly as ``models/har.py`` does."""
        fitted = LightGBMRV().fit(log_ar1_rv(n=400))
        one = fitted.predict(1).variance()
        buf = np.append(fitted.buffer[1:], one)
        stepped = FittedLightGBMRV(
            config=fitted.config,
            booster=fitted.booster,
            buffer=buf,
            smear=fitted.smear,
            resid_var=fitted.resid_var,
        )
        assert fitted.predict(2).variance() == pytest.approx(
            stepped.predict(1).variance(), rel=1e-15
        )


class TestInputContract:
    def test_fit_rejects_nonpositive_variance(self) -> None:
        rv = log_ar1_rv(n=300)
        rv[50] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            LightGBMRV().fit(rv)

    def test_fit_rejects_nonfinite_variance(self) -> None:
        rv = log_ar1_rv(n=300)
        rv[9] = np.nan
        with pytest.raises(ValueError, match="finite"):
            LightGBMRV().fit(rv)

    def test_fit_rejects_a_too_short_window(self) -> None:
        with pytest.raises(ValueError, match=f"at least {_MIN_TRAIN}"):
            LightGBMRV().fit(log_ar1_rv(n=_MIN_TRAIN - 1))

    def test_update_validates_like_fit_but_only_needs_the_buffer(self) -> None:
        """``update`` re-reads 22 lags, so it needs 22 observations, not a
        whole training window — the same asymmetry ``models/har.py`` has."""
        rv = log_ar1_rv(n=400)
        fitted = LightGBMRV().fit(rv[:300])
        assert fitted.update(rv[:_M_WINDOW]) is not None
        with pytest.raises(ValueError, match=f"at least {_M_WINDOW}"):
            fitted.update(rv[: _M_WINDOW - 1])
        bad = rv[100:400].copy()
        bad[-1] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            fitted.update(bad)

    def test_a_bad_origin_costs_one_row_not_the_cell(self) -> None:
        rv = log_ar1_rv(n=420)
        returns = np.sqrt(rv) * np.random.default_rng(0).standard_normal(rv.size)
        rv[305] = 0.0
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=5)
        frame = run_backtest(
            LightGBMRV, returns, rv, splitter, 7, asset="T", proxy_name="p", fit_series=rv
        )
        assert frame["forecast_var"].notna().any()
        assert frame["forecast_var"].isna().any()
        assert frame.loc[frame["forecast_var"].isna(), "missing_reason"].str.contains(
            "fit_error|update_error"
        ).all()


# --------------------------------------------------------------------------
# recovery / sanity
# --------------------------------------------------------------------------


class TestRecoveryAndSanity:
    def test_a_near_constant_rv_series_forecasts_that_constant(self) -> None:
        """The units canary: feed a series pinned at 4e-4 and the variance
        forecast must be 4e-4 — not its log, its square root or 252x it."""
        rng = np.random.default_rng(11)
        rv = 4e-4 * np.exp(rng.normal(0.0, 0.01, 500))
        assert LightGBMRV().fit(rv).predict(1).variance() == pytest.approx(4e-4, rel=0.02)

    def test_it_learns_a_deterministic_dependence_on_the_first_lag(self) -> None:
        """A noiseless RV series where ``log RV_{t+1} = 0.8 * log RV_t + c``.
        A model that has actually learned the mapping predicts the next value;
        one that has learned the unconditional mean does not."""
        n = 800
        y = np.empty(n)
        y[0] = -9.0
        for t in range(1, n):
            y[t] = -9.0 * 0.2 + 0.8 * y[t - 1] + 0.25 * math.sin(0.7 * t)
        rv = np.exp(y)
        fitted = LightGBMRV(num_boost_round=400, num_leaves=8, min_data_in_leaf=20).fit(rv)
        mu_hat = math.log(fitted.predict(1).variance() / fitted.smear)
        truth = -9.0 * 0.2 + 0.8 * y[-1] + 0.25 * math.sin(0.7 * n)
        naive = float(np.mean(y))
        assert abs(mu_hat - truth) < abs(naive - truth)
        assert abs(mu_hat - truth) < 0.25

    def test_the_first_lag_is_the_dominant_feature_on_persistent_rv(self) -> None:
        """Sanity on the plumbing, not on LightGBM: if the feature block were
        reversed or transposed, the split gains would not concentrate on the
        recent lags and the aggregates."""
        fitted = LightGBMRV().fit(toy_rv()[:500])
        gains = fitted.booster.feature_importance(importance_type="gain")
        assert list(fitted.booster.feature_name()) == list(FEATURE_NAMES)
        assert gains.sum() > 0.0
        recent = {"log_rv_lag1", f"log_mean_rv_{_W_WINDOW}", f"log_mean_rv_{_M_WINDOW}"}
        share = sum(
            g for name, g in zip(FEATURE_NAMES, gains, strict=True) if name in recent
        ) / float(gains.sum())
        assert share > 0.4

    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_forecast_tracks_the_level_of_the_toy_rv_series(
        self, config: LightGBMRV
    ) -> None:
        """A conditional mean against an unconditional one, so the band is
        wide by construction.

        The upper edge was 2.0 while the smearing factor was read off
        in-sample residuals; the out-of-fold factor is larger — that is the
        whole of the fix — and on this fixture's last origin it puts the ratio
        at 2.05. The band is widened rather than the factor trimmed: nothing
        here says the forecast is *right*, only that it is on the scale of the
        series it was fitted to.
        """
        rv = toy_rv()[:500]
        assert 0.5 < config.fit(rv).predict(1).variance() / float(np.mean(rv)) < 2.5


# --------------------------------------------------------------------------
# retransformation
# --------------------------------------------------------------------------


class TestRetransformation:
    def test_smearing_is_the_default_arm(self) -> None:
        assert LightGBMRV().retransform == "smearing"

    def test_the_smearing_factor_is_duans_mean_of_exponentiated_residuals(self) -> None:
        """Duan's formula itself, on the arm whose residuals are the fitted
        ones. The default arm's version of this is
        ``test_the_factor_is_exactly_duans_mean_over_the_out_of_fold_residuals``.
        """
        rv = toy_rv()[:500]
        fitted = LightGBMRV(smearing_residuals="in_sample").fit(rv)
        x, y = _design_matrix(rv)
        resid = y - np.asarray(fitted.booster.predict(x), dtype=np.float64).reshape(-1)
        assert fitted.smear == pytest.approx(float(np.mean(np.exp(resid))), rel=1e-15)
        assert fitted.smear == pytest.approx(smearing_factor(resid), rel=1e-15)
        assert fitted.resid_var == pytest.approx(float(np.mean(resid * resid)), rel=1e-15)

    def test_smearing_predict_is_exactly_exp_mu_times_the_factor(self) -> None:
        rv = toy_rv()[:500]
        fitted = LightGBMRV().fit(rv)
        row = _feature_row(fitted.buffer, _M_WINDOW - 1).reshape(1, _N_FEATURES)
        mu = float(np.asarray(fitted.booster.predict(row), dtype=np.float64).reshape(-1)[0])
        assert fitted.predict(1).variance() == pytest.approx(
            math.exp(mu) * fitted.smear, rel=1e-14
        )

    def test_gaussian_predict_is_exactly_exp_mu_plus_half_resid_var(self) -> None:
        rv = toy_rv()[:500]
        fitted = LightGBMRV(retransform="gaussian").fit(rv)
        row = _feature_row(fitted.buffer, _M_WINDOW - 1).reshape(1, _N_FEATURES)
        mu = float(np.asarray(fitted.booster.predict(row), dtype=np.float64).reshape(-1)[0])
        assert fitted.predict(1).variance() == pytest.approx(
            math.exp(mu + 0.5 * fitted.resid_var), rel=1e-14
        )

    def test_both_arms_exceed_the_median_and_disagree(self) -> None:
        rv = toy_rv()[:500]
        smeared = LightGBMRV().fit(rv)
        gaussian = LightGBMRV(retransform="gaussian").fit(rv)
        row = _feature_row(smeared.buffer, _M_WINDOW - 1).reshape(1, _N_FEATURES)
        median = math.exp(
            float(np.asarray(smeared.booster.predict(row), dtype=np.float64).reshape(-1)[0])
        )
        assert smeared.predict(1).variance() > median
        assert gaussian.predict(1).variance() > median
        assert smeared.predict(1).variance() != gaussian.predict(1).variance()

    def test_the_ensemble_does_not_memorize_its_own_residuals(self) -> None:
        """THE capacity guard (module docstring), on the arm it guards.

        The ``in_sample`` factor is estimated from residuals the booster has
        already fitted, so an ensemble with enough capacity to drive them
        toward zero drives that factor toward 1 and silently turns the
        variance forecast back into a median forecast.

        On the toy fixture the realized one-step log-space forecast-error
        variance is ~0.38 and HAR's in-sample residual variance is 0.377. The
        default *capacity* must stay in that neighbourhood — it is unchanged,
        and it is what bounds this arm. LightGBM's stock shape lands at 0.015
        — a 25x understatement — and the second half of this test shows the
        trap is real rather than hypothetical.
        """
        rv = toy_rv()[:500]
        in_sample = LightGBMRV(smearing_residuals="in_sample").fit(rv)
        assert in_sample.resid_var > 0.15, (
            f"in-sample residual variance {in_sample.resid_var:.4f} is far below the ~0.38 "
            "realized forecast-error variance: the ensemble is memorizing and the "
            "retransformation has stopped correcting anything"
        )
        assert in_sample.smear > 1.05

        memorizing = LightGBMRV(
            smearing_residuals="in_sample", num_boost_round=300, num_leaves=15,
            min_data_in_leaf=20,
        ).fit(rv)
        assert memorizing.resid_var < 0.05
        assert memorizing.smear < 1.02  # the correction has collapsed

    def test_out_of_fold_is_the_default_and_corrects_by_more(self) -> None:
        """The fix, as a property rather than a panel number.

        Out-of-fold residuals are genuine one-step errors of the same
        estimator, so they are larger than the fitted ones and the factor they
        imply is larger too. docs/P3_LGBM_SMEARING_AUDIT.md puts the panel
        medians at 1.371 in-sample against a realized 1.678, and the
        out-of-fold estimate at 1.703; the direction is what generalizes to
        this fixture, not the magnitude.
        """
        assert LightGBMRV().smearing_residuals == "out_of_fold"
        rv = toy_rv()[:500]
        oof = LightGBMRV().fit(rv)
        in_sample = LightGBMRV(smearing_residuals="in_sample").fit(rv)
        assert oof.smear > in_sample.smear
        # The point forecast is the same booster on both arms: only the
        # retransformation moved.
        assert oof.predict(1).variance() == pytest.approx(
            in_sample.predict(1).variance() * oof.smear / in_sample.smear, rel=1e-12
        )

    def test_the_factor_is_exactly_duans_mean_over_the_out_of_fold_residuals(self) -> None:
        rv = toy_rv()[:500]
        config = LightGBMRV()
        fitted = config.fit(rv)
        x, y = _design_matrix(rv)
        resid = out_of_fold_residuals(config, x, y, config.oof_folds)
        assert fitted.smear == pytest.approx(float(np.mean(np.exp(resid))), rel=1e-15)
        assert fitted.smear == pytest.approx(smearing_factor(resid), rel=1e-15)

    def test_the_gaussian_arm_keeps_the_in_sample_residual_variance(self) -> None:
        """``resid_var`` is HAR's estimator on this model's residuals and must
        stay in-sample on both arms, or the like-for-like comparison arm stops
        being like for like."""
        rv = toy_rv()[:500]
        x, y = _design_matrix(rv)
        for residuals in ("out_of_fold", "in_sample"):
            fitted = LightGBMRV(smearing_residuals=residuals).fit(rv)  # type: ignore[arg-type]
            in_sample = y - np.asarray(
                fitted.booster.predict(x), dtype=np.float64
            ).reshape(-1)
            assert fitted.resid_var == pytest.approx(
                float(np.mean(in_sample * in_sample)), rel=1e-15
            )


# --------------------------------------------------------------------------
# the out-of-fold folds: the second place this adapter could leak
# --------------------------------------------------------------------------


class TestOutOfFoldFoldsAreCausal:
    """The canary, ported from ``tests/test_lgbm_smearing_probe.py``.

    K's probe carried it while the construction lived there; the construction
    now ships inside ``fit``, and a probe being leakage-clean is not evidence
    that the adapter is. Same three claims, against the adapter's own
    function: corruption cannot travel backwards, the boundary is causal with
    no gap, and every fold but the first contributes.
    """

    def test_corrupting_the_future_leaves_earlier_folds_bit_identical(self) -> None:
        """Replace every design row from the second fold boundary onward with
        noise; the residuals of the fold *before* it must not move by one bit.
        If a fold ever trained on data after its own block, they would."""
        config = LightGBMRV()
        x, y = _design_matrix(log_ar1_rv(n=600, seed=11))
        edges = np.linspace(0, y.size, DEFAULT_OOF_FOLDS + 1).astype(int)
        first_block = int(edges[2] - edges[1])

        clean = out_of_fold_residuals(config, x, y, DEFAULT_OOF_FOLDS)

        rng = np.random.default_rng(1)
        x2, y2 = x.copy(), y.copy()
        x2[edges[2] :] = rng.standard_normal(x2[edges[2] :].shape)
        y2[edges[2] :] = rng.standard_normal(y2[edges[2] :].shape)
        corrupted = out_of_fold_residuals(config, x2, y2, DEFAULT_OOF_FOLDS)

        assert np.array_equal(clean[:first_block], corrupted[:first_block])
        assert not np.array_equal(clean, corrupted)  # later folds must react

    def test_a_fitted_factor_ignores_everything_after_its_own_window(self) -> None:
        """The same claim through ``fit``, which is what the grid calls: the
        smearing factor of a window must not depend on what follows it in the
        array the window was sliced from."""
        long = log_ar1_rv(n=800, seed=12)
        poisoned = long.copy()
        poisoned[400:] = 1e3
        assert LightGBMRV().fit(long[:400]).smear == LightGBMRV().fit(poisoned[:400]).smear

    def test_the_last_training_target_never_postdates_the_predicted_rows_origin(self) -> None:
        """The boundary the canary cannot see, stated arithmetically.

        Design row ``i`` reads ``rv[i : i+22]`` and predicts ``rv[i+22]``. A
        booster trained on rows ``0..train_end-1`` has therefore seen targets
        up to ``rv[train_end+21]`` — which is exactly the last observation
        known at the origin of row ``train_end``, whose own target is
        ``rv[train_end+22]``. Causal with no gap, and no gap needed.
        """
        rv = log_ar1_rv(n=200, seed=13)
        _x, y = _design_matrix(rv)
        train_end = 50
        last_train_target_pos = train_end - 1 + _M_WINDOW
        predicted_row_origin_pos = train_end + _M_WINDOW - 1
        predicted_row_target_pos = train_end + _M_WINDOW
        assert last_train_target_pos == predicted_row_origin_pos
        assert last_train_target_pos < predicted_row_target_pos
        assert y[train_end - 1] == pytest.approx(float(np.log(rv[last_train_target_pos])))

    def test_every_fold_but_the_first_contributes(self) -> None:
        config = LightGBMRV()
        x, y = _design_matrix(log_ar1_rv(n=600, seed=14))
        edges = np.linspace(0, y.size, DEFAULT_OOF_FOLDS + 1).astype(int)
        residuals = out_of_fold_residuals(config, x, y, DEFAULT_OOF_FOLDS)
        assert residuals.size == y.size - edges[1]

    def test_a_window_too_short_to_fold_raises_rather_than_returning_nothing(self) -> None:
        """In the adapter, unlike in the probe, an empty residual set must not
        be survivable: it would fall through to a factor that is not the one
        ``spec()`` names. ``run_backtest`` turns the raise into one NaN row."""
        config = LightGBMRV()
        x, y = _design_matrix(log_ar1_rv(n=_M_WINDOW + 2, seed=15))
        assert y.size == 2
        with pytest.raises(ValueError, match="too short to form"):
            out_of_fold_residuals(config, x, y, folds=20)

    def test_the_shipped_minimum_window_always_folds(self) -> None:
        """``_MIN_TRAIN`` is the shortest window ``fit`` accepts, so it must be
        long enough that the default folds exist — otherwise the raise above
        would be reachable from the grid."""
        fitted = LightGBMRV().fit(log_ar1_rv(n=_MIN_TRAIN, seed=16))
        assert fitted.smear > 0.0

    def test_the_folds_are_deterministic(self) -> None:
        config = LightGBMRV()
        x, y = _design_matrix(log_ar1_rv(n=600, seed=17))
        first = out_of_fold_residuals(config, x, y, DEFAULT_OOF_FOLDS)
        second = out_of_fold_residuals(config, x, y, DEFAULT_OOF_FOLDS)
        assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# update(): exact under fixed trees
# --------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_update_on_the_fit_window_is_the_fit(self, config: LightGBMRV) -> None:
        rv = log_ar1_rv(n=400, seed=21)
        fitted = config.fit(rv[:300])
        again = fitted.update(rv[:300])
        assert np.array_equal(again.buffer, fitted.buffer)
        for h in (1, 5):
            assert again.predict(h) == fitted.predict(h)

    def test_update_moves_only_the_buffer(self) -> None:
        rv = log_ar1_rv(n=500, seed=22)
        fitted = LightGBMRV().fit(rv[:300])
        again = fitted.update(rv[10:310])
        assert again.booster is fitted.booster
        assert again.smear == fitted.smear
        assert again.resid_var == fitted.resid_var
        assert again.spec() == fitted.spec() and again.name == fitted.name
        assert np.array_equal(again.buffer, rv[10:310][-_M_WINDOW:])
        assert not np.array_equal(again.buffer, fitted.buffer)
        assert again.predict(1) != fitted.predict(1)

    def test_update_is_exact_not_approximate(self) -> None:
        """Refreshing features under fixed trees IS the forecast this
        parameterization implies at the later origin — so it equals what a
        model carrying these trees and that buffer would produce, built any
        other way."""
        rv = log_ar1_rv(n=500, seed=23)
        fitted = LightGBMRV().fit(rv[:300])
        again = fitted.update(rv[10:310])
        rebuilt = FittedLightGBMRV(
            config=fitted.config,
            booster=fitted.booster,
            buffer=rv[10:310][-_M_WINDOW:].copy(),
            smear=fitted.smear,
            resid_var=fitted.resid_var,
        )
        assert again.predict(1) == rebuilt.predict(1)

    def test_predicting_does_not_mutate_the_shared_booster(self) -> None:
        rv = log_ar1_rv(n=500, seed=24)
        fitted = LightGBMRV().fit(rv[:300])
        first = fitted.predict(1)
        fitted.update(rv[10:310]).predict(5)
        assert fitted.predict(1) == first

    def test_the_evaluator_honours_the_refit_cadence(self) -> None:
        """``refit_every=10`` must mean ten re-conditioned origins per fit,
        not ten frozen ones — M1 report §4.3, closed for this model because
        ``update`` exists."""
        rv = log_ar1_rv(n=420, seed=25)
        returns = np.sqrt(rv) * np.random.default_rng(26).standard_normal(rv.size)
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=10)
        frame = run_backtest(
            LightGBMRV, returns, rv, splitter, 3, asset="T", proxy_name="p", fit_series=rv
        )
        assert (frame["conditioned_through"] == frame["origin_index"]).all()
        assert frame["forecast_var"].nunique() == len(frame)


# --------------------------------------------------------------------------
# determinism (CLAUDE.md rule 3) — the pinned reproducibility test
# --------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_same_window_in_bit_identical_forecast_out_twice(
        self, config: LightGBMRV
    ) -> None:
        rv = toy_rv()[:500]
        first = config.fit(rv)
        second = config.fit(rv)
        assert first.predict(1).sigma == second.predict(1).sigma  # bit-identical
        assert first.predict(22).sigma == second.predict(22).sigma
        assert first.smear == second.smear
        assert first.resid_var == second.resid_var

    def test_two_fits_produce_the_identical_serialized_model(self) -> None:
        """Stronger than equal forecasts: every split, threshold and leaf
        value must match, so a difference cannot be hiding in a region the
        forecast happens not to reach."""
        rv = toy_rv()[:500]
        a = LightGBMRV().fit(rv).booster.model_to_string()
        b = LightGBMRV().fit(rv).booster.model_to_string()
        assert a == b

    def test_the_determinism_settings_are_all_pinned_and_hashed(self) -> None:
        """``deterministic`` is inert without a forced histogram strategy, and
        both are inert if the thread count or the seed can drift."""
        spec = LightGBMRV().spec()
        assert spec["deterministic"] is True
        assert spec["force_row_wise"] is True
        assert spec["num_threads"] == 1
        assert isinstance(spec["seed"], int)

    def test_a_different_seed_gives_a_different_model(self) -> None:
        """The seed must actually reach the sampling, or pinning it is
        theatre."""
        rv = toy_rv()[:500]
        a = LightGBMRV(seed=1).fit(rv)
        b = LightGBMRV(seed=2).fit(rv)
        assert a.booster.model_to_string() != b.booster.model_to_string()

    def test_the_backtest_is_reproducible_end_to_end(self) -> None:
        rv = log_ar1_rv(n=400, seed=27)
        returns = np.sqrt(rv) * np.random.default_rng(28).standard_normal(rv.size)
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=5)
        kwargs: dict[str, Any] = dict(asset="T", proxy_name="p", fit_series=rv)
        first = run_backtest(LightGBMRV, returns, rv, splitter, 9, **kwargs)
        second = run_backtest(LightGBMRV, returns, rv, splitter, 9, **kwargs)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        assert first.attrs["config_hash"] == second.attrs["config_hash"]


# --------------------------------------------------------------------------
# spec(): complete, hashable, discriminating
# --------------------------------------------------------------------------


class TestSpec:
    @pytest.mark.parametrize("config", ARMS, ids=lambda c: c.name)
    def test_spec_is_json_serializable_and_matches_the_fitted_one(
        self, config: LightGBMRV
    ) -> None:
        json.dumps(config.spec(), sort_keys=True)
        assert config.fit(log_ar1_rv(n=300)).spec() == config.spec()

    def test_spec_carries_every_parameter_handed_to_lightgbm(self) -> None:
        """A hyperparameter that reaches the backend but not the spec can move
        a published number without moving its ``config_hash``."""
        config = LightGBMRV()
        spec = config.spec()
        for key, value in config._params().items():
            assert spec[key] == value
        for extra in (
            "model",
            "backend",
            "target",
            "max_lag",
            "w_window",
            "m_window",
            "n_features",
            "num_boost_round",
            "retransform",
            "smearing_residuals",
            "oof_folds",
            "min_train",
        ):
            assert extra in spec

    def test_spec_is_stable_and_discriminating(self) -> None:
        assert LightGBMRV().spec() == LightGBMRV().spec()
        assert LightGBMRV().spec() != LightGBMRV(retransform="gaussian").spec()
        # The whole point of naming the construction in spec(): a fix that
        # changed the numbers without moving the hash would let one
        # config_hash name two different sets of forecasts.
        assert LightGBMRV().spec() != LightGBMRV(smearing_residuals="in_sample").spec()
        assert LightGBMRV().spec() != LightGBMRV(oof_folds=4).spec()
        assert LightGBMRV().spec() != LightGBMRV(num_leaves=8).spec()
        assert LightGBMRV().spec() != LightGBMRV(seed=1).spec()
        assert LightGBMRV().name != LightGBMRV(retransform="gaussian").name

    def test_spec_does_not_collide_with_the_other_rv_models(self) -> None:
        pytest.importorskip("statsforecast")
        from volbench.models.har import HAR
        from volbench.models.sf import AutoARIMARV, AutoETSRV

        specs = [
            json.dumps(m.spec(), sort_keys=True)
            for m in (LightGBMRV(), AutoETSRV(), AutoARIMARV(), HAR())
        ]
        assert len(set(specs)) == len(specs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"retransform": "lognormal"}, "retransform"),
            ({"smearing_residuals": "oob"}, "smearing_residuals"),
            ({"oof_folds": 1}, "oof_folds"),
            ({"num_boost_round": 0}, "num_boost_round"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"num_leaves": 1}, "num_leaves"),
            ({"feature_fraction": 1.5}, "feature_fraction"),
            ({"bagging_fraction": 0.0}, "bagging_fraction"),
            ({"num_threads": 0}, "num_threads"),
        ],
    )
    def test_an_invalid_configuration_is_rejected_at_construction(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            LightGBMRV(**kwargs)


# --------------------------------------------------------------------------
# leakage, through the real evaluator
# --------------------------------------------------------------------------


class _FitSpy:
    """Wraps the real config and records every array ``fit``/``update`` saw."""

    def __init__(self, config: LightGBMRV) -> None:
        self.config = config
        self.fit_windows: list[NDArray[np.float64]] = []
        self.update_windows: list[NDArray[np.float64]] = []

    @property
    def name(self) -> str:
        return self.config.name

    def spec(self) -> dict[str, Any]:
        return dict(self.config.spec())

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _FitSpyFitted:
        self.fit_windows.append(np.asarray(train, dtype=np.float64).copy())
        return _FitSpyFitted(self, self.config.fit(train))


class _FitSpyFitted:
    def __init__(self, owner: _FitSpy, inner: FittedLightGBMRV) -> None:
        self.owner = owner
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    def spec(self) -> dict[str, Any]:
        return self.inner.spec()

    def predict(self, h: int) -> Distribution:
        return self.inner.predict(h)

    def update(self, train: NDArray[np.float64]) -> _FitSpyFitted:
        self.owner.update_windows.append(np.asarray(train, dtype=np.float64).copy())
        return _FitSpyFitted(self.owner, self.inner.update(train))


def _panel(n: int = 420, seed: int = 31) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rv = log_ar1_rv(n=n, seed=seed)
    returns = np.sqrt(rv) * np.random.default_rng(seed + 1).standard_normal(n)
    return returns, rv


class TestLeakage:
    def test_fit_and_update_see_only_the_origins_own_train_window(self) -> None:
        """The canary from ``tests/test_evaluate.py::
        test_fit_series_is_sliced_with_the_same_splitter_indices``, aimed at
        this adapter: every array either entry point receives is exactly
        ``rv[origin.train]`` — splitter indices, nothing hand-rolled."""
        returns, rv = _panel()
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=5)
        spy = _FitSpy(LightGBMRV())
        run_backtest(
            lambda: spy, returns, rv, splitter, 3, asset="T", proxy_name="p", fit_series=rv
        )
        origins = list(splitter.split(returns.size))
        for window, origin in zip(
            spy.fit_windows, [o for o in origins if o.refit], strict=True
        ):
            assert np.array_equal(window, rv[origin.train])
        for window, origin in zip(
            spy.update_windows, [o for o in origins if not o.refit], strict=True
        ):
            assert np.array_equal(window, rv[origin.train])
        assert len(spy.fit_windows) > 1 and len(spy.update_windows) > 1
        assert all(int(o.train[-1]) <= o.origin for o in origins)

    def test_the_proxy_never_reaches_the_model(self) -> None:
        returns, rv = _panel()
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=10)
        kwargs: dict[str, Any] = dict(asset="T", proxy_name="p", fit_series=rv)
        base = run_backtest(LightGBMRV, returns, rv, splitter, 3, **kwargs)
        other = run_backtest(
            LightGBMRV, returns, np.full_like(rv, 4.2e-4), splitter, 3, **kwargs
        )
        pd.testing.assert_frame_equal(
            base[["forecast_mean", "forecast_var"]], other[["forecast_mean", "forecast_var"]]
        )

    def test_future_data_cannot_touch_earlier_forecasts(self) -> None:
        """THE canary (`.claude/skills/leakage-check`). For a tree ensemble
        this is the check that matters most: histogram binning is a
        data-dependent transform, and a binning fitted over anything wider
        than the training window would show up here and nowhere else."""
        returns, rv = _panel(n=400, seed=41)
        splitter = RollingOriginSplitter(window=300, horizon=1, step=1, refit_every=5)
        origins = list(splitter.split(returns.size))
        cutoff = int(origins[40].test[-1])

        rng = np.random.default_rng(99)
        dirty_returns, dirty_rv = returns.copy(), rv.copy()
        n_bad = dirty_rv[cutoff + 1 :].size
        dirty_returns[cutoff + 1 :] = rng.normal(0.0, 0.5, n_bad)
        dirty_rv[cutoff + 1 :] = np.exp(rng.normal(0.0, 1.0, n_bad))

        kwargs: dict[str, Any] = dict(asset="T", proxy_name="p")
        clean = run_backtest(LightGBMRV, returns, rv, splitter, 5, fit_series=rv, **kwargs)
        dirty = run_backtest(
            LightGBMRV, dirty_returns, dirty_rv, splitter, 5, fit_series=dirty_rv, **kwargs
        )

        scores = [c for c in clean.columns if c != "config_hash"]
        before = clean.loc[clean["target_index"] <= cutoff, scores].reset_index(drop=True)
        after = dirty.loc[dirty["target_index"] <= cutoff, scores].reset_index(drop=True)
        assert len(before) > 20
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        assert clean.attrs["config_hash"] != dirty.attrs["config_hash"]

        # A canary that cannot die proves nothing. Row 0 of the tail is exempt
        # and must be: its target is ``cutoff + 1``, so its training window
        # ends at ``cutoff`` and holds no corrupted observation.
        late_clean = clean.loc[clean["target_index"] > cutoff, "forecast_var"].to_numpy()
        late_dirty = dirty.loc[dirty["target_index"] > cutoff, "forecast_var"].to_numpy()
        assert len(late_clean) > 20
        assert late_clean[0] == late_dirty[0]
        assert (late_clean[1:] != late_dirty[1:]).all()
