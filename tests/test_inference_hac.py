"""Tests for the pre-whitened HAC variance, the kernel-mode DM test, the
semi-quadratic MCS statistic and the public bootstrap wrappers.

Every test here has a decidable answer: a closed form worked by hand, a known
long-run variance, a bit-identity against an estimator already in the tree, or
a size that a correct test must have and an incorrect one must not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from volbench import analysis
from volbench.inference import (
    MAX_PREWHITEN_RHO,
    HACSpec,
    _ppw_block_length,
    _semi_quadratic_bootstrap,
    andrews_bandwidth,
    ar1_block_length,
    ar1_coefficient,
    autocorrelation,
    bootstrap_column_means,
    diebold_mariano,
    dm_matrix,
    effective_sample_size,
    long_run_variance,
    model_confidence_set,
    moving_block_indices,
    politis_white_block_length,
    rule_of_thumb_bandwidth,
)


def _ar1(rng: np.random.Generator, rho: float, n: int, sigma: float = 1.0) -> NDArray[np.float64]:
    shocks = sigma * rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + shocks[t]
    return x


# --------------------------------------------------------------------------
# the pieces, by hand
# --------------------------------------------------------------------------


class TestPieces:
    def test_ar1_coefficient_and_autocorrelation_by_hand(self) -> None:
        x = np.array([1.0, 3.0, 2.0, 6.0])  # mean 3 -> centred [-2, 0, -1, 3]
        # OLS slope: sum(c[1:] c[:-1]) / sum(c[:-1]^2) = (0 + 0 - 3) / (4 + 0 + 1)
        assert ar1_coefficient(x) == pytest.approx(-3.0 / 5.0)
        # autocorrelation: gamma_1 / gamma_0 = (-3) / (4 + 0 + 1 + 9)
        assert autocorrelation(x, 1) == pytest.approx(-3.0 / 14.0)
        assert math.isnan(autocorrelation(np.full(5, 2.0)))
        assert ar1_coefficient(np.full(5, 2.0)) == 0.0

    def test_andrews_bandwidth_is_the_published_formula(self) -> None:
        rng = np.random.default_rng(1)
        x = _ar1(rng, 0.6, 3000)
        rho = ar1_coefficient(x)
        alpha1 = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
        assert andrews_bandwidth(x) == pytest.approx(1.1447 * (alpha1 * x.size) ** (1.0 / 3.0))

    def test_andrews_bandwidth_of_white_noise_is_small(self) -> None:
        x = np.random.default_rng(2).standard_normal(5000)
        # rho_hat ~ 1/sqrt(n) makes the plug-in ~ 1.1447 (4 rho^2 n)^{1/3} ~ 2: a lag or two.
        assert andrews_bandwidth(x) < 3.0
        assert andrews_bandwidth(_ar1(np.random.default_rng(2), 0.9, 5000)) > 20.0

    def test_effective_sample_size(self) -> None:
        assert effective_sample_size(100, 0.5) == pytest.approx(100.0 / 3.0)
        assert effective_sample_size(100, 0.0) == pytest.approx(100.0)
        assert effective_sample_size(4904, 0.9) == pytest.approx(4904 * 0.1 / 1.9)
        assert math.isnan(effective_sample_size(100, math.nan))

    def test_ar1_block_length_yardstick(self) -> None:
        assert ar1_block_length(0.0, 4900) == 1.0
        b90 = ar1_block_length(0.9, 4900)
        assert b90 == pytest.approx((6 * 0.81 / (0.01 * 3.61)) ** (1 / 3) * 4900 ** (1 / 3))
        assert ar1_block_length(0.5, 4900) < b90
        assert ar1_block_length(0.99, 4900) == math.ceil(min(3 * math.sqrt(4900), 4900 / 3))
        with pytest.raises(ValueError):
            ar1_block_length(1.0, 100)

    def test_politis_white_wrapper_is_the_rule_the_mcs_uses(self) -> None:
        x = _ar1(np.random.default_rng(3), 0.7, 500)
        assert politis_white_block_length(x) == _ppw_block_length(x)
        with pytest.raises(ValueError):
            politis_white_block_length(np.array([1.0, np.nan, 2.0]))

    def test_hacspec_validation_and_scale(self) -> None:
        with pytest.raises(ValueError):
            HACSpec(bandwidth="newey")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            HACSpec(bandwidth=0.0)
        with pytest.raises(ValueError):
            HACSpec(scale=0.0)
        with pytest.raises(ValueError):
            HACSpec(max_rho=1.0)
        x = _ar1(np.random.default_rng(4), 0.5, 2000)
        once = long_run_variance(x, HACSpec())
        twice = long_run_variance(x, HACSpec(scale=2.0))
        assert twice.bandwidth == pytest.approx(2.0 * once.bandwidth)
        assert twice.n_lags >= once.n_lags


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


class TestLongRunVariance:
    def test_fixed_bandwidth_without_prewhitening_is_j2s_estimator_to_the_bit(self) -> None:
        """``analysis.hac_mean_se`` at truncation lag L is this estimator at
        bandwidth L + 1: same weights ``1 - j/(L+1)``, same arithmetic."""
        rng = np.random.default_rng(5)
        for n in (100, 2791, 4904):
            x = _ar1(rng, 0.8, n)
            lag = analysis.hac_bandwidth(n)
            reference = analysis.hac_mean_se(x)
            assert reference["bandwidth"] == lag
            mine = long_run_variance(x, HACSpec(bandwidth=lag + 1.0, prewhiten=False))
            assert mine.n_lags == lag
            assert math.sqrt(max(mine.omega, 0.0) / n) == reference["se"]
            named = long_run_variance(x, HACSpec(bandwidth="rule_of_thumb", prewhiten=False))
            assert named.bandwidth == rule_of_thumb_bandwidth(n) == lag + 1.0
            assert named.omega == mine.omega

    def test_it_recovers_a_known_long_run_variance_where_the_fixed_rule_does_not(self) -> None:
        """An AR(1) at rho has long-run variance ``1/(1-rho)^2`` (unit shocks).
        The fixed rule loses ~40 % of it at rho = 0.9 (``tests/test_analysis.py::
        TestHAC``); pre-whitening with a data-driven bandwidth must not."""
        rng = np.random.default_rng(2026)
        n = 200_000
        for rho in (0.0, 0.5, 0.9, 0.95):
            x = _ar1(rng, rho, n)
            truth = 1.0 / (1.0 - rho) ** 2
            auto = long_run_variance(x, HACSpec())
            assert auto.omega / truth == pytest.approx(1.0, rel=0.05), rho
            assert not auto.rho_capped
            assert auto.rho1 == pytest.approx(rho, abs=0.01)
            if rho >= 0.9:
                fixed = long_run_variance(
                    x, HACSpec(bandwidth=analysis.hac_bandwidth(n) + 1.0, prewhiten=False)
                )
                assert fixed.omega / truth < 0.8

    def test_the_cap_binds_and_is_reported(self) -> None:
        x = _ar1(np.random.default_rng(6), 0.995, 50_000)
        estimate = long_run_variance(x, HACSpec())
        assert estimate.rho_capped
        assert estimate.rho == MAX_PREWHITEN_RHO
        uncapped = long_run_variance(x, HACSpec(max_rho=0.9999))
        assert not uncapped.rho_capped
        assert uncapped.omega > estimate.omega

    def test_without_prewhitening_a_unit_bandwidth_is_the_plain_variance(self) -> None:
        x = np.random.default_rng(7).standard_normal(300)
        estimate = long_run_variance(x, HACSpec(bandwidth=1.0, prewhiten=False))
        assert estimate.n_lags == 0
        assert estimate.omega == pytest.approx(float(np.var(x)))
        assert estimate.rho == 0.0 and not estimate.prewhiten

    def test_it_refuses_holes_and_degenerate_input(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            long_run_variance(np.array([1.0, np.nan, 2.0]))
        with pytest.raises(ValueError):
            long_run_variance(np.array([1.0]))
        flat = long_run_variance(np.full(20, 3.0))
        assert flat.omega == 0.0 and math.isnan(flat.rho1)


# --------------------------------------------------------------------------
# the DM test in kernel mode
# --------------------------------------------------------------------------


class TestDieboldMarianoHAC:
    def test_lag_and_hac_are_alternatives(self) -> None:
        a, b = np.zeros(20), np.arange(20.0)
        with pytest.raises(ValueError, match="not both"):
            diebold_mariano(a, b, lag=2, hac=HACSpec())

    def test_unit_bandwidth_without_prewhitening_is_the_h1_test_exactly(self) -> None:
        """Bandwidth 1 weights no autocovariance, which is the h = 1 window;
        both must then give the one-sample t statistic, bit for bit."""
        rng = np.random.default_rng(8)
        a, b = rng.normal(size=500), rng.normal(size=500)
        windowed = diebold_mariano(a, b)
        kernel = diebold_mariano(a, b, hac=HACSpec(bandwidth=1.0, prewhiten=False))
        assert kernel.statistic == windowed.statistic
        assert kernel.p_value == windowed.p_value
        assert kernel.hln_factor == windowed.hln_factor == math.sqrt(499 / 500)
        assert math.isnan(windowed.bandwidth) and kernel.bandwidth == 1.0
        assert kernel.lag == 0 and kernel.kernel == "bartlett"

    def test_the_result_carries_the_diagnostics(self) -> None:
        rng = np.random.default_rng(9)
        d = _ar1(rng, 0.8, 3000)
        result = diebold_mariano(d, np.zeros_like(d), hac=HACSpec())
        assert result.prewhiten and 0.7 < result.rho < 0.9
        assert result.rho1 == pytest.approx(autocorrelation(d, 1))
        assert result.n_eff == pytest.approx(effective_sample_size(result.n, result.rho1))
        assert result.bandwidth > 0.0
        assert result.hln_factor == pytest.approx(math.sqrt((result.n - 1) / result.n))
        assert result.config_hash != diebold_mariano(d, np.zeros_like(d)).config_hash
        assert (
            result.config_hash
            != diebold_mariano(d, np.zeros_like(d), hac=HACSpec(scale=2.0)).config_hash
        )

    def test_size_under_a_persistent_null_where_the_windowed_test_fails(self) -> None:
        """Under the null E[d] = 0 with an AR(1) differential at rho = 0.9, a
        5 % test must reject about 5 % of the time. The h = 1 window rejects
        far more often — that is the silent error the kernel mode exists for.
        Both rates are decided over the same draws."""
        rng = np.random.default_rng(20260829)
        reps, n = 300, 2000
        rejected_windowed = 0
        rejected_kernel = 0
        for _ in range(reps):
            d = _ar1(rng, 0.9, n)
            zero = np.zeros(n)
            rejected_windowed += diebold_mariano(d, zero).p_value < 0.05
            rejected_kernel += diebold_mariano(d, zero, hac=HACSpec()).p_value < 0.05
        assert rejected_windowed / reps > 0.30
        assert rejected_kernel / reps < 0.12

    def test_the_matrix_reports_the_bandwidth_per_pair(self) -> None:
        rng = np.random.default_rng(10)
        losses = rng.normal(size=(400, 3))
        losses[:, 0] += _ar1(rng, 0.9, 400)
        matrix = dm_matrix(losses, model_names=["a", "b", "c"], hac=HACSpec())
        assert matrix.hac == HACSpec() and matrix.kernel == "bartlett"
        for frame in (matrix.bandwidth, matrix.rho1, matrix.n_eff):
            assert np.isnan(np.diag(frame.to_numpy())).all()
            off = frame.to_numpy()[~np.eye(3, dtype=bool)]
            assert np.isfinite(off).all()
            assert np.allclose(frame.to_numpy(), frame.to_numpy().T, equal_nan=True)
        assert matrix.rho1.loc["a", "b"] > matrix.rho1.loc["b", "c"]
        assert matrix.n_eff.loc["a", "b"] < matrix.n.loc["a", "b"]
        plain = dm_matrix(losses, model_names=["a", "b", "c"])
        assert plain.hac is None and np.isnan(plain.bandwidth.to_numpy()).all()


# --------------------------------------------------------------------------
# semi-quadratic MCS statistic and the public resampler
# --------------------------------------------------------------------------


class TestSemiQuadratic:
    def test_bootstrap_statistic_by_hand(self) -> None:
        centred = np.array([[1.0, 0.0, -1.0], [2.0, 2.0, 0.0]])
        sd = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        # resample 0: pairs (0,1): 1/1, (0,2): 2/2, (1,2): 1/1 -> 1 + 1 + 1 = 3
        # resample 1: (0,1): 0, (0,2): 2/2, (1,2): 2/1 -> 0 + 1 + 4 = 5
        assert _semi_quadratic_bootstrap(centred, sd).tolist() == [3.0, 5.0]

    def test_two_models_give_the_range_p_values_exactly(self) -> None:
        """With two models T_SQ = t^2 and T_R = |t|, and the bootstrap
        distributions are the same monotone transform of each other, so the
        step p-values coincide."""
        rng = np.random.default_rng(11)
        losses = rng.normal(size=(300, 2))
        losses[:, 1] += 0.1
        r = model_confidence_set(losses, seed=3, n_boot=500, statistic="range")
        q = model_confidence_set(losses, seed=3, n_boot=500, statistic="semi_quadratic")
        assert r.step_p_values == q.step_p_values
        assert r.elimination_order == q.elimination_order

    def test_identical_losses_keep_every_model(self) -> None:
        losses = np.tile(np.random.default_rng(12).normal(size=(100, 1)), (1, 4))
        result = model_confidence_set(losses, seed=1, n_boot=200, statistic="semi_quadratic")
        assert set(result.included) == set(result.models)
        assert all(p == 1.0 for p in result.p_values.values())

    def test_a_dominated_model_goes_first_and_the_run_is_seeded(self) -> None:
        rng = np.random.default_rng(13)
        losses = rng.normal(size=(500, 4))
        losses[:, 2] += 1.5
        first = model_confidence_set(losses, seed=5, n_boot=400, statistic="semi_quadratic")
        second = model_confidence_set(losses, seed=5, n_boot=400, statistic="semi_quadratic")
        assert first.elimination_order[0] == "model_2"
        assert first.p_values["model_2"] < 0.05
        assert first.p_values == second.p_values and first.config_hash == second.config_hash
        assert first.statistic == "semi_quadratic"
        with pytest.raises(ValueError, match="semi_quadratic"):
            model_confidence_set(losses, seed=1, statistic="quadratic")  # type: ignore[arg-type]

    def test_public_resampler_matches_explicit_indices(self) -> None:
        rng = np.random.default_rng(14)
        values = rng.normal(size=(97, 3))
        means = bootstrap_column_means(values, block_length=7, n_boot=50, seed=21)
        indices = moving_block_indices(97, 7, 50, 21)
        explicit = values[indices].mean(axis=1)
        assert np.allclose(means, explicit, atol=1e-12)
        with pytest.raises(ValueError, match="finite"):
            bootstrap_column_means(np.array([[1.0], [np.nan]]), 1, 5, 0)
        with pytest.raises(ValueError, match="2-D"):
            bootstrap_column_means(np.arange(5.0), 1, 5, 0)
