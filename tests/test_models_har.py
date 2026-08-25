"""HAR-RV (Corsi 2009): coefficient recovery, the RV input contract, retransformation.

The retransformation block is new at v0.5.0 (D-030): HAR moved off its own
Gaussian ``exp(mu + sigma^2/2)`` onto the shared ``volbench.models._rv``
correction with Duan smearing as the default, which is what every other
log-RV model in the package already used (D-025). Both arms are pinned here —
the ``gaussian`` arm because it must still reproduce the pre-0.5.0 number
exactly, the ``smearing`` arm because it is now the default.
"""

import math

import numpy as np
import pytest

from volbench.models import HAR
from volbench.models._rv import gaussian_factor, smearing_factor
from volbench.models.har import _M_WINDOW, _har_features


def _synthetic_har_series(beta: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Build an RV series whose own strict-window HAR design matrix satisfies
    log(RV_{t+1}) = x_t @ beta exactly (no noise), by constructing it
    recursively: each new RV is generated from the features of the RV
    values already placed, exactly mirroring what `_design_matrix` computes.
    """
    rng = np.random.default_rng(seed)
    rv = np.empty(n)
    rv[:_M_WINDOW] = np.exp(rng.normal(-8.0, 0.3, _M_WINDOW))
    for t in range(_M_WINDOW - 1, n - 1):
        d, w, m = _har_features(rv, t)
        x = np.array([1.0, np.log(d), np.log(w), np.log(m)])
        rv[t + 1] = np.exp(float(x @ beta))
    return rv


def _noisy_rv(seed: int, n: int = 400) -> np.ndarray:
    """A right-skewed RV series with real log-space residuals, so the two
    retransformation factors genuinely differ (they coincide only when the
    residuals are exactly Gaussian)."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(-9.0, 0.8, n))


class TestHARCoefficientRecovery:
    def test_recovers_known_coefficients_on_synthetic_design(self) -> None:
        beta_true = np.array([-0.5, 0.5, 0.3, 0.15])
        rv = _synthetic_har_series(beta_true, n=300, seed=0)

        fitted = HAR().fit(rv)

        assert fitted.beta == pytest.approx(beta_true, abs=1e-6)
        assert fitted.resid_var == pytest.approx(0.0, abs=1e-12)
        # A noiseless fit has no correction to make, under either arm.
        assert fitted.smear == pytest.approx(1.0, abs=1e-9)


class TestHARInputContract:
    def test_fit_rejects_nonpositive_values(self) -> None:
        rv = np.abs(np.random.default_rng(1).normal(0.0, 1e-4, 60)) + 1e-6
        rv[10] = 0.0
        with pytest.raises(ValueError):
            HAR().fit(rv)

    def test_fit_rejects_too_short_series(self) -> None:
        rv = np.abs(np.random.default_rng(1).normal(0.0, 1e-4, _M_WINDOW)) + 1e-6
        with pytest.raises(ValueError):
            HAR().fit(rv)

    def test_an_unknown_retransform_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="retransform"):
            HAR(retransform="lognormal")  # type: ignore[arg-type]


class TestHARPredict:
    def test_predict_returns_positive_sigma_and_is_zero_mean(self) -> None:
        rng = np.random.default_rng(6)
        rv = np.abs(rng.normal(0.0, 1e-4, 200)) + 1e-6
        fitted = HAR().fit(rv)
        dist = fitted.predict(1)
        assert dist.mu == 0.0
        assert dist.sigma > 0.0

    def test_multistep_predict_uses_recursive_buffer(self) -> None:
        rng = np.random.default_rng(7)
        rv = np.abs(rng.normal(0.0, 1e-4, 200)) + 1e-6
        fitted = HAR().fit(rv)
        d1 = fitted.predict(1)
        d5 = fitted.predict(5)
        assert np.isfinite(d1.sigma)
        assert np.isfinite(d5.sigma)

    def test_predict_rejects_nonpositive_horizon(self) -> None:
        rng = np.random.default_rng(8)
        rv = np.abs(rng.normal(0.0, 1e-4, 200)) + 1e-6
        fitted = HAR().fit(rv)
        with pytest.raises(ValueError):
            fitted.predict(0)


class TestRetransformation:
    """D-030: HAR goes through ``volbench.models._rv`` like every other log-RV
    model, smearing by default, the Gaussian arm kept for like-for-like."""

    def test_the_default_is_smearing(self) -> None:
        assert HAR().retransform == "smearing"

    def test_each_arm_applies_exactly_its_own_factor(self) -> None:
        rv = _noisy_rv(seed=11)
        smearing = HAR().fit(rv)
        gaussian = HAR(retransform="gaussian").fit(rv)

        # Same OLS underneath: only the factor differs.
        assert np.array_equal(smearing.beta, gaussian.beta)
        assert smearing.smear == gaussian.smear
        assert smearing.resid_var == gaussian.resid_var

        ratio = smearing.predict(1).sigma ** 2 / gaussian.predict(1).sigma ** 2
        expected = smearing.smear / gaussian_factor(gaussian.resid_var)
        assert ratio == pytest.approx(expected, rel=1e-12)

    def test_the_gaussian_arm_is_exactly_the_pre_0_5_0_formula(self) -> None:
        """The like-for-like arm must reproduce ``exp(y_hat + resid_var/2)``,
        the only correction HAR had up to v0.4.0 — that equality is what makes
        the two arms comparable and the D-030 measurement meaningful."""
        rv = _noisy_rv(seed=12)
        fitted = HAR(retransform="gaussian").fit(rv)
        d, w, m = _har_features(fitted.buffer, fitted.buffer.size - 1)
        y_hat = float(
            fitted.beta[0]
            + fitted.beta[1] * math.log(d)
            + fitted.beta[2] * math.log(w)
            + fitted.beta[3] * math.log(m)
        )
        expected = math.exp(y_hat + 0.5 * fitted.resid_var)
        assert fitted.predict(1).sigma ** 2 == pytest.approx(expected, rel=1e-12)

    def test_the_factors_are_the_shared_ones_computed_on_the_fit_window(self) -> None:
        rv = _noisy_rv(seed=13)
        fitted = HAR().fit(rv)
        from volbench.models.har import _design_matrix

        x, y = _design_matrix(rv)
        resid = y - x @ fitted.beta
        assert fitted.smear == pytest.approx(smearing_factor(resid), rel=1e-12)

    def test_smearing_is_at_least_the_gaussian_median_correction(self) -> None:
        """Jensen: ``mean(exp(e)) >= exp(mean(e)) == 1`` on OLS residuals, so
        the correction is never a *de*-flation."""
        for seed in (14, 15, 16):
            assert HAR().fit(_noisy_rv(seed=seed)).smear >= 1.0

    def test_update_re_estimates_neither_factor(self) -> None:
        rv = _noisy_rv(seed=17, n=500)
        fitted = HAR().fit(rv[:400])
        later = fitted.update(rv[100:400])
        assert later.smear == fitted.smear
        assert later.resid_var == fitted.resid_var
        assert np.array_equal(later.beta, fitted.beta)


class TestHARSpec:
    def test_spec_stable_across_identical_constructions(self) -> None:
        assert HAR().spec() == HAR().spec()

    def test_spec_reports_windows_and_the_retransformation(self) -> None:
        assert HAR().spec() == {
            "model": "har_rv",
            "d_window": 1,
            "w_window": 5,
            "m_window": 22,
            "retransform": "smearing",
        }

    def test_the_two_arms_cannot_share_a_name_or_a_spec(self) -> None:
        """D-025's rule: the arm is in both, so no config hash can hold both."""
        assert HAR().name == "har_rv-smearing"
        assert HAR(retransform="gaussian").name == "har_rv-gaussian"
        assert HAR().spec() != HAR(retransform="gaussian").spec()

    def test_the_fitted_model_reports_its_config(self) -> None:
        fitted = HAR(retransform="gaussian").fit(_noisy_rv(seed=18))
        assert fitted.name == "har_rv-gaussian"
        assert fitted.spec() == HAR(retransform="gaussian").spec()
