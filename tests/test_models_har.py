"""HAR-RV (Corsi 2009): coefficient recovery and the RV-series input contract."""

import numpy as np
import pytest

from volbench.models import HAR
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


class TestHARCoefficientRecovery:
    def test_recovers_known_coefficients_on_synthetic_design(self) -> None:
        beta_true = np.array([-0.5, 0.5, 0.3, 0.15])
        rv = _synthetic_har_series(beta_true, n=300, seed=0)

        fitted = HAR().fit(rv)

        assert fitted.beta == pytest.approx(beta_true, abs=1e-6)
        assert fitted.resid_var == pytest.approx(0.0, abs=1e-12)


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


class TestHARSpec:
    def test_spec_stable_across_identical_constructions(self) -> None:
        assert HAR().spec() == HAR().spec()

    def test_spec_reports_windows(self) -> None:
        assert HAR().spec() == {
            "model": "har_rv",
            "d_window": 1,
            "w_window": 5,
            "m_window": 22,
        }
