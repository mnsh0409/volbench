"""Random-walk volatility: forecast sigma = trailing RMS return."""

import numpy as np
import pytest

from volbench.models import NaiveVol


class TestNaiveVol:
    def test_fit_predict_matches_rms_return(self) -> None:
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        fitted = NaiveVol().fit(r)
        expected_sigma = float(np.sqrt(np.mean(r * r)))
        dist = fitted.predict(1)
        assert dist.sigma == pytest.approx(expected_sigma)
        assert dist.mu == 0.0

    def test_forecast_flat_across_horizon(self) -> None:
        rng = np.random.default_rng(1)
        fitted = NaiveVol().fit(rng.normal(0.0, 0.01, 100))
        assert fitted.predict(1).sigma == fitted.predict(10).sigma

    def test_predict_rejects_nonpositive_horizon(self) -> None:
        fitted = NaiveVol().fit(np.array([0.01, -0.02, 0.03]))
        with pytest.raises(ValueError):
            fitted.predict(0)

    def test_fit_rejects_too_short_series(self) -> None:
        with pytest.raises(ValueError):
            NaiveVol().fit(np.array([0.01]))

    def test_degenerate_flat_window_floors_sigma_instead_of_raising(self) -> None:
        fitted = NaiveVol().fit(np.zeros(50))
        dist = fitted.predict(1)
        assert dist.sigma > 0.0

    def test_spec_stable_and_differs_by_name_only(self) -> None:
        assert NaiveVol().spec() == NaiveVol().spec()
        assert NaiveVol().name == "naive_rw_vol"
