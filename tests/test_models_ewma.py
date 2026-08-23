"""RiskMetrics EWMA: recursion correctness and horizon-flat forecast."""

import numpy as np
import pytest

from volbench.models import EWMA


class TestEWMA:
    def test_matches_hand_computed_recursion_on_five_point_series(self) -> None:
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        lam = 0.94
        expected_var = r[0] ** 2
        for x in r[1:]:
            expected_var = lam * expected_var + (1.0 - lam) * x**2

        fitted = EWMA(lambda_=lam).fit(r)
        assert fitted.sigma2 == pytest.approx(expected_var)
        assert fitted.predict(1).sigma == pytest.approx(np.sqrt(expected_var))

    def test_default_lambda_is_riskmetrics_0_94(self) -> None:
        assert EWMA().lambda_ == pytest.approx(0.94)

    def test_forecast_flat_across_horizon(self) -> None:
        rng = np.random.default_rng(2)
        fitted = EWMA().fit(rng.normal(0.0, 0.01, 100))
        assert fitted.predict(1).sigma == fitted.predict(20).sigma

    def test_different_lambda_gives_different_forecast(self) -> None:
        r = np.array([0.01, -0.03, 0.02, -0.015, 0.025, 0.01, -0.02])
        low = EWMA(lambda_=0.80).fit(r).sigma2
        high = EWMA(lambda_=0.97).fit(r).sigma2
        assert low != pytest.approx(high)

    def test_invalid_lambda_rejected(self) -> None:
        with pytest.raises(ValueError):
            EWMA(lambda_=0.0)
        with pytest.raises(ValueError):
            EWMA(lambda_=1.0)

    def test_degenerate_flat_window_floors_variance_instead_of_raising(self) -> None:
        fitted = EWMA().fit(np.zeros(50))
        assert fitted.predict(1).sigma > 0.0

    def test_spec_includes_lambda_and_is_stable(self) -> None:
        assert EWMA(lambda_=0.9).spec() == {"model": "ewma", "lambda": 0.9}
        assert EWMA(lambda_=0.9).spec() == EWMA(lambda_=0.9).spec()
        assert EWMA(lambda_=0.9).spec() != EWMA(lambda_=0.94).spec()
