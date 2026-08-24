"""M1 report §4.2, closed at the root: a perfect Student-t forecast scores QLIKE 0.

The report's falsification table measured, for a *perfectly specified*
Student-t forecast scored against a *perfect* proxy — where QLIKE should be
exactly 0 — floors of 0.0407 / 0.0035 / 0.0010 / 0.0003 at nu = 3 / 5 / 8 /
20. The cause was two defensible decisions composing badly: GARCH emitted a
199-point quantile grid over tau in [0.005, 0.995], and ``forecast_moments``
took the grid's flat tails literally, understating the variance by 24% at
nu=3. The penalty grew exactly where the Student-t specification is supposed
to win.

This file reproduces that table against the old path (so the numbers stay
falsifiable, not folklore) and pins the new one below 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from volbench.dist import Distribution, Normal, StudentT
from volbench.evaluate import DEFAULT_LEVELS, forecast_moments
from volbench.metrics import qlike
from volbench.models import EWMA, GARCH, NaiveVol, gjr_garch

SIGMA2 = 1e-4  # a daily equity-index variance, ~1.6% annualized-vol-equivalent... in daily units

#: docs/M1_REPORT.md §4.2, measured under v0.1.0-m1.
REPORT_FLOORS = {3.0: 0.0407, 5.0: 0.0035, 8.0: 0.0010, 20.0: 0.0003}
REPORT_VARIANCE_UNDERSTATEMENT = {3.0: 0.238, 5.0: 0.079, 8.0: 0.043, 20.0: 0.024}


def old_grid_path(sigma2: float, nu: float) -> Distribution:
    """The v0.1.0-m1 construction, verbatim: what `models/garch.py` emitted."""
    taus = np.linspace(0.005, 0.995, 199)
    scale = math.sqrt(sigma2 * (nu - 2.0) / nu)
    return Distribution.from_quantiles(taus, scale * stats.t.ppf(taus, df=nu))


@pytest.mark.parametrize("nu", sorted(REPORT_FLOORS))
def test_a_perfect_student_t_forecast_now_scores_qlike_zero(nu: float) -> None:
    forecast = StudentT.from_variance(0.0, SIGMA2, nu)
    mean, variance = forecast_moments(forecast)
    assert mean == 0.0
    assert qlike(variance, SIGMA2) < 1e-6


@pytest.mark.parametrize("nu", sorted(REPORT_FLOORS))
def test_the_report_floors_are_reproduced_on_the_old_path(nu: float) -> None:
    """The fix is only real if the bug still is. The grid path must still
    show the report's floor — otherwise the new number proves nothing."""
    _, grid_variance = forecast_moments(old_grid_path(SIGMA2, nu))
    assert 1.0 - grid_variance / SIGMA2 == pytest.approx(
        REPORT_VARIANCE_UNDERSTATEMENT[nu], abs=0.005
    )
    assert qlike(grid_variance, SIGMA2) == pytest.approx(REPORT_FLOORS[nu], rel=0.05)
    assert qlike(grid_variance, SIGMA2) > 100 * 1e-6  # and it is not "almost zero" either


def test_a_fitted_student_t_garch_scores_qlike_zero_against_its_own_variance() -> None:
    """End to end through the model: the object GARCH hands back has exactly
    arch's conditional variance as its variance forecast."""
    rng = np.random.default_rng(11)
    r = rng.standard_t(df=6, size=1000) * 0.01
    fitted = GARCH(dist="studentst").fit(r)
    assert not fitted.fallback and fitted.result is not None
    sigma2 = float(fitted.result.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    sigma2 /= fitted.scale**2

    _, variance = forecast_moments(fitted.predict(1))
    assert qlike(variance, sigma2) < 1e-12


class TestNormalInnovationConfigsAreUnchanged:
    """Nothing on the Gaussian path may move (`make reproduce` byte-identity
    is the 200-origin version of this; these are the unit-level pins)."""

    def test_forecast_moments_of_a_normal_are_the_same_expression_as_before(self) -> None:
        # v0.1.0-m1 computed `dist.mu, dist.sigma * dist.sigma`; bit-for-bit.
        for sigma in (0.007, 0.01, 0.0123456789):
            assert forecast_moments(Normal(0.0, sigma)) == (0.0, sigma * sigma)

    def test_normal_innovation_garch_still_returns_a_normal_and_scores_as_one(self) -> None:
        rng = np.random.default_rng(3)
        r = rng.normal(0.0, 0.01, 600)
        for model in (GARCH(dist="normal"), gjr_garch(dist="normal"), EWMA(), NaiveVol()):
            dist = model.fit(r).predict(1)
            assert type(dist) is Normal
            reference = Normal(mu=dist.mu, sigma=dist.sigma)
            for y in (-0.03, 0.0, 0.012):
                assert dist.crps(y) == reference.crps(y)
                assert dist.log_score(y) == reference.log_score(y)
                for level in DEFAULT_LEVELS:
                    assert dist.pinball(y, level) == reference.pinball(y, level)

    def test_gaussian_closed_forms_are_the_known_values(self) -> None:
        # CRPS(N(0,1), 0) = sqrt(2/pi) - 1/sqrt(pi); pinball at the median is |y|/2.
        assert Normal(0.0, 1.0).crps(0.0) == pytest.approx(
            math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi), abs=1e-12
        )
        assert Normal(0.0, 1.0).pinball(1.0, 0.5) == pytest.approx(0.5)
