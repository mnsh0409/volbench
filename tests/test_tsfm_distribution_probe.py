"""The grid-law arithmetic behind docs/P3_METRIC_TARGETS.md and the variance audit.

These are the decidable parts of the measurement: the moments of a flat-tailed
quantile grid, the tail closures that estimate what the flat tails cost, and
the scale mixture that says what the ``Normal(0, sqrt(vhat))`` reduction leaves
out on the return axis. Each is checked against an independent representation —
a closed form or a fine quadrature — never against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from volbench.benchmarks.tsfm_distribution_probe import (
    grid_law_moments,
    mixture_excess_kurtosis,
    mixture_return_quantile,
    normal_var_es,
    tail_closed_mean,
)
from volbench.models.tsfm_common import quantile_grid_mean

BOLT_TAUS = np.round(np.arange(0.1, 0.95, 0.1), 10)


def _lognormal_grid(mu: float, sigma: float, taus: np.ndarray) -> np.ndarray:
    return np.exp(mu + sigma * stats.norm.ppf(taus))


class TestGridLawMoments:
    def test_mean_is_the_adapters_own_vhat(self) -> None:
        """The first moment must be exactly what the adapter scores, or this
        probe would be measuring a different object than the grid did."""
        grid = _lognormal_grid(-7.0, 0.6, BOLT_TAUS)
        assert grid_law_moments(BOLT_TAUS, grid).mean == pytest.approx(
            quantile_grid_mean(BOLT_TAUS, grid), rel=1e-15
        )

    def test_matches_a_fine_quadrature_of_the_same_law(self) -> None:
        """Independent representation: the closed-form sums against the
        empirical moments of the law's own quantile function, sampled densely."""
        grid = _lognormal_grid(-7.0, 0.8, BOLT_TAUS)
        u = (np.arange(2_000_000) + 0.5) / 2_000_000
        draws = np.interp(u, BOLT_TAUS, grid)  # flat outside, as the law is
        got = grid_law_moments(BOLT_TAUS, grid)
        assert got.mean == pytest.approx(float(np.mean(draws)), rel=1e-6)
        assert got.variance == pytest.approx(float(np.var(draws)), rel=1e-5)
        assert got.skewness == pytest.approx(float(stats.skew(draws)), rel=1e-4)
        assert got.excess_kurtosis == pytest.approx(float(stats.kurtosis(draws)), rel=1e-4)

    def test_a_degenerate_grid_has_no_shape(self) -> None:
        """A collapsed forecast is a real state (Moirai at raw units), and it
        must report zero variance rather than dividing by it."""
        got = grid_law_moments(BOLT_TAUS, np.full(BOLT_TAUS.size, 3.0))
        assert got.mean == pytest.approx(3.0)
        assert got.variance == 0.0
        assert math.isnan(got.skewness) and math.isnan(got.excess_kurtosis)

    def test_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError):
            grid_law_moments(np.asarray([0.1, 0.9]), np.asarray([1.0, 2.0, 3.0]))


class TestTailClosures:
    def test_the_closure_removes_most_but_not_all_of_the_flat_tail_gap(self) -> None:
        """Two error sources, and the audit must not conflate them. Closing the
        tails removes the large downward one; what remains is the *interior*
        trapezoid, which overstates the integral of a convex quantile function.
        So the closure lands slightly ABOVE the truth, and the flat-tailed grid
        mean well below it."""
        mu, sigma = -7.0, 0.7
        grid = _lognormal_grid(mu, sigma, BOLT_TAUS)
        truth = math.exp(mu + 0.5 * sigma**2)
        flat = quantile_grid_mean(BOLT_TAUS, grid)
        closed = tail_closed_mean(BOLT_TAUS, grid, "lognormal")
        assert flat < truth < closed
        assert abs(closed / truth - 1.0) < 0.01  # interior trapezoid alone
        assert flat / truth < 0.95  # flat tails cost far more

    def test_loglinear_closure_agrees_when_the_grid_is_exactly_lognormal(self) -> None:
        """Fitted on the outer pairs only, it must still land on the same
        answer when the grid has no curvature in log-z space to miss."""
        grid = _lognormal_grid(-7.0, 0.7, BOLT_TAUS)
        assert tail_closed_mean(BOLT_TAUS, grid, "loglinear") == pytest.approx(
            tail_closed_mean(BOLT_TAUS, grid, "lognormal"), rel=1e-6
        )

    def test_a_clipped_quantile_makes_the_closure_undefined_not_wrong(self) -> None:
        """chronos clips negative RV quantiles to zero; a lognormal cannot
        describe a zero quantile, and NaN is the honest answer."""
        grid = _lognormal_grid(-7.0, 0.7, BOLT_TAUS)
        grid[0] = 0.0
        assert math.isnan(tail_closed_mean(BOLT_TAUS, grid, "lognormal"))

    def test_rejects_an_unknown_closure(self) -> None:
        with pytest.raises(ValueError):
            tail_closed_mean(BOLT_TAUS, _lognormal_grid(-7.0, 0.7, BOLT_TAUS), "gpd")


class TestReturnAxis:
    def test_excess_kurtosis_is_three_times_the_squared_cv(self) -> None:
        assert mixture_excess_kurtosis(4.0, 2.0) == pytest.approx(3.0)
        assert mixture_excess_kurtosis(0.0, 2.0) == 0.0

    def test_a_degenerate_grid_reduces_to_the_scored_normal(self) -> None:
        """With no RV uncertainty the mixture IS Normal(0, sqrt(vhat)), so the
        comparison the audit draws must vanish exactly there."""
        grid = np.full(BOLT_TAUS.size, 1e-4)
        for alpha in (0.01, 0.025, 0.05):
            want, _ = normal_var_es(1e-4, alpha)
            assert mixture_return_quantile(BOLT_TAUS, grid, alpha) == pytest.approx(want, rel=1e-6)

    def test_the_var_error_changes_sign_inside_the_evaluated_levels(self) -> None:
        """The property that makes "the reduction understates tail risk" too
        simple a sentence to write. Matching the variance of a symmetric fat-
        tailed law with a Normal moves mass from the shoulders to the far tail,
        so the two quantile functions CROSS: the mixture is more extreme deep in
        the tail and less extreme nearer the shoulder. On this grid the crossing
        falls between 0.025 and 0.05 — inside the three levels volbench scores."""
        grid = _lognormal_grid(-7.0, 0.8, BOLT_TAUS)
        vhat = quantile_grid_mean(BOLT_TAUS, grid)

        def ratio(alpha: float) -> float:
            scored, _ = normal_var_es(vhat, alpha)
            return mixture_return_quantile(BOLT_TAUS, grid, alpha) / scored

        assert ratio(0.001) > ratio(0.01) > ratio(0.025) > 1.0 > ratio(0.05) > ratio(0.10)

    def test_mixture_variance_equals_the_grid_mean(self) -> None:
        """The reduction is variance-preserving by construction — that is why
        the difference is a shape difference and not a level one."""
        grid = _lognormal_grid(-7.0, 0.8, BOLT_TAUS)
        u = (np.arange(200_001) + 0.5) / 200_001
        v = np.interp(u, BOLT_TAUS, grid)
        assert float(np.mean(v)) == pytest.approx(quantile_grid_mean(BOLT_TAUS, grid), rel=1e-4)

    def test_rejects_a_level_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            mixture_return_quantile(BOLT_TAUS, _lognormal_grid(-7.0, 0.7, BOLT_TAUS), 1.0)


class TestNormalVarEs:
    def test_matches_the_stored_columns_closed_form(self) -> None:
        """What the store holds for the 12 Normal-emitting configs, so the
        audit can rebuild a scored VaR without re-running a cell."""
        var, es = normal_var_es(0.0004, 0.05)
        assert var == pytest.approx(0.02 * float(stats.norm.ppf(0.05)))
        assert es == pytest.approx(-0.02 * float(stats.norm.pdf(stats.norm.ppf(0.05))) / 0.05)
        assert es < var < 0.0
