"""GARCH(1,1)/GJR-GARCH via `arch`: parameter recovery, dist types, non-convergence fallback."""

import numpy as np
import pytest

from volbench.dist import Normal, StudentT
from volbench.models import GARCH, gjr_garch
from volbench.models.ewma import FittedEWMA
from volbench.models.garch import FIT_TOL, NU_BOUNDS


def _simulate_garch11(
    omega: float, alpha: float, beta: float, n: int, seed: int, burn: int = 500
) -> np.ndarray:
    """GARCH(1,1) DGP with known parameters, for the recovery test below."""
    rng = np.random.default_rng(seed)
    total = n + burn
    eps = np.zeros(total)
    sigma2 = np.zeros(total)
    sigma2[0] = omega / (1.0 - alpha - beta)
    eps[0] = rng.standard_normal() * np.sqrt(sigma2[0])
    for t in range(1, total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    return eps[burn:]


class TestGARCHParameterRecovery:
    def test_recovers_known_garch11_params_within_tolerance(self) -> None:
        omega, alpha, beta = 0.05, 0.10, 0.85
        r = _simulate_garch11(omega, alpha, beta, n=4000, seed=42)

        fitted = GARCH(dist="normal").fit(r)
        assert not fitted.fallback
        assert fitted.result is not None

        omega_hat = float(fitted.result.params["omega"]) / (fitted.scale**2)
        alpha_hat = float(fitted.result.params["alpha[1]"])
        beta_hat = float(fitted.result.params["beta[1]"])

        assert omega_hat == pytest.approx(omega, abs=0.03)
        assert alpha_hat == pytest.approx(alpha, abs=0.04)
        assert beta_hat == pytest.approx(beta, abs=0.05)


class TestGARCHDistributionTypes:
    def test_normal_innovations_return_normal_distribution(self) -> None:
        rng = np.random.default_rng(3)
        fitted = GARCH(dist="normal").fit(rng.normal(0.0, 0.01, 500))
        assert isinstance(fitted.predict(1), Normal)

    def test_studentt_innovations_return_a_parametric_student_t(self) -> None:
        """Was a 199-point QuantileGrid until m2/evaluator-hardening; the
        grid's truncated tails biased QLIKE (docs/M1_REPORT.md §4.2)."""
        rng = np.random.default_rng(4)
        r = rng.standard_t(df=6, size=800) * 0.01
        fitted = GARCH(dist="studentst").fit(r)
        assert not fitted.fallback
        assert fitted.result is not None
        dist = fitted.predict(1)
        assert isinstance(dist, StudentT)
        assert dist.loc == 0.0
        assert dist.df == float(fitted.result.params["nu"])
        # The object's variance is exactly arch's conditional variance, in
        # the caller's units (rescale undone), not the t's scale parameter.
        forecast = fitted.result.forecast(horizon=1, reindex=False)
        sigma2 = float(forecast.variance.values[-1, 0]) / fitted.scale**2
        assert dist.variance() == pytest.approx(sigma2, rel=1e-12)

    def test_studentt_forecast_is_deterministic_without_an_rng(self) -> None:
        rng = np.random.default_rng(4)
        r = rng.standard_t(df=6, size=800) * 0.01
        first = GARCH(dist="studentst").fit(r).predict(1)
        second = GARCH(dist="studentst").fit(r).predict(1)
        assert first == second  # value equality on (loc, scale, df)
        assert first.crps(0.012) == second.crps(0.012)

    def test_gjr_garch_fits_and_predicts(self) -> None:
        rng = np.random.default_rng(5)
        fitted = gjr_garch(dist="normal").fit(rng.normal(0.0, 0.01, 500))
        dist = fitted.predict(1)
        assert isinstance(dist, Normal)
        assert dist.sigma > 0.0


class TestGARCHNonConvergenceFallback:
    def test_degenerate_series_falls_back_to_ewma_and_still_forecasts(self) -> None:
        fitted = GARCH(dist="normal").fit(np.zeros(300))
        assert fitted.fallback is True
        assert fitted.result is None
        assert isinstance(fitted.fallback_fit, FittedEWMA)

        dist = fitted.predict(1)
        assert isinstance(dist, Normal)
        assert dist.sigma > 0.0
        assert np.isfinite(dist.sigma)

    def test_fallback_spec_still_reports_hyperparameters(self) -> None:
        fitted = GARCH(dist="normal", fallback_lambda=0.9).fit(np.zeros(300))
        assert fitted.spec()["fallback_lambda"] == 0.9

    def test_a_fallback_says_so_and_names_the_estimator_that_ran(self) -> None:
        """D-032. Falling back is the right behaviour and was invisible: a cell
        that ran EWMA on some of its origins scored like one that ran none."""
        fitted = GARCH(dist="normal").fit(np.zeros(300))
        diagnostics = fitted.fit_diagnostics()
        assert diagnostics.fallback == "ewma"
        assert diagnostics.converged is False
        assert diagnostics.status().startswith("fallback=ewma")

    def test_a_clean_fit_says_ok_and_carries_the_optimizer_flag(self) -> None:
        rng = np.random.default_rng(4)
        fitted = GARCH(dist="normal").fit(rng.standard_normal(400) * 0.01)
        status = fitted.fit_diagnostics().status()
        assert status.startswith("ok|flag=0")

    def test_the_status_survives_re_conditioning(self) -> None:
        """``update`` runs no optimizer, so a window re-filtered at the
        parameters of a fit that fell back is still a fallback forecast."""
        train = np.zeros(300)
        fitted = GARCH(dist="normal").fit(train)
        assert fitted.update(train).fit_diagnostics().status() == (
            fitted.fit_diagnostics().status()
        )


class TestNuIsBounded:
    """D-032 item 3. ``arch`` bounds ``nu`` at 500; a 500-observation window
    carries no information about tail thickness up there, so the likelihood is
    flat along it — and a flat direction is what let a last-ulp BLAS difference
    move SLSQP into a different local optimum.
    """

    def test_gaussian_data_pins_nu_at_the_upper_bound_instead_of_wandering(self) -> None:
        rng = np.random.default_rng(3)
        fitted = GARCH(dist="studentst").fit(rng.standard_normal(500) * 0.01)
        assert fitted.result is not None
        nu = float(fitted.result.params["nu"])
        assert nu <= NU_BOUNDS[1] + 1e-9
        # Not merely "inside the bound": on Gaussian data the bound is where
        # the answer lands, which is the honest reading of "nu is not
        # identified here" rather than a number picked out of a flat region.
        assert nu > NU_BOUNDS[1] - 1e-6
        assert "nu_at_bound" in fitted.fit_diagnostics().status()

    def test_genuinely_fat_tails_stay_inside_the_bound(self) -> None:
        """The bound must not turn every Student-t into a Gaussian. Real tail
        thickness is estimable and has to survive."""
        rng = np.random.default_rng(5)
        heavy = rng.standard_t(4.0, size=1500) * 0.01
        fitted = GARCH(dist="studentst").fit(heavy)
        assert fitted.result is not None
        nu = float(fitted.result.params["nu"])
        assert NU_BOUNDS[0] < nu < NU_BOUNDS[1] - 1.0, nu
        assert "nu_at_bound" not in fitted.fit_diagnostics().status()

    def test_the_bounds_are_hashed_only_where_they_bind(self) -> None:
        """A normal-innovations GARCH has no degrees of freedom to bound, so
        recording them there would split two identical experiments over a
        setting neither used."""
        assert "nu_bounds" in GARCH(dist="studentst").spec()
        assert "nu_bounds" not in GARCH(dist="normal").spec()
        assert GARCH(dist="studentst").spec() != GARCH(
            dist="studentst", nu_bounds=(2.1, 40.0)
        ).spec()

    def test_the_tolerance_is_hashed_for_both_distributions(self) -> None:
        """It changes the optimizer's stopping point whatever the innovations."""
        for dist in ("normal", "studentst"):
            assert GARCH(dist=dist).spec()["fit_tol"] == FIT_TOL  # type: ignore[arg-type]
            assert GARCH(dist=dist).spec() != GARCH(  # type: ignore[arg-type]
                dist=dist, fit_tol=1e-8  # type: ignore[arg-type]
            ).spec()

    def test_bounds_that_admit_an_undefined_variance_are_refused(self) -> None:
        """At nu <= 2 the predictive variance nu/(nu-2) does not exist, and a
        model whose variance forecast is undefined is not a volatility model."""
        with pytest.raises(ValueError, match="nu_bounds"):
            GARCH(dist="studentst", nu_bounds=(2.0, 50.0))
        with pytest.raises(ValueError, match="nu_bounds"):
            GARCH(dist="studentst", nu_bounds=(30.0, 10.0))
        with pytest.raises(ValueError, match="fit_tol"):
            GARCH(dist="studentst", fit_tol=0.0)


class TestGARCHSpec:
    def test_spec_stable_across_identical_constructions(self) -> None:
        assert GARCH(o=0, dist="normal").spec() == GARCH(o=0, dist="normal").spec()

    def test_spec_differs_by_variant_and_dist(self) -> None:
        base = GARCH(o=0, dist="normal").spec()
        assert base != GARCH(o=1, dist="normal").spec()
        assert base != GARCH(o=0, dist="studentst").spec()
        assert base != GARCH(o=0, dist="normal", fallback_lambda=0.9).spec()
        assert base != GARCH(o=0, dist="normal", fit_tol=1e-8).spec()

    def test_invalid_construction_rejected(self) -> None:
        with pytest.raises(ValueError):
            GARCH(o=2)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            GARCH(dist="gaussian")  # type: ignore[arg-type]
