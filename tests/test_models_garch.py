"""GARCH(1,1)/GJR-GARCH via `arch`: parameter recovery, dist types, non-convergence fallback."""

import numpy as np
import pytest

from volbench.dist import Normal, StudentT
from volbench.models import GARCH, gjr_garch
from volbench.models.ewma import FittedEWMA
from volbench.models.garch import FIT_TOL, NU_BOUNDS, TerminalFit, _BoundedStudentsT


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


class TestTerminalFit:
    """A fit that fell back used to discard everything but its exit flag.

    The forensic question "which parameter was at which bound when SLSQP
    stopped?" is not answerable from a flag, and it is the question a flat
    likelihood surface and a coding defect answer differently. So the
    optimizer's terminal state is retained on every scheduled fit — which
    must cost no number, no ``spec()`` field and no config hash.
    """

    @staticmethod
    def _clean() -> np.ndarray:
        rng = np.random.default_rng(4)
        return rng.standard_normal(400) * 0.01

    def test_a_converged_fit_keeps_the_optimizers_own_numbers(self) -> None:
        fitted = GARCH(dist="normal").fit(self._clean())
        terminal = fitted.terminal
        assert terminal is not None
        assert fitted.result is not None
        assert terminal.convergence_flag == 0
        assert dict(terminal.params) == pytest.approx(dict(fitted.result.params))
        assert terminal.loglikelihood == pytest.approx(float(fitted.result.loglikelihood))
        assert terminal.scale == pytest.approx(float(fitted.result.scale))
        assert terminal.iterations > 0 and terminal.function_evals > 0
        assert "success" in terminal.message.lower()

    def test_a_fit_that_did_not_converge_keeps_them_too(self) -> None:
        """The whole point: this one falls back, so ``result`` is None and the
        terminal state is the only surviving evidence about the optimizer."""
        fitted = GARCH(dist="normal").fit(np.zeros(300))
        assert fitted.fallback is True
        assert fitted.result is None
        terminal = fitted.terminal
        assert terminal is not None
        assert terminal.convergence_flag != 0
        assert fitted.fit_diagnostics().status() == (
            f"fallback=ewma|flag={terminal.convergence_flag}"
        )
        assert len(terminal.params) == len(terminal.bounds) == 3

    def test_a_fit_that_raised_has_no_terminal_state_to_keep(self) -> None:
        fitted = GARCH(dist="normal").fit(np.full(300, np.nan))
        assert fitted.fallback is True
        assert fitted.terminal is None
        assert fitted.fit_diagnostics().status().startswith("fallback=ewma|raised ")

    def test_every_parameter_carries_the_box_it_was_optimized_inside(self) -> None:
        fitted = gjr_garch(dist="normal").fit(self._clean())
        terminal = fitted.terminal
        assert terminal is not None
        assert [name for name, _ in terminal.params] == ["omega", "alpha[1]", "gamma[1]", "beta[1]"]
        assert [name for name, _, _ in terminal.bounds] == [name for name, _ in terminal.params]
        for name, (below, above) in terminal.slack().items():
            assert below >= -1e-8 and above >= -1e-8, name

    def test_the_student_t_box_recorded_is_the_one_the_optimizer_was_given(self) -> None:
        """``arch`` computes a distribution's bounds from *standardized*
        residuals and the volatility process's from the raw ones; this module
        records both from the raw ones, which is only sound because the two
        distributions it uses ignore the argument. Pinned, because a future
        distribution that reads it would make the recorded box a fiction."""
        resids = self._clean()
        assert _BoundedStudentsT(NU_BOUNDS).bounds(resids) == [NU_BOUNDS]
        assert _BoundedStudentsT(NU_BOUNDS).bounds(resids * 1e6) == [NU_BOUNDS]

        fitted = GARCH(dist="studentst").fit(resids)
        assert fitted.terminal is not None
        assert fitted.terminal.bounds[-1] == ("nu", *NU_BOUNDS)

    def test_omega_is_recorded_on_the_rescaled_series_and_can_be_undone(self) -> None:
        """``rescale=True`` means ``omega`` is in units of ``scale ** 2`` — the
        one place in this record where a reader can mistake a scale for a
        finding. The same returns in different units give the same fit."""
        returns = self._clean()
        one = GARCH(dist="normal").fit(returns).terminal
        ten = GARCH(dist="normal").fit(returns * 10.0).terminal
        assert one is not None and ten is not None
        assert ten.scale == pytest.approx(one.scale / 10.0)
        assert dict(ten.params)["alpha[1]"] == pytest.approx(dict(one.params)["alpha[1]"], rel=1e-5)
        assert ten.omega_on_the_return_scale == pytest.approx(
            100.0 * one.omega_on_the_return_scale, rel=1e-5
        )

    def test_re_conditioning_carries_the_scheduled_fits_terminal_state(self) -> None:
        """Same rule as ``detail``: ``update`` runs no optimizer, so it has no
        terminal state of its own to report."""
        train = self._clean()
        fitted = GARCH(dist="normal").fit(train)
        assert fitted.update(train).terminal is fitted.terminal

    def test_retention_is_not_in_the_spec_and_so_moves_no_config_hash(self) -> None:
        fitted = GARCH(dist="normal").fit(self._clean())
        assert "terminal" not in fitted.spec()
        assert fitted.spec() == GARCH(dist="normal").spec()

    def test_a_bound_must_be_paired_with_its_own_parameter(self) -> None:
        with pytest.raises(ValueError, match="bounds"):
            TerminalFit(
                params=(("omega", 1.0), ("alpha[1]", 0.1)),
                loglikelihood=-1.0,
                convergence_flag=0,
                message="",
                iterations=1,
                function_evals=1,
                scale=1.0,
                bounds=(("omega", 0.0, 1.0),),
            )
        with pytest.raises(ValueError, match="paired with the bound"):
            TerminalFit(
                params=(("omega", 1.0),),
                loglikelihood=-1.0,
                convergence_flag=0,
                message="",
                iterations=1,
                function_evals=1,
                scale=1.0,
                bounds=(("alpha[1]", 0.0, 1.0),),
            )
