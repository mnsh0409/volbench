"""``update()`` on every baseline: fixed parameters, fresh conditioning.

M1 report §4.3. Each model's ``update(train)`` must (1) leave every estimated
parameter untouched, (2) re-condition on ``train`` — observations dated at or
before the forecast origin, the same array ``fit`` would have been given —
and (3) reproduce ``fit`` exactly when handed the fit window itself, which is
what makes re-conditioning a no-op at ``refit_every=1``.

GARCH is the one with real machinery behind it: ``arch``'s ``ARCHModel.fix``
re-filters at fixed parameters. Its correctness pin is that ``fix`` on the
fit's own window reproduces the fitted forecast and every in-sample
conditional variance — a check that only passes if the ``rescale`` handling
is right too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from volbench.dist import Normal, StudentT
from volbench.models import EWMA, GARCH, HAR, FittedGARCH, NaiveVol, gjr_garch


def returns(n: int = 600, seed: int = 0) -> NDArray[np.float64]:
    """GARCH(1,1) returns with equity-like persistence, so fits are clean."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = 2.4e-6 / 0.02
    r[0] = math.sqrt(s2) * rng.standard_normal()
    for t in range(1, n):
        s2 = 2.4e-6 + 0.06 * r[t - 1] ** 2 + 0.92 * s2
        r[t] = math.sqrt(s2) * rng.standard_normal()
    return r


def realized_variance(n: int = 600, seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(1e-4), 0.4, size=n))


class TestEWMA:
    def test_update_on_the_fit_window_is_the_fit(self) -> None:
        r = returns()
        fitted = EWMA(lambda_=0.94).fit(r[:500])
        assert fitted.update(r[:500]) == fitted  # value equality, bit for bit

    def test_update_never_touches_lambda_and_matches_the_hand_recursion(self) -> None:
        fitted = EWMA(lambda_=0.9).fit(np.array([0.01, -0.02, 0.015, 0.0, 0.03]))
        window = np.array([-0.02, 0.015, 0.0, 0.03, 0.01])
        again = fitted.update(window)
        var = window[0] ** 2
        for x in window[1:]:
            var = 0.9 * var + 0.1 * x * x
        assert again.lambda_ == 0.9
        assert again.sigma2 == pytest.approx(var, rel=1e-15)

    def test_update_on_a_shifted_window_moves_the_forecast(self) -> None:
        r = returns()
        fitted = EWMA().fit(r[:500])
        assert fitted.update(r[1:501]).sigma2 != fitted.sigma2


class TestNaiveVol:
    def test_update_on_the_fit_window_is_the_fit(self) -> None:
        r = returns()
        fitted = NaiveVol().fit(r[:500])
        assert fitted.update(r[:500]) == fitted

    def test_update_slides_the_window(self) -> None:
        r = returns()
        fitted = NaiveVol().fit(r[:500])
        again = fitted.update(r[1:501])
        assert again.sigma == pytest.approx(float(np.sqrt(np.mean(r[1:501] ** 2))))
        assert again.sigma != fitted.sigma

    def test_update_rejects_what_fit_rejects(self) -> None:
        fitted = NaiveVol().fit(returns()[:500])
        with pytest.raises(ValueError, match="at least 2"):
            fitted.update(np.array([0.01]))


class TestHAR:
    def test_update_keeps_the_coefficients_and_moves_the_lags(self) -> None:
        rv = realized_variance()
        fitted = HAR().fit(rv[:500])
        again = fitted.update(rv[1:501])
        assert np.array_equal(again.beta, fitted.beta)
        assert again.resid_var == fitted.resid_var
        assert np.array_equal(again.buffer, rv[1:501][-22:])
        assert not np.array_equal(again.buffer, fitted.buffer)
        assert again.predict(1) != fitted.predict(1)

    def test_update_on_the_fit_window_is_the_fit(self) -> None:
        rv = realized_variance()
        fitted = HAR().fit(rv[:500])
        again = fitted.update(rv[:500])
        assert np.array_equal(again.buffer, fitted.buffer)
        assert again.predict(1) == fitted.predict(1)

    def test_update_validates_like_fit(self) -> None:
        rv = realized_variance()
        fitted = HAR().fit(rv[:500])
        bad = rv[1:501].copy()
        bad[-3] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            fitted.update(bad)
        with pytest.raises(ValueError, match="at least 22"):
            fitted.update(rv[:10])


class TestGARCH:
    """`fix()` re-filters at fixed parameters; these pin that it does only that."""

    @pytest.mark.parametrize(
        "model",
        [GARCH(dist="normal"), gjr_garch(dist="normal"), GARCH(dist="studentst")],
        ids=lambda m: m.name,
    )
    def test_update_on_the_fit_window_reproduces_the_fit_exactly(self, model: GARCH) -> None:
        r = returns(seed=1)
        window = r[:500]
        fitted = model.fit(window)
        assert not fitted.fallback and fitted.result is not None
        again = fitted.update(window)
        assert again.result is not None

        # Every in-sample conditional variance, in the caller's units.
        fit_path = fitted.result.conditional_volatility ** 2 / fitted.scale**2
        fix_path = again.result.conditional_volatility ** 2 / again.scale**2
        assert float(np.max(np.abs(fit_path - fix_path))) < 1e-8
        # And the forecast itself.
        assert again.predict(1) == fitted.predict(1)

    @pytest.mark.parametrize(
        "model",
        [GARCH(dist="normal"), gjr_garch(dist="studentst")],
        ids=lambda m: m.name,
    )
    def test_update_re_estimates_nothing(self, model: GARCH) -> None:
        r = returns(seed=2)
        fitted = model.fit(r[:500])
        assert not fitted.fallback and fitted.result is not None
        again = fitted.update(r[7:507])
        assert again.result is not None
        assert list(again.result.params.index) == list(fitted.result.params.index)
        assert np.array_equal(again.result.params.to_numpy(), fitted.result.params.to_numpy())
        assert again.scale == fitted.scale
        assert again.spec() == fitted.spec() and again.name == fitted.name
        assert again.fallback is False

    def test_update_on_a_shifted_window_re_conditions_the_forecast(self) -> None:
        r = returns(seed=3)
        fitted = GARCH(dist="normal").fit(r[:500])
        moved = fitted.update(r[1:501]).predict(1)
        held = fitted.predict(1)
        assert isinstance(moved, Normal) and isinstance(held, Normal)
        assert moved.sigma != held.sigma
        assert 0.5 < moved.sigma / held.sigma < 2.0  # same units: a scale slip would be ~100x

    def test_student_t_update_keeps_nu_and_returns_student_t(self) -> None:
        rng = np.random.default_rng(4)
        r = rng.standard_t(df=6, size=800) * 0.01
        fitted = GARCH(dist="studentst").fit(r[:700])
        assert not fitted.fallback and fitted.result is not None
        again = fitted.update(r[100:800])
        dist = again.predict(1)
        assert isinstance(dist, StudentT)
        assert dist.df == float(fitted.result.params["nu"])

    def test_fallback_fit_updates_its_ewma(self) -> None:
        fitted = GARCH().fit(np.zeros(300))  # degenerate: the optimizer cannot converge
        assert fitted.fallback and fitted.fallback_fit is not None
        again = fitted.update(returns()[:200])
        assert isinstance(again, FittedGARCH) and again.fallback
        assert again.fallback_fit is not None
        assert again.fallback_fit.sigma2 != fitted.fallback_fit.sigma2
        assert again.fallback_fit.lambda_ == fitted.fallback_lambda

    def test_update_is_deterministic(self) -> None:
        r = returns(seed=5)
        fitted = GARCH().fit(r[:500])
        assert fitted.update(r[3:503]).predict(1) == fitted.update(r[3:503]).predict(1)

    def test_update_rejects_a_short_window(self) -> None:
        fitted = GARCH().fit(returns()[:500])
        with pytest.raises(ValueError, match="at least 20"):
            fitted.update(returns()[:10])
