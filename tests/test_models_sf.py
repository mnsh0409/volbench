"""AutoETS / AutoARIMA on log-RV: recovery, retransformation, update, leakage.

The four things this file is here to pin, in order of how much would break if
they silently changed:

1. **The retransformation is what it says it is.** Duan's smearing factor is
   ``mean(exp(residual))`` over the fit window, computed by hand here and
   compared bit for bit; the Gaussian arm is ``exp(mu + s_h^2/2)`` with the
   model's own h-step interval. Getting this wrong moves every variance
   forecast by a few percent — a size that looks like a modelling result.
2. **``update`` re-conditions and re-estimates nothing.** Including the one
   sharp edge: ``forward_ets`` recomputes the ETS innovation variance on the
   window it is handed, so the adapter reads its forecast variance from the
   *scheduled fit* instead.
3. **``fit`` sees only ``origin.train``.** The leakage canary, in the shape
   ``tests/test_evaluate.py`` uses: poison everything after a cutoff and
   require every earlier forecast to be bit-identical.
4. **The backends behave under the installed pandas**, which is newer than
   the ``pandas<3.0.0`` cap the ``[tool.uv] override-dependencies`` block in
   pyproject.toml lifts.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

pytest.importorskip("statsforecast", reason="the `classical` extra is not installed")

from volbench.dist import Distribution, Normal
from volbench.evaluate import (
    FittedModel,
    ForecastModel,
    SupportsUpdate,
    run_backtest,
)
from volbench.models._rv import smearing_factor
from volbench.models.sf import (
    _LEVEL,
    _MIN_TRAIN,
    _Z,
    AutoARIMARV,
    AutoETSRV,
    FittedStatsForecastRV,
)
from volbench.splitter import RollingOriginSplitter

CONFIGS = [AutoETSRV(), AutoARIMARV()]
ALL_ARMS = [
    AutoETSRV(),
    AutoETSRV(retransform="gaussian"),
    AutoARIMARV(),
    AutoARIMARV(retransform="gaussian"),
]


def log_ar1_rv(
    n: int = 600, phi: float = 0.9, sigma: float = 0.35, level: float = -9.0, seed: int = 0
) -> NDArray[np.float64]:
    """RV whose log is a stationary AR(1) — the textbook shape of daily log-RV."""
    rng = np.random.default_rng(seed)
    y = np.empty(n, dtype=np.float64)
    y[0] = level
    for t in range(1, n):
        y[t] = level * (1.0 - phi) + phi * y[t - 1] + sigma * rng.standard_normal()
    return np.exp(y)


def toy_rv() -> NDArray[np.float64]:
    """The committed toy fixture's close-to-close variance series."""
    from volbench.benchmarks.toy import SCORING_TARGET, load_series

    return load_series().targets[SCORING_TARGET].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# the interface contract
# --------------------------------------------------------------------------


class TestInterface:
    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_satisfies_the_one_model_protocol(self, config: Any) -> None:
        """The protocols come from ``volbench.models.base`` via ``evaluate``;
        this file must never grow its own copy (M1 report §4.1)."""
        assert isinstance(config, ForecastModel)
        fitted = config.fit(log_ar1_rv(n=200))
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted, SupportsUpdate)

    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_predict_returns_a_return_distribution_in_daily_units(self, config: Any) -> None:
        """CLAUDE.md rule 2: a distribution over the next-period RETURN, zero
        mean (these models are never shown a return), variance = the forecast."""
        rv = log_ar1_rv(n=400)
        dist = config.fit(rv).predict(1)
        assert isinstance(dist, Normal)
        assert dist.mu == 0.0
        assert dist.sigma > 0.0
        # Daily units: the forecast must sit in the neighbourhood of the RV
        # series itself, not 252x it.
        assert 0.1 < dist.variance() / float(np.median(rv)) < 10.0

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_predict_rejects_a_nonpositive_horizon(self, config: Any) -> None:
        fitted = config.fit(log_ar1_rv(n=200))
        with pytest.raises(ValueError, match="h must be >= 1"):
            fitted.predict(0)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_multistep_forecasts_are_usable(self, config: Any) -> None:
        fitted = config.fit(log_ar1_rv(n=400))
        for h in (1, 5, 22):
            assert math.isfinite(fitted.predict(h).sigma)
            assert fitted.predict(h).sigma > 0.0


# --------------------------------------------------------------------------
# the input contract: an RV series, and it raises rather than falling back
# --------------------------------------------------------------------------


class TestInputContract:
    """Same shape as HAR's: this fits on realized variances, not returns, and
    a degenerate window raises. ``run_backtest`` records the exception as one
    NaN row with a ``fit_error@`` reason rather than losing the cell."""

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_fit_rejects_nonpositive_variance(self, config: Any) -> None:
        rv = log_ar1_rv(n=200)
        rv[100] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            config.fit(rv)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_fit_rejects_nonfinite_variance(self, config: Any) -> None:
        rv = log_ar1_rv(n=200)
        rv[7] = np.inf
        with pytest.raises(ValueError, match="finite"):
            config.fit(rv)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_fit_rejects_a_too_short_window(self, config: Any) -> None:
        with pytest.raises(ValueError, match=f"at least {_MIN_TRAIN}"):
            config.fit(log_ar1_rv(n=_MIN_TRAIN - 1))

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_update_validates_exactly_like_fit(self, config: Any) -> None:
        rv = log_ar1_rv(n=300)
        fitted = config.fit(rv[:200])
        bad = rv[100:300].copy()
        bad[-1] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            fitted.update(bad)
        with pytest.raises(ValueError, match=f"at least {_MIN_TRAIN}"):
            fitted.update(rv[:10])

    def test_a_bad_origin_costs_one_row_not_the_cell(self) -> None:
        """The M1 §4.5 promise, end to end: one poisoned RV day yields NaN
        rows with a ``fit_error``/``update_error`` reason, and the backtest
        still returns every other origin."""
        rv = log_ar1_rv(n=320)
        returns = np.sqrt(rv) * np.random.default_rng(0).standard_normal(rv.size)
        rv[210] = 0.0
        splitter = RollingOriginSplitter(window=200, horizon=1, step=1, refit_every=5)
        frame = run_backtest(
            AutoETSRV, returns, rv, splitter, 7, asset="T", proxy_name="p", fit_series=rv
        )
        assert frame["forecast_var"].notna().any()
        assert frame["forecast_var"].isna().any()
        assert frame.loc[frame["forecast_var"].isna(), "missing_reason"].str.contains(
            "fit_error|update_error"
        ).all()


# --------------------------------------------------------------------------
# parameter recovery
# --------------------------------------------------------------------------


class TestParameterRecovery:
    """Generate from the model, then check the backend found it. These are
    tests of the *adapter's plumbing* — the log transform, the fit call, the
    residual read — as much as of statsforecast: a transposed array or a
    forgotten ``np.log`` breaks them immediately."""

    def test_autoarima_recovers_a_known_ar1_in_logs(self) -> None:
        phi = 0.7
        fitted = AutoARIMARV().fit(log_ar1_rv(n=1500, phi=phi, sigma=0.3, seed=11))
        coef = fitted.backend.model_["coef"]  # type: ignore[attr-defined]
        assert "ar1" in coef, f"expected a plain AR(1) to be selected, got {sorted(coef)}"
        assert float(coef["ar1"]) == pytest.approx(phi, abs=0.05)

    def test_autoets_recovers_a_known_local_level_alpha(self) -> None:
        """ETS(A,N,N) with alpha = 0.3: y_t = l_{t-1} + e_t, l_t = l_{t-1} + alpha*e_t."""
        alpha = 0.3
        rng = np.random.default_rng(12)
        n = 2000
        y = np.empty(n)
        level = -9.0
        for t in range(n):
            e = 0.3 * rng.standard_normal()
            y[t] = level + e
            level = level + alpha * e
        fitted = AutoETSRV().fit(np.exp(y))
        model = fitted.backend.model_  # type: ignore[attr-defined]
        assert model["components"][:3] == "ANN"
        assert float(model["par"][0]) == pytest.approx(alpha, abs=0.06)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_a_near_constant_rv_series_forecasts_that_constant(self, config: Any) -> None:
        """The crudest sanity check there is, and the one that catches a units
        or transform slip: feed a series pinned at 4e-4 and the variance
        forecast must be 4e-4, not its log, its square or 252x it."""
        rng = np.random.default_rng(13)
        rv = 4e-4 * np.exp(rng.normal(0.0, 0.01, 400))
        assert config.fit(rv).predict(1).variance() == pytest.approx(4e-4, rel=0.02)


class TestOnTheToyFixture:
    """Sanity on the committed fixture — the same series ``make reproduce``
    runs — rather than on data generated to suit the model."""

    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_forecast_tracks_the_level_of_the_toy_rv_series(self, config: Any) -> None:
        rv = toy_rv()[:500]
        forecast = config.fit(rv).predict(1).variance()
        assert 0.5 < forecast / float(np.mean(rv)) < 2.0

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_the_smearing_factor_is_a_mild_positive_correction(self, config: Any) -> None:
        """On real-shaped log-RV the Duan factor is a few tens of percent — if
        it ever comes back at 1.0 the residuals are not being read, and if it
        comes back at 10 the fit is in the wrong units."""
        fitted = config.fit(toy_rv()[:500])
        assert 1.0 < fitted.smear < 2.0


# --------------------------------------------------------------------------
# the retransformation
# --------------------------------------------------------------------------


class TestRetransformation:
    def test_smearing_is_the_default_arm(self) -> None:
        assert AutoETSRV().retransform == "smearing"
        assert AutoARIMARV().retransform == "smearing"

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_the_smearing_factor_is_duans_mean_of_exponentiated_residuals(
        self, config: Any
    ) -> None:
        """Duan (1983): ``E[Y|x0] = exp(x0'b) * ave_i(exp(e_i))``. Recomputed
        here straight from the backend's own in-sample fits."""
        rv = toy_rv()[:500]
        fitted = config.fit(rv)
        y = np.log(rv)
        in_sample = np.asarray(
            fitted.backend.predict_in_sample()["fitted"], dtype=np.float64  # type: ignore[attr-defined]
        )
        expected = float(np.mean(np.exp(y - in_sample)))
        assert fitted.smear == pytest.approx(expected, rel=1e-15)
        assert fitted.smear == pytest.approx(smearing_factor(y - in_sample), rel=1e-15)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_smearing_predict_is_exactly_exp_mu_times_the_factor(self, config: Any) -> None:
        rv = toy_rv()[:500]
        fitted = config.fit(rv)
        mu = float(np.asarray(fitted.backend.forward(np.log(rv), h=1)["mean"])[0])
        assert fitted.predict(1).variance() == pytest.approx(math.exp(mu) * fitted.smear, rel=1e-14)

    @pytest.mark.parametrize(
        "config",
        [AutoETSRV(retransform="gaussian"), AutoARIMARV(retransform="gaussian")],
        ids=lambda c: c.name,
    )
    def test_gaussian_predict_is_exactly_exp_mu_plus_half_h_step_variance(
        self, config: Any
    ) -> None:
        rv = toy_rv()[:500]
        fitted = config.fit(rv)
        for h in (1, 5):
            mu = float(np.asarray(fitted.backend.forward(np.log(rv), h=h)["mean"])[h - 1])
            interval = fitted.backend.predict(h=h, level=[_LEVEL])
            hi = float(np.asarray(interval[f"hi-{_LEVEL}"], dtype=np.float64)[h - 1])
            lo = float(np.asarray(interval[f"lo-{_LEVEL}"], dtype=np.float64)[h - 1])
            sd = (hi - lo) / (2.0 * _Z)
            expected = math.exp(mu + 0.5 * sd * sd)
            assert fitted.predict(h).variance() == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("kind", [AutoETSRV, AutoARIMARV], ids=lambda k: k.__name__)
    def test_the_two_arms_disagree_and_both_exceed_the_median(self, kind: Any) -> None:
        """Both corrections lift the exponentiated point forecast (the
        conditional median) toward the mean; they are not the same number,
        which is the whole reason the choice is an option and not a constant."""
        rv = toy_rv()[:500]
        smeared = kind().fit(rv)
        gaussian = kind(retransform="gaussian").fit(rv)
        median = math.exp(float(np.asarray(smeared.backend.forward(np.log(rv), h=1)["mean"])[0]))
        v_s = smeared.predict(1).variance()
        v_g = gaussian.predict(1).variance()
        assert v_s > median and v_g > median
        assert v_s != v_g

    @pytest.mark.parametrize("kind", [AutoETSRV, AutoARIMARV], ids=lambda k: k.__name__)
    def test_the_gaussian_arm_grows_with_the_horizon_and_smearing_does_not(
        self, kind: Any
    ) -> None:
        """The documented asymmetry: the Gaussian factor uses the model's real
        h-step forecast variance, so it widens with h; the smearing factor is
        a one-step quantity reused at every horizon (module docstring of
        ``volbench.models._rv``, "Horizon caveat")."""
        rv = toy_rv()[:500]
        gaussian = kind(retransform="gaussian").fit(rv)
        smeared = kind().fit(rv)
        ratio_g = gaussian.predict(5).variance() / gaussian.predict(1).variance()
        ratio_s = smeared.predict(5).variance() / smeared.predict(1).variance()
        assert ratio_g > ratio_s


# --------------------------------------------------------------------------
# update(): re-condition, never re-estimate
# --------------------------------------------------------------------------


class TestUpdate:
    """``SupportsUpdate`` IS implementable here: both backends expose
    ``forward``, which re-filters at fixed parameters. These pin that it does
    only that."""

    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_update_on_the_fit_window_reproduces_the_fit_exactly(self, config: Any) -> None:
        rv = log_ar1_rv(n=400, seed=21)
        window = rv[:300]
        fitted = config.fit(window)
        again = fitted.update(window)
        for h in (1, 5):
            assert again.predict(h) == fitted.predict(h)

    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_update_on_a_shifted_window_re_conditions_the_forecast(self, config: Any) -> None:
        rv = log_ar1_rv(n=500, seed=22)
        fitted = config.fit(rv[:300])
        moved = fitted.update(rv[100:400]).predict(1)
        held = fitted.predict(1)
        assert moved.sigma != held.sigma
        assert 0.1 < moved.sigma / held.sigma < 10.0  # same units, not a scale slip

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_update_re_estimates_nothing(self, config: Any) -> None:
        rv = log_ar1_rv(n=500, seed=23)
        fitted = config.fit(rv[:300])
        before = _parameters(fitted)
        again = fitted.update(rv[100:400])
        assert again.smear == fitted.smear
        assert again.spec() == fitted.spec() and again.name == fitted.name
        assert _parameters(again) == before
        # ...and the shared backend object was not mutated on the way through.
        assert _parameters(fitted) == before

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_forward_does_not_mutate_the_fitted_backend(self, config: Any) -> None:
        """The whole reason ``FittedStatsForecastRV`` may share one backend by
        reference across every ``update`` derived from it."""
        rv = log_ar1_rv(n=500, seed=24)
        fitted = config.fit(rv[:300])
        first = fitted.predict(1)
        fitted.update(rv[100:400]).predict(1)
        assert fitted.predict(1) == first

    def test_the_forecast_variance_comes_from_the_scheduled_fit(self) -> None:
        """statsforecast's ``forward_ets`` recomputes the ETS innovation
        variance on whatever window it is handed — a re-estimated scale, which
        ``SupportsUpdate`` forbids between refits. The adapter therefore reads
        its h-step forecast variance from the scheduled fit's own ``predict``.

        The check: the Gaussian arm's *correction factor* must be identical
        before and after an update, even though the point forecast moved. If
        the adapter ever started reading the width off ``forward``, this
        fails for AutoETS (the two widths differ by ~3% on this data).
        """
        rv = log_ar1_rv(n=500, seed=25)
        config = AutoETSRV(retransform="gaussian")
        fitted = config.fit(rv[:300])
        again = fitted.update(rv[100:400])

        def factor(model: FittedStatsForecastRV, window: NDArray[np.float64]) -> float:
            mu = float(np.asarray(model.backend.forward(window, h=1)["mean"])[0])
            return model.predict(1).variance() / math.exp(mu)

        assert factor(fitted, np.log(rv[:300])) == pytest.approx(
            factor(again, np.log(rv[100:400])), rel=1e-15
        )
        # And prove the trap was real: forward's own interval IS different.
        w_fit = _interval_width(fitted.backend.predict(h=1, level=[_LEVEL]))
        w_fwd = _interval_width(fitted.backend.forward(np.log(rv[100:400]), h=1, level=[_LEVEL]))
        assert w_fit != pytest.approx(w_fwd, rel=1e-9)

    def test_the_arima_interval_is_already_window_invariant(self) -> None:
        """The same check for AutoARIMA, which ``arima2`` already handles by
        restoring the fit's ``sigma2`` — recorded so a backend change that
        broke it would show up here rather than in a paper number."""
        rv = log_ar1_rv(n=500, seed=26)
        fitted = AutoARIMARV(retransform="gaussian").fit(rv[:300])
        w_fit = _interval_width(fitted.backend.predict(h=1, level=[_LEVEL]))
        w_fwd = _interval_width(fitted.backend.forward(np.log(rv[100:400]), h=1, level=[_LEVEL]))
        assert w_fit == pytest.approx(w_fwd, rel=1e-12)


def _parameters(fitted: FittedStatsForecastRV) -> str:
    """A stable string of everything the backend estimated."""
    model = fitted.backend.model_  # type: ignore[attr-defined]
    if "coef" in model:  # AutoARIMA
        return repr(
            (sorted(model["coef"].items()), tuple(model["arma"]), float(model["sigma2"]))
        )
    return repr((model["components"], np.asarray(model["par"]).tolist()))  # AutoETS


def _interval_width(forecast: dict[str, Any]) -> float:
    hi = float(np.asarray(forecast[f"hi-{_LEVEL}"], dtype=np.float64)[0])
    lo = float(np.asarray(forecast[f"lo-{_LEVEL}"], dtype=np.float64)[0])
    return hi - lo


# --------------------------------------------------------------------------
# determinism (CLAUDE.md rule 3)
# --------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_the_same_window_gives_a_bit_identical_forecast_twice(self, config: Any) -> None:
        rv = toy_rv()[:400]
        first = config.fit(rv).predict(1)
        second = config.fit(rv).predict(1)
        assert isinstance(first, Normal) and isinstance(second, Normal)
        assert first.sigma == second.sigma  # bit-identical, not approx

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
    def test_no_seed_reaches_these_models(self, config: Any) -> None:
        """Neither backend samples, so ``spec()`` carries no seed and two runs
        cannot diverge through an RNG."""
        assert "seed" not in config.spec()


# --------------------------------------------------------------------------
# spec(): complete, hashable, discriminating
# --------------------------------------------------------------------------


class TestSpec:
    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_spec_is_json_serializable_and_matches_the_fitted_one(self, config: Any) -> None:
        json.dumps(config.spec(), sort_keys=True)
        assert config.fit(log_ar1_rv(n=200)).spec() == config.spec()

    @pytest.mark.parametrize("config", ALL_ARMS, ids=lambda c: c.name)
    def test_spec_is_stable_across_identical_constructions(self, config: Any) -> None:
        assert type(config)(retransform=config.retransform).spec() == config.spec()

    def test_spec_and_name_separate_every_arm(self) -> None:
        """Two configs that would land in one results table must not collide
        onto one ``config_hash`` — nor onto one label."""
        specs = [c.spec() for c in ALL_ARMS]
        names = [c.name for c in ALL_ARMS]
        assert len({json.dumps(s, sort_keys=True) for s in specs}) == len(ALL_ARMS)
        assert len(set(names)) == len(ALL_ARMS)

    def test_spec_records_every_setting_that_determines_the_fit(self) -> None:
        """A spec that omits a search setting lets a backend default change
        move a published number without moving its hash."""
        ets = AutoETSRV().spec()
        assert set(ets) == {
            "model",
            "backend",
            "target",
            "ets_model",
            "season_length",
            "damped",
            "retransform",
            "min_train",
        }
        arima = AutoARIMARV().spec()
        assert set(arima) == {
            "model",
            "backend",
            "target",
            "season_length",
            "seasonal",
            "max_p",
            "max_q",
            "max_d",
            "stepwise",
            "approximation",
            "ic",
            "test",
            "retransform",
            "min_train",
        }

    def test_a_changed_hyperparameter_changes_the_spec(self) -> None:
        assert AutoARIMARV(max_p=5).spec() != AutoARIMARV(max_p=3).spec()
        assert AutoETSRV(damped=True).spec() != AutoETSRV(damped=None).spec()

    @pytest.mark.parametrize("kind", [AutoETSRV, AutoARIMARV], ids=lambda k: k.__name__)
    def test_an_invalid_configuration_is_rejected_at_construction(self, kind: Any) -> None:
        with pytest.raises(ValueError, match="retransform"):
            kind(retransform="lognormal")
        with pytest.raises(ValueError, match="season_length"):
            kind(season_length=0)


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


class _FitSpy:
    """Wraps a real config and records the exact array ``fit`` was handed."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.fit_windows: list[NDArray[np.float64]] = []
        self.update_windows: list[NDArray[np.float64]] = []

    @property
    def name(self) -> str:
        return str(self.config.name)

    def spec(self) -> dict[str, Any]:
        return dict(self.config.spec())

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _FitSpyFitted:
        self.fit_windows.append(np.asarray(train, dtype=np.float64).copy())
        return _FitSpyFitted(self, self.config.fit(train))


class _FitSpyFitted:
    def __init__(self, owner: _FitSpy, inner: FittedStatsForecastRV) -> None:
        self.owner = owner
        self.inner = inner

    @property
    def name(self) -> str:
        return str(self.inner.name)

    def spec(self) -> dict[str, Any]:
        return self.inner.spec()

    def predict(self, h: int) -> Distribution:
        return self.inner.predict(h)

    def update(self, train: NDArray[np.float64]) -> _FitSpyFitted:
        self.owner.update_windows.append(np.asarray(train, dtype=np.float64).copy())
        return _FitSpyFitted(self.owner, self.inner.update(train))


def _panel(n: int = 320, seed: int = 31) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rv = log_ar1_rv(n=n, seed=seed)
    returns = np.sqrt(rv) * np.random.default_rng(seed + 1).standard_normal(n)
    return returns, rv


class TestLeakage:
    def test_fit_and_update_see_only_the_origins_own_train_window(self) -> None:
        """The canary in ``tests/test_evaluate.py::
        test_fit_series_is_sliced_with_the_same_splitter_indices``, pointed at
        this adapter: every array either entry point receives must be exactly
        ``rv[origin.train]``, produced by the splitter and nothing else."""
        returns, rv = _panel()
        splitter = RollingOriginSplitter(window=200, horizon=1, step=1, refit_every=5)
        spy = _FitSpy(AutoETSRV())
        run_backtest(
            lambda: spy, returns, rv, splitter, 3, asset="T", proxy_name="p", fit_series=rv
        )

        origins = list(splitter.split(returns.size))
        for window, origin in zip(
            spy.fit_windows, [o for o in origins if o.refit], strict=True
        ):
            assert np.array_equal(window, rv[origin.train])
        for window, origin in zip(
            spy.update_windows, [o for o in origins if not o.refit], strict=True
        ):
            assert np.array_equal(window, rv[origin.train])
        assert len(spy.fit_windows) > 1 and len(spy.update_windows) > 1
        # And every one of those windows ends at or before its origin.
        assert all(int(o.train[-1]) <= o.origin for o in origins)

    def test_the_proxy_never_reaches_the_model(self) -> None:
        """QLIKE's scoring target must not be an input, or the variance
        forecasts would be scoring themselves."""
        returns, rv = _panel()
        splitter = RollingOriginSplitter(window=200, horizon=1, step=1, refit_every=10)
        kwargs: dict[str, Any] = dict(asset="T", proxy_name="p", fit_series=rv)
        base = run_backtest(AutoETSRV, returns, rv, splitter, 3, **kwargs)
        other = run_backtest(
            AutoETSRV, returns, np.full_like(rv, 4.2e-4), splitter, 3, **kwargs
        )
        pd.testing.assert_frame_equal(
            base[["forecast_mean", "forecast_var"]], other[["forecast_mean", "forecast_var"]]
        )

    def test_future_data_cannot_touch_earlier_forecasts(self) -> None:
        """THE canary (`.claude/skills/leakage-check`): corrupt everything
        strictly after a cutoff, and every forecast for a target at or before
        it must be bit-identical."""
        returns, rv = _panel(n=300, seed=41)
        splitter = RollingOriginSplitter(window=200, horizon=1, step=1, refit_every=5)
        origins = list(splitter.split(returns.size))
        cutoff = int(origins[40].test[-1])

        rng = np.random.default_rng(99)
        dirty_returns, dirty_rv = returns.copy(), rv.copy()
        n_bad = dirty_rv[cutoff + 1 :].size
        dirty_returns[cutoff + 1 :] = rng.normal(0.0, 0.5, n_bad)
        dirty_rv[cutoff + 1 :] = np.exp(rng.normal(0.0, 1.0, n_bad))

        kwargs: dict[str, Any] = dict(asset="T", proxy_name="p")
        clean = run_backtest(
            AutoETSRV, returns, rv, splitter, 5, fit_series=rv, **kwargs
        )
        dirty = run_backtest(
            AutoETSRV, dirty_returns, dirty_rv, splitter, 5, fit_series=dirty_rv, **kwargs
        )

        scores = [c for c in clean.columns if c != "config_hash"]
        before = clean.loc[clean["target_index"] <= cutoff, scores].reset_index(drop=True)
        after = dirty.loc[dirty["target_index"] <= cutoff, scores].reset_index(drop=True)
        assert len(before) > 20
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        assert clean.attrs["config_hash"] != dirty.attrs["config_hash"]

        # A canary that cannot die proves nothing: the poison must have moved
        # every forecast that could see it. Row 0 of the tail is exempt and
        # must be — its target is ``cutoff + 1``, so its training window ends
        # at ``cutoff`` and contains no corrupted observation. Every later one
        # conditions on poisoned data and has to move.
        late_clean = clean.loc[clean["target_index"] > cutoff, "forecast_var"].to_numpy()
        late_dirty = dirty.loc[dirty["target_index"] > cutoff, "forecast_var"].to_numpy()
        assert len(late_clean) > 20
        assert late_clean[0] == late_dirty[0]
        assert (late_clean[1:] != late_dirty[1:]).all()


# --------------------------------------------------------------------------
# the lifted pandas cap
# --------------------------------------------------------------------------


class TestBackendCompatibility:
    """statsforecast declares ``pandas<3.0.0``; pyproject's ``[tool.uv]
    override-dependencies`` lifts that so the ``classical`` extra can resolve
    against volbench's ``pandas>=3.0.5``. The cap is a compatibility guess,
    and this is the test that makes it an observed fact rather than a hope:
    if a future pandas genuinely breaks statsforecast, it breaks here first
    and the override gets revisited instead of silently shipping.
    """

    def test_both_backends_run_their_full_path_under_the_installed_pandas(self) -> None:
        import pandas
        import statsforecast

        assert pandas.__version__ >= "3"
        rv = toy_rv()[:400]
        for config in ALL_ARMS:
            fitted = config.fit(rv)
            fitted.update(rv[50:])
            for h in (1, 5):
                assert math.isfinite(fitted.predict(h).sigma)
        assert statsforecast.__version__  # the version that carried the cap
