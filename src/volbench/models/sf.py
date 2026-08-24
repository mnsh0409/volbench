"""AutoETS and AutoARIMA on log realized variance, via `statsforecast`.

Contract — SAME INPUT AS ``models/har.py``, NOT THE RETURN-FED BASELINES:
``fit(train)`` takes a 1-D REALIZED-VARIANCE series in daily units, never
returns and never annualized (CLAUDE.md rule 2). The evaluator routes it
there through ``run_backtest(..., fit_series=...)``, which slices the RV
series with the *same* ``RollingOriginSplitter`` origins as the return series
(``tests/test_evaluate.py::test_fit_series_is_sliced_with_the_same_splitter_indices``).

What is modelled
================

``y_t = log RV_t``, fit by statsforecast's ``AutoETS`` (Hyndman et al.'s
state-space exponential smoothing with AICc model selection) or ``AutoARIMA``
(Hyndman-Khandakar stepwise selection). Logs, like HAR-RV: RV is positive and
right-skewed, and a Gaussian-error model on the level would put mass on
negative variances.

``AutoETS`` is pinned to ``model="AZN"`` — additive error, automatic trend,
no seasonality. Additive error is not a restriction on log-RV (statsforecast's
``ets_f`` already skips multiplicative-error candidates when the series is not
strictly positive, and ``log RV`` is negative for any daily variance below 1),
but stating it explicitly is what makes the additive-in-log-space residuals
that the retransformation is built on a guarantee rather than a coincidence.
``season_length=1`` throughout: daily volatility has no weekly period this
model set claims to capture, and ``seasonal`` is switched off for AutoARIMA
accordingly.

Retransformation — an explicit, config-hashed choice
====================================================

Both models forecast ``E[log RV]``; the benchmark needs a variance, i.e. a
mean. ``volbench.models._rv`` implements both corrections and documents the
formulae and the evidence; the summary is:

- ``retransform="smearing"`` (DEFAULT): Duan (1983) — ``exp(mu) * mean(exp(e))``
  over the fit window's one-step residuals. Nonparametric.
- ``retransform="gaussian"``: ``exp(mu + s_h^2 / 2)``, exact under Gaussian
  log-space errors, where ``s_h^2`` is the model's genuine **h-step** forecast
  variance (read off statsforecast's own prediction interval, so this arm is
  exact at every horizon, unlike HAR's reuse of a one-step residual variance).

Smearing is the default because ``docs/M2_NOTES.md`` measured the Gaussian
correction over-inflating on this repo's own target: HAR's forecast moved to
1.13x the fixture's known true variance once the target gained its overnight
component, because the Gaussian-log-residual assumption is mis-specified for
a near-chi-squared overnight shock. Both arms are kept, both are in
``spec()`` and therefore in every ``config_hash``, and the model's ``name``
carries the choice so two arms can never collide in one results table.

Re-conditioning between refits — ``SupportsUpdate`` IS implemented
==================================================================

statsforecast can re-filter at fixed parameters, which is the precondition
the Phase-2 brief set. ``AutoETS.forward`` / ``AutoARIMA.forward`` apply an
already-fitted model to a new series:

- ``forward_ets`` calls ``ets_f(y, m, model=<fitted dict>)``, whose
  ``isinstance(model, dict)`` branch lifts ``alpha, beta, gamma, phi`` and the
  initial state straight out of the fitted model and runs ``pegelsresid_C``
  — pure filtering, no optimizer.
- ``forward_arima`` calls ``Arima(x=y, model=<fitted dict>)`` -> ``arima2``,
  which rebuilds the selected order with ``fixed=coefs`` (every coefficient
  pinned) and then restores ``sigma2`` from the original fit.

Verified behaviourally, not just by reading: ``forward`` on the fit's own
window reproduces the fitted forecast exactly, a shifted window moves it, and
the fitted object is not mutated (``tests/test_models_sf.py::TestUpdate``).

A modelling note that follows from ``forward``'s semantics and is worth
stating rather than discovering later: re-conditioning re-runs the filter over
the *whole* new window starting from the initial state estimated at the last
scheduled fit — the state is not carried forward from where the previous
window ended. That is what R's ``ets(y, model=fit)`` and ``Arima(x,
model=fit)`` do, it reads nothing later than the origin, and at the window
lengths this package backtests on (500 observations) the state has long
converged, so the initialization is immaterial. It would not be for a model
with a near-zero smoothing parameter on a short window.

One wrinkle, handled: ``forward_ets`` *recomputes* the innovation variance
``sigma2`` from the window it is handed (it is derived from that window's
residuals), whereas ``arima2`` restores the fit's. Re-estimating a scale
between refits is exactly what ``SupportsUpdate`` forbids, so this adapter
never reads its forecast variance from ``forward``: the h-step forecast-error
variance is a function of the *fixed* parameters alone — the conditioning
window shifts the forecast's level, not its uncertainty — so it is read from
the scheduled fit's own ``predict(h, level=...)``. That is a no-op for
AutoARIMA (its interval width is already window-invariant) and it is what
keeps the ``gaussian`` arm honest for AutoETS. Pinned in
``tests/test_models_sf.py::TestUpdate::test_the_forecast_variance_comes_from_the_scheduled_fit``.

Failure policy: like HAR and unlike GARCH, a degenerate window raises. The
evaluator turns that into one NaN row with a ``fit_error@`` / ``update_error@``
reason and carries on (M1 report §4.5), so nothing silently substitutes a
different model for this one.

Determinism (CLAUDE.md rule 3): neither backend samples — AutoETS optimizes
by Nelder-Mead from a deterministic start and AutoARIMA's stepwise search is
deterministic given the data — so no seed enters here and two fits on the
same window are bit-identical (``TestDeterminism``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from statsforecast.models import AutoARIMA, AutoETS

from volbench.dist import Distribution, Normal
from volbench.models._rv import (
    Retransform,
    gaussian_factor,
    smearing_factor,
    validated_rv,
    variance_from_log,
)

__all__ = ["AutoARIMARV", "AutoETSRV", "FittedStatsForecastRV"]

#: Enough history for AutoARIMA's differencing tests and AutoETS's parameter
#: count to mean something. Below this the backends raise anyway, later and
#: with a less legible message.
_MIN_TRAIN: Final = 60

#: Prediction-interval level used to recover the h-step log-space forecast
#: sd for the ``gaussian`` arm. statsforecast builds symmetric Gaussian
#: intervals, so ``(hi - lo) / (2 * z)`` returns ``s_h`` exactly, whatever the
#: level; 95 is used only because it is the one everybody sanity-checks by eye.
_LEVEL: Final = 95
_Z: Final = NormalDist().inv_cdf(0.5 + 0.005 * _LEVEL)

#: AutoETS with additive error, automatic trend, no season. See module docstring.
_ETS_MODEL: Final = "AZN"

#: AutoARIMA search settings pinned here rather than left to the library's
#: defaults, so ``spec()`` fully determines the fitted model and a backend
#: default change cannot silently move a published number.
_ARIMA_IC: Final = "aicc"
_ARIMA_TEST: Final = "kpss"
_ARIMA_APPROXIMATION: Final = False


#: The two statsforecast classes this adapter drives. A union of the concrete
#: classes rather than a structural Protocol: statsforecast ships ``py.typed``,
#: so mypy checks the real signatures, and a hand-written Protocol would only
#: be a second, drift-prone copy of them.
_Backend = AutoETS | AutoARIMA


def _at(forecast: dict[str, Any], key: str, h: int) -> float:
    """Element ``h - 1`` of a statsforecast forecast entry.

    ``mean`` comes back as an ndarray but the interval bounds can be a pandas
    Series (AutoARIMA), so this normalizes before indexing.
    """
    return float(np.asarray(forecast[key], dtype=np.float64)[h - 1])


@dataclass(frozen=True, eq=False)
class FittedStatsForecastRV:
    """A fitted AutoETS/AutoARIMA on log-RV, conditioned on ``window``.

    ``eq=False``: numpy array and backend-object fields make the generated
    ``__eq__`` raise (the trap documented on ``Origin`` in splitter.py), so
    this falls back to identity comparison.

    ``backend`` is shared by reference with every object ``update`` derives
    from this one. That is safe *because* ``forward`` does not mutate the
    fitted model — asserted in ``tests/test_models_sf.py``, not assumed.
    """

    config: AutoETSRV | AutoARIMARV
    backend: _Backend
    #: The log-RV window the next forecast conditions on. ``fit``'s own window
    #: at the scheduled refit; a later origin's window after ``update``.
    window: NDArray[np.float64]
    #: Duan's smearing factor over the fit window's residuals. Estimated once,
    #: at the scheduled fit, and never re-estimated by ``update``.
    smear: float

    @property
    def name(self) -> str:
        return self.config.name

    def spec(self) -> dict[str, Any]:
        return self.config.spec()

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        mu = _at(self.backend.forward(self.window, h=h), "mean", h)
        if self.config.retransform == "smearing":
            factor = self.smear
        else:
            # From the SCHEDULED FIT, never from the re-conditioned window:
            # see the module docstring's "One wrinkle, handled".
            interval = self.backend.predict(h=h, level=[_LEVEL])
            sd = (_at(interval, f"hi-{_LEVEL}", h) - _at(interval, f"lo-{_LEVEL}", h)) / (2.0 * _Z)
            factor = gaussian_factor(sd * sd)
        return Normal(mu=0.0, sigma=math.sqrt(variance_from_log(mu, factor)))

    def update(self, train: NDArray[np.float64]) -> FittedStatsForecastRV:
        """Re-condition on ``train``; re-estimate nothing.

        The backend's parameters, the model order/components chosen at the
        scheduled fit and the smearing factor all stay exactly as estimated.
        Only ``window`` moves, and with it the state the filter runs forward
        to produce the next forecast.
        """
        rv = validated_rv(train, minimum=_MIN_TRAIN)
        return dataclasses.replace(self, window=np.log(rv))


def _fit_backend(
    config: AutoETSRV | AutoARIMARV, backend: _Backend, train: NDArray[np.float64]
) -> FittedStatsForecastRV:
    """Shared ``fit`` body: log the RV, fit, read the residuals, pin the smear."""
    rv = validated_rv(train, minimum=_MIN_TRAIN)
    y = np.log(rv)
    backend.fit(y)
    fitted = np.asarray(backend.predict_in_sample()["fitted"], dtype=np.float64)
    if fitted.shape != y.shape:
        raise ValueError(
            f"backend returned {fitted.size} in-sample fits for {y.size} observations"
        )
    return FittedStatsForecastRV(
        config=config, backend=backend, window=y, smear=smearing_factor(y - fitted)
    )


@dataclass(frozen=True)
class AutoETSRV:
    """AutoETS on log realized variance, retransformed to a variance forecast."""

    retransform: Retransform = "smearing"
    season_length: int = 1
    damped: bool | None = None

    def __post_init__(self) -> None:
        if self.retransform not in ("smearing", "gaussian"):
            raise ValueError("retransform must be 'smearing' or 'gaussian'")
        if self.season_length < 1:
            raise ValueError("season_length must be >= 1")

    @property
    def name(self) -> str:
        return f"autoets_rv-{self.retransform}"

    def spec(self) -> dict[str, Any]:
        return {
            "model": "autoets_rv",
            "backend": "statsforecast",
            "target": "log_rv",
            "ets_model": _ETS_MODEL,
            "season_length": self.season_length,
            "damped": self.damped,
            "retransform": self.retransform,
            "min_train": _MIN_TRAIN,
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedStatsForecastRV:
        backend = AutoETS(
            season_length=self.season_length, model=_ETS_MODEL, damped=self.damped
        )
        return _fit_backend(self, backend, train)


@dataclass(frozen=True)
class AutoARIMARV:
    """AutoARIMA on log realized variance, retransformed to a variance forecast."""

    retransform: Retransform = "smearing"
    season_length: int = 1
    max_p: int = 5
    max_q: int = 5
    max_d: int = 2
    stepwise: bool = True

    def __post_init__(self) -> None:
        if self.retransform not in ("smearing", "gaussian"):
            raise ValueError("retransform must be 'smearing' or 'gaussian'")
        if self.season_length < 1:
            raise ValueError("season_length must be >= 1")
        if min(self.max_p, self.max_q, self.max_d) < 0:
            raise ValueError("max_p, max_q and max_d must be non-negative")

    @property
    def name(self) -> str:
        return f"autoarima_rv-{self.retransform}"

    def spec(self) -> dict[str, Any]:
        return {
            "model": "autoarima_rv",
            "backend": "statsforecast",
            "target": "log_rv",
            "season_length": self.season_length,
            "seasonal": self.season_length > 1,
            "max_p": self.max_p,
            "max_q": self.max_q,
            "max_d": self.max_d,
            "stepwise": self.stepwise,
            "approximation": _ARIMA_APPROXIMATION,
            "ic": _ARIMA_IC,
            "test": _ARIMA_TEST,
            "retransform": self.retransform,
            "min_train": _MIN_TRAIN,
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedStatsForecastRV:
        backend = AutoARIMA(
            season_length=self.season_length,
            seasonal=self.season_length > 1,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            stepwise=self.stepwise,
            approximation=_ARIMA_APPROXIMATION,
            ic=_ARIMA_IC,
            test=_ARIMA_TEST,
        )
        return _fit_backend(self, backend, train)
