"""Shared plumbing for the models that fit on a realized-variance series in logs.

Two Phase-2 adapters — :mod:`volbench.models.sf` (AutoETS / AutoARIMA) and
:mod:`volbench.models.lgbm` (gradient boosting) — do the same three things
HAR-RV does: take an RV series, model ``log RV``, and turn a forecast of
``log RV`` back into a *variance* forecast. The validation and the
retransformation are identical across them, so they live here once rather
than three times.

Private on purpose (leading underscore, not re-exported from
``volbench.models``): it is an implementation detail of those adapters, not
part of the model interface.

Retransformation — the choice this module exists to make explicit
================================================================

A model fit in logs forecasts ``E[log RV]``, and ``exp(E[log RV])`` estimates
the conditional *median* of RV, not its mean. Since the number wanted is a
variance forecast — a mean — a correction factor is needed::

    RV_hat = exp(mu_hat) * factor

Two factors are implemented, and which one is used is a documented,
config-hashed option (never a silent default):

``gaussian`` — :func:`gaussian_factor`
    ``factor = exp(sigma^2 / 2)``: the exact lognormal mean correction *if*
    the log-space error is Gaussian with variance ``sigma^2``. This is what
    ``models/har.py`` does, using its own in-sample residual variance.

``smearing`` — :func:`smearing_factor`, the DEFAULT
    Duan's (1983) smearing estimate: ``factor = mean(exp(e_i))`` over the
    fit window's residuals ``e_i``. Nonparametric — it makes no distributional
    assumption about the log-space error, only that the errors are iid, and
    it is consistent for ``E[RV | x]`` under that condition. Duan, N. (1983),
    "Smearing Estimate: A Nonparametric Retransformation Method", *Journal of
    the American Statistical Association* 78(383), 605-610: the estimate of
    ``E[Y | x0]`` after fitting ``log Y = x'b + e`` is
    ``ave_i(exp(x0'b_hat + e_hat_i)) = exp(x0'b_hat) * ave_i(exp(e_hat_i))``.

**Why smearing is the default here.** ``docs/M2_NOTES.md`` measured the
Gaussian correction on this repo's own target and found it over-inflates when
the target is noisy in log space: HAR's forecast went from 0.98x to 1.13x the
fixture's *known* true close-to-close variance once the target gained its
overnight component, because "the overnight jump is a near-chi-squared single
shock" and HAR's Gaussian-log-residual assumption is mis-specified for it.
The smearing factor reads the correction off the realized residuals instead
of assuming their shape, so a heavy right tail in log space inflates it by
what the data show rather than by what a normal law would imply. Both are
kept: the Gaussian arm is the like-for-like comparison against HAR, and the
difference between them is itself a diagnostic of the mis-specification.

Note the two factors are equal only when the residuals are exactly Gaussian;
by Jensen, ``mean(exp(e))`` is at least ``exp(mean(e))``, and for real
log-RV residuals the two typically differ by a few percent.

Both factors are estimated **from the fit window's own in-sample residuals**
and never re-estimated afterwards, so they are parameters of the scheduled
fit in exactly the sense ``SupportsUpdate`` requires (``update`` may
re-condition, never re-estimate).

Horizon caveat, stated once and referenced from both adapters: the factor is
computed from *one-step* in-sample residuals. At ``h = 1`` it is the quantity
the theory describes. At ``h > 1`` the correct factor would come from the
h-step error distribution; applying the one-step factor is an approximation,
the same one ``models/har.py`` makes with its ``resid_var``. (The ``gaussian``
arm of ``models/sf.py`` is the exception: statsforecast reports a genuine
h-step forecast variance, so that arm is exact at every horizon.)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Retransform",
    "gaussian_factor",
    "smearing_factor",
    "validated_rv",
    "variance_from_log",
]

#: Which log-to-level retransformation a fitted model applies. See the module
#: docstring; ``"smearing"`` is every adapter's default.
Retransform = Literal["smearing", "gaussian"]


def validated_rv(train: NDArray[np.float64], minimum: int) -> NDArray[np.float64]:
    """A 1-D realized-variance window, checked the way ``models/har.py`` checks it.

    Raises rather than falling back: ``run_backtest`` turns the exception into
    a single NaN row with a ``fit_error@`` / ``update_error@`` reason and
    carries on, so one bad day costs one origin, not a cell (M1 report §4.5).
    """
    rv = np.asarray(train, dtype=np.float64)
    if rv.ndim != 1 or rv.size < minimum:
        raise ValueError(
            f"train must be a 1-D realized-variance series with at least {minimum} observations"
        )
    if not np.isfinite(rv).all() or (rv <= 0.0).any():
        raise ValueError("realized-variance series must be finite and strictly positive")
    return rv


def smearing_factor(resid: NDArray[np.float64]) -> float:
    """Duan's (1983) smearing factor ``mean(exp(e_i))`` over log-space residuals.

    ``resid`` are the in-sample residuals of a model fit on ``log RV``. Non-
    finite entries are dropped first: some backends report a NaN fitted value
    for observations consumed by differencing or state initialization, and
    those rows carry no residual information rather than an infinite one.
    """
    e = np.asarray(resid, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        raise ValueError("smearing factor needs at least one finite in-sample residual")
    factor = float(np.mean(np.exp(e)))
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"smearing factor is not a usable positive number: {factor!r}")
    return factor


def gaussian_factor(sigma2: float) -> float:
    """The lognormal mean correction ``exp(sigma^2 / 2)`` for log-space variance ``sigma2``."""
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise ValueError(f"log-space variance must be finite and non-negative: {sigma2!r}")
    factor = math.exp(0.5 * sigma2)
    if not math.isfinite(factor):
        raise ValueError("Gaussian retransformation overflowed: log-space variance is too large")
    return factor


def variance_from_log(mu: float, factor: float) -> float:
    """``exp(mu) * factor``, checked to be a usable variance."""
    if not math.isfinite(mu):
        raise ValueError(f"log-space forecast is not finite: {mu!r}")
    value = math.exp(mu) * factor
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"retransformed variance forecast is not positive and finite: {value!r}")
    return value
