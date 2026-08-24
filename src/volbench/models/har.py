"""HAR-RV (Corsi, 2009) via OLS in logs.

Contract — DIFFERENT INPUT FROM THE OTHER MODELS IN THIS PACKAGE:
`fit(train)` takes a 1-D REALIZED-VARIANCE series (RV_1, ..., RV_n), not
returns. Every other model here fits on returns; HAR-RV is defined
directly on a variance proxy, which must already be in daily units, never
annualized (CLAUDE.md rule 2).

Design (strict, non-truncated windows only — the first usable row needs a
full 22-day history, never a shorter, truncated one):

    log(RV_{t+1}) = b0 + b1*log(RV_d,t) + b2*log(RV_w,t) + b3*log(RV_m,t) + e_t
    RV_d,t = RV_t
    RV_w,t = mean(RV_{t-4 .. t})     (5 obs)
    RV_m,t = mean(RV_{t-21 .. t})    (22 obs)

fit via OLS (`np.linalg.lstsq`).

Retransformation: fitting in logs and exponentiating the point forecast
underestimates E[RV] (Jensen's inequality). We assume Gaussian residuals in
log-space and apply the standard lognormal correction
E[RV] = exp(y_hat + 0.5 * resid_var), with resid_var estimated from the
fit's own in-sample residuals (never from held-out data).

Multi-step forecasting: HAR is a direct 1-step model. For h > 1 this
forecasts recursively — each step's point forecast is fed back into the RV
buffer as if realized, and the design vector is rebuilt for the next step
(the standard iterated HAR multi-day forecast).

predict() returns a Distribution over the next-period RETURN (package
convention: FittedModel.predict -> Distribution over the return, variance
is a property of it), with the forecast RV as its variance and mu=0 — HAR
is only ever told RV, never returns, so — like the other baselines here —
it cannot recover a conditional mean and assumes none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal

__all__ = ["HAR", "FittedHAR"]

_D_WINDOW = 1
_W_WINDOW = 5
_M_WINDOW = 22
_N_PARAMS = 4
_MIN_ROWS = 5
_MIN_TRAIN = _M_WINDOW + _MIN_ROWS


def _validated_rv(train: NDArray[np.float64], minimum: int) -> NDArray[np.float64]:
    rv = np.asarray(train, dtype=np.float64)
    if rv.ndim != 1 or rv.size < minimum:
        raise ValueError(
            f"train must be a 1-D realized-variance series with at least {minimum} observations"
        )
    if not np.isfinite(rv).all() or (rv <= 0.0).any():
        raise ValueError("realized-variance series must be finite and strictly positive")
    return rv


def _har_features(rv: NDArray[np.float64], t: int) -> tuple[float, float, float]:
    """RV_d, RV_w, RV_m at time t, using only rv[0..t] (strict, full windows)."""
    d = float(rv[t])
    w = float(np.mean(rv[t - _W_WINDOW + 1 : t + 1]))
    m = float(np.mean(rv[t - _M_WINDOW + 1 : t + 1]))
    return d, w, m


def _design_matrix(rv: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = rv.size
    rows: list[list[float]] = []
    targets: list[float] = []
    for t in range(_M_WINDOW - 1, n - 1):
        d, w, m = _har_features(rv, t)
        rows.append([1.0, math.log(d), math.log(w), math.log(m)])
        targets.append(math.log(float(rv[t + 1])))
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class FittedHAR:
    """``eq=False``: numpy array fields make the dataclass default ``__eq__``
    raise (same trap documented on ``Origin`` in splitter.py); falls back to
    identity-based comparison instead of crashing.
    """

    beta: NDArray[np.float64]
    resid_var: float
    buffer: NDArray[np.float64]

    @property
    def name(self) -> str:
        return "har_rv"

    def spec(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "d_window": _D_WINDOW,
            "w_window": _W_WINDOW,
            "m_window": _M_WINDOW,
        }

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        buf = self.buffer.copy()
        rv_hat = 0.0
        for _ in range(h):
            d, w, m = _har_features(buf, buf.size - 1)
            y_hat = (
                self.beta[0]
                + self.beta[1] * math.log(d)
                + self.beta[2] * math.log(w)
                + self.beta[3] * math.log(m)
            )
            rv_hat = math.exp(y_hat + 0.5 * self.resid_var)
            buf = np.append(buf[1:], rv_hat)
        return Normal(mu=0.0, sigma=math.sqrt(rv_hat))

    def update(self, train: NDArray[np.float64]) -> FittedHAR:
        """Refresh the trailing RV lags under the fitted coefficients.

        ``beta`` and ``resid_var`` stay exactly as estimated at the last
        scheduled refit; only the buffer of the last 22 realized variances —
        the regressors of the next forecast — moves to the end of ``train``.
        Same validation as ``fit``: a non-positive or non-finite RV raises,
        which the evaluator records as an ``update_error`` row.
        """
        rv = _validated_rv(train, minimum=_M_WINDOW)
        return FittedHAR(beta=self.beta, resid_var=self.resid_var, buffer=rv[-_M_WINDOW:].copy())


@dataclass(frozen=True)
class HAR:
    """HAR-RV (Corsi 2009): OLS in logs on daily/weekly/monthly RV components."""

    @property
    def name(self) -> str:
        return "har_rv"

    def spec(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "d_window": _D_WINDOW,
            "w_window": _W_WINDOW,
            "m_window": _M_WINDOW,
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedHAR:
        rv = _validated_rv(train, minimum=_MIN_TRAIN)
        x, y = _design_matrix(rv)
        beta = np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float64)
        resid = y - x @ beta
        dof = max(x.shape[0] - _N_PARAMS, 1)
        resid_var = float(np.sum(resid * resid) / dof)
        buffer = rv[-_M_WINDOW:].copy()
        return FittedHAR(beta=beta, resid_var=resid_var, buffer=buffer)
