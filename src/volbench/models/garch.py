"""GARCH(1,1) / GJR-GARCH(1,1,1) via the `arch` package.

Contract: `fit(train)` takes a 1-D array of returns. Fit with `mean="Zero"`
(pure variance dynamics, consistent with the other baselines here).

Units: `arch_model` is fit with `rescale=True` unconditionally rather than
relying on its default auto-rescale (`rescale=None`, which rescales *and
warns* whenever it decides the data is poorly scaled) so the optimizer
always sees a well-conditioned series regardless of the caller's return
units. The catch: `res.forecast(...).variance` is reported in the
*rescaled* series' units, not the caller's — dividing by `res.scale ** 2`
before returning is required, or this would silently violate the
daily-units-only rule (CLAUDE.md rule 2).

Student-t innovations: the natural predictive distribution is a scaled
Student-t, for which `Distribution` has no closed-form constructor. We use
`Distribution.from_quantiles` over a fixed grid rather than
`from_samples`: it needs no RNG seed, so the same fitted model scores
bit-identically across runs (CLAUDE.md rule 3), and a fine grid's CRPS is
close to exact (Laio & Tamea, 2007) without Monte Carlo sampling error.

Non-convergence: if the optimizer fails to converge (`convergence_flag !=
0`, including a degenerate fitted `nu <= 2` for Student-t, where the
predictive variance would be undefined) or `fit()` raises outright, this
falls back to `EWMA(lambda_=fallback_lambda)` on the same training window
and records `fallback=True` on the fitted object, per the HARD RULES for
this stream: an origin must get a usable forecast, never a raised
exception that drops it from the grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from volbench.dist import Distribution, Normal
from volbench.models.ewma import EWMA, FittedEWMA

__all__ = ["GARCH", "FittedGARCH", "gjr_garch"]

logger = logging.getLogger(__name__)

_Dist = Literal["normal", "studentst"]
_MIN_TRAIN = 20
_MIN_NU = 2.02  # Student-t variance nu/(nu-2) blows up as nu -> 2
_N_QUANTILES = 199
_TAUS: NDArray[np.float64] = np.linspace(0.005, 0.995, _N_QUANTILES)


def _student_t_quantile_grid(sigma2: float, nu: float) -> Distribution:
    scale = math.sqrt(sigma2 * (nu - 2.0) / nu)
    values = scale * stats.t.ppf(_TAUS, df=nu)
    return Distribution.from_quantiles(_TAUS, values)


@dataclass(frozen=True)
class FittedGARCH:
    name: str
    o: int
    dist: _Dist
    fallback_lambda: float
    fallback: bool
    result: ARCHModelResult | None
    scale: float
    fallback_fit: FittedEWMA | None

    def spec(self) -> dict[str, Any]:
        return {
            "model": "gjr_garch" if self.o else "garch",
            "p": 1,
            "o": self.o,
            "q": 1,
            "dist": self.dist,
            "fallback_lambda": self.fallback_lambda,
        }

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        if self.fallback:
            assert self.fallback_fit is not None
            return self.fallback_fit.predict(h)
        assert self.result is not None

        forecast = self.result.forecast(horizon=h, reindex=False)
        sigma2 = float(forecast.variance.values[-1, h - 1]) / (self.scale**2)
        if self.dist == "normal":
            return Normal(mu=0.0, sigma=math.sqrt(sigma2))
        nu = float(self.result.params["nu"])
        return _student_t_quantile_grid(sigma2=sigma2, nu=nu)


@dataclass(frozen=True)
class GARCH:
    """GARCH(1,1) (o=0) / GJR-GARCH(1,1,1) (o=1), normal or Student-t innovations."""

    o: int = 0
    dist: _Dist = "normal"
    fallback_lambda: float = 0.94

    def __post_init__(self) -> None:
        if self.o not in (0, 1):
            raise ValueError("o must be 0 (GARCH) or 1 (GJR-GARCH)")
        if self.dist not in ("normal", "studentst"):
            raise ValueError("dist must be 'normal' or 'studentst'")
        if not 0.0 < self.fallback_lambda < 1.0:
            raise ValueError("fallback_lambda must lie strictly inside (0, 1)")

    @property
    def name(self) -> str:
        variant = "gjr_garch" if self.o else "garch"
        return f"{variant}(1,1)-{self.dist}"

    def spec(self) -> dict[str, Any]:
        return {
            "model": "gjr_garch" if self.o else "garch",
            "p": 1,
            "o": self.o,
            "q": 1,
            "dist": self.dist,
            "fallback_lambda": self.fallback_lambda,
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedGARCH:
        arr = np.asarray(train, dtype=np.float64)
        if arr.ndim != 1 or arr.size < _MIN_TRAIN:
            raise ValueError(f"train must be a 1-D array with at least {_MIN_TRAIN} returns")

        result: ARCHModelResult | None = None
        converged = False
        try:
            am = arch_model(
                arr, mean="Zero", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist, rescale=True
            )
            result = am.fit(disp="off", show_warning=False)
            converged = result.convergence_flag == 0
            if converged and self.dist == "studentst":
                converged = float(result.params["nu"]) > _MIN_NU
        except Exception:
            logger.warning("%s: fit raised, falling back to EWMA", self.name, exc_info=True)
            result = None
            converged = False

        if not converged:
            if result is not None:
                logger.warning(
                    "%s: optimizer did not converge (flag=%s), falling back to EWMA",
                    self.name,
                    result.convergence_flag,
                )
            fallback_fit = EWMA(lambda_=self.fallback_lambda).fit(arr)
            return FittedGARCH(
                name=self.name,
                o=self.o,
                dist=self.dist,
                fallback_lambda=self.fallback_lambda,
                fallback=True,
                result=None,
                scale=1.0,
                fallback_fit=fallback_fit,
            )

        assert result is not None
        return FittedGARCH(
            name=self.name,
            o=self.o,
            dist=self.dist,
            fallback_lambda=self.fallback_lambda,
            fallback=False,
            result=result,
            scale=float(result.scale),
            fallback_fit=None,
        )


def gjr_garch(dist: _Dist = "normal", fallback_lambda: float = 0.94) -> GARCH:
    return GARCH(o=1, dist=dist, fallback_lambda=fallback_lambda)
