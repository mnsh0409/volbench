"""RiskMetrics EWMA volatility.

Contract: `fit(train)` takes a 1-D array of returns.

Recursion (RiskMetrics 1996): sigma^2_t = lambda * sigma^2_{t-1} +
(1 - lambda) * r^2_{t-1}, seeded with sigma^2_1 = r_0^2 (the only variance
estimate available before any lagged variance exists — using only r_0,
never information beyond it). Iterating through the whole fit window gives
sigma^2_n, the forecast for the first out-of-sample return.

Horizon: RiskMetrics EWMA is an IGARCH(1,1) with no mean reversion, so its
conditional-variance forecast does not depend on h; `predict(h)` returns
the same sigma for every h >= 1 (RiskMetrics Technical Document, 1996,
§5.3 — documented limitation, not a bug).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal

__all__ = ["EWMA", "FittedEWMA"]

# See naive.py: a genuinely flat fit window is a legitimate market state,
# not an error; Normal requires sigma > 0, so floor rather than raise.
_MIN_VARIANCE = 1e-24


def _ewma_variance(train: NDArray[np.float64], lambda_: float) -> float:
    """Run the RiskMetrics recursion over ``train``; the last value is the forecast."""
    arr = np.asarray(train, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("train must be a 1-D array with at least 2 returns")
    var = float(arr[0] * arr[0])
    for r in arr[1:]:
        var = lambda_ * var + (1.0 - lambda_) * float(r * r)
    return max(var, _MIN_VARIANCE)


@dataclass(frozen=True)
class FittedEWMA:
    sigma2: float
    lambda_: float

    @property
    def name(self) -> str:
        return "ewma"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name, "lambda": self.lambda_}

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        return Normal(mu=0.0, sigma=math.sqrt(self.sigma2))

    def update(self, train: NDArray[np.float64]) -> FittedEWMA:
        """Re-run the recursion over ``train`` at the same ``lambda``.

        ``lambda`` is a fixed hyperparameter, never estimated, so this
        re-conditions without estimating anything. The recursion is seeded
        from the window's first squared return exactly as in ``fit``, so
        ``update`` on the fit window reproduces the fit bit for bit.
        """
        return FittedEWMA(sigma2=_ewma_variance(train, self.lambda_), lambda_=self.lambda_)


@dataclass(frozen=True)
class EWMA:
    """RiskMetrics EWMA volatility, default lambda = 0.94."""

    lambda_: float = 0.94

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_ < 1.0:
            raise ValueError("lambda_ must lie strictly inside (0, 1)")

    @property
    def name(self) -> str:
        return "ewma"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name, "lambda": self.lambda_}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedEWMA:
        return FittedEWMA(sigma2=_ewma_variance(train, self.lambda_), lambda_=self.lambda_)
