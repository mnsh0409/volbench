"""Random-walk volatility baseline.

Contract: `fit(train)` takes a 1-D array of returns. The forecast sigma is
the trailing RMS return over the fit window — the "no change" volatility
baseline. Returns are assumed zero-mean (standard for daily-return
volatility models, and consistent with the other baselines in this
package: EWMA and GARCH here fit `mean="Zero"` too), so the forecast RMS
return is exactly the trailing realized-volatility estimator, with no
separate mean-estimation step to leak information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal

__all__ = ["FittedNaiveVol", "NaiveVol"]

# A genuinely flat fit window (e.g. a thinly-traded asset) is a legitimate
# market state, not an error; Normal requires sigma > 0, so floor rather
# than raise.
_MIN_SIGMA = 1e-12


@dataclass(frozen=True)
class FittedNaiveVol:
    sigma: float

    @property
    def name(self) -> str:
        return "naive_rw_vol"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name}

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        return Normal(mu=0.0, sigma=self.sigma)


@dataclass(frozen=True)
class NaiveVol:
    """Random-walk volatility: forecast sigma = trailing RMS return."""

    @property
    def name(self) -> str:
        return "naive_rw_vol"

    def spec(self) -> dict[str, Any]:
        return {"model": self.name}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedNaiveVol:
        arr = np.asarray(train, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("train must be a 1-D array with at least 2 returns")
        sigma = max(float(np.sqrt(np.mean(arr * arr))), _MIN_SIGMA)
        return FittedNaiveVol(sigma=sigma)
