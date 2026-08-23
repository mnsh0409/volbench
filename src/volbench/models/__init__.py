"""Baseline forecast-model adapters (docs/design.md §Components, `ForecastModel`).

Every model here fits on a plain 1-D `numpy` array — never a `TimeSeriesFrame`
— and `predict(h)` returns a `Distribution` over the next-period return
(daily units, never annualized; CLAUDE.md rule 2). This package never imports
`volbench.data` or `volbench.evaluate`.
"""

from __future__ import annotations

from volbench.models.base import FittedModel, ForecastModel
from volbench.models.ewma import EWMA, FittedEWMA
from volbench.models.garch import GARCH, FittedGARCH, gjr_garch
from volbench.models.har import HAR, FittedHAR
from volbench.models.naive import FittedNaiveVol, NaiveVol

__all__ = [
    "EWMA",
    "GARCH",
    "HAR",
    "FittedEWMA",
    "FittedGARCH",
    "FittedHAR",
    "FittedModel",
    "FittedNaiveVol",
    "ForecastModel",
    "NaiveVol",
    "gjr_garch",
]
