"""Baseline forecast-model adapters (docs/design.md §Components, `ForecastModel`).

Every model here fits on a plain 1-D `numpy` array — never a `TimeSeriesFrame`
— and `predict(h)` returns a `Distribution` over the next-period return
(daily units, never annualized; CLAUDE.md rule 2). This package never imports
`volbench.data` or `volbench.evaluate`.

Deliberately NOT re-exported here: `volbench.models.sf` (AutoETS / AutoARIMA)
and `volbench.models.lgbm` (gradient boosting). Their backends live in the
optional `classical` extra, and importing them from this package root would
make `import volbench` fail for anyone who installed the core library — the
same reason `volbench.evaluate` imports `volbench.models.base` rather than
this module. Reach them by their own module path::

    from volbench.models.sf import AutoARIMARV, AutoETSRV
    from volbench.models.lgbm import LightGBMRV
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
