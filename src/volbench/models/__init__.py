"""Baseline forecast-model adapters (docs/design.md §Components, `ForecastModel`).

Every model here fits on a plain 1-D `numpy` array — never a `TimeSeriesFrame`
— and `predict(h)` returns a `Distribution` over the next-period return
(daily units, never annualized; CLAUDE.md rule 2). This package never imports
`volbench.data` or `volbench.evaluate`.

The zero-shot foundation-model adapters (`Chronos`, `TimesFM`, `Moirai`,
`TimeGPT`; see `tsfm_common.py` for the shared contract) import their heavy
backends lazily, so importing this package never needs the `tsfm` extra.
"""

from __future__ import annotations

from volbench.models.base import FittedModel, ForecastModel
from volbench.models.ewma import EWMA, FittedEWMA
from volbench.models.garch import GARCH, FittedGARCH, gjr_garch
from volbench.models.har import HAR, FittedHAR
from volbench.models.naive import FittedNaiveVol, NaiveVol
from volbench.models.tsfm_chronos import Chronos
from volbench.models.tsfm_common import FittedTSFM, TSFMBackend, ZeroShotRVModel

__all__ = [
    "EWMA",
    "GARCH",
    "HAR",
    "Chronos",
    "FittedEWMA",
    "FittedGARCH",
    "FittedHAR",
    "FittedModel",
    "FittedNaiveVol",
    "FittedTSFM",
    "ForecastModel",
    "NaiveVol",
    "TSFMBackend",
    "ZeroShotRVModel",
    "gjr_garch",
]
