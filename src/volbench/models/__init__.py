"""Baseline forecast-model adapters (docs/design.md §Components, `ForecastModel`).

Every model here fits on a plain 1-D `numpy` array — never a `TimeSeriesFrame`
— and `predict(h)` returns a `Distribution` over the next-period return
(daily units, never annualized; CLAUDE.md rule 2). This package never imports
`volbench.data` or `volbench.evaluate`.

Every adapter whose backend is optional — the classical `AutoETSRV` /
`AutoARIMARV` (statsforecast) and `LightGBMRV` (lightgbm) in the `classical`
extra; the zero-shot foundation-model adapters `Chronos`, `TimesFM`,
`Moirai`, `TimeGPT` (see `tsfm_common.py` for the shared contract) and the
trained `PatchTST` baseline in the `tsfm` / `torch-cpu` extras — imports that
backend lazily inside `fit`, so importing this package (and therefore
`import volbench`) never needs any extra installed. What fails without the
extra is the first `fit`, with the backend's own `ImportError`.
"""

from __future__ import annotations

from volbench.models.base import FittedModel, ForecastModel
from volbench.models.ewma import EWMA, FittedEWMA
from volbench.models.garch import GARCH, FittedGARCH, gjr_garch
from volbench.models.har import HAR, FittedHAR
from volbench.models.naive import FittedNaiveVol, NaiveVol
from volbench.models.patchtst import FittedPatchTST, PatchTST
from volbench.models.tsfm_chronos import Chronos
from volbench.models.tsfm_common import FittedTSFM, TSFMBackend, ZeroShotRVModel
from volbench.models.tsfm_moirai import Moirai
from volbench.models.tsfm_timegpt import TimeGPT
from volbench.models.tsfm_timesfm import TimesFM, TimesFMForecastOptions

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
    "FittedPatchTST",
    "FittedTSFM",
    "ForecastModel",
    "Moirai",
    "NaiveVol",
    "PatchTST",
    "TSFMBackend",
    "TimeGPT",
    "TimesFM",
    "TimesFMForecastOptions",
    "ZeroShotRVModel",
    "gjr_garch",
]
