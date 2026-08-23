"""Model adapter protocol (docs/design.md §Components, `ForecastModel`).

Concrete model classes (naive.py, ewma.py, garch.py, har.py) are plain
dataclasses that structurally satisfy these protocols — no shared base class
is required or provided.

Contract:

- `fit()` takes only a plain 1-D array of information available up to and
  including the fit window (never a `TimeSeriesFrame` — this package has no
  dependency on `volbench.data`, keeping the adapter boundary clean).
- `predict(h)` returns a `Distribution` over the NEXT-PERIOD RETURN, `h`
  steps past the fit window's end (CLAUDE.md convention: a model's variance
  forecast is a property of that distribution, never a bare number).
- `spec()` returns the model's hyperparameters as a JSON-serializable dict.
  The evaluation stream hashes this into `config_hash` (docs/design.md
  `ResultsStore`), so it must be stable across identical constructions and
  differ whenever a hyperparameter differs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution

__all__ = ["FittedModel", "ForecastModel"]


@runtime_checkable
class FittedModel(Protocol):
    """A model conditioned on a fit window, ready to forecast forward."""

    @property
    def name(self) -> str: ...

    def spec(self) -> dict[str, Any]: ...

    def predict(self, h: int) -> Distribution: ...


@runtime_checkable
class ForecastModel(Protocol):
    """A model specification, not yet conditioned on data."""

    @property
    def name(self) -> str: ...

    def spec(self) -> dict[str, Any]: ...

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedModel: ...
