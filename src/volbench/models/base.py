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
- `fit_diagnostics()` is OPTIONAL and reports how the fit *went* rather than
  what it produced (D-032). A model that estimates nothing does not
  implement it. Nothing here reaches `config_hash`: a diagnostic that
  changed a run's identity would make the act of observing a fit change
  which cell it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution

__all__ = [
    "FitDiagnostics",
    "FittedModel",
    "ForecastModel",
    "SupportsFitDiagnostics",
]


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


@dataclass(frozen=True)
class FitDiagnostics:
    """How one scheduled fit went — the part of a fit that is not a number.

    Volatility adapters degrade rather than raise: an origin must get a usable
    forecast, never an exception that drops it from the grid. That is the right
    call and it has one cost, which this closes — a model that quietly ran a
    different estimator on some origins scored exactly like one that did not.
    A GARCH-t cell that fell back to EWMA on 40 of 200 origins was, before
    D-032, indistinguishable in the results from one that fell back on none.

    ``fallback`` names the estimator that actually ran when it was not the one
    asked for, and is ``""`` when the model itself ran. ``converged`` is the
    optimizer's own verdict. ``detail`` is free-form evidence for a human
    reading a surprising row — a convergence flag, an estimated parameter at
    its bound — and is never parsed.

    :meth:`status` is the single string the evaluator records per row. The
    vocabulary is small and stable on purpose: it is a results column, so
    widening it later is a schema change, and ``"ok"`` must keep meaning
    exactly "the model asked for ran, and its optimizer said it converged".
    """

    converged: bool
    fallback: str = ""
    detail: str = ""

    def status(self) -> str:
        """Canonical token for the results column. ``""`` is never returned.

        ``ok`` | ``nonconverged`` | ``fallback=<name>`` | ``fallback=<name>|<detail>``
        — the empty string is reserved for *models that report nothing at all*,
        so an empty ``fit_status`` can never be confused with a clean fit.
        """
        if self.fallback:
            head = f"fallback={self.fallback}"
        elif self.converged:
            head = "ok"
        else:
            head = "nonconverged"
        detail = " ".join(self.detail.split())
        return f"{head}|{detail}" if detail else head

    @staticmethod
    def is_fallback(status: str) -> bool:
        """Whether a recorded ``fit_status`` says a fallback estimator ran."""
        return status.startswith("fallback=")

    @staticmethod
    def is_nonconverged(status: str) -> bool:
        """Whether a recorded ``fit_status`` says the optimizer did not converge.

        A fallback implies non-convergence for every adapter that has one
        today, so this counts both — a caller asking "how many fits did not
        converge?" must not get an answer that excludes the ones that gave up.
        """
        return status.startswith(("fallback=", "nonconverged"))


@runtime_checkable
class SupportsFitDiagnostics(Protocol):
    """A fitted model that can say how its fit went (D-032). Optional."""

    def fit_diagnostics(self) -> FitDiagnostics: ...
