"""Invalid-target policy (D-018): what an unusable variance day does to a fit window.

An **invalid target day** is a day whose primary variance target is NaN or
non-positive. Two mechanisms produce them in the real panel
(docs/PANEL_REPORT.md §3, §4), and neither is a modelling choice:

- a bar whose close printed outside its own session high/low, whose
  range-based targets are therefore NaN (TWSE 80 days, CAC 28, HSI 1);
- a *monotone* bar (open at the high, close at the low, or the reverse) on a
  day whose open equals the previous close, where Rogers-Satchell and the
  overnight term are **both exactly zero**, so the D-016 target is exactly 0
  (HSI 12 days, NKX 2).

Such a day plays two roles, and D-018 treats them differently on purpose:

- **as a target** it stays exactly where it is. The row is produced, every
  score is NaN, and ``missing_reason`` names the cause — the results contract
  this project has kept since M1: nothing is ever dropped from the scored
  table, because a model that quietly scores on a shortened sample is a model
  whose numbers cannot be compared to another's.
- **as a fit input** it is dropped. ``log(0)`` is ``-inf`` and QLIKE is
  undefined at ``y = 0``; before D-018 a single zero inside a window crashed
  ``HAR.fit`` and cost every origin whose window contained it (up to
  ``window`` origins per zero — on HSI, up to 12,000 origin-model cells).

This module implements the second half. :class:`FitSeries` wraps the series a
model is fitted on, keeps it on the **full calendar** — the splitter's origins
are calendar positions and stay that way — and materializes each origin's
window on demand as *the last N valid observations at or before that origin*.

Temporal integrity (CLAUDE.md rule 1). Compaction reaches **backwards** and
only backwards. For a splitter window of ``N`` positions ending at ``origin``,
:meth:`FitSeries.window` returns values at ``N`` valid positions all ``<=
origin``; where invalid days sit inside the span, the window's *calendar*
extent stretches further into the past to recover the missing observations,
and its last position never moves past the origin. Validity at a position
depends on that position's own value alone, so no future observation can
change which earlier days a window contains — the canary in
``tests/test_compaction.py`` pins that directly.

Two consequences worth stating because they are easy to get wrong:

- **An invalid day is still a valid origin.** Its own target is unscorable,
  but its history is intact, so the forecast issued at it is a normal forecast
  from the last ``N`` valid observations before it; only the row whose
  *target* is that day carries a ``missing_reason``.
- **Lag semantics change.** For models that read positional lags of the fit
  series — HAR's ``RV_d``/``RV_w``/``RV_m``, LightGBM's 22 lags — "lag 1"
  means *the previous valid observation*, which after a drop is two or more
  calendar days back. That is the intended reading (the alternative is
  imputing a variance that was never measured), but it is a change to what
  those regressors mean and it is documented in ``docs/design.md`` and in
  each adapter's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_INVALID_TARGET_POLICY",
    "FitSeries",
    "InsufficientHistoryError",
    "InvalidTargetPolicy",
    "invalid_target_mask",
    "valid_target_mask",
]

#: What happens to an invalid target day inside a fit window.
#:
#: ``"compact"`` — D-018's policy: the day is dropped and the window reaches
#: further back, so every fit sees exactly ``N`` valid observations.
#: ``"none"`` — the window is the splitter's own positions, invalid values and
#: all. This is the pre-D-018 behaviour, kept as an explicit arm (and as what
#: a bare array means) rather than as something a caller can fall into: on a
#: series with an invalid day it is the arm where models crash.
InvalidTargetPolicy = Literal["compact", "none"]

#: The policy panel runs use (D-018). Named here so the study's series-loading
#: layer and the config hash agree on one spelling.
DEFAULT_INVALID_TARGET_POLICY: Final[InvalidTargetPolicy] = "compact"


class InsufficientHistoryError(ValueError):
    """A window asked for more valid observations than exist at or before its origin.

    Raised rather than silently handing a model a short window: "fit on 500
    observations" is a protocol parameter that the config hash records, and an
    origin that quietly fit on 497 would be reported as if it had not. The
    evaluator turns this into the standard NaN-plus-``missing_reason`` row, so
    the shortfall is visible in the results instead of being averaged into
    them.

    Structurally this can only bite the earliest origins of a series: validity
    accumulates, so the origins with too little history are a prefix ending at
    the ``N``-th valid observation.
    """


def valid_target_mask(values: object) -> NDArray[np.bool_]:
    """Which days carry a usable variance target: finite and strictly positive.

    The complement of D-018's *invalid target day*. Non-positive counts as
    invalid, not just NaN: a variance of exactly zero is where ``log RV``
    diverges and where QLIKE — ``v/y - log(v/y) - 1`` at proxy ``y`` — is
    undefined, and the days that produce it are a source-data artefact (a
    stale open meeting a monotone bar), not a market fact.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {array.shape}")
    return np.isfinite(array) & (array > 0.0)


def invalid_target_mask(values: object) -> NDArray[np.bool_]:
    """Days whose primary variance target is NaN or non-positive (D-018)."""
    return ~valid_target_mask(values)


@dataclass(frozen=True, eq=False)
class FitSeries:
    """The series a model is fitted on, plus the policy for its unusable days.

    Stays on the **full calendar**: ``values`` is positionally aligned with
    the returns and the proxy that :func:`~volbench.evaluate.run_backtest`
    scores, and the splitter's origins index into it directly. Compaction
    happens when a window is *materialized*, never by reshaping the series the
    splitter sees — so which days are scored is a property of the calendar and
    of nothing else.

    ``eq=False``: holds numpy arrays, whose element-wise ``==`` would make the
    dataclass-generated ``__eq__``/``__hash__`` raise (the trap documented on
    ``Origin`` in splitter.py). Identity comparison instead.

    Attributes
    ----------
    values:
        The series itself, on the full calendar, invalid days included.
    policy:
        :data:`InvalidTargetPolicy`. Recorded in the config hash whenever it
        binds, so the compacted and uncompacted arms can never collide.
    index:
        The pandas index ``values`` arrived with, if any. Held so
        ``run_backtest`` can check this series against the returns and the
        proxy on one calendar; this module never reads it.
    """

    values: NDArray[np.float64]
    policy: InvalidTargetPolicy
    index: object | None = None
    #: Ascending positions of the valid days. Derived once — every window is a
    #: slice of it, so materializing a window is a binary search, not a scan.
    valid_positions: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"fit series must be 1-D, got shape {array.shape}")
        if self.policy not in ("compact", "none"):
            raise ValueError(
                f"invalid-target policy must be 'compact' or 'none', got {self.policy!r}"
            )
        object.__setattr__(self, "values", array)
        object.__setattr__(
            self,
            "valid_positions",
            np.flatnonzero(valid_target_mask(array)).astype(np.int64),
        )

    # --- constructors ----------------------------------------------------

    @classmethod
    def of(
        cls,
        values: object,
        *,
        policy: InvalidTargetPolicy = DEFAULT_INVALID_TARGET_POLICY,
    ) -> FitSeries:
        """Wrap ``values`` (a bare array or a pandas Series) under ``policy``."""
        index = getattr(values, "index", None)
        return cls(values=np.asarray(values, dtype=np.float64), policy=policy, index=index)

    @classmethod
    def compact(cls, values: object) -> FitSeries:
        """D-018's default: drop invalid days, fit on the last N valid observations."""
        return cls.of(values, policy="compact")

    @classmethod
    def raw(cls, values: object) -> FitSeries:
        """No compaction — the splitter's own positions, invalid values and all."""
        return cls.of(values, policy="none")

    # --- properties ------------------------------------------------------

    @property
    def size(self) -> int:
        """Length on the full calendar."""
        return int(self.values.size)

    @property
    def n_valid(self) -> int:
        """How many days carry a usable target."""
        return int(self.valid_positions.size)

    @property
    def n_invalid(self) -> int:
        """How many days are invalid target days (D-018)."""
        return self.size - self.n_valid

    # --- windows ---------------------------------------------------------

    def window_positions(self, train: NDArray[np.int64]) -> NDArray[np.int64]:
        """Calendar positions this fit window is made of, for ``train``'s origin.

        ``train`` is a :class:`~volbench.splitter.Origin`'s own index array —
        this method never invents one. Only its length (how many observations
        the protocol asks for) and its last entry (the origin, the information
        cutoff) are read, which is exactly the pair "N observations, ending at
        t" that the splitter guarantees.

        Under ``policy="none"`` the answer is ``train`` itself. Under
        ``"compact"`` it is the last ``len(train)`` valid positions ``<=
        train[-1]`` — the same count, reaching further back where invalid days
        sit inside the span, and never past the origin.

        Raises
        ------
        InsufficientHistoryError
            If fewer than ``len(train)`` valid observations exist at or before
            the origin.
        """
        positions = np.asarray(train, dtype=np.int64)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError(f"train must be a non-empty 1-D index array, got {positions.shape}")
        if np.any(np.diff(positions) <= 0):
            raise ValueError("train indices must be strictly increasing")
        origin = int(positions[-1])
        if positions[0] < 0 or origin >= self.size:
            raise IndexError(
                f"train indices {int(positions[0])}..{origin} fall outside a fit series of "
                f"length {self.size}"
            )
        if self.policy == "none":
            return positions

        n = int(positions.size)
        # Everything at or before the origin, and nothing after it: the whole
        # temporal-integrity claim of this module is this one bound.
        available = int(np.searchsorted(self.valid_positions, origin, side="right"))
        if available < n:
            raise InsufficientHistoryError(
                f"only {available} valid observations at or before origin {origin}, but the "
                f"protocol asks for a {n}-observation fit window (invalid-target policy "
                f"'compact'; {origin + 1 - available} of the first {origin + 1} days are "
                "invalid target days). Fitting on a short window would report a window length "
                "the run did not use"
            )
        return self.valid_positions[available - n : available]

    def window(self, train: NDArray[np.int64]) -> NDArray[np.float64]:
        """The values a model is fitted on at ``train``'s origin.

        A fresh array in every case (numpy fancy indexing copies), so a model
        that writes through its input cannot corrupt the series behind it.
        """
        return self.values[self.window_positions(train)]

    def dropped_positions(self, train: NDArray[np.int64]) -> NDArray[np.int64]:
        """The days compaction removed from this window's calendar span.

        Every one of them lies strictly inside ``[first kept position,
        origin]`` and is therefore in the past of the origin — the property
        ``tests/test_compaction.py`` asserts for every origin of every series
        it exercises. Empty under ``policy="none"``.
        """
        kept = self.window_positions(train)
        if self.policy == "none":
            return np.empty(0, dtype=np.int64)
        span = np.arange(int(kept[0]), int(kept[-1]) + 1, dtype=np.int64)
        return np.setdiff1d(span, kept, assume_unique=True)
