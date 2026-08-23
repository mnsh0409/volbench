"""Rolling-origin index generation — the only sanctioned source of train/test splits.

Temporal-integrity invariant (docs/design.md, CLAUDE.md rule 1): no code path
may let information from t' > t influence a forecast for t. This module makes
that guarantee *structural*: every evaluation consumes `Origin` objects
produced here, and by construction ``max(train) == origin < min(test)``.

Hand-rolled slices in model or evaluation code are a review-blocking defect.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Origin", "RollingOriginSplitter"]


@dataclass(frozen=True, eq=False)
class Origin:
    """One forecast origin.

    ``eq=False``: this holds numpy array fields, and dataclass's default
    generated ``__eq__``/``__hash__`` compare field tuples element-wise,
    which raises (``ValueError`` on ``==``, ``TypeError`` on ``hash()``) for
    any array with more than one element. Falling back to identity-based
    comparison/hash keeps this usable in sets/dicts without a silent
    correctness trap; use ``np.array_equal`` on ``.train``/``.test``
    explicitly if you need value equality.

    Attributes
    ----------
    train:
        Indices available for fitting/context, ending at ``origin`` inclusive.
    origin:
        The information-set cutoff t: the last observation index a model may see.
    test:
        Target indices ``origin+1 .. origin+horizon`` (within series bounds).
    refit:
        Whether models should re-estimate parameters at this origin (zero-shot
        models ignore this and always condition on ``train`` only).
    """

    train: NDArray[np.int64]
    origin: int
    test: NDArray[np.int64]
    refit: bool


@dataclass(frozen=True)
class RollingOriginSplitter:
    """Generate rolling-origin evaluation splits over a series of length ``n``.

    Parameters
    ----------
    window:
        Number of trailing observations in each training set (rolling window).
    horizon:
        Forecast horizon: targets are ``origin+1 .. origin+horizon``.
    step:
        Origin advance between consecutive splits.
    refit_every:
        Mark ``refit=True`` every this-many *origins* (the first origin always
        refits). ``1`` refits everywhere.
    """

    window: int
    horizon: int = 1
    step: int = 1
    refit_every: int = 1

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")

    def n_splits(self, n: int) -> int:
        first = self.window - 1
        last = n - 1 - self.horizon
        if last < first:
            return 0
        return (last - first) // self.step + 1

    def split(self, n: int) -> Iterator[Origin]:
        """Yield :class:`Origin` objects for a series of length ``n``.

        The first origin is the earliest index with a full window behind it;
        the last is the latest whose full horizon fits inside the series.
        """
        if n <= self.window + self.horizon - 1:
            raise ValueError(
                f"series too short: n={n} <= window+horizon-1={self.window + self.horizon - 1}"
            )
        first = self.window - 1
        last = n - 1 - self.horizon
        for k, origin in enumerate(range(first, last + 1, self.step)):
            train = np.arange(origin - self.window + 1, origin + 1, dtype=np.int64)
            test = np.arange(origin + 1, origin + 1 + self.horizon, dtype=np.int64)
            yield Origin(
                train=train,
                origin=origin,
                test=test,
                refit=(k % self.refit_every == 0),
            )
