"""Unified probabilistic forecast objects.

Every forecaster adapter in volbench returns a :class:`Distribution` — never a
bare point array. This module is the single currency for probabilistic
forecasts across model families that natively emit parameters (GARCH),
quantiles (quantile regressors, TSFMs), or samples (MC-based models).

Design invariant (docs/design.md §Components): downstream evaluation code
depends only on this interface, so adding a model family never touches the
evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Distribution", "Empirical", "Normal", "QuantileGrid"]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _phi(z: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: float) -> float:
    """Standard normal cdf."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


@dataclass(frozen=True, eq=False)
class Distribution:
    """Abstract probabilistic forecast for a single target.

    Concrete constructors:

    - :meth:`from_normal`    — parametric N(mu, sigma^2)
    - :meth:`from_samples`   — empirical ensemble
    - :meth:`from_quantiles` — quantile grid (tau -> value)

    All scores are negatively oriented (smaller is better).

    ``eq=False`` here (no own fields either way) matters for subclasses:
    a dataclass subclass declared ``eq=False`` does not generate its own
    ``__eq__``, so it inherits one via MRO. If this base class generated
    the default field-tuple ``__eq__`` (fields=() since Distribution has
    none), every subclass without its own ``__eq__`` would inherit an
    ``__eq__`` that always returns True for same-class instances,
    regardless of the subclass's actual field values — a silent
    wrong-answer bug, not just a crash. Leaving this base eq=False lets
    Empirical/QuantileGrid fall through to identity-based object.__eq__
    instead. Normal keeps the dataclass default (eq=True) since its
    fields are plain floats, so value equality there is correct and safe.
    """

    @staticmethod
    def from_normal(mu: float, sigma: float) -> Normal:
        return Normal(mu=float(mu), sigma=float(sigma))

    @staticmethod
    def from_samples(samples: Sequence[float] | NDArray[np.float64]) -> Empirical:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("samples must be a 1-D array with at least 2 values")
        if not np.isfinite(arr).all():
            raise ValueError("samples must be finite")
        return Empirical(samples=np.sort(arr))

    @staticmethod
    def from_quantiles(
        taus: Sequence[float] | NDArray[np.float64],
        values: Sequence[float] | NDArray[np.float64],
    ) -> QuantileGrid:
        t = np.asarray(taus, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if t.shape != v.shape or t.ndim != 1 or t.size < 2:
            raise ValueError("taus and values must be equal-length 1-D arrays (size >= 2)")
        if not ((t > 0.0) & (t < 1.0)).all():
            raise ValueError("taus must lie strictly inside (0, 1)")
        order = np.argsort(t)
        t, v = t[order], v[order]
        if np.unique(t).size != t.size:
            raise ValueError("taus must be distinct")
        if (np.diff(v) < 0).any():
            raise ValueError("quantile values must be non-decreasing in tau")
        return QuantileGrid(taus=t, values=v)

    # --- interface -------------------------------------------------------

    def quantile(self, tau: float) -> float:
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        raise NotImplementedError

    def crps(self, y: float) -> float:
        raise NotImplementedError

    def log_score(self, y: float) -> float:
        """Negative log predictive density at ``y``.

        Only defined for distributions with a density (parametric families).
        """
        raise NotImplementedError(f"{type(self).__name__} has no tractable density")

    def sample(self, n: int, seed: int) -> NDArray[np.float64]:
        raise NotImplementedError

    def pinball(self, y: float, tau: float) -> float:
        """Quantile (pinball) loss of this forecast's tau-quantile at ``y``."""
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly inside (0, 1)")
        q = self.quantile(tau)
        u = y - q
        return tau * u if u >= 0.0 else (tau - 1.0) * u


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian predictive distribution N(mu, sigma^2)."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu) or not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("require finite mu and sigma > 0")

    def quantile(self, tau: float) -> float:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly inside (0, 1)")
        # Acklam/Peter John Acklam-style inverse via numpy for robustness.
        from statistics import NormalDist

        return float(NormalDist(mu=self.mu, sigma=self.sigma).inv_cdf(tau))

    def cdf(self, x: float) -> float:
        return _Phi((x - self.mu) / self.sigma)

    def crps(self, y: float) -> float:
        """Closed form (Gneiting & Raftery, 2007, eq. 21)."""
        z = (y - self.mu) / self.sigma
        return self.sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRT_PI)

    def log_score(self, y: float) -> float:
        z = (y - self.mu) / self.sigma
        return 0.5 * z * z + math.log(self.sigma) + 0.5 * math.log(2.0 * math.pi)

    def sample(self, n: int, seed: int) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mu, self.sigma, size=n)


@dataclass(frozen=True, eq=False)
class Empirical(Distribution):
    """Ensemble/empirical predictive distribution (stored sorted).

    ``eq=False``: ``samples`` is a numpy array, and dataclass's default
    generated ``__eq__``/``__hash__`` raise on arrays with more than one
    element (see ``Origin`` in splitter.py for the same trap). Falls back
    to identity-based comparison/hash instead of crashing.
    """

    samples: NDArray[np.float64]

    def quantile(self, tau: float) -> float:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly inside (0, 1)")
        return float(np.quantile(self.samples, tau))

    def cdf(self, x: float) -> float:
        return float(np.searchsorted(self.samples, x, side="right")) / self.samples.size

    def crps(self, y: float) -> float:
        """Exact ensemble CRPS: E|X - y| - 0.5 E|X - X'|.

        The pairwise term uses the sorted-sample identity
        ``sum_{i<j}(x_(j) - x_(i)) = sum_i (2i - n + 1) x_(i)`` (0-indexed),
        giving O(n log n) overall (samples are stored sorted).
        """
        x = self.samples
        n = x.size
        term1 = float(np.mean(np.abs(x - y)))
        idx = np.arange(n, dtype=np.float64)
        pairwise_sum = float(np.sum((2.0 * idx - n + 1.0) * x))  # sum over i<j of gaps
        term2 = pairwise_sum / (n * n)
        return term1 - term2

    def sample(self, n: int, seed: int) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.choice(self.samples, size=n, replace=True)


@dataclass(frozen=True, eq=False)
class QuantileGrid(Distribution):
    """Predictive distribution known only through a finite quantile grid.

    Between grid points, quantiles interpolate linearly; outside the grid the
    edge values are returned (flat extrapolation, documented limitation —
    tail scores below min(tau) rely on the edge quantile).

    ``eq=False``: ``taus``/``values`` are numpy arrays, so the default
    generated ``__eq__``/``__hash__`` would raise instead of comparing (see
    ``Origin`` in splitter.py). Falls back to identity-based comparison/hash.
    """

    taus: NDArray[np.float64]
    values: NDArray[np.float64]

    def quantile(self, tau: float) -> float:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly inside (0, 1)")
        return float(np.interp(tau, self.taus, self.values))

    def cdf(self, x: float) -> float:
        if x <= self.values[0]:
            return float(self.taus[0]) if x >= self.values[0] else 0.0
        if x >= self.values[-1]:
            return 1.0 if x > self.values[-1] else float(self.taus[-1])
        return float(np.interp(x, self.values, self.taus))

    def crps(self, y: float) -> float:
        """Pinball-based approximation: CRPS = 2 * integral of pinball over tau.

        Uses the trapezoidal rule on the available grid — exact as the grid
        densifies (Laio & Tamea, 2007). Grids should span tails of interest.
        """
        losses = np.array([self.pinball(y, float(t)) for t in self.taus])
        return 2.0 * float(np.trapezoid(losses, self.taus))

    def sample(self, n: int, seed: int) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        u = rng.uniform(float(self.taus[0]), float(self.taus[-1]), size=n)
        return np.interp(u, self.taus, self.values)
