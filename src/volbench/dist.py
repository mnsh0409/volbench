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
from scipy import special, stats  # type: ignore[import-untyped]

__all__ = ["Distribution", "Empirical", "Normal", "QuantileGrid", "StudentT"]

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
    - :meth:`from_student_t` — parametric location-scale Student-t
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
    def from_student_t(loc: float, scale: float, df: float) -> StudentT:
        return StudentT(loc=float(loc), scale=float(scale), df=float(df))

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

    def mean(self) -> float:
        """Mean of the predictive law, in closed form.

        Implemented by parametric families only. Non-parametric objects
        (``Empirical``, ``QuantileGrid``) deliberately do not estimate one
        here: the evaluator owns that fallback and documents its bias.
        """
        raise NotImplementedError(f"{type(self).__name__} has no closed-form mean")

    def variance(self) -> float:
        """Variance of the predictive law, in closed form.

        For a distribution over the next-period return this *is* the variance
        forecast (CLAUDE.md rule 2), which is why it lives on the object rather
        than being re-derived downstream from a quantile grid.
        """
        raise NotImplementedError(f"{type(self).__name__} has no closed-form variance")

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

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma * self.sigma

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


@dataclass(frozen=True)
class StudentT(Distribution):
    """Location-scale Student-t predictive distribution ``loc + scale * T(df)``.

    ``scale`` is the t's scale parameter, *not* its standard deviation: the
    standard deviation is ``scale * sqrt(df / (df - 2))``. Build from a target
    variance — a GARCH conditional variance, say — with :meth:`from_variance`,
    which does that conversion and round-trips exactly through
    :meth:`variance`.

    ``df > 1`` is required. Below that the law has no mean and an infinite
    CRPS, so nothing here could score it. For ``1 < df <= 2`` the mean exists
    but the second moment diverges; :meth:`variance` raises rather than
    returning ``inf``, because in volbench a return distribution's variance IS
    the variance forecast (CLAUDE.md rule 2), and an infinite one is not a
    forecast.

    Why this exists: until it did, Student-t GARCH forecasts were a 199-point
    quantile grid over tau in [0.005, 0.995], and the evaluator's moments of
    that grid truncated the tails — a *perfectly specified* forecast could not
    reach QLIKE 0, with a floor of 0.0407 at nu=3 (docs/M1_REPORT.md §4.2).
    Closed-form moments and CRPS remove the bias at its source, with no RNG,
    so scoring stays bit-identical across runs (CLAUDE.md rule 3).

    Plain float fields, so the dataclass default value-``__eq__``/``__hash__``
    are correct (same reasoning as ``Normal``).
    """

    loc: float
    scale: float
    df: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.loc) and math.isfinite(self.scale) and math.isfinite(self.df)):
            raise ValueError("require finite loc, scale and df")
        if self.scale <= 0.0:
            raise ValueError("require scale > 0")
        if self.df <= 1.0:
            raise ValueError(
                f"require df > 1 (got df={self.df}): a Student-t with df <= 1 has no mean "
                "and an infinite CRPS, so it cannot be scored"
            )

    @classmethod
    def from_variance(cls, loc: float, variance: float, df: float) -> StudentT:
        """Build from a target *variance* instead of a scale.

        ``scale = sqrt(variance * (df - 2) / df)``, so that
        ``StudentT.from_variance(loc, v, df).variance() == v``. Needs
        ``df > 2`` — a finite variance does not exist otherwise.
        """
        if not (math.isfinite(variance) and variance > 0.0):
            raise ValueError("require finite variance > 0")
        if not (math.isfinite(df) and df > 2.0):
            raise ValueError(
                f"a Student-t with a finite variance needs df > 2 (got df={df}); "
                "for df <= 2 the variance is undefined and no scale reproduces it"
            )
        return cls(loc=float(loc), scale=math.sqrt(variance * (df - 2.0) / df), df=float(df))

    def _z(self, x: float) -> float:
        return (x - self.loc) / self.scale

    def quantile(self, tau: float) -> float:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must lie strictly inside (0, 1)")
        return self.loc + self.scale * float(stats.t.ppf(tau, df=self.df))

    def cdf(self, x: float) -> float:
        return float(stats.t.cdf(self._z(x), df=self.df))

    def mean(self) -> float:
        return self.loc  # df > 1 is guaranteed by __post_init__

    def variance(self) -> float:
        if self.df <= 2.0:
            raise ValueError(
                f"StudentT variance is undefined for df <= 2 (got df={self.df}): the second "
                "moment diverges, so this object cannot supply a variance forecast"
            )
        return self.scale * self.scale * self.df / (self.df - 2.0)

    def crps(self, y: float) -> float:
        """Closed form for the location-scale t, ``df > 1``.

        Jordan, Krüger & Lerch (2019, *J. Stat. Softw.* 90(12), Appendix A):
        with ``z = (y - loc) / scale``, ``F``/``f`` the standard-t cdf/pdf and
        ``B`` the beta function,

            CRPS = scale * [ z (2F(z) - 1) + 2 f(z) (df + z^2)/(df - 1)
                             - 2 sqrt(df)/(df - 1) * B(1/2, df - 1/2) / B(1/2, df/2)^2 ]

        The beta ratio is evaluated through ``betaln`` so large ``df`` cannot
        overflow; as ``df -> inf`` it tends to ``1/sqrt(pi)`` and the whole
        expression to the Gaussian closed form. ``tests/test_dist.py`` checks
        this against numerical integration of the quantile representation.
        """
        nu = self.df
        z = self._z(y)
        cdf = float(stats.t.cdf(z, df=nu))
        pdf = float(stats.t.pdf(z, df=nu))
        beta_ratio = math.exp(
            float(special.betaln(0.5, nu - 0.5)) - 2.0 * float(special.betaln(0.5, nu / 2.0))
        )
        constant = 2.0 * math.sqrt(nu) / (nu - 1.0) * beta_ratio
        return self.scale * (
            z * (2.0 * cdf - 1.0) + 2.0 * pdf * (nu + z * z) / (nu - 1.0) - constant
        )

    def log_score(self, y: float) -> float:
        # Density of the location-scale law: f_df(z) / scale.
        return -(float(stats.t.logpdf(self._z(y), df=self.df)) - math.log(self.scale))

    def sample(self, n: int, seed: int) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return self.loc + self.scale * rng.standard_t(self.df, size=n)


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
