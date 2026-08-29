"""Comparison inference on volbench scores: Diebold-Mariano and the MCS.

This module answers "who wins?" over the loss arrays that
:func:`volbench.evaluate.run_backtest` produces and
:class:`volbench.results.ResultsStore` persists. It only *consumes* scored
rows: nothing here touches the scored path, so no config hash and no stored
fragment can change because this module exists.

Two tools, deliberately kept together (docs/metrics_reference.md, "Comparison
inference"):

- :func:`diebold_mariano` — the pairwise test of equal predictive accuracy on
  a loss differential (Diebold & Mariano 1995) with the Harvey, Leybourne &
  Newbold (1997) small-sample correction. Pairwise p-values are **not**
  corrected for multiplicity; with ``m`` models there are ``m(m-1)/2`` of
  them, and the family-wise false-discovery rate grows accordingly.
- :func:`model_confidence_set` — the Model Confidence Set of Hansen, Lunde &
  Nason (2011): sequential elimination with a block-bootstrap null, yielding
  the set that contains the best model(s) with confidence ``1 - alpha``. This
  is the primary "who wins" tool; :func:`compare_models` returns it together
  with the DM matrix so the two are never reported apart.

Conventions
-----------
- Losses are **negatively oriented** (smaller is better), as every volbench
  score is: CRPS, log score (``-ln f(y)``), QLIKE, pinball and the FZ0 loss.
- Time order matters. The block bootstrap resamples *contiguous windows of
  the original time axis*, so every loss matrix must be in ascending origin
  order; :func:`loss_matrix` builds one from result rows and sorts it.
- **NaN policy.** A row with a ``missing_reason`` (or a non-finite loss) is
  never scored as zero and never silently skipped: the DM test drops it
  pairwise-complete, the MCS drops it listwise (one bootstrap index sequence
  has to serve every model in the set), and every result records how many
  origins that cost (``n_dropped``).
- **Determinism** (CLAUDE.md rule 3). The bootstrap takes a mandatory
  ``seed``; every result carries a ``config_hash`` over its inputs and
  settings, computed with the same :func:`volbench.results.config_hash` that
  identifies benchmark cells.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy.
*Journal of Business & Economic Statistics* 13(3), 253-263.

Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
prediction mean squared errors. *International Journal of Forecasting* 13(2),
281-291.

Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set.
*Econometrica* 79(2), 453-497; bootstrap implementation in their separate
appendix and in Hansen, Lunde & Nason (2003), *Choosing the best volatility
models: the model confidence set approach*, Brown University WP 2003-05,
Appendix B.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
observations. *Annals of Statistics* 17(3), 1217-1241 (moving block bootstrap).

Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
covariance matrix estimation. *Econometrica* 59(3), 817-858 (the AR(1)
plug-in bandwidth, eq. 6.4, and the Bartlett kernel's 1.1447 (alpha(1) T)^{1/3}).

Andrews, D. W. K. & Monahan, J. C. (1992). An improved heteroskedasticity and
autocorrelation consistent covariance matrix estimator. *Econometrica* 60(4),
953-966 (AR(1) pre-whitening and recolouring; the 0.97 cap on the coefficient).

Hansen, P. R., Lunde, A. & Nason, J. M. (2003), as above, and Hansen, P. R. &
Lunde, A. (2005). A forecast comparison of volatility models: does anything
beat a GARCH(1,1)? *Journal of Applied Econometrics* 20(7), 873-889 (the
semi-quadratic statistic ``T_SQ = sum_{i<j} t_ij^2``).

Politis, D. N. & White, H. (2004). Automatic block-length selection for the
dependent bootstrap. *Econometric Reviews* 23(1), 53-70; correction: Patton,
A., Politis, D. N. & White, H. (2009), *Econometric Reviews* 28(4), 372-375.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from volbench.results import ResultsStore, array_digest, config_hash, package_version

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_N_BOOT",
    "MAX_PREWHITEN_RHO",
    "Alternative",
    "DMMatrix",
    "DMResult",
    "HACSpec",
    "Kernel",
    "LongRunVariance",
    "LossMatrix",
    "MCSResult",
    "MCSStatistic",
    "MissingPolicy",
    "ModelComparison",
    "andrews_bandwidth",
    "ar1_block_length",
    "ar1_coefficient",
    "autocorrelation",
    "bootstrap_column_means",
    "compare_models",
    "default_block_length",
    "diebold_mariano",
    "dm_matrix",
    "effective_sample_size",
    "long_run_variance",
    "loss_matrix",
    "model_confidence_set",
    "moving_block_indices",
    "politis_white_block_length",
    "rule_of_thumb_bandwidth",
]

#: MCS confidence level complement, per docs/metrics_reference.md.
DEFAULT_ALPHA: Final = 0.10
#: Bootstrap resamples for the MCS, per docs/metrics_reference.md ("B = [10k]").
DEFAULT_N_BOOT: Final = 10_000

#: Lag window for the long-run variance of the loss differential.
#: ``"rectangular"`` is the Diebold-Mariano estimator (unit weight on the
#: ``h-1`` autocovariances a ``h``-step forecast error can carry); it is the
#: one Harvey, Leybourne & Newbold's correction is derived for, but it is not
#: guaranteed positive. ``"bartlett"`` uses Newey-West weights ``1 - k/h``,
#: which cannot go negative, at the cost of a slightly different estimator.
Kernel = Literal["rectangular", "bartlett"]
#: ``"two-sided"``: E[d] != 0. ``"less"``: E[d] < 0, i.e. ``loss_a`` is the
#: smaller (better) loss. ``"greater"``: E[d] > 0.
Alternative = Literal["two-sided", "less", "greater"]
#: ``"range"`` — ``T_R = max_{i,j} |t_ij|`` with elimination rule
#: ``arg max_i sup_j t_ij``; ``"max"`` — ``T_max = max_i t_i.`` with
#: elimination rule ``arg max_i t_i.`` (Hansen, Lunde & Nason 2011, §3.1.2);
#: ``"semi_quadratic"`` — ``T_SQ = sum_{i<j} t_ij^2`` (Hansen, Lunde & Nason
#: 2003; Hansen & Lunde 2005), a function of the same ``t_ij`` as the range
#: statistic and paired here with the same elimination rule
#: ``arg max_i sup_j t_ij``, so the two differ only in how the evidence across
#: pairs is pooled: the largest pair, or all of them squared.
MCSStatistic = Literal["range", "max", "semi_quadratic"]
#: Which result rows :func:`loss_matrix` treats as missing. ``"flagged"``:
#: any row carrying a ``missing_reason`` (the policy in the Phase 2 brief —
#: one set of origins for every score of a model set). ``"score"``: only rows
#: whose requested score is NaN.
MissingPolicy = Literal["flagged", "score"]

#: Resamples drawn per RNG call. Part of the bootstrap's identity: numpy's
#: bounded-integer sampler buffers 32-bit words *within* a call, so drawing the
#: same starts in different chunk sizes gives different streams. Fixed here so
#: :func:`moving_block_indices` and the internal block-sum path agree exactly.
_BOOT_CHUNK: Final = 256
#: Elements per temporary in the block-sum gather (bounds memory, not results).
_GATHER_ELEMENTS: Final = 1 << 21


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def _as_1d(values: object, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    return array


@dataclass(frozen=True, eq=False)
class LossMatrix:
    """One score for a set of models over one time axis, built from result rows.

    ``values`` has one row per origin (ascending) and one column per model.
    A NaN marks an origin that is *not usable* for that model under the
    :data:`MissingPolicy` that built the matrix; nothing has been dropped
    yet. The inference routines do the dropping — pairwise-complete for DM,
    listwise for the MCS — and each records what it cost, so exclusion
    happens exactly once and is always visible.

    ``n_flagged`` counts, per model, the origins marked unusable.
    ``config_hashes`` records which stored cell each column came from.
    """

    values: pd.DataFrame
    score: str
    asset: str
    horizon: int
    n_flagged: Mapping[str, int]
    config_hashes: Mapping[str, str]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.values.columns)


def _resolve_losses(
    losses: LossMatrix | pd.DataFrame | NDArray[np.float64] | Sequence[Sequence[float]],
    model_names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """``(n x m float array, model names)`` from any accepted loss input."""
    if isinstance(losses, LossMatrix):
        frame = losses.values
        names = losses.models
        matrix = np.asarray(frame.to_numpy(), dtype=np.float64)
    elif isinstance(losses, pd.DataFrame):
        names = tuple(str(c) for c in losses.columns)
        matrix = np.asarray(losses.to_numpy(), dtype=np.float64)
    else:
        matrix = np.asarray(losses, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"losses must be a 2-D (origins x models) array, got {matrix.shape}")
        names = (
            tuple(f"model_{j}" for j in range(matrix.shape[1]))
            if model_names is None
            else tuple(str(name) for name in model_names)
        )
    if (
        model_names is not None
        and isinstance(losses, LossMatrix | pd.DataFrame)
        and tuple(str(name) for name in model_names) != names
    ):
        raise ValueError("model_names must match the column labels of a DataFrame/LossMatrix")
    if matrix.ndim != 2:
        raise ValueError(f"losses must be 2-D (origins x models), got shape {matrix.shape}")
    if len(names) != matrix.shape[1]:
        raise ValueError(f"{len(names)} model names for {matrix.shape[1]} loss columns")
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be distinct, got {names}")
    return matrix, names


# --------------------------------------------------------------------------
# long-run variance: pre-whitened Bartlett HAC with a data-driven bandwidth
# --------------------------------------------------------------------------

#: Andrews & Monahan (1992, §3): the pre-whitening AR coefficient is held
#: below one in absolute value so that the recolouring factor ``1/(1-rho)^2``
#: stays finite; their eigenvalue adjustment reduces to this cap for a scalar.
MAX_PREWHITEN_RHO: Final = 0.97

#: Andrews (1991, eq. 6.2 and Table I): the Bartlett kernel's bandwidth is
#: ``1.1447 (alpha(1) T)^{1/3}``.
_BARTLETT_CONSTANT: Final = 1.1447
_BARTLETT_RATE: Final = 1.0 / 3.0
#: The AR(1) plug-in's coefficient is kept strictly inside the unit circle so
#: ``alpha(1)`` stays finite; an OLS slope at or beyond it means the series is
#: not stationary enough for the plug-in to mean anything, and the bandwidth
#: is then the largest one the formula can produce rather than infinity.
_PLUGIN_RHO_CAP: Final = 0.999

#: ``"andrews"``: the AR(1) plug-in of Andrews (1991). ``"rule_of_thumb"``:
#: the deterministic, ``n``-only ``floor(4 (n/100)^{2/9})`` truncation lag that
#: follows Newey & West (1994) and is most software's default, as a Bartlett
#: bandwidth ``L + 1``. A number is a Bartlett bandwidth ``S`` in the same
#: convention: lag ``j`` gets weight ``1 - j/S`` for ``0 < j < S``, so an
#: integer Newey-West truncation lag ``L`` with weights ``1 - j/(L+1)`` is
#: ``S = L + 1``.
BandwidthRule = Literal["andrews", "rule_of_thumb"]


@dataclass(frozen=True)
class HACSpec:
    """How the long-run variance of a loss differential is estimated.

    ``bandwidth`` is :data:`BandwidthRule` or a fixed Bartlett bandwidth;
    ``scale`` multiplies whichever results, which is how a sensitivity ladder
    (half, once, twice the automatic value) is expressed without re-deriving
    the rule. ``prewhiten`` applies Andrews & Monahan's (1992) AR(1)
    pre-whitening: the kernel sees the AR(1) residuals of the demeaned
    series and the result is recoloured by ``1/(1-rho)^2``; ``max_rho`` caps
    the coefficient. When pre-whitening, the automatic bandwidth is computed
    on the residuals, as Andrews & Monahan do.
    """

    bandwidth: float | BandwidthRule = "andrews"
    prewhiten: bool = True
    scale: float = 1.0
    max_rho: float = MAX_PREWHITEN_RHO

    def __post_init__(self) -> None:
        if isinstance(self.bandwidth, str):
            if self.bandwidth not in ("andrews", "rule_of_thumb"):
                raise ValueError(f"unknown bandwidth rule {self.bandwidth!r}")
        elif not (math.isfinite(self.bandwidth) and self.bandwidth > 0.0):
            raise ValueError(f"a fixed bandwidth must be finite and positive, got {self.bandwidth}")
        if not (math.isfinite(self.scale) and self.scale > 0.0):
            raise ValueError(f"scale must be finite and positive, got {self.scale}")
        if not 0.0 < self.max_rho < 1.0:
            raise ValueError(f"max_rho must lie strictly inside (0, 1), got {self.max_rho}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "bandwidth": self.bandwidth,
            "prewhiten": bool(self.prewhiten),
            "scale": float(self.scale),
            "max_rho": float(self.max_rho),
        }


@dataclass(frozen=True)
class LongRunVariance:
    """One long-run variance estimate and everything that went into it.

    ``omega`` estimates ``sum_k gamma_k`` of the series itself (recoloured
    when pre-whitened); ``bandwidth`` is the Bartlett bandwidth applied to
    the series the kernel actually saw and ``n_lags`` how many autocovariances
    received positive weight. ``rho`` is the pre-whitening coefficient (0 when
    not pre-whitening) and ``rho_capped`` whether :attr:`HACSpec.max_rho`
    bound it. ``rho1`` is the first-order sample autocorrelation of the
    series, reported so a reader can see how much the correction is doing.
    """

    omega: float
    bandwidth: float
    n_lags: int
    prewhiten: bool
    rho: float
    rho_capped: bool
    rho1: float
    n: int


def ar1_coefficient(values: object) -> float:
    """OLS slope of ``x_t`` on ``x_{t-1}`` after demeaning by the full-sample mean.

    ``0.0`` for a series too short or too flat to regress.
    """
    x = _as_1d(values, "values")
    if x.size < 3:
        return 0.0
    centred = x - x.mean()
    denominator = float(centred[:-1] @ centred[:-1])
    if not denominator > 0.0:
        return 0.0
    return float(centred[1:] @ centred[:-1]) / denominator


def autocorrelation(values: object, lag: int = 1) -> float:
    """Sample autocorrelation ``gamma_hat_lag / gamma_hat_0``; NaN for a constant series."""
    x = _as_1d(values, "values")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if x.size <= lag:
        return math.nan
    centred = x - x.mean()
    gamma0 = float(centred @ centred)
    if not gamma0 > 0.0:
        return math.nan
    return float(centred[lag:] @ centred[:-lag]) / gamma0


def effective_sample_size(n: int, rho1: float) -> float:
    """``n (1 - rho) / (1 + rho)``: the AR(1)-equivalent number of independent observations.

    The variance of the mean of an AR(1) at ``rho`` is ``sigma^2/n``
    times ``(1 + rho)/(1 - rho)`` to first order, so this is the ``n`` an iid
    sample would need to say as much about the mean. NaN when ``rho1`` is.
    """
    if not math.isfinite(rho1) or rho1 >= 1.0 or rho1 <= -1.0:
        return math.nan
    return n * (1.0 - rho1) / (1.0 + rho1)


def andrews_bandwidth(values: object) -> float:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    ``1.1447 (alpha(1) n)^{1/3}`` with ``alpha(1) = 4 rho^2 / ((1-rho)^2 (1+rho)^2)``
    (eq. 6.4 for a scalar AR(1) with unit weight), ``rho`` the OLS AR(1)
    coefficient of the series. ``0.0`` — no lag weighted — when the series is
    uncorrelated at lag one.
    """
    x = _as_1d(values, "values")
    rho = ar1_coefficient(x)
    rho = max(-_PLUGIN_RHO_CAP, min(_PLUGIN_RHO_CAP, rho))
    alpha1 = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    return float(_BARTLETT_CONSTANT * (alpha1 * x.size) ** _BARTLETT_RATE)


def rule_of_thumb_bandwidth(n: int) -> float:
    """``floor(4 (n/100)^{2/9}) + 1``: the fixed-rule truncation lag as a Bartlett bandwidth.

    The same rule as ``volbench.analysis.hac_bandwidth`` (whose ``L`` gives
    weights ``1 - j/(L+1)``), so a long-run variance at this bandwidth without
    pre-whitening is that module's estimator exactly. Stated for what it is:
    a function of ``n`` alone, blind to how persistent the series is.
    """
    if n < 2:
        return 1.0
    return float(int(4.0 * (n / 100.0) ** (2.0 / 9.0) // 1.0) + 1)


def _bartlett_long_run(centred: NDArray[np.float64], bandwidth: float) -> tuple[float, int]:
    """``gamma_0 + 2 sum_{0<j<S} (1 - j/S) gamma_j`` with ``1/n``-normalised autocovariances."""
    n = centred.size
    omega = float(centred @ centred) / n
    n_lags = 0
    j = 1
    while j < bandwidth and j < n:
        # Same operation order as ``analysis.hac_mean_se``, so that a fixed
        # bandwidth of ``L + 1`` reproduces that estimator to the bit.
        gamma_j = float(centred[j:] @ centred[:-j]) / n
        omega += 2.0 * (1.0 - j / bandwidth) * gamma_j
        n_lags += 1
        j += 1
    return omega, n_lags


def long_run_variance(values: object, spec: HACSpec | None = None) -> LongRunVariance:
    """Long-run variance of a series, Bartlett kernel, optionally pre-whitened.

    Without pre-whitening this is the Newey-West estimator on the demeaned
    series at ``spec.bandwidth``. With it (the default), Andrews & Monahan
    (1992): fit ``x_t - x̄ = rho (x_{t-1} - x̄) + e_t`` by OLS, cap ``rho`` at
    ``spec.max_rho``, apply the kernel to the demeaned residuals at a bandwidth
    chosen *on the residuals*, and recolour: ``omega = omega_e / (1 - rho)^2``.
    A fixed rule-of-thumb bandwidth on a persistent series understates the
    long-run variance by whatever the kernel cannot see past its window; the
    AR(1) fit carries that dependence instead, and the residuals are close
    enough to white for a short window to be right.

    Non-finite values are not accepted: the caller decides what a hole means.
    """
    spec = HACSpec() if spec is None else spec
    x = _as_1d(values, "values")
    if x.size < 2:
        raise ValueError(f"need at least two observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("long_run_variance needs a finite series; drop or mask holes first")
    centred = x - x.mean()
    rho1 = autocorrelation(x, 1)

    rho = 0.0
    capped = False
    series = centred
    if spec.prewhiten:
        if x.size < 3:
            raise ValueError("pre-whitening needs at least three observations")
        rho = ar1_coefficient(x)
        if abs(rho) > spec.max_rho:
            rho = math.copysign(spec.max_rho, rho)
            capped = True
        residuals = centred[1:] - rho * centred[:-1]
        series = residuals - residuals.mean()

    if spec.bandwidth == "andrews":
        base = andrews_bandwidth(series)
    elif spec.bandwidth == "rule_of_thumb":
        base = rule_of_thumb_bandwidth(series.size)
    else:
        base = float(spec.bandwidth)
    bandwidth = base * spec.scale
    omega, n_lags = _bartlett_long_run(series, bandwidth)
    if spec.prewhiten:
        omega /= (1.0 - rho) ** 2
    return LongRunVariance(
        omega=float(omega),
        bandwidth=float(bandwidth),
        n_lags=int(n_lags),
        prewhiten=bool(spec.prewhiten),
        rho=float(rho),
        rho_capped=bool(capped),
        rho1=float(rho1),
        n=int(x.size),
    )


# --------------------------------------------------------------------------
# Diebold-Mariano with the Harvey-Leybourne-Newbold correction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    """One Diebold-Mariano test.

    ``statistic`` is ``S_1*`` (HLN-corrected) when ``hln`` is true, else the
    original ``S_1``; a positive value means ``loss_a`` was the *larger*
    loss. ``variance`` is the estimated variance of the mean differential
    ``V̂(d̄)``; ``variance_nonpositive`` flags the case where the rectangular
    window produced ``V̂ <= 0`` and the Diebold-Mariano rule ("treat it as 0
    and automatically reject") was applied. ``n`` is the number of complete
    pairs used, ``n_dropped`` how many origins were excluded for a missing
    loss on either side.

    ``hln_factor`` is the correction factor actually applied (1 when ``hln``
    is off). Under a kernel estimator (``hac`` given) ``bandwidth`` is the
    Bartlett bandwidth used, ``lag`` the number of autocovariances it gave
    positive weight, ``prewhiten``/``rho``/``rho_capped`` the Andrews-Monahan
    step; under the windowed estimator ``bandwidth`` is NaN. ``rho1`` is the
    first-order autocorrelation of the differential and ``n_eff`` its
    AR(1)-equivalent sample size, both reported whichever estimator ran.
    """

    statistic: float
    p_value: float
    mean_diff: float
    variance: float
    n: int
    n_dropped: int
    horizon: int
    lag: int
    kernel: Kernel
    hln: bool
    alternative: Alternative
    variance_nonpositive: bool
    config_hash: str
    hln_factor: float
    bandwidth: float
    prewhiten: bool
    rho: float
    rho_capped: bool
    rho1: float
    n_eff: float


def _autocovariances(d: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """``gamma_hat_k = n^{-1} Σ_{t=k+1}^{n} (d_t - d̄)(d_{t-k} - d̄)`` for ``k = 0..max_lag``.

    Harvey, Leybourne & Newbold (1997, eq. 2); Diebold & Mariano (1995) use
    the same ``1/n`` normalisation.
    """
    n = d.size
    centred = d - d.mean()
    return np.array(
        [float(np.dot(centred[k:], centred[: n - k])) / n for k in range(max_lag + 1)],
        dtype=np.float64,
    )


def _dm_p_value(statistic: float, alternative: Alternative, df: int | None) -> float:
    dist: Any = stats.t(df=df) if df is not None else stats.norm()
    if alternative == "two-sided":
        return float(2.0 * dist.sf(abs(statistic)))
    if alternative == "less":
        return float(dist.cdf(statistic))
    return float(dist.sf(statistic))


def diebold_mariano(
    loss_a: object,
    loss_b: object,
    *,
    horizon: int = 1,
    lag: int | None = None,
    kernel: Kernel = "rectangular",
    hln: bool = True,
    alternative: Alternative = "two-sided",
    hac: HACSpec | None = None,
) -> DMResult:
    """Diebold-Mariano test of equal expected loss, HLN-corrected by default.

    With loss differential ``d_t = loss_a[t] - loss_b[t]`` over the ``n``
    origins where both losses are finite, the statistic of Diebold & Mariano
    (1995, §1.1) is

        ``S_1 = d̄ / sqrt(V̂(d̄))``,
        ``V̂(d̄) = n^{-1} [ gamma_hat_0 + 2 Σ_{k=1}^{h-1} w_k gamma_hat_k ]``,

    where ``gamma_hat_k`` is the sample autocovariance at lag ``k`` (``1/n``
    normalised, mean-corrected) and the sum runs over the ``h - 1``
    autocovariances a ``h``-step-ahead forecast error can carry: under
    optimal forecasts the differential is ``MA(h-1)``, so the rectangular
    (uniform) lag window truncated at ``h - 1`` is consistent (``w_k = 1``).
    ``S_1`` is asymptotically ``N(0, 1)`` under the null ``E[d_t] = 0``.

    Harvey, Leybourne & Newbold (1997, eq. 9) correct the finite-sample bias
    of ``V̂(d̄)`` and compare against Student's ``t`` with ``n - 1`` degrees
    of freedom:

        ``S_1* = sqrt( [n + 1 - 2h + n^{-1} h(h - 1)] / n ) · S_1``.

    The factor is exact when ``d_t`` is white noise: for ``h = 1`` the
    corrected statistic *is* the one-sample ``t`` statistic of ``d_t``, so
    the test has exact size under iid Gaussian differentials. That identity
    is pinned in ``tests/test_inference.py``.

    Parameters
    ----------
    loss_a, loss_b:
        Same-length loss arrays (smaller is better) in time order. Origins
        where either is non-finite are excluded pairwise-complete and
        counted in ``n_dropped``.
    horizon:
        Forecast horizon ``h``. Sets the default truncation lag ``h - 1``
        and enters the HLN factor.
    lag:
        Number of autocovariances to include; defaults to ``horizon - 1``.
        If overridden, the HLN factor uses ``h = lag + 1`` so the estimator
        and its bias correction stay consistent.
    kernel:
        Lag window, see :data:`Kernel`. Diebold & Mariano (1995) note the
        rectangular estimate "is not guaranteed to be positive semidefinite"
        and, "in the rare event that a negative estimate arises, we treat it
        as 0 and automatically reject the null". That rule is applied here
        and reported through ``variance_nonpositive``; the Bartlett window is
        the alternative they offer.
    hln:
        Apply the correction and the ``t_{n-1}`` reference distribution.
        ``False`` gives the original asymptotic test.
    alternative:
        See :data:`Alternative`.
    hac:
        Replace the ``h - 1`` window with a kernel estimator of the long-run
        variance (:func:`long_run_variance`): Bartlett kernel, a data-driven
        or fixed bandwidth, and Andrews-Monahan pre-whitening by default. The
        windowed estimator at ``h = 1`` uses *no* autocovariance at all, which
        is only right when the differential is white; a persistent
        differential — volatility losses cluster — then gets a denominator
        that is too small and a p-value that is too small with it. ``lag``
        and ``hac`` are alternatives, not combinable. The HLN factor is then
        computed with the forecast ``horizon``: the bandwidth is not a count
        of autocovariances the differential is assumed to carry.

    Notes
    -----
    Pairwise p-values are **not** multiplicity-corrected. Over a model set,
    use :func:`model_confidence_set` to decide who wins and report the DM
    matrix alongside it (:func:`compare_models`).
    """
    a = _as_1d(loss_a, "loss_a")
    b = _as_1d(loss_b, "loss_b")
    if a.size != b.size:
        raise ValueError(f"loss_a has length {a.size} but loss_b has length {b.size}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if hac is not None and lag is not None:
        raise ValueError("pass either lag (the truncated window) or hac (the kernel), not both")
    if lag is None:
        lag = horizon - 1
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if kernel not in ("rectangular", "bartlett"):
        raise ValueError(f"kernel must be 'rectangular' or 'bartlett', got {kernel!r}")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")

    valid = np.isfinite(a) & np.isfinite(b)
    n_dropped = int(a.size - int(valid.sum()))
    d = a[valid] - b[valid]
    n = int(d.size)
    # What the HLN factor sees: under the windowed estimator the truncation
    # lag plus one, so the estimator and its bias correction stay consistent;
    # under a kernel estimator the forecast horizon itself.
    h = horizon if hac is not None else lag + 1
    if n <= h or n < 2:
        raise ValueError(
            f"need more than lag + 1 = {h} complete loss pairs (and at least 2) to estimate "
            f"{lag} autocovariances; got {n} after dropping {n_dropped}"
        )

    lrv: LongRunVariance | None = None
    if hac is not None:
        if n < 3:
            raise ValueError(f"the kernel estimator needs at least three complete pairs, got {n}")
        lrv = long_run_variance(d, hac)
        variance = lrv.omega / n
        kernel = "bartlett"
        lag = lrv.n_lags
    else:
        gamma = _autocovariances(d, lag)
        if kernel == "rectangular":
            weights = np.ones(lag, dtype=np.float64)
        else:
            weights = 1.0 - np.arange(1, lag + 1, dtype=np.float64) / h
        variance = float(gamma[0] + 2.0 * float(np.dot(weights, gamma[1:]))) / n
    mean_diff = float(d.mean())
    factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if hln else 1.0

    variance_nonpositive = False
    if variance > 0.0:
        statistic = factor * mean_diff / math.sqrt(variance)
    elif mean_diff == 0.0:
        # Every differential is exactly zero: identical losses, no evidence
        # either way. (A rectangular window can only go negative when the
        # differentials vary, so here variance == 0 as well.)
        statistic = 0.0
        variance_nonpositive = variance < 0.0
    else:
        # Diebold & Mariano (1995, §1.1): a non-positive spectral estimate is
        # treated as zero and the null is rejected outright.
        variance_nonpositive = True
        statistic = math.copysign(math.inf, mean_diff)
        warnings.warn(
            f"Diebold-Mariano: the {kernel} long-run variance estimate is {variance:.3g} <= 0 "
            f"(n={n}, lag={lag}); applying Diebold & Mariano's rule of treating it as 0 and "
            "rejecting. Consider kernel='bartlett'.",
            RuntimeWarning,
            stacklevel=2,
        )
    p_value = _dm_p_value(statistic, alternative, n - 1 if hln else None)
    rho1 = lrv.rho1 if lrv is not None else autocorrelation(d, 1)

    digest = config_hash(
        {
            "method": "diebold_mariano",
            "loss_a_sha256": array_digest(a),
            "loss_b_sha256": array_digest(b),
            "horizon": int(horizon),
            "lag": int(lag),
            "kernel": kernel,
            "hln": bool(hln),
            "alternative": alternative,
            "hac": None if hac is None else hac.as_dict(),
            "package_version": package_version(),
        }
    )
    return DMResult(
        statistic=float(statistic),
        p_value=p_value,
        mean_diff=mean_diff,
        variance=variance,
        n=n,
        n_dropped=n_dropped,
        horizon=int(horizon),
        lag=int(lag),
        kernel=kernel,
        hln=bool(hln),
        alternative=alternative,
        variance_nonpositive=variance_nonpositive,
        config_hash=digest,
        hln_factor=float(factor),
        bandwidth=lrv.bandwidth if lrv is not None else math.nan,
        prewhiten=lrv.prewhiten if lrv is not None else False,
        rho=lrv.rho if lrv is not None else 0.0,
        rho_capped=lrv.rho_capped if lrv is not None else False,
        rho1=float(rho1),
        n_eff=effective_sample_size(n, float(rho1)),
    )


@dataclass(frozen=True, eq=False)
class DMMatrix:
    """Every pairwise Diebold-Mariano test over a model set.

    Entry ``(i, j)`` of each frame is ``diebold_mariano(loss_i, loss_j)``:
    the differential is ``L_i - L_j``, so a positive ``statistic`` means
    model ``i`` lost *more*. ``statistic`` is antisymmetric, ``p_value``
    (two-sided) symmetric, diagonals NaN. ``n`` and ``n_dropped`` are the
    pairwise-complete counts behind each entry — no exclusion is silent.

    These p-values are **not** multiplicity-corrected. The MCS is the
    primary "who wins" tool; see :func:`compare_models`.

    ``bandwidth``, ``rho1`` and ``n_eff`` carry each pair's Bartlett bandwidth
    (NaN under the windowed estimator), the first-order autocorrelation of its
    differential, and the AR(1)-equivalent sample size; ``hac`` is the
    estimator specification the matrix was built with, or ``None``.
    """

    models: tuple[str, ...]
    statistic: pd.DataFrame
    p_value: pd.DataFrame
    n: pd.DataFrame
    n_dropped: pd.DataFrame
    pairs: Mapping[tuple[str, str], DMResult]
    horizon: int
    lag: int
    kernel: Kernel
    hln: bool
    bandwidth: pd.DataFrame
    rho1: pd.DataFrame
    n_eff: pd.DataFrame
    hac: HACSpec | None


def dm_matrix(
    losses: LossMatrix | pd.DataFrame | NDArray[np.float64],
    *,
    horizon: int = 1,
    lag: int | None = None,
    kernel: Kernel = "rectangular",
    hln: bool = True,
    model_names: Sequence[str] | None = None,
    hac: HACSpec | None = None,
) -> DMMatrix:
    """Pairwise :func:`diebold_mariano` (two-sided) over every ordered pair.

    Each pair is evaluated on its own complete origins (pairwise-complete),
    so ``n`` can differ across entries; the ``n_dropped`` frame says by how
    much. See :class:`DMMatrix` for orientation and the multiplicity caveat.
    ``hac`` is passed through to every pair; the bandwidth it chooses is then
    a property of each pair's differential and is reported per entry.
    """
    matrix, names = _resolve_losses(losses, model_names)
    m = len(names)
    statistic = np.full((m, m), math.nan)
    p_value = np.full((m, m), math.nan)
    n_used = np.zeros((m, m), dtype=np.int64)
    n_dropped = np.zeros((m, m), dtype=np.int64)
    bandwidth = np.full((m, m), math.nan)
    rho1 = np.full((m, m), math.nan)
    n_eff = np.full((m, m), math.nan)
    pairs: dict[tuple[str, str], DMResult] = {}
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            result = diebold_mariano(
                matrix[:, i],
                matrix[:, j],
                horizon=horizon,
                lag=lag,
                kernel=kernel,
                hln=hln,
                alternative="two-sided",
                hac=hac,
            )
            pairs[(names[i], names[j])] = result
            statistic[i, j] = result.statistic
            p_value[i, j] = result.p_value
            n_used[i, j] = result.n
            n_dropped[i, j] = result.n_dropped
            bandwidth[i, j] = result.bandwidth
            rho1[i, j] = result.rho1
            n_eff[i, j] = result.n_eff
    index = pd.Index(names, name="model")
    resolved_lag = horizon - 1 if lag is None else lag
    return DMMatrix(
        models=names,
        statistic=pd.DataFrame(statistic, index=index, columns=index),
        p_value=pd.DataFrame(p_value, index=index, columns=index),
        n=pd.DataFrame(n_used, index=index, columns=index),
        n_dropped=pd.DataFrame(n_dropped, index=index, columns=index),
        pairs=pairs,
        horizon=int(horizon),
        lag=int(resolved_lag),
        kernel="bartlett" if hac is not None else kernel,
        hln=bool(hln),
        bandwidth=pd.DataFrame(bandwidth, index=index, columns=index),
        rho1=pd.DataFrame(rho1, index=index, columns=index),
        n_eff=pd.DataFrame(n_eff, index=index, columns=index),
        hac=hac,
    )


# --------------------------------------------------------------------------
# moving block bootstrap
# --------------------------------------------------------------------------


def _validate_bootstrap(n: int, block_length: int, n_boot: int) -> None:
    if n < 1:
        raise ValueError("need at least one observation to resample")
    if block_length < 1 or block_length > n:
        raise ValueError(f"block_length must lie in [1, n={n}], got {block_length}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")


def _block_starts(
    rng: np.random.Generator, n: int, block_length: int, n_blocks: int, count: int
) -> NDArray[np.int64]:
    """``count x n_blocks`` block start positions, uniform on ``0..n-l``."""
    starts: NDArray[np.int64] = rng.integers(
        0, n - block_length + 1, size=(count, n_blocks), dtype=np.int64
    )
    return starts


def moving_block_indices(n: int, block_length: int, n_boot: int, seed: int) -> NDArray[np.int64]:
    """Moving-block-bootstrap index resamples (Künsch 1989), ``n_boot x n``.

    Each resample concatenates ``ceil(n / block_length)`` blocks, each a
    **contiguous, time-ordered window** ``s, s+1, ..., s+l-1`` of the original
    axis with ``s`` drawn uniformly from ``0..n-l`` — never wrapping around
    the end of the sample — truncated to length ``n``. Within a block, time
    order is exactly the sample's; that is what preserves the serial
    dependence of loss differentials under the null (h-step losses are
    ``MA(h-1)``; volatility losses cluster).

    Hansen, Lunde & Nason's own implementation (2003, Appendix B; the 2011
    paper's separate appendix) draws starts on the *circular* axis
    (``τ + 1 mod n``), so a block may join the last observation to the first.
    The non-circular moving block is used here so that every block is a
    genuine window of the sample; the two are asymptotically equivalent.

    ``block_length = 1`` is the iid bootstrap; ``block_length = n`` returns
    the original ordering in every resample.

    Drawn in fixed chunks of :data:`_BOOT_CHUNK` resamples so that the
    indices reproduce exactly what :func:`model_confidence_set` consumed for
    the same ``(n, block_length, n_boot, seed)``.
    """
    _validate_bootstrap(n, block_length, n_boot)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    out = np.empty((n_boot, n), dtype=np.int64)
    for lo in range(0, n_boot, _BOOT_CHUNK):
        hi = min(lo + _BOOT_CHUNK, n_boot)
        starts = _block_starts(rng, n, block_length, n_blocks, hi - lo)
        full = (starts[:, :, None] + offsets[None, None, :]).reshape(hi - lo, -1)
        out[lo:hi] = full[:, :n]
    return out


def _bootstrap_column_means(
    values: NDArray[np.float64],
    block_length: int,
    n_boot: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Column means of ``values`` under each moving-block resample, ``n_boot x m``.

    Same draws as :func:`moving_block_indices`, but computed from
    precomputed block sums (cumulative sums over the time axis) so no
    ``n_boot x n`` index matrix is ever materialised.
    """
    n, m = values.shape
    n_blocks = -(-n // block_length)
    remainder = n - (n_blocks - 1) * block_length  # length of the last block, 1..l
    cumulative = np.vstack([np.zeros((1, m)), np.cumsum(values, axis=0)])
    n_starts = n - block_length + 1
    full_sums = cumulative[block_length : block_length + n_starts] - cumulative[:n_starts]
    last_sums = cumulative[remainder : remainder + n_starts] - cumulative[:n_starts]
    slab = max(1, _GATHER_ELEMENTS // max(1, _BOOT_CHUNK * m))

    out = np.empty((n_boot, m), dtype=np.float64)
    for lo in range(0, n_boot, _BOOT_CHUNK):
        hi = min(lo + _BOOT_CHUNK, n_boot)
        starts = _block_starts(rng, n, block_length, n_blocks, hi - lo)
        sums = last_sums[starts[:, -1]].copy()
        for a in range(0, n_blocks - 1, slab):
            sums += full_sums[starts[:, a : min(a + slab, n_blocks - 1)]].sum(axis=1)
        out[lo:hi] = sums / n
    return out


def bootstrap_column_means(
    values: object, block_length: int, n_boot: int, seed: int
) -> NDArray[np.float64]:
    """Column means of an ``n x m`` array under each moving-block resample, ``n_boot x m``.

    The same draws as :func:`moving_block_indices` for the same
    ``(n, block_length, n_boot, seed)`` — pinned in the tests — computed from
    block sums so the ``n_boot x n`` index matrix is never materialised. This
    is what :func:`model_confidence_set` consumes, exposed so that another
    statistic (a Sharpe ratio, say) can be bootstrapped under exactly the
    same scheme and seed and reported alongside it.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"values must be 2-D (observations x columns), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite; drop incomplete rows first")
    _validate_bootstrap(matrix.shape[0], block_length, n_boot)
    return _bootstrap_column_means(matrix, block_length, n_boot, np.random.default_rng(seed))


# --------------------------------------------------------------------------
# block length
# --------------------------------------------------------------------------


def _ppw_block_length(x: NDArray[np.float64]) -> float:
    """Optimal block length for one series (Politis & White 2004; Patton, Politis & White 2009).

    The circular/moving-block value ``b_CB = (2 G² / D_CB)^{1/3} n^{1/3}``
    with ``G = Σ_k h(k/M) |k| gamma_hat_k``, ``D_CB = (4/3) sigma_hat^4``,
    ``sigma_hat^2 = Σ_k h(k/M) gamma_hat_k`` (flat-top window ``h(x) = min(1, 2(1 - |x|))``),
    and ``M = 2 m̂`` where ``m̂`` is the first lag after which ``k_n = max(5,
    log10 n)`` consecutive autocorrelations all lie inside the band ``2
    sqrt(log10 n / n)``; ``M`` and ``b`` are capped at ``ceil(sqrt n) + k_n``
    and ``ceil(min(3 sqrt n, n / 3))``. Tuning constants follow Patton's
    MATLAB code, as in the reference implementation
    ``arch.bootstrap.optimal_block_length`` (K. Sheppard), which this mirrors
    step for step and is pinned against in ``tests/test_inference.py``.
    Returns the unrounded value; ``1.0`` for a series with no usable
    autocovariance (e.g. constant).
    """
    n = x.size
    eps = x - x.mean()
    b_max = float(math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))
    k_n = max(5, int(math.log10(n)))
    m_max = math.ceil(math.sqrt(n)) + k_n
    band = 2.0 * math.sqrt(math.log10(n) / n)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
    opt_m: int | None = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(m_max + 1):
            v1 = float(eps[i + 1 :] @ eps[i + 1 :])
            v2 = float(eps[: -(i + 1)] @ eps[: -(i + 1)])
            cross = float(eps[i:] @ eps[: n - i])
            acv[i] = cross / n
            abs_acorr[i] = abs(cross) / math.sqrt(v1 * v2) if v1 * v2 > 0.0 else math.nan
            if i >= k_n and opt_m is None and bool(np.all(abs_acorr[i - k_n : i] < band)):
                opt_m = i - k_n
    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)
    g = 0.0
    long_run = float(acv[0])
    for k in range(1, m + 1):
        weight = 1.0 if k / m <= 0.5 else 2.0 * (1.0 - k / m)
        g += 2.0 * weight * k * float(acv[k])
        long_run += 2.0 * weight * float(acv[k])
    d_cb = 4.0 / 3.0 * long_run * long_run
    if not (d_cb > 0.0) or not math.isfinite(g):
        return 1.0
    b_cb = (2.0 * g * g / d_cb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return min(b_cb, b_max) if math.isfinite(b_cb) else 1.0


def politis_white_block_length(values: object) -> float:
    """The Politis-White automatic block length of one series, unrounded.

    :func:`default_block_length` takes the largest of these over the pairwise
    loss differentials; this is the per-series value it is built from, so a
    chosen block length can be read against each series it was chosen for.
    """
    x = _as_1d(values, "values")
    if x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("need at least three finite observations")
    return _ppw_block_length(x)


def ar1_block_length(rho: float, n: int) -> float:
    """The block length the Politis-White rule would return for an AR(1) at ``rho``.

    With ``gamma_k = sigma^2 rho^|k|`` the rule's ``G = 2 sigma^2 rho /
    (1-rho)^2`` and ``D_CB = (4/3) sigma^4 ((1+rho)/(1-rho))^2``, so
    ``b = (6 rho^2 / ((1-rho)^2 (1+rho)^2))^{1/3} n^{1/3}``, capped where the
    rule caps (``ceil(min(3 sqrt n, n/3))``). A yardstick for a chosen block
    length: a series whose first-order autocorrelation is ``rho`` and whose
    chosen block is far below this has a rule that stopped short, not a
    series that is short-memory.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must lie strictly inside (-1, 1), got {rho}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if rho == 0.0:
        return 1.0
    b_max = float(math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))
    b = float((6.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)) ** (1.0 / 3.0))
    return max(1.0, min(b * float(n ** (1.0 / 3.0)), b_max))


def default_block_length(
    losses: LossMatrix | pd.DataFrame | NDArray[np.float64],
    *,
    horizon: int = 1,
    model_names: Sequence[str] | None = None,
) -> int:
    """Data-driven block length for :func:`model_confidence_set`.

    The automatic block length of Politis & White (2004), with the Patton,
    Politis & White (2009) correction, is computed for every pairwise loss
    differential ``d_ij,t`` (complete cases, ``i < j``) — the series whose
    dependence the MCS bootstrap must reproduce — and the largest value is
    taken, rounded up, floored at ``max(1, horizon)`` so the ``MA(h-1)``
    dependence of ``h``-step losses always fits inside a block, and capped
    at the sample length. The value used is recorded on every
    :class:`MCSResult`.

    Hansen, Lunde & Nason (2003, Appendix B) instead suggest the largest
    AR lag length selected by an information criterion over the ``d_ij,t``.
    That rule returns ``1`` — the iid bootstrap — for any AR(1) differential
    however persistent, which volatility losses often are; the Politis-White
    rule scales with the dependence *range* (and with ``n^{1/3}``) rather
    than the AR order, which is why it is the default here. Both are
    heuristics: pass ``block_length`` explicitly to override, and report it.
    """
    matrix, _ = _resolve_losses(losses, model_names)
    complete = matrix[np.all(np.isfinite(matrix), axis=1)]
    n, m = complete.shape
    floor = max(1, int(horizon))
    if n < 3 or m < 2:
        return floor
    longest = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            longest = max(longest, _ppw_block_length(complete[:, i] - complete[:, j]))
    return int(min(max(floor, math.ceil(longest)), n))


# --------------------------------------------------------------------------
# model confidence set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MCSResult:
    """A Model Confidence Set and everything needed to reproduce it.

    ``included`` is ``M̂*_{1-alpha}`` — the models whose MCS p-value is at
    least ``alpha`` — in the original model order; ``excluded`` the rest in
    the order they were eliminated. ``p_values`` are MCS p-values (Hansen,
    Lunde & Nason 2011, Definition 4: the cumulative maximum of the
    step-wise bootstrap p-values along the elimination sequence; ``1.0`` for
    the last survivor), so ``included`` for *any* level is ``{i : p_i >=
    level}``. ``step_p_values[k]`` is the raw bootstrap p-value of the
    equivalence test at step ``k`` (before the cumulative maximum).
    ``mean_loss`` is each model's average loss over the ``n`` complete
    origins. ``n_dropped`` counts the origins removed listwise because some
    model had no usable loss there.
    """

    models: tuple[str, ...]
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    p_values: Mapping[str, float]
    elimination_order: tuple[str, ...]
    step_p_values: tuple[float, ...]
    mean_loss: Mapping[str, float]
    alpha: float
    statistic: MCSStatistic
    horizon: int
    n: int
    n_dropped: int
    n_boot: int
    block_length: int
    seed: int
    config_hash: str


def _safe_ratio(numerator: NDArray[np.float64], sd: NDArray[np.float64]) -> NDArray[np.float64]:
    """``numerator / sd`` with ``0`` where ``sd == 0``.

    A zero bootstrap standard deviation means the compared losses were
    identical on every resample, i.e. in every complete origin: there is no
    evidence to standardise, so the t-statistic is defined as 0.
    """
    out = np.zeros(np.broadcast(numerator, sd).shape, dtype=np.float64)
    positive = sd > 0.0
    np.divide(numerator, sd, out=out, where=positive)
    return out


def _range_bootstrap(centred: NDArray[np.float64], sd: NDArray[np.float64]) -> NDArray[np.float64]:
    """``T*_{b,R} = max_{i,j} |c_{b,i} - c_{b,j}| / sd_ij`` for every resample ``b``."""
    n_boot, m = centred.shape
    out = np.empty(n_boot, dtype=np.float64)
    positive = sd > 0.0
    for lo in range(0, n_boot, _BOOT_CHUNK):
        hi = min(lo + _BOOT_CHUNK, n_boot)
        chunk = centred[lo:hi]
        diff = np.abs(chunk[:, :, None] - chunk[:, None, :])
        ratio = np.zeros_like(diff)
        np.divide(diff, sd[None, :, :], out=ratio, where=positive[None, :, :])
        out[lo:hi] = ratio.reshape(hi - lo, m * m).max(axis=1)
    return out


def _semi_quadratic_bootstrap(
    centred: NDArray[np.float64], sd: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``T*_{b,SQ} = sum_{i<j} ((c_{b,i} - c_{b,j}) / sd_ij)^2`` for every resample ``b``.

    The sum over ordered pairs counts every unordered pair twice and the
    diagonal not at all, hence the half.
    """
    n_boot, m = centred.shape
    out = np.empty(n_boot, dtype=np.float64)
    positive = sd > 0.0
    for lo in range(0, n_boot, _BOOT_CHUNK):
        hi = min(lo + _BOOT_CHUNK, n_boot)
        chunk = centred[lo:hi]
        diff = chunk[:, :, None] - chunk[:, None, :]
        ratio = np.zeros_like(diff)
        np.divide(diff, sd[None, :, :], out=ratio, where=positive[None, :, :])
        out[lo:hi] = 0.5 * (ratio * ratio).reshape(hi - lo, m * m).sum(axis=1)
    return out


def model_confidence_set(
    losses: LossMatrix | pd.DataFrame | NDArray[np.float64],
    *,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    block_length: int | None = None,
    statistic: MCSStatistic = "range",
    horizon: int = 1,
    model_names: Sequence[str] | None = None,
) -> MCSResult:
    """Model Confidence Set (Hansen, Lunde & Nason 2011) by sequential elimination.

    Let ``d_ij,t = L_i,t - L_j,t`` be the loss differentials of the models in
    the current set ``M``, ``d̄_ij`` their sample means, and ``d̄_i. = m^{-1}
    Σ_j d̄_ij = L̄_i - L̄.`` model ``i``'s loss relative to the set average
    (HLN 2011, §3.1.2). The equivalence test ``H_0,M : E[d_ij,t] = 0 ∀ i, j``
    uses one of

    - ``T_R,M = max_{i,j} |t_ij|``, ``t_ij = d̄_ij / sqrt(var̂(d̄_ij))``,
      elimination rule ``e_R,M = arg max_i sup_j t_ij``  (``statistic="range"``);
    - ``T_max,M = max_i t_i.``, ``t_i. = d̄_i. / sqrt(var̂(d̄_i.))``,
      elimination rule ``e_max,M = arg max_i t_i.``  (``statistic="max"``).

    The variances and the null distribution come from a moving block
    bootstrap of the time axis (HLN 2003, Appendix B; HLN 2011's separate
    appendix), with one index sequence per resample shared by every model so
    cross-sectional dependence is preserved:
    ``var̂(d̄_ij) = B^{-1} Σ_b (d̄*_{b,ij} - d̄_ij)²``, and the bootstrap
    statistic is the same maximum over the *re-centred* resample means, e.g.
    ``T*_{b,R} = max_{i,j} |d̄*_{b,ij} - d̄_ij| / sqrt(var̂(d̄_ij))``. The
    step p-value is ``P_M = B^{-1} Σ_b 1{T*_b ≥ T_M}``. The worst model is
    removed and the test repeated until one model remains; MCS p-values are
    the cumulative maxima of the step p-values along that sequence
    (Definition 4), the last survivor getting 1, and ``M̂*_{1-alpha} = {i :
    p̂_i ≥ alpha}``.

    Two implementation choices, both documented because they change what an
    exact tie returns:

    - The p-value uses ``≥`` where HLN write ``>``. For continuous losses the
      two coincide almost surely; the difference is exactly the tied case
      ``T_M = T*_b = 0`` (identical losses), which now yields ``P_M = 1``
      rather than ``0``. No evidence of a difference must not eliminate.
    - A t-statistic whose bootstrap standard deviation is zero is 0 (the two
      losses were identical on every complete origin), see ``_safe_ratio``.

    Parameters
    ----------
    losses:
        ``n x m`` losses in ascending time order — a :class:`LossMatrix`
        from :func:`loss_matrix`, a DataFrame (columns = models), or an
        array with ``model_names``. Rows with any non-finite entry are
        dropped *listwise* (one bootstrap index sequence must apply to every
        model); ``n_dropped`` records how many.
    seed:
        Mandatory. Seeds the block starts; identical inputs and seed give an
        identical result (CLAUDE.md rule 3).
    alpha:
        The MCS is the ``1 - alpha`` confidence set; default 0.10
        (docs/metrics_reference.md).
    n_boot:
        Bootstrap resamples ``B``; default 10 000.
    block_length:
        Moving block length ``l``; ``None`` uses :func:`default_block_length`
        (the Politis-White automatic block length, largest over the pairwise
        differentials, floored at ``max(1, horizon)``). Recorded on the
        result either way, so a reported MCS always states its block length.
    statistic:
        ``"range"`` (default), ``"max"`` or ``"semi_quadratic"``, see above
        and :data:`MCSStatistic`. The range statistic is the one HLN's
        corrigendum recommends (their published results had inadvertently
        used a minimum t-statistic); it is also the default of the reference
        ``arch`` implementation. The semi-quadratic statistic ``T_SQ =
        sum_{i<j} t_ij^2`` (HLN 2003; Hansen & Lunde 2005) pools every pair
        rather than taking the largest, and shares the range statistic's
        elimination rule.
    horizon:
        Forecast horizon of the losses; only used to floor the default block
        length.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly inside (0, 1), got {alpha}")
    if statistic not in ("range", "max", "semi_quadratic"):
        raise ValueError(f"statistic must be 'range', 'max' or 'semi_quadratic', got {statistic!r}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    matrix, names = _resolve_losses(losses, model_names)
    m = len(names)
    if m < 2:
        raise ValueError("a model confidence set needs at least two models")
    complete_rows = np.all(np.isfinite(matrix), axis=1)
    complete = matrix[complete_rows]
    n = int(complete.shape[0])
    n_dropped = int(matrix.shape[0] - n)
    if n < 2:
        raise ValueError(
            f"need at least two origins complete for every model, got {n} "
            f"(dropped {n_dropped} of {matrix.shape[0]})"
        )
    if block_length is None:
        block_length = default_block_length(complete, horizon=horizon)
    _validate_bootstrap(n, block_length, n_boot)

    rng = np.random.default_rng(seed)
    mean_loss = complete.mean(axis=0)
    centred = _bootstrap_column_means(complete, block_length, n_boot, rng) - mean_loss[None, :]
    # var̂(d̄_ij) = mean_b (c_bi - c_bj)² = G_ii + G_jj - 2 G_ij with G = CᵀC / B,
    # without materialising a B x m x m tensor.
    gram = centred.T @ centred / n_boot
    diagonal = np.diag(gram)
    pair_variance = np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)
    pair_sd = np.sqrt(pair_variance)

    remaining = list(range(m))
    elimination: list[int] = []
    step_p: list[float] = []
    while len(remaining) > 1:
        idx = np.array(remaining, dtype=np.int64)
        if statistic in ("range", "semi_quadratic"):
            d_bar = mean_loss[idx][:, None] - mean_loss[idx][None, :]
            sd = pair_sd[np.ix_(idx, idx)]
            t_stat = _safe_ratio(d_bar, sd)
            if statistic == "range":
                observed = float(np.abs(t_stat).max())
                boot = _range_bootstrap(centred[:, idx], sd)
            else:
                observed = 0.5 * float((t_stat * t_stat).sum())
                boot = _semi_quadratic_bootstrap(centred[:, idx], sd)
            worst = int(idx[int(np.argmax(t_stat.max(axis=1)))])
        else:
            sub = centred[:, idx]
            sub_dot = sub - sub.mean(axis=1, keepdims=True)
            d_dot = mean_loss[idx] - mean_loss[idx].mean()
            sd_dot = np.sqrt(np.mean(sub_dot * sub_dot, axis=0))
            t_dot = _safe_ratio(d_dot, sd_dot)
            observed = float(t_dot.max())
            boot = _safe_ratio(sub_dot, sd_dot[None, :]).max(axis=1)
            worst = int(idx[int(np.argmax(t_dot))])
        step_p.append(float(np.mean(boot >= observed)))
        elimination.append(worst)
        remaining.remove(worst)
    survivor = remaining[0]

    p_values: dict[str, float] = {}
    running = 0.0
    for model_index, p in zip(elimination, step_p, strict=True):
        running = max(running, p)
        p_values[names[model_index]] = running
    p_values[names[survivor]] = 1.0
    order = tuple(names[i] for i in [*elimination, survivor])
    included = tuple(name for name in names if p_values[name] >= alpha)
    excluded = tuple(name for name in order if p_values[name] < alpha)

    digest = config_hash(
        {
            "method": "model_confidence_set",
            "models": list(names),
            "losses_sha256": array_digest(matrix.reshape(-1)),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "alpha": float(alpha),
            "n_boot": int(n_boot),
            "block_length": int(block_length),
            "statistic": statistic,
            "horizon": int(horizon),
            "seed": int(seed),
            "package_version": package_version(),
        }
    )
    return MCSResult(
        models=names,
        included=included,
        excluded=excluded,
        p_values=p_values,
        elimination_order=order,
        step_p_values=tuple(step_p),
        mean_loss={name: float(mean_loss[i]) for i, name in enumerate(names)},
        alpha=float(alpha),
        statistic=statistic,
        horizon=int(horizon),
        n=n,
        n_dropped=n_dropped,
        n_boot=int(n_boot),
        block_length=int(block_length),
        seed=int(seed),
        config_hash=digest,
    )


# --------------------------------------------------------------------------
# from result rows to a loss matrix
# --------------------------------------------------------------------------


def _require_one_series(store: ResultsStore, config_hashes: Mapping[str, str], score: str) -> None:
    """Refuse cells that were not scored on the same series bytes.

    ``origin_index`` is a *position* in the series a cell was scored on, so
    two cells line up day for day only if they were scored on the same
    series (same asset, length and content digest — the ``data`` block of
    the config sidecar). For QLIKE the proxy must match as well: a variance
    loss against two different targets is not one comparison
    (docs/M2_NOTES.md, "One scoring target per cell").
    """
    specs = {name: store.read_config(digest)["data"] for name, digest in config_hashes.items()}
    keys = ["asset", "n", "series_sha256"]
    if score == "qlike":
        keys.append("proxy")
    reference_name, reference = next(iter(specs.items()))
    for name, spec in specs.items():
        for key in keys:
            if spec.get(key) != reference.get(key):
                raise ValueError(
                    f"cells {reference_name!r} and {name!r} were scored on different data "
                    f"({key!r} differs): their origin_index values are not the same days, so "
                    "they cannot be aligned into one loss matrix"
                )


def loss_matrix(
    frame: pd.DataFrame,
    score: str,
    *,
    model_col: str = "model",
    policy: MissingPolicy = "flagged",
    store: ResultsStore | None = None,
) -> LossMatrix:
    """Pivot :func:`~volbench.evaluate.run_backtest` rows into a :class:`LossMatrix`.

    ``frame`` holds result rows (one or many cells, e.g. from
    :meth:`~volbench.results.ResultsStore.read_all`) for **one asset and one
    horizon** — the block bootstrap resamples a single time axis, and
    stacking assets or horizons would splice unrelated calendars into one;
    filter first. Each value of ``model_col`` must map to exactly one
    ``config_hash`` (one cell), otherwise the same name would mix two
    experiments; pass ``model_col="config_hash"`` or filter to disambiguate.

    Models are aligned on ``origin_index``, which is a position in the
    series a cell was scored on. Rows alone cannot prove two cells share a
    series; pass ``store`` and the config sidecars are checked — same
    asset, length and series digest for every model, and the same proxy
    when ``score`` is ``"qlike"`` — before anything is aligned. Without a
    store that is the caller's guarantee.

    Under ``policy="flagged"`` (default; the Phase 2 brief) an origin is
    unusable for a model if that row carries any ``missing_reason``, whether
    or not the requested score itself is NaN — so every score of a model set
    is compared on the same origins. ``policy="score"`` marks only rows whose
    requested score is NaN. Either way nothing is dropped here: unusable
    entries are NaN, counted in ``n_flagged``, and the DM/MCS routines record
    what they drop.
    """
    if policy not in ("flagged", "score"):
        raise ValueError(f"policy must be 'flagged' or 'score', got {policy!r}")
    required = ["asset", "origin_index", "horizon", "config_hash", "missing_reason", score]
    if model_col not in required:
        required.append(model_col)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"result frame is missing columns {missing}")
    if frame.empty:
        raise ValueError("result frame is empty")

    assets = sorted({str(a) for a in frame["asset"].unique()})
    if len(assets) != 1:
        raise ValueError(
            f"loss_matrix needs rows for exactly one asset, got {assets}: the block bootstrap "
            "resamples one time axis, so filter to one asset per comparison"
        )
    horizons = sorted({int(h) for h in frame["horizon"].unique()})
    if len(horizons) != 1:
        raise ValueError(
            f"loss_matrix needs rows for exactly one horizon, got {horizons}: filter to one "
            "horizon per comparison"
        )
    keys = frame[model_col].astype(str)
    hashes_per_model = frame.assign(_key=keys).groupby("_key")["config_hash"].nunique()
    ambiguous = sorted(str(k) for k, count in hashes_per_model.items() if int(count) > 1)
    if ambiguous:
        raise ValueError(
            f"{model_col}={ambiguous} maps to more than one config_hash (several cells share "
            "the name); filter to one cell per model or use model_col='config_hash'"
        )
    duplicated = frame.assign(_key=keys).duplicated(subset=["_key", "origin_index"])
    if bool(duplicated.any()):
        raise ValueError("duplicate (model, origin_index) rows in result frame")

    values = pd.to_numeric(frame[score], errors="coerce").astype(float)
    if policy == "flagged":
        flagged = frame["missing_reason"].fillna("").astype(str).str.len() > 0
        values = values.where(~flagged.to_numpy(), np.nan)
    wide = (
        frame.assign(_key=keys, _loss=values.to_numpy())
        .pivot(index="origin_index", columns="_key", values="_loss")
        .sort_index()
    )
    wide.columns = pd.Index([str(c) for c in wide.columns], name=model_col)
    wide.index = pd.Index(wide.index.astype(np.int64), name="origin_index")
    n_flagged = {str(c): int(wide[c].isna().sum()) for c in wide.columns}
    config_hashes = {
        str(k): str(frame.loc[keys.to_numpy() == k, "config_hash"].iloc[0]) for k in wide.columns
    }
    if store is not None:
        _require_one_series(store, config_hashes, score)
    return LossMatrix(
        values=wide,
        score=score,
        asset=assets[0],
        horizon=horizons[0],
        n_flagged=n_flagged,
        config_hashes=config_hashes,
    )


# --------------------------------------------------------------------------
# the multiplicity-aware entry point
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelComparison:
    """MCS and pairwise DM matrix over one model set, reported together.

    ``mcs`` is the primary "who wins" answer; ``dm`` is context. See
    :func:`compare_models` for why they travel together.
    """

    mcs: MCSResult
    dm: DMMatrix
    config_hash: str

    def table(self) -> pd.DataFrame:
        """One row per model: mean loss, MCS p-value, membership, origins used."""
        rows = [
            {
                "model": name,
                "mean_loss": self.mcs.mean_loss[name],
                "mcs_p_value": self.mcs.p_values[name],
                "in_mcs": name in self.mcs.included,
                "n": self.mcs.n,
                "n_dropped": self.mcs.n_dropped,
            }
            for name in self.mcs.models
        ]
        return pd.DataFrame(rows).sort_values("mean_loss", kind="stable").reset_index(drop=True)


def compare_models(
    losses: LossMatrix | pd.DataFrame | NDArray[np.float64],
    *,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    block_length: int | None = None,
    statistic: MCSStatistic = "range",
    horizon: int | None = None,
    kernel: Kernel = "rectangular",
    hln: bool = True,
    model_names: Sequence[str] | None = None,
) -> ModelComparison:
    """The full model set in, the MCS **and** the pairwise DM matrix out.

    The two are returned together on purpose. Pairwise Diebold-Mariano
    p-values are **not multiplicity-corrected**: with ``m`` models there are
    ``m(m-1)/2`` tests, and reading the smallest p-value as "model A beats
    model B" ignores every other comparison that was run. The Model
    Confidence Set controls that family-wise error by construction and is
    the primary "who wins" tool (docs/metrics_reference.md, "Comparison
    inference"); the DM matrix is reported alongside it as descriptive
    context for individual pairs, never as the headline.

    ``horizon`` defaults to the :class:`LossMatrix`'s horizon (or 1 for bare
    inputs); it sets the DM truncation lag ``h - 1`` and floors the default
    block length. See :func:`model_confidence_set` and :func:`dm_matrix` for
    the remaining parameters and the NaN policy (listwise for the MCS,
    pairwise-complete for DM; both record ``n_dropped``).
    """
    if horizon is None:
        horizon = losses.horizon if isinstance(losses, LossMatrix) else 1
    mcs = model_confidence_set(
        losses,
        seed=seed,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        statistic=statistic,
        horizon=horizon,
        model_names=model_names,
    )
    dm = dm_matrix(losses, horizon=horizon, kernel=kernel, hln=hln, model_names=model_names)
    digest = config_hash(
        {
            "method": "compare_models",
            "mcs": mcs.config_hash,
            "dm": {"horizon": int(horizon), "lag": dm.lag, "kernel": kernel, "hln": bool(hln)},
            "package_version": package_version(),
        }
    )
    return ModelComparison(mcs=mcs, dm=dm, config_hash=digest)
