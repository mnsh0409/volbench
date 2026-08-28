#!/usr/bin/env python
"""What the TSFM adapters' grid-to-variance reduction discards, measured.

The three zero-shot TSFM adapters emit a quantile grid over **realized
variance** and reduce it to one number — the grid's mean — before the scored
object is built (``volbench.models.tsfm_common.FittedTSFM.predict``). Nothing
downstream ever sees the grid: ``run_backtest`` hashes the *unfitted* probe's
``spec()``, so the per-origin ``rv_forecasts`` block never reaches disk
(docs/P3_INSTRUMENTATION_GAP.md §2.1).

This module re-runs the reduction at the primary grid's own origins and records
both sides of it: the native grid as the checkpoint emitted it, and the
``Normal(0, sqrt(vhat))`` that was actually scored. It writes nothing to any
:class:`~volbench.results.ResultsStore`, moves no config hash and rewrites no
fragment — the same contract
:mod:`volbench.benchmarks.fit_diagnostics_probe` runs under, whose origin
bookkeeping it reuses.

Three axes, kept apart on purpose
=================================
The grid is over RV and the scored object is over the next-period **return**,
so "how far apart are they" is only a question once an axis is named:

- **RV axis.** What the reduction throws away is a whole predictive law over
  RV, replaced by its own mean. :func:`grid_law_moments` gives the four moments
  of that law — the mean is exactly the ``vhat`` the adapter scores.
- **RV axis, tail closure.** The grid's mean is computed with *flat tails*
  outside the outermost level, which puts a point mass of ``taus[0]`` at
  ``values[0]`` and ``1 - taus[-1]`` at ``values[-1]``. For a right-skewed RV
  law that understates the mean (D-014's family of bias; docs/design.md). The
  size of the understatement is not identified by the grid alone, so
  :func:`tail_closed_mean` reports it under a named closure and the caller
  reports the range across closures rather than one number.
- **Return axis.** The distributional shape the adapter drops shows up as a
  *scale mixture*: ``r = sqrt(V) Z`` with ``V`` the grid law and ``Z``
  standard normal. That mixture is symmetric like the scored Normal, has the
  same variance ``E[V] = vhat``, and differs from it in exactly one number —
  excess kurtosis ``3 Var(V) / E[V]^2`` (:func:`mixture_excess_kurtosis`) —
  plus the VaR/ES that follows from it (:func:`mixture_return_quantile`).

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical --extra tsfm python -m \\
        volbench.benchmarks.tsfm_distribution_probe --assets SPY --n-origins 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from volbench.benchmarks.grid_primary import ARM, asset_data, model_configs
from volbench.data.panel import build_panel
from volbench.results import ResultsStore
from volbench.runner import AssetData, ModelConfig

__all__ = [
    "TSFM_LABELS",
    "GridMoments",
    "grid_law_moments",
    "mixture_excess_kurtosis",
    "mixture_return_quantile",
    "normal_var_es",
    "probe_asset",
    "tail_closed_mean",
]

#: The three configs whose predictive object is a quantile grid. ``patchtst``
#: is deliberately absent: it emits a point forecast of log RV and has no grid
#: at all (see :func:`patchtst_record`).
TSFM_LABELS: Final = ("chronos", "timesfm", "moirai")

#: The scoring levels the primary grid used (``DEFAULT_LEVELS``, recorded in
#: every sidecar's ``scoring`` block).
ARM_LEVELS: Final = (0.01, 0.025, 0.05)

#: Below this ratio of variance to squared mean, a grid is treated as a point
#: mass: the closed-form ``m2 - m1^2`` of a collapsed grid is float noise, and a
#: real grid's ratio is ~0.4 (measured on the panel), so nothing lies between.
_DEGENERATE_REL_VAR: Final = 1e-12

#: Quadrature resolution for the scale mixture. Uniform in probability, so the
#: flat-tail point masses are represented by the share of nodes that falls in
#: them — no special-casing, and no RNG. The integrand is a smooth CDF, so this
#: resolves the mixture quantile to ~1e-4 relative, well inside what the audit
#: reports; the bisection below then costs nothing next to it.
#: Bisection steps for the mixture quantile: the bracket is 80 standard
#: deviations wide, so 60 halvings put the residual at ~1e-16 of it.
_MIXTURE_NODES: Final = 4_001
_BISECTION_STEPS: Final = 60


# --------------------------------------------------------------------------
# the grid law: exact moments of the flat-tailed linear interpolant
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridMoments:
    """Moments of the law whose quantile function is ``(taus, values)``.

    ``mean`` is the number the adapter scores as ``vhat`` — pinned equal to
    :func:`volbench.models.tsfm_common.quantile_grid_mean` by test.
    """

    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float


def _raw_moment(taus: NDArray[np.float64], values: NDArray[np.float64], k: int) -> float:
    """``E[X^k]`` for the flat-tailed linear-interpolant law, in closed form.

    On a segment of probability width ``w`` where the quantile function runs
    linearly from ``a`` to ``b``, ``∫ Q^k du = w (b^{k+1} - a^{k+1}) /
    ((k+1)(b - a))``, which degenerates to ``w a^k`` at ``a == b``. Outside the
    grid the quantile function is flat, contributing point masses ``taus[0]``
    at ``values[0]`` and ``1 - taus[-1]`` at ``values[-1]``. At ``k = 1`` this
    is ``volbench.models.tsfm_common.quantile_grid_mean`` term for term.
    """
    lo_mass, hi_mass = float(taus[0]), 1.0 - float(taus[-1])
    a, b = values[:-1], values[1:]
    w = np.diff(taus)
    flat = np.isclose(b, a)
    ratio = np.where(
        flat,
        np.power(a, k),
        (np.power(b, k + 1) - np.power(a, k + 1)) / ((k + 1) * np.where(flat, 1.0, b - a)),
    )
    interior = float(np.sum(w * ratio))
    return lo_mass * float(values[0]) ** k + interior + hi_mass * float(values[-1]) ** k


def grid_law_moments(taus: NDArray[np.float64], values: NDArray[np.float64]) -> GridMoments:
    """Mean, variance, skewness and excess kurtosis of the grid law.

    Exact for that law, not an estimate: the quantile function is piecewise
    linear, so every raw moment is a closed-form sum (:func:`_raw_moment`).
    A degenerate grid — every level at one value, which is what a collapsed
    forecast looks like (Moirai at raw units; see tsfm_moirai) — reports zero
    variance and NaN standardized moments rather than dividing by the float
    noise the closed-form difference leaves behind. The test is *relative*:
    ``m2 - m1^2`` cancels to ~1e-16 of ``m1^2`` on a collapsed grid and to
    ~0.4 of it on a real one, so no threshold in between can be reached by a
    forecast that has any spread at all.
    """
    t = np.asarray(taus, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or t.shape != v.shape or t.size < 2:
        raise ValueError("taus and values must be equal-length 1-D arrays (size >= 2)")
    m1, m2, m3, m4 = (_raw_moment(t, v, k) for k in (1, 2, 3, 4))
    var = max(m2 - m1 * m1, 0.0)
    if var <= _DEGENERATE_REL_VAR * m1 * m1:
        return GridMoments(mean=m1, variance=0.0, skewness=math.nan, excess_kurtosis=math.nan)
    sd = math.sqrt(var)
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
    return GridMoments(
        mean=m1,
        variance=var,
        skewness=mu3 / sd**3,
        excess_kurtosis=mu4 / var**2 - 3.0,
    )


# --------------------------------------------------------------------------
# tail closures: what the flat tails cost the mean
# --------------------------------------------------------------------------


def _lognormal_fit(
    taus: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, float] | None:
    """``(mu, sigma)`` of a lognormal fitted to the grid by OLS in log-z space.

    ``log q_tau = mu + sigma Phi^{-1}(tau)``. Returns ``None`` when the grid
    holds a non-positive value (a clipped quantile) or is degenerate, since a
    lognormal cannot describe either.
    """
    if np.any(values <= 0.0) or float(values[-1]) <= float(values[0]):
        return None
    z = stats.norm.ppf(taus)
    y = np.log(values)
    sigma = float(np.sum((z - z.mean()) * (y - y.mean())) / np.sum((z - z.mean()) ** 2))
    if not (math.isfinite(sigma) and sigma > 0.0):
        return None
    return float(y.mean() - sigma * z.mean()), sigma


def tail_closed_mean(
    taus: NDArray[np.float64], values: NDArray[np.float64], closure: str = "lognormal"
) -> float:
    """The grid's mean with the flat tails replaced by ``closure``'s own tails.

    The interior of the grid is left exactly as the checkpoint emitted it; only
    the two closures — the mass below ``taus[0]`` and above ``taus[-1]`` — are
    re-expressed. Two are implemented, and the honest reading is the range
    between them rather than either alone:

    ``"lognormal"``
        A lognormal fitted to the whole grid (:func:`_lognormal_fit`), whose
        partial expectations are closed form: the mass above ``tau`` integrates
        to ``exp(mu + sigma^2/2) (1 - Phi(z_tau - sigma))``.
    ``"loglinear"``
        The same shape fitted to the outermost *pair* of levels at each end,
        so the extrapolated tail follows the grid's own outer spacing rather
        than its global shape. Heavier whenever the grid fans out at the edges.

    Returns NaN when the closure cannot be fitted — a clipped (zero) quantile
    is the case that occurs in practice.
    """
    t = np.asarray(taus, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if closure == "lognormal":
        lo_fit = hi_fit = _lognormal_fit(t, v)
    elif closure == "loglinear":
        lo_fit = _lognormal_fit(t[:2], v[:2])
        hi_fit = _lognormal_fit(t[-2:], v[-2:])
    else:
        raise ValueError(f"unknown closure {closure!r}")
    if lo_fit is None or hi_fit is None:
        return math.nan
    a, b = v[:-1], v[1:]
    w = np.diff(t)
    interior = float(np.sum(w * (a + b) / 2.0))
    lo_mu, lo_sigma = lo_fit
    hi_mu, hi_sigma = hi_fit
    # E[X 1{X < q}] = exp(mu + s^2/2) Phi(z_q - s); the upper tail is its complement.
    lo_total = math.exp(lo_mu + 0.5 * lo_sigma**2)
    hi_total = math.exp(hi_mu + 0.5 * hi_sigma**2)
    lower = lo_total * float(stats.norm.cdf(stats.norm.ppf(t[0]) - lo_sigma))
    upper = hi_total * float(stats.norm.sf(stats.norm.ppf(t[-1]) - hi_sigma))
    return lower + interior + upper


# --------------------------------------------------------------------------
# the return axis: the scale mixture the reduction replaces with a Normal
# --------------------------------------------------------------------------


def mixture_excess_kurtosis(variance_of_v: float, mean_of_v: float) -> float:
    """Excess kurtosis of ``r = sqrt(V) Z``, ``Z`` standard normal, ``Z ⊥ V``.

    ``E[r^2] = E[V]`` and ``E[r^4] = 3 E[V^2]``, so the excess kurtosis is
    ``3 Var(V) / E[V]^2`` — three times the squared coefficient of variation of
    the RV law. It is the *whole* of what the Gaussian reduction discards on
    the return axis: the mixture and the scored Normal share their mean (0),
    their variance (``vhat``), and their symmetry.
    """
    if not (math.isfinite(mean_of_v) and mean_of_v > 0.0):
        return math.nan
    return 3.0 * variance_of_v / (mean_of_v * mean_of_v)


def _mixture_nodes(taus: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Q(u)`` on a uniform probability grid — the mixing law, discretized."""
    u = (np.arange(_MIXTURE_NODES, dtype=np.float64) + 0.5) / _MIXTURE_NODES
    return np.interp(u, taus, values)


def _mixture_cdf(sigmas: NDArray[np.float64], x: float) -> float:
    """``P(sqrt(V) Z <= x)`` averaged over the discretized mixing law.

    A zero scale — the state a clipped quantile leaves behind — is a point mass
    at zero, so it contributes ``1{x > 0}`` (and ``0.5`` at ``x == 0``) rather
    than a division by zero.
    """
    out = np.empty_like(sigmas)
    positive = sigmas > 0.0
    out[positive] = stats.norm.cdf(x / sigmas[positive])
    out[~positive] = 1.0 if x > 0.0 else (0.5 if x == 0.0 else 0.0)
    return float(np.mean(out))


def mixture_return_quantile(
    taus: NDArray[np.float64], values: NDArray[np.float64], alpha: float
) -> float:
    """The ``alpha``-quantile of ``r = sqrt(V) Z`` with ``V`` the grid law.

    What the adapter would have scored had it carried the model's RV
    uncertainty into the return distribution instead of collapsing it to a
    single variance. Solved by bisection on a deterministic quadrature — no
    RNG, so the number is reproducible bit for bit.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    sigmas = np.sqrt(np.maximum(_mixture_nodes(taus, values), 0.0))
    scale = float(np.max(sigmas))
    if scale <= 0.0:
        return 0.0
    lo, hi = -40.0 * scale, 40.0 * scale
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if _mixture_cdf(sigmas, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_var_es(variance: float, alpha: float) -> tuple[float, float]:
    """``(VaR, ES)`` of ``Normal(0, sqrt(variance))`` — what the store holds."""
    sigma = math.sqrt(variance)
    z = float(stats.norm.ppf(alpha))
    return sigma * z, -sigma * float(stats.norm.pdf(z)) / alpha


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def _fit_origins(store: ResultsStore, manifest: pd.DataFrame, asset: str, label: str) -> list[int]:
    """Every origin the grid scored for this cell, read off its own fragment."""
    rows = manifest[(manifest["asset"] == asset) & (manifest["model"] == label)]
    if rows.empty:
        return []
    frame = store.read(str(rows["config_hash"].iloc[0]))
    return sorted({int(v) for v in frame["origin_index"]})


def _record(
    fitted: Any, levels: Sequence[float], asset: str, label: str, origin: int
) -> dict[str, Any]:
    """One origin's grid, its reduction, and the two axes' comparison."""
    raw = fitted.rv_forecast(1)  # pre-repair, in daily-variance units
    fitted.predict(1)  # populates the repair record on the fitted spec
    meta = fitted.spec()["rv_forecasts"]["1"]
    taus = np.asarray(meta["taus"], dtype=np.float64)
    grid = np.asarray(meta["values"], dtype=np.float64)  # post-repair: sorted, clipped
    moments = grid_law_moments(taus, grid)
    vhat = float(meta["mean"])
    row: dict[str, Any] = {
        "asset": asset,
        "model": label,
        "origin": origin,
        "n_levels": int(taus.size),
        "vhat": vhat,
        "grid_variance": moments.variance,
        "grid_skewness": moments.skewness,
        "grid_excess_kurtosis": moments.excess_kurtosis,
        "grid_cv": math.sqrt(moments.variance) / vhat if vhat > 0 else math.nan,
        "return_excess_kurtosis": mixture_excess_kurtosis(moments.variance, vhat),
        "crossings_rearranged": int(meta["crossings_rearranged"]),
        "clipped_at_zero": int(meta["clipped_at_zero"]),
        "native_mean": meta["native_mean"],
        "mean_lognormal_tail": tail_closed_mean(taus, grid, "lognormal"),
        "mean_loglinear_tail": tail_closed_mean(taus, grid, "loglinear"),
    }
    for i, tau in enumerate(taus):
        row[f"q_{tau:g}"] = float(grid[i])
        row[f"raw_q_{tau:g}"] = float(raw.values[0][i])
    for alpha in levels:
        tag = f"{alpha:.10g}".replace(".", "p")
        scored_var, scored_es = normal_var_es(vhat, alpha)
        row[f"var_{tag}_scored"] = scored_var
        row[f"es_{tag}_scored"] = scored_es
        row[f"var_{tag}_mixture"] = mixture_return_quantile(taus, grid, alpha)
    return row


def patchtst_record(fitted: Any, asset: str, label: str, origin: int) -> dict[str, Any]:
    """PatchTST's analogue: there is no grid, so there is nothing to reduce.

    The net emits ``max_horizon`` direct point outputs under an MSE objective
    and has no quantile head, so its predictive object is one number in log
    space plus a Duan smearing factor. Recorded so the four "foundation models"
    of the addendum are separated rather than assumed alike.
    """
    spec = fitted.spec()
    mu = float(spec["log_forecast"][0])
    factor = float(spec["smearing_factor"][0])
    return {
        "asset": asset,
        "model": label,
        "origin": origin,
        "n_levels": 0,
        "vhat": math.exp(mu) * factor,
        "log_forecast": mu,
        "smearing_factor": factor,
        "epochs_run": spec.get("epochs_run"),
        "best_val_mse": spec.get("best_val_mse"),
        "final_train_mse": spec.get("final_train_mse"),
    }


def probe_asset(
    config: ModelConfig,
    data: AssetData,
    origins: Sequence[int],
    levels: Sequence[float],
) -> list[dict[str, Any]]:
    """Re-run ``config`` at ``origins`` and record both sides of the reduction."""
    splitter = ARM.splitter(1)
    fit_series = data.fit_series(ARM.invalid_target_policy)
    wanted = set(origins)
    rows: list[dict[str, Any]] = []
    for origin in splitter.split(len(data.returns)):
        if origin.origin not in wanted:
            continue
        try:
            fitted = config.factory().fit(fit_series.window(origin.train))
            if config.label == "patchtst":
                row = patchtst_record(fitted, data.asset, config.label, origin.origin)
            else:
                row = _record(fitted, levels, data.asset, config.label, origin.origin)
            row["probe_error"] = None
        except Exception as exc:  # a probe must never take the report down
            row = {
                "asset": data.asset,
                "model": config.label,
                "origin": int(origin.origin),
                "probe_error": f"{type(exc).__name__}: {exc}",
            }
        row["target_index"] = int(origin.test[0])
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/P3_GRID_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/tsfm_dist_probe/probe.parquet"))
    parser.add_argument("--assets", nargs="*", default=["SPY"])
    parser.add_argument("--models", nargs="*", default=[*TSFM_LABELS, "patchtst"])
    parser.add_argument("--n-origins", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    store = ResultsStore(args.store)
    manifest = pd.DataFrame(json.loads(args.manifest.read_text(encoding="utf-8"))["cells"])
    panel = {k: v for k, v in build_panel().items() if k in set(args.assets)}
    data = {name: asset_data(series) for name, series in panel.items()}
    by_label = {c.label: c for c in model_configs(device=args.device)}
    levels = tuple(ARM_LEVELS)

    rows: list[dict[str, Any]] = []
    for label in args.models:
        started = time.perf_counter()
        for asset, datum in data.items():
            every = _fit_origins(store, manifest, asset, label)
            # Evenly spaced across the whole evaluation span, not the first N:
            # a contiguous prefix would sample one volatility regime.
            stride = max(1, len(every) // args.n_origins)
            origins = every[::stride][: args.n_origins]
            rows.extend(probe_asset(by_label[label], datum, origins, levels))
        print(f"  {label:10s} {time.perf_counter() - started:7.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"\n{len(frame)} probed origins -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.argv[0] = "tsfm_distribution_probe"
    raise SystemExit(main())
