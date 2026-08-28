"""Shared machinery for the zero-shot time-series foundation-model adapters.

The concrete adapters — :mod:`tsfm_chronos`, :mod:`tsfm_timesfm`,
:mod:`tsfm_moirai`, :mod:`tsfm_timegpt` — differ only in how they load a
checkpoint and turn a context array into a quantile forecast. Everything that
touches the evaluation contract lives here, once, so the four adapters cannot
drift apart on it.

THE MAPPING, STATED ONCE AND PROMINENTLY (a design choice flagged for review,
not a hidden convention):

1. ``fit(train)`` takes a 1-D **realized-variance** series in daily units —
   the same input contract as HAR, never returns — and records the trailing
   ``context_length`` observations as the model's context. Nothing is
   estimated: decision D-005 makes every TSFM zero-shot, and there is no
   fine-tuning path in this package by construction.
2. ``predict(h)`` feeds that context to the foundation model, takes the
   model's own predictive distribution of RV at ``t+h`` — the models emit it
   as a quantile grid — and uses its **MEAN** as the variance forecast
   ``vhat``, computed under a **lognormal tail closure**
   (:func:`tail_closed_grid_mean`, and the section below on why that closure).
   The mean is the right functional — QLIKE and MSE are both minimized at the
   conditional mean — so what had to be fixed was never the choice of
   functional but how the mean was computed. The model's native point head,
   where one exists (TimesFM, TimeGPT), is recorded next to the grid in the
   fitted ``spec()`` but is not used, so that every adapter is scored on one
   estimator.
3. The scored object is ``Normal(mu=0, sigma=sqrt(vhat))`` over the
   next-period **return** — the same shape HAR emits (CLAUDE.md rule 2: a
   model's variance forecast is the variance of its return distribution). The
   RV quantiles themselves are kept in the fitted ``spec()`` under
   ``rv_forecasts`` for inspection; they are never what gets scored.

Units. The models are not scale-free at the level of a daily variance
(~1e-4): Moirai-2's standard scaler adds ``1e-5`` to the variance before
taking the square root, which flattens such a series into a constant, and
other pipelines carry similar epsilons. ``input_scale`` (default ``1e4``, i.e.
variance in percent-squared) multiplies the context before it reaches the
model and divides the quantiles on the way back. It is a fixed, data-
independent constant recorded in ``spec()`` — a unit convention, not a fitted
transform, and therefore cannot leak.

Post-processing of the returned RV quantiles, in this order and each counted
in the fitted ``spec()``: (i) quantile crossing is repaired by sorting the
grid (the rearrangement of Chernozhukov, Fernández-Val & Galichon, 2010);
(ii) negative quantiles — the models are not constrained to a positive
support — are clipped at zero. A forecast whose mean is still non-positive or
non-finite raises ``ValueError``, which the evaluator records as a
``predict_error`` row rather than scoring a meaningless variance.

THE TAIL CLOSURE, AND WHY IT IS LOGNORMAL
=========================================

A 9-level 0.1...0.9 grid says nothing about the law outside its outermost
levels, and **20 % of the probability mass lives there**. Reading the tails
as flat — a point atom of ``taus[0]`` at ``values[0]`` and ``1 - taus[-1]``
at ``values[-1]`` — is one assumption among several, and on a right-skewed RV
law it is the one that understates the mean. That is the D-014 truncation
bias in the same family and for the same reason (``docs/design.md``), and
``docs/P3_TSFM_VARIANCE_AUDIT.md`` measured it on the real panel: the
tail-closed mean is **11 % to 21 % above** the flat-tailed one at the panel
median (``chronos`` 1.135, ``timesfm`` 1.201, ``moirai`` 1.111), one-directional
on every asset and every config, moving VaR and ES by exactly its square root.

**The closure is lognormal because realized volatility is approximately
lognormally distributed.** That is one of the better-established stylized
facts in the realized-volatility literature — Andersen, Bollerslev, Diebold &
Labys (2001), "The Distribution of Realized Exchange Rate Volatility",
*JASA* 96(453), 42-55, and (2003), "Modeling and Forecasting Realized
Volatility", *Econometrica* 71(2), 579-625, which is also the reason every
log-RV model in this package (``models/har.py``, ``models/lgbm.py``,
``models/sf.py``) works in logs at all. A lognormal tail on an RV quantile
grid is therefore the closure with literature behind it, not the closure in
the middle of the range.

The grid's own interior is left exactly as the checkpoint emitted it; only
the two atoms are re-expressed. **The tail beyond q(0.1) / q(0.9) is
genuinely unidentified**, so :func:`grid_mean_under_closures` reports the
flat, lognormal and log-linear readings side by side on demand — the paper
states a range rather than implying the grid pins one number.

**When the closure cannot be fitted, the flat tails stand.** A lognormal
cannot describe a grid holding a zero — a clipped ``chronos``/``moirai``
quantile or a package-floored ``timesfm`` one — and those origins (119, 215
and 12 of 2,199 respectively, at h=1 on the primary grid) keep the flat-tailed
mean and record ``tail_closure: "flat"`` in the fitted ``spec()``. They retain
the understatement; imputing a shape onto a grid that contradicts it would be
worse than reporting how often it happened.

``update(train)`` is context extension: the new fitted object holds the
trailing window of ``train`` and nothing else, which is exactly what ``fit``
would have produced — re-conditioning is exact, no estimation is involved,
and ``update`` on the fit window reproduces the fit bit for bit. Because every
origin costs one forward pass whether it is a "refit" or an "update",
``refit_every`` changes nothing for a TSFM; the runner's ``conditioned_through``
column stays honest either way.

Determinism. None of the wrapped inference paths sample: Chronos-Bolt,
Chronos-2, TimesFM 2.5 and Moirai-2 emit quantiles directly. The
torch-backed adapters still seed torch before every forward pass and run
under ``inference_mode`` so that assumption is checkable, and each
tsfm-marked test pins bit-identity (same context in, same forecast out,
twice). ``spec()`` carries the checkpoint id and the resolved commit hash of
its weights, so the config hash moves whenever the weights do.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from volbench.dist import Distribution, Normal

__all__ = [
    "CLOSURES",
    "DEFAULT_INPUT_SCALE",
    "MIN_CONTEXT",
    "VARIANCE_FROM",
    "FittedTSFM",
    "RVQuantileForecast",
    "TSFMBackend",
    "ZeroShotRVModel",
    "checkpoint_slug",
    "grid_mean_under_closures",
    "quantile_grid_mean",
    "rearrange_quantiles",
    "resolve_hf_revision",
    "tail_closed_grid_mean",
    "validated_rv",
]

#: Context multiplier: daily variance -> percent-squared (see module docstring).
DEFAULT_INPUT_SCALE: Final = 1e4

#: Shortest context ``fit``/``update`` accept: one TimesFM input patch. The
#: models left-pad shorter inputs, but a forecast from a handful of points is
#: not the experiment; the protocol's windows are 1000 observations.
MIN_CONTEXT: Final = 32

#: The tail closures :func:`grid_mean_under_closures` reports, in the order a
#: sensitivity table should read them: the reading that shipped before the fix,
#: the one that ships now, and the one that follows the grid's own edge
#: spacing. Not a config option — the shipped closure is fixed, and this is the
#: vocabulary for reporting what the others would have given.
CLOSURES: Final = ("flat", "lognormal", "loglinear")

#: What ``spec()`` records as the estimator behind ``vhat``, and therefore part
#: of every TSFM config hash. It names the closure because the closure decides
#: the number: leaving the label at ``mean_of_rv_quantile_grid`` while changing
#: how that mean is computed would make every sidecar a false statement.
VARIANCE_FROM: Final = "lognormal_tail_closed_mean_of_rv_quantile_grid"

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------


def validated_rv(train: NDArray[np.float64], minimum: int = MIN_CONTEXT) -> NDArray[np.float64]:
    """A finite, non-negative 1-D realized-variance series of at least ``minimum`` points."""
    rv = np.asarray(train, dtype=np.float64)
    if rv.ndim != 1 or rv.size < minimum:
        raise ValueError(
            f"train must be a 1-D realized-variance series with at least {minimum} observations"
        )
    if not np.isfinite(rv).all() or (rv < 0.0).any():
        raise ValueError("realized-variance series must be finite and non-negative")
    return rv


def quantile_grid_mean(taus: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    """Mean of the law whose quantile function linearly interpolates ``(taus, values)``.

    Flat outside the grid: probability mass ``taus[0]`` sits at ``values[0]``
    and ``1 - taus[-1]`` at ``values[-1]``. This is the first moment of
    ``volbench.evaluate._moments_from_quantile_grid`` — the models package
    must not import the evaluator, so the formula is repeated here and the
    equality is pinned by test.
    """
    t = np.asarray(taus, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or t.shape != v.shape or t.size < 2:
        raise ValueError("taus and values must be equal-length 1-D arrays (size >= 2)")
    lo_mass, hi_mass = float(t[0]), 1.0 - float(t[-1])
    w = np.diff(t)
    return (
        lo_mass * float(v[0]) + float(np.sum(w * (v[:-1] + v[1:]) / 2.0)) + hi_mass * float(v[-1])
    )


def _lognormal_fit(
    taus: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, float] | None:
    """``(mu, sigma)`` of a lognormal fitted to the grid by OLS in log-z space.

    ``log q_tau = mu + sigma Phi^{-1}(tau)``, so the fit is one regression of
    the logged quantiles on the standard-normal quantiles of their own levels.
    Returns ``None`` — never a guess — when the grid holds a non-positive value
    (a clipped quantile) or is degenerate, since a lognormal can describe
    neither.
    """
    if np.any(values <= 0.0) or float(values[-1]) <= float(values[0]):
        return None
    z = stats.norm.ppf(taus)
    y = np.log(values)
    sigma = float(np.sum((z - z.mean()) * (y - y.mean())) / np.sum((z - z.mean()) ** 2))
    if not (math.isfinite(sigma) and sigma > 0.0):
        return None
    return float(y.mean() - sigma * z.mean()), sigma


def tail_closed_grid_mean(
    taus: NDArray[np.float64], values: NDArray[np.float64], closure: str = "lognormal"
) -> float:
    """The grid's mean with the flat tails replaced by ``closure``'s own tails.

    The interior — the mass between ``taus[0]`` and ``taus[-1]`` — is the same
    trapezoid :func:`quantile_grid_mean` uses and is left exactly as the
    checkpoint emitted it. Only the two atoms are re-expressed:

    ``"flat"``
        No closure at all: identical to :func:`quantile_grid_mean`, kept in
        this vocabulary so a sensitivity table can name what shipped before.
    ``"lognormal"``
        A lognormal fitted to the whole grid (:func:`_lognormal_fit`), whose
        partial expectations are closed form:
        ``E[X 1{X < q_tau}] = exp(mu + s^2/2) Phi(z_tau - s)`` and its
        complement above. **This is the shipped closure** (module docstring).
    ``"loglinear"``
        The same shape fitted to the outermost *pair* of levels at each end,
        so the extrapolated tail follows the grid's own outer spacing rather
        than its global shape. Heavier whenever the grid fans out at the edges.

    Returns NaN when the closure cannot be fitted; callers decide what to do
    about that, and :meth:`FittedTSFM.predict` falls back to the flat reading
    and records having done so.
    """
    t = np.asarray(taus, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or t.shape != v.shape or t.size < 2:
        raise ValueError("taus and values must be equal-length 1-D arrays (size >= 2)")
    if closure == "flat":
        return quantile_grid_mean(t, v)
    if closure == "lognormal":
        lo_fit = hi_fit = _lognormal_fit(t, v)
    elif closure == "loglinear":
        lo_fit = _lognormal_fit(t[:2], v[:2])
        hi_fit = _lognormal_fit(t[-2:], v[-2:])
    else:
        raise ValueError(f"unknown closure {closure!r}; known: {CLOSURES}")
    if lo_fit is None or hi_fit is None:
        return math.nan
    w = np.diff(t)
    interior = float(np.sum(w * (v[:-1] + v[1:]) / 2.0))
    lo_mu, lo_sigma = lo_fit
    hi_mu, hi_sigma = hi_fit
    lower = math.exp(lo_mu + 0.5 * lo_sigma**2) * float(
        stats.norm.cdf(stats.norm.ppf(t[0]) - lo_sigma)
    )
    upper = math.exp(hi_mu + 0.5 * hi_sigma**2) * float(
        stats.norm.sf(stats.norm.ppf(t[-1]) - hi_sigma)
    )
    return lower + interior + upper


def grid_mean_under_closures(
    taus: NDArray[np.float64], values: NDArray[np.float64]
) -> dict[str, float]:
    """Every closure in :data:`CLOSURES`, for one grid — the sensitivity, on demand.

    The tail beyond the outermost levels is genuinely unidentified by the grid,
    so a single number overstates what the data support. This is what lets the
    paper state a range instead: ``docs/P3_TSFM_VARIANCE_AUDIT.md`` §3.2 puts
    the panel medians of ``closed / flat`` at 1.11-1.20 (lognormal) against
    1.09-1.10 (log-linear) and 1.21 (an assumption-free empirical closure,
    computable only after the fact and therefore not here).

    A closure that cannot be fitted reports NaN rather than being dropped: how
    often that happens is itself part of the sensitivity.
    """
    return {name: tail_closed_grid_mean(taus, values, name) for name in CLOSURES}


def rearrange_quantiles(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """Sort a quantile grid into non-decreasing order; also count the adjacent pairs that cross."""
    v = np.asarray(values, dtype=np.float64)
    crossings = int(np.sum(np.diff(v) < 0.0))
    return (np.sort(v) if crossings else v.copy()), crossings


def checkpoint_slug(family: str, checkpoint: str) -> str:
    """``"amazon/chronos-bolt-small" -> "chronos_bolt_small"``; the family prefix is added once."""
    base = checkpoint.rstrip("/").rsplit("/", 1)[-1].lower()
    slug = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return slug if slug.startswith(family) else f"{family}_{slug}"


def resolve_hf_revision(repo_id: str, revision: str | None = None) -> str:
    """The commit hash the local Hugging Face cache holds for ``repo_id`` at ``revision``.

    Pure filesystem: no network. A 40-hex ``revision`` is already a commit and
    is returned as is; a branch or tag name (default ``main``) is read from
    the cache's ``refs/`` directory, which ``from_pretrained`` populates when
    it fetches the checkpoint. Call it *after* loading, never before.
    """
    if revision is not None and _SHA1.match(revision):
        return revision
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    folder = str(repo_folder_name(repo_id=repo_id, repo_type="model"))
    repo_dir: Path = Path(str(constants.HF_HUB_CACHE)) / folder
    ref: Path = repo_dir / "refs" / (revision or "main")
    if not ref.is_file():
        raise RuntimeError(
            f"{repo_id}@{revision or 'main'} is not in the local Hugging Face cache "
            f"({repo_dir}); load the checkpoint before asking for its revision"
        )
    sha = ref.read_text(encoding="utf-8").strip()
    if not _SHA1.match(sha):
        raise RuntimeError(f"unexpected ref content for {repo_id}@{revision or 'main'}: {sha!r}")
    return sha


# --------------------------------------------------------------------------
# backend protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RVQuantileForecast:
    """A model's own forecast of RV over ``h`` steps, in the caller's units.

    ``eq=False``: numpy fields (see ``Origin`` in splitter.py).

    Attributes
    ----------
    taus:
        The quantile levels the model was trained to emit, ascending.
    values:
        Shape ``(h, len(taus))``: row ``k`` is the grid for step ``k+1``.
    native_mean:
        Shape ``(h,)`` when the model has a point/mean head of its own
        (TimesFM, TimeGPT), else ``None``. Recorded, never scored.
    """

    taus: tuple[float, ...]
    values: NDArray[np.float64]
    native_mean: NDArray[np.float64] | None = None


@runtime_checkable
class TSFMBackend(Protocol):
    """One loaded foundation model behind a uniform quantile interface.

    A backend is dumb on purpose: it receives the context already in model
    units and returns quantiles in those same units. Scaling, clipping,
    rearrangement and the return-distribution mapping are all done by
    :class:`FittedTSFM`, identically for every model.
    """

    @property
    def taus(self) -> tuple[float, ...]: ...

    @property
    def max_context(self) -> int: ...

    def identity(self) -> dict[str, Any]:
        """Everything that pins the weights and numerics: checkpoint, revision, versions, dtype."""
        ...

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        """Quantiles of the next ``h`` values given ``context`` (1-D, model units)."""
        ...


# --------------------------------------------------------------------------
# the shared adapter
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroShotRVModel:
    """Base of the four adapters: zero-shot, RV in, return distribution out.

    Subclasses add the checkpoint fields, implement ``name`` (a stable slug
    such as ``chronos_bolt_small``) and ``_load_backend()``. ``backend`` is an
    injection point: tests pass a fake so the whole contract runs in CI
    without weights; it never enters ``spec()`` (``identity()`` does), and
    ``None`` — the default, and the only picklable form a
    :data:`~volbench.evaluate.ModelFactory` should produce — loads the real
    checkpoint lazily on first use.

    Parameters
    ----------
    context_length:
        Trailing observations handed to the model. ``None`` means the whole
        training window, capped at the backend's maximum context.
    input_scale:
        Unit multiplier applied to the context (see module docstring).
    """

    context_length: int | None = None
    input_scale: float = DEFAULT_INPUT_SCALE
    backend: TSFMBackend | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.context_length is not None and self.context_length < MIN_CONTEXT:
            raise ValueError(f"context_length must be >= {MIN_CONTEXT} or None")
        if not (math.isfinite(self.input_scale) and self.input_scale > 0.0):
            raise ValueError("input_scale must be a positive finite number")

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _load_backend(self) -> TSFMBackend:
        raise NotImplementedError

    def _identity(self) -> dict[str, Any]:
        return self._resolved_backend().identity()

    def _resolved_backend(self) -> TSFMBackend:
        return self.backend if self.backend is not None else self._load_backend()

    def spec(self) -> dict[str, Any]:
        """Hyperparameters plus the backend's identity.

        Resolving the identity needs the checkpoint's commit hash, so on a
        model whose weights are not loaded yet this loads them (downloading
        on first use) — the same work ``fit`` would do a moment later.
        """
        return {
            "model": self.name,
            "family": "tsfm",
            "zero_shot": True,
            "context_length": self.context_length,
            "input_scale": self.input_scale,
            "variance_from": VARIANCE_FROM,
            **self._identity(),
        }

    def _context_of(self, rv: NDArray[np.float64], backend: TSFMBackend) -> NDArray[np.float64]:
        cap = backend.max_context
        if self.context_length is not None:
            cap = min(cap, self.context_length)
        return rv[-cap:].copy()

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedTSFM:
        """Record the trailing context; nothing is estimated (zero-shot, D-005)."""
        rv = validated_rv(train)
        backend = self._resolved_backend()
        return FittedTSFM(model=self, backend=backend, context=self._context_of(rv, backend))


@dataclass(frozen=True, eq=False)
class FittedTSFM:
    """A zero-shot model with its context window fixed at a forecast origin.

    ``eq=False``: numpy field (see ``Origin`` in splitter.py). The last
    element of ``context`` is the origin's own observation — the context ends
    at the origin and never touches ``t+1``; ``predict(h)`` is the model's
    step-``h`` forecast past that point.
    """

    model: ZeroShotRVModel
    backend: TSFMBackend
    context: NDArray[np.float64]
    _cache: dict[int, RVQuantileForecast] = field(default_factory=dict, repr=False)
    _meta: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.model.name

    def spec(self) -> dict[str, Any]:
        """The model's spec plus what this context produced so far (cheap: no inference)."""
        return {
            **self.model.spec(),
            "n_context": int(self.context.size),
            "rv_forecasts": {k: dict(v) for k, v in sorted(self._meta.items())},
        }

    def rv_forecast(self, h: int) -> RVQuantileForecast:
        """The model's raw RV quantile forecast for steps ``1..h``, in daily-variance units."""
        if h < 1:
            raise ValueError("h must be >= 1")
        cached = self._cache.get(h)
        if cached is not None:
            return cached
        scale = self.model.input_scale
        raw = self.backend.forecast(self.context * scale, h)
        values = np.asarray(raw.values, dtype=np.float64)
        if values.shape != (h, len(raw.taus)):
            raise RuntimeError(
                f"{type(self.backend).__name__} returned quantiles of shape {values.shape}, "
                f"expected {(h, len(raw.taus))}"
            )
        native = None if raw.native_mean is None else np.asarray(raw.native_mean) / scale
        out = RVQuantileForecast(taus=raw.taus, values=values / scale, native_mean=native)
        self._cache[h] = out
        return out

    def predict(self, h: int) -> Distribution:
        """``Normal(0, sqrt(vhat))`` over the return at ``t+h``; ``vhat`` = mean of the RV grid.

        The mean is taken under the lognormal tail closure (module docstring).
        Where that closure cannot be fitted — a grid holding a zero — the flat
        reading stands and ``tail_closure`` records ``"flat"`` for that origin,
        so the fallback is countable rather than invisible.
        """
        fc = self.rv_forecast(h)
        taus = np.asarray(fc.taus, dtype=np.float64)
        sorted_values, crossings = rearrange_quantiles(fc.values[h - 1])
        clipped = int(np.sum(sorted_values < 0.0))
        grid = np.maximum(sorted_values, 0.0)
        flat = quantile_grid_mean(taus, grid)
        closed = tail_closed_grid_mean(taus, grid, "lognormal")
        applied = "lognormal" if math.isfinite(closed) else "flat"
        vhat = closed if applied == "lognormal" else flat
        self._meta[str(h)] = {
            "taus": [float(t) for t in taus],
            "values": [float(v) for v in grid],
            "mean": vhat,
            "flat_tail_mean": flat,
            "tail_closure": applied,
            "native_mean": None if fc.native_mean is None else float(fc.native_mean[h - 1]),
            "crossings_rearranged": crossings,
            "clipped_at_zero": clipped,
        }
        if not (math.isfinite(vhat) and vhat > 0.0):
            raise ValueError(
                f"{self.name}: predictive mean of RV at h={h} is {vhat!r}, not positive"
            )
        return Normal(mu=0.0, sigma=math.sqrt(vhat))

    def update(self, train: NDArray[np.float64]) -> FittedTSFM:
        """Re-condition by extending the context — exact, nothing re-estimated.

        Handing ``update`` the fit window reproduces the fit exactly; handing
        it the next origin's window moves the context forward by one day.
        Per-origin inference costs the same either way, which is why
        ``refit_every`` is irrelevant to a TSFM (the module docstring).
        """
        rv = validated_rv(train)
        return FittedTSFM(
            model=self.model, backend=self.backend, context=self.model._context_of(rv, self.backend)
        )
