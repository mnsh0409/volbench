"""Rolling-origin backtesting: run a model over a series and score it.

The whole module exists to turn one *cell* — an ``(asset, model, splitter,
seed)`` unit, per :mod:`volbench.execute` — into a tidy frame of scored
forecasts, reproducibly.

Conventions this module assumes (fixed for Phase 1, see
``docs/phase1_prompts.md``):

- ``FittedModel.predict(h)`` returns a :class:`~volbench.dist.Distribution`
  over the **next-period return**, never over variance. The variance forecast
  is a property of that distribution.
- Variance is in **daily** units, never annualized.
- CRPS, log score, pinball and VaR hits are therefore computed against the
  realized *return*; QLIKE compares the distribution's variance against a
  realized-variance *proxy*.

Temporal integrity (CLAUDE.md rule 1): every index used here comes from
``RollingOriginSplitter``. Models see ``series[origin.train]`` and nothing
else; targets are read at ``origin.test``, which the splitter guarantees to
start strictly after ``origin``. There is no slicing arithmetic in this file
that the splitter did not produce.

One whole-series statistic does exist here — the content digest that goes
into the config hash — and it is deliberately *not* a leak: it is computed
after the fact, reaches no model, and only ever narrows what the cache is
allowed to serve (:mod:`volbench.results`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volbench.dist import Distribution, Empirical, QuantileGrid
from volbench.execute import Executor, SerialExecutor
from volbench.metrics import qlike
from volbench.models.base import FittedModel, ForecastModel
from volbench.results import (
    ResultsStore,
    array_digest,
    build_config,
    config_hash,
    normalize_frame,
)
from volbench.splitter import Origin, RollingOriginSplitter

__all__ = [
    "DEFAULT_LEVELS",
    "FittedModel",
    "ForecastModel",
    "ModelFactory",
    "Recondition",
    "SupportsUpdate",
    "forecast_moments",
    "run_backtest",
]

#: What happens at the origins between two scheduled refits.
#:
#: ``"daily"`` — the reading of "refit every N days" fixed after M1 report
#: §4.3: parameters come from the last scheduled refit and the model's
#: conditional state is re-filtered on every origin's window through
#: :class:`SupportsUpdate`. ``"none"`` — the forecast is frozen between refits:
#: the behaviour every baseline had before ``update`` existed, kept as an
#: explicit ablation arm rather than a default anyone can fall into.
Recondition = Literal["daily", "none"]

logger = logging.getLogger(__name__)

#: Tail levels for pinball loss and VaR, per docs/metrics_reference.md.
DEFAULT_LEVELS: Final = (0.01, 0.025, 0.05)

#: Fallback quadrature grid for distributions with no closed-form moments.
_FALLBACK_TAUS: Final = np.linspace(1.0 / 2048.0, 1.0 - 1.0 / 2048.0, 1024)


# --------------------------------------------------------------------------
# model interface
# --------------------------------------------------------------------------
#
# During Phase 1 this module declared ``FittedModel``/``ForecastModel`` as its
# own local Protocols, because ``volbench.models`` was being built in a
# parallel stream (docs/phase1_prompts.md streams B and C). At M1 integration
# the two were reconciled: there is now exactly ONE definition of the model
# interface, in :mod:`volbench.models.base`, and this module imports it. They
# are re-exported here so ``volbench.evaluate.ForecastModel`` keeps working.
#
# The reconciliation went C -> B (the local Protocol was widened to the real
# classes' surface, gaining ``name``/``spec()`` on the *fitted* side), because
# all four Phase 1 models honour the return-distribution convention that would
# have forced the opposite direction.
#
# ``volbench.models.base`` is imported rather than ``volbench.models`` on
# purpose: the package root of that subpackage pulls in ``arch`` and ``scipy``,
# and evaluation should not depend on a model backend being installed.


@runtime_checkable
class SupportsUpdate(Protocol):
    """Optional capability: refresh conditioning without re-estimating.

    A GARCH forecast for day ``t`` normally uses parameters estimated at the
    last scheduled refit but conditions on returns *through* ``t``. With a
    ``predict(h)``-only interface there is nowhere to put that newer data, so
    a model that filters recent observations implements ``update``: volbench
    calls it at every non-refit origin with that origin's training window —
    observations dated at or before the origin, exactly what ``fit`` would
    have been handed — when the backtest runs with ``recondition="daily"``
    (the default). Under ``recondition="none"`` it is never called.

    Implementations **must not re-estimate parameters** in ``update`` — that
    would silently defeat the refit schedule and make the reported refit
    cadence a lie. Every Phase 1 baseline implements it (M1 report §4.3 is
    closed); a model without it holds its forecast constant between refits,
    which the ``conditioned_through`` result column records either way.
    """

    def update(self, train: NDArray[np.float64]) -> FittedModel:
        """Re-condition on ``train`` keeping the current parameters."""
        ...


#: Builds a fresh, identically-specified model. Must be picklable and
#: deterministic — Phase 2 executors will call it in another process.
ModelFactory = Callable[[], ForecastModel]


# --------------------------------------------------------------------------
# forecast moments
# --------------------------------------------------------------------------


def _moments_from_quantile_grid(
    taus: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, float]:
    """Mean and variance of the law whose quantile function is the linear
    interpolant of ``(taus, values)``, flat outside the grid.

    Exact for that law: on a segment where the quantile function runs linearly
    from ``a`` to ``b`` over probability width ``w``, ``∫Q du = w(a+b)/2`` and
    ``∫Q² du = w(a²+ab+b²)/3``. The flat tails contribute point masses of
    ``taus[0]`` at ``values[0]`` and ``1-taus[-1]`` at ``values[-1]``.
    """
    lo_mass, hi_mass = float(taus[0]), 1.0 - float(taus[-1])
    lo_val, hi_val = float(values[0]), float(values[-1])
    a, b = values[:-1], values[1:]
    w = np.diff(taus)
    m1 = lo_mass * lo_val + float(np.sum(w * (a + b) / 2.0)) + hi_mass * hi_val
    m2 = (
        lo_mass * lo_val**2 + float(np.sum(w * (a * a + a * b + b * b) / 3.0)) + hi_mass * hi_val**2
    )
    return m1, max(m2 - m1 * m1, 0.0)


def forecast_moments(dist: Distribution) -> tuple[float, float]:
    """Mean and variance of a predictive distribution over returns.

    The variance is the model's variance forecast (the fixed Phase 1
    convention), so it is what QLIKE scores against the proxy. Computed
    through :class:`~volbench.dist.Distribution`'s public interface only.

    - Parametric families — :class:`~volbench.dist.Normal`,
      :class:`~volbench.dist.StudentT`, anything implementing ``mean()`` and
      ``variance()``: their own closed forms, asked for first. This is what
      closed M1 report §4.2: a Student-t forecast used to arrive as a quantile
      grid, and the grid's moments truncated its tails.
    - :class:`~volbench.dist.Empirical`: the plug-in moments of the sample.
      ``ddof=0`` deliberately — the ensemble *is* the predictive law here, the
      same reading under which ``Empirical.crps`` is exact rather than an
      estimate.
    - :class:`~volbench.dist.QuantileGrid`: exact for the interpolated law.
    - anything else: quadrature over a dense quantile grid, which truncates
      mass beyond the outermost tau and so understates very heavy tails.
      Genuinely non-parametric objects are the only ones that should land
      here.
    """
    try:
        return dist.mean(), dist.variance()
    except NotImplementedError:
        pass
    if isinstance(dist, Empirical):
        return float(np.mean(dist.samples)), float(np.var(dist.samples))
    if isinstance(dist, QuantileGrid):
        return _moments_from_quantile_grid(dist.taus, dist.values)
    grid = np.array([dist.quantile(float(t)) for t in _FALLBACK_TAUS], dtype=np.float64)
    return _moments_from_quantile_grid(_FALLBACK_TAUS, grid)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def _level_tag(level: float) -> str:
    """Column suffix for a tail level: ``0.025`` -> ``"0p025"``."""
    return f"{level:.10g}".replace(".", "p").replace("-", "m")


def _score(
    dist: Distribution,
    realized_return: float,
    proxy_var: float,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    """Score one forecast, recording why anything unscorable is missing.

    Nothing is ever dropped: a row is always produced, with NaN where a score
    is undefined and ``missing_reason`` naming every cause.
    """
    reasons: set[str] = set()
    mean, variance = forecast_moments(dist)

    target_ok = math.isfinite(realized_return)
    if not target_ok:
        reasons.add("target_nan")

    out: dict[str, Any] = {
        "forecast_mean": mean,
        "forecast_var": variance,
        "crps": dist.crps(realized_return) if target_ok else math.nan,
    }

    if not target_ok:
        out["log_score"] = math.nan
    else:
        try:
            out["log_score"] = dist.log_score(realized_return)
        except NotImplementedError:
            out["log_score"] = math.nan
            reasons.add("log_score_undefined")

    if math.isnan(proxy_var):
        out["qlike"] = math.nan
        reasons.add("proxy_nan")
    elif math.isinf(proxy_var):
        out["qlike"] = math.nan
        reasons.add("proxy_not_finite")
    elif proxy_var <= 0.0:
        out["qlike"] = math.nan
        reasons.add("proxy_nonpositive")
    elif not (math.isfinite(variance) and variance > 0.0):
        out["qlike"] = math.nan
        reasons.add("forecast_var_nonpositive")
    else:
        out["qlike"] = qlike(variance, proxy_var)

    for level in levels:
        tag = _level_tag(level)
        var_quantile = dist.quantile(level)
        out[f"var_{tag}"] = var_quantile
        out[f"pinball_{tag}"] = dist.pinball(realized_return, level) if target_ok else math.nan
        out[f"hit_{tag}"] = float(realized_return < var_quantile) if target_ok else math.nan

    out["missing_reason"] = "|".join(sorted(reasons))
    return out


def _result_dtypes(levels: tuple[float, ...]) -> dict[str, str]:
    """Pinned dtypes for the result frame.

    Explicit rather than inferred: inference depends on the values a
    particular run happened to produce, and the determinism canary compares
    parquet bytes.
    """
    dtypes: dict[str, str] = {
        "config_hash": "str",
        "asset": "str",
        "model": "str",
        "origin_index": "int64",
        "horizon": "int64",
        "target_index": "int64",
        "fit_origin": "int64",  # -1 on rows that had no fitted model (see _run_block)
        "conditioned_through": "int64",  # likewise
        "refit": "bool",
        "seed": "int64",
        "forecast_mean": "float64",
        "forecast_var": "float64",
        "realized_return": "float64",
        "proxy_name": "str",
        "proxy_var": "float64",
        "crps": "float64",
        "log_score": "float64",
        "qlike": "float64",
        "missing_reason": "str",
    }
    for level in levels:
        tag = _level_tag(level)
        dtypes[f"var_{tag}"] = "float64"
        dtypes[f"pinball_{tag}"] = "float64"
        dtypes[f"hit_{tag}"] = "float64"
    return dtypes


# --------------------------------------------------------------------------
# fault isolation — one bad origin costs one row, never the cell
# --------------------------------------------------------------------------


def _exception_reason(stage: str, exc: BaseException, *, origin: int | None = None) -> str:
    """``missing_reason`` token for an exception: stage, type and message.

    ``fit_error@499: ValueError: realized-variance series must be finite and
    strictly positive``. The origin is named for the stages whose failure can
    affect rows at *other* origins (a scheduled fit serves its whole block).
    Whitespace inside the message is collapsed so the token stays one line in
    a table; nothing else is altered or truncated.
    """
    message = " ".join(str(exc).split())
    where = f"@{origin}" if origin is not None else ""
    return f"{stage}{where}: {type(exc).__name__}" + (f": {message}" if message else "")


def _failure_scores(levels: tuple[float, ...], reason: str) -> dict[str, Any]:
    """The score columns of a row whose forecast never happened.

    Same keys as :func:`_score`, NaN throughout, so a failure row has the
    frame's full schema and the pinned dtypes still apply.
    """
    out: dict[str, Any] = {
        "forecast_mean": math.nan,
        "forecast_var": math.nan,
        "crps": math.nan,
        "log_score": math.nan,
        "qlike": math.nan,
    }
    for level in levels:
        tag = _level_tag(level)
        out[f"var_{tag}"] = math.nan
        out[f"pinball_{tag}"] = math.nan
        out[f"hit_{tag}"] = math.nan
    out["missing_reason"] = reason
    return out


def _log_failure(model_name: str, asset: str, origin: int, reason: str, exc: BaseException) -> None:
    # One line per failed origin, always: a run that failed 300 origins should
    # look like one. The traceback is attached only at DEBUG, because one bad
    # day inside a long window fails every origin whose window contains it,
    # and a traceback per origin would bury the signal.
    logger.warning(
        "%s on %s: %s (origin %d)",
        model_name,
        asset,
        reason,
        origin,
        exc_info=exc if logger.isEnabledFor(logging.DEBUG) else None,
    )


# --------------------------------------------------------------------------
# refit blocks — the unit of distributable work inside a cell
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _BlockTask:
    """One refit block: a refit origin plus the origins reusing its fit.

    Self-contained and picklable so a Phase 2 process- or Slurm-backed
    :class:`~volbench.execute.Executor` can run it unchanged. ``eq=False``
    because it holds arrays (same trap as ``Origin``; see splitter.py).
    """

    model_factory: ModelFactory
    fit_series: NDArray[np.float64]
    series: NDArray[np.float64]
    proxy: NDArray[np.float64]
    origins: tuple[Origin, ...]
    levels: tuple[float, ...]
    asset: str
    model_name: str
    proxy_name: str
    config_hash: str
    seed: int
    recondition: Recondition


def _blocks(origins: Sequence[Origin]) -> list[tuple[Origin, ...]]:
    """Partition origins into maximal runs starting at a refit origin.

    Blocks are independent — each fits from its own training window — so
    running them in any order, in any process, gives identical results.
    """
    blocks: list[list[Origin]] = []
    for origin in origins:
        if origin.refit or not blocks:
            blocks.append([origin])
        else:
            blocks[-1].append(origin)
    return [tuple(block) for block in blocks]


def _forecast_and_score(
    fitted: FittedModel | None,
    horizon: int,
    realized_return: float,
    proxy_var: float,
    task: _BlockTask,
    origin: int,
    origin_failure: str | None,
) -> dict[str, Any]:
    """Score one ``(origin, horizon)``, or explain why it could not be."""
    if origin_failure is not None:
        return _failure_scores(task.levels, origin_failure)
    assert fitted is not None  # an origin without a model always carries a failure
    try:
        dist = fitted.predict(horizon)
    except Exception as exc:
        reason = _exception_reason("predict_error", exc)
        _log_failure(task.model_name, task.asset, origin, reason, exc)
        return _failure_scores(task.levels, reason)
    try:
        return _score(dist, realized_return, proxy_var, task.levels)
    except Exception as exc:
        reason = _exception_reason("score_error", exc)
        _log_failure(task.model_name, task.asset, origin, reason, exc)
        return _failure_scores(task.levels, reason)


def _run_block(task: _BlockTask) -> list[dict[str, Any]]:
    """Fit once, then forecast and score every origin in the block.

    Faults are isolated per origin (M1 report §4.5). An exception from
    ``fit``, ``update``, ``predict`` or scoring becomes the NaN-plus-
    ``missing_reason`` row the results contract already promises for
    unscorable data, instead of taking the whole cell down with it; the
    reason keeps the exception's type and message.

    A failed *scheduled* fit leaves its whole block without a model, so every
    origin in the block gets a failure row naming the origin whose fit failed.
    The alternative — refitting off-schedule at the next origin — would
    silently change the refit cadence that the config hash describes. A
    failed ``update`` costs only its own origin: later origins re-condition
    from the last good state on their own full window.

    Rows without a fitted model carry ``fit_origin = conditioned_through =
    -1``: there is no fit to point at, and those columns are pinned int64.

    Only :class:`Exception` is caught. ``KeyboardInterrupt``, ``SystemExit``
    and friends propagate — a user ending a run is not a bad origin.
    """
    model = task.model_factory()
    fitted: FittedModel | None = None
    fit_origin = -1
    conditioned_through = -1
    block_failure: str | None = None  # set while the block's scheduled fit is missing
    rows: list[dict[str, Any]] = []

    for origin in task.origins:
        origin_failure: str | None
        if origin.refit or (fitted is None and block_failure is None):
            try:
                fitted = model.fit(task.fit_series[origin.train])
            except Exception as exc:
                fitted = None
                fit_origin = conditioned_through = -1
                block_failure = _exception_reason("fit_error", exc, origin=origin.origin)
                _log_failure(task.model_name, task.asset, origin.origin, block_failure, exc)
            else:
                block_failure = None
                fit_origin = conditioned_through = origin.origin
            origin_failure = block_failure
        elif block_failure is not None:
            # The block's scheduled fit failed: nothing to hold or update, and
            # no off-schedule refit (see the docstring).
            origin_failure = block_failure
        elif task.recondition == "daily" and isinstance(fitted, SupportsUpdate):
            try:
                fitted = fitted.update(task.fit_series[origin.train])
            except Exception as exc:
                origin_failure = _exception_reason("update_error", exc, origin=origin.origin)
                _log_failure(task.model_name, task.asset, origin.origin, origin_failure, exc)
            else:
                conditioned_through = origin.origin
                origin_failure = None
        else:
            # Hold: either the model cannot re-condition, or
            # recondition="none" asked for the frozen forecast (the ablation
            # arm). Either way the forecast still rests on the last refit's
            # information set, which is what gets recorded.
            conditioned_through = fit_origin
            origin_failure = None

        for horizon, target_index in enumerate(origin.test, start=1):
            target = int(target_index)
            row: dict[str, Any] = {
                "config_hash": task.config_hash,
                "asset": task.asset,
                "model": task.model_name,
                "origin_index": origin.origin,
                "horizon": horizon,
                "target_index": target,
                "fit_origin": fit_origin,
                "conditioned_through": conditioned_through,
                "refit": bool(origin.refit),
                "seed": task.seed,
                "proxy_name": task.proxy_name,
                "realized_return": float(task.series[target]),
                "proxy_var": float(task.proxy[target]),
            }
            row.update(
                _forecast_and_score(
                    fitted,
                    horizon,
                    row["realized_return"],
                    row["proxy_var"],
                    task,
                    origin.origin,
                    origin_failure,
                )
            )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _calendar_of(values: object) -> pd.Index | None:
    """The index a pandas input carries, or ``None`` for a bare array."""
    index = getattr(values, "index", None)
    return index if isinstance(index, pd.Index) else None


def _first_mismatch(reference: pd.Index, other: pd.Index) -> int:
    """Position of the first entry where two indexes disagree.

    If one index is a prefix of the other, that is the position where the
    shorter one runs out.
    """
    n = min(len(reference), len(other))
    if n:
        equal = np.asarray(reference[:n].to_numpy() == other[:n].to_numpy(), dtype=bool)
        unequal = np.flatnonzero(~equal)
        if unequal.size:
            return int(unequal[0])
    return n


def _require_one_calendar(*inputs: tuple[str, object]) -> None:
    """Refuse inputs that are not on one shared calendar.

    ``run_backtest`` aligns its inputs by position, so the only way to know
    that position ``i`` of the proxy is the same day as position ``i`` of the
    returns is for both to carry that day. Pandas inputs must therefore have
    identical indexes — values and order, not just length — and mixing an
    indexed input with a bare array is refused rather than guessed at.

    Bare arrays across the board are still accepted: they carry no calendar
    to check, so passing them is the caller's explicit statement that the
    alignment is theirs to guarantee. ``_as_series`` still checks lengths.
    """
    calendars = [(name, _calendar_of(values)) for name, values in inputs if values is not None]
    indexed = [(name, index) for name, index in calendars if index is not None]
    if not indexed:
        return
    bare = [name for name, index in calendars if index is None]
    if bare:
        raise ValueError(
            f"{indexed[0][0]} carries a pandas index but {', '.join(bare)} is a bare array: "
            "pass every input on one shared index so their alignment can be checked, or "
            "every input as a bare array to take responsibility for positional alignment "
            "yourself"
        )
    reference_name, reference = indexed[0]
    for name, index in indexed[1:]:
        if reference.equals(index):
            continue
        position = _first_mismatch(reference, index)
        if position < len(reference) and position < len(index):
            where = (
                f"{reference_name} has {reference[position]!s} and {name} has {index[position]!s}"
            )
        elif position < len(reference):
            where = f"{name} has run out (length {len(index)}) while {reference_name} continues"
        else:
            where = f"{reference_name} has run out (length {len(reference)}) while {name} continues"
        raise ValueError(
            f"{name} is not on the same calendar as {reference_name}: first mismatch at "
            f"position {position}, where {where}. run_backtest aligns its inputs by "
            "position, so same-length inputs on different calendars would silently score "
            "every forecast against the wrong day's realization — put "
            f"{reference_name}, {name} and any fit_series on one shared index"
        )


def _as_series(values: object, name: str, expected: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if expected is not None and array.size != expected:
        raise ValueError(f"{name} has length {array.size}, expected {expected} to match series")
    return array


def run_backtest(
    model_factory: ModelFactory,
    series: object,
    proxy: object,
    splitter: RollingOriginSplitter,
    seed: int,
    *,
    asset: str,
    proxy_name: str,
    data_spec: Mapping[str, Any] | None = None,
    fit_series: object | None = None,
    levels: Sequence[float] = DEFAULT_LEVELS,
    executor: Executor | None = None,
    store: ResultsStore | None = None,
    overwrite: bool = False,
    recondition: Recondition = "daily",
) -> pd.DataFrame:
    """Score ``model_factory`` over ``series`` at every rolling origin.

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning a **fresh**, identically-specified
        model. Called once per refit block, plus once up front to read
        ``name`` and ``spec()`` for the config hash. Freshness is a leakage
        requirement, not a style preference: a factory that hands back one
        shared, stateful instance lets that state cross block boundaries, and
        blocks span time. volbench cannot check this — a model constructed
        with a reference to the full series can always cheat — so the
        guarantee this module makes is narrower and exact: *volbench itself
        passes a model nothing but ``fit_series[origin.train]``.*
    series:
        Daily returns. Supplies both the training input and the realized
        target for CRPS, log score, pinball and VaR hits.
    proxy:
        Daily realized-variance proxy aligned index-for-index with ``series``,
        in daily units. QLIKE scores the forecast variance against it. NaNs
        are expected (thin trading days, missing intraday data) and are
        recorded rather than dropped.

        Alignment is positional, so one calendar must govern ``series``,
        ``proxy`` and ``fit_series``. Pass them as pandas objects on one shared
        index and that is *checked*: the indexes must be identical in values
        and order, and a mismatch raises naming the first offending position.
        Mixing an indexed input with a bare array is refused. Bare arrays for
        every input are still accepted — they carry no calendar to check, so
        only lengths are compared and the alignment is the caller's to
        guarantee. The failure this guards against is silent: two same-length
        inputs off by a day would score each forecast against the following
        day's realization, which is leakage no other test here can see.
        Calendars are the data layer's job (docs/design.md,
        ``TimeSeriesFrame``); pass series that came from one.
    splitter:
        The only sanctioned source of train/test indices (CLAUDE.md rule 1).
    seed:
        Recorded on every row and hashed into the config (design.md
        invariant 3).
    asset, proxy_name:
        Provenance, recorded on every row.
    data_spec:
        Extra description of the data (source, span, ...). Content digests of
        ``series``, ``proxy`` and ``fit_series`` are added automatically, so
        the cache cannot serve results computed from different numbers even
        if the caller's label is unchanged.
    fit_series:
        Optional alternative model input, sliced with the *same* splitter
        indices as ``series``. For models whose input is not the return series
        — HAR-RV takes realized variances — pass it here; scoring still uses
        ``series``, keeping the return-distribution convention intact.
    levels:
        Tail levels for pinball loss and VaR. Part of the config hash, since
        they determine the output columns.
    executor:
        Where per-block work runs. Defaults to
        :class:`~volbench.execute.SerialExecutor`.
    store:
        If given, a stored result for this config short-circuits the run
        entirely — no fit, no predict, no scoring — and a fresh run is
        persisted before returning.
    overwrite:
        Recompute and replace a stored result instead of reading it.
    recondition:
        What happens between scheduled refits. ``"daily"`` (default):
        parameters from the last refit, conditional state re-filtered on each
        origin's window via :class:`SupportsUpdate`. ``"none"``: the forecast
        is frozen until the next refit — the ablation arm. Part of the config
        hash whenever it can change a number, i.e. whenever
        ``splitter.refit_every > 1``; at ``refit_every == 1`` every origin
        refits, ``update`` is unreachable, and the run is the same experiment
        under either setting — so it is not recorded and the hash does not
        move.

    Returns
    -------
    A tidy frame, one row per ``(origin, horizon)``, in canonical order.
    ``frame.attrs`` carries ``config_hash`` and ``config``.

    Notes
    -----
    Refit cadence comes from ``splitter``: a fit happens only at origins where
    ``origin.refit`` is true, so the number of ``fit`` calls equals the number
    of refit origins. Under ``recondition="daily"``, models implementing
    :class:`SupportsUpdate` re-condition (without re-estimating) at the
    origins in between; under ``"none"``, and for models without the
    capability, the forecast is held. ``conditioned_through`` records which
    happened on every row, so the two are never confused after the fact.

    Exceptions are isolated per origin. A ``fit``, ``update``, ``predict`` or
    scoring exception produces a row with NaN scores and a ``missing_reason``
    of the form ``<stage>[@origin]: <ExceptionType>: <message>`` — for
    example ``fit_error@499: ValueError: realized-variance series must be
    finite and strictly positive`` — rather than aborting the cell. A failed
    scheduled fit fails every origin in its block (the cadence is never
    changed to work around it); a failed update fails its own origin only.
    Such rows carry ``fit_origin = conditioned_through = -1``. Each failure is
    also logged at WARNING on this module's logger.
    """
    levels_tuple = tuple(float(level) for level in levels)
    if not levels_tuple:
        raise ValueError("levels must not be empty")
    if len(set(levels_tuple)) != len(levels_tuple):
        raise ValueError(f"levels must be distinct, got {levels_tuple}")
    if not all(0.0 < level < 1.0 for level in levels_tuple):
        raise ValueError(f"levels must lie strictly inside (0, 1), got {levels_tuple}")
    if recondition not in ("daily", "none"):
        raise ValueError(f"recondition must be 'daily' or 'none', got {recondition!r}")

    _require_one_calendar(("series", series), ("proxy", proxy), ("fit_series", fit_series))
    series_array = _as_series(series, "series")
    proxy_array = _as_series(proxy, "proxy", expected=series_array.size)
    fit_array = (
        series_array
        if fit_series is None
        else _as_series(fit_series, "fit_series", expected=series_array.size)
    )

    probe = model_factory()
    config = build_config(
        model_name=probe.name,
        model_spec=probe.spec(),
        data_spec={
            "asset": asset,
            "n": int(series_array.size),
            "series_sha256": array_digest(series_array),
            "fit_series_sha256": array_digest(fit_array),
            "proxy": {"name": proxy_name, "sha256": array_digest(proxy_array)},
            **dict(data_spec or {}),
        },
        splitter=splitter,
        seed=seed,
        scoring={"levels": list(levels_tuple)},
        # Recorded exactly when it binds (see the parameter docstring): with
        # one refit per origin there is nothing to re-condition, and a setting
        # that cannot change a number is not part of the experiment's identity.
        protocol={"recondition": recondition} if splitter.refit_every > 1 else None,
    )
    hash_value = config_hash(config)

    if store is not None and not overwrite and store.has(hash_value):
        cached = store.read(hash_value)
        cached.attrs["config_hash"] = hash_value
        cached.attrs["config"] = config
        cached.attrs["cached"] = True
        return cached

    origins = list(splitter.split(series_array.size))
    tasks = [
        _BlockTask(
            model_factory=model_factory,
            fit_series=fit_array,
            series=series_array,
            proxy=proxy_array,
            origins=block,
            levels=levels_tuple,
            asset=asset,
            model_name=probe.name,
            proxy_name=proxy_name,
            config_hash=hash_value,
            seed=int(seed),
            recondition=recondition,
        )
        for block in _blocks(origins)
    ]

    runner = executor if executor is not None else SerialExecutor()
    rows = [row for block_rows in runner.map(_run_block, tasks) for row in block_rows]

    frame = normalize_frame(pd.DataFrame(rows).astype(_result_dtypes(levels_tuple)))
    if store is not None:
        store.write(frame, config=config, overwrite=overwrite)
    frame.attrs["config_hash"] = hash_value
    frame.attrs["config"] = config
    frame.attrs["cached"] = False
    return frame
