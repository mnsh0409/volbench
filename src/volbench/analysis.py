"""Analysis layer: read the completed grid out of the store and describe it.

What this module consumes, and what it must never do
====================================================
It reads **stored result rows** — a :class:`~volbench.results.ResultsStore`
and the run manifest that names which of its fragments belong to one grid —
and nothing else. It never fits a model, never touches a splitter, and never
imports :mod:`volbench.models` or :mod:`volbench.evaluate`. That is a
structural boundary, not a style choice, and it is the same one
:mod:`volbench.econ` draws for the same reason: everything here runs *after*
the fact on a table whose temporal integrity has already been established and
hashed, so nothing in this file can reach a model, a training window, or a
future observation even by mistake. ``tests/test_analysis.py::TestBoundary``
asserts the import graph.

The one import from the package is :mod:`volbench.results`, which is how a
fragment is addressed and read. That module imports no model either.

Why the loss functions are re-implemented here
==============================================
:func:`normal_crps`, :func:`student_t_crps` and :func:`qlike_loss` duplicate
closed forms that :mod:`volbench.dist` and :mod:`volbench.metrics` already
have. The duplication is the point. They exist to *check the stored columns*,
and a check that calls the same function that produced the number under test
verifies only that the function is deterministic. Written out here from the
published forms, they can disagree — which is what makes agreement evidence.

They are not a second opinion about what a loss *is*: the definitions come
from docs/metrics_reference.md and the references named in each docstring,
identically to the ones in :mod:`volbench.dist`. If the two ever disagree,
that is a finding to report, never something to reconcile by editing this
file to match.

The stored predictive law
=========================
The store does not persist a :class:`~volbench.dist.Distribution`; it persists
that distribution's moments and its tail quantiles. Every adapter in the
primary grid emits ``Normal(0, sqrt(v))`` over the next-period return except
``garch11_t``, which emits a location-scale Student-t at the same variance
whenever its own estimator ran. So the predictive law of a stored row is
recoverable from ``forecast_mean``, ``forecast_var`` and *two* tail quantiles,
and :func:`recover_predictive_law` does that recovery — the degrees of freedom
read back out of the quantile ratio rather than assumed, and then required to
reproduce the row's own variance before the law is accepted.

Not exported from the package root. ``volbench/__init__.py`` is a public API
surface and docs/design.md is its record; that document is a read-only mirror
here (CLAUDE.md), so widening the root is a decision for the planning machine
rather than a side effect of adding this module. ``import volbench.analysis``
works today and is how the study drivers reach it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, special, stats  # type: ignore[import-untyped]

from volbench.results import ResultsStore

__all__ = [
    "CELL_KEY",
    "EXCEPTION_STAGES",
    "HAC_BANDWIDTH_RULE",
    "LOSS_ORDER",
    "NU_BOUNDS",
    "SCORE_REASONS",
    "AlignmentCheck",
    "GridFrame",
    "PredictiveLaw",
    "alignment_check",
    "alignment_table",
    "cell_index",
    "fallback_rates",
    "forecast_floor_report",
    "fz0_column",
    "fz0_loss",
    "hac_bandwidth",
    "hac_mean_se",
    "level_tags",
    "load_grid",
    "load_manifest",
    "loss_columns",
    "loss_table",
    "missing_accounting",
    "nonfinite_report",
    "normal_crps",
    "normal_quantile",
    "qlike_loss",
    "qlike_positivity",
    "reason_kind",
    "reason_kinds",
    "recover_predictive_law",
    "student_t_crps",
    "student_t_df_from_quantile_ratio",
    "student_t_scale_from_variance",
    "with_derived_losses",
]

#: What identifies one cell of the grid in the manifest. ``model`` here is the
#: grid's *label* (``garch11_t``), not the adapter's own ``name``
#: (``garch(1,1)-studentst``); the fragments carry the latter, the manifest the
#: former, and :func:`load_grid` joins them so both are readable.
CELL_KEY: Final = ("asset", "model", "horizon", "arm")

#: ``missing_reason`` tokens the *scorer* emits, joined by ``"|"`` when more
#: than one applies. Everything else is an exception token (below).
SCORE_REASONS: Final = (
    "target_nan",
    "proxy_nan",
    "proxy_not_finite",
    "proxy_nonpositive",
    "forecast_var_nonpositive",
    "log_score_undefined",
    "es_undefined",
)

#: Stages whose exception becomes a ``missing_reason`` of the form
#: ``<stage>[@origin]: <ExceptionType>: <message>``.
EXCEPTION_STAGES: Final = ("fit_error", "update_error", "predict_error", "score_error")

_EXCEPTION_RE: Final = re.compile(
    r"^(?P<stage>" + "|".join(EXCEPTION_STAGES) + r")(?:@(?P<origin>-?\d+))?: (?P<type>\w+)"
)
_LEVEL_COLUMN_RE: Final = re.compile(r"^(?P<kind>var|es|pinball|hit)_(?P<tag>[0-9pm]+)$")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


#: A long/tidy frame of stored rows with the manifest's cell identity joined on.
GridFrame = pd.DataFrame


def load_manifest(path: str | Path) -> pd.DataFrame:
    """The manifest's ``cells`` as a frame, one row per cell, in manifest order.

    Read from JSON rather than reconstructed through
    :class:`~volbench.runner.RunManifest`, so that describing a grid never
    imports the machinery that could run one.
    """
    payload: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path}: manifest has no 'cells' list")
    frame = pd.DataFrame(cells)
    missing = [c for c in (*CELL_KEY, "config_hash", "status") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: manifest cells are missing columns {missing}")
    return frame


def load_grid(
    store: ResultsStore,
    manifest: pd.DataFrame,
    *,
    require_all: bool = True,
) -> GridFrame:
    """Every stored row of the manifest's cells, with the cell identity joined on.

    The result is the long/tidy frame the rest of this module consumes: one row
    per ``(cell, origin, horizon)``, carrying the fragment's own columns plus
    ``model_label``, ``arm`` and ``lane`` from the manifest. The fragment's
    ``model`` column (the adapter's name) is left untouched, so a reader can
    see both spellings and neither is silently substituted for the other.

    ``require_all`` refuses a partial read: an analysis run over 140 of 143
    cells that does not say so is how a missing column becomes a published
    number. Set it False deliberately, never to get past a surprise.
    """
    usable = manifest.loc[manifest["config_hash"].notna()].copy()
    present = usable["config_hash"].map(store.has)
    if require_all and not bool(present.all()):
        absent = usable.loc[~present, list(CELL_KEY)].to_dict("records")
        raise ValueError(
            f"{len(absent)} of {len(manifest)} manifest cells have no fragment in {store!r}: "
            f"{absent[:5]}"
        )
    usable = usable.loc[present]
    if usable.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for row in usable.itertuples(index=False):
        frame = store.read(str(row.config_hash))
        frame = frame.assign(
            model_label=str(row.model),
            arm=str(row.arm),
            lane=str(getattr(row, "lane", "")),
        )
        parts.append(frame)
    grid = pd.concat(parts, ignore_index=True)
    return grid.sort_values(
        ["asset", "model_label", "horizon", "arm", "origin_index"], kind="stable"
    ).reset_index(drop=True)


def cell_index(grid: GridFrame) -> pd.DataFrame:
    """One row per cell: identity, config hash, and the row count behind it."""
    keys = ["asset", "model_label", "horizon", "arm"]
    out = grid.groupby(keys, observed=True, sort=True).agg(
        config_hash=("config_hash", "first"),
        model_name=("model", "first"),
        proxy_name=("proxy_name", "first"),
        n_rows=("origin_index", "size"),
    )
    return out.reset_index()


# --------------------------------------------------------------------------
# column vocabulary
# --------------------------------------------------------------------------


def level_tags(frame: pd.DataFrame) -> list[str]:
    """The tail-level tags the frame actually carries, e.g. ``["0p01", ...]``.

    Read off the columns rather than assumed from
    :data:`volbench.evaluate.DEFAULT_LEVELS`: which levels a run evaluated is a
    property of that run's config, and the analysis must report the levels in
    front of it, not the ones the library currently defaults to.
    """
    tags: list[str] = []
    for column in frame.columns:
        match = _LEVEL_COLUMN_RE.match(str(column))
        if match and match.group("kind") == "var":
            tags.append(match.group("tag"))
    return sorted(tags, key=_tag_to_level)


def _tag_to_level(tag: str) -> float:
    return float(tag.replace("m", "-").replace("p", "."))


def loss_columns(frame: pd.DataFrame) -> list[str]:
    """Every per-row loss column present, in a stable order.

    ``var_*`` and ``es_*`` are excluded: they describe the *forecast* and are
    written even on rows whose target is unscorable, so counting a NaN there
    as a missing loss would misreport what happened. ``hit_*`` is an indicator,
    not a loss, and is likewise left out.
    """
    present = [c for c in ("crps", "log_score", "qlike") if c in frame.columns]
    for tag in level_tags(frame):
        column = f"pinball_{tag}"
        if column in frame.columns:
            present.append(column)
    return present


# --------------------------------------------------------------------------
# missing-row accounting
# --------------------------------------------------------------------------


def reason_kinds(reason: str) -> tuple[str, ...]:
    """The kinds a ``missing_reason`` string names. ``()`` for a scored row.

    Score-side reasons are a ``"|"``-joined sorted set of the small vocabulary
    in :data:`SCORE_REASONS`; an exception-side reason is one token of the form
    ``<stage>[@origin]: <Type>: <message>`` and is reduced to ``<stage>/<Type>``
    so that a message quoting a varying number never fragments a count.
    """
    text = (reason or "").strip()
    if not text:
        return ()
    match = _EXCEPTION_RE.match(text)
    if match:
        return (f"{match.group('stage')}/{match.group('type')}",)
    parts = tuple(part for part in text.split("|") if part)
    return parts or (text,)


def reason_kind(reason: str) -> str:
    """A single label for a ``missing_reason``; joins multiple kinds with ``+``."""
    kinds = reason_kinds(reason)
    return "+".join(kinds) if kinds else ""


def missing_accounting(grid: GridFrame) -> pd.DataFrame:
    """Per asset x model: origins, scored rows, and NaN rows by ``missing_reason``.

    ``scored`` is rows whose ``missing_reason`` is empty — the strict reading,
    under which a row that lost only QLIKE (a non-positive proxy) counts as
    unscored. ``crps_scored`` and ``qlike_scored`` are the per-metric counts,
    which differ from it and from each other; reporting only one of the three
    is how a table comes to mean something other than what it says.
    """
    frame = grid.copy()
    frame["reason"] = frame["missing_reason"].fillna("").astype(str)
    keys = ["asset", "model_label"]
    rows: list[dict[str, Any]] = []
    for (asset, model), group in frame.groupby(keys, observed=True, sort=True):
        counts: dict[str, int] = {}
        for reason in group.loc[group["reason"] != "", "reason"]:
            for kind in reason_kinds(reason):
                counts[kind] = counts.get(kind, 0) + 1
        rows.append(
            {
                "asset": asset,
                "model": model,
                "origins": int(group["origin_index"].nunique()),
                "rows": len(group),
                "scored": int((group["reason"] == "").sum()),
                "missing": int((group["reason"] != "").sum()),
                "crps_scored": int(group["crps"].notna().sum()),
                "qlike_scored": int(group["qlike"].notna().sum()),
                "reasons": dict(sorted(counts.items())),
            }
        )
    return pd.DataFrame(rows)


def fallback_rates(grid: GridFrame) -> pd.DataFrame:
    """Per asset x model: scheduled fits, fallbacks, and the rate as ``k/n``.

    ``fit_status`` describes the *scheduled fit* a row rests on, not the row
    (docs/P3_ANALYSIS_ASSUMPTIONS.md §5), so the per-fit view is the first
    status of each ``fit_origin`` group and counting rows would multiply every
    fit by the length of its refit block.

    ``n_fits`` is ``<NA>`` — never ``0`` — for a model whose adapter
    implements no ``fit_diagnostics``, and the ``instrumented`` column says
    which. Ten of the thirteen configs are in that state, and a table printing
    ``0`` for them would say "never failed" where the truth is "never
    measured" (docs/P3_INSTRUMENTATION_GAP.md).
    """
    rows: list[dict[str, Any]] = []
    for (asset, model), group in grid.groupby(["asset", "model_label"], observed=True, sort=True):
        scheduled = group.loc[group["fit_origin"] >= 0]
        status = scheduled.groupby("fit_origin")["fit_status"].first().astype(str)
        status = status[status != ""]
        n_fits = int(status.size)
        fallback = int(status.str.startswith("fallback=").sum())
        rows.append(
            {
                "asset": asset,
                "model": model,
                "instrumented": n_fits > 0,
                "n_fits": n_fits if n_fits else pd.NA,
                "n_fallback": fallback if n_fits else pd.NA,
                "rate": (fallback / n_fits) if n_fits else pd.NA,
                "n_nonconverged": (
                    int((~status.str.startswith("ok")).sum()) if n_fits else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# finiteness
# --------------------------------------------------------------------------


def nonfinite_report(grid: GridFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Per asset x model x column: how many values are NaN, +inf or -inf.

    NaN is separated from the infinities on purpose. A NaN loss is the
    contract's own way of saying "unscorable, and here is why"; an infinity is
    not — nothing in the scorer produces one, so a non-zero ``pos_inf`` or
    ``neg_inf`` is a defect and a NaN may not be.
    """
    targets = list(columns) if columns is not None else loss_columns(grid)
    rows: list[dict[str, Any]] = []
    for (asset, model), group in grid.groupby(["asset", "model_label"], observed=True, sort=True):
        for column in targets:
            values = group[column].to_numpy(dtype=np.float64)
            nan = int(np.isnan(values).sum())
            pos = int(np.isposinf(values).sum())
            neg = int(np.isneginf(values).sum())
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "column": column,
                    "n": int(values.size),
                    "nan": nan,
                    "pos_inf": pos,
                    "neg_inf": neg,
                    "nonfinite_not_nan": pos + neg,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# QLIKE positivity
# --------------------------------------------------------------------------


def qlike_positivity(grid: GridFrame) -> pd.DataFrame:
    """Per asset: what the realized target and the forecasts do near zero.

    QLIKE is ``p/f - log(p/f) - 1``: it needs a strictly positive proxy ``p``
    and a strictly positive forecast ``f``, and it diverges as either
    approaches zero. This tabulates how close each asset actually gets.

    The proxy is a property of the asset, not of the model, so its columns are
    identical across that asset's cells; ``proxy_distinct_series`` is 1 when
    that holds and is worth reading, because a value above 1 would mean two
    cells of one asset were scored against different targets.
    """
    rows: list[dict[str, Any]] = []
    for asset, group in grid.groupby("asset", observed=True, sort=True):
        proxy_by_target = group.groupby("target_index", observed=True)["proxy_var"]
        distinct = int(proxy_by_target.nunique(dropna=False).max())
        proxy = group.drop_duplicates("target_index").set_index("target_index")["proxy_var"]
        values = proxy.to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0.0]
        forecast = group["forecast_var"].to_numpy(dtype=np.float64)
        f_finite = forecast[np.isfinite(forecast)]
        f_positive = f_finite[f_finite > 0.0]
        rows.append(
            {
                "asset": asset,
                "n_targets": int(values.size),
                "proxy_distinct_series": distinct,
                "proxy_min": float(positive.min()) if positive.size else math.nan,
                "proxy_min_incl_zero": float(finite.min()) if finite.size else math.nan,
                "proxy_nan": int(np.isnan(values).sum()),
                "proxy_zero": int((finite == 0.0).sum()),
                "proxy_negative": int((finite < 0.0).sum()),
                "forecast_min": float(f_positive.min()) if f_positive.size else math.nan,
                "forecast_nan": int(np.isnan(forecast).sum()),
                "forecast_nonpositive": int((f_finite <= 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def forecast_floor_report(grid: GridFrame) -> pd.DataFrame:
    """Per asset x model: the smallest positive variance forecast, and the
    largest ratio ``proxy/forecast`` a QLIKE term was actually evaluated at.

    The ratio is the quantity QLIKE's sensitivity lives in — the loss is
    ``r - log r - 1`` — so a column whose maximum ``r`` is enormous is a column
    whose mean is decided by a handful of days, whatever its forecasts look
    like elsewhere.
    """
    rows: list[dict[str, Any]] = []
    for (asset, model), group in grid.groupby(["asset", "model_label"], observed=True, sort=True):
        f = group["forecast_var"].to_numpy(dtype=np.float64)
        p = group["proxy_var"].to_numpy(dtype=np.float64)
        usable = np.isfinite(f) & (f > 0.0) & np.isfinite(p) & (p > 0.0)
        ratio = p[usable] / f[usable]
        positive = f[np.isfinite(f) & (f > 0.0)]
        rows.append(
            {
                "asset": asset,
                "model": model,
                "forecast_min": float(positive.min()) if positive.size else math.nan,
                "forecast_max": float(positive.max()) if positive.size else math.nan,
                "n_qlike_terms": int(usable.sum()),
                "ratio_max": float(ratio.max()) if ratio.size else math.nan,
                "ratio_min": float(ratio.min()) if ratio.size else math.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# loss tables: FZ0, HAC standard errors, per-asset means
# --------------------------------------------------------------------------


def fz0_loss(realized_return: float, var: float, es: float, level: float) -> float:
    """Elementwise FZ0 joint (VaR, ES) loss — Patton, Ziegel & Chen (2019) eq. 6.

    ``L(Y, v, e; a) = -(1/(a e)) 1{Y <= v} (v - Y) + v/e + log(-e) - 1``, with
    ``Y`` the realized return, ``v`` the ``a``-quantile of the predictive
    return law and ``e`` the mean below it. Written out here from the paper
    rather than imported from :mod:`volbench.backtests`, for the reason the
    module docstring gives: a recomputation that calls the implementation
    under test checks only that it is deterministic.

    Its domain is the paper's: ``e < 0`` (the log term) and ``e <= v``
    (consistency, their footnote 1). Both raise rather than returning a number,
    because the usual way to violate them is a sign-convention bug. The whole
    grid was checked against both before this was used (J1 §2).
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    if not es < 0.0:
        raise ValueError(
            "FZ0 needs an ES forecast strictly below zero (return-side sign convention)"
        )
    if es > var:
        raise ValueError("FZ0 needs ES <= VaR (the mean below a quantile cannot exceed it)")
    shortfall = (var - realized_return) if realized_return <= var else 0.0
    return -shortfall / (level * es) + var / es + math.log(-es) - 1.0


def fz0_column(frame: pd.DataFrame, level: float) -> NDArray[np.float64]:
    """FZ0 at ``level`` for every row, vectorized, NaN where any input is not finite.

    The scalar :func:`fz0_loss` is the definition and this is the same
    arithmetic over arrays; ``tests/test_analysis.py`` pins them equal, so the
    fast path cannot drift from the one written out from the paper.
    """
    tag = _level_tag(level)
    y = frame["realized_return"].to_numpy(dtype=np.float64)
    v = frame[f"var_{tag}"].to_numpy(dtype=np.float64)
    e = frame[f"es_{tag}"].to_numpy(dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(v) & np.isfinite(e)
    pair = np.isfinite(v) & np.isfinite(e)
    if np.any(e[np.isfinite(e)] >= 0.0) or np.any(e[pair] > v[pair]):
        raise ValueError(f"FZ0 domain violated at level {level}: some ES >= 0 or ES > VaR")
    out = np.full(y.shape, np.nan)
    yy, vv, ee = y[finite], v[finite], e[finite]
    shortfall = np.where(yy <= vv, vv - yy, 0.0)
    out[finite] = -shortfall / (level * ee) + vv / ee + np.log(-ee) - 1.0
    return out


def with_derived_losses(grid: GridFrame, levels: Sequence[float] | None = None) -> GridFrame:
    """``grid`` plus the losses that are recomputable but not stored.

    Adds ``fz0_<tag>`` at each level, ``fz0_avg`` and ``pinball_avg``. The two
    averages are the unweighted mean **across levels of one row**, which is a
    different object from the mean across rows and is named so it cannot be
    mistaken for one.
    """
    tags = level_tags(grid) if levels is None else [_level_tag(level) for level in levels]
    frame = grid.copy()
    for tag in tags:
        frame[f"fz0_{tag}"] = fz0_column(frame, _tag_to_level(tag))
    frame["fz0_avg"] = frame[[f"fz0_{tag}" for tag in tags]].mean(axis=1)
    frame["pinball_avg"] = frame[[f"pinball_{tag}" for tag in tags]].mean(axis=1)
    return frame


#: How the Bartlett kernel's truncation lag is chosen when none is given:
#: ``floor(4 (n/100)^(2/9))``, the deterministic rule of thumb that follows
#: Newey & West (1994) and is what most software uses as its default. Stated
#: rather than left implicit, because a HAC standard error is a statement about
#: a bandwidth as much as about the data.
HAC_BANDWIDTH_RULE: Final = "floor(4 * (n / 100) ** (2 / 9)), Bartlett kernel"


def hac_bandwidth(n: int) -> int:
    """The truncation lag :data:`HAC_BANDWIDTH_RULE` gives for ``n`` observations."""
    if n < 2:
        return 0
    return int(4.0 * (n / 100.0) ** (2.0 / 9.0) // 1.0)


def hac_mean_se(values: NDArray[np.float64], bandwidth: int | None = None) -> dict[str, Any]:
    """Newey-West standard error of the sample mean of ``values``.

    ``sqrt(omega / n)`` with ``omega = gamma_0 + 2 sum_{j=1}^{L} (1 - j/(L+1))
    gamma_j``, the Bartlett-kernel long-run variance. The kernel's weights make
    ``omega`` non-negative by construction, so this cannot return a NaN from a
    negative variance the way a truncated (unweighted) estimator can.

    **Non-finite values are dropped and the remainder treated as adjacent.**
    A loss series with an unscorable day in it has a hole, and a HAC estimator
    has no way to represent one: lag ``j`` after the hole spans more than ``j``
    days. The alternative — treating the hole as a zero deviation — would bias
    the autocovariances toward zero, which is worse and silent. ``n_dropped``
    is returned so the size of the approximation is visible next to the number
    it affects.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    x = array[finite]
    n = int(x.size)
    result: dict[str, Any] = {
        "n": n,
        "n_dropped": int(array.size - n),
        "mean": float(x.mean()) if n else math.nan,
        "bandwidth": 0,
        "se": math.nan,
        "se_iid": math.nan,
    }
    if n < 2:
        return result
    lag = hac_bandwidth(n) if bandwidth is None else int(bandwidth)
    lag = max(0, min(lag, n - 1))
    centered = x - x.mean()
    gamma0 = float(centered @ centered) / n
    omega = gamma0
    for j in range(1, lag + 1):
        gamma_j = float(centered[j:] @ centered[:-j]) / n
        omega += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    result["bandwidth"] = lag
    result["se"] = math.sqrt(max(omega, 0.0) / n)
    result["se_iid"] = math.sqrt(gamma0 / (n - 1))
    return result


#: The losses a per-asset table reports, in the order it reports them.
LOSS_ORDER: Final = (
    "crps",
    "log_score",
    "pinball_0p01",
    "pinball_0p025",
    "pinball_0p05",
    "pinball_avg",
    "qlike",
    "fz0_0p01",
    "fz0_0p025",
    "fz0_0p05",
    "fz0_avg",
)


def loss_table(grid: GridFrame, asset: str, *, losses: Sequence[str] = LOSS_ORDER) -> pd.DataFrame:
    """One asset's per-model mean loss, HAC standard error and ``n``.

    One row per (model, loss). ``grid`` must already carry the derived columns
    (:func:`with_derived_losses`). Rows are ordered by origin before the HAC
    estimator sees them, because a long-run variance computed on a frame in
    some other order is a number about that order.

    Nothing is aggregated across assets, and this function cannot be made to:
    equities score against an overnight-plus-range variance target and crypto
    against 5-minute realized variance, so a mean over the eleven would be a
    mean over two different units.
    """
    cell = grid.loc[grid["asset"] == asset]
    if cell.empty:
        raise ValueError(f"no rows for asset {asset!r}")
    rows: list[dict[str, Any]] = []
    for model, group in cell.groupby("model_label", observed=True, sort=True):
        ordered = group.sort_values("origin_index", kind="stable")
        origins = int(ordered["origin_index"].nunique())
        for loss in losses:
            summary = hac_mean_se(ordered[loss].to_numpy(dtype=np.float64))
            rows.append(
                {"asset": asset, "model": model, "loss": loss, "origins": origins, **summary}
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# independent loss recomputation (see the module docstring)
# --------------------------------------------------------------------------


_INV_SQRT_PI: Final = 1.0 / math.sqrt(math.pi)


def qlike_loss(forecast_var: float, proxy_var: float) -> float:
    """``r - log r - 1`` with ``r = proxy/forecast`` (docs/metrics_reference.md)."""
    if not (forecast_var > 0.0 and proxy_var > 0.0):
        raise ValueError("QLIKE needs strictly positive forecast and proxy variances")
    r = proxy_var / forecast_var
    return r - math.log(r) - 1.0


def normal_quantile(mu: float, sigma: float, level: float) -> float:
    """``mu + sigma * Phi^{-1}(level)``."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    return float(mu + sigma * stats.norm.ppf(level))


def normal_crps(mu: float, sigma: float, y: float) -> float:
    """CRPS of ``N(mu, sigma^2)`` at ``y`` — Gneiting & Raftery (2007), eq. 21.

    ``sigma * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]`` with
    ``z = (y - mu) / sigma``. Written out from the reference rather than
    imported, for the reason the module docstring gives.
    """
    if not sigma > 0.0:
        raise ValueError("sigma must be strictly positive")
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


def student_t_crps(loc: float, scale: float, df: float, y: float) -> float:
    """CRPS of the location-scale Student-t — Jordan, Krüger & Lerch (2019), App. A.

    ``scale * [ z(2F(z) - 1) + 2 f(z)(df + z^2)/(df - 1)
                - 2 sqrt(df)/(df - 1) B(1/2, df - 1/2) / B(1/2, df/2)^2 ]``.
    """
    if not scale > 0.0:
        raise ValueError("scale must be strictly positive")
    if not df > 1.0:
        raise ValueError("df must exceed 1 for a finite Student-t CRPS")
    z = (y - loc) / scale
    cdf = float(stats.t.cdf(z, df=df))
    pdf = float(stats.t.pdf(z, df=df))
    beta_ratio = math.exp(
        float(special.betaln(0.5, df - 0.5)) - 2.0 * float(special.betaln(0.5, df / 2.0))
    )
    constant = 2.0 * math.sqrt(df) / (df - 1.0) * beta_ratio
    return scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf * (df + z * z) / (df - 1.0) - constant)


def student_t_scale_from_variance(variance: float, df: float) -> float:
    """``sqrt(variance (df - 2) / df)`` — the scale that reproduces ``variance``."""
    if not df > 2.0:
        raise ValueError("a finite Student-t variance needs df > 2")
    return math.sqrt(variance * (df - 2.0) / df)


#: Bracket for the ``nu`` recovery. D-032 bounds the GARCH-t optimizer to
#: ``(2.1, 50)``; quoted as a literal rather than imported, because importing
#: ``garch.NU_BOUNDS`` would mean importing the model package (module docstring).
NU_BOUNDS: Final = (2.1, 50.0)


def student_t_df_from_quantile_ratio(
    q_low: float,
    q_high: float,
    level_low: float,
    level_high: float,
    *,
    bounds: tuple[float, float] = NU_BOUNDS,
    edge_tolerance: float = 1e-12,
) -> float | None:
    """Degrees of freedom implied by the *ratio* of two stored tail quantiles.

    A stored row records the predictive variance and its tail quantiles but not
    ``nu``, so ``nu`` has to be inverted out of what is there. The ratio is
    what is invertible. Inverting a single quantile at a fixed *variance* does
    not work: ``nu -> sqrt(v(nu-2)/nu) t_nu(level)`` is **not** monotone — at
    ``level = 0.01`` it turns around near ``nu = 4`` — so a bracketed root
    find on it either fails or returns whichever of two roots it lands on.

    The ratio ``t_nu(level_low) / t_nu(level_high)`` has the scale divided out
    of it, is free of the variance entirely, and is strictly decreasing in
    ``nu`` (2.31 at ``nu = 2.1`` down to 1.41 as ``nu -> inf``, for
    ``0.01/0.05``). That is what makes the recovery well posed.

    ``edge_tolerance`` accepts a ratio that sits on a bound to within
    floating-point noise. It is not slack: D-032 *bounds* the GARCH-t
    optimizer at ``nu = 50``, so ``nu`` estimated at its bound is a normal
    outcome, and the ratio a stored row then carries misses the bound's own
    ratio by ~6e-15 — enough for a sign test to reject a root that is really
    there. Without it those origins read as unrecovered and would be scored as
    Gaussian, which is wrong by ~1% of the CRPS.

    Returns ``None`` when the observed ratio lies outside what ``bounds``
    can produce by more than that — the honest answer for a row whose law is
    not a Student-t at all, which is exactly what a ``garch11_t`` origin that
    fell back to the Gaussian EWMA looks like.
    """
    if not (math.isfinite(q_low) and math.isfinite(q_high)) or q_high == 0.0:
        return None
    if not (0.0 < level_low < level_high < 1.0):
        raise ValueError("require 0 < level_low < level_high < 1")
    observed = q_low / q_high

    def ratio(df: float) -> float:
        return float(stats.t.ppf(level_low, df=df) / stats.t.ppf(level_high, df=df))

    def residual(df: float) -> float:
        return ratio(df) - observed

    lo, hi = bounds
    r_lo, r_hi = residual(lo), residual(hi)
    if not (math.isfinite(r_lo) and math.isfinite(r_hi)):
        return None
    if r_lo == 0.0:
        return lo
    if r_hi == 0.0:
        return hi
    if r_lo * r_hi > 0.0:
        for bound, residual_at in ((lo, r_lo), (hi, r_hi)):
            if abs(residual_at) <= edge_tolerance * max(1.0, abs(ratio(bound))):
                return bound
        return None
    return float(optimize.brentq(residual, lo, hi, xtol=1e-12, rtol=8.9e-16))


@dataclass(frozen=True)
class PredictiveLaw:
    """The predictive law a stored row implies, recovered from its own columns."""

    family: Literal["normal", "student_t"]
    loc: float
    #: Standard deviation for ``normal``; the t's *scale* parameter otherwise.
    scale: float
    #: ``nan`` for ``normal``.
    df: float

    def crps(self, y: float) -> float:
        if self.family == "normal":
            return normal_crps(self.loc, self.scale, y)
        return student_t_crps(self.loc, self.scale, self.df, y)

    def quantile(self, level: float) -> float:
        if self.family == "normal":
            return normal_quantile(self.loc, self.scale, level)
        return float(self.loc + self.scale * stats.t.ppf(level, df=self.df))


def recover_predictive_law(
    row: Mapping[str, Any],
    *,
    levels: tuple[float, float] = (0.01, 0.05),
    tolerance: float = 1e-9,
) -> PredictiveLaw | None:
    """The predictive law behind one stored row, or ``None`` if unrecoverable.

    Nothing is assumed about which adapter wrote the row: the row's own two
    tail quantiles decide, so a ``garch11_t`` origin that fell back to a
    Gaussian estimator is recognised as Gaussian and no cell is labelled by its
    config's *intent*.

    Gaussian first — accepted when ``Normal(forecast_mean, sqrt(forecast_var))``
    reproduces **both** stored quantiles to ``tolerance`` relative. Otherwise
    ``nu`` comes out of the quantile ratio (:func:`student_t_df_from_quantile_ratio`)
    and the recovered law is accepted only if the variance it implies also
    matches ``forecast_var``. That second check is what makes the recovery
    falsifiable: a law reconstructed from one quantile can always be made to
    fit that quantile, and would then hide exactly the disagreement the
    alignment canary exists to find.
    """
    mean = float(row["forecast_mean"])
    variance = float(row["forecast_var"])
    low, high = levels
    q_low = float(row[f"var_{_level_tag(low)}"])
    q_high = float(row[f"var_{_level_tag(high)}"])
    if not (math.isfinite(mean) and math.isfinite(variance) and variance > 0.0):
        return None
    if not (math.isfinite(q_low) and math.isfinite(q_high)):
        return None

    sigma = math.sqrt(variance)
    scale_of = max(1.0, abs(q_low), abs(q_high))
    gaussian_low = normal_quantile(mean, sigma, low)
    gaussian_high = normal_quantile(mean, sigma, high)
    if (
        abs(gaussian_low - q_low) <= tolerance * scale_of
        and abs(gaussian_high - q_high) <= tolerance * scale_of
    ):
        return PredictiveLaw("normal", mean, sigma, math.nan)

    if abs(mean) > tolerance * scale_of:
        return None  # the t recovery assumes the zero location every adapter emits
    df = student_t_df_from_quantile_ratio(q_low, q_high, low, high)
    if df is None:
        return None
    scale = student_t_scale_from_variance(variance, df)
    implied = scale * float(stats.t.ppf(low, df=df))
    if abs(implied - q_low) > 1e-6 * scale_of:
        return None  # a t at this nu does not carry this variance and this quantile
    return PredictiveLaw("student_t", mean, scale, df)


def _level_tag(level: float) -> str:
    """``0.025 -> "0p025"`` — the evaluator's own column-suffix spelling."""
    return f"{level:.10g}".replace(".", "p").replace("-", "m")


# --------------------------------------------------------------------------
# alignment canary
# --------------------------------------------------------------------------


def _abs_error(stored: float, recomputed: float) -> float:
    """``|stored - recomputed|``, with two NaNs counting as agreement.

    A NaN on both sides is the contract agreeing that a row is unscorable. A
    NaN on one side only is a disagreement, and returning NaN there would let
    it pass a ``max() < tol`` gate — so it returns infinity instead.
    """
    if math.isnan(stored) and math.isnan(recomputed):
        return 0.0
    if math.isnan(stored) or math.isnan(recomputed):
        return math.inf
    return abs(stored - recomputed)


@dataclass(frozen=True)
class AlignmentCheck:
    """One row's independent re-derivation, beside what the store recorded."""

    asset: str
    model: str
    origin_index: int
    horizon: int
    target_index: int
    family: str
    df: float
    stored_crps: float
    recomputed_crps: float
    stored_qlike: float
    recomputed_qlike: float
    stored_return: float
    stored_proxy: float
    #: ``returns`` and ``proxy`` read back out of the series the study was run
    #: on, at ``target_index``. NaN when no series was supplied — which is what
    #: ``series_checked`` distinguishes from a series that is NaN there.
    series_return: float
    series_proxy: float
    #: Whether a comparison series was supplied at all. Without it the
    #: alignment half of the check did not run, and its errors are ``nan``
    #: (not checked) rather than ``inf`` (checked and disagreed) — the two must
    #: not look alike to a ``max() < tol`` gate.
    series_checked: bool = False

    @property
    def crps_abs_error(self) -> float:
        return _abs_error(self.stored_crps, self.recomputed_crps)

    @property
    def qlike_abs_error(self) -> float:
        return _abs_error(self.stored_qlike, self.recomputed_qlike)

    @property
    def return_abs_error(self) -> float:
        if not self.series_checked:
            return math.nan
        return _abs_error(self.stored_return, self.series_return)

    @property
    def proxy_abs_error(self) -> float:
        if not self.series_checked:
            return math.nan
        return _abs_error(self.stored_proxy, self.series_proxy)

    @property
    def target_index_is_consistent(self) -> bool:
        """``target_index == origin_index + horizon`` — the splitter's promise."""
        return self.target_index == self.origin_index + self.horizon


def alignment_check(
    row: Mapping[str, Any],
    *,
    levels: tuple[float, float] = (0.01, 0.05),
    returns: NDArray[np.float64] | None = None,
    proxy: NDArray[np.float64] | None = None,
) -> AlignmentCheck:
    """Re-derive one stored row's CRPS and QLIKE, and re-read its target.

    Two independent things happen here, and they catch different failures:

    - the losses are recomputed from the row's own recovered predictive law and
      its own stored target, which catches a scoring bug but *cannot* catch a
      misalignment — a loss computed against the wrong day's realization is
      still self-consistent;
    - ``returns`` and ``proxy``, when given, are indexed at ``target_index``
      and compared against the row's ``realized_return`` and ``proxy_var``,
      which is what catches an off-by-one between forecast and realization.

    Pass the series the study was actually run on, on the study's own calendar.
    The primary grid's driver trims one leading bar, so position ``j`` of a
    results frame is position ``j + 1`` of the untrimmed raw frame — the
    arithmetic that has already produced one false leak report.
    """
    law = recover_predictive_law(row, levels=levels)
    realized = float(row["realized_return"])
    proxy_var = float(row["proxy_var"])
    forecast_var = float(row["forecast_var"])
    target = int(row["target_index"])

    if law is None:
        recomputed_crps = math.nan
        family, df = "unrecovered", math.nan
    else:
        family, df = law.family, law.df
        recomputed_crps = law.crps(realized) if math.isfinite(realized) else math.nan

    if forecast_var > 0.0 and math.isfinite(proxy_var) and proxy_var > 0.0:
        recomputed_qlike = qlike_loss(forecast_var, proxy_var)
    else:
        recomputed_qlike = math.nan

    return AlignmentCheck(
        asset=str(row["asset"]),
        model=str(row.get("model_label", row["model"])),
        origin_index=int(row["origin_index"]),
        horizon=int(row["horizon"]),
        target_index=target,
        family=family,
        df=df,
        stored_crps=float(row["crps"]),
        recomputed_crps=recomputed_crps,
        stored_qlike=float(row["qlike"]),
        recomputed_qlike=recomputed_qlike,
        stored_return=realized,
        stored_proxy=proxy_var,
        series_return=float(returns[target]) if returns is not None else math.nan,
        series_proxy=float(proxy[target]) if proxy is not None else math.nan,
        series_checked=returns is not None or proxy is not None,
    )


def alignment_table(checks: Iterable[AlignmentCheck]) -> pd.DataFrame:
    """The checks as a frame, with the absolute errors worked out."""
    rows: list[dict[str, Any]] = []
    for check in checks:
        rows.append(
            {
                "asset": check.asset,
                "model": check.model,
                "origin_index": check.origin_index,
                "target_index": check.target_index,
                "target_ok": check.target_index_is_consistent,
                "family": check.family,
                "df": check.df,
                "stored_crps": check.stored_crps,
                "recomputed_crps": check.recomputed_crps,
                "crps_abs_err": check.crps_abs_error,
                "stored_qlike": check.stored_qlike,
                "recomputed_qlike": check.recomputed_qlike,
                "qlike_abs_err": check.qlike_abs_error,
                "return_abs_err": check.return_abs_error,
                "proxy_abs_err": check.proxy_abs_error,
            }
        )
    return pd.DataFrame(rows)
