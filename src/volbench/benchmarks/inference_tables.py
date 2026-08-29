#!/usr/bin/env python
"""P3 comparison inference off the completed primary grid.

Diebold-Mariano matrices, model confidence sets, VaR/ES backtests, the
volatility-targeting backtest with a bootstrap on its Sharpe differences,
rank-only cross-asset summaries, crisis sub-samples on the pre-registered
windows — and the checks the J3 brief asked for before any of them:
the boundary-pinned GARCH fits against EWMA, QLIKE with and without the five
near-zero target days, and the AutoARIMA optimiser rate carried beside the
GARCH fallback rates.

**Reported, not interpreted.** Every function here computes a number or
arranges numbers in a table. Nothing ranks a model as better, calls a result
large or small, or draws a conclusion; where a table is ordered it says by
what. The results review happens elsewhere.

**Read-only over the store, and structurally unable to fit.** The numerics
come from :mod:`volbench.inference`, :mod:`volbench.backtests`,
:mod:`volbench.econ` and :mod:`volbench.analysis`, none of which imports the
model package; this module is held to the same boundary by
``tests/test_inference_tables.py::TestBoundary``. The one thing it rebuilds
is the study's *calendar* — through the panel and the driver's own leading
trim — and it refuses to proceed unless the rebuilt return series reproduces
every stored ``realized_return`` at its ``target_index`` to the bit, so a date
attached to a row is the date the row was scored on.

**Determinism.** One master seed; every bootstrap seed is derived from it and
the run's identity and written into ``docs/P3_ANALYSIS_manifest.json`` with
the block lengths, the bandwidth summaries, the config hashes, the data
digests, the package versions, the thread pins and the git SHA.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run python -m volbench.benchmarks.inference_tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from volbench import analysis, backtests, econ, inference
from volbench.benchmarks.loss_tables import LOSS_HEADINGS, frame_markdown
from volbench.data.crisis import CALM_TAG, CRISIS_WINDOWS, PENDING_WINDOWS, CrisisWindow, tag_dates
from volbench.data.panel import build_panel
from volbench.data.proxies import log_returns
from volbench.results import ResultsStore

__all__ = [
    "ALPHAS",
    "BASELINE",
    "COST_BPS",
    "HAC_LADDER",
    "LOSSES",
    "MASTER_SEED",
    "NEAR_ZERO_TARGET",
    "N_BOOT",
    "QLIKE_EX",
    "STATISTICS",
    "WIDE_GFC",
    "Inputs",
    "attach_calendar",
    "block_diagnostics",
    "dm_long",
    "dm_matrix_markdown",
    "dm_significance_changes",
    "dm_summary",
    "drop_model",
    "garch_recursion_by_block",
    "kendall_between",
    "load_inputs",
    "longest_run",
    "loss_matrix_for",
    "mcs_records",
    "mcs_run",
    "near_zero_targets",
    "rank_table",
    "regime_tags",
    "runs_of_hits",
    "seed_for",
    "sharpe_difference_ci",
    "study_calendars",
    "with_qlike_excluding_near_zero",
]

#: One seed for the whole analysis; every bootstrap seed is derived from it
#: (:func:`seed_for`) and recorded, so re-running reproduces every number.
MASTER_SEED: Final = 20260829
#: Bootstrap resamples for every MCS and for the Sharpe-difference intervals.
N_BOOT: Final = 10_000
#: MCS levels reported: the design's 0.10 and the brief's 0.25.
ALPHAS: Final = (0.10, 0.25)
#: Transaction costs, basis points of turnover, for the economic-value table.
COST_BPS: Final = (0.0, 1.0, 5.0, 10.0)
#: The model every Sharpe difference is taken against.
BASELINE: Final = "garch11"
#: A scored target below this is one of J1's "near-zero" days: scored, and
#: contributing a QLIKE term of 8.6-13.9 from a target six to seven orders of
#: magnitude below its asset's median (docs/P3_ANALYSIS_VALIDITY.md §1.4).
NEAR_ZERO_TARGET: Final = 1e-8
#: QLIKE with those days set to NaN — every QLIKE figure is reported twice.
QLIKE_EX: Final = "qlike_ex_near_zero"
#: The losses every section runs over: J2's eleven plus the excluded QLIKE.
LOSSES: Final = (*analysis.LOSS_ORDER, QLIKE_EX)
#: The MCS statistics reported side by side.
STATISTICS: Final = ("range", "semi_quadratic")
#: Uncorrected pairwise significance level the DM counts are taken at.
SIGNIFICANCE: Final = 0.05
#: Wider GFC definition, **sensitivity only, never the headline**: the
#: fallback window J1 was told to use had the codebase defined none.
WIDE_GFC: Final = CrisisWindow(
    tag="gfc_wide",
    start=date(2007, 7, 1),
    end=date(2009, 6, 30),
    label="Global financial crisis, wider definition (sensitivity)",
    source_phrase="J1 fallback 'GFC 2007-07-01 -> 2009-06-30'; sensitivity only",
)
#: The HAC ladder for the DM matrices: J2's fixed rule without pre-whitening
#: (its exact estimator), the automatic bandwidth with pre-whitening, and
#: twice the automatic bandwidth.
HAC_LADDER: Final[tuple[tuple[str, inference.HACSpec], ...]] = (
    ("fixed", inference.HACSpec(bandwidth="rule_of_thumb", prewhiten=False)),
    ("auto", inference.HACSpec()),
    ("twice_auto", inference.HACSpec(scale=2.0)),
)
#: The rung whose matrices the markdown renders in full.
HEADLINE_RUNG: Final = "auto"
#: Tolerance at which alpha+beta is called "within 1e-6 of 1", the brief's figure.
PINNED_TOL: Final = 1e-6
#: Where the nu bound of D-032 sits; a recovered nu at it is a bound that binds.
NU_UPPER: Final = analysis.NU_BOUNDS[1]

FLOAT_FORMAT: Final = "%.17g"


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------


def seed_for(*parts: object) -> int:
    """A 31-bit seed derived from :data:`MASTER_SEED` and the run's identity.

    Derived rather than shared so that two bootstraps in one run do not
    consume the same stream, and derived from *names* rather than from a
    counter so that adding a run cannot re-seed every run after it.
    """
    text = "|".join([str(MASTER_SEED), *(str(p) for p in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# inputs: the grid, its calendar, its regimes, the excluded QLIKE
# --------------------------------------------------------------------------


def near_zero_targets(grid: analysis.GridFrame) -> pd.DataFrame:
    """The scored target days below :data:`NEAR_ZERO_TARGET`, one row per (asset, day).

    Each row carries the target's ratio to its asset's median positive target
    rather than the target itself (docs/P3_ORDER_STATISTICS.md), and the range
    of QLIKE over the 13 models on that day, which is the study's own output.
    """
    proxy = grid["proxy_var"].to_numpy(dtype=np.float64)
    near = (proxy > 0.0) & (proxy < NEAR_ZERO_TARGET)
    rows: list[dict[str, Any]] = []
    for (asset, target), group in grid.loc[near].groupby(
        ["asset", "target_index"], observed=True, sort=True
    ):
        own = grid.loc[grid["asset"] == asset]
        positive = own.drop_duplicates("target_index")["proxy_var"]
        positive = positive[positive > 0.0]
        median = float(positive.median())
        rows.append(
            {
                "asset": asset,
                "target_index": _count(target),
                "date": group["date"].iloc[0] if "date" in group.columns else pd.NaT,
                "target_over_asset_median": float(group["proxy_var"].iloc[0]) / median,
                "n_models_scored": int(group["qlike"].notna().sum()),
                "qlike_min_over_models": float(group["qlike"].min()),
                "qlike_max_over_models": float(group["qlike"].max()),
            }
        )
    return pd.DataFrame(rows)


def with_qlike_excluding_near_zero(grid: analysis.GridFrame) -> analysis.GridFrame:
    """``grid`` plus :data:`QLIKE_EX`: QLIKE with the near-zero target days set to NaN."""
    proxy = grid["proxy_var"].to_numpy(dtype=np.float64)
    near = (proxy > 0.0) & (proxy < NEAR_ZERO_TARGET)
    frame = grid.copy()
    frame[QLIKE_EX] = frame["qlike"].where(~near, np.nan)
    return frame


def study_calendars(
    panel: Mapping[str, Any],
) -> dict[str, tuple[pd.DatetimeIndex, NDArray[np.float64]]]:
    """Per asset, the trimmed calendar and return series the grid was scored on.

    The driver's bridge (``benchmarks.grid_primary.asset_data``) computes
    ``log_returns`` of the close and drops the leading bar that has no
    ``C_{t-1}``; a results-frame ``target_index`` is a position in *that*
    series. The two lines are repeated here rather than imported because the
    bridge lives beside the model configs, and :func:`attach_calendar` then
    proves the repetition faithful row by row.
    """
    out: dict[str, tuple[pd.DatetimeIndex, NDArray[np.float64]]] = {}
    for asset, series in panel.items():
        returns = log_returns(series.frame.close)
        first = returns.first_valid_index()
        if first is None:
            raise ValueError(f"{asset}: no finite returns")
        returns = returns[returns.index >= first]
        out[str(asset)] = (pd.DatetimeIndex(returns.index), returns.to_numpy(dtype=np.float64))
    return out


def attach_calendar(
    grid: analysis.GridFrame,
    calendars: Mapping[str, tuple[pd.DatetimeIndex, NDArray[np.float64]]],
) -> analysis.GridFrame:
    """``grid`` with a ``date`` column: the calendar date of each row's target.

    Refuses unless every row's stored ``realized_return`` equals the rebuilt
    series at its ``target_index`` exactly. A calendar that is off by one
    would tag every crisis window one day late and nothing downstream would
    notice; this is the check that makes the date a fact about the row.
    """
    frame = grid.copy()
    dates = np.empty(len(frame), dtype="datetime64[ns]")
    for asset, positions in frame.groupby("asset", observed=True).indices.items():
        if str(asset) not in calendars:
            raise KeyError(f"no calendar for asset {asset!r}")
        index, returns = calendars[str(asset)]
        targets = frame["target_index"].to_numpy(dtype=np.int64)[positions]
        if targets.min() < 0 or targets.max() >= returns.size:
            raise ValueError(f"{asset}: target_index outside the rebuilt series")
        stored = frame["realized_return"].to_numpy(dtype=np.float64)[positions]
        if not np.array_equal(returns[targets], stored):
            worst = float(np.nanmax(np.abs(returns[targets] - stored)))
            raise ValueError(
                f"{asset}: the rebuilt return series does not reproduce realized_return at "
                f"target_index (max abs error {worst}); the calendar cannot be trusted"
            )
        dates[positions] = index.tz_convert("UTC").tz_localize(None).to_numpy()[targets]
    frame["date"] = pd.DatetimeIndex(dates).tz_localize("UTC")
    return frame


def regime_tags(dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """``(headline, wide)`` regime labels for ``dates``.

    ``headline`` is :func:`volbench.data.crisis.tag_dates` verbatim — the
    four pre-registered windows and ``calm``; the pending ``stress_2025_26``
    window is undated and tags nothing. ``wide`` is the same labelling with
    :data:`WIDE_GFC` in place of ``gfc``, for the sensitivity table only.
    """
    headline = tag_dates(dates).astype(str)
    days = pd.Series(dates.tz_convert("UTC").date, index=dates)
    in_wide = days.map(WIDE_GFC.contains).astype(bool).to_numpy()
    wide = headline.to_numpy(dtype=object).copy()
    wide[in_wide] = WIDE_GFC.tag
    return headline, pd.Series(wide, index=dates, dtype=str)


@dataclass(frozen=True)
class Inputs:
    """Everything the sections read: the grid with its derived columns, and the store."""

    grid: analysis.GridFrame
    store: ResultsStore
    manifest: pd.DataFrame
    manifest_digest: str
    store_digest: str
    levels: tuple[float, ...]
    near_zero: pd.DataFrame

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(sorted(str(a) for a in self.grid["asset"].unique()))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(str(m) for m in self.grid["model_label"].unique()))

    def cell(self, asset: str, model: str) -> pd.DataFrame:
        rows = self.grid.loc[(self.grid["asset"] == asset) & (self.grid["model_label"] == model)]
        return rows.sort_values("origin_index", kind="stable").reset_index(drop=True)


def load_inputs(
    store_root: Path,
    manifest_path: Path,
    *,
    panel_builder: Callable[[], Mapping[str, Any]] = build_panel,
) -> Inputs:
    """Read the grid, attach the calendar and regimes, add the excluded QLIKE."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = analysis.load_manifest(manifest_path)
    store = ResultsStore(store_root)
    grid = analysis.with_derived_losses(analysis.load_grid(store, manifest))
    grid = with_qlike_excluding_near_zero(grid)
    grid = attach_calendar(grid, study_calendars(panel_builder()))
    dates = pd.DatetimeIndex(grid["date"])
    headline, wide = regime_tags(dates)
    grid["regime"] = headline.to_numpy()
    grid["regime_wide"] = wide.to_numpy()
    levels = tuple(analysis._tag_to_level(tag) for tag in analysis.level_tags(grid))
    return Inputs(
        grid=grid,
        store=store,
        manifest=manifest,
        manifest_digest=str(payload.get("manifest_digest", "")),
        store_digest=str(payload.get("store_digest", "")),
        levels=levels,
        near_zero=near_zero_targets(grid),
    )


def loss_matrix_for(
    grid: analysis.GridFrame, store: ResultsStore | None, asset: str, loss: str
) -> inference.LossMatrix:
    """One asset's origin x model matrix of ``loss``, NaN where the loss is not finite.

    ``policy="score"``: an origin is unusable for a model exactly where that
    loss is NaN, which is J2's pairwise-complete accounting (a row that lost
    only its QLIKE keeps its CRPS). With a store, the thirteen cells are
    checked to have been scored on one series before they are aligned.
    """
    rows = grid.loc[grid["asset"] == asset]
    return inference.loss_matrix(rows, loss, model_col="model_label", policy="score", store=store)


def drop_model(matrix: inference.LossMatrix, model: str) -> inference.LossMatrix:
    """``matrix`` without one model's column, for the sensitivity runs."""
    if model not in matrix.models:
        raise KeyError(f"{model!r} is not in the matrix ({matrix.models})")
    return inference.LossMatrix(
        values=matrix.values.drop(columns=[model]),
        score=matrix.score,
        asset=matrix.asset,
        horizon=matrix.horizon,
        n_flagged={k: v for k, v in matrix.n_flagged.items() if k != model},
        config_hashes={k: v for k, v in matrix.config_hashes.items() if k != model},
    )


# --------------------------------------------------------------------------
# §1 Diebold-Mariano
# --------------------------------------------------------------------------


def dm_long(
    matrix: inference.LossMatrix,
    *,
    ladder: Sequence[tuple[str, inference.HACSpec]] = HAC_LADDER,
    expected_n: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every unordered pair at every rung of the ladder, one row each.

    ``expected_n`` is J2's pairwise-complete ``n_used`` matrix for this asset
    and loss; when given, every pair's ``n`` must equal it, so the test is
    provably on the sample J2 accounted for and not on one of its own.
    """
    rows: list[dict[str, Any]] = []
    for rung, spec in ladder:
        result = inference.dm_matrix(matrix, hac=spec)
        models = result.models
        for i, a in enumerate(models):
            for j in range(i + 1, len(models)):
                b = models[j]
                pair = result.pairs[(a, b)]
                if expected_n is not None and _count(expected_n.loc[a, b]) != pair.n:
                    raise ValueError(
                        f"{matrix.asset}/{matrix.score}: pair {a}/{b} ran on n={pair.n} but "
                        f"J2's pairwise-complete matrix says {_count(expected_n.loc[a, b])}"
                    )
                rows.append(
                    {
                        "asset": matrix.asset,
                        "loss": matrix.score,
                        "rung": rung,
                        "model_a": a,
                        "model_b": b,
                        "n": pair.n,
                        "n_dropped": pair.n_dropped,
                        "rho1": pair.rho1,
                        "n_eff": pair.n_eff,
                        "rho_prewhiten": pair.rho,
                        "rho_capped": pair.rho_capped,
                        "prewhiten": pair.prewhiten,
                        "bandwidth": pair.bandwidth,
                        "n_lags": pair.lag,
                        "mean_diff": pair.mean_diff,
                        "variance": pair.variance,
                        "statistic": pair.statistic,
                        "p_value": pair.p_value,
                        "hln_factor": pair.hln_factor,
                        "variance_nonpositive": pair.variance_nonpositive,
                        "significant_5pct": bool(pair.p_value < SIGNIFICANCE),
                    }
                )
    return pd.DataFrame(rows)


def _quantiles(values: pd.Series) -> dict[str, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    if finite.empty:
        return {"min": math.nan, "median": math.nan, "max": math.nan}
    return {
        "min": float(finite.min()),
        "median": float(finite.median()),
        "max": float(finite.max()),
    }


def dm_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Per (asset, loss, rung): significant pairs against chance, and the diagnostics' spread."""
    rows: list[dict[str, Any]] = []
    for (asset, loss, rung), group in long.groupby(
        ["asset", "loss", "rung"], observed=True, sort=True
    ):
        pairs = len(group)
        row: dict[str, Any] = {
            "asset": asset,
            "loss": loss,
            "rung": rung,
            "pairs": pairs,
            "significant_5pct": int(group["significant_5pct"].sum()),
            "expected_by_chance": pairs * SIGNIFICANCE,
            "rho_capped": int(group["rho_capped"].sum()),
            "variance_nonpositive": int(group["variance_nonpositive"].sum()),
            "nonfinite_statistic": int((~np.isfinite(group["statistic"])).sum()),
            "p_exactly_0_or_1": int(((group["p_value"] == 0.0) | (group["p_value"] == 1.0)).sum()),
        }
        for column in ("bandwidth", "rho1", "n_eff", "n"):
            for key, value in _quantiles(group[column]).items():
                row[f"{column}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def dm_significance_changes(long: pd.DataFrame) -> pd.DataFrame:
    """Per (asset, loss): how many pairs change 5 % significance between rungs."""
    wide = long.pivot_table(
        index=["asset", "loss", "model_a", "model_b"],
        columns="rung",
        values="significant_5pct",
        aggfunc="first",
    ).astype(bool)
    rungs = [name for name, _ in HAC_LADDER if name in wide.columns]
    rows: list[dict[str, Any]] = []
    for (asset, loss), group in wide.groupby(level=["asset", "loss"], observed=True, sort=True):
        row: dict[str, Any] = {"asset": asset, "loss": loss, "pairs": len(group)}
        for i, first in enumerate(rungs):
            for second in rungs[i + 1 :]:
                changed = group[first] != group[second]
                row[f"changed_{first}_vs_{second}"] = int(changed.sum())
                row[f"lost_{first}_to_{second}"] = int((group[first] & ~group[second]).sum())
                row[f"gained_{first}_to_{second}"] = int((~group[first] & group[second]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def nu_bound_shares(inputs: Inputs, model: str = "garch11_t") -> pd.DataFrame:
    """Per asset: the ``garch11_t`` scheduled fits whose recovered nu sits on the D-032 bound.

    Read from the store, not the model: each fit's law is recovered from the
    first row resting on it (:func:`volbench.analysis.recover_predictive_law`).
    Counted at two tolerances, ``1e-9`` and ``1e-4``, the two J2 reported
    (57 and 61 over the grid, docs/P3_CONVERGENCE_FORENSICS.md §8).
    Where the bound binds the Student-t is at nu = 50 and the config collapses
    toward ``garch11``, so the two are not independent by construction there.
    """
    rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        cell = inputs.cell(asset, model)
        scheduled = cell.loc[cell["fit_origin"] >= 0]
        first = scheduled.groupby("fit_origin", sort=True).head(1)
        n_fits = student = at_bound = near_bound = gaussian = unrecovered = 0
        for row in first.to_dict("records"):
            n_fits += 1
            law = analysis.recover_predictive_law({str(k): v for k, v in row.items()})
            if law is None:
                unrecovered += 1
            elif law.family == "normal":
                gaussian += 1
            else:
                student += 1
                at_bound += law.df >= NU_UPPER - 1e-9
                near_bound += law.df >= NU_UPPER - 1e-4
        rows.append(
            {
                "asset": asset,
                "fits": n_fits,
                "student_t": student,
                "gaussian_fallback": gaussian,
                "unrecovered": unrecovered,
                "nu_at_bound_1e-9": at_bound,
                "nu_at_bound_1e-4": near_bound,
                "share_at_bound_of_fits": (at_bound / n_fits) if n_fits else math.nan,
                "share_at_bound_of_student_t": (at_bound / student) if student else math.nan,
            }
        )
    return pd.DataFrame(rows)


def dm_matrix_markdown(long: pd.DataFrame, asset: str, loss: str, rung: str) -> str:
    """One 13x13 matrix as markdown, ``statistic (p) n`` in every off-diagonal cell.

    Entry ``(row i, column j)`` is the test on ``L_i - L_j``: positive means
    the row model lost *more*. The p-values are two-sided, HLN-corrected
    against ``t_{n-1}``, and **uncorrected for multiplicity**.
    """
    block = long.loc[(long["asset"] == asset) & (long["loss"] == loss) & (long["rung"] == rung)]
    models = sorted(set(block["model_a"]) | set(block["model_b"]))
    cells: dict[tuple[str, str], str] = {}
    for row in block.itertuples(index=False):
        stat, p, n = _num(row.statistic), _num(row.p_value), _count(row.n)
        cells[(str(row.model_a), str(row.model_b))] = f"{stat:+.2f} ({p:.3f}) {n}"
        cells[(str(row.model_b), str(row.model_a))] = f"{-stat:+.2f} ({p:.3f}) {n}"
    header = "| | " + " | ".join(f"`{m}`" for m in models) + " |"
    rule = "|---|" + "---:|" * len(models)
    lines = [header, rule]
    for a in models:
        lines.append(f"| `{a}` | " + " | ".join(cells.get((a, b), "—") for b in models) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# §2 Model Confidence Set
# --------------------------------------------------------------------------


def mcs_run(
    matrix: inference.LossMatrix,
    statistic: str,
    seed: int,
    *,
    n_boot: int = N_BOOT,
    block_length: int | None = None,
) -> inference.MCSResult:
    """One MCS at ``alpha=0.10`` (membership at any level is read off the p-values)."""
    if statistic not in ("range", "max", "semi_quadratic"):
        raise ValueError(f"unknown statistic {statistic!r}")
    return inference.model_confidence_set(
        matrix,
        seed=seed,
        alpha=ALPHAS[0],
        n_boot=n_boot,
        block_length=block_length,
        statistic=statistic,  # type: ignore[arg-type]
        horizon=matrix.horizon,
    )


def mcs_records(
    result: inference.MCSResult, asset: str, loss: str, variant: str, regime: str = "full"
) -> pd.DataFrame:
    """One row per model: mean loss, MCS p-value, elimination step, membership at each level."""
    step = {name: k + 1 for k, name in enumerate(result.elimination_order)}
    rows: list[dict[str, Any]] = []
    for model in result.models:
        row: dict[str, Any] = {
            "asset": asset,
            "loss": loss,
            "regime": regime,
            "variant": variant,
            "statistic": result.statistic,
            "model": model,
            "mean_loss": result.mean_loss[model],
            "mcs_p_value": result.p_values[model],
            "eliminated_at_step": step[model],
            "n": result.n,
            "n_dropped": result.n_dropped,
            "block_length": result.block_length,
            "n_boot": result.n_boot,
            "seed": result.seed,
            "config_hash": result.config_hash,
        }
        for alpha in ALPHAS:
            row[f"in_mcs_{alpha:g}"] = bool(result.p_values[model] >= alpha)
        rows.append(row)
    return pd.DataFrame(rows)


def block_diagnostics(matrix: inference.LossMatrix) -> dict[str, Any]:
    """The Politis-White block length against the persistence it was chosen for.

    On the listwise-complete sample the MCS resamples: the rule's value on
    every pairwise differential (the chosen block is the largest, rounded up)
    and on every loss series itself, each series' first-order autocorrelation,
    and the block length an AR(1) at the most persistent differential's rho_1
    would call for — the yardstick a short block is read against.
    """
    values = matrix.values.to_numpy(dtype=np.float64)
    complete = values[np.all(np.isfinite(values), axis=1)]
    n, m = complete.shape
    names = matrix.models
    pair_pw: list[float] = []
    pair_rho: list[float] = []
    worst_pair = ("", "")
    worst_rho = -math.inf
    for i in range(m):
        for j in range(i + 1, m):
            d = complete[:, i] - complete[:, j]
            pw = inference.politis_white_block_length(d) if n >= 3 else 1.0
            rho = inference.autocorrelation(d, 1) if n >= 2 else math.nan
            pair_pw.append(pw)
            pair_rho.append(rho)
            if math.isfinite(rho) and rho > worst_rho:
                worst_rho, worst_pair = rho, (names[i], names[j])
    series_pw = [
        inference.politis_white_block_length(complete[:, k]) if n >= 3 else 1.0 for k in range(m)
    ]
    series_rho = [
        inference.autocorrelation(complete[:, k], 1) if n >= 2 else math.nan for k in range(m)
    ]
    chosen = inference.default_block_length(complete, horizon=matrix.horizon)
    implied = inference.ar1_block_length(worst_rho, n) if math.isfinite(worst_rho) else math.nan
    return {
        "n": n,
        "block_length": chosen,
        "block_is_1": chosen == 1,
        "block_exceeds_n_over_4": chosen > n / 4.0,
        "pw_pairs_min": float(np.min(pair_pw)),
        "pw_pairs_median": float(np.median(pair_pw)),
        "pw_pairs_max": float(np.max(pair_pw)),
        "rho1_pairs_min": float(np.nanmin(pair_rho)),
        "rho1_pairs_median": float(np.nanmedian(pair_rho)),
        "rho1_pairs_max": float(np.nanmax(pair_rho)),
        "rho1_pairs_max_pair": f"{worst_pair[0]}/{worst_pair[1]}",
        "ar1_implied_block_at_max_rho1": implied,
        "block_below_ar1_implied": bool(math.isfinite(implied) and chosen < implied),
        "pw_series_min": float(np.min(series_pw)),
        "pw_series_median": float(np.median(series_pw)),
        "pw_series_max": float(np.max(series_pw)),
        "rho1_series_min": float(np.nanmin(series_rho)),
        "rho1_series_median": float(np.nanmedian(series_rho)),
        "rho1_series_max": float(np.nanmax(series_rho)),
    }


# --------------------------------------------------------------------------
# §3 VaR / ES backtests
# --------------------------------------------------------------------------


def runs_of_hits(hits: NDArray[np.float64]) -> list[int]:
    """Lengths of every maximal run of consecutive exceedances; a NaN breaks a run."""
    runs: list[int] = []
    current = 0
    for value in hits:
        if value == 1.0:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def longest_run(hits: NDArray[np.float64]) -> int:
    """The longest run of consecutive exceedances, 0 when there is none."""
    runs = runs_of_hits(hits)
    return max(runs) if runs else 0


def backtest_records(inputs: Inputs) -> pd.DataFrame:
    """Every cell at every evaluated level: Kupiec, Christoffersen, counts, runs, FZ0."""
    rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        for model in inputs.models:
            cell = inputs.cell(asset, model)
            for level in inputs.levels:
                result = backtests.var_backtest(cell, level, policy="score", warn=False)
                tag = analysis._level_tag(level)
                hits = cell[f"hit_{tag}"].to_numpy(dtype=np.float64)
                runs = runs_of_hits(hits)
                markov = result.christoffersen
                rows.append(
                    {
                        "asset": asset,
                        "model": model,
                        "level": level,
                        "n": result.n,
                        "n_dropped": result.n_dropped,
                        "n_hits": result.n_hits,
                        "expected_hits": result.expected_hits,
                        "hit_rate": result.hit_rate,
                        "kupiec_lr": result.kupiec.lr,
                        "kupiec_p": result.kupiec.p_value,
                        "ind_lr": markov.lr_ind,
                        "ind_p": markov.p_ind,
                        "cc_lr": markov.lr_cc,
                        "cc_p": markov.p_cc,
                        "n00": markov.n00,
                        "n01": markov.n01,
                        "n10": markov.n10,
                        "n11": markov.n11,
                        "pi01": markov.pi01,
                        "pi11": markov.pi11,
                        "longest_run": max(runs) if runs else 0,
                        "runs_of_2_or_more": int(sum(1 for r in runs if r >= 2)),
                        "fz0_mean": result.fz0_mean,
                        "fz0_n": result.fz0_n,
                        "small_sample": result.small_sample,
                        "backtest_hash": result.config_hash,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# §4 economic value
# --------------------------------------------------------------------------


def _bootstrap_sharpe(
    mean: NDArray[np.float64], mean_sq: NDArray[np.float64], n: int, ppy: float
) -> NDArray[np.float64]:
    variance = (mean_sq - mean * mean) * n / (n - 1)
    out = np.full(mean.shape, np.nan)
    positive = variance > 0.0
    out[positive] = mean[positive] / np.sqrt(variance[positive]) * math.sqrt(ppy)
    return out


def sharpe_difference_ci(
    net_model: NDArray[np.float64],
    net_base: NDArray[np.float64],
    *,
    periods_per_year: float,
    seed: int,
    n_boot: int = N_BOOT,
    block_length: int | None = None,
) -> dict[str, Any]:
    """Percentile moving-block-bootstrap interval on ``Sharpe(model) - Sharpe(baseline)``.

    Both net-return series are resampled with **one** index sequence, so the
    dependence between them survives; the Sharpe of each resample is its mean
    over its ``ddof=1`` standard deviation, annualised — the point estimate's
    own definition (:func:`volbench.econ._sharpe`). The block length is the
    Politis-White rule's largest value over the two differentials the
    statistic is built from, ``r_m - r_b`` and ``r_m^2 - r_b^2``, unless given.
    """
    a = np.asarray(net_model, dtype=np.float64)
    b = np.asarray(net_base, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("the two net-return series must be 1-D and the same length")
    n = a.size
    if n < 3:
        raise ValueError("need at least three periods")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("net returns must be finite")
    if block_length is None:
        candidates = []
        for d in (a - b, a * a - b * b):
            candidates.append(1.0 if np.ptp(d) == 0.0 else inference.politis_white_block_length(d))
        block_length = int(min(n, max(1, math.ceil(max(candidates)))))
    values = np.column_stack([a, a * a, b, b * b])
    means = inference.bootstrap_column_means(values, block_length, n_boot, seed)
    delta = _bootstrap_sharpe(means[:, 0], means[:, 1], n, periods_per_year) - _bootstrap_sharpe(
        means[:, 2], means[:, 3], n, periods_per_year
    )
    finite = delta[np.isfinite(delta)]
    low, high = (
        (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
        if finite.size
        else (math.nan, math.nan)
    )
    return {
        "block_length": block_length,
        "n_boot": n_boot,
        "n_boot_finite": int(finite.size),
        "ci_low": low,
        "ci_high": high,
        "seed": seed,
    }


def econ_records(inputs: Inputs, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Every cell at every cost, with the Sharpe difference against the baseline and its CI."""
    rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        ppy = econ.periods_per_year_for(asset)
        for cost in COST_BPS:
            results: dict[str, econ.VolTargetBacktest] = {}
            origins: dict[str, NDArray[np.int64]] = {}
            for model in inputs.models:
                cell = inputs.cell(asset, model)
                results[model] = econ.volatility_target_backtest(cell, cost_bps=cost)
                origins[model] = cell["origin_index"].to_numpy(dtype=np.int64)
            base = results[BASELINE]
            for model in inputs.models:
                if not np.array_equal(origins[model], origins[BASELINE]):
                    raise ValueError(f"{asset}: {model} and {BASELINE} are not on one origin axis")
                result = results[model]
                row: dict[str, Any] = {
                    "asset": asset,
                    "model": model,
                    "cost_bps": cost,
                    "periods_per_year": ppy,
                    "n_periods": result.n_periods,
                    "n_flat": result.n_flat,
                    "annual_return": result.annual_return,
                    "annual_vol": result.annual_vol,
                    "sharpe": result.sharpe,
                    "gross_sharpe": result.gross_sharpe,
                    "max_drawdown": result.max_drawdown,
                    "avg_leverage": result.avg_leverage,
                    "annual_turnover": result.annual_turnover,
                    "annual_cost_drag": result.annual_cost_drag,
                    "ruined": result.ruined,
                    "sharpe_diff_vs_baseline": result.sharpe - base.sharpe,
                }
                if model == BASELINE:
                    row.update(
                        {
                            "block_length": 0,
                            "n_boot": 0,
                            "n_boot_finite": 0,
                            "ci_low": 0.0,
                            "ci_high": 0.0,
                            "seed": 0,
                        }
                    )
                else:
                    row.update(
                        sharpe_difference_ci(
                            result.net_returns,
                            base.net_returns,
                            periods_per_year=ppy,
                            seed=seed_for("sharpe", asset, model, cost),
                            n_boot=n_boot,
                        )
                    )
                row["ci_excludes_zero"] = bool(row["ci_low"] > 0.0 or row["ci_high"] < 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# §5 cross-asset: ranks only
# --------------------------------------------------------------------------


def rank_table(mean_losses: pd.DataFrame) -> pd.DataFrame:
    """``mean_losses`` (asset x model) to ranks: 1 is the smallest mean loss in that asset."""
    return mean_losses.rank(axis=1, method="min", ascending=True)


def kendall_between(rank_a: pd.Series, rank_b: pd.Series) -> float:
    """Kendall's τ (τ-b) between two rankings of the same models."""
    joined = pd.concat([rank_a.rename("a"), rank_b.rename("b")], axis=1).dropna()
    if len(joined) < 2:
        return math.nan
    return float(stats.kendalltau(joined["a"], joined["b"]).statistic)


def rank_summary(ranks: pd.DataFrame, block: str) -> pd.DataFrame:
    """Per model: mean rank and the distribution of its ranks across ``ranks``' assets."""
    rows: list[dict[str, Any]] = []
    for model in ranks.columns:
        values = ranks[model].dropna()
        rows.append(
            {
                "block": block,
                "model": model,
                "assets": int(values.size),
                "mean_rank": float(values.mean()),
                "rank_min": float(values.min()),
                "rank_q25": float(values.quantile(0.25)),
                "rank_median": float(values.median()),
                "rank_q75": float(values.quantile(0.75)),
                "rank_max": float(values.max()),
                "ranks": " ".join(f"{a}:{int(r)}" for a, r in values.items()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# §amendment 3: the GARCH recursion, recovered from stored forecasts
# --------------------------------------------------------------------------


def garch_recursion_by_block(cell: pd.DataFrame) -> pd.DataFrame:
    """Per scheduled fit: ``(omega, alpha, beta)`` solved from the stored forecasts.

    Within one refit block the variance forecast is re-conditioned daily
    without re-estimation, so consecutive rows satisfy
    ``h_{t+1} = omega + alpha r_t^2 + beta h_t`` with the block's fixed
    parameters, ``h_t`` the row's ``forecast_var`` and ``r_t`` its
    ``realized_return``. Twenty equations in three unknowns per block: the
    least-squares solution is the parameter vector, and the largest relative
    residual says whether the recursion actually holds — it must be at
    floating-point noise, or the block is not a GARCH(1,1) block. An EWMA
    fallback block satisfies the same recursion with ``omega = 0`` and
    ``alpha + beta = 1`` by construction, which is the point of asking.
    """
    ordered = cell.sort_values("origin_index", kind="stable")
    rows: list[dict[str, Any]] = []
    for fit_origin, block in ordered.groupby("fit_origin", sort=True):
        if _count(fit_origin) < 0:
            continue
        h = block["forecast_var"].to_numpy(dtype=np.float64)
        r = block["realized_return"].to_numpy(dtype=np.float64)
        origins = block["origin_index"].to_numpy(dtype=np.int64)
        consecutive = np.diff(origins) == 1
        usable = np.isfinite(h[:-1]) & np.isfinite(h[1:]) & np.isfinite(r[:-1]) & consecutive
        row: dict[str, Any] = {
            "fit_origin": _count(fit_origin),
            "fit_status": str(block["fit_status"].iloc[0]),
            "fallback": str(block["fit_status"].iloc[0]).startswith("fallback="),
            "n_rows": len(block),
            "n_equations": int(usable.sum()),
        }
        if int(usable.sum()) < 4:
            row.update(
                {
                    "omega": math.nan,
                    "alpha": math.nan,
                    "beta": math.nan,
                    "max_rel_residual": math.nan,
                }
            )
        else:
            design = np.column_stack(
                [np.ones(int(usable.sum())), r[:-1][usable] ** 2, h[:-1][usable]]
            )
            target = h[1:][usable]
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
            residual = target - design @ solution
            row.update(
                {
                    "omega": float(solution[0]),
                    "alpha": float(solution[1]),
                    "beta": float(solution[2]),
                    "max_rel_residual": float(np.max(np.abs(residual) / np.abs(target))),
                }
            )
        row["alpha_plus_beta"] = row["alpha"] + row["beta"]
        rows.append(row)
    return pd.DataFrame(rows)


def ratio_summary(ratio: NDArray[np.float64]) -> dict[str, float]:
    finite = ratio[np.isfinite(ratio)]
    keys = ("q01", "q05", "q25", "median", "q75", "q95", "q99", "min", "max", "share_within_5pct")
    if finite.size == 0:
        return dict.fromkeys(keys, math.nan)
    q = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "share_within_5pct": float(np.mean(np.abs(finite - 1.0) <= 0.05)),
    }


def recursion_residuals(
    cell: pd.DataFrame, params: Mapping[int, tuple[float, float, float]]
) -> pd.Series:
    """Per fit: the largest relative one-step residual at *given* ``(omega, alpha, beta)``.

    The same recursion :func:`garch_recursion_by_block` solves for, evaluated
    instead at parameters supplied from outside — J2's re-fit — so the
    stored forecasts can be checked against the fit that produced them.
    """
    ordered = cell.sort_values("origin_index", kind="stable")
    out: dict[int, float] = {}
    for fit_origin, block in ordered.groupby("fit_origin", sort=True):
        key = _count(fit_origin)
        if key < 0 or key not in params:
            continue
        omega, alpha, beta = params[key]
        h = block["forecast_var"].to_numpy(dtype=np.float64)
        r = block["realized_return"].to_numpy(dtype=np.float64)
        origins = block["origin_index"].to_numpy(dtype=np.int64)
        usable = np.isfinite(h[:-1]) & np.isfinite(h[1:]) & np.isfinite(r[:-1])
        usable &= np.diff(origins) == 1
        if not usable.any():
            out[key] = math.nan
            continue
        predicted = omega + alpha * r[:-1][usable] ** 2 + beta * h[:-1][usable]
        out[key] = float(np.max(np.abs(predicted - h[1:][usable]) / np.abs(h[1:][usable])))
    return pd.Series(out, name="ref_max_rel_residual", dtype=float)


#: Below this relative residual the stored forecasts are read as following the
#: one-step recursion; the blocks that do sit at 1e-14 and below.
RECURSION_TOL: Final = 1e-8


def boundary_persistence(
    inputs: Inputs, asset: str = "BTC-USD", fits_path: Path | None = None
) -> dict[str, Any]:
    """Amendment 3: the GARCH cells against EWMA at the same origins, split by the governing fit.

    The parameter behind each origin's forecast comes from J2's re-fit
    (``fits_path``, the parquet that reproduced every stored ``fit_status``)
    when it is on disk; the store's own one-step recursion is the cross-check,
    and also the fallback source where the parquet is absent — but only on
    blocks where that recursion reproduces the stored forecasts, because a
    filter re-run over a sliding window at ``alpha = 0`` and ``beta`` near 1
    keeps the window's initial condition alive through ``beta^window`` and the
    one-step recursion then does not hold between consecutive forecasts.
    """
    ewma = inputs.cell(asset, "ewma").set_index("origin_index")["forecast_var"]
    out: dict[str, Any] = {"asset": asset, "configs": {}, "reference": fits_path is not None}
    reference = pd.read_parquet(fits_path) if fits_path is not None else None
    for config in ("garch11_t", "garch11"):
        cell = inputs.cell(asset, config)
        window = _count(
            inputs.store.read_config(str(cell["config_hash"].iloc[0]))["splitter"]["window"]
        )
        blocks = garch_recursion_by_block(cell).rename(
            columns={
                "omega": "omega_store",
                "alpha": "alpha_store",
                "beta": "beta_store",
                "alpha_plus_beta": "alpha_plus_beta_store",
                "max_rel_residual": "store_max_rel_residual",
            }
        )
        blocks["store_recursion_reproduced"] = blocks["store_max_rel_residual"] <= RECURSION_TOL
        governing = blocks["alpha_plus_beta_store"].where(blocks["store_recursion_reproduced"])
        source = "store recursion (blocks where it is reproduced)"
        if reference is not None:
            ref = reference.loc[
                (reference["asset"] == asset) & (reference["config"] == config)
            ].set_index("fit_origin")
            ref = ref.loc[~ref.index.duplicated()]
            joined = blocks.set_index("fit_origin").join(
                ref[
                    ["omega_return_scale", "alpha[1]", "beta[1]", "alpha_plus_beta", "fallback"]
                ].rename(
                    columns={
                        "omega_return_scale": "omega_ref",
                        "alpha[1]": "alpha_ref",
                        "beta[1]": "beta_ref",
                        "alpha_plus_beta": "alpha_plus_beta_ref",
                        "fallback": "fallback_ref",
                    }
                ),
                how="left",
            )
            params = {
                _count(k): (_num(v["omega_ref"]), _num(v["alpha_ref"]), _num(v["beta_ref"]))
                for k, v in joined.iterrows()
                if math.isfinite(_num(v["alpha_ref"]))
            }
            joined = joined.join(recursion_residuals(cell, params), how="left")
            joined["beta_ref_pow_window"] = joined["beta_ref"] ** window
            blocks = joined.reset_index()
            governing = blocks["alpha_plus_beta_ref"]
            source = "J2 re-fit (docs/P3_CONVERGENCE_FITS.parquet)"
        blocks["alpha_plus_beta_governing"] = governing.to_numpy(dtype=np.float64)
        by_fit = blocks.set_index("fit_origin")
        origin_fit = cell["fit_origin"].to_numpy(dtype=np.int64)
        apb = by_fit["alpha_plus_beta_governing"].reindex(origin_fit).to_numpy(dtype=np.float64)
        is_fallback = by_fit["fallback"].reindex(origin_fit).to_numpy(dtype=bool)
        ratio = cell["forecast_var"].to_numpy(dtype=np.float64) / ewma.reindex(
            cell["origin_index"]
        ).to_numpy(dtype=np.float64)
        group = np.where(
            is_fallback,
            "fallback_ewma",
            np.where(
                ~np.isfinite(apb),
                "unresolved",
                np.where(np.abs(apb - 1.0) <= PINNED_TOL, "converged_pinned", "converged_interior"),
            ),
        )
        groups: dict[str, Any] = {}
        for name in ("fallback_ewma", "converged_pinned", "converged_interior", "unresolved"):
            mask = group == name
            if not mask.any():
                continue
            groups[name] = {
                "n_rows": int(mask.sum()),
                "n_fits": int(cell.loc[mask, "fit_origin"].nunique()),
                **ratio_summary(ratio[mask]),
            }
        converged = blocks.loc[~blocks["fallback"]]
        gov = converged["alpha_plus_beta_governing"]
        not_reproduced = converged.loc[~converged["store_recursion_reproduced"]]
        entry: dict[str, Any] = {
            "window": window,
            "alpha_plus_beta_source": source,
            "fits": len(blocks),
            "fallback_fits": int(blocks["fallback"].sum()),
            "converged_fits": len(converged),
            "converged_pinned_1e-6": int((np.abs(gov - 1.0) <= PINNED_TOL).sum()),
            "converged_pinned_1e-4": int((np.abs(gov - 1.0) <= 1e-4).sum()),
            "converged_pinned_1e-2": int((np.abs(gov - 1.0) <= 1e-2).sum()),
            "converged_unresolved": int(gov.isna().sum()),
            "store_recursion_reproduced_1e-12": int(
                (converged["store_max_rel_residual"] <= 1e-12).sum()
            ),
            "store_recursion_reproduced_1e-8": int(converged["store_recursion_reproduced"].sum()),
            "store_recursion_reproduced_1e-4": int(
                (converged["store_max_rel_residual"] <= 1e-4).sum()
            ),
            "store_recursion_max_rel_residual": float(converged["store_max_rel_residual"].max()),
            "blocks_too_short_to_solve": int(blocks["omega_store"].isna().sum()),
            "ratio_all": ratio_summary(ratio),
            "groups": groups,
            "blocks": blocks,
        }
        if reference is not None:
            entry["reference_fits_matched"] = int(blocks["alpha_plus_beta_ref"].notna().sum())
            entry["reference_fallback_flags_agree"] = int(
                (blocks["fallback_ref"].astype(bool) == blocks["fallback"]).sum()
            )
            same = converged.loc[converged["store_max_rel_residual"] <= 1e-12]
            entry["max_abs_diff_store_vs_reference_where_reproduced"] = (
                float(
                    np.nanmax(np.abs(same["alpha_plus_beta_store"] - same["alpha_plus_beta_ref"]))
                )
                if len(same)
                else math.nan
            )
            entry["reference_recursion_reproduced_1e-8"] = int(
                (converged["ref_max_rel_residual"] <= RECURSION_TOL).sum()
            )
            entry["reference_recursion_max_rel_residual"] = float(
                converged["ref_max_rel_residual"].max()
            )
            entry["not_reproduced_alpha_ref_below_1e-6"] = int(
                (not_reproduced["alpha_ref"] <= 1e-6).sum()
            )
            entry["not_reproduced_beta_pow_window_min"] = (
                float(not_reproduced["beta_ref_pow_window"].min())
                if len(not_reproduced)
                else math.nan
            )
            entry["not_reproduced_beta_pow_window_max"] = (
                float(not_reproduced["beta_ref_pow_window"].max())
                if len(not_reproduced)
                else math.nan
            )
            entry["reproduced_beta_pow_window_max"] = (
                float(
                    converged.loc[
                        converged["store_recursion_reproduced"], "beta_ref_pow_window"
                    ].max()
                )
                if converged["store_recursion_reproduced"].any()
                else math.nan
            )
        out["configs"][config] = entry
    return out


# --------------------------------------------------------------------------
# §amendment 4: QLIKE twice, AutoARIMA carried
# --------------------------------------------------------------------------


def qlike_twice(inputs: Inputs) -> pd.DataFrame:
    """Per asset x model: mean QLIKE with and without the near-zero days, both SEs."""
    rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        for model in inputs.models:
            cell = inputs.cell(asset, model)
            row: dict[str, Any] = {"asset": asset, "model": model}
            for label, column in (("all", "qlike"), ("ex_near_zero", QLIKE_EX)):
                values = cell[column].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                fixed = analysis.hac_mean_se(values)
                auto = (
                    inference.long_run_variance(finite, inference.HACSpec())
                    if finite.size >= 3
                    else None
                )
                row[f"n_{label}"] = int(finite.size)
                row[f"mean_{label}"] = float(finite.mean()) if finite.size else math.nan
                row[f"se_fixed_{label}"] = fixed["se"]
                row[f"se_auto_{label}"] = (
                    math.sqrt(max(auto.omega, 0.0) / finite.size) if auto else math.nan
                )
            row["rows_excluded"] = row["n_all"] - row["n_ex_near_zero"]
            row["mean_shift_relative"] = (
                (row["mean_ex_near_zero"] - row["mean_all"]) / row["mean_all"]
                if row["mean_all"]
                else math.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_diagnostics_table(inputs: Inputs, fit_probe: Path | None) -> pd.DataFrame:
    """Per asset x model: the fallback / optimiser-status rate, or ``not instrumented``.

    GARCH-family rates come from the store (``fit_status`` per scheduled
    fit); AutoARIMA's non-zero optimiser status from J1's re-fit probe when
    its parquet is present (it is under ``data/`` and is not committed), and
    every other config reads ``not instrumented`` — never ``0`` and never
    ``nan`` (docs/P3_INSTRUMENTATION_GAP.md).
    """
    rates = analysis.fallback_rates(inputs.grid).set_index(["asset", "model"])
    probe = None
    if fit_probe is not None and fit_probe.is_file():
        probe = pd.read_parquet(fit_probe)
        probe = probe.loc[probe["model"] == "autoarima"]
    rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        for model in inputs.models:
            entry: Any = rates.loc[(asset, model)]
            row: dict[str, Any] = {"asset": asset, "model": model}
            if bool(entry["instrumented"]):
                k, n = _count(entry["n_fallback"]), _count(entry["n_fits"])
                row.update(
                    {
                        "signal": "ewma fallback (store, fit_status)",
                        "k": k,
                        "n": n,
                        "rate": f"{k}/{n} = {100.0 * k / n:.2f}%",
                    }
                )
            elif model == "autoarima" and probe is not None:
                own = probe.loc[probe["asset"] == asset]
                k, n = int((own["code"].fillna(0) != 0).sum()), len(own)
                row.update(
                    {
                        "signal": "optimizer status != 0 (J1 re-fit probe)",
                        "k": k,
                        "n": n,
                        "rate": f"{k}/{n} = {100.0 * k / n:.2f}%" if n else "not instrumented",
                    }
                )
            else:
                row.update(
                    {
                        "signal": "not instrumented",
                        "k": pd.NA,
                        "n": pd.NA,
                        "rate": "not instrumented",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# crisis sub-samples
# --------------------------------------------------------------------------


def regime_loss_table(
    rows: pd.DataFrame, asset: str, regime: str, losses: Sequence[str] = LOSSES
) -> pd.DataFrame:
    """Per model x loss inside one regime: mean, pre-whitened HAC SE, n."""
    out: list[dict[str, Any]] = []
    for model, group in rows.groupby("model_label", observed=True, sort=True):
        ordered = group.sort_values("origin_index", kind="stable")
        for loss in losses:
            values = ordered[loss].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            se = math.nan
            if finite.size >= 3:
                lrv = inference.long_run_variance(finite, inference.HACSpec())
                se = math.sqrt(max(lrv.omega, 0.0) / finite.size)
            out.append(
                {
                    "asset": asset,
                    "regime": regime,
                    "model": model,
                    "loss": loss,
                    "n": int(finite.size),
                    "mean": float(finite.mean()) if finite.size else math.nan,
                    "se_auto": se,
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# environment for the manifest
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    for name in ("volbench", "numpy", "scipy", "pandas", "statsmodels", "arch", "pyarrow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: Any) -> float:
    """A pandas scalar as a float, for the stubs' sake."""
    return float(value)


def _count(value: Any) -> int:
    """A pandas scalar as an int, for the stubs' sake."""
    return int(value)


def _row(frame: pd.DataFrame, label: str) -> pd.Series:
    """One row of ``frame`` as a Series, typed as one."""
    return frame.T[label]


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}g}"


# --------------------------------------------------------------------------
# the sections, assembled
# --------------------------------------------------------------------------


@dataclass
class Outputs:
    """What a run produced: frames to write, manifest entries, timings, notes."""

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def pairwise_reference(path: Path | None) -> dict[tuple[str, str], pd.DataFrame]:
    """J2's ``n_used`` matrices keyed by (asset, loss), read rather than re-derived."""
    if path is None or not path.is_file():
        return {}
    long = pd.read_csv(path)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for (asset, loss), group in long.groupby(["asset", "loss"], sort=True):
        out[(str(asset), str(loss))] = group.pivot(
            index="model_a", columns="model_b", values="n_used"
        )
    return out


def loss_matrices(inputs: Inputs) -> dict[tuple[str, str], inference.LossMatrix]:
    """Every (asset, loss) matrix, built once and shared by the DM and MCS sections."""
    return {
        (asset, loss): loss_matrix_for(inputs.grid, inputs.store, asset, loss)
        for asset in inputs.assets
        for loss in LOSSES
    }


def run_dm(
    inputs: Inputs,
    matrices: Mapping[tuple[str, str], inference.LossMatrix],
    reference: Mapping[tuple[str, str], pd.DataFrame],
    out: Outputs,
) -> None:
    started = time.perf_counter()
    parts = [
        dm_long(matrices[(asset, loss)], expected_n=reference.get((asset, loss)))
        for asset in inputs.assets
        for loss in LOSSES
    ]
    long = pd.concat(parts, ignore_index=True)
    out.frames["P3_DM"] = long
    out.frames["P3_DM_SUMMARY"] = dm_summary(long)
    out.frames["P3_DM_CHANGES"] = dm_significance_changes(long)
    out.frames["P3_DM_NU_BOUND"] = nu_bound_shares(inputs)
    out.notes["dm_pairs_checked_against_j2"] = sum(
        1 for asset in inputs.assets for loss in LOSSES if (asset, loss) in reference
    )
    out.manifest["dm"] = {
        "ladder": {name: spec.as_dict() for name, spec in HAC_LADDER},
        "hln": "factor sqrt((n + 1 - 2h + h(h-1)/n)/n) at h = 1, reference t_{n-1}",
        "significance": SIGNIFICANCE,
        "bandwidth_by_asset_loss_rung": {
            f"{r.asset}|{r.loss}|{r.rung}": {
                "min": r.bandwidth_min,
                "median": r.bandwidth_median,
                "max": r.bandwidth_max,
            }
            for r in out.frames["P3_DM_SUMMARY"].itertuples(index=False)
        },
    }
    out.timings["dm"] = time.perf_counter() - started


def run_mcs(
    inputs: Inputs,
    matrices: Mapping[tuple[str, str], inference.LossMatrix],
    out: Outputs,
    *,
    n_boot: int = N_BOOT,
) -> None:
    started = time.perf_counter()
    records: list[pd.DataFrame] = []
    blocks: list[dict[str, Any]] = []
    seeds: dict[str, int] = {}
    block_lengths: dict[str, int] = {}
    sensitivity = {"BTC-USD": "garch11_t", "HSI": "gjr"}
    for asset in inputs.assets:
        for loss in LOSSES:
            matrix = matrices[(asset, loss)]
            diagnostics = {"asset": asset, "loss": loss, "variant": "headline"}
            diagnostics.update(block_diagnostics(matrix))
            blocks.append(diagnostics)
            for statistic in STATISTICS:
                seed = seed_for("mcs", asset, loss, statistic, "headline")
                result = mcs_run(matrix, statistic, seed, n_boot=n_boot)
                records.append(mcs_records(result, asset, loss, "headline"))
                key = f"{asset}|{loss}|{statistic}|headline"
                seeds[key] = seed
                block_lengths[key] = result.block_length
            if asset in sensitivity:
                dropped = sensitivity[asset]
                reduced = drop_model(matrix, dropped)
                diagnostics = {"asset": asset, "loss": loss, "variant": f"drop_{dropped}"}
                diagnostics.update(block_diagnostics(reduced))
                blocks.append(diagnostics)
                for statistic in STATISTICS:
                    seed = seed_for("mcs", asset, loss, statistic, f"drop_{dropped}")
                    result = mcs_run(reduced, statistic, seed, n_boot=n_boot)
                    records.append(mcs_records(result, asset, loss, f"drop_{dropped}"))
                    key = f"{asset}|{loss}|{statistic}|drop_{dropped}"
                    seeds[key] = seed
                    block_lengths[key] = result.block_length
    out.frames["P3_MCS"] = pd.concat(records, ignore_index=True)
    out.frames["P3_MCS_BLOCKS"] = pd.DataFrame(blocks)
    out.frames["P3_MCS_SENSITIVITY"] = mcs_sensitivity_table(out.frames["P3_MCS"], sensitivity)
    out.manifest["mcs"] = {
        "n_boot": n_boot,
        "alphas": list(ALPHAS),
        "statistics": list(STATISTICS),
        "block_length_rule": "Politis-White (2004; Patton, Politis & White 2009), largest over "
        "the pairwise loss differentials of the listwise-complete sample, rounded up, floored "
        "at the horizon",
        "bootstrap": "moving block (Kuensch 1989), non-circular, one index sequence per resample "
        "shared by every model",
        "seeds": seeds,
        "block_lengths": block_lengths,
        "sensitivity": sensitivity,
    }
    out.timings["mcs"] = time.perf_counter() - started


def survivors(records: pd.DataFrame, alpha: float) -> str:
    column = f"in_mcs_{alpha:g}"
    return ", ".join(f"`{m}`" for m in records.loc[records[column], "model"])


def mcs_sensitivity_table(records: pd.DataFrame, sensitivity: Mapping[str, str]) -> pd.DataFrame:
    """Headline survivors (the dropped model removed) beside the sensitivity survivors."""
    rows: list[dict[str, Any]] = []
    for asset, dropped in sensitivity.items():
        variant = f"drop_{dropped}"
        for loss in LOSSES:
            for statistic in STATISTICS:
                head = records.loc[
                    (records["asset"] == asset)
                    & (records["loss"] == loss)
                    & (records["statistic"] == statistic)
                    & (records["variant"] == "headline")
                    & (records["model"] != dropped)
                ]
                sens = records.loc[
                    (records["asset"] == asset)
                    & (records["loss"] == loss)
                    & (records["statistic"] == statistic)
                    & (records["variant"] == variant)
                ]
                row: dict[str, Any] = {
                    "asset": asset,
                    "dropped": dropped,
                    "loss": loss,
                    "statistic": statistic,
                    "block_length_headline": int(head["block_length"].iloc[0]),
                    "block_length_sensitivity": int(sens["block_length"].iloc[0]),
                }
                for alpha in ALPHAS:
                    column = f"in_mcs_{alpha:g}"
                    a = set(head.loc[head[column], "model"])
                    b = set(sens.loc[sens[column], "model"])
                    row[f"survivors_headline_{alpha:g}"] = ", ".join(sorted(a))
                    row[f"survivors_sensitivity_{alpha:g}"] = ", ".join(sorted(b))
                    row[f"changed_{alpha:g}"] = a != b
                    row[f"entered_{alpha:g}"] = ", ".join(sorted(b - a))
                    row[f"left_{alpha:g}"] = ", ".join(sorted(a - b))
                rows.append(row)
    return pd.DataFrame(rows)


def run_backtests(inputs: Inputs, out: Outputs) -> None:
    started = time.perf_counter()
    out.frames["P3_BACKTESTS"] = backtest_records(inputs)
    out.manifest["backtests"] = {
        "levels": list(inputs.levels),
        "policy": "score: a row is excluded only where its hit indicator is NaN",
        "hit": "1{r_t < VaR_t}, strict, as evaluate.py records it",
    }
    out.timings["backtests"] = time.perf_counter() - started


def run_econ(inputs: Inputs, out: Outputs, *, n_boot: int = N_BOOT) -> None:
    started = time.perf_counter()
    frame = econ_records(inputs, n_boot=n_boot)
    out.frames["P3_ECON"] = frame
    out.manifest["econ"] = {
        "costs_bps": list(COST_BPS),
        "baseline": BASELINE,
        "annual_target_vol": econ.DEFAULT_TARGET_VOL,
        "leverage_cap": econ.DEFAULT_LEVERAGE_CAP,
        "periods_per_year": {a: econ.periods_per_year_for(a) for a in inputs.assets},
        "sharpe_ci": "percentile moving-block bootstrap of Sharpe(model) - Sharpe(baseline), one "
        "index sequence for both series, 95 %",
        "n_boot": n_boot,
        "seeds": {
            f"{r.asset}|{r.model}|{r.cost_bps:g}": _count(r.seed)
            for r in frame.itertuples(index=False)
            if r.model != BASELINE
        },
        "block_lengths": {
            f"{r.asset}|{r.model}|{r.cost_bps:g}": _count(r.block_length)
            for r in frame.itertuples(index=False)
            if r.model != BASELINE
        },
    }
    out.timings["econ"] = time.perf_counter() - started


def run_cross_asset(inputs: Inputs, out: Outputs) -> None:
    started = time.perf_counter()
    records = out.frames["P3_MCS"]
    headline = records.loc[records["variant"] == "headline"]
    crypto = {a for a in inputs.assets if a in econ.CRYPTO_ASSETS}
    blocks = {
        "all": set(inputs.assets),
        "equity": set(inputs.assets) - crypto,
        "crypto": crypto,
    }
    summaries: list[pd.DataFrame] = []
    rank_rows: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    ranks_by_loss: dict[str, pd.DataFrame] = {}
    for loss in LOSSES:
        base = headline.loc[(headline["loss"] == loss) & (headline["statistic"] == "range")]
        means = base.pivot(index="asset", columns="model", values="mean_loss")
        ranks = rank_table(means)
        ranks_by_loss[loss] = ranks
        long = ranks.reset_index().melt(id_vars="asset", var_name="model", value_name="rank")
        long.insert(1, "loss", loss)
        rank_rows.append(long)
        for block, members in blocks.items():
            summary = rank_summary(ranks.loc[sorted(members)], block)
            summary.insert(0, "loss", loss)
            summaries.append(summary)
            for statistic in STATISTICS:
                sub = headline.loc[
                    (headline["loss"] == loss)
                    & (headline["statistic"] == statistic)
                    & headline["asset"].isin(members)
                ]
                for model, group in sub.groupby("model", sort=True):
                    row: dict[str, Any] = {
                        "loss": loss,
                        "statistic": statistic,
                        "block": block,
                        "model": model,
                        "assets": len(group),
                    }
                    for alpha in ALPHAS:
                        row[f"survives_{alpha:g}"] = int(group[f"in_mcs_{alpha:g}"].sum())
                    count_rows.append(row)
    kendall_rows: list[dict[str, Any]] = []
    for asset in inputs.assets:
        crps = _row(ranks_by_loss["crps"], asset)
        kendall_rows.append(
            {
                "asset": asset,
                "block": "crypto" if asset in crypto else "equity",
                "tau_crps_vs_qlike": kendall_between(crps, _row(ranks_by_loss["qlike"], asset)),
                "tau_crps_vs_qlike_ex_near_zero": kendall_between(
                    crps, _row(ranks_by_loss[QLIKE_EX], asset)
                ),
                "tau_crps_vs_log_score": kendall_between(
                    crps, _row(ranks_by_loss["log_score"], asset)
                ),
                "tau_crps_vs_fz0_avg": kendall_between(crps, _row(ranks_by_loss["fz0_avg"], asset)),
                "tau_qlike_vs_fz0_avg": kendall_between(
                    _row(ranks_by_loss["qlike"], asset), _row(ranks_by_loss["fz0_avg"], asset)
                ),
            }
        )
    out.frames["P3_CROSS_ASSET_RANKS"] = pd.concat(rank_rows, ignore_index=True)
    out.frames["P3_CROSS_ASSET_RANK_SUMMARY"] = pd.concat(summaries, ignore_index=True)
    out.frames["P3_CROSS_ASSET_MCS_COUNTS"] = pd.DataFrame(count_rows)
    out.frames["P3_CROSS_ASSET_KENDALL"] = pd.DataFrame(kendall_rows)
    out.manifest["cross_asset"] = {
        "rank": "1 = smallest mean loss within the asset, on the listwise-complete sample the MCS "
        "used; ties share the minimum rank",
        "blocks": {k: sorted(v) for k, v in blocks.items()},
        "kendall": "tau-b, scipy.stats.kendalltau, on the two rankings of the 13 models",
    }
    out.timings["cross_asset"] = time.perf_counter() - started


def run_crisis(inputs: Inputs, out: Outputs, *, n_boot: int = N_BOOT) -> None:
    started = time.perf_counter()
    tables: list[pd.DataFrame] = []
    records: list[pd.DataFrame] = []
    blocks: list[dict[str, Any]] = []
    seeds: dict[str, int] = {}
    coverage: list[dict[str, Any]] = []
    regimes = [(w.tag, "regime", "headline") for w in CRISIS_WINDOWS]
    regimes.append((CALM_TAG, "regime", "headline"))
    regimes.append((WIDE_GFC.tag, "regime_wide", "sensitivity"))
    for asset in inputs.assets:
        own = inputs.grid.loc[inputs.grid["asset"] == asset]
        for tag, column, kind in regimes:
            rows = own.loc[own[column] == tag]
            days = int(rows["target_index"].nunique())
            coverage.append(
                {
                    "asset": asset,
                    "regime": tag,
                    "kind": kind,
                    "days": days,
                    "first_date": rows["date"].min() if days else pd.NaT,
                    "last_date": rows["date"].max() if days else pd.NaT,
                }
            )
            if days < 3:
                continue
            tables.append(regime_loss_table(rows, asset, tag))
            for loss in LOSSES:
                matrix = inference.loss_matrix(
                    rows, loss, model_col="model_label", policy="score", store=None
                )
                complete = int(
                    np.all(np.isfinite(matrix.values.to_numpy(dtype=np.float64)), axis=1).sum()
                )
                if complete < 3:
                    continue
                diagnostics = {"asset": asset, "loss": loss, "variant": tag}
                diagnostics.update(block_diagnostics(matrix))
                blocks.append(diagnostics)
                seed = seed_for("mcs-regime", asset, tag, loss, "range")
                result = mcs_run(matrix, "range", seed, n_boot=n_boot)
                records.append(mcs_records(result, asset, loss, kind, regime=tag))
                seeds[f"{asset}|{tag}|{loss}|range"] = seed
    out.frames["P3_CRISIS_COVERAGE"] = pd.DataFrame(coverage)
    out.frames["P3_CRISIS"] = pd.concat(tables, ignore_index=True)
    out.frames["P3_CRISIS_MCS"] = pd.concat(records, ignore_index=True)
    out.frames["P3_CRISIS_BLOCKS"] = pd.DataFrame(blocks)
    out.manifest["crisis"] = {
        "windows": [
            {
                "tag": w.tag,
                "start": w.start.isoformat(),
                "end": w.end.isoformat(),
                "label": w.label,
                "source_phrase": w.source_phrase,
                "source": "volbench.data.crisis.CRISIS_WINDOWS (docs/research_design.md)",
            }
            for w in CRISIS_WINDOWS
        ],
        "pending": [
            {"tag": w.tag, "label": w.label, "blocked_on": w.blocked_on} for w in PENDING_WINDOWS
        ],
        "sensitivity_window": {
            "tag": WIDE_GFC.tag,
            "start": WIDE_GFC.start.isoformat(),
            "end": WIDE_GFC.end.isoformat(),
            "source_phrase": WIDE_GFC.source_phrase,
        },
        "n_boot": n_boot,
        "statistic": "range",
        "seeds": seeds,
    }
    out.timings["crisis"] = time.perf_counter() - started


def run_boundary(inputs: Inputs, out: Outputs, fits_path: Path | None) -> None:
    started = time.perf_counter()
    result = boundary_persistence(inputs, fits_path=fits_path)
    frames: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    for config, entry in result["configs"].items():
        blocks = entry["blocks"].copy()
        blocks.insert(0, "config", config)
        blocks.insert(0, "asset", result["asset"])
        frames.append(blocks)
        for group, values in entry["groups"].items():
            summary.append({"asset": result["asset"], "config": config, "group": group, **values})
        summary.append(
            {
                "asset": result["asset"],
                "config": config,
                "group": "all",
                "n_rows": len(inputs.cell(result["asset"], config)),
                "n_fits": entry["fits"],
                **entry["ratio_all"],
            }
        )
    out.frames["P3_BOUNDARY_FITS"] = pd.concat(frames, ignore_index=True)
    out.frames["P3_BOUNDARY_RATIOS"] = pd.DataFrame(summary)
    out.notes["boundary"] = {
        "reference_parquet_present": result["reference"],
        **{
            config: {k: v for k, v in entry.items() if k not in ("blocks", "groups", "ratio_all")}
            for config, entry in result["configs"].items()
        },
    }
    out.timings["boundary"] = time.perf_counter() - started


def run_qlike(inputs: Inputs, out: Outputs, fit_probe: Path | None) -> None:
    started = time.perf_counter()
    out.frames["P3_QLIKE_TWICE"] = qlike_twice(inputs)
    out.frames["P3_FIT_DIAGNOSTICS"] = fit_diagnostics_table(inputs, fit_probe)
    out.frames["P3_NEAR_ZERO_TARGETS"] = inputs.near_zero
    out.timings["qlike"] = time.perf_counter() - started


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def _table(frame: pd.DataFrame, formats: Mapping[str, Callable[[Any], str]] | None = None) -> str:
    """``frame`` as markdown with per-column formatters, defaulting to ``str``."""
    shown = frame.copy()
    for column in shown.columns:
        fmt = (formats or {}).get(str(column))
        if fmt is not None:
            shown[column] = [fmt(v) for v in shown[column]]
        elif shown[column].dtype.kind == "f":
            shown[column] = [_fmt(float(v)) for v in shown[column]]
        elif shown[column].dtype.kind == "b":
            shown[column] = ["yes" if bool(v) else "no" for v in shown[column]]
    return frame_markdown(shown)


def _provenance(inputs: Inputs, extra: Sequence[tuple[str, str]] = ()) -> str:
    rows = [
        (
            "Grid",
            f"`docs/P3_GRID_manifest.json`, `manifest_digest` `{inputs.manifest_digest[:16]}…`, "
            f"`store_digest` `{inputs.store_digest[:16]}…`",
        ),
        (
            "Rows",
            f"{len(inputs.grid):,} = {len(inputs.assets)} assets x {len(inputs.models)} configs x "
            f"h=1",
        ),
        (
            "Generated by",
            "`python -m volbench.benchmarks.inference_tables`; numerics in `volbench.inference`, "
            "`volbench.backtests`, `volbench.econ`, `volbench.analysis`",
        ),
        (
            "Samples",
            "J2's pairwise-complete accounting (`docs/P3_PAIRWISE_COMPLETE.csv`), read and "
            "asserted, not re-derived",
        ),
        (
            "Manifest",
            "`docs/P3_ANALYSIS_manifest.json` — seeds, B, block lengths, bandwidths, hashes, "
            "digests, versions, pins, git SHA",
        ),
        (
            "Environment",
            '`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES="X86_V4 '
            'AVX512_ICL AVX512_SPR"`',
        ),
        *extra,
    ]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


REPORTED_NOT_INTERPRETED: Final = (
    "**Reported, not interpreted.** No model is ranked as better, no number is called large or "
    "small, and nothing here says which forecast wins. Where a table is ordered it says by what."
)

QLIKE_TWICE_NOTE: Final = (
    "**Every QLIKE figure appears twice.** `qlike` is the stored column; `qlike_ex_near_zero` is "
    "the same column with the five near-zero target days of docs/P3_ANALYSIS_VALIDITY.md §1.4 set "
    "to NaN (two on CAC, two on TWSE, one on NKX — listed in docs/P3_QLIKE_LEVERAGE.md). The two "
    "differ only on those three assets; on the other eight they are the same numbers."
)


def fit_diagnostics_markdown(frame: pd.DataFrame) -> str:
    wide = frame.pivot(index="model", columns="asset", values="rate")
    wide = wide.reindex(columns=sorted(wide.columns))
    lines = [
        "| model | " + " | ".join(str(c) for c in wide.columns) + " |",
        "|---|" + "---|" * len(wide.columns),
    ]
    for model, row in wide.iterrows():
        lines.append(f"| `{model}` | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_dm(inputs: Inputs, out: Outputs, docs: Path) -> None:
    long = out.frames["P3_DM"]
    summary = out.frames["P3_DM_SUMMARY"]
    changes = out.frames["P3_DM_CHANGES"]
    nu = out.frames["P3_DM_NU_BOUND"]
    parts: list[str] = []
    parts.append("# P3 — Diebold-Mariano matrices\n")
    parts.append(
        "Per asset and per loss, every one of the 78 model pairs tested for equal expected loss on "
        "the differential `d_t = L_i,t - L_j,t` over J2's pairwise-complete sample, at three rungs "
        "of a HAC bandwidth ladder.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_DM.csv` (every pair, every rung), `docs/P3_DM_SUMMARY.csv`, "
                    "`docs/P3_DM_CHANGES.csv`, `docs/P3_DM_NU_BOUND.csv`",
                )
            ],
        )
    )
    parts.append(
        "\n## The test\n\n"
        "- **Statistic.** `S* = f · d̄ / sqrt(Omega_hat / n)` with the Harvey-Leybourne-Newbold "
        "(1997) factor "
        "`f = sqrt((n + 1 - 2h + h(h-1)/n) / n)` at the forecast horizon `h = 1`, i.e. `f = "
        "sqrt((n-1)/n)` "
        "(0.99990 at n = 4,904; 0.99982 at n = 2,791), compared against Student's `t_{n-1}`, "
        "two-sided. "
        "The factor is computed with the forecast horizon, not with the bandwidth: the bandwidth "
        "is a "
        "property of the estimator, not a count of autocovariances the differential is assumed to "
        "carry.\n"
        "- **Long-run variance `Omega_hat`.** Bartlett kernel throughout, so the three rungs "
        "differ only in the "
        "bandwidth and in whether the series is pre-whitened:\n"
        "  - `fixed` — J2's rule `floor(4 (n/100)^(2/9))` (bandwidth 9 on the equity series, 8 on "
        "crypto, as a Bartlett bandwidth L+1), **no pre-whitening**. This is exactly the estimator "
        "behind J2's standard errors, which J2 measured to recover 99 % / 94 % / 60 % of a known "
        "AR(1) long-run variance at rho = 0 / 0.5 / 0.9 (`tests/test_analysis.py::TestHAC`).\n"
        "  - `auto` — **Andrews-Monahan (1992) pre-whitening**: an AR(1) is fitted to the demeaned "
        "differential (|rho| capped at 0.97), the kernel is applied to its residuals at the "
        "**Andrews (1991) AR(1) plug-in bandwidth** `1.1447 (alpha(1) n)^(1/3)` computed on those "
        "residuals, and the result is recoloured by `1/(1-rho)²`. `tests/test_inference_hac.py` "
        "recovers the same AR(1) truth within 5 % at rho = 0 / 0.5 / 0.9 / 0.95.\n"
        "  - `twice_auto` — pre-whitened, at twice the automatic bandwidth.\n"
        "- **Sample.** Each pair runs on the origins where both losses are finite — J2's "
        "pairwise-complete matrix, read from `docs/P3_PAIRWISE_COMPLETE.csv` and asserted equal to "
        "the `n` every pair actually used "
        f"({out.notes.get('dm_pairs_checked_against_j2', 0)} (asset, loss) matrices checked; "
        f"`qlike_ex_near_zero` is new here and has no J2 counterpart). `n` is printed inside every "
        f"cell.\n"
        "- **Diagnostics per pair.** `rho1`, the first-order autocorrelation of the differential; "
        "`n_eff = n (1-rho1)/(1+rho1)`, the AR(1)-equivalent number of independent observations; "
        "the bandwidth chosen; whether the 0.97 cap bound.\n"
        "- **Orientation.** Entry (row `i`, column `j`) tests `L_i - L_j`: a **positive** "
        "statistic means the row model's loss was the larger. The matrix is antisymmetric in the "
        "statistic and symmetric in `p` and `n`.\n"
        "- **These p-values are uncorrected for multiplicity.** 78 pairs per matrix, 3.9 expected "
        "significant at 5 % by chance under the null. They are descriptive; the model confidence "
        "set (docs/P3_MCS.md) is the multiple-comparison instrument. No Bonferroni is applied and "
        "none should be read in.\n"
    )
    parts.append("\n" + QLIKE_TWICE_NOTE + "\n")
    parts.append(
        "\n## Pairs that are not independent by construction\n\n"
        "Where D-032's bound `nu ≤ 50` binds, `garch11_t`'s Student-t is at nu = 50 and the config "
        "collapses "
        "toward `garch11`; the `garch11`/`garch11_t` entry is then a comparison of near-identical "
        "forecasts "
        "on those origins. Per asset, the share of `garch11_t` scheduled fits whose recovered nu "
        "(from the "
        "stored tail quantiles, docs/P3_ANALYSIS_ASSUMPTIONS.md §3) sits on the bound:\n"
    )
    parts.append(
        _table(
            nu,
            {
                "share_at_bound_of_fits": lambda v: f"{100 * v:.1f}%",
                "share_at_bound_of_student_t": lambda v: f"{100 * v:.1f}%",
            },
        )
        + "\n"
    )
    parts.append(
        "\n## Fit diagnostics carried beside the tables\n\n"
        + fit_diagnostics_markdown(out.frames["P3_FIT_DIAGNOSTICS"])
        + "\n"
    )
    parts.append(
        "\n`garch11`, `garch11_t`, `gjr`: EWMA fallbacks per scheduled fit, from the store's "
        "`fit_status`. "
        "`autoarima`: the share of fits on which scipy's optimiser returned a non-zero status "
        "(2,334 of 2,366 = 98.6 % over the grid, docs/P3_INSTRUMENTATION_GAP.md §3.1), from J1's "
        "re-fit "
        "probe where its parquet is present. Every other config reads `not instrumented` — never "
        "`0`, "
        "never `nan`.\n"
    )
    parts.append("\n## Significant pairs at 5 %, by rung\n\n")
    parts.append(
        "Of 78 pairs; 3.9 expected by chance under the null of no difference anywhere. `changed` "
        "counts "
        "pairs whose 5 % verdict differs between the two rungs named.\n\n"
    )
    wide = summary.pivot(
        index=["asset", "loss"], columns="rung", values="significant_5pct"
    ).reset_index()
    wide = wide.merge(
        changes[
            [
                "asset",
                "loss",
                "changed_fixed_vs_auto",
                "changed_auto_vs_twice_auto",
                "changed_fixed_vs_twice_auto",
            ]
        ],
        on=["asset", "loss"],
    )
    wide = wide[
        [
            "asset",
            "loss",
            "fixed",
            "auto",
            "twice_auto",
            "changed_fixed_vs_auto",
            "changed_auto_vs_twice_auto",
            "changed_fixed_vs_twice_auto",
        ]
    ]
    parts.append(_table(wide) + "\n")
    totals = {
        r: int(summary.loc[summary["rung"] == r, "significant_5pct"].sum()) for r, _ in HAC_LADDER
    }
    pairs_total = int(summary.loc[summary["rung"] == "auto", "pairs"].sum())
    parts.append(
        f"\nOver all {pairs_total:,} (asset, loss, pair) tests: significant at 5 % on "
        f"**{totals['fixed']:,}** "
        f"at the fixed rule, **{totals['auto']:,}** at the automatic bandwidth, "
        f"**{totals['twice_auto']:,}** at twice it; "
        f"{int(changes['changed_fixed_vs_auto'].sum()):,} pairs change verdict between the fixed "
        f"rule and the automatic "
        f"bandwidth, {int(changes['changed_auto_vs_twice_auto'].sum()):,} between the automatic "
        f"bandwidth and twice it.\n"
    )
    parts.append(
        "\n## Bandwidth, persistence and effective sample size, by asset x loss (rung `auto`)\n\n"
    )
    auto = summary.loc[
        summary["rung"] == "auto",
        [
            "asset",
            "loss",
            "n_min",
            "n_max",
            "bandwidth_min",
            "bandwidth_median",
            "bandwidth_max",
            "rho1_min",
            "rho1_median",
            "rho1_max",
            "n_eff_min",
            "n_eff_median",
            "n_eff_max",
            "rho_capped",
            "p_exactly_0_or_1",
            "nonfinite_statistic",
        ],
    ]
    parts.append(
        _table(
            auto,
            {
                "n_min": lambda v: str(int(v)),
                "n_max": lambda v: str(int(v)),
                "rho_capped": lambda v: str(int(v)),
                "p_exactly_0_or_1": lambda v: str(int(v)),
                "nonfinite_statistic": lambda v: str(int(v)),
            },
        )
        + "\n"
    )
    fixed = summary.loc[summary["rung"] == "fixed", ["asset", "loss", "bandwidth_median"]].rename(
        columns={"bandwidth_median": "fixed_bandwidth"}
    )
    parts.append(
        "\nThe fixed rung's Bartlett bandwidth (`L + 1`) per asset x loss: "
        + ", ".join(
            f"{a} {int(b)}" for a, b in fixed.groupby("asset")["fixed_bandwidth"].first().items()
        )
        + ".\n"
    )
    parts.append("\n## The matrices — rung `auto` (pre-whitened, Andrews bandwidth)\n\n")
    parts.append(
        "Cell: `statistic (p) n`. The `fixed` and `twice_auto` rungs are in `docs/P3_DM.csv`.\n"
    )
    for asset in inputs.assets:
        parts.append(f"\n### {asset}\n")
        for loss in LOSSES:
            row = summary.loc[(summary["asset"] == asset) & (summary["loss"] == loss)].set_index(
                "rung"
            )
            heading = LOSS_HEADINGS.get(loss, loss)
            auto_row, fixed_row, twice_row = (
                row.loc["auto"],
                row.loc["fixed"],
                row.loc["twice_auto"],
            )
            bw = " / ".join(
                _fmt(_num(auto_row[k]), 3)
                for k in ("bandwidth_min", "bandwidth_median", "bandwidth_max")
            )
            rho = " / ".join(
                _fmt(_num(auto_row[k]), 3) for k in ("rho1_min", "rho1_median", "rho1_max")
            )
            eff = " / ".join(
                _fmt(_num(auto_row[k]), 4) for k in ("n_eff_min", "n_eff_median", "n_eff_max")
            )
            sig = (
                f"fixed {_count(fixed_row['significant_5pct'])}, "
                f"auto {_count(auto_row['significant_5pct'])}, "
                f"twice {_count(twice_row['significant_5pct'])} of {_count(auto_row['pairs'])}"
            )
            parts.append(
                f"\n#### {asset} — `{loss}` ({heading}) — bandwidth {bw} (min / median / max over "
                f"pairs), "
                f"rho1 {rho}, n_eff {eff}; significant at 5 %: {sig}\n\n"
            )
            parts.append(dm_matrix_markdown(long, asset, loss, HEADLINE_RUNG) + "\n")
    (docs / "P3_DM.md").write_text("\n".join(parts), encoding="utf-8")


def write_mcs(inputs: Inputs, out: Outputs, docs: Path) -> None:
    records = out.frames["P3_MCS"]
    blocks = out.frames["P3_MCS_BLOCKS"]
    sensitivity = out.frames["P3_MCS_SENSITIVITY"]
    parts: list[str] = []
    parts.append("# P3 — model confidence sets\n")
    parts.append(
        "Per asset and per loss, the Model Confidence Set of Hansen, Lunde & Nason (2011) over the "
        "13 configs: the surviving set at alpha = 0.10 and alpha = 0.25, every model's MCS p-value "
        "and the step at which it was eliminated, for two statistics, with the block length the "
        "bootstrap used and what it was chosen against.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_MCS.csv` (one row per asset x loss x statistic x variant x model), "
                    "`docs/P3_MCS_BLOCKS.csv`, `docs/P3_MCS_SENSITIVITY.csv`",
                )
            ],
        )
    )
    parts.append(
        "\n## The procedure\n\n"
        f"- **Bootstrap.** Moving block (Künsch 1989), non-circular, **B = "
        f"{out.manifest['mcs']['n_boot']:,}**, one index sequence per resample shared by all "
        f"models so cross-sectional dependence survives. Seeds derived from the master seed "
        f"{MASTER_SEED} and the run's name; every seed is in the manifest.\n"
        "- **Block length.** The Politis-White (2004; Patton, Politis & White 2009) automatic rule "
        "on every pairwise loss differential of the listwise-complete sample, the largest taken "
        "and rounded up (`volbench.inference.default_block_length`). Flagged below where it is 1 "
        "or exceeds n/4, and read against the persistence of the series it was chosen for.\n"
        "- **Sample.** Listwise-complete: the origins where all 13 losses are finite (one index "
        "sequence must serve every model). `n_dropped` is reported.\n"
        "- **Statistics.** `range`: `T_R = max_{i,j} |t_ij|`, elimination `arg max_i sup_j t_ij` "
        "(HLN 2011 §3.1.2). `semi_quadratic`: `T_SQ = Σ_{i<j} t_ij²` (HLN 2003; Hansen & Lunde "
        "2005), the same `t_ij` and the same elimination rule, so the two differ only in how the "
        "evidence across pairs is pooled.\n"
        "- **p-values.** MCS p-values are the cumulative maxima of the step p-values along the "
        "elimination sequence (HLN 2011, Definition 4), the last survivor at 1. The set at level "
        "alpha is `{i : p_i ≥ alpha}`; both levels are read off the same run.\n"
        "- **Elimination step.** 1 = eliminated first; 13 = the last survivor.\n"
    )
    parts.append("\n" + QLIKE_TWICE_NOTE + "\n")
    parts.append(
        "\n## Fit diagnostics carried beside the tables\n\n"
        + fit_diagnostics_markdown(out.frames["P3_FIT_DIAGNOSTICS"])
        + "\n"
    )
    parts.append("\n## Block lengths, against the persistence they were chosen for\n\n")
    parts.append(
        "`block` is the length used. `pw pairs` is the Politis-White value over the 78 "
        "differentials (min / median / max); `rho1 pairs` their first-order autocorrelations, with "
        "the pair carrying the largest; `AR(1)-implied` is the block length the same rule returns "
        "for an AR(1) at that rho1 and this n (`volbench.inference.ar1_block_length`); `pw series` "
        "and `rho1 series` are the same two quantities on the 13 loss series themselves. `flag` "
        "marks a block of 1, a block above n/4, or a block below the AR(1)-implied length for the "
        "most persistent differential.\n\n"
    )
    head = blocks.loc[blocks["variant"] == "headline"].copy()
    head["flag"] = ["1" if a else "" for a in head["block_is_1"]]
    head["flag"] = [
        "; ".join(
            x
            for x in (
                ("block = 1" if r.block_is_1 else ""),
                ("> n/4" if r.block_exceeds_n_over_4 else ""),
                ("< AR(1)-implied" if r.block_below_ar1_implied else ""),
            )
            if x
        )
        for r in head.itertuples(index=False)
    ]
    head["pw pairs"] = [
        f"{_fmt(a, 3)} / {_fmt(b, 3)} / {_fmt(c, 3)}"
        for a, b, c in zip(
            head["pw_pairs_min"], head["pw_pairs_median"], head["pw_pairs_max"], strict=True
        )
    ]
    head["rho1 pairs"] = [
        f"{_fmt(a, 3)} / {_fmt(b, 3)} / {_fmt(c, 3)}"
        for a, b, c in zip(
            head["rho1_pairs_min"], head["rho1_pairs_median"], head["rho1_pairs_max"], strict=True
        )
    ]
    head["pw series"] = [
        f"{_fmt(a, 3)} / {_fmt(b, 3)} / {_fmt(c, 3)}"
        for a, b, c in zip(
            head["pw_series_min"], head["pw_series_median"], head["pw_series_max"], strict=True
        )
    ]
    head["rho1 series"] = [
        f"{_fmt(a, 3)} / {_fmt(b, 3)} / {_fmt(c, 3)}"
        for a, b, c in zip(
            head["rho1_series_min"],
            head["rho1_series_median"],
            head["rho1_series_max"],
            strict=True,
        )
    ]
    shown = head[
        [
            "asset",
            "loss",
            "n",
            "block_length",
            "pw pairs",
            "rho1 pairs",
            "rho1_pairs_max_pair",
            "ar1_implied_block_at_max_rho1",
            "pw series",
            "rho1 series",
            "flag",
        ]
    ].rename(
        columns={
            "block_length": "block",
            "rho1_pairs_max_pair": "max-rho1 pair",
            "ar1_implied_block_at_max_rho1": "AR(1)-implied",
        }
    )
    parts.append(_table(shown, {"n": lambda v: str(int(v)), "block": lambda v: str(int(v))}) + "\n")
    zero_p = records.loc[(records["variant"] == "headline") & (records["mcs_p_value"] == 0.0)]
    parts.append(
        f"\nMCS p-values exactly 0 (no resample reached the observed statistic at any step before "
        f"the model's elimination): **{len(zero_p)}** of "
        f"{int((records['variant'] == 'headline').sum())} headline (asset, loss, statistic, model) "
        f"entries, over {zero_p.groupby(['asset', 'loss', 'statistic']).ngroups} runs.\n"
    )
    flagged = head.loc[head["block_is_1"] | head["block_exceeds_n_over_4"]]
    parts.append(
        f"\nBlocks equal to 1: **{int(head['block_is_1'].sum())}**; above n/4: "
        f"**{int(head['block_exceeds_n_over_4'].sum())}**; below the AR(1)-implied length for the "
        f"most persistent differential: **{int(head['block_below_ar1_implied'].sum())}** of "
        f"{len(head)}.\n"
    )
    if not flagged.empty:
        parts.append(
            "Flagged as 1 or above n/4: "
            + ", ".join(
                f"{r.asset}/{r.loss} ({_count(r.block_length)}, n={_count(r.n)})"
                for r in flagged.itertuples(index=False)
            )
            + ".\n"
        )
    parts.append("\n## Surviving sets\n\n")
    headline = records.loc[records["variant"] == "headline"]
    rows: list[dict[str, Any]] = []
    for (asset, loss, statistic), group in headline.groupby(
        ["asset", "loss", "statistic"], sort=True
    ):
        rows.append(
            {
                "asset": asset,
                "loss": loss,
                "statistic": statistic,
                "n": int(group["n"].iloc[0]),
                "block": int(group["block_length"].iloc[0]),
                "alpha = 0.10": survivors(group, 0.10) or "—",
                "alpha = 0.25": survivors(group, 0.25) or "—",
                "|M| 0.10": int(group["in_mcs_0.1"].sum()),
                "|M| 0.25": int(group["in_mcs_0.25"].sum()),
            }
        )
    parts.append(_table(pd.DataFrame(rows)) + "\n")
    parts.append("\n## Elimination order and MCS p-values, per asset x loss\n\n")
    parts.append(
        "Models in alphabetical order. `p R` / `step R` under the range statistic, `p SQ` / `step "
        "SQ` under the semi-quadratic; `n` and the block length are shared.\n"
    )
    for asset in inputs.assets:
        parts.append(f"\n### {asset}\n")
        for loss in LOSSES:
            r = headline.loc[
                (headline["asset"] == asset)
                & (headline["loss"] == loss)
                & (headline["statistic"] == "range")
            ].set_index("model")
            q = headline.loc[
                (headline["asset"] == asset)
                & (headline["loss"] == loss)
                & (headline["statistic"] == "semi_quadratic")
            ].set_index("model")
            parts.append(
                f"\n#### {asset} — `{loss}` (n = {int(r['n'].iloc[0]):,}, dropped "
                f"{int(r['n_dropped'].iloc[0])}, block {int(r['block_length'].iloc[0])})\n\n"
            )
            table = pd.DataFrame(
                {
                    "model": [f"`{m}`" for m in r.index],
                    "mean loss": [_fmt(v, 6) for v in r["mean_loss"]],
                    "p R": [_fmt(v, 4) for v in r["mcs_p_value"]],
                    "step R": [int(v) for v in r["eliminated_at_step"]],
                    "in 0.10 R": ["yes" if v else "no" for v in r["in_mcs_0.1"]],
                    "in 0.25 R": ["yes" if v else "no" for v in r["in_mcs_0.25"]],
                    "p SQ": [_fmt(v, 4) for v in q.loc[r.index, "mcs_p_value"]],
                    "step SQ": [int(v) for v in q.loc[r.index, "eliminated_at_step"]],
                    "in 0.10 SQ": ["yes" if v else "no" for v in q.loc[r.index, "in_mcs_0.1"]],
                    "in 0.25 SQ": ["yes" if v else "no" for v in q.loc[r.index, "in_mcs_0.25"]],
                }
            )
            parts.append(frame_markdown(table) + "\n")
    parts.append("\n## Sensitivity: BTC-USD without `garch11_t`, HSI without `gjr`\n\n")
    parts.append(
        "Reported alongside the headline, never instead of it. BTC-USD `garch11_t` is 15/133 EWMA "
        "fallback (11.28 %) and HSI `gjr` 14/230 (6.09 %) by construction "
        "(docs/P3_CONVERGENCE_FORENSICS.md T8). `changed` compares the sensitivity survivors "
        "against the headline survivors with the dropped model removed; the block length can move "
        "because the rule takes a maximum over fewer pairs.\n\n"
    )
    shown = sensitivity[
        [
            "asset",
            "dropped",
            "loss",
            "statistic",
            "block_length_headline",
            "block_length_sensitivity",
            "survivors_headline_0.1",
            "survivors_sensitivity_0.1",
            "changed_0.1",
            "survivors_headline_0.25",
            "survivors_sensitivity_0.25",
            "changed_0.25",
        ]
    ]
    parts.append(_table(shown) + "\n")
    parts.append(
        f"\nSurviving sets that change: **{int(sensitivity['changed_0.1'].sum())}** of "
        f"{len(sensitivity)} at alpha = 0.10, **{int(sensitivity['changed_0.25'].sum())}** at "
        f"alpha = 0.25.\n"
    )
    (docs / "P3_MCS.md").write_text("\n".join(parts), encoding="utf-8")


def write_backtests(inputs: Inputs, out: Outputs, docs: Path) -> None:
    frame = out.frames["P3_BACKTESTS"]
    parts: list[str] = []
    parts.append("# P3 — VaR / ES backtests\n")
    parts.append(
        "Per asset x model at each evaluated level: Kupiec's (1995) unconditional-coverage test, "
        "Christoffersen's (1998) independence and conditional-coverage tests, observed against "
        "expected exceedances, the longest run of consecutive exceedances, and the mean FZ0 joint "
        "(VaR, ES) loss.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(_provenance(inputs, [("Machine-readable", "`docs/P3_BACKTESTS.csv`")]))
    parts.append(
        "\n## The tests\n\n"
        f"- **Levels** {', '.join(f'{lv:g}' for lv in inputs.levels)}, lower tail, return-side "
        f"sign convention; hit `H_t = 1{{r_t < VaR_t}}`, strict, as the evaluator stored it.\n"
        "- **Sample.** A row is excluded only where its hit is NaN (no forecast was made — NKX's "
        "first 21 origins on the eight variance-fed configs); a row that lost only its QLIKE keeps "
        "its hit. `n` and `n_dropped` are printed.\n"
        "- **Kupiec** `LR_uc ~ chi^2(1)`; **Christoffersen** `LR_ind ~ chi^2(1)` against a "
        "first-order Markov alternative and `LR_cc = LR_uc + LR_ind ~ chi^2(2)`, all conditional "
        "on the first observation; transitions counted only between rows adjacent in origin "
        "order.\n"
        "- **Runs.** `longest run` is the longest sequence of consecutive exceedances; `runs ≥ 2` "
        "counts the maximal runs of length two or more; `n11` is Christoffersen's "
        "exceedance-after-exceedance count.\n"
        "- **FZ0** is Patton, Ziegel & Chen (2019) eq. 6 on the cell's own stored VaR and ES, "
        "averaged over the rows with a forecast.\n"
        "- **These p-values are uncorrected across 13 models x 11 assets x 3 levels** (429 tests "
        "per statistic). At 5 %, 21 rejections per statistic are expected by chance under the null "
        "everywhere.\n"
    )
    for statistic in ("kupiec_p", "ind_p", "cc_p"):
        parts.append(
            f"- Rejections at 5 % on `{statistic}`: **{int((frame[statistic] < 0.05).sum())}** of "
            f"{len(frame)}; p exactly 0: {int((frame[statistic] == 0.0).sum())}; p exactly 1: "
            f"{int((frame[statistic] == 1.0).sum())}.\n"
        )
    parts.append(
        f"- Small-sample flags (expected exceedances below {backtests.MIN_EXPECTED_HITS:g}): "
        f"**{int(frame['small_sample'].sum())}** of {len(frame)}.\n"
    )
    for asset in inputs.assets:
        parts.append(f"\n## {asset}\n")
        for level in inputs.levels:
            block = frame.loc[(frame["asset"] == asset) & (frame["level"] == level)]
            parts.append(
                f"\n### {asset} — level {level:g} (n = {int(block['n'].iloc[0]):,}, expected "
                f"exceedances {block['expected_hits'].iloc[0]:.1f})\n\n"
            )
            table = pd.DataFrame(
                {
                    "model": [f"`{m}`" for m in block["model"]],
                    "n": [int(v) for v in block["n"]],
                    "hits": [int(v) for v in block["n_hits"]],
                    "rate": [f"{v:.4f}" for v in block["hit_rate"]],
                    "Kupiec LR (p)": [
                        f"{a:.2f} ({b:.3f})"
                        for a, b in zip(block["kupiec_lr"], block["kupiec_p"], strict=True)
                    ],
                    "IND LR (p)": [
                        f"{a:.2f} ({b:.3f})"
                        for a, b in zip(block["ind_lr"], block["ind_p"], strict=True)
                    ],
                    "CC LR (p)": [
                        f"{a:.2f} ({b:.3f})"
                        for a, b in zip(block["cc_lr"], block["cc_p"], strict=True)
                    ],
                    "n11": [int(v) for v in block["n11"]],
                    "longest run": [int(v) for v in block["longest_run"]],
                    "runs ≥ 2": [int(v) for v in block["runs_of_2_or_more"]],
                    "mean FZ0 (n)": [
                        f"{a:.4f} ({int(b)})"
                        for a, b in zip(block["fz0_mean"], block["fz0_n"], strict=True)
                    ],
                }
            )
            parts.append(frame_markdown(table) + "\n")
    (docs / "P3_BACKTESTS.md").write_text("\n".join(parts), encoding="utf-8")


def write_econ(inputs: Inputs, out: Outputs, docs: Path) -> None:
    frame = out.frames["P3_ECON"]
    parts: list[str] = []
    parts.append("# P3 — economic value\n")
    parts.append(
        "The volatility-targeting backtest of `volbench.econ` (D-029) per asset x model at four "
        "transaction-cost levels, and a moving-block-bootstrap interval on each model's Sharpe "
        "ratio minus the `garch11` baseline's.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(_provenance(inputs, [("Machine-readable", "`docs/P3_ECON.csv`")]))
    parts.append(
        "\n## The backtest\n\n"
        f"- `position_t = min(target / forecast_vol_t, {econ.DEFAULT_LEVERAGE_CAP:g})` with a "
        f"{100 * econ.DEFAULT_TARGET_VOL:.0f} % annualised target, held into `t+1`, earning the "
        f"simple return of the stored log return; an unusable forecast sizes to zero (`n_flat`).\n"
        f"- **Costs** {', '.join(f'{c:g}' for c in COST_BPS)} bps of turnover `|position_t - "
        f"position_{{t-1}}|`, the first day charged against a flat book.\n"
        "- **Annualisation per asset class:** 252 on the nine equity series, 365 on BTC-USD and "
        "ETH-USD (`econ.periods_per_year_for`); the target, the return, the volatility and the "
        "Sharpe all scale with it.\n"
        "- Annualised return is geometric; the Sharpe is the arithmetic mean over the `ddof = 1` "
        "standard deviation of per-period net returns, annualised, risk-free rate 0. `gross` is "
        "before costs.\n"
        f"- **Sharpe difference.** `dS = S(model) - S(garch11)` on the same origin axis, with a "
        f"**percentile moving-block bootstrap** (B = {out.manifest['econ']['n_boot']:,}): one "
        f"index sequence resamples both net-return series together, each resample's Sharpe is its "
        f"mean over its `ddof = 1` standard deviation, and the 2.5 / 97.5 percentiles of the "
        f"resampled differences are the interval. Block length: the Politis-White rule's larger "
        f"value over `r_m - r_b` and `r_m² - r_b²`, rounded up. Seeds are in the manifest. `CI "
        f"excludes 0` is a mechanical reading of the two bounds.\n"
    )
    parts.append(
        f"- Cells whose levered book was wiped out (`ruined`): **{int(frame['ruined'].sum())}** of "
        f"{len(frame)}.\n"
    )
    parts.append(
        f"- Intervals excluding zero: **{int(frame['ci_excludes_zero'].sum())}** of "
        f"{int((frame['model'] != BASELINE).sum())} model x asset x cost cells.\n"
    )
    for asset in inputs.assets:
        parts.append(f"\n## {asset} ({econ.periods_per_year_for(asset):g} periods per year)\n")
        for cost in COST_BPS:
            block = frame.loc[(frame["asset"] == asset) & (frame["cost_bps"] == cost)]
            parts.append(
                f"\n### {asset} — {cost:g} bps (n = {int(block['n_periods'].iloc[0]):,})\n\n"
            )
            table = pd.DataFrame(
                {
                    "model": [f"`{m}`" for m in block["model"]],
                    "ann. return": [f"{v:+.2%}" for v in block["annual_return"]],
                    "ann. vol": [f"{v:.2%}" for v in block["annual_vol"]],
                    "Sharpe": [f"{v:.3f}" for v in block["sharpe"]],
                    "gross Sharpe": [f"{v:.3f}" for v in block["gross_sharpe"]],
                    "max DD": [f"{v:.2%}" for v in block["max_drawdown"]],
                    "ann. turnover": [f"{v:.2f}" for v in block["annual_turnover"]],
                    "n_flat": [int(v) for v in block["n_flat"]],
                    "ruined": ["yes" if v else "no" for v in block["ruined"]],
                    "dSharpe vs garch11": [f"{v:+.3f}" for v in block["sharpe_diff_vs_baseline"]],
                    "95 % CI": [
                        ("—" if m == BASELINE else f"[{lo:+.3f}, {hi:+.3f}]")
                        for m, lo, hi in zip(
                            block["model"], block["ci_low"], block["ci_high"], strict=True
                        )
                    ],
                    "block": [int(v) for v in block["block_length"]],
                }
            )
            parts.append(frame_markdown(table) + "\n")
    (docs / "P3_ECON.md").write_text("\n".join(parts), encoding="utf-8")


def write_cross_asset(inputs: Inputs, out: Outputs, docs: Path) -> None:
    summary = out.frames["P3_CROSS_ASSET_RANK_SUMMARY"]
    counts = out.frames["P3_CROSS_ASSET_MCS_COUNTS"]
    kendall = out.frames["P3_CROSS_ASSET_KENDALL"]
    parts: list[str] = []
    parts.append("# P3 — cross-asset aggregation, ranks only\n")
    parts.append(
        "The nine equity series are scored against an overnight-plus-range variance target and the "
        "two crypto series against 5-minute realized variance, so no loss is pooled or averaged "
        "across assets anywhere in this file. Every summary is a rank or a count.\n"
    )
    parts.append(
        REPORTED_NOT_INTERPRETED
        + " No line here names an overall best; none can be derived from this file without "
        "adding an assumption it does not make.\n"
    )
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_CROSS_ASSET_RANKS.csv`, `docs/P3_CROSS_ASSET_RANK_SUMMARY.csv`, "
                    "`docs/P3_CROSS_ASSET_MCS_COUNTS.csv`, `docs/P3_CROSS_ASSET_KENDALL.csv`",
                )
            ],
        )
    )
    parts.append(
        "\n## Conventions\n\n"
        "- **Rank** = position of a model's mean loss among the 13 within one asset, 1 = smallest, "
        "on the listwise-complete sample the MCS ran on (docs/P3_MCS.md); ties share the minimum "
        "rank.\n"
        "- **Blocks:** `all` (11 assets), `equity` (CAC, DAX, DIA, HSI, KOSPI, NDX, NKX, SPY, "
        "TWSE), `crypto` (BTC-USD, ETH-USD).\n"
        "- **MCS membership counts:** in how many of the block's assets the model survives at "
        "alpha = 0.10 and 0.25, per statistic (docs/P3_MCS.md, headline runs).\n"
        "- **Kendall's τ** (τ-b) between two rankings of the 13 models within one asset.\n"
    )
    parts.append("\n" + QLIKE_TWICE_NOTE + "\n")
    parts.append("\n## Kendall's τ between the CRPS ranking and the QLIKE ranking, per asset\n\n")
    parts.append(_table(kendall) + "\n")
    parts.append(
        f"\nMinimum over the 11 assets: **{kendall['tau_crps_vs_qlike'].min():.3f}** (`qlike`), "
        f"**{kendall['tau_crps_vs_qlike_ex_near_zero'].min():.3f}** (`qlike_ex_near_zero`); median "
        f"{kendall['tau_crps_vs_qlike'].median():.3f} / "
        f"{kendall['tau_crps_vs_qlike_ex_near_zero'].median():.3f}. Equity block minimum "
        f"{kendall.loc[kendall['block'] == 'equity', 'tau_crps_vs_qlike'].min():.3f}, crypto block "
        f"minimum {kendall.loc[kendall['block'] == 'crypto', 'tau_crps_vs_qlike'].min():.3f}.\n"
    )
    parts.append("\n## Mean rank and rank distribution, per loss\n\n")
    for loss in LOSSES:
        parts.append(f"\n### `{loss}` ({LOSS_HEADINGS.get(loss, loss)})\n\n")
        for block in ("all", "equity", "crypto"):
            sub = summary.loc[(summary["loss"] == loss) & (summary["block"] == block)]
            parts.append(
                f"\n**{block}** ({int(sub['assets'].iloc[0])} assets), models in alphabetical "
                f"order:\n\n"
            )
            table = pd.DataFrame(
                {
                    "model": [f"`{m}`" for m in sub["model"]],
                    "mean rank": [f"{v:.2f}" for v in sub["mean_rank"]],
                    "min": [int(v) for v in sub["rank_min"]],
                    "q25": [f"{v:.1f}" for v in sub["rank_q25"]],
                    "median": [f"{v:.1f}" for v in sub["rank_median"]],
                    "q75": [f"{v:.1f}" for v in sub["rank_q75"]],
                    "max": [int(v) for v in sub["rank_max"]],
                    "ranks by asset": list(sub["ranks"]),
                }
            )
            parts.append(frame_markdown(table) + "\n")
    parts.append("\n## MCS membership counts, per loss\n\n")
    for loss in LOSSES:
        parts.append(f"\n### `{loss}`\n\n")
        sub = counts.loc[counts["loss"] == loss]
        wide = sub.pivot_table(
            index="model",
            columns=["statistic", "block"],
            values=["survives_0.1", "survives_0.25"],
            aggfunc="first",
        )
        keys = [tuple(c) for c in wide.columns.to_list()]
        columns = [f"{s} {b} alpha={a[len('survives_') :]}" for a, s, b in keys]
        flat = pd.DataFrame(wide.to_numpy(), index=wide.index, columns=columns).reset_index()
        flat["model"] = [f"`{m}`" for m in flat["model"]]
        parts.append(frame_markdown(flat.astype({c: int for c in columns})) + "\n")
    (docs / "P3_CROSS_ASSET.md").write_text("\n".join(parts), encoding="utf-8")


def write_crisis(inputs: Inputs, out: Outputs, docs: Path) -> None:
    coverage = out.frames["P3_CRISIS_COVERAGE"]
    table = out.frames["P3_CRISIS"]
    records = out.frames["P3_CRISIS_MCS"]
    blocks = out.frames["P3_CRISIS_BLOCKS"]
    parts: list[str] = []
    parts.append("# P3 — crisis sub-samples on the pre-registered windows\n")
    parts.append(
        "Per asset x regime: each model's mean loss with a pre-whitened HAC standard error, and "
        "the model confidence set inside the window. The headline uses the windows exactly as "
        "`volbench.data.crisis` defines them; a wider GFC definition is reported as a sensitivity, "
        "never as the headline.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_CRISIS.csv` (means), `docs/P3_CRISIS_MCS.csv`, "
                    "`docs/P3_CRISIS_BLOCKS.csv`, `docs/P3_CRISIS_COVERAGE.csv`",
                )
            ],
        )
    )
    parts.append("\n## The windows, as fixed in advance\n\n")
    parts.append(
        "Source: `src/volbench/data/crisis.py::CRISIS_WINDOWS`, verbatim from "
        'docs/research_design.md ("GFC Sep 08-Mar 09 · COVID Feb-Apr 20 · 2022 tightening Jan-Oct '
        '22 · Aug-2024 spike · latest 2025-26 stress window (fixed at grid freeze)"), resolved to '
        "calendar-month boundaries, both ends inclusive. A row belongs to the window containing "
        "its **target** date.\n\n"
    )
    parts.append(
        "| tag | start | end | label | source phrase |\n|---|---|---|---|---|\n"
        + "\n".join(
            f"| `{w.tag}` | {w.start} | {w.end} | {w.label} | {w.source_phrase} |"
            for w in CRISIS_WINDOWS
        )
        + "\n"
    )
    parts.append(
        "\n**`stress_2025_26` is an unset window, not an absence of stress.** "
        + "; ".join(f"`{w.tag}` — {w.label}: {w.blocked_on}" for w in PENDING_WINDOWS)
        + " It is undated in the codebase, tags nothing, and is excluded from the headline. "
        "Thirteen of J2's 38 GARCH fallbacks fall in the span it names "
        "(docs/P3_CONVERGENCE_FORENSICS.md T7); dating it now, after seeing where results fall, "
        "would be selection on the outcome, "
        "so it is not dated here.\n"
    )
    parts.append(
        f"\n**Sensitivity window, never the headline:** `{WIDE_GFC.tag}` = {WIDE_GFC.start} → "
        f"{WIDE_GFC.end} ({WIDE_GFC.source_phrase}). DIA's fallback cluster sits at 2008-07-23 and "
        f"2008-08-21, outside the headline window that opens 2008-09-01; the headline window is "
        f"not moved.\n"
    )
    parts.append(
        "\nEverything outside the four dated windows is `calm`. Standard errors are Bartlett, "
        "Andrews-Monahan pre-whitened, Andrews bandwidth, on the regime's rows in origin order; "
        "`calm` is a union of stretches and the estimator treats the joins as adjacent, as J2's "
        "did across holes.\n"
    )
    parts.append("\n## Coverage\n\n")
    cov = coverage.copy()
    cov["first_date"] = [
        ("" if pd.isna(v) else str(pd.Timestamp(v).date())) for v in cov["first_date"]
    ]
    cov["last_date"] = [
        ("" if pd.isna(v) else str(pd.Timestamp(v).date())) for v in cov["last_date"]
    ]
    parts.append(_table(cov) + "\n")
    parts.append("\n## Block lengths inside the windows\n\n")
    parts.append(
        "Small windows give the Politis-White rule little to work with; a block of 1 or a block "
        "above n/4 is flagged, as is a block below the AR(1)-implied length for the most "
        "persistent differential.\n\n"
    )
    shown = blocks[
        [
            "asset",
            "variant",
            "loss",
            "n",
            "block_length",
            "block_is_1",
            "block_exceeds_n_over_4",
            "block_below_ar1_implied",
            "rho1_pairs_max",
            "ar1_implied_block_at_max_rho1",
        ]
    ].rename(columns={"variant": "regime"})
    parts.append(
        _table(shown, {"n": lambda v: str(int(v)), "block_length": lambda v: str(int(v))}) + "\n"
    )
    parts.append(
        f"\nBlocks equal to 1: **{int(blocks['block_is_1'].sum())}**; above n/4: "
        f"**{int(blocks['block_exceeds_n_over_4'].sum())}**; below the AR(1)-implied length: "
        f"**{int(blocks['block_below_ar1_implied'].sum())}** of {len(blocks)}.\n"
    )
    primary = ("crps", "log_score", "pinball_avg", "qlike", QLIKE_EX, "fz0_avg")
    for kind, title in (
        ("headline", "Headline — the pre-registered windows"),
        ("sensitivity", "Sensitivity — the wider GFC definition (not the headline)"),
    ):
        parts.append(f"\n## {title}\n")
        tags = (
            [w.tag for w in CRISIS_WINDOWS] + [CALM_TAG] if kind == "headline" else [WIDE_GFC.tag]
        )
        for asset in inputs.assets:
            for tag in tags:
                sub = table.loc[(table["asset"] == asset) & (table["regime"] == tag)]
                if sub.empty:
                    continue
                days = int(
                    coverage.loc[
                        (coverage["asset"] == asset) & (coverage["regime"] == tag), "days"
                    ].iloc[0]
                )
                parts.append(f"\n### {asset} — `{tag}` ({days} target days)\n\n")
                wide = sub.pivot(index="model", columns="loss", values="mean")
                ses = sub.pivot(index="model", columns="loss", values="se_auto")
                ns = sub.pivot(index="model", columns="loss", values="n")
                rows: list[dict[str, Any]] = []
                for model in wide.index:
                    row: dict[str, Any] = {"model": f"`{model}`"}
                    for loss in primary:
                        row[LOSS_HEADINGS.get(loss, loss)] = (
                            f"{wide.loc[model, loss]:.6g} ({ses.loc[model, loss]:.3g}) "
                            f"n={int(ns.loc[model, loss])}"
                        )
                    rows.append(row)
                parts.append(frame_markdown(pd.DataFrame(rows)) + "\n")
                mcs = records.loc[(records["asset"] == asset) & (records["regime"] == tag)]
                if not mcs.empty:
                    lines = [
                        "",
                        "| loss | n | block | alpha = 0.10 | alpha = 0.25 | elimination order "
                        "(first out → survivor) |",
                        "|---|---:|---:|---|---|---|",
                    ]
                    for loss in LOSSES:
                        group = mcs.loc[mcs["loss"] == loss].sort_values("eliminated_at_step")
                        if group.empty:
                            lines.append(
                                f"| `{loss}` | — | — | — | — | fewer than 3 complete origins |"
                            )
                            continue
                        order = " → ".join(f"`{m}`" for m in group["model"])
                        lines.append(
                            f"| `{loss}` | {int(group['n'].iloc[0])} | "
                            f"{int(group['block_length'].iloc[0])} | "
                            f"{survivors(group, 0.10) or '—'} | {survivors(group, 0.25) or '—'} | "
                            f"{order} |"
                        )
                    parts.append("\n".join(lines) + "\n")
    (docs / "P3_CRISIS.md").write_text("\n".join(parts), encoding="utf-8")


def write_boundary(inputs: Inputs, out: Outputs, docs: Path) -> None:
    notes = out.notes["boundary"]
    ratios = out.frames["P3_BOUNDARY_RATIOS"]
    fits = out.frames["P3_BOUNDARY_FITS"]
    has_ref = bool(notes["reference_parquet_present"])
    parts: list[str] = []
    parts.append("# P3 — boundary-pinned GARCH fits against EWMA, from the store\n")
    parts.append(
        "BTC-USD's `garch11_t` and `garch11` variance forecasts against BTC-USD's `ewma` forecasts "
        "at the same origins, split by whether the fit governing each origin sits on the "
        "persistence boundary. Both series are stored; no model was run.\n"
    )
    parts.append(
        REPORTED_NOT_INTERPRETED + " The numbers are reported; no conclusion is drawn from them "
        "here.\n"
    )
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_BOUNDARY_FITS.csv` (one row per scheduled fit), "
                    "`docs/P3_BOUNDARY_RATIOS.csv`",
                )
            ],
        )
    )
    parts.append(
        "\n## Where alpha+beta comes from, and the cross-check\n\n"
        "**Primary source: J2's re-fit** (`docs/P3_CONVERGENCE_FITS.parquet`, uncommitted, "
        "docs/P3_CONVERGENCE_FORENSICS.md §0), which reproduced every one of the store's "
        "7,101 `fit_status` strings before it was used"
        + (
            " — on disk for this run."
            if has_ref
            else " — **not on disk for this run**, so the store recursion below is the only source."
        )
        + "\n\n"
        "**Cross-check from the store alone.** Within one refit block the forecast is "
        "re-conditioned daily without re-estimation, so consecutive rows should obey "
        "`h_{t+1} = omega + alpha r_t^2 + beta h_t` with the block's fixed parameters (`h` the "
        "stored `forecast_var`, `r` the stored `realized_return`). Two things are computed per "
        "block: the least-squares solution of that recursion from the stored rows alone "
        "(`_store` columns), and — when the re-fit is on disk — the largest relative one-step "
        "residual at the re-fit's own parameters (`ref_max_rel_residual`). The re-conditioning "
        "is `arch`'s `ARCHModel.fix` re-filtering the *current 500-day window* "
        "(`models/garch.py`), so the filter restarts from that window's initial condition each "
        "day; the one-step recursion between consecutive forecasts holds only once "
        "`beta^window` has decayed, and a block is called reproduced when the residual is at or "
        f"below {RECURSION_TOL:g}. The least-squares read-back is ill-conditioned where alpha is 0 "
        "and `h` is nearly constant, so its agreement with the re-fit is quoted on the blocks "
        "reproduced at 1e-12 only. "
        "An EWMA fallback block obeys the same recursion with omega = 0 and alpha + beta = 1 by "
        "construction.\n"
    )
    lines = [
        "\n## Counts\n",
        "| config | source of alpha+beta | fits | fallback | converged | alpha+beta within 1e-6 of "
        "1 "
        "(converged) | within 1e-4 | within 1e-2 | unresolved | store recursion reproduced at "
        "1e-12 / 1e-8 / 1e-4 (of converged) | store recursion max residual | blocks too short |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for config in ("garch11_t", "garch11"):
        e = notes[config]
        lines.append(
            f"| `{config}` | {e['alpha_plus_beta_source']} | {e['fits']} | {e['fallback_fits']} | "
            f"{e['converged_fits']} | **{e['converged_pinned_1e-6']}** | "
            f"{e['converged_pinned_1e-4']} | "
            f"{e['converged_pinned_1e-2']} | {e['converged_unresolved']} | "
            f"{e['store_recursion_reproduced_1e-12']} / {e['store_recursion_reproduced_1e-8']} / "
            f"{e['store_recursion_reproduced_1e-4']} | {e['store_recursion_max_rel_residual']:.2e} "
            f"| "
            f"{e['blocks_too_short_to_solve']} |"
        )
    parts.append("\n".join(lines) + "\n")
    if has_ref:
        lines = [
            "\n### The store against the re-fit\n",
            "| config | re-fit rows matched | fallback flags agree | max abs diff in alpha+beta, "
            "store LS "
            "vs re-fit, on blocks reproduced at 1e-12 | recursion reproduced at the re-fit's "
            "parameters (of converged) | its max residual | not-reproduced blocks with alpha <= "
            "1e-6 "
            "| beta^window on not-reproduced blocks (min / max) | beta^window max on reproduced |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for config in ("garch11_t", "garch11"):
            e = notes[config]
            lines.append(
                f"| `{config}` | {e['reference_fits_matched']} | "
                f"{e['reference_fallback_flags_agree']} | "
                f"{e['max_abs_diff_store_vs_reference_where_reproduced']:.2e} | "
                f"{e['reference_recursion_reproduced_1e-8']} | "
                f"{e['reference_recursion_max_rel_residual']:.2e} | "
                f"{e['not_reproduced_alpha_ref_below_1e-6']} | "
                f"{_fmt(e['not_reproduced_beta_pow_window_min'], 3)} / "
                f"{_fmt(e['not_reproduced_beta_pow_window_max'], 3)} | "
                f"{_fmt(e['reproduced_beta_pow_window_max'], 3)} |"
            )
        parts.append("\n".join(lines) + "\n")
    parts.append(
        "\n## The ratio `forecast_var(config) / forecast_var(ewma)`, by the governing fit\n\n"
    )
    parts.append(
        "Quantiles of the per-origin ratio; `within 5 %` is the share of origins with the ratio "
        "in [0.95, 1.05]. Ratios of two forecasts are the study's own output. `converged_pinned` "
        "is "
        "alpha+beta within 1e-6 of 1 on the governing fit; `unresolved` would be a converged block "
        "whose alpha+beta could not be read (none when the re-fit is on disk).\n\n"
    )
    shown = ratios[
        [
            "config",
            "group",
            "n_rows",
            "n_fits",
            "q01",
            "q05",
            "q25",
            "median",
            "q75",
            "q95",
            "q99",
            "min",
            "max",
            "share_within_5pct",
        ]
    ].rename(columns={"share_within_5pct": "within 5 %"})
    parts.append(
        _table(
            shown,
            {
                "n_rows": lambda v: str(int(v)),
                "n_fits": lambda v: str(int(v)),
                "within 5 %": lambda v: f"{100 * v:.1f}%",
            },
        )
        + "\n"
    )
    parts.append("\n## Every scheduled fit\n\n")
    columns = [
        "config",
        "fit_origin",
        "fallback",
        "n_equations",
        "alpha_plus_beta_governing",
        "store_max_rel_residual",
    ]
    if has_ref:
        columns += [
            "alpha_ref",
            "beta_ref",
            "omega_ref",
            "beta_ref_pow_window",
            "ref_max_rel_residual",
        ]
    columns += ["alpha_store", "beta_store", "alpha_plus_beta_store"]
    parts.append(
        _table(
            fits[columns],
            {
                "fit_origin": lambda v: str(int(v)),
                "n_equations": lambda v: str(int(v)),
                "alpha_plus_beta_governing": lambda v: f"{v:.9f}",
                "alpha_plus_beta_store": lambda v: f"{v:.9f}",
                "store_max_rel_residual": lambda v: f"{v:.1e}",
                "ref_max_rel_residual": lambda v: f"{v:.1e}",
                "beta_ref_pow_window": lambda v: f"{v:.2e}",
                "omega_ref": lambda v: f"{v:.2e}",
            },
        )
        + "\n"
    )
    (docs / "P3_BOUNDARY_PERSISTENCE.md").write_text("\n".join(parts), encoding="utf-8")


def write_qlike(inputs: Inputs, out: Outputs, docs: Path) -> None:
    twice = out.frames["P3_QLIKE_TWICE"]
    near = out.frames["P3_NEAR_ZERO_TARGETS"]
    parts: list[str] = []
    parts.append("# P3 — QLIKE with and without the five near-zero target days\n")
    parts.append(
        "J2's loss tables report QLIKE once. This file reports it twice — with and without the "
        "five scored target days that sit within 1e-8 of zero — and carries the two fit-diagnostic "
        "rates the J3 brief asked to see beside every loss table.\n"
    )
    parts.append(REPORTED_NOT_INTERPRETED + "\n")
    parts.append(
        _provenance(
            inputs,
            [
                (
                    "Machine-readable",
                    "`docs/P3_QLIKE_TWICE.csv`, `docs/P3_NEAR_ZERO_TARGETS.csv`, "
                    "`docs/P3_FIT_DIAGNOSTICS.csv`",
                )
            ],
        )
    )
    parts.append("\n## The five days\n\n")
    parts.append(
        "Identified from the store by the rule `0 < proxy_var < 1e-8` on scored rows; the target "
        "is given as its ratio to the asset's median positive target over the grid's scored "
        "target days, not as itself — J1's ratios used the whole panel as the denominator, so "
        "they differ slightly — "
        "(docs/P3_ORDER_STATISTICS.md). The QLIKE range is over the 13 models on that day.\n\n"
    )
    shown = near.copy()
    shown["date"] = [str(pd.Timestamp(v).date()) for v in shown["date"]]
    parts.append(
        _table(
            shown,
            {
                "target_index": lambda v: str(int(v)),
                "n_models_scored": lambda v: str(int(v)),
                "target_over_asset_median": lambda v: f"{v:.3e}",
            },
        )
        + "\n"
    )
    parts.append(
        f"\n{len(near)} rows found; the brief and docs/P3_ANALYSIS_VALIDITY.md §1.4 name five.\n"
    )
    parts.append("\n## Mean QLIKE, twice, per asset x model\n\n")
    parts.append(
        "`all` is the stored column; `ex` excludes the days above. Standard errors: `fixed` is "
        "J2's rule-of-thumb Bartlett (no pre-whitening), `auto` the pre-whitened Andrews-bandwidth "
        "estimator of docs/P3_DM.md. Assets on which nothing is excluded have identical columns "
        "and are listed once for completeness.\n\n"
    )
    for asset in inputs.assets:
        sub = twice.loc[twice["asset"] == asset]
        excluded = int(sub["rows_excluded"].max())
        parts.append(
            f"\n### {asset} ({excluded} row{'s' if excluded != 1 else ''} excluded per model)\n\n"
        )
        table = pd.DataFrame(
            {
                "model": [f"`{m}`" for m in sub["model"]],
                "n all": [int(v) for v in sub["n_all"]],
                "QLIKE all (fixed SE) [auto SE]": [
                    f"{a:.6g} ({b:.3g}) [{c:.3g}]"
                    for a, b, c in zip(
                        sub["mean_all"], sub["se_fixed_all"], sub["se_auto_all"], strict=True
                    )
                ],
                "n ex": [int(v) for v in sub["n_ex_near_zero"]],
                "QLIKE ex (fixed SE) [auto SE]": [
                    f"{a:.6g} ({b:.3g}) [{c:.3g}]"
                    for a, b, c in zip(
                        sub["mean_ex_near_zero"],
                        sub["se_fixed_ex_near_zero"],
                        sub["se_auto_ex_near_zero"],
                        strict=True,
                    )
                ],
                "shift": [f"{v:+.2%}" for v in sub["mean_shift_relative"]],
            }
        )
        parts.append(frame_markdown(table) + "\n")
    parts.append(
        "\n## Fit diagnostics carried\n\n"
        + fit_diagnostics_markdown(out.frames["P3_FIT_DIAGNOSTICS"])
        + "\n"
    )
    parts.append(
        "\n`garch11`, `garch11_t`, `gjr`: EWMA fallbacks per scheduled fit (store, `fit_status`). "
        "`autoarima`: scipy optimiser status non-zero, from J1's re-fit probe "
        "(`data/fit_probe/cpu.parquet`, uncommitted; grid total 2,334 / 2,366 = 98.6 %, "
        "docs/P3_INSTRUMENTATION_GAP.md §3.1). The other nine configs report nothing and read `not "
        "instrumented`.\n"
    )
    (docs / "P3_QLIKE_LEVERAGE.md").write_text("\n".join(parts), encoding="utf-8")


def write_csvs(out: Outputs, docs: Path) -> dict[str, int]:
    """Every frame as ``docs/<name>.csv`` at %.17g; returns the row count per file."""
    counts: dict[str, int] = {}
    for name, frame in out.frames.items():
        path = docs / f"{name}.csv"
        frame.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        counts[path.name] = len(frame)
    return counts


def write_manifest(inputs: Inputs, out: Outputs, docs: Path, *, data_digests: Path) -> Path:
    digests = (
        json.loads(data_digests.read_text(encoding="utf-8")) if data_digests.is_file() else None
    )
    files = sorted(p for p in docs.glob("P3_*.csv")) + sorted(p for p in docs.glob("P3_*.md"))
    ours = {
        p.name
        for p in files
        if p.stem in out.frames
        or p.stem
        in (
            "P3_DM",
            "P3_MCS",
            "P3_BACKTESTS",
            "P3_ECON",
            "P3_CROSS_ASSET",
            "P3_CRISIS",
            "P3_BOUNDARY_PERSISTENCE",
            "P3_QLIKE_LEVERAGE",
        )
    }
    payload: dict[str, Any] = {
        "analysis": "P3 comparison inference (prompt J3)",
        "master_seed": MASTER_SEED,
        "grid_manifest": {
            "path": "docs/P3_GRID_manifest.json",
            "manifest_digest": inputs.manifest_digest,
            "store_digest": inputs.store_digest,
            "n_cells": len(inputs.manifest),
            "config_hashes": {
                f"{r.asset}|{r.model}": r.config_hash
                for r in inputs.manifest.itertuples(index=False)
            },
        },
        "data_digests": digests,
        "git_sha": _git_sha(),
        "versions": _versions(),
        "environment": {
            "thread_pins": {
                k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        },
        "losses": list(LOSSES),
        "near_zero_target_rule": f"0 < proxy_var < {NEAR_ZERO_TARGET:g}",
        "near_zero_rows": [
            {"asset": r.asset, "target_index": _count(r.target_index)}
            for r in inputs.near_zero.itertuples(index=False)
        ],
        "sections": out.manifest,
        "notes": out.notes,
        "wall_clock_s": {k: round(v, 3) for k, v in out.timings.items()},
        "outputs": {
            p.name: {
                "sha256": _sha256(p),
                "rows": len(out.frames[p.stem]) if p.stem in out.frames else None,
            }
            for p in files
            if p.name in ours
        },
    }
    path = docs / "P3_ANALYSIS_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


SECTIONS: Final = ("qlike", "boundary", "dm", "mcs", "backtests", "econ", "cross_asset", "crisis")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/P3_GRID_manifest.json"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--pairwise", type=Path, default=Path("docs/P3_PAIRWISE_COMPLETE.csv"))
    parser.add_argument("--fit-probe", type=Path, default=Path("data/fit_probe/cpu.parquet"))
    parser.add_argument("--fits", type=Path, default=Path("docs/P3_CONVERGENCE_FITS.parquet"))
    parser.add_argument("--data-digests", type=Path, default=Path("docs/P3_DATA_DIGESTS.json"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--sections", default=",".join(SECTIONS))
    args = parser.parse_args(argv)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown sections {unknown}; known: {SECTIONS}")
    args.docs.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    inputs = load_inputs(args.store_root, args.manifest)
    out = Outputs()
    out.timings["load"] = time.perf_counter() - started
    print(
        f"grid {inputs.grid.shape} in {out.timings['load']:.1f}s; manifest_digest "
        f"{inputs.manifest_digest[:16]}; {len(inputs.near_zero)} near-zero target rows",
        flush=True,
    )

    # The two amendment checks first: everything below reads their columns.
    run_qlike(inputs, out, args.fit_probe if args.fit_probe.is_file() else None)
    print(f"qlike/diagnostics {out.timings['qlike']:.1f}s", flush=True)
    if "boundary" in sections:
        run_boundary(inputs, out, args.fits if args.fits.is_file() else None)
        print(f"boundary {out.timings['boundary']:.1f}s", flush=True)
    matrices = loss_matrices(inputs) if {"dm", "mcs", "cross_asset"} & set(sections) else {}
    if "dm" in sections:
        run_dm(inputs, matrices, pairwise_reference(args.pairwise), out)
        print(f"dm {out.timings['dm']:.1f}s", flush=True)
    if "mcs" in sections:
        run_mcs(inputs, matrices, out, n_boot=args.n_boot)
        print(f"mcs {out.timings['mcs']:.1f}s", flush=True)
    if "backtests" in sections:
        run_backtests(inputs, out)
        print(f"backtests {out.timings['backtests']:.1f}s", flush=True)
    if "econ" in sections:
        run_econ(inputs, out, n_boot=args.n_boot)
        print(f"econ {out.timings['econ']:.1f}s", flush=True)
    if "cross_asset" in sections and "P3_MCS" in out.frames:
        run_cross_asset(inputs, out)
        print(f"cross_asset {out.timings['cross_asset']:.1f}s", flush=True)
    if "crisis" in sections:
        run_crisis(inputs, out, n_boot=args.n_boot)
        print(f"crisis {out.timings['crisis']:.1f}s", flush=True)

    counts = write_csvs(out, args.docs)
    write_qlike(inputs, out, args.docs)
    if "boundary" in sections:
        write_boundary(inputs, out, args.docs)
    if "dm" in sections:
        write_dm(inputs, out, args.docs)
    if "mcs" in sections:
        write_mcs(inputs, out, args.docs)
    if "backtests" in sections:
        write_backtests(inputs, out, args.docs)
    if "econ" in sections:
        write_econ(inputs, out, args.docs)
    if "cross_asset" in sections and "P3_CROSS_ASSET_RANKS" in out.frames:
        write_cross_asset(inputs, out, args.docs)
    if "crisis" in sections:
        write_crisis(inputs, out, args.docs)
    out.timings["total"] = time.perf_counter() - started
    manifest = write_manifest(inputs, out, args.docs, data_digests=args.data_digests)
    for name, rows in sorted(counts.items()):
        print(f"  {name}: {rows} rows")
    print(f"manifest -> {manifest}; total {out.timings['total']:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
