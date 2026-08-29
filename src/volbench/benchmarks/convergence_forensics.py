#!/usr/bin/env python
"""The 38 EWMA fallbacks of the primary grid, re-fitted and described.

``docs/P3_GRID_manifest.json`` reports 38 fallbacks in 7,101 GARCH-family
fits and ``fit_status`` records each one's exit flag. That is everything the
store knows about them: :class:`~volbench.models.garch.FittedGARCH` used to
set ``result=None`` on a fit that did not converge, so **the parameter vector
the optimizer stopped at was discarded**. A flag cannot say which parameter
sat at which bound, and that is the question separating a flat likelihood
surface from a coding defect.

So two things happen here, in this order:

1. ``models/garch.py`` now retains the terminal state of every scheduled fit
   (:class:`~volbench.models.garch.TerminalFit`) — added rather than guessed
   at, and hashed into nothing, so it moves no config hash and changes no
   stored number.
2. This module re-fits **every** GARCH-family scheduled fit of the grid at the
   grid's own origins and reads that state back. The re-fit is checked against
   the store fit by fit: all 7,101 ``fit_status`` strings must reproduce, or
   the re-fit is describing a different experiment and nothing below it means
   anything. :func:`refit_all` refuses to return otherwise.

It writes nothing to any :class:`~volbench.results.ResultsStore`, touches no
fragment, and re-fits only the three configs that report convergence at all
(``garch11``, ``garch11_t``, ``gjr`` — docs/P3_INSTRUMENTATION_GAP.md); the
other ten estimate nothing that can fall back.

**Reported, not interpreted.** Every function here computes a number or
arranges numbers in a table. None of them ranks a model or reads a fallback
as evidence about a forecast.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run python -m volbench.benchmarks.convergence_forensics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from volbench.benchmarks.grid_primary import ARM, asset_data, model_configs
from volbench.data.crisis import CALM_TAG, CRISIS_WINDOWS, tag_dates
from volbench.data.panel import build_panel
from volbench.models.garch import FittedGARCH, TerminalFit
from volbench.results import ResultsStore
from volbench.runner import AssetData, ModelConfig

__all__ = [
    "BOUNDARY_TOLERANCES",
    "GARCH_LABELS",
    "at_bound_table",
    "boundary_flags",
    "constraint_slack",
    "dia_intersection",
    "gamma_persistence_table",
    "paired_windows",
    "refit_all",
    "refit_cell",
    "window_stats",
]

#: The three configs whose adapter implements ``fit_diagnostics``. The other
#: ten report nothing, so "0 fallbacks" would be a claim about instrumentation
#: rather than about them (docs/P3_INSTRUMENTATION_GAP.md).
GARCH_LABELS: Final = ("garch11", "garch11_t", "gjr")

#: Absolute tolerances the "at a bound" question is answered at, smallest
#: first. A boundary claim that moves across this ladder is a claim about the
#: tolerance; one that does not is a claim about the fit.
BOUNDARY_TOLERANCES: Final = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


# --------------------------------------------------------------------------
# re-fitting the grid's own fits
# --------------------------------------------------------------------------


def _scheduled_fits(
    store: ResultsStore, manifest: pd.DataFrame, asset: str, label: str
) -> pd.Series:
    """``fit_origin -> fit_status`` for one cell, as the store recorded it.

    The per-fit view of a per-row column: every origin resting on one
    scheduled fit carries that fit's status, so the first of each group is the
    fit (docs/P3_ANALYSIS_ASSUMPTIONS.md §5).
    """
    cells = manifest[(manifest["asset"] == asset) & (manifest["model"] == label)]
    if cells.empty:
        return pd.Series(dtype=str)
    frame = store.read(str(cells["config_hash"].iloc[0]))
    scheduled = frame[frame["fit_origin"] >= 0]
    status = scheduled.groupby("fit_origin")["fit_status"].first()
    status.index = status.index.astype(np.int64)
    return status.astype(str)


def window_stats(window: np.ndarray) -> dict[str, float]:
    """What the fit window looks like, independently of any model.

    Kurtosis is Pearson's (a normal scores 3), bias-corrected — stated rather
    than left to a default, because the Fisher/Pearson choice is a three-unit
    difference that reads as a finding.
    """
    return {
        "n": float(window.size),
        "std": float(np.std(window, ddof=1)),
        "kurtosis": float(stats.kurtosis(window, fisher=False, bias=False)),
        "skew": float(stats.skew(window, bias=False)),
        "max_abs_return": float(np.max(np.abs(window))),
        "n_zero_returns": float(np.count_nonzero(window == 0.0)),
    }


def _terminal_row(terminal: TerminalFit | None) -> dict[str, Any]:
    """One fit's terminal state, flattened, with the derived persistence."""
    if terminal is None:
        return {"converged_flag": np.nan, "loglikelihood": np.nan}
    row: dict[str, Any] = terminal.as_dict()
    params = dict(terminal.params)
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)
    gamma = params.get("gamma[1]", 0.0)
    row["omega_return_scale"] = terminal.omega_on_the_return_scale
    row["alpha_plus_beta"] = alpha + beta
    # GJR's stationarity condition weights the leverage term by the
    # probability of a negative shock under a symmetric innovation.
    row["persistence"] = alpha + gamma / 2.0 + beta
    # Distance to each box bound. Derivable from the value and the bound, and
    # kept anyway: "at a bound" is the question this table exists to answer,
    # and a reader should not have to subtract two columns to ask it.
    for name, (below, above) in terminal.slack().items():
        row[f"slack_low_{name}"] = below
        row[f"slack_high_{name}"] = above
    return row


def refit_cell(
    config: ModelConfig, data: AssetData, status: pd.Series, dates: pd.DatetimeIndex
) -> list[dict[str, Any]]:
    """Re-fit one cell at the origins the grid fitted at, and read the optimizer."""
    splitter = ARM.splitter(1)
    returns = data.returns.to_numpy(dtype=np.float64)
    wanted = set(status.index)
    rows: list[dict[str, Any]] = []
    for origin in splitter.split(len(returns)):
        if origin.origin not in wanted:
            continue
        window = returns[origin.train]
        fitted = config.factory().fit(window)
        if not isinstance(fitted, FittedGARCH):
            raise TypeError(
                f"{config.label} fitted a {type(fitted).__name__}; this probe reads the "
                "optimizer's terminal state, which only the GARCH family retains"
            )
        row: dict[str, Any] = {
            "asset": data.asset,
            "config": config.label,
            "fit_origin": int(origin.origin),
            "date": dates[origin.origin],
            "window_start_date": dates[int(origin.train[0])],
            "stored_status": str(status.loc[origin.origin]),
            "refit_status": fitted.fit_diagnostics().status(),
            "fallback": bool(fitted.fallback),
        }
        row.update(window_stats(window))
        row.update(_terminal_row(fitted.terminal))
        rows.append(row)
    return rows


def refit_all(
    *,
    store_root: Path,
    manifest_path: Path,
    labels: Sequence[str] = GARCH_LABELS,
    assets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Every GARCH-family scheduled fit of the grid, re-fitted and described.

    Raises if a single re-fitted ``fit_status`` differs from the stored one:
    a forensic table built on fits that are not the grid's fits would look
    exactly like one built on the grid's.
    """
    store = ResultsStore(store_root)
    manifest = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8"))["cells"])
    panel = build_panel()
    if assets:
        panel = {k: v for k, v in panel.items() if k in set(assets)}
    data = {name: asset_data(series) for name, series in panel.items()}
    by_label = {c.label: c for c in model_configs(device="cpu")}

    rows: list[dict[str, Any]] = []
    for label in labels:
        started = time.perf_counter()
        for asset, datum in data.items():
            status = _scheduled_fits(store, manifest, asset, label)
            if status.empty:
                continue
            dates = pd.DatetimeIndex(datum.returns.index)
            rows.extend(refit_cell(by_label[label], datum, status, dates))
        print(f"  {label:10s} {time.perf_counter() - started:7.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    disagree = frame[frame["stored_status"] != frame["refit_status"]]
    if not disagree.empty:
        raise RuntimeError(
            f"{len(disagree)} of {len(frame)} re-fits do not reproduce the store's fit_status; "
            f"first: {disagree.iloc[0][['asset', 'config', 'fit_origin']].to_dict()} "
            f"stored={disagree.iloc[0]['stored_status']!r} "
            f"refit={disagree.iloc[0]['refit_status']!r}"
        )
    frame["crisis_tag"] = tag_dates(pd.DatetimeIndex(frame["date"])).to_numpy()
    return frame.sort_values(["asset", "config", "fit_origin"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# the boundary question
# --------------------------------------------------------------------------


def boundary_flags(fits: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Per fit, which parameter sits within ``tolerance`` of which bound.

    Two kinds of boundary, kept apart because they are different objects: a
    **box** bound, which SLSQP is given per parameter, and the **stationarity
    constraint** ``alpha + gamma/2 + beta <= 1``, which it is given as a linear
    inequality. A parameter can be at a box bound while the constraint is
    slack, and the other way round.
    """
    out = pd.DataFrame(index=fits.index)
    for name in ("omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"):
        low, high = f"slack_low_{name}", f"slack_high_{name}"
        if low not in fits.columns:
            continue
        out[f"{name}@low"] = fits[low].to_numpy() <= tolerance
        out[f"{name}@high"] = fits[high].to_numpy() <= tolerance
    out["persistence@1"] = (1.0 - fits["persistence"].to_numpy()) <= tolerance
    return out


def at_bound_table(
    fits: pd.DataFrame, tolerances: Sequence[float] = BOUNDARY_TOLERANCES
) -> pd.DataFrame:
    """How many of ``fits`` sit at each bound, at each tolerance of the ladder."""
    rows: list[dict[str, Any]] = []
    for tolerance in tolerances:
        flags = boundary_flags(fits, tolerance)
        row: dict[str, Any] = {"tolerance": tolerance, "n_fits": len(fits)}
        row.update({column: int(flags[column].sum()) for column in flags.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def constraint_slack(fits: pd.DataFrame) -> pd.DataFrame:
    """The three linear inequalities SLSQP was handed, as signed slack.

    ``arch`` gives the optimizer box bounds *and* linear constraints, and they
    are different objects: a parameter can sit at a box bound while every
    constraint is slack, and the other way round. Non-negative slack means
    inside; a small negative value means the optimizer stopped just outside,
    which SLSQP allows within its own constraint tolerance.
    """
    alpha = fits["alpha[1]"].to_numpy(dtype=float)
    beta = fits["beta[1]"].to_numpy(dtype=float)
    gamma = np.nan_to_num(fits.get("gamma[1]", pd.Series(0.0, index=fits.index)).to_numpy(float))
    return pd.DataFrame(
        {
            # 1 - alpha - gamma/2 - beta >= 0: stationarity. The IGARCH edge.
            "stationarity": 1.0 - (alpha + gamma / 2.0 + beta),
            # alpha + gamma >= 0: a negative shock's coefficient (GJR only).
            "alpha_plus_gamma": alpha + gamma,
            # alpha >= 0: a positive shock's coefficient.
            "alpha": alpha,
        },
        index=fits.index,
    )


def gamma_persistence_table(fits: pd.DataFrame, asset: str, config: str) -> pd.DataFrame:
    """One cell's fits in ``gamma`` / ``alpha+beta`` / persistence space.

    Fallbacks and clean fits described the same way and side by side, because
    the question is where the former sit *relative to* the latter.
    """
    cell = fits[(fits["asset"] == asset) & (fits["config"] == config)]
    slack = constraint_slack(cell)
    described = cell.assign(
        alpha_plus_gamma=slack["alpha_plus_gamma"], stationarity_slack=slack["stationarity"]
    )
    columns = [
        "gamma[1]",
        "alpha[1]",
        "beta[1]",
        "alpha_plus_beta",
        "persistence",
        "alpha_plus_gamma",
        "kurtosis",
        "max_abs_return",
        "std",
    ]
    grouped = described.groupby("fallback")[columns]
    table = grouped.describe(percentiles=[0.05, 0.5, 0.95]).transpose()
    table.columns = pd.Index(["clean", "fallback"][: table.shape[1]])
    return table


def paired_windows(
    fits: pd.DataFrame, config: str, left: str, right: str, origins: Sequence[int]
) -> pd.DataFrame:
    """Two assets' fit windows at the same origins, described identically.

    Both sides must be on one calendar and one origin grid, which is checked
    rather than assumed: comparing "the same dates" across two series whose
    scheduled origins had drifted apart would compare different windows.
    ``nearest_converged`` is the closest origin in the same cell whose fit did
    not fall back, ties resolved to the earlier origin.
    """
    sides = {}
    for asset in (left, right):
        cell = fits[(fits["asset"] == asset) & (fits["config"] == config)]
        sides[asset] = cell.set_index("fit_origin").sort_index()
    if not sides[left].index.equals(sides[right].index):
        raise ValueError(f"{left} and {right} were not fitted at the same origins")
    if not (sides[left]["date"].to_numpy() == sides[right]["date"].to_numpy()).all():
        raise ValueError(f"{left} and {right} are not on one calendar")

    def nearest_converged(frame: pd.DataFrame, origin: int) -> int:
        converged = frame.index[~frame["fallback"]].to_numpy(dtype=np.int64)
        if converged.size == 0:
            raise ValueError("no converged fit in the cell to compare against")
        distance = np.abs(converged - origin)
        return int(converged[np.lexsort((converged, distance))[0]])

    rows: list[dict[str, Any]] = []
    for origin in origins:
        row: dict[str, Any] = {"fit_origin": int(origin), "date": sides[left].loc[origin, "date"]}
        for asset in (left, right):
            frame = sides[asset]
            near = nearest_converged(frame, int(origin))
            here = frame.loc[origin]
            row[f"{asset}_fallback"] = bool(here["fallback"])
            for column in ("kurtosis", "max_abs_return", "std"):
                row[f"{asset}_{column}"] = float(np.asarray(here[column], dtype=float))
            row[f"{asset}_nearest_converged"] = near
            row[f"{asset}_nearest_gap"] = abs(near - int(origin))
            row[f"{asset}_nearest_alpha_plus_beta"] = float(
                np.asarray(frame.loc[near, "alpha_plus_beta"], dtype=float)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def dia_intersection(fits: pd.DataFrame, asset: str = "DIA") -> dict[str, Any]:
    """The fallback origins of one asset's three configs, and their overlaps."""
    failed = fits[(fits["asset"] == asset) & fits["fallback"]]
    per_config = {
        label: sorted(int(o) for o in failed.loc[failed["config"] == label, "fit_origin"])
        for label in GARCH_LABELS
    }
    sets = {label: set(origins) for label, origins in per_config.items()}
    three_way = set.intersection(*sets.values()) if sets else set()
    pairwise = {
        f"{a}&{b}": sorted(sets[a] & sets[b])
        for i, a in enumerate(GARCH_LABELS)
        for b in GARCH_LABELS[i + 1 :]
    }
    return {
        "per_config": per_config,
        "union": sorted(set.union(*sets.values())) if sets else [],
        "three_way_intersection": sorted(three_way),
        "pairwise_intersections": pairwise,
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/P3_GRID_manifest.json")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/P3_CONVERGENCE_FITS.parquet"))
    parser.add_argument("--assets", nargs="*", default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    fits = refit_all(
        store_root=args.store_root, manifest_path=args.manifest, assets=args.assets
    )
    if args.out.suffix == ".csv":
        fits.to_csv(args.out, index=False)
    else:
        fits.to_parquet(args.out, index=False)
    print(f"{len(fits)} fits re-fitted in {time.perf_counter() - started:.1f}s -> {args.out}")
    print(f"fallbacks: {int(fits['fallback'].sum())}")
    print(f"crisis tags: {fits.loc[fits['fallback'], 'crisis_tag'].value_counts().to_dict()}")
    print(f"windows named: {[w.tag for w in CRISIS_WINDOWS]} + {CALM_TAG!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
