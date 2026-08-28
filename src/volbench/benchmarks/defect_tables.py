#!/usr/bin/env python
"""K's per-asset defect tables, and the acceptance test the fixes must pass.

`docs/P3_LGBM_SMEARING_AUDIT.md` and `docs/P3_TSFM_VARIANCE_AUDIT.md` were
aggregated from throwaway scripts. Their numbers were re-derivable — every
input is a committed probe writing a parquet — but not *re-runnable*, and a
table nobody can re-run is a table nobody can check after the code moves under
it. Both fixes moved the code under them, so this module is what the aggregation
should have been from the start.

It consumes probe output and the primary store, and it fits nothing.

Three things it reports
=======================
:func:`lgbm_factor_table`
    Audit §2, per asset: the in-sample factor, the out-of-fold factor, the
    factor ``fit`` actually used, and the **realized** factor implied by the
    grid's own one-step forecast errors. The last of those reads the future
    and is therefore not a candidate estimator — it is the quantity the other
    three are trying to estimate, so it is what they are judged against.
:func:`lgbm_acceptance`
    The acceptance test for the out-of-fold fix, as a decidable claim rather
    than a hope: the shipped factor must now agree with the realized one to
    within a stated tolerance, at the panel median and on every asset.
:func:`tsfm_closure_table`
    Audit §3, per config per asset: the mean under each tail closure as a
    ratio to the flat-tailed reading that shipped before the fix. Three
    closures, because the tail beyond the outermost quantile level is
    genuinely unidentified and one number would overstate what the grid says.

Run, after both probes have been re-run against the fixed adapters::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical python -m volbench.benchmarks.defect_tables
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from volbench.models.tsfm_common import CLOSURES
from volbench.results import ResultsStore

__all__ = [
    "EMPIRICAL_CLOSURE",
    "PANEL_TOLERANCE",
    "PER_ASSET_TOLERANCE",
    "Acceptance",
    "lgbm_acceptance",
    "lgbm_factor_table",
    "realized_lgbm_factors",
    "tsfm_closure_table",
]

#: How far the shipped factor may sit from the realized one at the **panel
#: median** before the out-of-fold fix is judged not to have worked. The
#: audit's own prediction was a 1.5 % overshoot (1.703 against 1.678); 5 %
#: leaves room for the sampling noise of a per-asset median without leaving
#: room for the 21.8 % gap the fix exists to close.
PANEL_TOLERANCE: Final = 0.05

#: The same claim per asset, where the audit's spread ran from a 3.2 %
#: undershoot (BTC-USD) to an 8.2 % overshoot (SPY). A per-asset bound of 15 %
#: still fails on every asset if the fix is reverted.
PER_ASSET_TOLERANCE: Final = 0.15

#: The name of the closure that can only be computed after the fact. It is
#: reported next to the two implementable ones as a bound, never as a
#: candidate: it reads realizations *after* the origin it would correct, so
#: nothing could compute it at forecast time.
EMPIRICAL_CLOSURE: Final = "empirical"


# --------------------------------------------------------------------------
# LightGBM: the four factors, and whether the shipped one landed
# --------------------------------------------------------------------------


def _cell_hash(manifest: pd.DataFrame, asset: str, model: str) -> str | None:
    rows = manifest[(manifest["asset"] == asset) & (manifest["model"] == model)]
    if rows.empty or rows["config_hash"].isna().all():
        return None
    return str(rows["config_hash"].iloc[0])


def realized_lgbm_factors(
    store: ResultsStore, manifest: pd.DataFrame, probe: pd.DataFrame
) -> pd.DataFrame:
    """The factor the grid's own one-step forecast errors imply, per asset.

    ``mu_hat`` is never re-run. Within a refit block ``update`` re-conditions
    at fixed parameters and re-estimates nothing, so the factor is constant
    across the block by construction and

        mu_hat_t = log(forecast_var_t) - log(smear_shipped[fit_origin_t])

    inverts the stored variance exactly. The realized log-space error is then
    ``log(proxy_var_{t+1}) - mu_hat_t`` at the cell's own scored origins, and
    Duan's factor over those errors is what the estimator was aiming at.

    Rows with no fit behind them, or an unscorable target, carry NaN and are
    dropped rather than imputed.
    """
    rows: list[dict[str, Any]] = []
    for asset in sorted(probe["asset"].unique()):
        digest = _cell_hash(manifest, asset, "lgbm")
        if digest is None or not store.has(digest):
            continue
        frame = store.read(digest)
        shipped = (
            probe.loc[probe["asset"] == asset]
            .dropna(subset=["smear_shipped"])
            .set_index("origin")["smear_shipped"]
        )
        factor = frame["fit_origin"].map(shipped)
        # An unscorable row — a NaN or non-positive proxy, or an origin with no
        # fit behind it — has no realized error to contribute. ``errstate``
        # because ``log(0)`` on those rows is the expected path, not a
        # surprise: they are dropped immediately after, never imputed, and the
        # surviving count is reported so the drop is visible.
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log(frame["proxy_var"]) - (np.log(frame["forecast_var"]) - np.log(factor))
        resid = resid.to_numpy(dtype=np.float64)
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            continue
        rows.append(
            {
                "asset": asset,
                "n_scored": int(resid.size),
                "realized_resid_var": float(np.mean(resid * resid)),
                "smear_realized": float(np.mean(np.exp(resid))),
            }
        )
    return pd.DataFrame(rows)


def lgbm_factor_table(
    probe: pd.DataFrame, store: ResultsStore, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Audit §2 as a frame: three estimable factors against the realized one.

    Medians over each asset's refit origins for the in-window scales, and the
    single pooled figure for the realized one — the same reduction the audit
    reported, so the two tables are comparable line for line.
    """
    ok = probe[probe["probe_error"].isna()]
    per_asset = (
        ok.groupby("asset")
        .agg(
            fits=("origin", "size"),
            resid_var_in_sample=("resid_var_in_sample", "median"),
            resid_var_out_of_fold=("resid_var_out_of_fold", "median"),
            smear_in_sample=("smear_in_sample", "median"),
            smear_out_of_fold=("smear_out_of_fold", "median"),
            smear_shipped=("smear_shipped", "median"),
        )
        .reset_index()
    )
    table = per_asset.merge(realized_lgbm_factors(store, manifest, ok), on="asset", how="left")
    table["shipped_over_realized"] = table["smear_shipped"] / table["smear_realized"]
    table["in_sample_over_realized"] = table["smear_in_sample"] / table["smear_realized"]
    return table.sort_values("asset").reset_index(drop=True)


class Acceptance:
    """Whether the out-of-fold factor landed on the quantity it estimates.

    A fix without an assertion that it worked is a hope, so this is a verdict
    with a stated tolerance rather than a table to read. ``passed`` is the
    conjunction of the panel-median claim and the per-asset one; the worst
    asset is carried so a failure names itself.
    """

    def __init__(self, table: pd.DataFrame, panel: float, per_asset: float) -> None:
        ratios = table["shipped_over_realized"].dropna()
        self.n_assets = int(ratios.size)
        self.panel_tolerance = panel
        self.per_asset_tolerance = per_asset
        self.panel_median = float(ratios.median()) if self.n_assets else math.nan
        worst = (ratios - 1.0).abs().idxmax() if self.n_assets else None
        self.worst_asset = None if worst is None else str(table.loc[worst, "asset"])
        self.worst_ratio = math.nan if worst is None else float(ratios.loc[worst])

    @property
    def panel_ok(self) -> bool:
        return bool(abs(self.panel_median - 1.0) <= self.panel_tolerance)

    @property
    def per_asset_ok(self) -> bool:
        return bool(abs(self.worst_ratio - 1.0) <= self.per_asset_tolerance)

    @property
    def passed(self) -> bool:
        return self.n_assets > 0 and self.panel_ok and self.per_asset_ok

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"acceptance ({self.n_assets} assets): {verdict}\n"
            f"  panel median shipped/realized {self.panel_median:.4f} "
            f"(tolerance {self.panel_tolerance:.0%}) -> {'ok' if self.panel_ok else 'FAILED'}\n"
            f"  worst asset {self.worst_asset} at {self.worst_ratio:.4f} "
            f"(tolerance {self.per_asset_tolerance:.0%}) "
            f"-> {'ok' if self.per_asset_ok else 'FAILED'}"
        )


def lgbm_acceptance(
    table: pd.DataFrame,
    panel: float = PANEL_TOLERANCE,
    per_asset: float = PER_ASSET_TOLERANCE,
) -> Acceptance:
    """The acceptance test of the out-of-fold fix. See :class:`Acceptance`."""
    return Acceptance(table, panel, per_asset)


# --------------------------------------------------------------------------
# TSFM: what each tail closure would have given
# --------------------------------------------------------------------------


def _grid_columns(probe: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in probe.columns if c.startswith("q_")),
        key=lambda c: float(c[2:]),
    )


def _empirical_ratio(probe: pd.DataFrame, store: ResultsStore, manifest: pd.DataFrame) -> pd.Series:
    """Per-origin ``empirical_mean / flat_mean``, the assumption-free bound.

    The two atoms are replaced by what the realizations actually did beyond
    the grid's outer levels. Because each origin's grid has its own scale, the
    exceedances are pooled **as ratios** to the level they exceeded — one
    number per cell, ``mean(RV / q)`` over the origins where ``RV`` fell
    outside ``q`` — and applied back to each origin's own outer quantile.

    This is a diagnostic and never a candidate closure: it reads realizations
    after the origin it would correct, so no forecast-time code could compute
    it. It bounds the two implementable closures without assuming a shape.
    """
    levels = _grid_columns(probe)
    lo_col, hi_col = levels[0], levels[-1]
    lo_tau, hi_tau = float(lo_col[2:]), float(hi_col[2:])
    out = pd.Series(np.nan, index=probe.index, dtype=float)
    for (asset, model), block in probe.groupby(["asset", "model"]):
        digest = _cell_hash(manifest, str(asset), str(model))
        if digest is None or not store.has(digest):
            continue
        realized = store.read(digest).set_index("target_index")["proxy_var"]
        rv = block["target_index"].map(realized).to_numpy(dtype=np.float64)
        lo_q = block[lo_col].to_numpy(dtype=np.float64)
        hi_q = block[hi_col].to_numpy(dtype=np.float64)
        flat = block["flat_tail_mean"].to_numpy(dtype=np.float64)
        below = np.isfinite(rv) & (lo_q > 0.0) & (rv < lo_q)
        above = np.isfinite(rv) & (hi_q > 0.0) & (rv > hi_q)
        if not below.any() or not above.any():
            continue
        lo_ratio = float(np.mean(rv[below] / lo_q[below]))
        hi_ratio = float(np.mean(rv[above] / hi_q[above]))
        interior = flat - lo_tau * lo_q - (1.0 - hi_tau) * hi_q
        closed = lo_tau * lo_ratio * lo_q + interior + (1.0 - hi_tau) * hi_ratio * hi_q
        out.loc[block.index] = closed / flat
    return out


def tsfm_closure_table(
    probe: pd.DataFrame,
    store: ResultsStore | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit §3 as a frame: each closure's mean over the flat-tailed reading.

    One row per config per asset plus a pooled ``panel`` row per config. The
    ratios are medians over the probed origins; an origin whose closure could
    not be fitted contributes to ``n_flat_fallback`` instead, since how often
    that happens is part of the sensitivity rather than a footnote to it.

    ``store``/``manifest`` are optional and only enable the empirical column.
    """
    grid = probe[probe["probe_error"].isna() & probe["flat_tail_mean"].notna()].copy()
    grid["lognormal"] = grid["mean_lognormal_tail"] / grid["flat_tail_mean"]
    grid["loglinear"] = grid["mean_loglinear_tail"] / grid["flat_tail_mean"]
    grid["scored_over_flat"] = grid["vhat"] / grid["flat_tail_mean"]
    if store is not None and manifest is not None:
        grid[EMPIRICAL_CLOSURE] = _empirical_ratio(grid, store, manifest)

    columns = ["lognormal", "loglinear", "scored_over_flat"]
    if EMPIRICAL_CLOSURE in grid:
        columns.append(EMPIRICAL_CLOSURE)

    def _reduce(block: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {
            "n_origins": len(block),
            "n_flat_fallback": int((block["tail_closure"] == "flat").sum()),
        }
        for column in columns:
            row[column] = float(block[column].median())
        return row

    rows: list[dict[str, Any]] = []
    for model in sorted(grid["model"].unique()):
        block = grid[grid["model"] == model]
        for asset in sorted(block["asset"].unique()):
            rows.append({"model": model, "asset": asset, **_reduce(block[block["asset"] == asset])})
        rows.append({"model": model, "asset": "panel", **_reduce(block)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/grid_primary/manifest_fix.json")
    )
    parser.add_argument(
        "--lgbm-probe", type=Path, default=Path("data/lgbm_smear_probe/probe.parquet")
    )
    parser.add_argument(
        "--tsfm-probe", type=Path, default=Path("data/tsfm_dist_probe/probe.parquet")
    )
    args = parser.parse_args(argv)

    store = ResultsStore(args.store)
    manifest = pd.DataFrame(json.loads(args.manifest.read_text(encoding="utf-8"))["cells"])
    pd.set_option("display.width", 200)

    status = 0
    if args.lgbm_probe.is_file():
        table = lgbm_factor_table(pd.read_parquet(args.lgbm_probe), store, manifest)
        print("LightGBM smearing factors, per asset (docs/P3_LGBM_SMEARING_AUDIT.md §2)\n")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\npanel medians:")
        for column in ("smear_in_sample", "smear_out_of_fold", "smear_shipped", "smear_realized"):
            print(f"  {column:22s} {table[column].median():.3f}")
        verdict = lgbm_acceptance(table)
        print("\n" + str(verdict))
        status = status or (0 if verdict.passed else 1)

    if args.tsfm_probe.is_file():
        probe = pd.read_parquet(args.tsfm_probe)
        if "flat_tail_mean" in probe.columns:
            print("\n\nTSFM tail closures, ratio to the flat-tailed mean "
                  "(docs/P3_TSFM_VARIANCE_AUDIT.md §3)\n")
            print(
                tsfm_closure_table(probe, store, manifest).to_string(
                    index=False, float_format=lambda v: f"{v:.3f}"
                )
            )
            print(f"\nclosures reported: {(*CLOSURES, EMPIRICAL_CLOSURE)}")
    return status


if __name__ == "__main__":
    sys.argv[0] = "defect_tables"
    raise SystemExit(main())
