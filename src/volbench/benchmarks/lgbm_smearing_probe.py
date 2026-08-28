#!/usr/bin/env python
"""LightGBM's Duan smearing factor, measured on the real panel.

``models/lgbm.py`` retransforms ``E[log RV]`` to a variance with Duan's (1983)
smearing factor ``mean(exp(e))`` over the fit window's **in-sample** residuals.
Duan's estimator wants residuals that behave like draws from the *forecast-
error* distribution, and a boosted ensemble's in-sample residuals do not: the
ensemble shrinks them, which drives the factor toward 1 and quietly turns the
variance forecast back into a median forecast. The size of that was measured
once, on a toy fixture (docs/P2_INTEGRATION.md §3.2). This module measures it on
21 years of index data.

Three residual scales, and the difference between them is the whole point
=========================================================================
``in_sample``
    ``y - f(x)`` on the rows the booster was trained on. What ships.
``out_of_fold``
    Expanding **temporal** folds inside the training window: the rows are cut
    into contiguous blocks and each block is predicted by a booster trained
    only on the blocks before it. Never a random split — a random fold of a
    time series lets tomorrow's neighbours predict today (docs/design.md).
    This is the implementable fix, and it trains on less data than the real fit
    does, so it is an upper bound on what the fix would deliver.
``realized``
    ``log RV_{t+1} - mu_hat_t`` at the grid's own scored origins: the genuine
    one-step forecast error. Not implementable as a factor — it reads the
    future — but it is the quantity the factor is trying to estimate, so it is
    the target both other scales are judged against.

``mu_hat_t`` is not re-run: it is recovered exactly from the stored fragment as
``log(forecast_var_t) - log(smear)``, with ``smear`` the factor of the refit
block that row rests on (``fit_origin``). ``update`` never re-estimates the
factor, so it is constant within a block by construction.

The primary store is read-only here: this module writes only to its own
``--out`` parquet, moves no config hash and rewrites no fragment.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical python -m volbench.benchmarks.lgbm_smearing_probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volbench.benchmarks.grid_primary import ARM, asset_data
from volbench.data.panel import build_panel
from volbench.models._rv import smearing_factor, validated_rv
from volbench.models.lgbm import FEATURE_NAMES, LightGBMRV, _design_matrix
from volbench.results import ResultsStore
from volbench.runner import AssetData

__all__ = [
    "DEFAULT_FOLDS",
    "ROUND_LADDER",
    "out_of_fold_residuals",
    "probe_asset",
    "round_ladder",
]

#: Contiguous, chronological blocks the training window is cut into. Five
#: leaves the first block as training-only and predicts the other four, so 80 %
#: of the window's rows contribute an out-of-fold residual.
DEFAULT_FOLDS: Final = 5

#: Boosting-round counts for the capacity question: is the shipped 100-round
#: cap binding on the *training* loss, i.e. would more rounds keep shrinking the
#: residuals the factor is read off?
ROUND_LADDER: Final = (25, 50, 100, 200, 400, 800)


def _train(config: LightGBMRV, x: NDArray[np.float64], y: NDArray[np.float64], rounds: int) -> Any:
    import lightgbm as lgb

    dataset = lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=False)
    return lgb.train(config._params(), dataset, num_boost_round=rounds)


def out_of_fold_residuals(
    config: LightGBMRV,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    folds: int = DEFAULT_FOLDS,
) -> NDArray[np.float64]:
    """Residuals from expanding chronological folds inside one training window.

    Block ``k`` is predicted by a booster trained on blocks ``0..k-1`` only, so
    no residual is ever produced by a model that saw its own row or any later
    one. Block ``0`` yields none — there is nothing before it — which is the
    unavoidable price of keeping the fold causal.
    """
    n = y.size
    edges = np.linspace(0, n, folds + 1).astype(int)
    out: list[NDArray[np.float64]] = []
    for k in range(1, folds):
        train_end, block_end = edges[k], edges[k + 1]
        if train_end < 2 or block_end <= train_end:
            continue
        booster = _train(config, x[:train_end], y[:train_end], config.num_boost_round)
        pred = np.asarray(booster.predict(x[train_end:block_end]), dtype=np.float64).reshape(-1)
        out.append(y[train_end:block_end] - pred)
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def round_ladder(
    config: LightGBMRV, x: NDArray[np.float64], y: NDArray[np.float64]
) -> list[dict[str, Any]]:
    """Training loss and smearing factor along :data:`ROUND_LADDER`.

    Answers the addendum's amendment: a booster that runs its full round budget
    at every origin says nothing on its own about whether the budget *binds*.
    If the training MSE is still falling at 100 rounds and the factor is still
    heading toward 1, then "capacity-capped" and "collapsed in-sample residual"
    are one story, not two.
    """
    rows: list[dict[str, Any]] = []
    for rounds in ROUND_LADDER:
        booster = _train(config, x, y, rounds)
        resid = y - np.asarray(booster.predict(x), dtype=np.float64).reshape(-1)
        rows.append(
            {
                "rounds": rounds,
                "trees": int(booster.num_trees()),
                "train_mse": float(np.mean(resid * resid)),
                "smear": smearing_factor(resid),
            }
        )
    return rows


def probe_asset(
    data: AssetData,
    origins: Sequence[int],
    *,
    folds: int = DEFAULT_FOLDS,
    ladder_every: int = 0,
) -> list[dict[str, Any]]:
    """Fit ``lgbm`` at each refit origin and read all three residual scales."""
    config = LightGBMRV()
    splitter = ARM.splitter(1)
    fit_series = data.fit_series(ARM.invalid_target_policy)
    wanted = set(origins)
    rows: list[dict[str, Any]] = []
    seen = 0
    for origin in splitter.split(len(data.returns)):
        if origin.origin not in wanted:
            continue
        row: dict[str, Any] = {"asset": data.asset, "origin": int(origin.origin)}
        try:
            window = validated_rv(fit_series.window(origin.train), minimum=1)
            x, y = _design_matrix(window)
            fitted = config.fit(window)
            resid = y - np.asarray(fitted.booster.predict(x), dtype=np.float64).reshape(-1)
            oof = out_of_fold_residuals(config, x, y, folds)
            row.update(
                {
                    "n_rows": int(y.size),
                    "smear_in_sample": float(fitted.smear),
                    "resid_var_in_sample": float(np.mean(resid * resid)),
                    "smear_out_of_fold": smearing_factor(oof) if oof.size else math.nan,
                    "resid_var_out_of_fold": float(np.mean(oof * oof)) if oof.size else math.nan,
                    "n_oof": int(oof.size),
                    "probe_error": None,
                }
            )
            if ladder_every and seen % ladder_every == 0:
                row["ladder"] = json.dumps(round_ladder(config, x, y))
        except Exception as exc:  # a probe must never take the report down
            row["probe_error"] = f"{type(exc).__name__}: {exc}"
        seen += 1
        rows.append(row)
    return rows


def _refit_origins(store: ResultsStore, manifest: pd.DataFrame, asset: str) -> list[int]:
    rows = manifest[(manifest["asset"] == asset) & (manifest["model"] == "lgbm")]
    if rows.empty:
        return []
    frame = store.read(str(rows["config_hash"].iloc[0]))
    refits = frame.loc[frame["refit"] & (frame["fit_origin"] >= 0), "origin_index"]
    return sorted({int(v) for v in refits})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/P3_GRID_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/lgbm_smear_probe/probe.parquet"))
    parser.add_argument("--assets", nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--ladder-every",
        type=int,
        default=0,
        help="run the boosting-round ladder at every Nth refit origin (0 = never)",
    )
    args = parser.parse_args(argv)

    store = ResultsStore(args.store)
    manifest = pd.DataFrame(json.loads(args.manifest.read_text(encoding="utf-8"))["cells"])
    panel = build_panel()
    if args.assets:
        panel = {k: v for k, v in panel.items() if k in set(args.assets)}
    rows: list[dict[str, Any]] = []
    for name, series in panel.items():
        datum = asset_data(series)
        started = time.perf_counter()
        rows.extend(
            probe_asset(
                datum,
                _refit_origins(store, manifest, name),
                folds=args.folds,
                ladder_every=args.ladder_every,
            )
        )
        print(f"  {name:9s} {time.perf_counter() - started:7.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"\n{len(frame)} probed fits -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.argv[0] = "lgbm_smearing_probe"
    raise SystemExit(main())
