#!/usr/bin/env python
"""What each adapter's backend *could* report about a fit, measured.

D-032 shipped ``fit_diagnostics()`` and the results' ``fit_status`` column, and
three configs implement it: ``garch11``, ``garch11_t``, ``gjr``. The other ten
report nothing, so the primary grid's manifest carries ``n_fits = 0`` and a
fallback rate of ``nan`` for them. For ``naive`` and ``ewma`` that is exactly
right — they estimate nothing that can fail. For the other eight it is not, and
"never failed" and "never measured" are different claims that a table of
zeroes and NaNs cannot tell apart.

This module answers the prior question — *is there a signal at all?* — by
fitting each adapter at the primary grid's own scheduled refit origins and
reading whatever its backend exposes. It writes nothing to any
:class:`~volbench.results.ResultsStore`, changes no config hash, and touches no
fragment: it is a measurement of the gap, not a repair of it. See
docs/P3_INSTRUMENTATION_GAP.md.

The origins are read out of the completed store's ``fit_origin`` column rather
than re-derived from a splitter, so the probe fits exactly where the grid fit
and a probe/grid mismatch cannot be mistaken for a finding. The windows come
from the same ``AssetData.fit_series(policy)`` the runner hands a cell, so
D-018's compaction applies here as it did there.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical --extra tsfm python -m volbench.benchmarks.fit_diagnostics_probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from volbench.benchmarks.grid_primary import ARM, asset_data, model_configs
from volbench.data.panel import build_panel
from volbench.results import ResultsStore
from volbench.runner import AssetData, ModelConfig

__all__ = ["PROBES", "probe_grid", "probe_one"]


def _har(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """HAR's OLS solve. ``np.linalg.lstsq`` returns the design matrix's rank
    and singular values and ``models/har.py`` keeps only the coefficients, so
    the rank-deficiency signal exists but is discarded rather than absent."""
    from volbench.models._rv import validated_rv
    from volbench.models.har import _design_matrix

    x, y = _design_matrix(validated_rv(window, minimum=1))
    _, _residuals, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    return {
        "rank": int(rank),
        "full_rank": bool(rank == x.shape[1]),
        "cond": float(singular.max() / singular.min()) if singular.min() > 0 else math.inf,
        "smear": float(fitted.smear),
        "resid_var": float(fitted.resid_var),
    }


def _statsforecast(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """AutoETS/AutoARIMA. ``backend.model_`` carries the selection outcome and,
    for AutoARIMA, ``code`` — which is ``scipy.optimize.minimize``'s own
    ``status`` for the selected model's fit (statsforecast/arima.py line 923;
    the same value its "possible convergence problem" warning quotes)."""
    model = getattr(fitted.backend, "model_", None)
    if not isinstance(model, dict):
        return {"model_": None}
    out: dict[str, Any] = {}
    if "code" in model:
        out["code"] = int(model["code"])
    if "arma" in model:
        out["order"] = [int(v) for v in model["arma"]]
    if "method" in model:
        out["method"] = str(model["method"])
    for key in ("aicc", "loglik", "sigma2"):
        if key in model and model[key] is not None:
            value = float(model[key])
            out[key] = value if math.isfinite(value) else None
    return out


def _lgbm(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """LightGBM has no convergence criterion; boosting always terminates. The
    honest signal is whether it built the rounds it was asked for and whether
    any tree split at all — a booster of stumps is a constant forecast."""
    booster = fitted.booster
    trees = int(booster.num_trees())
    return {
        "trees_built": trees,
        "iterations": int(booster.current_iteration()),
        "requested": int(fitted.config.num_boost_round),
        "all_rounds_built": bool(trees >= int(fitted.config.num_boost_round)),
        "smear": float(fitted.smear),
    }


def _tsfm(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """Zero-shot (D-005): nothing is estimated, so there is no convergence to
    report. There *is* a per-origin post-processing record — how many quantile
    crossings had to be rearranged and how many quantiles were clipped at zero
    — and ``FittedTSFM.spec()`` already carries it. It needs a forecast to
    exist, so one is drawn here."""
    fitted.predict(1)
    meta = fitted.spec().get("rv_forecasts", {}).get("1", {})
    return {
        "n_context": int(fitted.spec()["n_context"]),
        "crossings_rearranged": int(meta.get("crossings_rearranged", -1)),
        "clipped_at_zero": int(meta.get("clipped_at_zero", -1)),
        "rv_mean": float(meta.get("mean", math.nan)),
    }


def _patchtst(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """PatchTST trains per origin and keeps a full training record on the
    fitted object — epochs run, the best epoch, whether early stopping fired.
    ``FittedPatchTST.spec()`` already exposes all of it."""
    spec = fitted.spec()
    return {
        key: spec[key]
        for key in (
            "epochs_run",
            "best_epoch",
            "stopped_early",
            "best_val_mse",
            "final_train_mse",
            "n_train_windows",
            "n_val_windows",
        )
        if key in spec
    }


def _none(fitted: Any, window: np.ndarray) -> dict[str, Any]:
    """Nothing is estimated: the empty ``fit_status`` is already correct."""
    return {}


#: Per config label: what to read off a fitted model, and how many scheduled
#: refits to sample. ``1`` means every one; ``n`` means every ``n``-th, which is
#: how the four torch-backed configs stay affordable.
PROBES: Final[dict[str, tuple[Callable[[Any, np.ndarray], dict[str, Any]], int]]] = {
    "naive": (_none, 1),
    "ewma": (_none, 1),
    "har": (_har, 1),
    "autoets": (_statsforecast, 1),
    "autoarima": (_statsforecast, 1),
    "lgbm": (_lgbm, 1),
    "chronos": (_tsfm, 10),
    "timesfm": (_tsfm, 10),
    "moirai": (_tsfm, 10),
    "patchtst": (_patchtst, 10),
}


def _fit_origins(store: ResultsStore, manifest: pd.DataFrame, asset: str, label: str) -> list[int]:
    """The origins the grid actually fitted at, read off its own fragments."""
    rows = manifest[(manifest["asset"] == asset) & (manifest["model"] == label)]
    if rows.empty:
        return []
    frame = store.read(str(rows["config_hash"].iloc[0]))
    refits = frame.loc[frame["refit"] & (frame["fit_origin"] >= 0), "origin_index"]
    return sorted({int(v) for v in refits})


def probe_one(
    config: ModelConfig,
    data: AssetData,
    origins: Sequence[int],
    reader: Callable[[Any, np.ndarray], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fit ``config`` at each origin and read what its backend exposes."""
    splitter = ARM.splitter(1)
    fit_series = (
        data.fit_series(ARM.invalid_target_policy)
        if config.fits_on_variance
        else None
    )
    wanted = set(origins)
    rows: list[dict[str, Any]] = []
    for origin in splitter.split(len(data.returns)):
        if origin.origin not in wanted:
            continue
        window = (
            fit_series.window(origin.train)
            if fit_series is not None
            else data.returns.to_numpy(dtype=np.float64)[origin.train]
        )
        record: dict[str, Any] = {
            "asset": data.asset,
            "model": config.label,
            "origin": int(origin.origin),
        }
        try:
            fitted = config.factory().fit(window)
            record["fit_error"] = None
            record.update(reader(fitted, window))
        except Exception as exc:  # a probe must never take the report down
            record["fit_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    return rows


def probe_grid(
    *,
    store_root: Path,
    manifest_path: Path,
    labels: Sequence[str],
    device: str,
    assets: Sequence[str] | None = None,
) -> pd.DataFrame:
    store = ResultsStore(store_root)
    manifest = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8"))["cells"])
    panel = build_panel()
    if assets:
        panel = {k: v for k, v in panel.items() if k in set(assets)}
    data = {name: asset_data(series) for name, series in panel.items()}
    by_label = {c.label: c for c in model_configs(device=device)}

    rows: list[dict[str, Any]] = []
    for label in labels:
        reader, stride = PROBES[label]
        if reader is _none:
            continue
        started = time.perf_counter()
        for asset, datum in data.items():
            origins = _fit_origins(store, manifest, asset, label)[::stride]
            rows.extend(probe_one(by_label[label], datum, origins, reader))
        print(f"  {label:10s} {time.perf_counter() - started:7.1f}s", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/P3_GRID_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/fit_probe/probe.parquet"))
    parser.add_argument("--models", nargs="*", default=sorted(PROBES))
    parser.add_argument("--assets", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    frame = probe_grid(
        store_root=args.store,
        manifest_path=args.manifest,
        labels=args.models,
        device=args.device,
        assets=args.assets,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"\n{len(frame)} probed fits -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.argv[0] = "fit_diagnostics_probe"
    raise SystemExit(main())
