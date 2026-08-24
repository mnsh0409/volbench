"""The M1 toy benchmark: four baselines, 200 rolling origins, one scored table.

This is the end-to-end smoke signal for volbench — the first thing that runs
all three Phase 1 streams in series, on one series:

    data (ingest -> TimeSeriesFrame -> variance proxy)
      -> models (naive / EWMA / GARCH(1,1) / HAR-RV)
        -> evaluation (RollingOriginSplitter -> run_backtest -> ResultsStore)

It exists to prove the wiring holds and the numbers are reproducible, not to
say anything about the models. The series is synthetic (see
``make_toy_asset.py`` for why a real asset cannot be committed), so the
rankings below are a plausibility check and nothing more. **No number this
module produces belongs in the paper.**

Reproduce::

    make reproduce            # rebuilds the fixture, then this benchmark
    uv run python -m volbench.benchmarks.toy --out-dir data/toy_benchmark

Determinism: no code path here samples. Every model returns either a closed-
form ``Normal`` or a fixed quantile grid, and the GARCH optimizer is
deterministic given its data, so two runs with the same seed produce
byte-identical parquet. ``tests/test_m1_smoke.py`` is the canary for that.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volbench.benchmarks.make_toy_asset import DEFAULT_PATH
from volbench.data import load_ohlc_csv, log_returns, overnight_plus_range_variance, parkinson
from volbench.evaluate import DEFAULT_LEVELS, ModelFactory, Recondition, run_backtest
from volbench.models import EWMA, GARCH, HAR, NaiveVol
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

__all__ = [
    "ModelEntry",
    "ToyBenchmarkResult",
    "ToySeries",
    "build_summary",
    "load_series",
    "models",
    "run_toy_benchmark",
]

#: Trailing observations each model fits on. With 700 usable returns this
#: leaves exactly 200 origins at step=1, horizon=1.
WINDOW = 500
HORIZON = 1
STEP = 1
SEED = 20260823
ASSET_ID = "TOY"
#: Variance target for the return-fed models (naive, EWMA, GARCH): unchanged
#: since M1, so their numbers stay comparable across the M2 change below.
PROXY_NAME = "parkinson"
#: Variance target for range-fed models (HAR): the per-day overnight-plus-
#: Rogers-Satchell estimator of the *close-to-close* variance, which is the
#: quantity HAR's forecast is scored on (M1 report §4.4; docs/M2_NOTES.md).
HAR_TARGET = "overnight_plus_range"


@dataclass(frozen=True)
class ModelEntry:
    """One model in the benchmark: what it fits on, and what its variance is scored against.

    ``fits_on_variance`` is the whole reason this dataclass exists rather than
    a bare list of factories: HAR-RV takes a realized-variance series where
    every other baseline takes returns (models/har.py). ``run_backtest``
    supports that through ``fit_series``, and getting the flag wrong would
    quietly feed a model the wrong units rather than raise.

    ``target`` names the variance series the model is scored against on
    QLIKE — and, for a range-fed model, also fed as its input, so what it
    forecasts and what it is scored on are the same quantity.
    """

    label: str
    factory: ModelFactory
    fits_on_variance: bool = False
    target: str = PROXY_NAME


def models() -> list[ModelEntry]:
    """The toy model set: M1's four baselines plus a Student-t GARCH.

    The Student-t config was added at M2 so that `make reproduce` exercises
    the parametric ``StudentT`` path that closed M1 report §4.2 (D-014), not
    only the unit tests. HAR is fed and scored on the overnight-plus-range
    target (§4.4, D-016); the return-fed models keep Parkinson.

    Factories are module-level classes or ``functools.partial`` over them, so
    they stay picklable for the Phase 2 process/Slurm executors — a lambda
    here would work today and break the moment a backend crosses a process
    boundary.
    """
    return [
        ModelEntry("naive", NaiveVol),
        ModelEntry("ewma", functools.partial(EWMA, lambda_=0.94)),
        ModelEntry("garch11", functools.partial(GARCH, o=0, dist="normal")),
        ModelEntry("garch11_t", functools.partial(GARCH, o=0, dist="studentst")),
        ModelEntry("har", HAR, fits_on_variance=True, target=HAR_TARGET),
    ]


@dataclass(frozen=True, eq=False)
class ToyBenchmarkResult:
    """Everything one benchmark run produced. ``eq=False``: holds DataFrames."""

    results: pd.DataFrame
    summary: pd.DataFrame
    n_origins: int
    config_hashes: dict[str, str]


@dataclass(frozen=True, eq=False)
class ToySeries:
    """The fixture, ingested: returns plus every variance target, on one calendar.

    ``eq=False``: holds pandas objects.
    """

    returns: pd.Series
    targets: dict[str, pd.Series]

    def inputs_for(self, entry: ModelEntry) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        """``(series, proxy, fit_series)`` for ``run_backtest``, per the entry's contract."""
        proxy = self.targets[entry.target]
        return self.returns, proxy, (proxy if entry.fits_on_variance else None)


def load_series(path: Path = DEFAULT_PATH) -> ToySeries:
    """Ingest the fixture: log returns and the variance targets, all on one index.

    Everything comes back as pandas Series still carrying the fixture's
    calendar, so that ``run_backtest`` — which aligns its inputs by position
    — can verify they are on one index rather than take it on trust. The
    first bar is dropped from *every* series because ``log_returns`` and the
    overnight term have no ``C_{t-1}`` for it — a leading-edge trim of
    unusable rows, applied identically to all, which moves no information
    backwards in time.
    """
    frame = load_ohlc_csv(path, asset_id=ASSET_ID, source="synthetic")
    return_series = log_returns(frame.close)
    targets = {
        PROXY_NAME: parkinson(frame.high, frame.low),
        HAR_TARGET: overnight_plus_range_variance(frame.open, frame.high, frame.low, frame.close),
    }
    # Belt to run_backtest's braces: it re-checks the calendars on the way in,
    # but two helpers that fell out of step — one gaining a `dropna()`, say —
    # is a data-layer bug, and this is the data-layer side of the seam.
    for name, target in targets.items():
        if not return_series.index.equals(target.index):
            raise ValueError(f"returns and {name} are not on the same calendar")
    returns = return_series.iloc[1:]
    if not np.isfinite(returns.to_numpy(dtype=np.float64)).all():
        raise ValueError("returns contain non-finite values after the leading trim")
    trimmed: dict[str, pd.Series] = {}
    for name, target in targets.items():
        series = target.iloc[1:]
        values = series.to_numpy(dtype=np.float64)
        if not (np.isfinite(values) & (values > 0.0)).all():
            raise ValueError(f"{name} target must be finite and strictly positive")
        trimmed[name] = series
    return ToySeries(returns=returns, targets=trimmed)


def run_toy_benchmark(
    *,
    out_dir: Path | str | None = None,
    fixture: Path = DEFAULT_PATH,
    seed: int = SEED,
    window: int = WINDOW,
    refit_every: int = 1,
    levels: Sequence[float] = DEFAULT_LEVELS,
    use_store: bool = True,
    recondition: Recondition = "daily",
) -> ToyBenchmarkResult:
    """Run every baseline over the toy series and return the scored tables.

    Parameters
    ----------
    out_dir:
        Where the :class:`~volbench.results.ResultsStore` fragments and the
        summary CSV land. ``None`` runs entirely in memory.
    refit_every:
        Origins between scheduled re-estimations. The toy runs at 1 — every
        origin refits — because that is the M1 protocol its byte-identity gate
        pins. Above 1, ``recondition`` decides what happens in between.
    recondition:
        ``"daily"`` (default): parameters from the last scheduled refit, the
        model's conditional state re-filtered on each origin's window — the
        reading of "refit every N days" fixed after M1 report §4.3.
        ``"none"``: the forecast is frozen between refits — the pre-M2
        behaviour, kept as an explicit ablation arm. The ``conditioned_
        through`` column records which happened on every row.
    """
    toy = load_series(fixture)
    splitter = RollingOriginSplitter(
        window=window, horizon=HORIZON, step=STEP, refit_every=refit_every
    )
    store = ResultsStore(Path(out_dir)) if (use_store and out_dir is not None) else None

    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for entry in models():
        returns, proxy, fit_series = toy.inputs_for(entry)
        frame = run_backtest(
            entry.factory,
            returns,
            proxy,
            splitter,
            seed,
            asset=ASSET_ID,
            proxy_name=entry.target,
            data_spec={"fixture": fixture.name, "generator": "volbench.benchmarks.make_toy_asset"},
            fit_series=fit_series,
            levels=levels,
            store=store,
            recondition=recondition,
        )
        # `model` comes from the model's own `name`; `label` is the benchmark's
        # short handle for it. Keeping both means the table stays readable
        # without losing the identity that went into the config hash.
        frame = frame.assign(label=entry.label)
        frames.append(frame)
        hashes[entry.label] = str(frame.attrs["config_hash"])

    results = pd.concat(frames, ignore_index=True)
    summary = build_summary(results, levels=tuple(float(x) for x in levels))
    n_origins = int(splitter.n_splits(toy.returns.size))

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)
        (out / "summary.md").write_text(_markdown(summary, n_origins), encoding="utf-8")

    return ToyBenchmarkResult(
        results=results, summary=summary, n_origins=n_origins, config_hashes=hashes
    )


def _level_tag(level: float) -> str:
    """Mirrors ``evaluate._level_tag`` — the result columns are named by it."""
    return f"{level:.10g}".replace(".", "p").replace("-", "m")


def build_summary(
    results: pd.DataFrame, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> pd.DataFrame:
    """Average each score per model, plus realized VaR hit rates.

    Means skip NaN, and ``n_scored`` reports how many rows actually carried a
    score, so a model that failed to produce one on half its origins cannot
    look good by averaging over the half that worked.
    """
    rows: list[dict[str, Any]] = []
    for label, group in results.groupby("label", sort=True):
        row: dict[str, Any] = {
            "label": str(label),
            "model": str(group["model"].iloc[0]),
            "target": str(group["proxy_name"].iloc[0]),
            "n": len(group),
            "n_scored": int(group["crps"].notna().sum()),
            "crps": float(group["crps"].mean()),
            "log_score": float(group["log_score"].mean()),
            "qlike": float(group["qlike"].mean()),
            "mean_forecast_vol": float(np.sqrt(group["forecast_var"]).mean()),
        }
        for level in levels:
            tag = _level_tag(level)
            row[f"pinball_{tag}"] = float(group[f"pinball_{tag}"].mean())
            row[f"hitrate_{tag}"] = float(group[f"hit_{tag}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


def _markdown(summary: pd.DataFrame, n_origins: int) -> str:
    header = (
        f"# volbench toy benchmark (M1)\n\n"
        f"Synthetic series, {n_origins} rolling origins, h=1, daily units.\n"
        f"Smoke signal only — see `src/volbench/benchmarks/toy.py`. "
        f"No number here belongs in the paper.\n\n"
    )
    # Hand-rolled rather than `DataFrame.to_markdown`, which needs the
    # optional `tabulate` package — not worth a dependency for one table.
    columns = list(summary.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for record in summary.to_dict("records"):
        cells = [
            f"{record[c]:.6g}" if isinstance(record[c], float) else str(record[c])
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the volbench M1 toy benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/toy_benchmark"))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refit-every", type=int, default=1)
    parser.add_argument("--recondition", choices=("daily", "none"), default="daily")
    args = parser.parse_args()

    result = run_toy_benchmark(
        out_dir=args.out_dir,
        fixture=args.fixture,
        seed=args.seed,
        refit_every=args.refit_every,
        recondition=args.recondition,
    )
    print(f"origins: {result.n_origins}   rows: {len(result.results)}")
    print(result.summary.to_string(index=False))
    print(f"\nwrote {args.out_dir}/summary.csv, summary.md and one parquet fragment per model")
    for label, hash_value in sorted(result.config_hashes.items()):
        print(f"  {label:<8} {hash_value}")


if __name__ == "__main__":
    main()
