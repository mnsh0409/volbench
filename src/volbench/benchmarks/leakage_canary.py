#!/usr/bin/env python
"""The leakage canary, run through the primary grid's own bridge.

``.claude/skills/leakage-check`` ends with a test rather than a checklist item:

    corrupt all data strictly after date T with noise; every forecast for
    targets <= T must be bit-identical. If not, there is leakage.

docs/P3_GRID.md §6 ran that for ``naive``, ``ewma``, ``garch11`` and ``har``.
This module is the same canary, driven by the same code, extended to the
configs where leakage is hardest to argue from source alone — a per-origin
trained network, a gradient-boosted regressor with a known out-of-fold
question, a zero-shot foundation model with a context window, and a model
selection procedure.

What makes it evidence rather than a re-assertion
=================================================
Three legs run, and all three are needed:

1. **Determinism.** The same clean inputs, twice, into two different stores so
   neither read is a cache hit. Bit-identity is the null hypothesis of legs 2
   and 3; without it "identical" and "differs" mean nothing. This is the leg
   that matters for ``patchtst``, which trains per origin with an optimizer
   and a dropout RNG in the path.
2. **Future corruption.** Raw OHLC strictly after the cutoff is corrupted, and
   every row at or before the cutoff must be bit-identical to the clean run.
3. **Past corruption.** Raw OHLC *at or before* the cutoff is corrupted, and
   the same rows must now **differ**. A canary that cannot fail proves nothing;
   a model reporting "past-corruption: identical" means the test is inert
   there, which is a worse finding than a leak because it invalidates leg 2.

The corruption goes in at the **raw CSV**, so the whole production path runs
on it: :func:`~volbench.data.stooq.ingest_manual_csv` ->
:func:`~volbench.data.panel.repair_bars` ->
:func:`~volbench.data.panel.build_targets` ->
:class:`~volbench.data.panel.PanelSeries` -> the grid driver's own
:func:`~volbench.benchmarks.grid_primary.asset_data` bridge ->
:func:`~volbench.runner.run_grid`. Nothing here is a stand-in for a stage.

Each bar's four prices are perturbed independently and the high and low are
then reset to the max and min of the four, so the corrupted bar is still a
valid bar. Scaling a bar by one factor would leave every within-bar log ratio
untouched — Rogers-Satchell and Parkinson are scale-free — and would corrupt
only the overnight term, which is a much weaker test than it looks.

The index arithmetic, which has already produced one false leak report: the
driver trims one leading bar (``log_returns`` has no ``C_{t-1}`` on the first
one), so a results-frame ``target_index`` of ``j`` is position ``j + 1`` of the
windowed OHLC frame. :data:`CUTOFF_TARGET_INDEX` is a results-frame index and
is converted once, here, rather than at each use.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical --extra tsfm python -m volbench.benchmarks.leakage_canary
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from volbench.benchmarks.grid_primary import ARM, SEED, asset_data, model_configs
from volbench.data.panel import (
    DEFAULT_RAW_ROOT,
    build_equity_series,
    equity_spec,
    resolve_equity_path,
)
from volbench.execute import SerialExecutor
from volbench.results import ResultsStore
from volbench.runner import AssetData, GridSpec, ModelConfig, RunManifest, run_grid

__all__ = [
    "ASSET",
    "CUTOFF_TARGET_INDEX",
    "DEFAULT_MODELS",
    "CanaryVerdict",
    "corrupt_archive",
    "run_canary",
]

#: The series the canary runs on, as in docs/P3_GRID.md §6.
ASSET: Final = "SPY"

#: Results-frame ``target_index`` of the last row required to be unaffected.
#: Everything strictly after it is what gets corrupted.
CUTOFF_TARGET_INDEX: Final = 560

#: The nine configs P3_GRID's canary did not cover, priority order first. The
#: five the extension is required to reach are the head of this list.
DEFAULT_MODELS: Final = ("patchtst", "lgbm", "chronos", "autoarima", "garch11_t")

#: How many bars before the cutoff the past-corruption leg disturbs. One refit
#: block: enough to be inside both the compared rows and the training windows
#: they rest on.
PAST_CORRUPTION_BARS: Final = 21

#: Relative size of the per-price perturbation. Large enough that no comparison
#: turns on floating-point noise.
CORRUPTION_SCALE: Final = 0.05


# --------------------------------------------------------------------------
# corrupting the raw archive
# --------------------------------------------------------------------------


def corrupt_archive(
    source: Path,
    dest: Path,
    *,
    corrupt_from: date | None = None,
    corrupt_through: date | None = None,
    corrupt_after: date | None = None,
    scale: float = CORRUPTION_SCALE,
    seed: int = 20260828,
) -> int:
    """Write a copy of a Stooq CSV with some rows' OHLC perturbed.

    Rows are selected by *date*, never by position: the raw file, the windowed
    frame and the results frame have three different origins, and a positional
    rule would be the fourth chance to make the off-by-one this canary exists
    to detect.

    ``corrupt_after`` selects rows strictly after a date (the future leg);
    ``corrupt_from``/``corrupt_through`` select an inclusive band (the past
    leg). Returns the number of rows actually altered — zero means the canary
    would be testing nothing, and callers check it.

    Dates are plain :class:`datetime.date`. The panel index is tz-aware UTC and
    the raw file's ``<DATE>`` column is tz-naive, and comparing the two raises
    rather than silently offsetting — which is the right behaviour and the
    reason the boundary is expressed in calendar days on both sides.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], lines[1:]
    columns = [c.strip("<>").lower() for c in header.split(",")]
    idx = {name: i for i, name in enumerate(columns)}
    rng = np.random.default_rng(seed)

    out: list[str] = [header]
    changed = 0
    for line in body:
        if not line.strip():
            continue
        fields = line.split(",")
        day = pd.Timestamp(fields[idx["date"]]).date()
        selected = (
            (corrupt_after is not None and day > corrupt_after)
            or (
                corrupt_from is not None
                and corrupt_through is not None
                and corrupt_from <= day <= corrupt_through
            )
        )
        if selected:
            prices = np.array(
                [float(fields[idx[k]]) for k in ("open", "high", "low", "close")], dtype=np.float64
            )
            perturbed = prices * (1.0 + scale * rng.standard_normal(4))
            open_, close = float(perturbed[0]), float(perturbed[3])
            high, low = float(perturbed.max()), float(perturbed.min())
            for key, value in (("open", open_), ("high", high), ("low", low), ("close", close)):
                fields[idx[key]] = f"{value:.6f}"
            changed += 1
        out.append(",".join(fields))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def _staged_archive(root: Path, asset: str) -> Path:
    """Path of ``asset``'s raw file inside a staging root, directories made."""
    spec = equity_spec(asset)
    path = root / spec.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# running one variant
# --------------------------------------------------------------------------


def _truncate(data: AssetData, n: int | None) -> AssetData:
    """The first ``n`` observations of an asset, on one calendar still.

    Used only by the cheap legs. Truncation removes rows from the *end*, so it
    cannot move information backwards; and it changes the content digests, so a
    truncated run can never be served a full-length cell's fragment.
    """
    if n is None:
        return data
    return dataclasses.replace(
        data,
        returns=data.returns.iloc[:n],
        proxy=data.proxy.iloc[:n],
        variance=None if data.variance is None else data.variance.iloc[:n],
    )


def _configs(labels: Sequence[str], device: str) -> list[ModelConfig]:
    """The grid's own configs, selected by label.

    Taken from :func:`~volbench.benchmarks.grid_primary.model_configs` rather
    than re-declared: a canary run against a re-declaration would prove
    something about the re-declaration.
    """
    by_label = {c.label: c for c in model_configs(device=device)}
    unknown = sorted(set(labels) - set(by_label))
    if unknown:
        raise SystemExit(f"unknown model labels: {unknown}; known: {sorted(by_label)}")
    return [by_label[label] for label in labels]


def _run_variant(
    *,
    raw_root: Path,
    cache_root: Path,
    store_root: Path,
    labels: Sequence[str],
    device: str,
    n_bars: int | None,
) -> tuple[dict[str, pd.DataFrame], RunManifest, pd.Index]:
    """Build the series through the driver's bridge and score ``labels`` on it.

    Returns the per-label result frames, the manifest, and the windowed OHLC
    frame's own index (which is what :data:`CUTOFF_TARGET_INDEX` is resolved
    against).
    """
    series = build_equity_series(ASSET, raw_root=raw_root, cache_root=cache_root)
    data = _truncate(asset_data(series), n_bars)
    configs = _configs(labels, device)
    grid = GridSpec(
        assets=(ASSET,), models=tuple(configs), horizons=(1,), arms=(ARM,), seed=SEED
    )
    store = ResultsStore(store_root)
    manifest = run_grid(
        grid,
        {ASSET: data},
        store,
        cpu_executor=SerialExecutor(),
        gpu_executor=SerialExecutor(),
    )
    frames: dict[str, pd.DataFrame] = {}
    for cell in manifest.cells:
        if cell.config_hash is not None and store.has(cell.config_hash):
            frames[cell.model] = store.read(cell.config_hash)
    return frames, manifest, series.frame.close.index


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

#: Excluded from the row comparison. ``config_hash`` is a digest of the whole
#: input series, so it differs between a clean and a corrupted run *by design*
#: — that is the cache doing its job (D-011), not a forecast changing.
_NOT_COMPARED: Final = ("config_hash",)


def _rows_at_or_before(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    return (
        frame.loc[frame["target_index"] <= cutoff]
        .sort_values(["origin_index", "horizon"], kind="stable")
        .reset_index(drop=True)
    )


def _identical(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, list[str]]:
    """Bit-identity over every compared column. NaN equals NaN here.

    A NaN-blind comparison would call two unscorable rows different and two
    differently-unscorable rows the same; both are wrong for this purpose.
    """
    columns = [c for c in left.columns if c not in _NOT_COMPARED]
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False, ["shape or schema"]
    differing: list[str] = []
    for column in columns:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            same = bool(np.array_equal(a, b, equal_nan=True))
        else:
            same = bool(np.array_equal(a, b))
        if not same:
            differing.append(column)
    return not differing, differing


@dataclass(frozen=True)
class CanaryVerdict:
    """One model's two-line verdict, plus what it was measured over."""

    model: str
    n_compared_rows: int
    deterministic: bool
    future_identical: bool
    past_differs: bool
    future_differing_columns: tuple[str, ...] = ()
    determinism_differing_columns: tuple[str, ...] = ()

    @property
    def alive(self) -> bool:
        """Whether the canary can fail here at all."""
        return self.past_differs and self.deterministic

    @property
    def passed(self) -> bool:
        return self.alive and self.future_identical

    def lines(self) -> str:
        future = "identical" if self.future_identical else "DIFFERS  <-- LEAK"
        past = "differs (canary alive)" if self.past_differs else "IDENTICAL  <-- CANARY INERT"
        det = "" if self.deterministic else "   determinism: FAILED (clean vs clean differs)"
        return f"  {self.model:<10s} future-corruption: {future:<22s} past-corruption: {past}{det}"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_canary(
    *,
    work_root: Path,
    labels: Sequence[str] = DEFAULT_MODELS,
    device: str = "cuda",
    cutoff: int = CUTOFF_TARGET_INDEX,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
) -> list[CanaryVerdict]:
    """Run all three legs and return one verdict per model."""
    work_root.mkdir(parents=True, exist_ok=True)
    source = resolve_equity_path(equity_spec(ASSET), raw_root)

    clean_root = work_root / "raw_clean"
    shutil.copyfile(source, _staged_archive(clean_root, ASSET))

    # Resolve the cutoff to a date once, through the clean series, and convert
    # the results-frame index to the windowed frame's with the driver's trim.
    reference = build_equity_series(ASSET, raw_root=clean_root, cache_root=work_root / "cache")
    calendar = reference.frame.close.index
    cutoff_date = pd.Timestamp(calendar[cutoff + 1]).date()
    past_from = pd.Timestamp(calendar[cutoff + 1 - PAST_CORRUPTION_BARS]).date()
    n_full = len(calendar) - 1  # the driver's leading trim
    print(
        f"{ASSET}: {len(calendar)} windowed bars, {n_full} after the driver's leading trim\n"
        f"  cutoff target_index {cutoff} -> windowed position {cutoff + 1} "
        f"-> {cutoff_date}\n"
        f"  rows compared: target_index in [500, {cutoff}]"
    )

    future_root = work_root / "raw_future"
    n_future = corrupt_archive(
        source, _staged_archive(future_root, ASSET), corrupt_after=cutoff_date
    )
    past_root = work_root / "raw_past"
    n_past = corrupt_archive(
        source,
        _staged_archive(past_root, ASSET),
        corrupt_from=past_from,
        corrupt_through=cutoff_date,
        seed=20260829,
    )
    print(f"  corrupted bars: future leg {n_future}, past leg {n_past}")
    if not n_future or not n_past:
        raise SystemExit("refusing to report: a corruption leg altered no bars")

    short = cutoff + 1  # enough origins to produce every compared row, no more

    def run(tag: str, root: Path, n_bars: int | None) -> dict[str, pd.DataFrame]:
        started = time.perf_counter()
        frames, manifest, _ = _run_variant(
            raw_root=root,
            cache_root=work_root / f"cache_{tag}",
            store_root=work_root / f"store_{tag}",
            labels=labels,
            device=device,
            n_bars=n_bars,
        )
        print(
            f"  [{tag}] {len(frames)} cells, {manifest.n_failed} failed, "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        for cell in manifest.failures:
            print(f"      FAILED {cell.model}: {cell.error}")
        return frames

    print("\nleg 1/3  determinism (clean vs clean, short)")
    clean_a = run("clean_a", clean_root, short)
    clean_b = run("clean_b", clean_root, short)
    print("leg 2/3  past corruption (short)")
    past = run("past", past_root, short)
    print("leg 3/3  future corruption (full length)")
    clean_full = run("clean_full", clean_root, None)
    future = run("future", future_root, None)

    verdicts: list[CanaryVerdict] = []
    for label in labels:
        base = _rows_at_or_before(clean_a[label], cutoff)
        det_ok, det_cols = _identical(base, _rows_at_or_before(clean_b[label], cutoff))
        past_same, _ = _identical(base, _rows_at_or_before(past[label], cutoff))
        fut_ok, fut_cols = _identical(
            _rows_at_or_before(clean_full[label], cutoff),
            _rows_at_or_before(future[label], cutoff),
        )
        verdicts.append(
            CanaryVerdict(
                model=label,
                n_compared_rows=len(base),
                deterministic=det_ok,
                future_identical=fut_ok,
                past_differs=not past_same,
                future_differing_columns=tuple(fut_cols),
                determinism_differing_columns=tuple(det_cols),
            )
        )
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extend the leakage canary to more configs.")
    parser.add_argument("--work-root", type=Path, default=Path("data/leakage_canary"))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutoff", type=int, default=CUTOFF_TARGET_INDEX)
    args = parser.parse_args(argv)

    verdicts = run_canary(
        work_root=args.work_root, labels=args.models, device=args.device, cutoff=args.cutoff
    )
    print(f"\n{ASSET}: {verdicts[0].n_compared_rows} rows at or before target_index {args.cutoff}")
    for verdict in verdicts:
        print(verdict.lines())
    inert = [v.model for v in verdicts if not v.alive]
    leaked = [v.model for v in verdicts if v.alive and not v.future_identical]
    if inert:
        print(f"\nVERDICT: INCONCLUSIVE — the canary is inert for {inert}")
        return 2
    if leaked:
        print(f"\nVERDICT: FAIL — future corruption changed a past forecast for {leaked}")
        for verdict in verdicts:
            if verdict.model in leaked:
                print(f"  {verdict.model}: {list(verdict.future_differing_columns)}")
        return 1
    print("\nVERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.argv[0] = "leakage_canary"
    raise SystemExit(main())
