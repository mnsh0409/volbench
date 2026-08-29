#!/usr/bin/env python
"""The primary grid: 11 assets x 13 model configs, h=1, window 500, refit 21.

The driver that produced every number in the primary grid — docs/P3_GRID.md and
docs/P3_GRID_manifest.json are its output. It lives here, beside
``benchmarks.toy`` and ``benchmarks.smoke_tsfm``, because a study driver is
part of the study: reproducibility is this project's claim, and a reader who
can install the package but cannot run the study cannot check it.

It was originally written to ``data/grid_primary/run_grid.py``, next to the
store it filled, and was therefore not committed — ``/data/`` is gitignored and
``tests/test_licensing_guard.py::TestNoDataIsTracked`` forbids tracking
anything under it absolutely. That rule is right and unchanged; the error was
putting the driver under ``data/`` at all. See docs/P3_DRIVER_PROVENANCE.md.

**The outputs are split, and the split is the point.** The *store* — the
parquet fragments and their sidecars — goes under ``--out-dir`` (default
``data/grid_primary``) and can never be tracked: it holds values derived from
Stooq and Binance data. The *manifest* goes under ``--manifest-dir`` (default
``docs/``) and is committed: it holds config hashes, package and interpreter
versions, environment pins, timings and per-cell status, and no series value
anywhere. The store is the results; the manifest is the only thing that says
which of an append-only store's fragments are the current ones, so a manifest
that lands under ``data/`` leaves a clean checkout unable to tell. See
docs/P3_MANIFEST_INVENTORY.md, and ``benchmarks.manifest_provenance`` for the
two digests the manifest carries about itself.

Protocol (the arm's settings are all in every cell's config hash):
    horizon 1, window 500 (D-019), step 1, refit_every 21 with daily
    re-conditioning (D-015), invalid-target policy at its D-018 default.

Scoring target is each asset's own primary (D-017: a property of the cell):
``overnight_plus_range`` on the nine equity series (D-016),
``realized_variance`` on the two crypto series (D-004). ``build_crypto_series``
states in terms why crypto is not scored on the range estimator: its
"overnight" term is a one-minute gap on a 24/7 market.

Determinism: this refuses to start unless D-026's kernel pin and D-032's
thread pin are in force, because both must be set before numpy is imported and
neither can be repaired from in here. Run it through the Makefile's exports::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run --extra classical --extra tsfm python -m volbench.benchmarks.grid_primary

Run it from the repository root: ``--out-dir`` defaults to a path relative to
the working directory, the same convention ``benchmarks.toy`` uses.

Resumable by construction: the ResultsStore is append-only and ``run_grid``
skips any cell whose config hash is already stored, so re-running after an
interruption adds only the missing cells and leaves existing fragments
byte-identical and unrewritten. The manifest is written on every run and
covers every cell, cached ones included.

``--tag`` names the run. The default tag owns ``docs/P3_GRID_manifest.json``,
which is *always* the current grid manifest; the manifest it replaces is
archived under ``docs/archive/`` before the run starts, never overwritten, so
the evidence anchored to a superseded grid stays checkable. Any other tag —
a resume check, a determinism gate — writes a sibling that reads as one. A run
restricted with ``--assets``/``--models`` is refused under the default tag: a
partial grid must not end up wearing the whole grid's name.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# The pins, checked before numpy is imported by anything below.
# ---------------------------------------------------------------------------

KERNEL_PIN = "X86_V4 AVX512_ICL AVX512_SPR"


def _check_pins() -> None:
    problems = []
    if os.environ.get("NPY_DISABLE_CPU_FEATURES") != KERNEL_PIN:
        problems.append(
            f"  NPY_DISABLE_CPU_FEATURES is {os.environ.get('NPY_DISABLE_CPU_FEATURES')!r}, "
            f"expected {KERNEL_PIN!r}  (D-026)"
        )
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(var) != "1":
            problems.append(f"  {var} is {os.environ.get(var)!r}, expected '1'  (D-032)")
    if problems:
        raise SystemExit(
            "refusing to run: the determinism pins are not in force.\n"
            + "\n".join(problems)
            + "\n\nBoth must be set before numpy is imported, so this cannot be fixed from\n"
            "inside the process. See the Makefile."
        )


_check_pins()

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volbench.benchmarks.manifest_provenance import (  # noqa: E402
    ARCHIVE_DIR,
    annotate,
    supersede,
)
from volbench.data.panel import PanelSeries, build_panel  # noqa: E402
from volbench.data.proxies import log_returns  # noqa: E402
from volbench.determinism import environment_report  # noqa: E402
from volbench.execute import ProcessExecutor  # noqa: E402
from volbench.models import (  # noqa: E402
    EWMA,
    GARCH,
    HAR,
    AutoARIMARV,
    AutoETSRV,
    Chronos,
    LightGBMRV,
    Moirai,
    NaiveVol,
    PatchTST,
    TimesFM,
    gjr_garch,
)
from volbench.results import ResultsStore  # noqa: E402
from volbench.runner import (  # noqa: E402
    AssetData,
    CellOutcome,
    GridSpec,
    ModelConfig,
    ProtocolArm,
    RunManifest,
    run_grid,
)

#: Where the store and the run reports go. Relative to the working directory,
#: not to this file: this module lives inside the package, and a
#: ``Path(__file__).parent`` default would write a study's results into the
#: source tree. Same convention as ``benchmarks.toy`` (``data/toy_benchmark``).
DEFAULT_OUT_DIR = Path("data/grid_primary")

#: Where the *manifest* goes — under ``docs/``, in version control, not beside
#: the store. The store holds values derived from licensed market data and can
#: never be tracked (docs/data_licenses.md); the manifest holds config hashes,
#: versions, environment pins, timings and per-cell status and no series value
#: at all, which is what has always let ``docs/P3_GRID_manifest.json`` be
#: committed. Defaulting it here is what stops the two from drifting apart:
#: a manifest that lands under ``data/`` is a manifest a clean checkout cannot
#: read, and the study's current fragment set is then unidentifiable.
DEFAULT_MANIFEST_DIR = Path("docs")

#: The tag of the study run itself. Its manifest is *the* current grid manifest
#: (``manifest_provenance.CURRENT_MANIFEST``); every other tag names a
#: verification or exploratory run and gets a sibling that cannot be mistaken
#: for it.
DEFAULT_TAG = "primary"

SEED = 20260825
ARM = ProtocolArm(label="headline", window=500, refit_every=21, step=1, recondition="daily")


def manifest_name(tag: str) -> str:
    """The manifest filename for a run tag.

    The default tag owns the canonical name, so the committed current manifest
    is what a default run produces rather than something a human remembers to
    copy afterwards. Any other tag is a verification run and is named so that
    it reads as one.
    """
    return "P3_GRID_manifest.json" if tag == DEFAULT_TAG else f"P3_GRID_manifest_{tag}.json"


# ---------------------------------------------------------------------------
# the 13 configs
# ---------------------------------------------------------------------------


def model_configs(*, device: str = "cuda") -> list[ModelConfig]:
    """docs/research_design.md's model list, minus TimeGPT, plus ``garch11_t``.

    Thirteen: the design's twelve non-TimeGPT configs (which include GJR) and
    the Student-t GARCH that D-014 introduced and D-032 was measured on. TimeGPT
    is out because it is an API model behind a key and the design excludes it
    from the headline where access is unstable.

    Factories are module-level classes or ``functools.partial`` over them, never
    lambdas: the process backend pickles them across a boundary.

    ``lane`` is declared, never inferred from the label (D-027). The four
    torch-backed configs share the one GPU and are serialized on it.
    """
    return [
        ModelConfig("naive", NaiveVol),
        ModelConfig("ewma", functools.partial(EWMA, lambda_=0.94)),
        ModelConfig("garch11", functools.partial(GARCH, o=0, dist="normal")),
        ModelConfig("garch11_t", functools.partial(GARCH, o=0, dist="studentst")),
        ModelConfig("gjr", functools.partial(gjr_garch, dist="normal")),
        ModelConfig("har", HAR, fits_on_variance=True),
        ModelConfig("autoets", AutoETSRV, fits_on_variance=True),
        ModelConfig("autoarima", AutoARIMARV, fits_on_variance=True),
        ModelConfig("lgbm", LightGBMRV, fits_on_variance=True),
        ModelConfig(
            "chronos",
            functools.partial(Chronos, device=device),
            fits_on_variance=True,
            lane="gpu",
        ),
        ModelConfig("timesfm", TimesFM, fits_on_variance=True, lane="gpu"),
        ModelConfig(
            "moirai",
            functools.partial(Moirai, device=device),
            fits_on_variance=True,
            lane="gpu",
        ),
        ModelConfig(
            "patchtst",
            functools.partial(PatchTST, seed=SEED, device=device),
            fits_on_variance=True,
            lane="gpu",
        ),
    ]


# ---------------------------------------------------------------------------
# panel -> AssetData
# ---------------------------------------------------------------------------


def asset_data(series: PanelSeries) -> AssetData:
    """One panel series as the runner's inputs, on one calendar.

    The leading trim is the same one ``benchmarks.toy.load_series`` applies and
    for the same reason: ``log_returns`` and the overnight term have no
    ``C_{t-1}`` on the first bar. It is applied identically to returns, proxy
    and fit series, and it removes rows from the *start* only, so it moves no
    information backwards in time. On the seven index series the archive
    predates the panel window and nothing is trimmed.

    Returns must be finite everywhere after the trim — a NaN return would be a
    data defect, not a modelling one, and it would silently poison every fit
    window containing it. The *proxy* may carry NaNs and does (109 panel days
    whose close printed outside their own session range); those are D-018's
    business and are recorded per row as ``missing_reason``, never dropped.
    """
    returns = log_returns(series.frame.close)
    target = series.primary  # the asset's own primary: D-016 equity, D-004 crypto
    if not returns.index.equals(target.index):
        raise ValueError(f"{series.asset_id}: returns and target are not on one calendar")

    first = returns.first_valid_index()
    if first is None:
        raise ValueError(f"{series.asset_id}: no finite returns")
    keep = returns.index >= first
    returns, target = returns[keep], target[keep]

    values = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{series.asset_id}: {bad} non-finite returns after the leading trim")

    return AssetData(
        asset=series.asset_id,
        returns=returns,
        proxy=target,
        proxy_name=series.primary_target,
        variance=target,
        data_spec={
            "source": series.source,
            "role": series.role,
            "panel_start": str(returns.index[0].date()),
            "panel_end": str(returns.index[-1].date()),
            "raw_sha256": series.raw_sha256,
        },
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def missing_reason_counts(store: ResultsStore, manifest: RunManifest) -> dict[str, dict[str, int]]:
    """Per cell, how many rows carry each kind of ``missing_reason``.

    ``n_missing`` on the manifest is the total; this says what they were. Read
    off the stored fragments rather than kept in memory, so it works just as
    well on a resumed run whose cells were all cache hits.
    """
    out: dict[str, dict[str, int]] = {}
    for cell in manifest.cells:
        if cell.config_hash is None or not store.has(cell.config_hash):
            continue
        frame = store.read(cell.config_hash)
        reasons = frame["missing_reason"].astype(str)
        reasons = reasons[reasons != ""]
        if reasons.empty:
            continue
        # "fit_error@499: ValueError: ..." -> "fit_error/ValueError"
        kinds = Counter(
            f"{r.split('@', 1)[0].split(':', 1)[0]}/{r.split(': ', 2)[1] if ': ' in r else '?'}"
            for r in reasons
        )
        out[f"{cell.asset}/{cell.model}"] = dict(sorted(kinds.items()))
    return out


def peak_rss_gib() -> float:
    """Peak RSS of this process and every worker it reaped, in GiB."""
    kb = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    return kb / (1024.0 * 1024.0)


def report(manifest: RunManifest, reasons: dict[str, dict[str, int]], elapsed: float) -> str:
    rows = [c.as_json() for c in manifest.cells]
    frame = pd.DataFrame(rows)
    lines: list[str] = []
    add = lines.append

    add(f"cells attempted {len(manifest.cells)}  "
        f"computed {manifest.n_computed}  cached {manifest.n_cached}  failed {manifest.n_failed}")
    add(f"wall clock {elapsed / 60:.1f} min   peak RSS {peak_rss_gib():.2f} GiB")
    add("")

    add("per model:")
    by_model = frame.groupby("model").agg(
        cells=("index", "size"),
        failed=("status", lambda s: int((s == "failed").sum())),
        rows=("n_rows", "sum"),
        missing=("n_missing", "sum"),
        fits=("n_fits", "sum"),
        fallback=("n_fits_fallback", "sum"),
        nonconv=("n_fits_nonconverged", "sum"),
        wall_s=("wall_clock_s", "sum"),
    )
    by_model["wall_min"] = (by_model.pop("wall_s") / 60).round(2)
    add(by_model.to_string())
    add("")

    if manifest.n_failed:
        add("FAILURES:")
        for cell in manifest.failures:
            add(f"  {cell.asset}/{cell.model}/h{cell.horizon}/{cell.arm}: {cell.error}")
        add("")

    if reasons:
        add("missing_reason rows, per cell:")
        for key, kinds in sorted(reasons.items()):
            add(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    else:
        add("missing_reason rows: none, in any cell")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="the store and the run reports"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="the manifest, which is committed and carries no series values",
    )
    parser.add_argument("--assets", nargs="*", default=None, help="default: the whole panel")
    parser.add_argument("--models", nargs="*", default=None, help="default: all 13 configs")
    parser.add_argument("--cpu-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    args = parser.parse_args(argv)

    if args.tag == DEFAULT_TAG and (args.assets or args.models):
        parser.error(
            f"--assets/--models restrict the grid, so this run is not the study run, but "
            f"--tag is still {DEFAULT_TAG!r} and its manifest would be written to "
            f"{args.manifest_dir / manifest_name(DEFAULT_TAG)} — the committed manifest of "
            f"the whole grid. Give the run its own --tag."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    out_dir = args.out_dir
    store = ResultsStore(out_dir / "store")

    print("environment:")
    print(json.dumps(environment_report(), indent=2, default=str))

    t0 = time.perf_counter()
    panel = build_panel()
    if args.assets:
        panel = {k: v for k, v in panel.items() if k in set(args.assets)}
        missing = set(args.assets) - set(panel)
        if missing:
            raise SystemExit(f"unknown assets: {sorted(missing)}")
    data = {name: asset_data(series) for name, series in panel.items()}
    print(f"panel: {len(data)} assets in {time.perf_counter() - t0:.1f}s")
    for name, datum in data.items():
        print(f"  {name:9s} n={len(datum.returns):5d}  target={datum.proxy_name}")

    configs = model_configs(device=args.device)
    if args.models:
        wanted = set(args.models)
        configs = [c for c in configs if c.label in wanted]
        missing = wanted - {c.label for c in configs}
        if missing:
            raise SystemExit(f"unknown models: {sorted(missing)}")

    grid = GridSpec(
        assets=tuple(sorted(data)),
        models=tuple(configs),
        horizons=(1,),
        arms=(ARM,),
        seed=SEED,
    )
    print(f"\ngrid: {grid.size} cells "
          f"({len(grid.assets)} assets x {len(grid.models)} configs x 1 horizon x 1 arm)")

    def progress(outcome: CellOutcome) -> None:
        flag = "" if outcome.status != "failed" else "  <-- FAILED"
        print(
            f"  [{outcome.index:3d}] {outcome.asset:9s} {outcome.model:10s} "
            f"lane={outcome.lane} {outcome.status:8s} rows={outcome.n_rows:5d} "
            f"missing={outcome.n_missing:4d} fits={outcome.n_fits:4d} "
            f"fb={outcome.n_fits_fallback:3d} nc={outcome.n_fits_nonconverged:3d} "
            f"{outcome.wall_clock_s:8.1f}s{flag}",
            flush=True,
        )

    manifest_path = args.manifest_dir / manifest_name(args.tag)
    superseded = supersede(manifest_path, args.archive_dir)
    if superseded is not None:
        print(f"\narchived the manifest this run replaces -> {superseded['archived_path']}")

    started = time.perf_counter()
    manifest = run_grid(
        grid,
        data,
        store,
        cpu_executor=ProcessExecutor(workers=args.cpu_workers),
        gpu_executor=ProcessExecutor(workers=1),
        manifest_path=manifest_path,
        on_cell=progress,
    )
    elapsed = time.perf_counter() - started

    reasons = missing_reason_counts(store, manifest)
    text = report(manifest, reasons, elapsed)
    print("\n" + text)

    summary: dict[str, Any] = {
        "tag": args.tag,
        "elapsed_s": elapsed,
        "peak_rss_gib": peak_rss_gib(),
        "missing_reasons": reasons,
    }
    (out_dir / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"report_{args.tag}.txt").write_text(text + "\n")

    # The manifest run_grid wrote is the run's own record; this adds the three
    # fields that say which manifest it is, which fragment set it names and
    # what it replaced. Digests, not commit SHAs: the release repository will
    # not carry this history, so a manifest has to be verifiable by
    # recomputation from the files a reader actually has.
    annotated = annotate(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        store_root=out_dir / "store",
        supersedes=superseded,
    )
    manifest_path.write_text(json.dumps(annotated, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest: {manifest_path}")
    print(f"  manifest_digest {annotated['manifest_digest']}")
    print(f"  store_digest    {annotated['store_digest']}")
    return 1 if manifest.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
