"""Local-only smoke run of the heavy models over the toy series.

The zero-shot foundation models (Chronos, TimesFM, Moirai) and the trained
PatchTST baseline stay OUT of ``volbench.benchmarks.toy`` and therefore out of
``make reproduce``: they need the ``tsfm`` extra (a CUDA torch and the
foundation-model backends), cached Hugging Face weights, and a GPU to be quick,
and none of that is available in CI. This module is their equivalent of the
toy benchmark — the same fixture, the same splitter, the same scoring target,
the same ``ResultsStore`` — run by hand on the GPU box::

    make smoke-tsfm                                # data/smoke_tsfm/
    uv run --extra classical --extra tsfm python -m volbench.benchmarks.smoke_tsfm \\
        --out-dir data/smoke_tsfm --refit-every 21

Same caveat as the toy benchmark, only more so: the series is synthetic, so
the numbers are a plausibility check on the *wiring* (weights load, contexts
are cut, quantile grids become variances, results land in the store with a
config hash that moves with the weights). **No number this module produces
belongs in the paper.**

What differs from the toy run, and why:

- ``refit_every`` defaults to 21, not 1. For the zero-shot adapters the
  cadence changes no number (``fit`` and ``update`` are the same operation;
  pinned in ``tests/test_models_tsfm_common.py``); for PatchTST, which runs
  frozen between refits, it is what keeps ~10 trainings from becoming 200.
- TimeGPT is not in the default set: it calls a paid remote API and cannot
  pin its weights (docs/design.md). ``--timegpt`` adds it; the key comes from
  ``NIXTLA_API_KEY`` only.
- The fitted specs (checkpoint commit hashes, crossing/clipping counts,
  PatchTST's epochs) are in the store's config files, not in the summary.

Determinism: each adapter's tsfm-/gpu-marked tests pin bit-identity on this
box; ``device`` is not part of any config hash, and PatchTST reproduces per
device class (models/patchtst.py). Two runs on the same device with the same
seed must therefore rebuild every parquet byte for byte.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from volbench.benchmarks.make_toy_asset import DEFAULT_PATH
from volbench.benchmarks.toy import (
    ASSET_ID,
    HORIZON,
    SCORING_TARGET,
    SEED,
    STEP,
    WINDOW,
    ModelEntry,
    ToyBenchmarkResult,
    _markdown,
    build_summary,
    load_series,
)
from volbench.evaluate import DEFAULT_LEVELS, Recondition, run_backtest
from volbench.models import Chronos, Moirai, PatchTST, TimeGPT, TimesFM
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

__all__ = ["DEFAULT_OUT_DIR", "REFIT_EVERY", "models", "run_smoke_tsfm"]

DEFAULT_OUT_DIR = Path("data/smoke_tsfm")
#: See the module docstring: irrelevant to the zero-shot adapters, and what
#: bounds PatchTST's training count.
REFIT_EVERY = 21


def models(*, device: str = "auto", timegpt: bool = False) -> list[ModelEntry]:
    """The heavy model set. Every one fits on the realized-variance series.

    Default checkpoints throughout (Chronos-Bolt-small, TimesFM 2.5 200M,
    Moirai 2.0-R-small); PatchTST at its documented default size. ``device``
    reaches only the adapters that take one and is hashed by none of them.
    """
    entries = [
        ModelEntry("chronos", functools.partial(Chronos, device=device), fits_on_variance=True),
        ModelEntry("timesfm", TimesFM, fits_on_variance=True),
        ModelEntry("moirai", functools.partial(Moirai, device=device), fits_on_variance=True),
        ModelEntry(
            "patchtst",
            functools.partial(PatchTST, seed=SEED, device=device),
            fits_on_variance=True,
        ),
    ]
    if timegpt:
        entries.append(
            ModelEntry("timegpt", functools.partial(TimeGPT, enabled=True), fits_on_variance=True)
        )
    return entries


def run_smoke_tsfm(
    *,
    out_dir: Path | str | None = DEFAULT_OUT_DIR,
    fixture: Path = DEFAULT_PATH,
    seed: int = SEED,
    window: int = WINDOW,
    refit_every: int = REFIT_EVERY,
    levels: Sequence[float] = DEFAULT_LEVELS,
    recondition: Recondition = "daily",
    entries: Sequence[ModelEntry] | None = None,
    device: str = "auto",
) -> ToyBenchmarkResult:
    """Run the heavy models over the toy series; same tables as the toy run.

    ``entries`` overrides :func:`models` — that is how the CPU test drives
    this function with weight-free fake backends and a two-epoch PatchTST.
    """
    toy = load_series(fixture)
    splitter = RollingOriginSplitter(
        window=window, horizon=HORIZON, step=STEP, refit_every=refit_every
    )
    store = ResultsStore(Path(out_dir)) if out_dir is not None else None
    chosen = list(entries) if entries is not None else models(device=device)

    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for entry in chosen:
        returns, proxy, fit_series = toy.inputs_for(entry, target=SCORING_TARGET)
        frame = run_backtest(
            entry.factory,
            returns,
            proxy,
            splitter,
            seed,
            asset=ASSET_ID,
            proxy_name=SCORING_TARGET,
            data_spec={"fixture": fixture.name, "generator": "volbench.benchmarks.make_toy_asset"},
            fit_series=fit_series,
            levels=levels,
            store=store,
            recondition=recondition,
        ).assign(label=entry.label)
        frames.append(frame)
        hashes[entry.label] = str(frame.attrs["config_hash"])

    results = pd.concat(frames, ignore_index=True)
    summary = build_summary(results, levels=tuple(float(x) for x in levels))
    n_origins = int(splitter.n_splits(toy.returns.size))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)
        (out / "summary.md").write_text(
            _markdown(summary, n_origins, title="TSFM / PatchTST smoke run (local only)"),
            encoding="utf-8",
        )
    return ToyBenchmarkResult(
        results=results, summary=summary, n_origins=n_origins, config_hashes=hashes
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local-only smoke run of the foundation models and PatchTST on the toy series."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refit-every", type=int, default=REFIT_EVERY)
    parser.add_argument("--recondition", choices=("daily", "none"), default="daily")
    parser.add_argument("--device", default="auto", help='"auto", "cuda", "cuda:0" or "cpu"')
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="labels to run (default: all of chronos timesfm moirai patchtst)",
    )
    parser.add_argument(
        "--timegpt", action="store_true", help="also run TimeGPT (needs NIXTLA_API_KEY)"
    )
    args = parser.parse_args()

    entries = models(device=args.device, timegpt=args.timegpt)
    if args.only:
        unknown = set(args.only) - {e.label for e in entries}
        if unknown:
            parser.error(f"unknown model labels: {sorted(unknown)}")
        entries = [e for e in entries if e.label in set(args.only)]

    result = run_smoke_tsfm(
        out_dir=args.out_dir,
        fixture=args.fixture,
        seed=args.seed,
        refit_every=args.refit_every,
        recondition=args.recondition,
        entries=entries,
        device=args.device,
    )
    print(f"origins: {result.n_origins}   rows: {len(result.results)}")
    print(result.summary.to_string(index=False))
    print(f"\nwrote {args.out_dir}/summary.csv, summary.md and one parquet fragment per model")
    for label, hash_value in sorted(result.config_hashes.items()):
        print(f"  {label:<10} {hash_value}")


if __name__ == "__main__":
    main()
