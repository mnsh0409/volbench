#!/usr/bin/env python
"""Per-asset loss tables and pairwise-complete accounting, out of the store.

Two deliverables, both read-only over a completed grid:

1. **Per asset, one 13-row table** of mean CRPS, log score, pinball (per level
   and averaged), QLIKE and FZ0 (per level and averaged), each with a
   Newey-West standard error that accounts for serial dependence, the
   bandwidth rule and chosen bandwidth, and the ``n`` actually used.
2. **Per asset and per loss, the 13x13 matrices** of ``n used`` on the
   intersection where both models score finitely, and of rows dropped against
   the asset's origin count.

**Nothing is aggregated across assets, and this module has no code path that
could be.** The nine equity series are scored against an overnight-plus-range
variance target and the two crypto series against 5-minute realized variance
(D-016, D-004); the levels are not comparable, so a pooled or averaged loss
over the eleven would be a mean over two different units. Cross-asset
summaries have to be rank-based and are a later prompt's.

**Reported, not interpreted.** No model is ranked here and no number is called
large or small.

Everything numeric comes from :mod:`volbench.analysis`, which is structurally
forbidden from importing the model package (``tests/test_analysis.py::
TestBoundary``) — so nothing in this pipeline can re-run a model. This module
only orchestrates it and renders markdown.

Run::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run python -m volbench.benchmarks.loss_tables
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from volbench import analysis
from volbench.results import ResultsStore

__all__ = [
    "LOSS_HEADINGS",
    "frame_markdown",
    "largest_drops",
    "loss_table_markdown",
    "pairwise_markdown",
    "read_grid",
]

#: Column heading per loss key, for the rendered tables only. The keys are
#: :data:`volbench.analysis.LOSS_ORDER` and are what the CSVs carry.
LOSS_HEADINGS: Final[dict[str, str]] = {
    "crps": "CRPS",
    "log_score": "log score",
    "pinball_0p01": "pinball 1%",
    "pinball_0p025": "pinball 2.5%",
    "pinball_0p05": "pinball 5%",
    "pinball_avg": "pinball avg",
    "qlike": "QLIKE",
    "fz0_0p01": "FZ0 1%",
    "fz0_0p025": "FZ0 2.5%",
    "fz0_0p05": "FZ0 5%",
    "fz0_avg": "FZ0 avg",
}


def read_grid(store_root: Path, manifest_path: Path) -> analysis.GridFrame:
    """The whole grid with the derived losses on it."""
    store = ResultsStore(store_root)
    manifest = analysis.load_manifest(manifest_path)
    return analysis.with_derived_losses(analysis.load_grid(store, manifest))


def _cell(mean: float, se: float) -> str:
    return f"{mean:.6g} ({se:.3g})"


def frame_markdown(frame: pd.DataFrame) -> str:
    """A frame as a markdown table, without pulling in ``tabulate``.

    ``DataFrame.to_markdown`` needs an optional dependency this project does
    not carry, and a report generator should not be the reason a dependency
    enters a study's environment.
    """
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def loss_table_markdown(table: pd.DataFrame, asset: str) -> str:
    """One asset's 13-row table, as markdown. ``table`` is :func:`analysis.loss_table`'s."""
    cell = table.loc[table["asset"] == asset]
    wide = cell.pivot(index="model", columns="loss", values="mean")
    ses = cell.pivot(index="model", columns="loss", values="se")
    counts = cell.pivot(index="model", columns="loss", values="n")
    bandwidths = sorted({int(b) for b in cell["bandwidth"]})
    origins = int(cell["origins"].iloc[0])

    losses = [loss for loss in analysis.LOSS_ORDER if loss in wide.columns]
    lines = [
        f"### {asset}",
        "",
        f"{origins} origins. Cells are **mean (Newey-West SE)**. "
        f"Bandwidth {'/'.join(str(b) for b in bandwidths)} "
        f"by `{analysis.HAC_BANDWIDTH_RULE}`.",
        "",
        "| model | n (dist.) | n (QLIKE) | "
        + " | ".join(LOSS_HEADINGS[loss] for loss in losses)
        + " |",
        "|---|---:|---:|" + "---:|" * len(losses),
    ]
    for model in wide.index:
        cells = [
            _cell(float(wide.loc[model, loss]), float(ses.loc[model, loss])) for loss in losses
        ]
        lines.append(
            f"| `{model}` | {int(counts.loc[model, 'crps'])} | {int(counts.loc[model, 'qlike'])} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def pairwise_markdown(result: analysis.PairwiseComplete) -> str:
    """One (asset, loss) pair of 13x13 matrices, as markdown."""
    models = result.models
    header = "| | " + " | ".join(f"`{m}`" for m in models) + " |"
    rule = "|---|" + "---:|" * len(models)
    lines = [f"#### {result.asset} — `{result.loss}` ({result.origins} origins)", ""]
    for title, frame in (("n used", result.n_used), ("rows dropped", result.dropped)):
        lines += [f"**{title}**", "", header, rule]
        values = frame.to_numpy()
        for index, model in enumerate(models):
            lines.append(
                f"| `{model}` | " + " | ".join(str(int(v)) for v in values[index]) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def largest_drops(long: pd.DataFrame) -> pd.DataFrame:
    """Per asset, the worst-overlapping (loss, pair) and by how much.

    The diagonal is included deliberately: ``dropped[i, i]`` is what model *i*
    loses against the asset's origin count on its own, and it lower-bounds
    every pair it appears in. The largest off-diagonal drop is reported
    alongside, because a comparison is what the matrices are for.
    """
    rows: list[dict[str, Any]] = []
    for asset, group in long.groupby("asset", observed=True, sort=True):
        worst = group.loc[group["dropped"].idxmax()]
        off = group.loc[group["model_a"] != group["model_b"]]
        worst_off = off.loc[off["dropped"].idxmax()]
        rows.append(
            {
                "asset": asset,
                "origins": int(str(worst["origins"])),
                "largest_drop": int(str(worst["dropped"])),
                "loss": str(worst["loss"]),
                "pair": f"{worst['model_a']} / {worst['model_b']}",
                "n_used": int(str(worst["n_used"])),
                "largest_off_diagonal_drop": int(str(worst_off["dropped"])),
                "off_diagonal_loss": str(worst_off["loss"]),
                "off_diagonal_pair": f"{worst_off['model_a']} / {worst_off['model_b']}",
            }
        )
    return pd.DataFrame(rows)


LOSS_PREAMBLE: Final = """# P3 — per-asset loss tables

Mean CRPS, log score, pinball (per level and averaged), QLIKE and FZ0 (per
level and averaged) for each of the 13 configs, on each of the 11 assets, with
a standard error that accounts for serial dependence and the `n` actually used.

**Reported, not interpreted.** No model is ranked, no number is called large or
small, and nothing here says which forecast is better.

**Nothing is aggregated across assets.** The nine equity series are scored
against an overnight-plus-range variance target (D-016) and the two crypto
series against 5-minute realized variance (D-004). The levels are not
comparable, so a pooled or averaged loss over the eleven would be a mean over
two different units. Cross-asset summaries have to be rank-based and are a
later prompt's; `volbench.benchmarks.loss_tables` has no code path that could
produce one.

| | |
|---|---|
| Grid | `data/grid_primary/manifest_fix.json` — the **post-fix** manifest (see the note below) |
| Rows | 645,151 = 11 assets x 13 configs x h=1 x arm `headline` |
| Generated by | `python -m volbench.benchmarks.loss_tables`; numerics in `volbench.analysis` |
| Machine-readable | `docs/P3_LOSS_TABLES.csv`, one row per (asset, model, loss) |
| Environment | P3's thread and kernel pins (`OMP`/`OPENBLAS` = 1, `NPY_DISABLE_CPU_FEATURES`) |

**Which grid this is.** `docs/P3_MODEL_DEFECT_FIXES.md` §3 re-ran 44 of the 143
cells after the LightGBM smearing and TSFM tail-closure fixes, moving those
cells' config hashes; `lgbm`, `chronos`, `timesfm` and `moirai` are affected on
all 11 assets. `docs/P3_GRID_manifest.json` — the manifest J1 was written
against — describes the **pre-fix** grid and its numbers for those four
configs are superseded. Everything here reads `manifest_fix.json`. The row
count, the missing-reason census and the GARCH-family fit counts are identical
across the two manifests; the 99 unaffected cells are byte-identical fragments.

## The two losses that are not stored

CRPS, log score, pinball at 0.01 / 0.025 / 0.05 and QLIKE are stored per row.
**FZ0 is not**, and is recomputed exactly from the four stored columns
`realized_return`, `var_<level>`, `es_<level>` at each level — Patton, Ziegel &
Chen (2019) eq. 6, written out in `volbench.analysis.fz0_loss` from the paper
rather than imported from `volbench.backtests`, for the reason
`analysis.py`'s module docstring gives. Its domain (`e < 0`, `e <= v`) was
checked over all 645,151 rows before it was used: no violation, at any level
(J1 §2).

FZ0 is reported per level **and** averaged, exactly as pinball is. FZ0 is a
per-level object like pinball, no level is privileged by the design, and a
single "mean FZ0" would hide which level it came from. `pinball_avg` and
`fz0_avg` are the
unweighted mean **across the three levels of one row**, which is a different
object from a mean across rows and is named so it cannot be mistaken for one.

## The standard errors

Newey-West with the Bartlett kernel, on the loss series ordered by origin.
Bandwidth rule: **`floor(4 * (n / 100) ** (2 / 9))`** — the deterministic
rule of thumb following Newey & West (1994), which is most software's default.
It gives **9** on the nine equity series and **8** on the two crypto series;
each table's caption states its own. `se_iid` in the CSV is the ordinary
`sqrt(gamma_0 / (n-1))` for the same series, so the inflation the serial
dependence buys is visible next to the number it applies to.

**How sensitive the numbers are to that bandwidth**, measured rather than
asserted. Recomputing all 1,573 standard errors at twice the rule's bandwidth
raises them by a median of **9 %** (5th to 95th percentile 1.7 % to 26 %, maximum
29 %); at four times, by a median of **19 %** (2.1 % to 60 %, maximum 68 %).
It is not uniform across losses — at twice the bandwidth the median rise is
26 % for CRPS and 20 % for the log score, against 3.5 % for QLIKE and for FZ0
at the 1 % level. A Bartlett kernel at a fixed rule-of-thumb bandwidth
**understates** the long-run variance of a highly persistent series (on a
simulated AR(1) at rho = 0.9, where the truth is known, it recovers 60 % of
it; at rho = 0.5, 94 %; at rho = 0, 99 % —
`tests/test_analysis.py::TestHAC`). Every standard error here is therefore a
lower bound in the direction that matters, by the amounts above.

**Holes.** A loss series with an unscorable day in it has a hole, and a HAC
estimator has no way to represent one: lag `j` across a hole spans more than
`j` days. Non-finite values are dropped and the remainder treated as adjacent;
the alternative — a zero deviation at the hole — biases the autocovariances
toward zero and does it silently. `n_dropped` is in the CSV next to every
number it affects, and it is 0 on 132 of the 143 cells: only NKX (21 rows on
the eight variance-fed configs, `InsufficientHistoryError` at the earliest
origins), TWSE (80), CAC (28), HSI (13) and NKX's one `proxy_nonpositive` day
have any at all, all on QLIKE except NKX's.

## What was checked before these numbers were published

- **Every published mean is the stored column's own mean, bit for bit.**
  All **858** (asset x model x loss) means over the five stored losses were
  recomputed straight off the fragments, outside this pipeline, as the mean of
  the column's finite values: **858 / 858 identical**, relative difference
  exactly 0.
- **FZ0 agrees with the package's own implementation exactly.**
  `analysis.fz0_loss` (written here from the paper) and
  `volbench.backtests.fz0_loss` (written independently, earlier, from the same
  paper) were compared over **1,934,949** finite rows — the whole grid, all
  three levels: **maximum absolute difference 0.000e+00**, and the two NaN
  sets identical. The analysis layer is structurally forbidden from importing
  `backtests`, which is what makes this an agreement and not a tautology.
- **Reading the CSV back.** It is written at `%.17g` so a float64 round-trips.
  `pandas.read_csv` truncates by default; read it with
  `float_precision="round_trip"` if bit-identical values matter.

**`n` differs by loss, within a cell.** `n (dist.)` is the count for CRPS, log
score, pinball and FZ0, which are NaN on exactly the same rows; `n (QLIKE)` is
QLIKE's, which is NaN on a strict superset of them (a non-positive or missing
proxy costs QLIKE and nothing else). Both are given per model. The full
per-loss `n` is in the CSV.

---

"""

PAIRWISE_PREAMBLE: Final = """# P3 — pairwise-complete accounting

Every model-versus-model comparison must run on the **intersection of origins
where both models have a finite score**. This file is that intersection,
measured per asset and per loss, so the next stage consumes it rather than
recomputing it.

**Reported, not interpreted.** These are counts of rows, not statements about
forecasts.

| | |
|---|---|
| Grid | `data/grid_primary/manifest_fix.json` — the post-fix manifest |
| Machine-readable | `docs/P3_PAIRWISE_COMPLETE.csv`, one row per pair |
| Generated by | `python -m volbench.benchmarks.loss_tables` |

`n_used[i, j]` counts the origins where **both** models score finitely;
`dropped[i, j]` is that subtracted from the asset's own origin count. Both
matrices are symmetric and carry each model's own scored count on the diagonal
— which is not decoration: `dropped[i, i]` lower-bounds every pair model *i*
appears in.

The pivot is on `origin_index`, never on row position, so "the same sample"
means the same days.

**Why this is not bookkeeping.** On NKX the variance-fed configs are scored on
4,773 QLIKE rows and the return-fed ones on 4,794, out of 4,795 origins. A
comparison between one of each that quietly used each side's own sample would
be comparing two different samples. The matrix says 4,773 for every such pair,
which is the number the comparison has to use.

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/grid_primary/store"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/grid_primary/manifest_fix.json")
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    grid = read_grid(args.store_root, args.manifest)
    assets: Sequence[str] = sorted(str(a) for a in grid["asset"].unique())
    print(f"grid {grid.shape} in {time.perf_counter() - started:.1f}s; {len(assets)} assets")

    # --- 1. loss tables -----------------------------------------------------
    table = pd.concat([analysis.loss_table(grid, asset) for asset in assets], ignore_index=True)
    # %.17g, not the default: pandas' default float formatting drops the last
    # digit of a float64, so a "machine-readable" table would not round-trip to
    # the number it was computed from. The markdown rounds on purpose; the CSV
    # must not.
    table.to_csv(args.docs / "P3_LOSS_TABLES.csv", index=False, float_format="%.17g")
    parts = [LOSS_PREAMBLE]
    parts += [loss_table_markdown(table, asset) + "\n" for asset in assets]
    (args.docs / "P3_LOSS_TABLES.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"loss tables: {table.shape} -> P3_LOSS_TABLES.{{csv,md}}")

    # --- 2. pairwise-complete ----------------------------------------------
    long = analysis.pairwise_complete_long(grid)
    long.to_csv(args.docs / "P3_PAIRWISE_COMPLETE.csv", index=False)
    patterns = analysis.missingness_patterns(grid)

    parts = [PAIRWISE_PREAMBLE, "## The distinct matrices\n"]
    parts.append(
        "Over the whole grid the eleven losses fall into exactly **"
        f"{len(patterns)} finite/NaN patterns**, checked element-wise over all\n"
        f"{len(grid):,} rows rather than assumed. Losses inside one pattern are NaN on\n"
        "precisely the same rows, so their matrices are the *same* matrix and are given\n"
        "once here. All eleven are in the CSV regardless.\n"
    )
    parts.append(frame_markdown(patterns) + "\n")
    parts.append("## Largest drop per asset\n")
    parts.append(
        "Where the largest drop is **0** the asset loses nothing anywhere, and the `loss` and "
        "`pair` columns name the first cell of a frame that is entirely zero — they carry no "
        "information in that case. `largest_off_diagonal_drop` is the worst *comparison*, which "
        "is what the matrices are for; the unqualified maximum can sit on the diagonal.\n"
    )
    parts.append(frame_markdown(largest_drops(long)) + "\n")
    parts.append("## The matrices\n")
    for asset in assets:
        parts.append(f"### {asset}\n")
        for _, pattern in patterns.iterrows():
            members = str(pattern["losses"]).split(", ")
            result = analysis.pairwise_complete(grid, asset, members[0])
            if len(members) > 1:
                parts.append(
                    f"Shared by {len(members)} losses: "
                    + ", ".join(f"`{m}`" for m in members)
                    + "\n"
                )
            parts.append(pairwise_markdown(result))
    (args.docs / "P3_PAIRWISE_COMPLETE.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"pairwise:    {long.shape} -> P3_PAIRWISE_COMPLETE.{{csv,md}}")

    print(patterns.to_string(index=False))
    print(largest_drops(long).to_string(index=False))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
