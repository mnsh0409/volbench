#!/usr/bin/env python
"""The panel's content digests, committed so a clean checkout can verify itself.

Every config hash in the study is built over ``series_sha256``,
``fit_series_sha256``, ``proxy.sha256`` and ``raw_sha256``
(``volbench.evaluate.run_backtest``), and until now those existed **only** in
the store's sidecars — which live under ``data/`` and are gitignored, because
``tests/test_licensing_guard.py::TestNoDataIsTracked`` forbids tracking
anything there and is right to. The consequence was that a clean checkout could
*run* the study but could not check that it had rebuilt the right inputs: a
silently different Stooq archive would produce a different grid with no error
anywhere, only a full cache miss that looks like an empty store.

A digest manifest under ``docs/`` closes that. It records no data — a SHA-256 of
a float64 buffer is not redistribution of the buffer — so the licensing guard is
untouched, and it is the same recipe ``run_backtest`` hashes rather than a
parallel one:

* ``raw_sha256`` — the source archive, as ``volbench.data.panel`` recorded it
  (``null`` for the two crypto series, which are assembled from the exchange
  API rather than a file).
* ``series_sha256`` — the log-return series after the driver's leading trim.
* ``proxy.sha256`` — the asset's own primary target (D-016 / D-004), likewise.
* ``fit_series_sha256`` — the variance series the variance-fed models fit on,
  under the study's D-018 compaction policy, which is the one the primary arm
  ran with.

Write it, or check the tree against it::

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    uv run python -m volbench.benchmarks.data_digests --check

``--check`` exits non-zero on the first disagreement and prints which asset and
which digest moved. That is the whole point: a mismatch means the inputs are
not the study's inputs, and every number computed from them is a different
experiment wearing the same name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

from volbench.benchmarks.grid_primary import ARM, asset_data
from volbench.data.panel import build_panel
from volbench.results import array_digest

__all__ = ["DEFAULT_MANIFEST", "compare", "panel_digests"]

#: Where the committed manifest lives. Under ``docs/``, never under ``data/``.
DEFAULT_MANIFEST: Final = Path("docs/P3_DATA_DIGESTS.json")


def panel_digests() -> dict[str, Any]:
    """Rebuild the panel and digest it, in the grid driver's own bridge.

    ``asset_data`` is called rather than reimplemented: it applies the leading
    trim, and a digest of an untrimmed series would be a digest of something
    the study never scored.
    """
    entries: list[dict[str, Any]] = []
    for name, series in sorted(build_panel().items()):
        datum = asset_data(series)
        returns = datum.returns.to_numpy(dtype=np.float64)
        proxy = datum.proxy.to_numpy(dtype=np.float64)
        fit_series = datum.fit_series(ARM.invalid_target_policy)
        entries.append(
            {
                "asset": name,
                "source": series.source,
                "role": series.role,
                "panel_start": str(datum.returns.index[0].date()),
                "panel_end": str(datum.returns.index[-1].date()),
                "n": int(returns.size),
                "raw_sha256": series.raw_sha256,
                "series_sha256": array_digest(returns),
                "proxy": {"name": datum.proxy_name, "sha256": array_digest(proxy)},
                "fit_series_sha256": array_digest(fit_series.values),
                "invalid_target_policy": ARM.invalid_target_policy,
            }
        )
    return {
        "what": "content digests of the primary grid's inputs; see the module docstring",
        "arm": ARM.label,
        "assets": entries,
    }


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Every disagreement between two manifests, as readable lines.

    Missing and extra assets are reported as disagreements too — a panel that
    gained or lost a series is not the panel the study ran on.
    """
    left = {entry["asset"]: entry for entry in expected["assets"]}
    right = {entry["asset"]: entry for entry in actual["assets"]}
    problems = [f"{a}: absent from the rebuilt panel" for a in sorted(set(left) - set(right))]
    problems += [f"{a}: present but not in the manifest" for a in sorted(set(right) - set(left))]
    for asset in sorted(set(left) & set(right)):
        for key in ("raw_sha256", "series_sha256", "fit_series_sha256", "n"):
            if left[asset].get(key) != right[asset].get(key):
                problems.append(
                    f"{asset}.{key}: manifest {left[asset].get(key)!r} != "
                    f"rebuilt {right[asset].get(key)!r}"
                )
        if left[asset]["proxy"] != right[asset]["proxy"]:
            problems.append(
                f"{asset}.proxy: manifest {left[asset]['proxy']} != rebuilt {right[asset]['proxy']}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the rebuilt panel against the committed manifest instead of writing it",
    )
    args = parser.parse_args(argv)

    actual = panel_digests()
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{len(actual['assets'])} assets -> {args.out}")
        return 0

    expected = json.loads(args.out.read_text(encoding="utf-8"))
    problems = compare(expected, actual)
    if problems:
        print(f"DIGEST MISMATCH ({len(problems)}):")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"digests match on all {len(actual['assets'])} assets ({args.out})")
    return 0


if __name__ == "__main__":
    sys.argv[0] = "data_digests"
    raise SystemExit(main())
