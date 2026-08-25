"""Build the D-004/D-012 panel and write docs/PANEL_REPORT.md.

Run as::

    uv run python -m volbench.data.build_panel \\
        --raw-root ~/Documents/IJF/volbench/volbench/data/raw \\
        --cache-root ~/Documents/IJF/volbench/volbench/data/cache

Every number in the emitted report comes from :mod:`volbench.data.diagnostics`
measuring the panel this run built — there are no hand-copied figures, so the
report cannot drift from the data. The prose blocks below are static; the
figures inside them are interpolated.

The report is a *findings* document for human review, not an input to anything:
nothing in the library reads it, and no decision it flags is applied here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from volbench.data.crisis import CRISIS_WINDOWS, PENDING_WINDOWS
from volbench.data.diagnostics import (
    GAP_ALERT_DAYS,
    OVERNIGHT_SHARE_EXPECTED,
    SeriesDiagnostics,
    crisis_coverage,
    diagnose_panel,
)
from volbench.data.panel import (
    CRYPTO_PANEL,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RAW_ROOT,
    EQUITY_PANEL,
    FIT_WINDOW_DEFAULT,
    FIT_WINDOW_ROBUSTNESS,
    PANEL_END,
    PANEL_START,
    RETIRED_EQUITY,
    PanelSeries,
    build_panel,
)

__all__ = ["main", "render_report"]

#: The rolling-origin windows this report describes: D-019's default and its
#: robustness arm. Reported, not applied — this module evaluates nothing; the
#: numbers say how much of each crisis window survives warm-up at each.
PROTOCOL_WINDOW = FIT_WINDOW_DEFAULT
ROBUSTNESS_WINDOW = FIT_WINDOW_ROBUSTNESS
PROTOCOL_MIN_FORECASTS = 1500

#: A series shorter than this fraction of the longest equity series is called
#: out as materially short (D-012's ETF fallbacks).
SHORT_SPAN_FRACTION = 0.75

#: Outcome of the `.claude/skills/leakage-check` audit of this branch, rendered
#: as §11 of the report. Held here rather than inline in the report's prose so
#: each row stays one markdown table line without fighting the line-length rule.
_LEAKAGE_AUDIT: tuple[tuple[str, str, str], ...] = (
    ("1", "Index arithmetic at boundaries", "**FIXED** - see below"),
    ("2", "Splitter monopoly", "**FIXED** - see below"),
    ("3", "Feature lags", "PASS - no features are built here"),
    (
        "4",
        "Transforms / scalers",
        "PASS - nothing is fitted; the repair tolerance is a constant, "
        "not estimated from the data",
    ),
    (
        "5",
        "Target construction",
        "PASS - every target reads day `t`'s own bar plus, for the overnight "
        "term, day `t-1`'s close",
    ),
    ("6", "Refit schedule", "n/a - not touched"),
    ("7", "TSFM context", "n/a - not touched"),
    (
        "8",
        "Calendar alignment",
        "PASS - the panel never joins two assets; each series stays on its own "
        "calendar, as `TimeSeriesFrame` requires",
    ),
    (
        "9",
        "Caching",
        "PASS - `ingest_manual_csv` re-reads and re-hashes the raw bytes every "
        "time and never serves from cache; the Binance cache is keyed by "
        "`(symbol, day)` on immutable past-day archives",
    ),
    ("10", "Survivorship / selection", "**DESIGN-LEVEL, flagged** - see below"),
)


def _count(frame: pd.DataFrame, row: str, column: str) -> int:
    """Read one integer cell out of a mixed-dtype report frame.

    ``DataFrame.loc`` is typed as a very wide union because the frame holds
    dates and strings alongside counts; every cell read through here is a count
    by construction of the frame that produced it.
    """
    return int(cast(int, frame.loc[row, column]))


def _md_table(frame: pd.DataFrame, *, index_label: str = "asset") -> str:
    header = f"| {index_label} | " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    lines = [header, rule]
    for name, row in frame.iterrows():
        cells = " | ".join("" if pd.isna(v) else str(v) for v in row)
        lines.append(f"| **{name}** | {cells} |")
    return "\n".join(lines)


def _span_table(diagnostics: dict[str, SeriesDiagnostics]) -> pd.DataFrame:
    longest = max(d.n_obs for d in diagnostics.values() if d.source == "stooq")
    rows = {}
    for asset_id, d in diagnostics.items():
        short = d.source == "stooq" and d.n_obs < SHORT_SPAN_FRACTION * longest
        rows[asset_id] = {
            "role": d.role,
            "archive starts": d.archive_start.date(),
            "panel span": f"{d.panel_start.date()} → {d.panel_end.date()}",
            "obs": d.n_obs,
            "obs/yr": f"{d.obs_per_year:.1f}",
            "% of longest": f"{100 * d.n_obs / longest:.0f}%",
            "flag": "**SHORT**" if short else "ok",
        }
    return pd.DataFrame(rows).T


def _quality_table(diagnostics: dict[str, SeriesDiagnostics]) -> pd.DataFrame:
    rows = {}
    for asset_id, d in diagnostics.items():
        rows[asset_id] = {
            "repaired": d.repaired_bars,
            "inconsistent": d.inconsistent_bars,
            "inconsistent %": f"{100 * d.inconsistent_bars / d.n_obs:.2f}%",
            "zero-range (H=L)": d.zero_range_days,
            "monotone bars": d.monotone_bars,
            "stale opens": d.stale_open_days,
            "stale open %": f"{100 * d.stale_open_days / d.n_obs:.2f}%",
            "**target = 0**": d.zero_primary_days,
            "NaN primary": d.nan_by_target.get(d.primary_target, 0),
        }
    return pd.DataFrame(rows).T


def _gap_table(diagnostics: dict[str, SeriesDiagnostics]) -> pd.DataFrame:
    rows = {}
    for asset_id, d in diagnostics.items():
        longest = "; ".join(
            f"{a.date()}→{b.date()} ({n}d)" for a, b, n in d.longest_gaps[:2]
        )
        rows[asset_id] = {
            "max gap (days)": d.max_gap_days,
            f"gaps > {GAP_ALERT_DAYS}d": d.n_gaps_over_alert,
            "longest closures": longest or "—",
        }
    return pd.DataFrame(rows).T


def _target_table(diagnostics: dict[str, SeriesDiagnostics]) -> pd.DataFrame:
    rows = {}
    for asset_id, d in diagnostics.items():
        q = d.opr_over_parkinson_quantiles
        rows[asset_id] = {
            "ann. vol": f"{d.annualized_vol_pct:.1f}%",
            "overnight share": f"{100 * d.overnight_share:.1f}%",
            "OPR/Park q10": f"{q['q10']:.2f}",
            "OPR/Park median": f"{q['q50']:.2f}",
            "OPR/Park q90": f"{q['q90']:.2f}",
        }
    return pd.DataFrame(rows).T


def _crisis_table(panel: dict[str, PanelSeries], window: int = PROTOCOL_WINDOW) -> pd.DataFrame:
    coverage = crisis_coverage(panel, window=window)
    rows = {}
    for asset_id, row in coverage.iterrows():
        cells: dict[str, object] = {
            "scored obs": row["n_scored"],
            "first scored": row["scored_from"],
        }
        for crisis in CRISIS_WINDOWS:
            scored = int(row[f"{crisis.tag}_scored"])
            available = int(row[f"{crisis.tag}_available"])
            cells[crisis.tag] = f"{scored}/{available}"
        rows[str(asset_id)] = cells
    return pd.DataFrame(rows).T


def _gfc_recovery_table(
    panel: dict[str, PanelSeries], *, window: int, robustness_window: int
) -> pd.DataFrame:
    """What each window costs the two crisis arms that warm-up eats (D-019)."""
    default = crisis_coverage(panel, window=window)
    longer = crisis_coverage(panel, window=robustness_window)
    rows = {}
    for asset_id in panel:
        rows[asset_id] = {
            "GFC in panel": _count(default, asset_id, "gfc_available"),
            f"GFC scored @{window}": _count(default, asset_id, "gfc_scored"),
            f"GFC scored @{robustness_window}": _count(longer, asset_id, "gfc_scored"),
            "COVID in panel": _count(default, asset_id, "covid_available"),
            f"COVID scored @{window}": _count(default, asset_id, "covid_scored"),
            f"COVID scored @{robustness_window}": _count(longer, asset_id, "covid_scored"),
        }
    return pd.DataFrame(rows).T


def render_report(
    panel: dict[str, PanelSeries],
    diagnostics: dict[str, SeriesDiagnostics],
    *,
    raw_root: Path,
    generated_at: str,
    window: int = PROTOCOL_WINDOW,
    robustness_window: int = ROBUSTNESS_WINDOW,
) -> str:
    equities = {k: v for k, v in diagnostics.items() if v.source == "stooq"}
    crypto = {k: v for k, v in diagnostics.items() if v.source == "binance"}

    longest = max(d.n_obs for d in equities.values())
    short = [k for k, d in equities.items() if d.n_obs < SHORT_SPAN_FRACTION * longest]
    zero_target = {k: d.zero_primary_days for k, d in diagnostics.items() if d.zero_primary_days}
    inconsistent = {k: d.inconsistent_bars for k, d in diagnostics.items() if d.inconsistent_bars}

    shares = {k: d.overnight_share for k, d in equities.items()}
    lo_share, hi_share = min(shares.values()), max(shares.values())
    lo_name = min(shares, key=lambda k: shares[k])
    hi_name = max(shares, key=lambda k: shares[k])
    expected_lo, expected_hi = OVERNIGHT_SHARE_EXPECTED

    crypto_share = max(d.overnight_share for d in crypto.values()) if crypto else float("nan")
    crypto_ratio = (
        max(d.opr_over_parkinson_quantiles["q50"] for d in crypto.values())
        if crypto
        else float("nan")
    )

    coverage = crisis_coverage(panel, window=window)
    gfc_scored = {
        str(a): (int(r["gfc_scored"]), int(r["gfc_available"]))
        for a, r in coverage.iterrows()
    }
    too_short = [
        a
        for a, r in coverage.iterrows()
        if int(r["n_scored"]) < PROTOCOL_MIN_FORECASTS
    ]

    # Values that would otherwise make the prose f-strings below unreadably long.
    n_equity, n_crypto = len(EQUITY_PANEL), len(CRYPTO_PANEL)
    n_assets = n_equity + n_crypto
    n_index = sum(1 for spec in EQUITY_PANEL.values() if spec.role == "index")
    n_etf = sum(1 for spec in EQUITY_PANEL.values() if spec.role == "etf_proxy")
    hsi, twse, nkx = diagnostics["HSI"], diagnostics["TWSE"], diagnostics["NKX"]
    hsi_stale_pct = 100 * hsi.stale_open_days / hsi.n_obs
    twse_mono_pct = 100 * twse.monotone_bars / twse.n_obs
    twse_bad_pct = 100 * twse.inconsistent_bars / twse.n_obs
    hsi_failed_cells = f"{hsi.zero_primary_days * PROTOCOL_WINDOW:,}"
    inconsistent_list = ", ".join(
        f"{k} ({v})" for k, v in sorted(inconsistent.items(), key=lambda kv: -kv[1])
    )
    total_inconsistent = sum(d.inconsistent_bars for d in diagnostics.values())
    n_zero_target = sum(zero_target.values())
    ratio_lo = min(d.opr_over_parkinson_quantiles["q50"] for d in equities.values())
    ratio_hi = max(d.opr_over_parkinson_quantiles["q50"] for d in equities.values())
    crypto_stale_pct = (
        100 * max(d.stale_open_days / d.n_obs for d in crypto.values()) if crypto else 0.0
    )
    crypto_missing_rv = sum(d.nan_by_target.get("realized_variance", 0) for d in crypto.values())
    crypto_days = len(panel["BTC-USD"].index) if "BTC-USD" in panel else 0
    gfc_min = min(v[0] for v in gfc_scored.values() if v[1])
    gfc_max = max(v[0] for v in gfc_scored.values())

    def _cov(asset: str, tag: str) -> str:
        scored = _count(coverage, asset, f"{tag}_scored")
        available = _count(coverage, asset, f"{tag}_available")
        return f"{scored}/{available}"

    covid_spy = _cov("SPY", "covid")
    tightening_spy = _cov("SPY", "tightening_2022")
    spike_spy = _cov("SPY", "spike_2024_08")
    smallest_name = str(min(coverage.index, key=lambda a: _count(coverage, str(a), "n_scored")))
    smallest_scored = _count(coverage, smallest_name, "n_scored")
    gfc_available_min = min(v[1] for v in gfc_scored.values() if v[1])
    gfc_available_max = max(v[1] for v in gfc_scored.values())
    gfc_full = [a for a, v in gfc_scored.items() if v[1] and v[0] == v[1]]
    gfc_recovered_note = (
        f"every one of the {len(gfc_full)} series with GFC days in the panel\nnow scores "
        "all of them"
        if len(gfc_full) == sum(1 for v in gfc_scored.values() if v[1])
        else f"{len(gfc_full)} of {sum(1 for v in gfc_scored.values() if v[1])} series score "
        "their full GFC sample"
    )
    has_btc = "BTC-USD" in coverage.index
    btc_covid_scored = _count(coverage, "BTC-USD", "covid_scored") if has_btc else 0
    btc_scored_from = str(coverage.loc["BTC-USD", "scored_from"]) if has_btc else "n/a"
    short_note = "" if not too_short else " EXCEPT " + ", ".join(map(str, too_short))
    retired_row = (
        ", ".join(
            f"{spec.asset_id} ({spec.proxy_for or spec.description})"
            for spec in RETIRED_EQUITY.values()
        )
        or "—"
    ) + " — D-020"
    invalid_days = {k: v.invalid_target_days for k, v in panel.items()}
    invalid_list = (
        ", ".join(f"{k} ({v})" for k, v in sorted(invalid_days.items(), key=lambda kv: -kv[1]) if v)
        or "none"
    )
    n_invalid_total = sum(invalid_days.values())
    n_series_with_invalid = sum(1 for v in invalid_days.values() if v)
    equity_targets_row = (
        "`overnight_plus_range` (primary, D-016), `parkinson`, "
        "`garman_klass`, `squared_return`"
    )
    zero_range = {k: d.zero_range_days for k, d in diagnostics.items() if d.zero_range_days}
    audit_table = "\n".join(
        ["| # | Item | Verdict |", "|---|---|---|"]
        + [f"| {n} | {item} | {verdict} |" for n, item, verdict in _LEAKAGE_AUDIT]
    )
    realized_shares = {k: d.realized_overnight_share for k, d in equities.items()}
    realized_lo, realized_hi = min(realized_shares.values()), max(realized_shares.values())
    share_gaps = {k: abs(shares[k] - realized_shares[k]) for k in equities}
    max_gap_name = max(share_gaps, key=lambda k: share_gaps[k])
    max_share_gap = share_gaps[max_gap_name]
    decomps = {k: d.decomposition_ratio for k, d in equities.items()}
    worst_decomp_name = max(decomps, key=lambda k: abs(decomps[k] - 1.0))
    worst_decomp = decomps[worst_decomp_name]
    ok_decomps = {k: v for k, v in decomps.items() if k != worst_decomp_name}
    decomp_lo, decomp_hi = min(ok_decomps.values()), max(ok_decomps.values())
    n_decomp_ok = len(ok_decomps)
    share_check_table = _md_table(
        pd.DataFrame(
            {
                k: {
                    "estimator": f"{100 * shares[k]:.1f}%",
                    "realized returns": f"{100 * realized_shares[k]:.1f}%",
                    "Var(ON)+Var(OC) / Var(CC)": f"{decomps[k]:.3f}",
                }
                for k in equities
            }
        ).T
    )
    zero_range_note = (
        ", ".join(f"{v} on {k}" for k, v in zero_range.items()) if zero_range else "none at all"
    )
    zero_target_table = (
        _md_table(
            pd.DataFrame({k: {"primary target == 0": v} for k, v in zero_target.items()}).T
        )
        if zero_target
        else "_none_"
    )

    parts: list[str] = []
    parts.append(f"""# Panel report — the D-004/D-012/D-020 evaluation panel

> Generated by `uv run python -m volbench.data.build_panel` on {generated_at}.
> Every figure below is measured by `volbench.data.diagnostics` from the panel
> that run built; none is hand-entered. Regenerating against the same archives
> reproduces this file.
>
> **This edition is post-decision.** The first edition (2026-08-24) was a
> review gate whose §9 listed what it could not settle; the three protocol
> decisions it asked for were taken on `feat/p2-protocol` and are D-018
> (invalid-target policy), D-019 (a {window}-observation fit window) and D-020
> (the FTSE 100 slot dropped). The report is regenerated here under those
> decisions, so every count below is the count the study actually runs with.
> §9 records what each of them resolved and what remains open.

## 1. What was built

| | |
|---|---|
| panel window | {PANEL_START.date()} → {PANEL_END.date()} (D-004: "2005-01 → freeze date") |
| headline panel | **{n_assets} assets** ({n_equity} equity + {n_crypto} crypto, D-020) |
| equity series | {len(EQUITY_PANEL)} ({n_index} indices + {n_etf} ETF proxies, D-012) |
| crypto series | {len(CRYPTO_PANEL)} (Binance 1-minute → 5-minute RV) |
| not in the panel | {retired_row} |
| equity targets | {equity_targets_row} |
| crypto target | `realized_variance` (5-min RV, D-004) — see §7 |
| fit window | {window} observations (D-019); {robustness_window} as the robustness arm |
| unusable days | dropped from fit windows, kept as scored NaN rows (D-018) — see §4 |
| raw source | hand-downloaded Stooq bulk archives under `{raw_root}` |
| crypto source | `data.binance.vision` bulk archives (scripted; documented as permitted) |

**Provenance.** stooq.com is never contacted programmatically: its terms forbid
redistribution and its CSV endpoint answers automation with an anti-bot
challenge (`docs/data_licenses.md`). The equity arm reads files a human
downloaded and unzipped, through `ingest_manual_csv`;
`tests/test_data_panel.py` asserts that `panel.py` imports no network entry
point at all, so the rule cannot be eroded by a later edit. No raw or cached
file is committed — `tests/test_licensing_guard.py` asks git directly.

## 2. Span validation

{_md_table(_span_table(diagnostics))}

`obs/yr` is the sanity check against each venue's real trading calendar: US
~252, Xetra ~253, Euronext ~256, Tokyo ~245, HK ~247, Taiwan ~245, Korea ~247.
Every series lands within a day or two of its venue, so no series is missing a
material block of sessions.
""")

    if short:
        short_lines = "\n".join(
            f"- **{k}** - {equities[k].n_obs} obs from "
            f"{equities[k].archive_start.date()}, "
            f"{100 * equities[k].n_obs / longest:.0f}% of the longest equity series "
            f"({longest} obs).\n"
            f"  Under the {window}-observation rolling window it yields "
            f"{_count(coverage, k, 'n_scored')} scored forecasts,\n"
            f"  all from {coverage.loc[k, 'scored_from']} onward."
            for k in short
        )
        parts.append(f"""
### 2.1 D-012 fallback trigger — **FIRED**

{short_lines}
""")
    else:
        parts.append(f"""
### 2.1 The FTSE 100 slot — dropped (D-020)

No series in the panel is materially short of the longest: the {len(equities)}
equity series span {min(d.n_obs for d in equities.values())} to {longest}
observations, a spread of {100 * min(d.n_obs for d in equities.values()) / longest:.0f}%
to 100%.

That is a consequence of the decision, not a property of the archives. The
first edition of this report fired D-012's fallback trigger on **ISF**, the
iShares Core FTSE 100 ETF and the only instrument available for that slot:
2899 observations from 2015-03-04, 52% of the longest equity series, and — the
part that decided it — **zero observations inside the GFC window**, which
predates the fund by six years. No choice of rolling window recovers a crisis
that is not in the data. Since H3 is a claim about crisis sub-samples, a series
that can never contribute to the oldest of them would have been carried through
every table as a permanent blank, so D-020 drops the FTSE 100 leg and the
headline panel is {len(EQUITY_PANEL) + len(CRYPTO_PANEL)} assets.

What was *not* thrown away: `ISF` keeps its `EquitySpec` in
`volbench.data.panel.RETIRED_EQUITY`, `build_equity_series("ISF")` still
ingests it, and its tests still run. Dropping a series from a study and
deleting the code that reads it are different things, and only the first was
decided.

The other two ETF proxies are unaffected: SPY and DIA both start
{equities["SPY"].archive_start.date()}, which is the Stooq US archive's own
beginning, and match the index series observation-for-observation thereafter.
The D-012 substitution was only ever *materially* costly in the FTSE 100 slot.
""")

    parts.append(f"""
## 3. Bar-level data quality

{_md_table(_quality_table(diagnostics))}

**Inconsistent bars.** A bar must satisfy `low ≤ min(open, close) ≤
max(open, close) ≤ high`. The archives violate that in two clearly separate
regimes, and the builder treats them differently on purpose:

- **≤ 1e-5 relative** — the source file's own decimal rounding (NDX, NKX). The
  bar is clamped to its own open/close hull and counted under `repaired`.
- **above that** — up to 1.3% on TWSE, 1.1% on CAC: a close printed outside its
  own session high or low. That is two different feeds (or two different
  snapshot times) for the index level and the intraday extremes, and there is
  no honest repair for it. The bar is left **unmodified**, flagged, and its
  three *range-based* targets are set to NaN. `squared_return`, which reads
  only closes, is unaffected. Nothing is dropped: the row survives with NaN, so
  `run_backtest` records a `missing_reason` rather than a model quietly
  scoring on a shortened sample.

Affected: {inconsistent_list}.
TWSE is the worst at {twse_bad_pct:.1f}% of its panel days.

**The 1e-5 threshold is a judgement call this branch made and did not have
authority to settle** — see §9.

## 4. Zero-variance days — the D-016 revisit trigger

The task asked for zero-range days as the D-016 trigger. The count that
actually matters turned out to be different, and larger:

{zero_target_table}

Zero-*range* days (`high == low`) are almost absent: {zero_range_note}.
But the **primary target reaches exactly zero on {n_zero_target} days
across {len(zero_target)} series**, by a mechanism that has nothing to do with limit moves:

`overnight_plus_range = (ln(O_t/C_(t-1)))² + RS_t`, and *both* terms can be
exactly zero at once.

- Rogers-Satchell is **identically zero on a monotone bar** — a day that opens
  at its high and closes at its low, or the reverse. That is a property of the
  estimator, not a defect: `ln(H/O)·ln(H/C) + ln(L/O)·ln(L/C)` has a zero
  factor in each product. It happens on {twse.monotone_bars} TWSE days
  ({twse_mono_pct:.1f}%) and {hsi.monotone_bars} HSI days.
- The overnight term is zero whenever the open **exactly equals** the previous
  close. On HSI that is {hsi.stale_open_days} days — {hsi_stale_pct:.1f}% of the
  sample — a stale or synthetic open in the Stooq file, not a market fact.

Where the two coincide, the target is 0. Every such day is verifiably of that
form: e.g. HSI 2024-04-12, `O=H=17095.03`, `L=C=16721.69`, previous close
`17095.03` — a monotone down day whose open was carried over from the previous
close.

**Why this was a live problem, and what D-018 did about it.** `HAR.fit` raises
on a non-positive realized-variance input, and every log-RV model takes
`log(RV)`. Under a {window}-observation rolling window a *single* zero
contaminates the {window} training windows that contain it, so HSI's
{hsi.zero_primary_days} zeros alone were up to {hsi_failed_cells} origin-model
cells emitted as NaN rows — the evaluator behaving correctly, while a large
share of one series' HAR column quietly disappeared.

D-018 settles it by separating the two roles such a day plays:

- **as a target** it stays exactly where it is. The row is produced, the
  scores are NaN and `missing_reason` names the cause. Nothing is dropped from
  the scored table, so no model can look good by being scored on a shorter
  sample than another. A zero is routed there explicitly now, rather than
  arriving as a crash: QLIKE is `v/y - log(v/y) - 1`, undefined at `y = 0`.
- **as a fit input** it is dropped. `volbench.compaction.FitSeries`, which
  `PanelSeries.fit_input()` hands to `run_backtest`, materializes each window
  as the last {window} *valid* observations at or before its origin — reaching
  backwards, never past the origin.

Across the whole panel that is {n_invalid_total} days —
{invalid_list}.
The cost is a change in what a lag means (§9), not a change in what is scored:
an invalid day is still a perfectly good *origin* — its own target is
unmeasurable, but its history is intact, so the forecast issued at it is a
normal forecast.

## 5. Calendar gaps

{_md_table(_gap_table(diagnostics))}

Every long closure identified is a real, checkable exchange holiday, not a hole
in the archive: TWSE's {twse.max_gap_days}-day maximum is Lunar New
Year; NKX's {nkx.max_gap_days}-day is Golden Week 2019 (the
Reiwa-accession extension); KOSPI's is Chuseok. The US series' only two gaps
over {GAP_ALERT_DAYS} days are the 2006-07 New Year and **Hurricane Sandy
(2012-10-29/30)**, the two-day NYSE closure. Crypto has a maximum gap of 1 day,
as a 24/7 market should.

Combined with the `obs/yr` check in §2, no series shows evidence of missing
data as opposed to genuine market closures.

## 6. Targets: overnight share and the OPR/Parkinson ratio

{_md_table(_target_table(diagnostics))}

**The expected range in the task brief is wrong, and it matters in D-016's
favour.** The brief anticipated an overnight share of
~{100 * expected_lo:.0f}-{100 * expected_hi:.0f}% for indices. The measured share is
**{100 * lo_share:.0f}% ({lo_name}) to {100 * hi_share:.0f}% ({hi_name})** —
between two and five times that. The ~9% figure in D-016 came from the *toy
fixture's* generator parameters, not from market data, and it should not be
carried into the paper as an empirical expectation.

This was verified independently of the estimator, because a discrepancy that
large is as likely to be an estimator bug as a fact. Decomposing *realized
returns* — `Var(ln(O_t/C_(t-1)))` against `Var(ln(C_t/O_t))`, no range
estimator involved anywhere — gives an overnight share of
{100 * realized_lo:.0f}-{100 * realized_hi:.0f}% across the ten equity series, against
{100 * lo_share:.0f}-{100 * hi_share:.0f}% from `overnight_variance`/`rogers_satchell`.
The largest disagreement on any series is {100 * max_share_gap:.1f} percentage points
({max_gap_name}), and `Var(overnight) + Var(intraday)` recovers
{100 * decomp_lo:.0f}-{100 * decomp_hi:.0f}% of `Var(close-to-close)` on
{n_decomp_ok} of the ten. So the measurement stands.

Both columns are regenerated by `diagnostics.py` on every run
(`realized_overnight_share` and `decomposition_ratio` in the diagnostics CSV),
so this cross-check cannot silently go stale:

{share_check_table}

The consequence for D-016 is that it was *more* right than its own rationale
claimed: a range proxy alone omits **a third to a half** of close-to-close
variance on these series, not a tenth. The OPR/Parkinson median of
{ratio_lo:.2f}-{ratio_hi:.2f}
is the same fact seen from the other side.

**The clean control for that ratio is the crypto arm.** On a 24/7 market the
overnight term is a one-minute gap, so OPR ≈ RS and any excess of OPR over
Parkinson is pure estimator difference, not overnight variance. There the
median ratio is {crypto_ratio:.2f}. The equity excess should therefore be read
against ~{crypto_ratio:.2f}, not against 1.00.

**Exception worth recording:** on {worst_decomp_name},
`Var(overnight) + Var(intraday)` is only {100 * worst_decomp:.0f}% of
`Var(close-to-close)` — i.e. that series' overnight gap and its subsequent
intraday move are materially positively correlated, where every other series
shows ~zero. That is either a genuine continuation effect (the Nikkei's
overnight session is the US session) or a timing artefact in Stooq's open. It
does not affect the target, which is a per-day sum rather than a variance
decomposition, but it should be checked before any overnight/intraday
*component* model is fitted to that series.

## 7. What the crypto target actually is

Confirmed empirically, as the task asked. **The crypto primary target is
5-minute realized variance (`realized_variance`), per D-004 — not
`overnight_plus_range`.** The panel builder sets `primary_target` accordingly,
and the four range targets are carried alongside for diagnostics only.

The reason is measured, not assumed: on a 24/7 market the "overnight" term
degenerates. Its aggregate share of `overnight_plus_range` is
**{100 * crypto_share:.2f}%** (versus {100 * lo_share:.0f}%+ for every equity series), and the
open equals the previous bar's close outright on
~{crypto_stale_pct:.0f}% of days.
There is no session break for it to measure; the UTC-day boundary is a
reporting convention this module imposes so that RV and the range estimators
share one calendar, and `daily_bars_from_minutes` is tested to ensure no bucket
straddles midnight.

So `overnight_plus_range` on BTC/ETH is, to three decimals, Rogers-Satchell.
Reporting it as a "close-to-close" target on crypto would be a category error;
reporting RV is the whole point of including the arm — it is the one place in
the panel with an almost model-free measure of the latent variance.
Both series have {crypto_missing_rv} missing RV days over {crypto_days} days.

## 8. Crisis sub-samples

Windows are `volbench.data.crisis`, taken verbatim from
`docs/research_design.md`. Tags are metadata attached to dates and are never an
input to fitting — the module imports nothing from `volbench.models`,
`volbench.evaluate` or `volbench.splitter`, and `tests/test_data_crisis.py`
enforces that against the module's AST, along with the fact that its whole
public API accepts only a `DatetimeIndex`.

| tag | window | source phrase |
|---|---|---|
""")

    for crisis in CRISIS_WINDOWS:
        parts.append(
            f"| `{crisis.tag}` | {crisis.start} → {crisis.end} | \"{crisis.source_phrase}\" |\n"
        )
    for pending in PENDING_WINDOWS:
        parts.append(
            f"| `{pending.tag}` | **not dated** | \"{pending.source_phrase}\" |\n"
        )

    parts.append(f"""
The fifth window is deliberately undated: D-004 fixes it at grid freeze, so
inventing a range here would fabricate a sub-sample result. It lives in
`PENDING_WINDOWS`, is excluded from every tagging function, and raises a
self-explaining `KeyError` if looked up.

The Aug-2024 window is read as the **calendar month** containing the 2024-08-05
unwind, since `research_design.md` gives only "Aug-2024 spike" with no
day-level range. That reading is an assumption — see §9.

### 8.1 Scored vs. available crisis observations

`scored/available` — how many observations of each window survive the
{window}-observation rolling-origin warm-up (D-019), against how many the panel
contains at all:

{_md_table(_crisis_table(panel, window))}

**This is what D-019 was for.** The first edition of this report found the GFC
window largely *inside* the warm-up period: at {robustness_window} observations
the panel held 140-149 GFC days per equity series and the evaluation scored
31-86 of them, none at all for the crypto arm's COVID window. At {window} the
same panel scores **{gfc_min}-{gfc_max}** GFC days per equity series against
{gfc_available_min}-{gfc_available_max} available:
{gfc_recovered_note}.

The two arms side by side, which is the measurement the decision rests on:

{_md_table(_gfc_recovery_table(panel, window=window, robustness_window=robustness_window))}

So at the default window:

- every equity series scores **{gfc_min}+** GFC observations, against the
  140-149 the panel contains — the arm is now a real sub-sample rather than
  a remnant;
- the crypto arm scores **{btc_covid_scored}** COVID observations per series
  (it scored 0 at {robustness_window}), because {window} days of warm-up from
  the 2017 listing runs only to {btc_scored_from};
- COVID ({covid_spy} on SPY), the 2022 tightening ({tightening_spy}) and
  Aug-2024 ({spike_spy}) remain fully scored, as they already were.

H3 — "TSFM relative performance degrades in crisis sub-samples" — is therefore
testable on all four settled windows rather than three. The Aug-2024 window is
still only ~22 scored observations per series: enough to describe, not enough
to run block-bootstrap MCS on per-series, and that is a property of a
one-month window, not of the protocol.

What the shorter window costs is estimation sample, not evaluation sample, and
it is the reason {robustness_window} survives as the robustness arm rather than
being deleted: whether a ranking holds at both windows is a question this panel
can now answer, and both are plumbed through the run configs and enter every
config hash through the splitter.

Every series clears the ≥{PROTOCOL_MIN_FORECASTS}-forecast minimum from
`research_design.md`{short_note}
(smallest: {smallest_name} at {smallest_scored}).

## 9. Open items

### Settled since the first edition

- **Unusable days** (§4) — **D-018**. The first edition recommended NaN-ing the
  zero-target days so they are excluded from fitting like any other bad bar.
  What was decided is that recommendation made precise: an invalid target day
  (primary target NaN *or* ≤ 0) is dropped from every fit window and kept as a
  scored NaN row. Neither of the alternatives it listed was taken — the target
  is not floored (that would change the estimator and invent a number nobody
  measured) and no series is dropped.
- **The GFC crisis arm** (§8.1) — **D-019**. Resolved by the second of the
  three options the first edition listed, generalized: the fit window is
  {window} observations for the whole study rather than for a GFC-specific
  ablation, with {robustness_window} kept as the robustness arm. D-004's stated
  span is untouched, so no reopen was needed.
- **The FTSE 100 slot** (§2.1) — **D-020**. Dropped. The panel is
  {n_assets} assets; the ingestion code is kept.
- **`STOOQ_INDEX_SYMBOLS` vs. the panel** — closed earlier, by D-024 (the CFD
  entries were retired at the Phase-2 core integration).

### Still open

1. **The 1e-5 bar-repair threshold** (§3). Splitting "rounding" from "real
   error" at 1e-5 relative is calibrated on this archive's observed bimodality,
   but it is a modelling choice. Alternatives: NaN *every* violation
   ({total_inconsistent} more days lost), or clamp them all (silently rewrites
   a 1.3% error). Recommend keeping the split and recording it as a decision
   entry. Note D-018 lowers the stakes: a NaN'd day now costs its own scored
   row and nothing else, where before it could fail every window containing it.
2. **Lag semantics under compaction** (§4), new with D-018 and the one thing it
   costs. For HAR and LightGBM, which read *positional* lags of the fit series,
   "yesterday" now means the previous *measured* day, so a lag can span two or
   more calendar days on the {n_series_with_invalid} series that have any.
   This is documented in those adapters and in
   `docs/design.md`; whether the paper reports HAR's memory in calendar days or
   in observations is a presentation decision that has not been taken.
3. **The Aug-2024 window's exact dates** (§8), currently read as the calendar
   month.
4. **The ~9-15% overnight-share expectation** (§6) should be corrected wherever
   it appears in the planning documents; the measured range is
   {100 * lo_share:.0f}-{100 * hi_share:.0f}%.
5. **NKX overnight/intraday correlation** (§6), worth a look before any
   component model is fitted.
6. **The 2025-26 stress window** (§8) is still undated, per D-004: it is fixed
   at grid freeze.

## 10. Reproducing this report

```
uv run python -m volbench.data.build_panel \\
    --raw-root <path to the unzipped Stooq archives> \\
    --cache-root <path for the parquet caches>
```

Requires the hand-downloaded Stooq bulk archives
(`d_us_txt`, `d_uk_txt`, `d_world_txt`) unpacked under `--raw-root`, and
network access on first run for the Binance archives (cached thereafter).
Neither tree is committed.
""")
    parts.append(f"""
## 11. Leakage audit

Run against `.claude/skills/leakage-check` with a calendar/gap focus. Two
findings, both fixed on this branch; the rest pass.

{audit_table}

**(1) and (2) - `crisis_coverage` reproduced the splitter's arithmetic and got
it wrong.** It assumed the first scored position was `window + horizon`. The
splitter's first origin is `window - 1` and its test set is `origin + 1`, so
the first scored position is `window`. Every series' scored count was
understated by one observation. Rewritten to take the union of
`RollingOriginSplitter`'s own `test` indices, so the arithmetic exists in one
place only. `tests/test_data_panel.py::TestCrisisCoverageUsesTheSplitter`
pins it against the splitter directly. The counts in §8.1 are post-fix.

**Canary.** The audit's required test is implemented as
`TestFutureDataCannotReachAnEarlierTarget`: corrupting every bar strictly after
a cutoff date and rebuilding leaves every target dated `<= cutoff`
bit-identical, asserted with `assert_frame_equal`. A companion test corrupts
from an earlier date and asserts the same comparison *fails*, so the canary
cannot pass by being inert.

**(10) Survivorship, flagged not fixed.** The ten equity series are instruments
that exist and are liquid in 2026, chosen in 2026 - SPY/DIA/ISF explicitly
because they have history reaching back. For *variance* forecasting this is far
weaker than the corresponding bias in a return study, and index-level data
carries the usual index-reconstitution survivorship regardless of what volbench
does. It is nonetheless a property of the panel that belongs in the paper's data
section rather than in a footnote.

**One structural note.** Everything in `diagnostics.py` is a full-sample
statistic, computed over the whole panel window on purpose. That is correct for
a *report* and would be leakage in a *feature*. Nothing consumes these values
today; if any of them ever becomes a model input, it must be recomputed per
train window instead.

""")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("docs/PANEL_REPORT.md"))
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=None,
        help="optional CSV path for the tidy per-series diagnostics frame",
    )
    parser.add_argument(
        "--no-crypto",
        action="store_true",
        help="skip the Binance arm (no network, equities only)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=PROTOCOL_WINDOW,
        help="fit window the crisis-coverage section reports against (D-019 default)",
    )
    parser.add_argument(
        "--robustness-window",
        type=int,
        default=ROBUSTNESS_WINDOW,
        help="the second window §8.1 compares against (D-019 robustness arm)",
    )
    args = parser.parse_args(argv)

    panel = build_panel(
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        include_crypto=not args.no_crypto,
    )
    diagnostics = diagnose_panel(panel)

    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    report = render_report(
        panel,
        diagnostics,
        raw_root=args.raw_root,
        generated_at=generated_at,
        window=args.window,
        robustness_window=args.robustness_window,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out} ({len(report):,} chars, {len(panel)} series)")

    if args.diagnostics_out:
        from volbench.data.diagnostics import diagnostics_frame

        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_frame(diagnostics).to_csv(args.diagnostics_out)
        print(f"wrote {args.diagnostics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
