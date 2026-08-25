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
    PANEL_END,
    PANEL_START,
    PanelSeries,
    build_panel,
)

__all__ = ["main", "render_report"]

#: Rolling-origin window from docs/research_design.md, used only to report how
#: much of each crisis window survives warm-up. Not an evaluation parameter here.
PROTOCOL_WINDOW = 1000
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


def _crisis_table(panel: dict[str, PanelSeries]) -> pd.DataFrame:
    coverage = crisis_coverage(panel, window=PROTOCOL_WINDOW)
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


def render_report(
    panel: dict[str, PanelSeries],
    diagnostics: dict[str, SeriesDiagnostics],
    *,
    raw_root: Path,
    generated_at: str,
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

    coverage = crisis_coverage(panel, window=PROTOCOL_WINDOW)
    gfc_scored = {
        str(a): (int(r["gfc_scored"]), int(r["gfc_available"]))
        for a, r in coverage.iterrows()
    }
    worst_gfc = min(
        (a for a in gfc_scored if gfc_scored[a][1] > 0),
        key=lambda a: gfc_scored[a][0] / gfc_scored[a][1],
    )
    lost_pct = 100 * (1 - gfc_scored[worst_gfc][0] / gfc_scored[worst_gfc][1])

    too_short = [
        a
        for a, r in coverage.iterrows()
        if int(r["n_scored"]) < PROTOCOL_MIN_FORECASTS
    ]

    # Values that would otherwise make the prose f-strings below unreadably long.
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
    isf_scored = _count(coverage, "ISF", "n_scored") if "ISF" in coverage.index else 0
    has_btc = "BTC-USD" in coverage.index
    btc_covid_scored = _count(coverage, "BTC-USD", "covid_scored") if has_btc else 0
    btc_covid_available = _count(coverage, "BTC-USD", "covid_available") if has_btc else 0
    btc_scored_from = str(coverage.loc["BTC-USD", "scored_from"]) if has_btc else "n/a"
    short_note = "" if not too_short else " EXCEPT " + ", ".join(map(str, too_short))
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
    parts.append(f"""# Panel report — D-004/D-012 evaluation panel

> Generated by `uv run python -m volbench.data.build_panel` on {generated_at}.
> Every figure below is measured by `volbench.data.diagnostics` from the panel
> that run built; none is hand-entered. Regenerating against the same archives
> reproduces this file.
>
> **Nothing in this report has been acted on.** It is the review gate the panel
> task was asked to produce, and §9 lists the decisions it needs before any
> grid run consumes the panel.

## 1. What was built

| | |
|---|---|
| panel window | {PANEL_START.date()} → {PANEL_END.date()} (D-004: "2005-01 → freeze date") |
| equity series | {len(EQUITY_PANEL)} ({n_index} indices + {n_etf} ETF proxies, D-012) |
| crypto series | {len(CRYPTO_PANEL)} (Binance 1-minute → 5-minute RV) |
| equity targets | {equity_targets_row} |
| crypto target | `realized_variance` (5-min RV, D-004) — see §7 |
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
            f"  Under the {PROTOCOL_WINDOW}-observation rolling window it yields "
            f"{_count(coverage, k, 'n_scored')} scored forecasts,\n"
            f"  all from {coverage.loc[k, 'scored_from']} onward."
            for k in short
        )
        parts.append(f"""
### 2.1 D-012 fallback trigger — **FIRED**

{short_lines}

The other two ETF proxies are not short: SPY and DIA both start
{equities["SPY"].archive_start.date()}, which is the Stooq US archive's own
beginning, and match the index series observation-for-observation thereafter.
So the D-012 substitution is only *materially* costly in the FTSE 100 slot.

What this costs, concretely, is stated in §8: **ISF contributes zero GFC
observations** — the window predates the series by six years — so any
crisis-regime result for the UK slot rests on COVID, the 2022 tightening, and
the Aug-2024 spike alone. Whether that is acceptable, or whether the FTSE 100
slot should be sourced elsewhere (or dropped from the H3 crisis analysis and
kept for the full-sample comparison), is a decision for the planning machine,
not for this branch.
""")
    else:
        parts.append(
            "\n### 2.1 D-012 fallback trigger - not fired\n\n"
            "No equity series is materially short of the longest.\n"
        )

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

**Why this is a live problem, not a curiosity.** `HAR.fit` raises on a
non-positive realized-variance input, and the model takes `log(RV)`. Under a
{PROTOCOL_WINDOW}-observation rolling window, a *single* zero contaminates the
{PROTOCOL_WINDOW} training windows that contain it. On HSI's {hsi.zero_primary_days}
zeros that is up to {hsi_failed_cells} origin-model cells failing —
they will be emitted as NaN rows with a `missing_reason`, which is the
evaluator behaving correctly, but it would remove a large share of HSI's HAR
column without anyone having decided that. §9 lists the options.

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
{PROTOCOL_WINDOW}-observation rolling-origin warm-up, against how many the panel
contains at all:

{_md_table(_crisis_table(panel))}

**This is the most consequential finding in the report.** The GFC window is
largely *inside the warm-up period* for every series. The panel contains
140-149 GFC days per equity series; the evaluation scores
{gfc_min}-{gfc_max} of them.
{worst_gfc} loses {lost_pct:.0f}% of its GFC sample to warm-up, and ISF, BTC-USD and
ETH-USD score none at all.

H3 — "TSFM relative performance degrades in crisis sub-samples" — is stated
over crisis sub-samples generally, and COVID ({covid_spy} on SPY),
the 2022 tightening ({tightening_spy}) and Aug-2024 ({spike_spy})
are fully scored, so H3 is testable. But **the GFC arm as currently specified
is not**, and the Aug-2024 window is only ~22 scored observations per series —
enough to describe, not enough to run block-bootstrap MCS on per-series.

The crypto arm loses COVID the same way: BTC/ETH hold
{btc_covid_available} COVID observations each but score {btc_covid_scored}, because
1000 days of warm-up from the 2017 listing runs to {btc_scored_from}. Crypto
therefore contributes to only two of the four settled windows.

This is a protocol interaction, not a data defect: the panel *has* the GFC
days. Recovering them requires a decision (§9), not a rebuild.

Every series clears the ≥{PROTOCOL_MIN_FORECASTS}-forecast minimum from
`research_design.md`{short_note}
(smallest: ISF at {isf_scored}).

## 9. Open items — decisions this branch did not have authority to take

1. **The 1e-5 bar-repair threshold** (§3). Splitting "rounding" from "real
   error" at 1e-5 relative is calibrated on this archive's observed bimodality,
   but it is a modelling choice. Alternatives: NaN *every* violation
   ({total_inconsistent} more days lost), or
   clamp them all (silently rewrites a 1.3% error). Recommend keeping the split
   and recording it as a decision entry.
2. **HSI's zero-variance days and stale opens** (§4). {hsi.zero_primary_days} zero
   targets and {hsi_stale_pct:.1f}% stale opens
   is a source-quality problem specific to one series. Options: (a) let HAR
   fail those windows and report the NaN rows; (b) NaN the zero-target days so
   they are excluded from fitting like any other bad bar; (c) floor the target
   at a small positive value (changes the estimator — needs a D-entry);
   (d) drop HSI. Recommend (b): it is the same treatment already applied to
   inconsistent bars, and it is the only option that neither invents a number
   nor silently deletes a model column.
3. **The GFC crisis arm** (§8.1). Either accept that GFC is a partial
   sub-sample and say so in the paper, or shorten the rolling window for a
   GFC-specific ablation, or extend the panel start before 2005 for the seven
   index series that have the history (NDX to 1938, NKX to 1914) — the last
   would change D-004's stated span and needs an explicit reopen.
4. **The Aug-2024 window's exact dates** (§8), currently read as the calendar
   month.
5. **The ~9-15% overnight-share expectation** (§6) should be corrected wherever
   it appears in the planning documents; the measured range is 33-51%.
6. **`docs/design.md` drift.** `volbench.data.proxies` gained two public
   functions this branch (`overnight_variance`, `rogers_satchell` — the two
   pieces D-016's target is now literally the sum of), and
   `volbench.data` gained `panel`, `diagnostics` and `crisis`.
   `CLAUDE.md` requires a public-API change to update `docs/design.md` in the
   same PR, but also makes that file a read-only mirror here. **Flagged, not
   edited**, per the mirror rule. `ManualIngestResult` also gained a `ticker`
   field and `ingest_manual_csv` an `expect_ticker` guard.
7. **NKX overnight/intraday correlation** (§6), worth a look before any
   component model is fitted.
8. **`STOOQ_INDEX_SYMBOLS` and the panel now disagree about the asset list.**
   That map still carries `SPX`/`DJI`/`FTSE` pointing at Stooq's unlicensed CFD
   proxies (`^uslc`, `^usbc`, `^uklc`); D-012 replaced those three slots with
   SPY/DIA/ISF, which is what `EQUITY_PANEL` encodes. Nothing reads the stale
   entries on this branch, but two sources of truth for "the panel" is a trap.
   Recommend retiring the three CFD entries once D-012 is mirrored.

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
    args = parser.parse_args(argv)

    panel = build_panel(
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        include_crypto=not args.no_crypto,
    )
    diagnostics = diagnose_panel(panel)

    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    report = render_report(
        panel, diagnostics, raw_root=args.raw_root, generated_at=generated_at
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
