"""Per-series validation of a built panel — the numbers docs/PANEL_REPORT.md reports.

Pure measurement. Nothing here repairs, filters, or reweights anything: it
counts what the panel contains so a human can decide whether D-012's ETF
fallbacks trigger, whether D-016's zero-range revisit trigger is live, and
whether any series' calendar looks unlike the exchange it claims to come from.

Every statistic is computed over the panel window only, and every one of them
is a property of already-realized data — there is no forecast, no split, and
no train/test boundary in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from volbench.data.crisis import CRISIS_WINDOWS, tag_dates
from volbench.data.panel import PanelSeries
from volbench.splitter import RollingOriginSplitter

__all__ = [
    "GAP_ALERT_DAYS",
    "OVERNIGHT_SHARE_EXPECTED",
    "RATIO_QUANTILES",
    "SeriesDiagnostics",
    "crisis_coverage",
    "diagnose",
    "diagnose_panel",
    "diagnostics_frame",
]

#: A gap longer than this many calendar days is listed individually in the
#: report. Four covers a normal weekend plus a Friday/Monday holiday; anything
#: longer is either a real exchange closure (Golden Week, Lunar New Year,
#: Christmas) or a hole in the archive, and the two must be told apart by eye.
GAP_ALERT_DAYS = 4

#: Quantiles reported for per-day ratio distributions.
RATIO_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

#: The overnight share of close-to-close variance expected for a cash equity
#: index, per D-016's rationale (~9% on the toy fixture, "more on real
#: indices"). Used only to phrase the sanity check in the report; nothing
#: branches on it.
OVERNIGHT_SHARE_EXPECTED = (0.09, 0.15)


@dataclass(frozen=True)
class SeriesDiagnostics:
    """Everything docs/PANEL_REPORT.md states about one series."""

    asset_id: str
    source: str
    role: str
    description: str
    primary_target: str

    # --- span -------------------------------------------------------------
    archive_start: pd.Timestamp
    archive_end: pd.Timestamp
    panel_start: pd.Timestamp
    panel_end: pd.Timestamp
    n_obs: int
    obs_per_year: float

    # --- bar quality ------------------------------------------------------
    repaired_bars: int
    inconsistent_bars: int
    zero_range_days: int
    non_positive_prices: int
    #: Days whose PRIMARY target is exactly zero. A log-space model cannot take
    #: these — ``HAR.fit`` raises on a non-positive RV — so every training
    #: window containing one fails. D-016's revisit trigger, made countable.
    zero_primary_days: int
    #: Days whose open equals the previous close exactly: a stale or synthetic
    #: open. Their overnight variance is zero by construction, which biases the
    #: overnight share DOWN. A source-quality defect, not a market fact.
    stale_open_days: int
    #: Days that opened at their high and closed at their low, or the reverse.
    #: Rogers-Satchell is exactly zero on such a bar — a property of the
    #: estimator, not of the data.
    monotone_bars: int
    #: NaN count per target column, over the panel window.
    nan_by_target: dict[str, int]

    # --- calendar ---------------------------------------------------------
    max_gap_days: int
    max_gap_at: pd.Timestamp | None
    n_gaps_over_alert: int
    #: The longest gaps as ``(last bar before, first bar after, calendar days)``.
    longest_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp, int], ...]

    # --- target behaviour -------------------------------------------------
    mean_primary: float
    #: sqrt(252 * mean daily variance) — a readability aid for humans only. The
    #: library itself is daily-units throughout (CLAUDE.md rule 2).
    annualized_vol_pct: float
    #: Aggregate overnight share = sum(overnight) / sum(overnight + intraday).
    #: The aggregate, not the mean per-day ratio: overnight variance is
    #: extremely right-skewed, so a per-day mean is dominated by a handful of
    #: gap days and is not the "share of variance" anyone means.
    overnight_share: float
    #: Per-day ``overnight / (overnight + intraday)`` quantiles, for shape.
    overnight_share_quantiles: dict[str, float]
    #: The same share computed WITHOUT any range estimator: the sample variance
    #: of realized close-to-open log returns over the sum of that and the
    #: variance of realized open-to-close log returns. An independent check on
    #: :attr:`overnight_share` — if the two disagree materially, the estimator
    #: is suspect, not the market.
    realized_overnight_share: float
    #: ``(Var(overnight) + Var(intraday)) / Var(close-to-close)``. Near 1 when
    #: the overnight and intraday returns are roughly uncorrelated, which is
    #: what makes the additive decomposition meaningful; a value far from 1
    #: flags a series where they are not.
    decomposition_ratio: float
    #: Per-day ``overnight_plus_range / parkinson`` quantiles. >1 means the
    #: close-to-close target exceeds the intraday-only range proxy, which is
    #: what the overnight term is supposed to add.
    opr_over_parkinson_quantiles: dict[str, float]
    n_ratio_obs: int


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``log(a / b)`` as a Series. ``np.log`` of a Series *is* a Series (pandas
    implements ``__array_ufunc__``); numpy's stubs only know it as an ndarray."""
    return cast(pd.Series, np.log(numerator / denominator))


def _quantiles(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {f"q{int(q * 100):02d}": float("nan") for q in RATIO_QUANTILES}
    return {
        f"q{int(q * 100):02d}": float(clean.quantile(q)) for q in RATIO_QUANTILES
    }


Gap = tuple[pd.Timestamp, pd.Timestamp, int]


def _gaps(index: pd.DatetimeIndex) -> tuple[int, pd.Timestamp | None, int, tuple[Gap, ...]]:
    """Calendar-day gaps between consecutive bars: the max, and the longest few.

    Positional throughout: gap ``i`` is the step from bar ``i`` to bar ``i+1``,
    so a gap is always reported as the pair of real bars that bracket it rather
    than as a lookup that could land on the wrong side of a duplicate.
    """
    if index.size < 2:
        return 0, None, 0, ()
    # A tz-aware DatetimeIndex materializes as an *object* array of Timestamps,
    # whose differences are Python timedeltas that numpy cannot vectorize over.
    # Dropping the (always-UTC) tz first gives a real datetime64 array.
    stamps = index.tz_convert("UTC").tz_localize(None).to_numpy()
    days = np.rint(np.diff(stamps) / np.timedelta64(1, "D")).astype(np.int64)

    peak = int(days.argmax())
    over = np.flatnonzero(days > GAP_ALERT_DAYS)
    ranked = over[np.argsort(-days[over], kind="stable")][:5]
    longest: tuple[Gap, ...] = tuple(
        (index[i], index[i + 1], int(days[i])) for i in ranked.tolist()
    )
    return int(days[peak]), index[peak + 1], int(over.size), longest


def diagnose(series: PanelSeries) -> SeriesDiagnostics:
    """Measure one built :class:`~volbench.data.panel.PanelSeries`."""
    index = series.index
    targets = series.targets
    components = series.components

    span_days = float((index[-1] - index[0]).days) or 1.0
    obs_per_year = len(index) / (span_days / 365.25)

    max_gap, max_at, n_over, longest = _gaps(index)

    primary = series.primary
    mean_primary = float(primary.mean(skipna=True))

    overnight = components["overnight_variance"]
    intraday = components["rogers_satchell"]
    total = overnight + intraday
    valid = total.notna() & (total > 0.0)
    share_aggregate = (
        float(overnight[valid].sum() / total[valid].sum()) if bool(valid.any()) else float("nan")
    )
    per_day_share = (
        (overnight[valid] / total[valid]) if bool(valid.any()) else pd.Series(dtype=float)
    )

    prices = series.frame.data
    stale_open = int((prices["open"] == prices["close"].shift(1)).sum())

    # Model-free cross-check on the overnight share: realized return variances
    # only, no range estimator anywhere. Uses each day's own open/close and the
    # PREVIOUS close, so it is as backward-looking as the target itself.
    open_px, close_px = prices["open"], prices["close"]
    prev_close = close_px.shift(1)
    log_overnight = _log_ratio(open_px, prev_close)
    log_intraday = _log_ratio(close_px, open_px)
    log_daily = _log_ratio(close_px, prev_close)
    usable = log_overnight.notna() & log_intraday.notna() & log_daily.notna()
    if int(usable.sum()) > 2:
        var_on = float(log_overnight[usable].var())
        var_oc = float(log_intraday[usable].var())
        var_cc = float(log_daily[usable].var())
        realized_share = var_on / (var_on + var_oc) if var_on + var_oc > 0 else float("nan")
        decomposition = (var_on + var_oc) / var_cc if var_cc > 0 else float("nan")
    else:
        realized_share = decomposition = float("nan")
    monotone = int((intraday == 0.0).sum())
    zero_primary = int((primary.notna() & (primary <= 0.0)).sum())

    park = targets["parkinson"]
    opr = targets["overnight_plus_range"]
    ratio_ok = park.notna() & opr.notna() & (park > 0.0)
    ratio = (opr[ratio_ok] / park[ratio_ok]) if bool(ratio_ok.any()) else pd.Series(dtype=float)

    return SeriesDiagnostics(
        asset_id=series.asset_id,
        source=series.source,
        role=series.role,
        description=series.description,
        primary_target=series.primary_target,
        archive_start=series.archive_start,
        archive_end=series.archive_end,
        panel_start=index[0],
        panel_end=index[-1],
        n_obs=len(index),
        obs_per_year=obs_per_year,
        repaired_bars=series.quality.repaired,
        inconsistent_bars=series.quality.inconsistent,
        zero_range_days=series.quality.zero_range,
        non_positive_prices=series.quality.non_positive,
        zero_primary_days=zero_primary,
        stale_open_days=stale_open,
        monotone_bars=monotone,
        nan_by_target={str(c): int(targets[c].isna().sum()) for c in targets.columns},
        max_gap_days=max_gap,
        max_gap_at=max_at,
        n_gaps_over_alert=n_over,
        longest_gaps=longest,
        mean_primary=mean_primary,
        annualized_vol_pct=float(np.sqrt(max(mean_primary, 0.0) * 252.0) * 100.0),
        overnight_share=share_aggregate,
        overnight_share_quantiles=_quantiles(per_day_share),
        realized_overnight_share=realized_share,
        decomposition_ratio=decomposition,
        opr_over_parkinson_quantiles=_quantiles(ratio),
        n_ratio_obs=int(ratio.size),
    )


def diagnose_panel(panel: dict[str, PanelSeries]) -> dict[str, SeriesDiagnostics]:
    """Measure every series in a built panel, preserving order."""
    return {asset_id: diagnose(series) for asset_id, series in panel.items()}


def diagnostics_frame(diagnostics: dict[str, SeriesDiagnostics]) -> pd.DataFrame:
    """Flatten diagnostics into one tidy row per series, for tables and parquet."""
    rows = []
    for diag in diagnostics.values():
        row: dict[str, object] = {
            "asset_id": diag.asset_id,
            "source": diag.source,
            "role": diag.role,
            "primary_target": diag.primary_target,
            "archive_start": diag.archive_start.date(),
            "panel_start": diag.panel_start.date(),
            "panel_end": diag.panel_end.date(),
            "n_obs": diag.n_obs,
            "obs_per_year": round(diag.obs_per_year, 1),
            "repaired_bars": diag.repaired_bars,
            "inconsistent_bars": diag.inconsistent_bars,
            "zero_range_days": diag.zero_range_days,
            "non_positive_prices": diag.non_positive_prices,
            "zero_primary_days": diag.zero_primary_days,
            "stale_open_days": diag.stale_open_days,
            "monotone_bars": diag.monotone_bars,
            "max_gap_days": diag.max_gap_days,
            "n_gaps_over_alert": diag.n_gaps_over_alert,
            "annualized_vol_pct": round(diag.annualized_vol_pct, 2),
            "overnight_share": round(diag.overnight_share, 4),
            "realized_overnight_share": round(diag.realized_overnight_share, 4),
            "decomposition_ratio": round(diag.decomposition_ratio, 4),
            "opr_park_median": round(diag.opr_over_parkinson_quantiles["q50"], 3),
        }
        for name, count in diag.nan_by_target.items():
            row[f"nan_{name}"] = count
        rows.append(row)
    return pd.DataFrame(rows).set_index("asset_id")


def crisis_coverage(
    panel: dict[str, PanelSeries], *, window: int, horizon: int = 1, step: int = 1
) -> pd.DataFrame:
    """Scored vs. available observations per crisis regime, per series.

    The distinction the panel report exists to make: a crisis window can be
    fully present in the *data* and almost absent from the *evaluation*, because
    the first observations of every series are consumed warming up the rolling
    origin and are never scored. Counting what the panel contains would
    overstate every crisis sub-sample — and H3 is a claim about scored
    forecasts, not about rows.

    Which observations are scored is taken from
    :class:`~volbench.splitter.RollingOriginSplitter` itself — the union of its
    ``test`` indices — never from arithmetic reproduced here. An earlier version
    of this function assumed the first scored position was ``window + horizon``;
    the splitter's first origin is ``window - 1``, so the first scored position
    is ``window``, and the reproduction silently dropped one observation from
    every series' scored count. Driving the splitter makes that class of
    off-by-one impossible, and is what CLAUDE.md rule 1 requires of anything
    that talks about train/test boundaries.

    This reads indices to *describe* coverage; it produces none, and nothing it
    returns is fed back into fitting or scoring.
    """
    if window < 2:
        raise ValueError("window must be >= 2 (RollingOriginSplitter's own floor)")
    splitter = RollingOriginSplitter(window=window, horizon=horizon, step=step)

    rows: dict[str, dict[str, object]] = {}
    for asset_id, series in panel.items():
        index = series.index
        n = len(index)
        if n <= window + horizon - 1:
            scored = index[:0]
        else:
            positions = np.unique(
                np.concatenate([origin.test for origin in splitter.split(n)])
            )
            scored = index[positions]

        all_tags, scored_tags = tag_dates(index), tag_dates(scored)
        row: dict[str, object] = {
            "n_obs": n,
            "n_scored": len(scored),
            "scored_from": scored[0].date() if len(scored) else None,
        }
        for crisis in CRISIS_WINDOWS:
            row[f"{crisis.tag}_scored"] = int((scored_tags == crisis.tag).sum())
            row[f"{crisis.tag}_available"] = int((all_tags == crisis.tag).sum())
        rows[asset_id] = row
    return pd.DataFrame(rows).T
