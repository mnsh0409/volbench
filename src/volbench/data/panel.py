"""The D-004/D-012 evaluation panel: raw archives -> cached frames -> daily variance targets.

What this module is
-------------------
The single place that says *which* series the study runs on, *where* their raw
bytes live, and *how* the four daily variance targets are built from them. It
composes the existing adapters (:mod:`volbench.data.stooq`,
:mod:`volbench.data.crypto`) and the pure proxies (:mod:`volbench.data.proxies`)
— it does not re-implement parsing, downloading, or any estimator.

Provenance rules it obeys
-------------------------
- **Stooq is never fetched programmatically.** stooq.com's terms forbid
  redistribution and its CSV endpoint answers automation with an anti-bot
  challenge (``docs/data_licenses.md``). The equity arm reads *hand-downloaded*
  bulk archives that a human unzipped under ``raw_root`` and ingests them with
  :func:`volbench.data.stooq.ingest_manual_csv`. Nothing here opens a socket to
  stooq.com; there is deliberately no code path that could.
- **Binance bulk archives are scripted**, which that source documents and
  permits, so the crypto arm calls the adapter directly.
- **Nothing raw is ever committed.** ``raw_root`` and the cache root both live
  under the gitignored ``data/`` tree; ``tests/test_licensing_guard.py`` fails
  the build if either is ever staged.

Temporal integrity
------------------
Every target here is *contemporaneous or backward-looking*: day ``t``'s value
uses day ``t``'s own bar and, for the overnight term, day ``t-1``'s close.
Nothing reads forward. Targets are built on each series' **full** history and
only then trimmed to the panel window, so the first panel day keeps a real
previous close instead of a NaN — a strictly backward-looking use of data that
already existed before the window opened. This module produces no train/test
indices; that remains :class:`~volbench.splitter.RollingOriginSplitter`'s job
alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from volbench.data.crypto import CRYPTO_SYMBOLS, daily_realized_variance, fetch_and_cache_day
from volbench.data.proxies import (
    garman_klass,
    overnight_variance,
    parkinson,
    rogers_satchell,
    squared_return,
)
from volbench.data.stooq import ingest_manual_csv
from volbench.data.types import TimeSeriesFrame

__all__ = [
    "CRYPTO_PANEL",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_RAW_ROOT",
    "EQUITY_PANEL",
    "PANEL_END",
    "PANEL_START",
    "PRIMARY_TARGET_CRYPTO",
    "PRIMARY_TARGET_EQUITY",
    "TARGET_NAMES",
    "BarQuality",
    "CryptoSpec",
    "EquitySpec",
    "PanelSeries",
    "build_crypto_series",
    "build_equity_series",
    "build_panel",
    "build_targets",
    "daily_bars_from_minutes",
    "repair_bars",
    "resolve_equity_path",
]

#: Root of the hand-downloaded raw archives. Configurable so a worktree, a CI
#: box, or the Slurm shared filesystem (D-011) can point elsewhere; the default
#: is the repo-relative gitignored path.
DEFAULT_RAW_ROOT = Path("data/raw")

#: Root of the local parquet caches written by the adapters. Also gitignored.
DEFAULT_CACHE_ROOT = Path("data/cache")

#: Panel window per D-004 ("span 2005-01 -> freeze date"). ``PANEL_END`` is the
#: last bar the 2026-08 archive download contains, i.e. this build's freeze date.
PANEL_START = pd.Timestamp("2005-01-01", tz="UTC")
PANEL_END = pd.Timestamp("2026-08-21", tz="UTC")

#: The four daily-variance targets, primary first (D-016 settles the primary).
TARGET_NAMES = ("overnight_plus_range", "parkinson", "garman_klass", "squared_return")

PRIMARY_TARGET_EQUITY = "overnight_plus_range"

#: Crypto's primary target is 5-minute realized variance (D-004), not the
#: close-to-close range estimator. A 24/7 market has no overnight session, so
#: the overnight term of ``overnight_plus_range`` degenerates there — see
#: :func:`build_crypto_series` and docs/PANEL_REPORT.md.
PRIMARY_TARGET_CRYPTO = "realized_variance"

#: Relative tolerance below which an OHLC inconsistency is treated as decimal
#: rounding in the source file and repaired by clamping the bar to its own
#: open/close hull. Above it the bar is a genuine data error and its
#: range-based targets are set NaN instead of being silently rewritten.
#: Calibrated on the 2026-08 archives: the residual violations on NDX/NKX are
#: O(1e-6) (representation noise), while CAC/HSI/TWSE carry violations up to
#: 1.3% (a close printed outside its own session high — a real feed
#: disagreement that must not be papered over). See docs/PANEL_REPORT.md §3.
DEFAULT_REPAIR_TOLERANCE = 1e-5


@dataclass(frozen=True)
class EquitySpec:
    """Where one equity series lives in the hand-downloaded Stooq archives."""

    asset_id: str
    ticker: str
    #: Path relative to ``raw_root``, using the layout Stooq's bulk zips unpack to.
    relative_path: str
    description: str
    #: ``"index"`` for a direct index series, ``"etf_proxy"`` for a tradable
    #: fund standing in for an index Stooq no longer serves (D-012).
    role: str
    #: For ``role="etf_proxy"``, the index this instrument stands in for.
    proxy_for: str | None = None

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name


@dataclass(frozen=True)
class CryptoSpec:
    """One crypto series, built from Binance 1-minute bulk archives."""

    asset_id: str
    symbol: str
    description: str
    listed_on: date


#: The ten equity series per D-012: the seven indices Stooq still serves
#: directly, plus three ETFs standing in for SPX/DJI/FTSE 100, whose Stooq
#: symbols were retired in favour of unlicensed CFD proxies (docs/M1_REPORT.md
#: §, docs/data_licenses.md). An ETF on the index is a *tradable* instrument
#: with its own tracking error and its own session, not the index itself — that
#: substitution is D-012's, and it is restated in every report this panel feeds.
EQUITY_PANEL: dict[str, EquitySpec] = {
    "NDX": EquitySpec(
        asset_id="NDX",
        ticker="^NDX",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^ndx.txt",
        description="NASDAQ-100 index",
        role="index",
    ),
    "DAX": EquitySpec(
        asset_id="DAX",
        ticker="^DAX",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^dax.txt",
        description="DAX 40 index (Germany)",
        role="index",
    ),
    "CAC": EquitySpec(
        asset_id="CAC",
        ticker="^CAC",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^cac.txt",
        description="CAC 40 index (France)",
        role="index",
    ),
    "NKX": EquitySpec(
        asset_id="NKX",
        ticker="^NKX",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^nkx.txt",
        description="Nikkei 225 index (Japan)",
        role="index",
    ),
    "HSI": EquitySpec(
        asset_id="HSI",
        ticker="^HSI",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^hsi.txt",
        description="Hang Seng index (Hong Kong)",
        role="index",
    ),
    "TWSE": EquitySpec(
        asset_id="TWSE",
        ticker="^TWSE",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^twse.txt",
        description="TAIEX index (Taiwan)",
        role="index",
    ),
    "KOSPI": EquitySpec(
        asset_id="KOSPI",
        ticker="^KOSPI",
        relative_path="stooq/d_world_txt/data/daily/world/indices/^kospi.txt",
        description="KOSPI Composite index (South Korea)",
        role="index",
    ),
    "SPY": EquitySpec(
        asset_id="SPY",
        ticker="SPY.US",
        relative_path="stooq/d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt",
        description="SPDR S&P 500 ETF Trust",
        role="etf_proxy",
        proxy_for="S&P 500",
    ),
    "DIA": EquitySpec(
        asset_id="DIA",
        ticker="DIA.US",
        relative_path="stooq/d_us_txt/data/daily/us/nyse etfs/1/dia.us.txt",
        description="SPDR Dow Jones Industrial Average ETF Trust",
        role="etf_proxy",
        proxy_for="Dow Jones Industrial Average",
    ),
    "ISF": EquitySpec(
        asset_id="ISF",
        ticker="ISF.UK",
        relative_path="stooq/d_uk_txt/data/daily/uk/lse etfs/2/isf.uk.txt",
        description="iShares Core FTSE 100 UCITS ETF",
        role="etf_proxy",
        proxy_for="FTSE 100",
    ),
}

#: The crypto arm (D-004). Listing dates are Binance's spot listing for the
#: USDT pairs; USDT stands in for USD (see :mod:`volbench.data.crypto`).
CRYPTO_PANEL: dict[str, CryptoSpec] = {
    "BTC-USD": CryptoSpec(
        asset_id="BTC-USD",
        symbol=CRYPTO_SYMBOLS["BTC-USD"],
        description="Bitcoin / USD (Binance BTCUSDT spot)",
        listed_on=date(2017, 8, 17),
    ),
    "ETH-USD": CryptoSpec(
        asset_id="ETH-USD",
        symbol=CRYPTO_SYMBOLS["ETH-USD"],
        description="Ether / USD (Binance ETHUSDT spot)",
        listed_on=date(2017, 8, 17),
    ),
}


@dataclass(frozen=True)
class BarQuality:
    """Per-series bar-level data-quality counts, reported not silently absorbed."""

    n_bars: int
    #: Bars clamped to their own OHLC hull because the violation was within
    #: ``repair_tolerance`` (source-file decimal rounding).
    repaired: int
    #: Bars whose open/close lie outside [low, high] by more than the tolerance.
    #: Their range-based targets are NaN — the bar is not a bar.
    inconsistent: int
    #: Bars with ``high == low`` (a limit day, or a halted/untraded session).
    #: Parkinson and Garman-Klass are exactly 0 there, which is a floor a log
    #: -space model cannot take — the D-016 revisit trigger.
    zero_range: int
    #: Bars with a non-positive price in any OHLC field.
    non_positive: int


@dataclass(frozen=True)
class PanelSeries:
    """One panel asset: its price frame, its variance targets, and how clean they are.

    ``targets`` and ``components`` share ``frame``'s index exactly, which is
    what ``run_backtest`` requires of series it aligns positionally.
    """

    asset_id: str
    source: str
    role: str
    description: str
    frame: TimeSeriesFrame
    targets: pd.DataFrame
    #: ``overnight_variance`` and ``rogers_satchell``, the two pieces the
    #: primary equity target is the sum of. Diagnostics only; never a target.
    components: pd.DataFrame
    quality: BarQuality
    primary_target: str
    #: Full extent of the underlying source before the panel window was applied.
    #: Reported so "this series is short" is always visible as a property of the
    #: source rather than an artefact of the trim (D-012's fallback trigger).
    archive_start: pd.Timestamp
    archive_end: pd.Timestamp
    #: SHA256 of the raw bytes this series was built from, where a single file
    #: defines it (equities). ``None`` for series assembled from many files.
    raw_sha256: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def primary(self) -> pd.Series:
        series: pd.Series = self.targets[self.primary_target]
        return series


def resolve_equity_path(spec: EquitySpec, raw_root: Path | str = DEFAULT_RAW_ROOT) -> Path:
    """Locate ``spec``'s raw file under ``raw_root``.

    Tries the recorded layout first, then falls back to a filename search under
    ``raw_root`` — re-extracting a Stooq bulk zip with a different tool can add
    or drop a directory level, and a panel that dies on that is brittle for no
    reason. A search that finds several candidates raises rather than guessing.
    """
    root = Path(raw_root)
    exact = root / spec.relative_path
    if exact.is_file():
        return exact

    matches = sorted(p for p in root.rglob(spec.filename) if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"no raw file for {spec.asset_id} ({spec.ticker}) under {root}: "
            f"expected {exact}. Stooq is never downloaded programmatically "
            "(docs/data_licenses.md) — unpack the hand-downloaded bulk archive there."
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"ambiguous raw file for {spec.asset_id}: {len(matches)} files named "
            f"{spec.filename!r} under {root}: {[str(m) for m in matches[:5]]}"
        )
    return matches[0]


def repair_bars(
    data: pd.DataFrame, *, tolerance: float = DEFAULT_REPAIR_TOLERANCE
) -> tuple[pd.DataFrame, pd.Series, BarQuality]:
    """Classify and minimally repair OHLC bars; never silently rewrite a real error.

    A well-formed bar satisfies ``low <= min(open, close) <= max(open, close)
    <= high``. Real archive files violate this two ways, and the two deserve
    different treatment:

    - by O(1e-6) relative, which is the source file's own decimal rounding. The
      bar is clamped to its open/close hull and counted as ``repaired``.
    - by up to ~1%, which means the close was printed outside its own session
      high/low — different feeds or different snapshot times for the index
      level and the intraday extremes. There is no honest repair for that, so
      the bar is flagged ``inconsistent`` and its range-based targets become
      NaN downstream. Nothing is dropped and nothing is invented.

    Returns the (possibly clamped) frame, a boolean Series flagging the
    inconsistent bars, and the :class:`BarQuality` counts.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"repair_bars needs full OHLC; missing {missing}")

    frame = data.copy()
    # copy=True is load-bearing: on a single-dtype frame ``to_numpy`` can hand
    # back a VIEW of the frame's own buffer, and the clamping below writes
    # through it. That made every statistic computed from these arrays describe
    # the *repaired* bar rather than the bar the source file actually contained
    # — ``zero_range`` silently lost any high==low day that a rounding repair
    # then widened. These arrays must stay the original observation.
    values = frame[required].to_numpy(dtype=np.float64, copy=True)
    open_, high, low, close = (values[:, i] for i in range(4))

    non_positive = int((values <= 0.0).any(axis=1).sum())

    hull_high = np.maximum.reduce([open_, high, low, close])
    hull_low = np.minimum.reduce([open_, high, low, close])

    with np.errstate(divide="ignore", invalid="ignore"):
        over = np.where(high > 0.0, (hull_high - high) / high, np.inf)
        under = np.where(low > 0.0, (low - hull_low) / low, np.inf)
    violation = np.maximum(np.maximum(over, 0.0), np.maximum(under, 0.0))

    violated = violation > 0.0
    repairable = violated & (violation <= tolerance)
    inconsistent = violated & ~repairable

    if repairable.any():
        frame.loc[repairable, "high"] = hull_high[repairable]
        frame.loc[repairable, "low"] = hull_low[repairable]

    zero_range = int((high == low).sum())

    flag = pd.Series(inconsistent, index=frame.index, name="inconsistent_bar")
    quality = BarQuality(
        n_bars=len(frame),
        repaired=int(repairable.sum()),
        inconsistent=int(inconsistent.sum()),
        zero_range=zero_range,
        non_positive=non_positive,
    )
    return frame, flag, quality


def build_targets(
    data: pd.DataFrame,
    *,
    inconsistent: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the four daily variance targets, plus the primary target's two components.

    Columns, in :data:`TARGET_NAMES` order: ``overnight_plus_range`` (D-016
    primary), ``parkinson``, ``garman_klass``, ``squared_return``. All in daily
    units, never annualized.

    Bars flagged ``inconsistent`` get NaN in all three *range-based* targets.
    Parkinson is arithmetically defined there (it needs only ``high >= low``),
    but on a day whose close printed outside the session range the high/low
    demonstrably is not that day's range, so the estimate is not meaningful
    either — masking it keeps the robustness arms honest rather than merely
    computable. ``squared_return`` reads only closes and is unaffected.

    Rows are never dropped: an unusable day is NaN in place, so the panel keeps
    one row per trading day and ``run_backtest`` records a ``missing_reason``
    instead of a model looking good on a shortened sample.
    """
    open_, high, low, close = (data["open"], data["high"], data["low"], data["close"])
    mask = (
        pd.Series(False, index=data.index)
        if inconsistent is None
        else inconsistent.reindex(data.index).fillna(False).astype(bool)
    )

    overnight = overnight_variance(open_, close)
    intraday = rogers_satchell(open_, high, low, close)

    targets = pd.DataFrame(
        {
            "overnight_plus_range": (overnight + intraday).where(~mask),
            "parkinson": parkinson(high, low).where(~mask),
            "garman_klass": garman_klass(open_, high, low, close).where(~mask),
            "squared_return": squared_return(close),
        },
        index=data.index,
    )[list(TARGET_NAMES)]

    components = pd.DataFrame(
        {
            "overnight_variance": overnight.where(~mask),
            "rogers_satchell": intraday.where(~mask),
        },
        index=data.index,
    )
    return targets, components


def _trim(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def build_equity_series(
    asset_id: str,
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    start: pd.Timestamp = PANEL_START,
    end: pd.Timestamp = PANEL_END,
    repair_tolerance: float = DEFAULT_REPAIR_TOLERANCE,
    ingested_on: date | None = None,
) -> PanelSeries:
    """Ingest one equity series from the hand-downloaded archives and build its targets.

    Targets are computed on the file's **full** history and trimmed to
    ``[start, end]`` afterwards, so the first in-window day keeps a genuine
    previous close for its overnight term. Both operations look only backward.
    """
    if asset_id not in EQUITY_PANEL:
        raise KeyError(f"unknown equity asset_id {asset_id!r}; known: {sorted(EQUITY_PANEL)}")
    spec = EQUITY_PANEL[asset_id]
    path = resolve_equity_path(spec, raw_root)

    ingested = ingest_manual_csv(
        path,
        asset_id=asset_id,
        cache_dir=Path(cache_root) / "stooq",
        ingested_on=ingested_on,
        expect_ticker=spec.ticker,
    )
    full = ingested.frame.data

    repaired, inconsistent, _ = repair_bars(full, tolerance=repair_tolerance)
    targets, components = build_targets(repaired, inconsistent=inconsistent)

    windowed = _trim(repaired, start, end)
    if windowed.empty:
        raise ValueError(
            f"{asset_id}: no observations in [{start.date()}, {end.date()}]; "
            f"file covers {full.index[0].date()}..{full.index[-1].date()}"
        )
    # Quality is reported for the panel WINDOW, not the whole archive: NDX's
    # history reaches back to 1938 and TWSE's to 1995, and defects in years the
    # study never evaluates would otherwise dominate the counts.
    _, _, quality = repair_bars(_trim(full, start, end), tolerance=repair_tolerance)

    notes: list[str] = []
    if spec.role == "etf_proxy":
        notes.append(
            f"D-012 substitution: tradable ETF standing in for {spec.proxy_for}; "
            "carries tracking error and its own session, and is not the index."
        )

    return PanelSeries(
        asset_id=asset_id,
        source="stooq",
        role=spec.role,
        description=spec.description,
        frame=TimeSeriesFrame(data=windowed, asset_id=asset_id, source="stooq"),
        targets=_trim(targets, start, end),
        components=_trim(components, start, end),
        quality=quality,
        primary_target=PRIMARY_TARGET_EQUITY,
        archive_start=full.index[0],
        archive_end=full.index[-1],
        raw_sha256=ingested.sha256,
        notes=tuple(notes),
    )


def daily_bars_from_minutes(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday OHLCV bars into UTC calendar-day OHLC bars.

    Used only for the crypto arm's *diagnostic* range targets. A UTC day is an
    arbitrary but fixed cut of a market that never closes: it is what makes
    "overnight" a definable quantity at all on a 24/7 series, and it is the same
    day boundary :func:`volbench.data.proxies.realized_variance_from_bars` uses,
    so RV and the range targets are on one calendar.

    Each day's open is its first bar's open and its close its last bar's close,
    both strictly within the day — no bucket can reach into day ``t+1``.
    """
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("bars must be indexed by a DatetimeIndex")
    if index.tz is None:
        raise ValueError("bars index must be tz-aware (UTC)")
    if not index.is_monotonic_increasing:
        raise ValueError("bars index must be strictly increasing")

    day = index.tz_convert("UTC").floor("D")
    grouped = bars.groupby(day)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )
    if "volume" in bars.columns:
        daily["volume"] = grouped["volume"].sum()
    daily.index = pd.DatetimeIndex(daily.index, name="timestamp")
    return daily.sort_index()


def build_crypto_series(
    asset_id: str,
    *,
    start: date | None = None,
    end: date | None = None,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    sample_interval: str = "5min",
    repair_tolerance: float = DEFAULT_REPAIR_TOLERANCE,
) -> PanelSeries:
    """Build one crypto series: 5-minute realized variance plus the range targets.

    The **primary** crypto target is ``realized_variance`` — daily RV from
    5-minute returns (D-004's "crypto = 5-min RV"), which is why this arm
    exists: it is the one place in the panel where an (almost) model-free
    measure of the latent variance is available. The four equity-style range
    targets are computed alongside it from UTC-day OHLC bars, purely so the
    panel report can state what a range estimator does to a market with no
    session break; ``overnight_plus_range``'s overnight term is a
    last-bar-to-first-bar gap of one minute on a continuous market, not an
    overnight jump, and it is not the target anything is scored against here.

    Both RV and the daily bars come from the same cached 1-minute archives, so
    they are on one calendar by construction.
    """
    if asset_id not in CRYPTO_PANEL:
        raise KeyError(f"unknown crypto asset_id {asset_id!r}; known: {sorted(CRYPTO_PANEL)}")
    spec = CRYPTO_PANEL[asset_id]
    first = start or spec.listed_on
    last = end or PANEL_END.date()
    cache_dir = Path(cache_root) / "crypto"

    daily = daily_bars_from_minutes(_minute_frame(spec.symbol, first, last, cache_dir))
    repaired, inconsistent, quality = repair_bars(daily, tolerance=repair_tolerance)
    targets, components = build_targets(repaired, inconsistent=inconsistent)

    realized = daily_realized_variance(
        asset_id, first, last, sample_interval=sample_interval, cache_dir=cache_dir
    )
    realized.index = pd.DatetimeIndex(realized.index, name="timestamp")
    targets["realized_variance"] = realized.reindex(targets.index)

    return PanelSeries(
        asset_id=asset_id,
        source="binance",
        role="crypto",
        description=spec.description,
        frame=TimeSeriesFrame(data=repaired, asset_id=asset_id, source="binance"),
        targets=targets,
        components=components,
        quality=quality,
        primary_target=PRIMARY_TARGET_CRYPTO,
        archive_start=repaired.index[0],
        archive_end=repaired.index[-1],
        raw_sha256=None,
        notes=(
            f"USDT quoted ({spec.symbol}) as a USD proxy; no comparably liquid "
            "direct fiat-USD pair exists on Binance.",
            "24/7 market: the UTC-day cut is a convention, not a session. The "
            "'overnight' term is a one-minute gap and is diagnostic only.",
        ),
    )


def _minute_frame(symbol: str, first: date, last: date, cache_dir: Path) -> pd.DataFrame:
    """Concatenate the cached 1-minute OHLCV frames over ``[first, last]``.

    :func:`volbench.data.crypto.load_minute_bars` returns closes only; the range
    targets need the full bar, and re-reading the same cached parquet files is
    cheap next to re-downloading them.
    """
    frames: list[pd.DataFrame] = []
    day = first
    while day <= last:
        frames.append(fetch_and_cache_day(symbol, day, cache_dir=cache_dir))
        day += timedelta(days=1)
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def build_panel(
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    start: pd.Timestamp = PANEL_START,
    end: pd.Timestamp = PANEL_END,
    include_crypto: bool = True,
    repair_tolerance: float = DEFAULT_REPAIR_TOLERANCE,
) -> dict[str, PanelSeries]:
    """Build the whole D-004/D-012 panel, in declaration order (equities, then crypto)."""
    panel: dict[str, PanelSeries] = {}
    for asset_id in EQUITY_PANEL:
        panel[asset_id] = build_equity_series(
            asset_id,
            raw_root=raw_root,
            cache_root=cache_root,
            start=start,
            end=end,
            repair_tolerance=repair_tolerance,
        )
    if include_crypto:
        for asset_id in CRYPTO_PANEL:
            panel[asset_id] = build_crypto_series(
                asset_id,
                end=end.date(),
                cache_root=cache_root,
                repair_tolerance=repair_tolerance,
            )
    return panel
