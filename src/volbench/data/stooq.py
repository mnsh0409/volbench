"""Stooq daily OHLC downloader for the D-004 equity-index panel.

STATUS AS VERIFIED 2026-08-23 (see docs/data_licenses.md): stooq.com's own
CSV export endpoint (``https://stooq.com/q/d/l/``) currently answers
automated HTTP requests with a JavaScript proof-of-work anti-bot challenge
page instead of data — confirmed with a plain ``curl`` GET, not just an
assumption. This module therefore does NOT attempt to solve or bypass that
challenge (doing so would be deliberate anti-bot evasion, not a data-access
question). ``fetch_stooq_csv``/``download_index`` are implemented against the
documented CSV endpoint for the case where it *is* reachable (e.g. a
differently-provisioned network, or Stooq lifting the gate), but today they
will raise :class:`StooqBlockedError` almost everywhere. The supported path
right now is :func:`ingest_manual_csv`: a human downloads the CSV from a
browser (which solves the JS challenge) and hands the file to this module,
which does the same parsing, validation, and caching either way.

Symbol verification caveat: because the endpoint is blocked, the ticker map
below could not be confirmed end-to-end against a live response. ``^spx``,
``^dji``, and ``^dax`` are corroborated by third-party usage (e.g. the
pandas-datareader Stooq test suite); ``^ndx`` (vs. ``^ndq``), the FTSE 100
code (community sources disagree between ``^ftse``, ``^ftm``, and
``^uk100``), ``^twse`` (vs. ``^twii``), and ``^kospi`` are best-effort and
UNVERIFIED — confirm against a real response (e.g. via a manual browser
download through :func:`ingest_manual_csv`) before trusting this panel.

Never commit downloaded data: caches live under a gitignored directory
(default ``data/cache/stooq/``) and are never vendored with the package.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import requests

from volbench.data.types import TimeSeriesFrame

__all__ = [
    "STOOQ_INDEX_SYMBOLS",
    "StooqBlockedError",
    "StooqDownloadError",
    "download_index",
    "fetch_stooq_csv",
    "ingest_manual_csv",
    "parse_stooq_csv",
]

STOOQ_CSV_URL = "https://stooq.com/q/d/l/"

#: canonical asset id -> Stooq ticker for the D-004 core equity-index panel.
#: See the module docstring: several of these are unverified best-effort guesses.
STOOQ_INDEX_SYMBOLS: dict[str, str] = {
    "SPX": "^spx",  # S&P 500 — corroborated
    "NDX": "^ndx",  # NASDAQ-100 — UNVERIFIED (possibly ^ndq for the Composite)
    "DJI": "^dji",  # Dow Jones Industrial Average — corroborated
    "DAX": "^dax",  # DAX — corroborated
    "FTSE": "^ftse",  # FTSE 100 — UNVERIFIED (sources also suggest ^ftm / ^uk100)
    "CAC": "^cac",  # CAC 40 — UNVERIFIED
    "NKX": "^nkx",  # Nikkei 225 — UNVERIFIED
    "HSI": "^hsi",  # Hang Seng — UNVERIFIED
    "TWSE": "^twse",  # TAIEX (Taiwan) — UNVERIFIED (sources also suggest ^twii)
    "KOSPI": "^kospi",  # KOSPI Composite (South Korea) — UNVERIFIED
}

_REQUEST_HEADERS = {"User-Agent": "volbench-research/0.1 (+mailto:martin.ai.nlp@gmail.com)"}


class StooqDownloadError(RuntimeError):
    """Raised when Stooq responds but the payload is not usable OHLC CSV."""


class StooqBlockedError(StooqDownloadError):
    """Raised when Stooq's response is its JS proof-of-work anti-bot challenge.

    This module will not attempt to solve that challenge. Use
    :func:`ingest_manual_csv` with a browser-downloaded CSV instead.
    """


def _looks_like_challenge_or_html(payload: bytes) -> bool:
    head = payload[:512].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<script" in head


def fetch_stooq_csv(
    symbol: str, *, session: requests.Session | None = None, timeout: float = 15.0
) -> bytes:
    """GET the raw CSV payload for ``symbol`` from Stooq's daily export endpoint.

    Raises :class:`StooqBlockedError` if the response is the anti-bot
    challenge page, and :class:`StooqDownloadError` for any other
    non-CSV or empty response.
    """
    http = session or requests
    resp = http.get(
        STOOQ_CSV_URL, params={"s": symbol, "i": "d"}, headers=_REQUEST_HEADERS, timeout=timeout
    )
    resp.raise_for_status()
    payload = resp.content
    if not payload:
        raise StooqDownloadError(f"empty response for symbol {symbol!r}")
    if _looks_like_challenge_or_html(payload):
        raise StooqBlockedError(
            f"stooq.com returned an HTML/anti-bot challenge page for symbol {symbol!r} "
            "instead of CSV data; this module does not attempt to bypass it. "
            "Download the CSV manually in a browser and pass it to ingest_manual_csv()."
        )
    if payload.strip().lower().startswith(b"exceeded"):
        raise StooqDownloadError(f"stooq.com rate limit hit for symbol {symbol!r}: {payload!r}")
    return payload


def parse_stooq_csv(raw: bytes) -> pd.DataFrame:
    """Parse a Stooq daily CSV payload (``Date,Open,High,Low,Close,Volume``).

    Returns a DataFrame indexed by UTC timestamp (midnight of the trading
    date) with lower-cased columns, sorted ascending by date.
    """
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise StooqDownloadError(f"Stooq CSV missing expected columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").set_index("date")
    df.index.name = "timestamp"
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("^", "idx_").replace("/", "_")


def _cache_paths(cache_dir: Path, symbol: str, download_date: date) -> tuple[Path, Path]:
    stem = f"stooq_{_sanitize_symbol(symbol)}_{download_date.isoformat()}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def download_index(
    asset_id: str,
    *,
    cache_dir: Path | str = Path("data/cache/stooq"),
    download_date: date | None = None,
    session: requests.Session | None = None,
    force: bool = False,
) -> TimeSeriesFrame:
    """Fetch (or load from cache) daily OHLC for one D-004 index.

    Cached under ``cache_dir`` as ``stooq_<symbol>_<download_date>.parquet``
    plus a JSON sidecar carrying the source symbol, download timestamp, and
    the SHA256 of the raw payload. Never committed (``data/`` is gitignored).
    """
    if asset_id not in STOOQ_INDEX_SYMBOLS:
        raise KeyError(
            f"unknown asset_id {asset_id!r}; known assets: {sorted(STOOQ_INDEX_SYMBOLS)}"
        )
    symbol = STOOQ_INDEX_SYMBOLS[asset_id]
    resolved_date = download_date or datetime.now(tz=UTC).date()

    cache_dir_path = Path(cache_dir)
    parquet_path, meta_path = _cache_paths(cache_dir_path, symbol, resolved_date)
    if not force and parquet_path.exists() and meta_path.exists():
        df = pd.read_parquet(parquet_path)
        return TimeSeriesFrame(data=df, asset_id=asset_id, source="stooq")

    raw = fetch_stooq_csv(symbol, session=session)
    sha256 = hashlib.sha256(raw).hexdigest()
    df = parse_stooq_csv(raw)
    frame = TimeSeriesFrame(data=df, asset_id=asset_id, source="stooq")

    cache_dir_path.mkdir(parents=True, exist_ok=True)
    frame.data.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "symbol": symbol,
                "download_date": resolved_date.isoformat(),
                "downloaded_at": datetime.now(tz=UTC).isoformat(),
                "sha256": sha256,
                "url": STOOQ_CSV_URL,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return frame


@dataclass(frozen=True)
class ManualIngestResult:
    """Result of ingesting a manually-downloaded Stooq CSV."""

    frame: TimeSeriesFrame
    sha256: str


def ingest_manual_csv(
    path: Path | str,
    *,
    asset_id: str,
    cache_dir: Path | str = Path("data/cache/stooq"),
    ingested_on: date | None = None,
) -> ManualIngestResult:
    """Parse and cache a CSV a human downloaded from stooq.com in a browser.

    This is the supported path while stooq.com's anti-bot gate blocks
    :func:`download_index` (see module docstring). Applies the same parsing,
    validation, hashing, and caching as the automated path.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    df = parse_stooq_csv(raw)
    frame = TimeSeriesFrame(data=df, asset_id=asset_id, source="stooq")

    symbol = STOOQ_INDEX_SYMBOLS.get(asset_id, asset_id)
    resolved_date = ingested_on or datetime.now(tz=UTC).date()
    cache_dir_path = Path(cache_dir)
    parquet_path, meta_path = _cache_paths(cache_dir_path, symbol, resolved_date)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    frame.data.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "symbol": symbol,
                "download_date": resolved_date.isoformat(),
                "ingested_at": datetime.now(tz=UTC).isoformat(),
                "sha256": sha256,
                "source": "manual-browser-download",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return ManualIngestResult(frame=frame, sha256=sha256)
