"""Stooq daily OHLC downloader for the D-004 equity-index panel.

STATUS AS VERIFIED 2026-08-23 (see docs/data_licenses.md): stooq.com's own
CSV export endpoint (``https://stooq.com/q/d/l/``) answers HTTP requests
with a 503 JavaScript proof-of-work anti-bot challenge instead of data —
confirmed three independent ways: a plain ``curl`` GET, a real Chrome
session with live stooq.com cookies calling ``fetch()`` directly (got back
an explicit "Access denied" body), and clicking the site's own "Download
data in csv file..." link from the quote page in that same session (still
503). This module does NOT attempt to solve or bypass that challenge
(doing so would be deliberate anti-bot evasion, not a data-access
question). ``fetch_stooq_csv``/``download_index`` are implemented against
the documented CSV endpoint for the case where it *is* reachable (e.g. a
differently-provisioned network, or Stooq lifting the gate), but today
they will raise :class:`StooqBlockedError` almost everywhere. The
supported path right now is :func:`ingest_manual_csv`: a human downloads
the CSV from a browser (which solves the JS challenge) and hands the file
to this module, which does the same parsing, validation, and caching
either way. Note the *interactive* HTML quote/history pages (unlike the
CSV endpoint) are not gated at all — that's how the verification below was
done and how a human can always get the numbers by hand if needed.

Symbol verification: confirmed live on 2026-08-23 against stooq.com's
"Main Indices" listing (``https://stooq.com/t/?i=510``) and each index's
own quote page. Three of the originally-guessed symbols turned out to be
retired: Stooq no longer serves the licensed S&P Dow Jones / FTSE index
series directly (consistent with the ToS §6.1 S&P DJI licensing terms in
docs/data_licenses.md) and replies with unlicensed CFD-tracked proxy
instruments instead:

- ``^spx`` -> no longer exists; redirects with "Symbol ^SPX został
  zmieniony na ^USLC" (renamed to ^USLC, "U.S. Large Cap CFD").
- ``^dji`` -> renamed to ``^usbc`` ("U.S. Blue Chip CFD").
- ``^ftse`` -> doesn't exist at all ("nie istnieje w bazie"); the FTSE 100
  slot in Stooq's Main Indices list is now ``^uklc`` ("United Kingdom
  Large Cap CFD"), the same rebrand pattern as the two above.
- ``^ndx``, ``^dax``, ``^cac``, ``^nkx``, ``^hsi``, ``^twse``, ``^kospi``
  all confirmed correct and unchanged.

That was a data-provenance change worth a human decision, not just a symbol
fix, and the decision was taken (D-012, Phase 2): the three slots are filled
by tradable ETFs on the index — SPY, DIA, ISF — never by the CFD proxies.
``STOOQ_INDEX_SYMBOLS`` below therefore carries only the seven indices Stooq
still serves as indices; the study's actual asset list, ETFs included, is
``volbench.data.panel.EQUITY_PANEL``, the single source of truth for what the
panel contains. (The CFD entries were retired at the Phase-2 integration;
docs/PANEL_REPORT.md §9 item 8.)

Never commit downloaded data: caches live under a gitignored directory
(default ``data/cache/stooq/``) and are never vendored with the package.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
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
#: All confirmed live on 2026-08-23 (see module docstring). Only the indices
#: Stooq serves *as indices*: the SPX/DJI/FTSE slots are NOT here — Stooq
#: retired those symbols in favour of unlicensed CFD proxies (^uslc, ^usbc,
#: ^uklc), and D-012 fills the slots with ETFs (SPY/DIA/ISF) instead. The
#: panel's asset list lives in ``volbench.data.panel.EQUITY_PANEL``.
STOOQ_INDEX_SYMBOLS: dict[str, str] = {
    "NDX": "^ndx",  # NASDAQ-100 — confirmed
    "DAX": "^dax",  # DAX — confirmed
    "CAC": "^cac",  # CAC 40 — confirmed
    "NKX": "^nkx",  # Nikkei 225 — confirmed
    "HSI": "^hsi",  # Hang Seng — confirmed
    "TWSE": "^twse",  # TAIEX (Taiwan) — confirmed
    "KOSPI": "^kospi",  # KOSPI Composite (South Korea) — confirmed
}

#: A contact address in a scraper's User-Agent is courtesy to the operator of
#: the endpoint, and this one used to be hardcoded. It is read from the
#: environment now: IJF review is double-blind, this file ships in the
#: reproducibility package, and an email address in it identifies the author
#: (``tests/test_identity_leakage.py``). Unset, the header still names the tool
#: and its version, which is the part the endpoint actually needs.
_CONTACT = os.environ.get("VOLBENCH_CONTACT", "").strip()
_REQUEST_HEADERS = {
    "User-Agent": "volbench-research/0.1" + (f" (+mailto:{_CONTACT})" if _CONTACT else "")
}


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


_ANGLE_HEADER = re.compile(r"^<(.+)>$")
_YYYYMMDD = re.compile(r"^\d{8}$")

#: bulk-archive column names -> the names used everywhere else in volbench.
_BULK_COLUMN_ALIASES = {"vol": "volume"}


def _normalize_columns(columns: pd.Index) -> list[str]:
    """Lower-case headers and unwrap the bulk archive's ``<ANGLE>`` brackets.

    The per-symbol CSV export writes ``Date,Open,High,Low,Close,Volume``; the
    bulk ``d_*_txt`` archives write ``<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,
    <HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`` for the same daily bars. Both reduce
    to the same lower-case names here so one parser serves both.
    """
    names: list[str] = []
    for column in columns:
        name = str(column).strip().lower()
        match = _ANGLE_HEADER.match(name)
        if match:
            name = match.group(1).strip()
        names.append(_BULK_COLUMN_ALIASES.get(name, name))
    return names


def _parse_date_column(dates: pd.Series) -> pd.Series:
    """Parse Stooq's date column, which is ISO in the export and ``YYYYMMDD`` in bulk.

    The bulk form must be parsed with an explicit format: as a bare integer
    ``20050225`` pandas would read it as a nanosecond epoch, and as a bare
    string it is ambiguous. Getting this wrong silently reorders the panel, so
    the branch is explicit rather than left to inference.
    """
    text = dates.astype(str).str.strip()
    if bool(text.map(lambda value: _YYYYMMDD.match(value) is not None).all()):
        parsed: pd.Series = pd.to_datetime(text, format="%Y%m%d", utc=True)
        return parsed
    fallback: pd.Series = pd.to_datetime(dates, utc=True)
    return fallback


def parse_stooq_csv(raw: bytes) -> pd.DataFrame:
    """Parse a Stooq daily payload into a UTC-indexed OHLCV DataFrame.

    Accepts either shape Stooq publishes for the same daily bars:

    - the per-symbol CSV export, ``Date,Open,High,Low,Close,Volume`` with ISO
      dates (what :func:`fetch_stooq_csv` targets and what a browser download
      of a single quote page yields); and
    - a per-symbol file out of a bulk ``d_{us,uk,jp,hk,world}_txt`` archive,
      ``<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,
      <OPENINT>`` with ``YYYYMMDD`` dates.

    Returns a DataFrame indexed by UTC timestamp (midnight of the trading
    date), sorted ascending, carrying only ``open``/``high``/``low``/``close``
    and ``volume`` when present. ``<PER>``, ``<TIME>`` and ``<OPENINT>`` are
    dropped: every file here is daily (``PER=D``, ``TIME=000000``) and open
    interest is not an OHLC field. ``<TICKER>`` is dropped from the frame but
    returned in ``frame.attrs["ticker"]`` so callers can verify they opened the
    file they meant to (see :func:`ingest_manual_csv`).
    """
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = pd.Index(_normalize_columns(df.columns))
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise StooqDownloadError(f"Stooq CSV missing expected columns: {sorted(missing)}")

    ticker = None
    if "ticker" in df.columns:
        tickers = set(df["ticker"].astype(str).str.strip().str.upper())
        if len(tickers) > 1:
            raise StooqDownloadError(
                f"Stooq file mixes several tickers: {sorted(tickers)}; expected exactly one"
            )
        ticker = tickers.pop() if tickers else None

    df["date"] = _parse_date_column(df["date"])
    df = df.sort_values("date").set_index("date")
    df.index.name = "timestamp"
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep]
    out.attrs["ticker"] = ticker
    return out


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
    """Result of ingesting a manually-downloaded Stooq CSV.

    ``ticker`` is the symbol the file declared in its own ``<TICKER>`` column
    (bulk-archive files only; ``None`` for the per-symbol CSV export, which
    carries no such column).
    """

    frame: TimeSeriesFrame
    sha256: str
    ticker: str | None = None


def ingest_manual_csv(
    path: Path | str,
    *,
    asset_id: str,
    cache_dir: Path | str = Path("data/cache/stooq"),
    ingested_on: date | None = None,
    expect_ticker: str | None = None,
) -> ManualIngestResult:
    """Parse and cache a Stooq file a human downloaded in a browser.

    This is the supported path while stooq.com's anti-bot gate blocks
    :func:`download_index` (see module docstring), and it is also how the
    per-symbol files inside a hand-downloaded bulk ``d_*_txt`` archive are
    ingested — :func:`parse_stooq_csv` reads both layouts. Applies the same
    parsing, validation, hashing, and caching either way.

    ``expect_ticker`` guards against opening the wrong file out of an archive
    holding tens of thousands of symbols: if given, and the file declares its
    own ``<TICKER>``, the two must match (case-insensitively) or this raises.
    Files without a ticker column cannot be checked and pass silently.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    df = parse_stooq_csv(raw)
    ticker = df.attrs.get("ticker")
    if (
        expect_ticker is not None
        and ticker is not None
        and ticker.upper() != expect_ticker.strip().upper()
    ):
        raise StooqDownloadError(
                f"{Path(path)} declares ticker {ticker!r}, expected {expect_ticker!r} "
                f"for asset_id {asset_id!r}"
            )
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
    return ManualIngestResult(frame=frame, sha256=sha256, ticker=ticker)
