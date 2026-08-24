"""Stooq downloader: parsing, caching, hashing, and anti-bot-block detection.

No network access here — fetch_stooq_csv is never exercised end-to-end;
only its response-classification helper and the offline parse/cache/ingest
paths are tested, against a committed fixture CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from volbench.data.stooq import (
    STOOQ_INDEX_SYMBOLS,
    StooqDownloadError,
    _looks_like_challenge_or_html,
    download_index,
    ingest_manual_csv,
    parse_stooq_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stooq_sample.csv"

# A short excerpt representative of what stooq.com actually returned when
# probed with a plain HTTP GET on 2026-08-23 (see stooq.py's module docstring).
_CHALLENGE_PAGE = (
    b'<!DOCTYPE html><html><head><meta charset="utf-8">'
    b'<meta name="robots" content="noindex,nofollow"></head><body>'
    b"<noscript>This site requires JavaScript to verify your browser.</noscript>"
    b'<script nonce="abc">(async()=>{})();</script></body></html>'
)


class TestSymbolMap:
    def test_expected_assets_present(self) -> None:
        expected = {
            "SPX", "NDX", "DJI", "DAX", "FTSE", "CAC", "NKX", "HSI", "TWSE", "KOSPI",
        }
        assert set(STOOQ_INDEX_SYMBOLS) == expected

    def test_symbols_are_caret_prefixed(self) -> None:
        assert all(sym.startswith("^") for sym in STOOQ_INDEX_SYMBOLS.values())


class TestChallengeDetection:
    def test_detects_known_challenge_page(self) -> None:
        assert _looks_like_challenge_or_html(_CHALLENGE_PAGE)

    def test_does_not_flag_real_csv(self) -> None:
        raw = FIXTURE.read_bytes()
        assert not _looks_like_challenge_or_html(raw)


class TestParseStooqCsv:
    def test_parses_fixture(self) -> None:
        raw = FIXTURE.read_bytes()
        df = parse_stooq_csv(raw)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.is_monotonic_increasing
        assert str(df.index.tz) == "UTC"
        assert len(df) == 5
        assert df["close"].iloc[0] == pytest.approx(4742.83)

    def test_missing_columns_raises(self) -> None:
        raw = b"Date,Open,High\n2024-01-02,1.0,2.0\n"
        with pytest.raises(StooqDownloadError, match="missing expected columns"):
            parse_stooq_csv(raw)


class TestIngestManualCsv:
    def test_ingest_builds_frame_and_caches(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "stooq_cache"
        ingested_on = pd.Timestamp("2024-06-01").date()
        result = ingest_manual_csv(
            FIXTURE, asset_id="SPX", cache_dir=cache_dir, ingested_on=ingested_on
        )
        assert result.frame.asset_id == "SPX"
        assert result.frame.source == "stooq"
        assert len(result.frame) == 5
        assert result.sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

        cached_files = list(cache_dir.glob("*.parquet"))
        meta_files = list(cache_dir.glob("*.json"))
        assert len(cached_files) == 1
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["asset_id"] == "SPX"
        assert meta["symbol"] == STOOQ_INDEX_SYMBOLS["SPX"]
        assert meta["sha256"] == result.sha256

    def test_cache_roundtrip_via_download_index(self, tmp_path: Path) -> None:
        # ingest_manual_csv and download_index's cache reader must agree on the
        # cache layout: ingest once manually, then confirm download_index(force=False)
        # reads the same cache instead of trying (and failing) to hit the network.
        cache_dir = tmp_path / "stooq_cache"
        ingested_date = pd.Timestamp("2024-06-01").date()
        ingest_manual_csv(FIXTURE, asset_id="SPX", cache_dir=cache_dir, ingested_on=ingested_date)

        frame = download_index("SPX", cache_dir=cache_dir, download_date=ingested_date)
        assert len(frame) == 5
        assert frame.asset_id == "SPX"

    def test_unknown_asset_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            download_index("NOT_A_REAL_ASSET", cache_dir=tmp_path)


BULK_FIXTURE = Path(__file__).parent / "fixtures" / "stooq_bulk_sample.txt"


class TestParseBulkArchiveFormat:
    """The bulk ``d_*_txt`` archives carry the same daily bars in a second shape.

    ``<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>``
    with ``YYYYMMDD`` dates, versus the per-symbol export's
    ``Date,Open,High,Low,Close,Volume`` with ISO dates. One parser reads both so
    ``ingest_manual_csv`` — the only sanctioned Stooq path — serves either.
    """

    def test_parses_the_bulk_layout(self) -> None:
        df = parse_stooq_csv(BULK_FIXTURE.read_bytes())
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 5
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing

    def test_yyyymmdd_dates_are_read_as_calendar_dates(self) -> None:
        # Read as a bare integer, 20050103 would become a nanosecond epoch and
        # the whole panel would silently land in 1970.
        df = parse_stooq_csv(BULK_FIXTURE.read_bytes())
        assert df.index[0] == pd.Timestamp("2005-01-03", tz="UTC")
        assert df.index[-1] == pd.Timestamp("2005-01-07", tz="UTC")

    def test_per_time_and_openint_are_dropped(self) -> None:
        df = parse_stooq_csv(BULK_FIXTURE.read_bytes())
        assert not {"per", "time", "openint", "ticker"} & set(df.columns)

    def test_ticker_is_reported_in_attrs(self) -> None:
        assert parse_stooq_csv(BULK_FIXTURE.read_bytes()).attrs["ticker"] == "FAKE.US"

    def test_export_format_has_no_ticker(self) -> None:
        assert parse_stooq_csv(FIXTURE.read_bytes()).attrs["ticker"] is None

    def test_prices_survive_the_round_trip(self) -> None:
        df = parse_stooq_csv(BULK_FIXTURE.read_bytes())
        assert df["open"].iloc[0] == pytest.approx(100.0)
        assert df["close"].iloc[-1] == pytest.approx(100.1)

    def test_a_file_mixing_tickers_is_refused(self) -> None:
        raw = (
            b"<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
            b"A.US,D,20050103,000000,1,2,0.5,1.5,1,0\n"
            b"B.US,D,20050104,000000,1,2,0.5,1.5,1,0\n"
        )
        with pytest.raises(StooqDownloadError, match="mixes several tickers"):
            parse_stooq_csv(raw)


class TestIngestTickerGuard:
    def test_matching_ticker_passes_and_is_reported(self, tmp_path: Path) -> None:
        result = ingest_manual_csv(
            BULK_FIXTURE,
            asset_id="FAKE",
            cache_dir=tmp_path,
            expect_ticker="FAKE.US",
            ingested_on=pd.Timestamp("2026-08-25").date(),
        )
        assert result.ticker == "FAKE.US"
        assert len(result.frame) == 5

    def test_ticker_match_is_case_insensitive(self, tmp_path: Path) -> None:
        result = ingest_manual_csv(
            BULK_FIXTURE,
            asset_id="FAKE",
            cache_dir=tmp_path,
            expect_ticker="fake.us",
            ingested_on=pd.Timestamp("2026-08-25").date(),
        )
        assert result.ticker == "FAKE.US"

    def test_wrong_ticker_raises(self, tmp_path: Path) -> None:
        # The bulk archives hold tens of thousands of symbols; opening the
        # wrong one must fail loudly, not produce a plausible wrong panel.
        with pytest.raises(StooqDownloadError, match="declares ticker"):
            ingest_manual_csv(
                BULK_FIXTURE,
                asset_id="NDX",
                cache_dir=tmp_path,
                expect_ticker="^NDX",
                ingested_on=pd.Timestamp("2026-08-25").date(),
            )

    def test_export_format_cannot_be_checked_and_passes(self, tmp_path: Path) -> None:
        result = ingest_manual_csv(
            FIXTURE,
            asset_id="SPX",
            cache_dir=tmp_path,
            expect_ticker="^SPX",
            ingested_on=pd.Timestamp("2026-08-25").date(),
        )
        assert result.ticker is None
