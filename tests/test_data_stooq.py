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
