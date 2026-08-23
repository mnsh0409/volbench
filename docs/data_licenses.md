# Data licenses

> Rule: only redistributable sources may be vendored with the package.
> Every other source — including anything not yet verified below — goes
> through the bring-your-own-data (`byo`) adapter instead of being shipped.
> A row only belongs in this table once its license has been *investigated*;
> "confirmed" below means the check was actually done and the true answer
> recorded, even where that answer is "unconfirmed" or "no" — not that the
> outcome was favorable. See `docs/data_sources.md` for candidates not yet
> implemented at all.
>
> volbench never commits downloaded data (see `.gitignore`); every adapter
> below caches locally only, and every cache entry records a SHA256 of the
> raw payload it was built from.

| source | series | frequency | license | redistributable | access method | checksum | verified_on |
|--------|--------|-----------|---------|------------------|----------------|----------|--------------|
| Stooq (stooq.com) | D-004 equity indices: SPX, NDX, DJI, DAX, FTSE 100, CAC 40, Nikkei 225 (NKX), Hang Seng (HSI), TAIEX (TWSE), KOSPI | daily OHLC | **UNCONFIRMED.** No accessible terms-of-service text was found: `stooq.com/regulamin.aspx` returns HTTP 404, and the homepage returned no legal/terms content to automated retrieval. | **NO.** Automated download is technically blocked, not just legally ambiguous: a plain HTTP GET to the CSV export endpoint returns a JavaScript proof-of-work anti-bot challenge page instead of data — verified directly with `curl` on 2026-08-23 (see `src/volbench/data/stooq.py` module docstring for the evidence). `volbench` does not attempt to solve/bypass that challenge. Redistribution rights would remain unconfirmed even if the gate were not there. | `src/volbench/data/stooq.py`: `fetch_stooq_csv()`/`download_index()` target `https://stooq.com/q/d/l/` (documented CSV export endpoint) but will raise `StooqBlockedError` today; `ingest_manual_csv()` is the supported path — a human downloads the CSV in a browser (which solves the JS challenge) and hands the file to this module for the same parsing/validation/caching. | SHA256 of the raw CSV payload, recorded in a JSON sidecar next to each cached parquet (`data/cache/stooq/*.json`) | 2026-08-23 |
| Binance (`data.binance.vision` bulk archive + `api.binance.com` REST) | BTC-USD, ETH-USD 1-minute bars (as `BTCUSDT`/`ETHUSDT` — USDT used as a USD proxy, see module docstring) → daily realized variance | 1-minute OHLCV | **UNCONFIRMED for the market data itself.** The companion GitHub client (`binance/binance-public-data`) is MIT-licensed, but that governs the *download script*, not the data it fetches. Binance's Terms of Use restrict commercial "data feeding/streaming" services and profiting from Binance market data without written consent; no explicit statement about non-commercial academic redistribution of a small derived series was found in the portions of the ToS retrieved (`binance.com/en/terms` gave only a footer/risk-warning excerpt; the full terms document was not reachable via automated fetch). | **NO (unconfirmed)** for raw bars — never vendored regardless of the answer (cached locally only, gitignored). The only thing this module hands callers is a heavily aggregated derived series (daily RV); whether *that* may be archived/redistributed (the Zenodo question already flagged as open in `docs/data_sources.md`) needs a human legal read of Binance's full ToS before any redistribution decision is made. Unlike Stooq, the endpoints themselves are reachable (verified with `curl`, 2026-08-23) — Binance actively documents this bulk-download path for exactly this kind of use. | `src/volbench/data/crypto.py`: `fetch_and_cache_day()` / `load_minute_bars()` / `daily_realized_variance()`, against `https://data.binance.vision/data/spot/daily/klines/...` | SHA256 of each day's raw zip archive, recorded in a JSON sidecar next to each cached parquet (`data/cache/crypto/*.json`) | 2026-08-23 |
| Bring-your-own-data (`byo`) | Any OHLC/close series the user already holds a license for (CRSP, Bloomberg, Refinitiv, a manually-downloaded Stooq CSV, ...) | any | User's own — volbench neither fetches nor vendors anything through this path. | N/A — not distributed by volbench; the user supplies and is responsible for the data's license. | `src/volbench/data/byo.py`: `load_ohlc_csv()` / `load_ohlc_parquet()` | N/A (no download, so no payload to hash) | 2026-08-23 |

## Open items for a human decision (flagging per CLAUDE.md — not resolved here)

- **Stooq symbol verification.** Because the endpoint is blocked, the ticker
  map in `STOOQ_INDEX_SYMBOLS` could not be confirmed end-to-end against a
  live response. `^spx`, `^dji`, `^dax` are corroborated by third-party
  usage (e.g. the pandas-datareader Stooq test suite); `^ndx` (vs. `^ndq`
  for the Composite), the FTSE 100 code (sources disagree between `^ftse`,
  `^ftm`, and `^uk100`), `^twse` (vs. `^twii`), and `^kospi` are best-effort
  guesses. Confirm via a manual browser download through `ingest_manual_csv`
  before trusting this panel for real results.
- **Stooq redistribution/access.** A human should (a) locate stooq.com's
  actual terms (not found by this session) and (b) decide whether the
  benchmark's index panel should instead be sourced from a provider with an
  unambiguous API and license, given the anti-bot gate.
- **Binance redistribution of the derived RV series.** Needed before the
  Zenodo-archival idea in `docs/data_sources.md` can proceed for the crypto
  arm — read Binance's full Terms of Use (not just the excerpt reachable
  here) or seek written permission.
