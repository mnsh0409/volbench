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
| Stooq (stooq.com) | The panel's equity leg (D-004, as amended by D-012 and D-020): the seven indices Stooq still serves — NDX, DAX, CAC 40, Nikkei 225 (NKX), Hang Seng (HSI), TAIEX (TWSE), KOSPI — plus the SPY and DIA ETF files standing in for SPX and DJI. FTSE 100 is no longer taken from here, or anywhere (D-020). | daily OHLC | **CONFIRMED, and it's a NO.** Read the actual ToS at `https://stooq.com/terms.html` on 2026-08-23 (found via a real browser session — it wasn't reachable by URL-guessing beforehand). §5.3: *"Redistribution of data found on the website is not allowed without the consent of Stooq."* §6.1 additionally restricts the S&P Dow Jones indices specifically to "personal, non-commercial purposes" and forbids using the data to build financial instruments/products. | **NO — explicit, not just unconfirmed.** On top of the license text: automated download is also technically blocked. A plain HTTP GET to the CSV export endpoint returns a 503 JS proof-of-work anti-bot page. This was re-verified from an authenticated, cookie-carrying real Chrome session (not just `curl`): a same-origin `fetch()` to the endpoint got back an explicit `"Access denied"` body, and clicking the site's own "Download data in csv file..." link from the quote page (proper referer, real click) still returned 503. `volbench` does not attempt to solve/bypass that challenge. Note the *interactive* HTML history-table pages are not gated — a human (or `ingest_manual_csv`, fed a browser-downloaded file) can always get the numbers that way. | `src/volbench/data/stooq.py`: `fetch_stooq_csv()`/`download_index()` target `https://stooq.com/q/d/l/` (documented CSV export endpoint) but will raise `StooqBlockedError` today; `ingest_manual_csv()` is the supported path and was exercised end-to-end on 2026-08-23 against 40 rows of genuine live `^UKLC` data read from its quote page — parsed, validated into a `TimeSeriesFrame`, cached to parquet+JSON, and scored with `proxies.squared_return`/`parkinson` successfully. | SHA256 of the raw CSV payload, recorded in a JSON sidecar next to each cached parquet (`data/cache/stooq/*.json`) | 2026-08-23 |
| Binance (`data.binance.vision` bulk archive + `api.binance.com` REST) | BTC-USD, ETH-USD 1-minute bars (as `BTCUSDT`/`ETHUSDT` — USDT used as a USD proxy, see module docstring) → daily realized variance | 1-minute OHLCV | **UNCONFIRMED for the market data itself.** The companion GitHub client (`binance/binance-public-data`) is MIT-licensed, but that governs the *download script*, not the data it fetches. Binance's Terms of Use restrict commercial "data feeding/streaming" services and profiting from Binance market data without written consent; no explicit statement about non-commercial academic redistribution of a small derived series was found in the portions of the ToS retrieved (`binance.com/en/terms` gave only a footer/risk-warning excerpt; the full terms document was not reachable via automated fetch). | **NO (unconfirmed)** for raw bars — never vendored regardless of the answer (cached locally only, gitignored). The only thing this module hands callers is a heavily aggregated derived series (daily RV); whether *that* may be archived/redistributed (the Zenodo question already flagged as open in `docs/data_sources.md`) needs a human legal read of Binance's full ToS before any redistribution decision is made. Unlike Stooq, the endpoints themselves are reachable (verified with `curl`, 2026-08-23) — Binance actively documents this bulk-download path for exactly this kind of use. | `src/volbench/data/crypto.py`: `fetch_and_cache_day()` / `load_minute_bars()` / `daily_realized_variance()`, against `https://data.binance.vision/data/spot/daily/klines/...` | SHA256 of each day's raw zip archive, recorded in a JSON sidecar next to each cached parquet (`data/cache/crypto/*.json`) | 2026-08-23 |
| Bring-your-own-data (`byo`) | Any OHLC/close series the user already holds a license for (CRSP, Bloomberg, Refinitiv, a manually-downloaded Stooq CSV, ...) | any | User's own — volbench neither fetches nor vendors anything through this path. | N/A — not distributed by volbench; the user supplies and is responsible for the data's license. | `src/volbench/data/byo.py`: `load_ohlc_csv()` / `load_ohlc_parquet()` | N/A (no download, so no payload to hash) | 2026-08-23 |

## Open items — and what has since been settled

### Settled

- **SPX and DJI are read from the SPY and DIA tracking ETFs** (D-012;
  `volbench.data.panel.EQUITY_PANEL`). Verified live on 2026-08-23 that Stooq
  no longer serves the literal indices: `^spx` redirects with Stooq's own
  message *"Symbol ^SPX został zmieniony na ^USLC"* (renamed to ^USLC, "U.S.
  Large Cap CFD") and `^dji` likewise redirects to `^usbc` ("U.S. Blue Chip
  CFD"). The CFD substitution was **not** accepted: D-024 removed those
  entries from `STOOQ_INDEX_SYMBOLS` entirely, so no code path points at them.

  What replaced it matters for this file specifically, because it changes
  *which product* is being read. **An ETF trade price is not the licensed
  index product.** SPY and DIA are exchange-traded funds whose prices are
  their own trades on NYSE Arca, not S&P Dow Jones Indices' calculated index
  values; the index licence that ToS §6.1 restricts to "personal,
  non-commercial purposes" governs the index level, and volbench never reads
  one. That does not make the ETF *files* redistributable — Stooq's §5.3 blanket
  "redistribution of data found on the website is not allowed" still applies to
  anything downloaded from stooq.com, and volbench still commits nothing and
  still fetches nothing programmatically. It means the narrower, index-specific
  restriction is not the binding constraint here, and the substitution is a
  *data* decision (tracking error, the fund's own session, its own dividends)
  rather than a licence one. Both properties are recorded on the series
  itself — `EquitySpec.role="etf_proxy"` and `proxy_for` — and restated in
  every report the panel feeds.

- **The FTSE 100 leg is dropped** (D-020), which closes the third of the three
  slots this section used to ask about. Stooq does not have it (`^ftse`
  returns *"Symbol ^FTSE nie istnieje w bazie"* — not in the database; the
  Main Indices list offers `^uklc`, another unlicensed CFD), and the one
  tradable stand-in with a UK listing, the iShares Core FTSE 100 ETF (ISF),
  starts 2015-03-04 and so holds zero observations in the GFC window. The slot
  was therefore dropped from the study rather than filled with a series that
  could not answer its crisis question; the panel is 11 assets. No licence
  question remains for it, because no FTSE 100 data is read at all.
  `RETIRED_EQUITY` keeps the ISF ingestion path, which reads a
  hand-downloaded file under exactly the same rules as every other Stooq file.

- **Stooq redistribution is a confirmed NO**, not merely unconfirmed — ToS
  §5.3, plus §6.1's tighter "personal, non-commercial" restriction on S&P Dow
  Jones data. Combined with the anti-bot gate on the CSV endpoint, the
  hand-download path (`ingest_manual_csv`) is the only supported one and the
  panel module is tested to import no network entry point at all. Nothing is
  vendored, and nothing about the ETF substitution above relaxes this.

### Still open

- **Binance redistribution of the derived RV series.** Unchanged: needed
  before the Zenodo-archival idea in `docs/data_sources.md` can proceed for the
  crypto arm. Binance's Terms of Use restrict commercial data-feeding services
  and profiting from Binance market data without written consent; no explicit
  statement about non-commercial academic redistribution of a small *derived*
  series was found in the portions retrievable here (`binance.com/en/terms`
  returned only a footer/risk-warning excerpt). Requires a human legal read of
  the full Terms of Use, or written permission. Until then the raw bars are
  cached locally and gitignored, as they are today, and nothing derived from
  them is published.

- **Whether the index panel should be re-sourced entirely** from a provider
  with an unambiguous API and licence. Not urgent — the hand-download path
  works, is reproducible from a recorded SHA256, and vendors nothing — but it
  is the structural fix, and it would also restore the FTSE 100 slot.
