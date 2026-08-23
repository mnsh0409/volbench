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
| Stooq (stooq.com) | D-004 equity indices: SPX, NDX, DJI, DAX, FTSE 100, CAC 40, Nikkei 225 (NKX), Hang Seng (HSI), TAIEX (TWSE), KOSPI | daily OHLC | **CONFIRMED, and it's a NO.** Read the actual ToS at `https://stooq.com/terms.html` on 2026-08-23 (found via a real browser session — it wasn't reachable by URL-guessing beforehand). §5.3: *"Redistribution of data found on the website is not allowed without the consent of Stooq."* §6.1 additionally restricts the S&P Dow Jones indices specifically to "personal, non-commercial purposes" and forbids using the data to build financial instruments/products. | **NO — explicit, not just unconfirmed.** On top of the license text: automated download is also technically blocked. A plain HTTP GET to the CSV export endpoint returns a 503 JS proof-of-work anti-bot page. This was re-verified from an authenticated, cookie-carrying real Chrome session (not just `curl`): a same-origin `fetch()` to the endpoint got back an explicit `"Access denied"` body, and clicking the site's own "Download data in csv file..." link from the quote page (proper referer, real click) still returned 503. `volbench` does not attempt to solve/bypass that challenge. Note the *interactive* HTML history-table pages are not gated — a human (or `ingest_manual_csv`, fed a browser-downloaded file) can always get the numbers that way. | `src/volbench/data/stooq.py`: `fetch_stooq_csv()`/`download_index()` target `https://stooq.com/q/d/l/` (documented CSV export endpoint) but will raise `StooqBlockedError` today; `ingest_manual_csv()` is the supported path and was exercised end-to-end on 2026-08-23 against 40 rows of genuine live `^UKLC` data read from its quote page — parsed, validated into a `TimeSeriesFrame`, cached to parquet+JSON, and scored with `proxies.squared_return`/`parkinson` successfully. | SHA256 of the raw CSV payload, recorded in a JSON sidecar next to each cached parquet (`data/cache/stooq/*.json`) | 2026-08-23 |
| Binance (`data.binance.vision` bulk archive + `api.binance.com` REST) | BTC-USD, ETH-USD 1-minute bars (as `BTCUSDT`/`ETHUSDT` — USDT used as a USD proxy, see module docstring) → daily realized variance | 1-minute OHLCV | **UNCONFIRMED for the market data itself.** The companion GitHub client (`binance/binance-public-data`) is MIT-licensed, but that governs the *download script*, not the data it fetches. Binance's Terms of Use restrict commercial "data feeding/streaming" services and profiting from Binance market data without written consent; no explicit statement about non-commercial academic redistribution of a small derived series was found in the portions of the ToS retrieved (`binance.com/en/terms` gave only a footer/risk-warning excerpt; the full terms document was not reachable via automated fetch). | **NO (unconfirmed)** for raw bars — never vendored regardless of the answer (cached locally only, gitignored). The only thing this module hands callers is a heavily aggregated derived series (daily RV); whether *that* may be archived/redistributed (the Zenodo question already flagged as open in `docs/data_sources.md`) needs a human legal read of Binance's full ToS before any redistribution decision is made. Unlike Stooq, the endpoints themselves are reachable (verified with `curl`, 2026-08-23) — Binance actively documents this bulk-download path for exactly this kind of use. | `src/volbench/data/crypto.py`: `fetch_and_cache_day()` / `load_minute_bars()` / `daily_realized_variance()`, against `https://data.binance.vision/data/spot/daily/klines/...` | SHA256 of each day's raw zip archive, recorded in a JSON sidecar next to each cached parquet (`data/cache/crypto/*.json`) | 2026-08-23 |
| Bring-your-own-data (`byo`) | Any OHLC/close series the user already holds a license for (CRSP, Bloomberg, Refinitiv, a manually-downloaded Stooq CSV, ...) | any | User's own — volbench neither fetches nor vendors anything through this path. | N/A — not distributed by volbench; the user supplies and is responsible for the data's license. | `src/volbench/data/byo.py`: `load_ohlc_csv()` / `load_ohlc_parquet()` | N/A (no download, so no payload to hash) | 2026-08-23 |

## Open items for a human decision (flagging per CLAUDE.md — not resolved here)

- **Stooq no longer serves the literal SPX/DJI/FTSE 100 indices.** Verified
  live on 2026-08-23 against stooq.com's "Main Indices" listing
  (`https://stooq.com/t/?i=510`) and each symbol's own quote page:
  - `^spx` no longer exists; Stooq's own redirect message reads *"Symbol
    ^SPX został zmieniony na ^USLC"* (renamed to ^USLC, "U.S. Large Cap
    CFD").
  - `^dji` similarly redirects to `^usbc` ("U.S. Blue Chip CFD").
  - `^ftse` doesn't exist at all (*"Symbol ^FTSE nie istnieje w bazie"* —
    not in the database); the FTSE 100 slot in the Main Indices list is now
    `^uklc` ("United Kingdom Large Cap CFD").
  - `^ndx`, `^dax`, `^cac`, `^nkx`, `^hsi`, `^twse`, `^kospi` are all
    confirmed correct and unchanged.

  `STOOQ_INDEX_SYMBOLS` has been updated to the CFD proxies so the module
  points at symbols that actually exist, but **this is a substitution a
  human should sign off on**, not just a ticker fix: a CFD instrument
  tracking an index is not the same series as the licensed index itself
  (different composition/calculation/timing are possible), and
  `docs/research_design.md` names "S&P 500, ... Dow, ... FTSE 100" as the
  panel. Options: (a) accept the CFD proxies as close-enough substitutes
  and document it in the paper's data section, or (b) source SPX/DJI/FTSE
  100 from elsewhere (FRED has some; a licensed provider is another route)
  and keep Stooq only for the indices it still serves directly (NDX, DAX,
  CAC, NKX, HSI, TWSE, KOSPI).
- **Stooq redistribution is a confirmed NO**, not just unconfirmed — see
  the table row above (ToS §5.3, plus §6.1's tighter "personal,
  non-commercial" restriction specifically on S&P Dow Jones data, which
  would have applied to the original `^spx`/`^dji` symbols anyway). This
  closes the earlier "locate stooq.com's actual terms" item. Given the
  anti-bot gate on top of that, a human should decide whether the D-004
  index panel should instead be sourced from a provider with an
  unambiguous API and license.
- **Binance redistribution of the derived RV series.** Needed before the
  Zenodo-archival idea in `docs/data_sources.md` can proceed for the crypto
  arm — read Binance's full Terms of Use (not just the excerpt reachable
  here) or seek written permission.
