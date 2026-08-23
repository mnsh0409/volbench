# 06 · Data sources & licenses

> Status: CANDIDATES — confirm license terms yourself before any data ships with the package. Rule: only redistributable sources are vendored; everything else goes through the bring-your-own-data adapter. Licensing is the #1 reproducibility trap in financial benchmarking.

| source | series | frequency | license | redistributable? | access method | checksum/version | used in |
|-----------|--------------------|-----------|--------------------|-----------|------------------|---------|-----------|
| Stooq | Equity indices (^SPX, ^DAX, …) | daily | TO-CONFIRM | TO-CONFIRM | CSV download script | [ ] | core panel |
| FRED | Rates, macro controls | daily | Public (check per-series) | mostly yes | fredapi / CSV | [ ] | controls |
| Crypto exchange (e.g., Binance public data) | BTC, ETH | 1-min → RV | TO-CONFIRM (usually permissive) | TO-CONFIRM | bulk dumps | [ ] | intraday RV arm |
| M6 competition data | 50 stocks + 50 ETFs | daily | Public competition data — confirm terms | TO-CONFIRM | GitHub | [ ] | robustness panel |
| Kaggle Optiver RV dataset | Book/trade snapshots | intraday | Kaggle competition rules — likely NOT redistributable | likely no | BYO adapter only | [ ] | optional |
| BYO (CRSP/Bloomberg/Refinitiv) | user-supplied | any | user's own | no | `byo` adapter | n/a | reviewers/practitioners |

## Decisions needed (→ 09_decisions_log)
- [ ] Final core panel (which indices, which span)
- [ ] Intraday source for RV targets (crypto only vs. also equities via a licensed BYO example)
- [ ] Snapshot strategy: Zenodo-archived copy of every redistributable input, checksummed

## Rules
- Every download script pins URL + date + SHA256.
- `06` table and paper Table T2 must match exactly at submission.
- yfinance/Yahoo scraping: avoid for the shipped benchmark (ToS gray zone) — BYO adapter only.
