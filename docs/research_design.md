# 02 · Research design v0.2 (updated 2026-08-19)

> Status: PROPOSED — concrete selections pre-filled per D-002…D-006; Martin confirms or edits, then this is the contract P4 audits. One page; out-of-scope list is the guard rail.

QUESTION
How do model families — GARCH/HAR econometrics, statistical forecasting, ML/DL, and zero-shot time-series foundation models — compare on probabilistic volatility and tail-risk forecasting, under a leakage-safe, finance-native evaluation?

HYPOTHESES (falsifiable)
- H1: Zero-shot TSFMs do not beat HAR-RV/GARCH baselines on CRPS for daily variance at h=1 (test: MCS membership at α=0.10).
- H2: Model rankings are metric-dependent: MCS membership differs materially across CRPS, QLIKE, and FZ loss.
- H3: TSFM relative performance degrades in crisis sub-samples versus calm periods (regime × model interaction).
- H4 (engineering): config-hash caching + process parallelism cut full-grid wall-clock ≥5× vs. naive sequential on 8 cores; TSFM GPU batching adds ≥3× on inference.

DATA (per D-004; licenses per 06_data_sources)
- 10 equity indices, Stooq daily OHLC: SPX, NDX, DJI, DAX, FTSE 100, CAC 40, Nikkei 225, HSI, TAIEX, KOSPI.
- Crypto true-RV arm: BTC-USD, ETH-USD, 1-min exchange bars → daily 5-min RV.
- Span: 2005-01 → freeze date (crypto from listing). Targets: daily variance.
- Proxy hierarchy: crypto = 5-min RV; indices = Parkinson + Garman–Klass range proxies, squared-return robustness check (Patton-robust losses only).
- Crisis sub-samples: GFC Sep 08–Mar 09 · COVID Feb–Apr 20 · 2022 tightening Jan–Oct 22 · Aug-2024 spike · latest 2025–26 stress window (fixed at grid freeze).

MODELS (13 configs; baselines first)
1 naive RW-vol · 2 EWMA λ=0.94 · 3 GARCH(1,1) · 4 GJR-GARCH · 5 HAR-RV
6 ARIMA on log-RV (statsforecast) · 7 ETS on log-RV
8 LightGBM on lagged features · 9 PatchTST (neuralforecast)
10 Chronos(-Bolt) · 11 TimesFM · 12 Moirai 2.0 · 13 TimeGPT [API-key flag, excluded from headline if access unstable]

PROTOCOL
- Rolling origin: window 1000 obs (~4 trading years), step 1 day, refit every 21 trading days (TSFMs zero-shot, context = trailing window).
- Horizons: h=1 primary; h=5, 22 secondary.
- Information rule: nothing later than t enters any forecast for t+1. No exceptions. Transforms fit per train window.
- Minimum evaluation length: ≥1500 forecasts per series.

METRICS (definitions frozen in 05_metrics_reference)
- Primary: CRPS, quantile loss @ {1%, 2.5%, 5%}, VaR/ES backtests (Kupiec, Christoffersen), FZ loss, QLIKE.
- Inference: DM pairwise (HLN correction), MCS at α=0.10 (block bootstrap).
- Economic value: volatility-targeting backtest, Sharpe net of 10 bps per rebalance.

OUT-OF-SCOPE for v1 (guard rail)
Return-direction forecasting; intraday horizons; options/IV data; TSFM fine-tuning; multivariate/covariance forecasting; additional asset classes; LLM/text-derived covariates (v2 candidate per TCSS comparison M3).
