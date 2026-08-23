---
name: leakage-check
description: Audit volbench code, designs, or experiment configs for temporal leakage / look-ahead bias. Use for any change touching data loading, splitting, feature construction, scaling, refitting, caching, or evaluation — and whenever a result looks suspiciously good.
---

# Temporal leakage audit

Work through every item; report a table of findings with verdict PASS / FIX (with the fix) / FATAL (why it invalidates results). Quote the exact lines implicated.

1. **Index arithmetic at boundaries.** Off-by-one at train/test joins: does the last train observation strictly precede the first forecast target's information cutoff? Check inclusive/exclusive slicing on both ends.
2. **Splitter monopoly.** All train/test indices must come from `RollingOriginSplitter`. Any hand-rolled slice, `.iloc` window, or ad-hoc date filter in model/eval code is a violation.
3. **Feature lags.** Every feature at time t must be computable from data ≤ t. Watch rolling means ending at t (must end at t, using ≤ t values), calendar features (fine), and anything derived from full-series statistics.
4. **Transforms and scalers.** Normalizers, Box-Cox parameters, winsorization thresholds fit on the train window only, refit per origin per the refit schedule — never on the full series.
5. **Target construction.** RV_t built only from day-t intraday data; log/variance transforms applied consistently; no forward-filled targets.
6. **Refit schedule.** Model parameters used for the forecast at t estimated only from data available at the scheduled refit date ≤ t.
7. **TSFM context windows.** The context passed to a foundation model for a forecast at t contains nothing after t; check batching code that pads or aligns across series with different calendars.
8. **Calendar alignment.** Cross-asset features/joins on asynchronous calendars (US vs. EU close, crypto 24/7): a same-timestamp join can smuggle future information across timezones.
9. **Caching.** Cache keys include the information cutoff; a cached artifact computed with later data must never serve an earlier origin.
10. **Survivorship & selection.** Series chosen using information unavailable at forecast time (e.g., "assets that survived to 2026") — flag as design-level leakage.

**Canary test to demand:** corrupt all data strictly after date T with noise; every forecast for targets ≤ T must be bit-identical. If not, there is leakage — find it before anything ships.
