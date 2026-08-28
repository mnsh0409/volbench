# P3 analysis — data validity checks

Four checks that can invalidate every table computed downstream, run over all
**645,151 rows** of the primary grid (143 cells = 11 assets x 13 configs x h=1
x arm `headline`).

**Reported, not interpreted.** No model is ranked, no column is called good or
bad, and nothing here says whether any finding matters. Anything mechanically
wrong is flagged as such.

Computed by `src/volbench/analysis.py`, which is structurally forbidden from
importing the model package (`tests/test_analysis.py::TestBoundary`) and
therefore cannot re-run anything it reads. Environment as pinned:
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`.

| check | verdict |
|---|---|
| 1. QLIKE positivity | **No forecast is ever non-positive.** 13 target days are exactly zero and are correctly dropped. 5 further days are within 1e-8 of zero and **are** scored |
| 2. Missing-row accounting | **Confirmed exactly**, including NKX's 4,794 / 4,773. No other asset has an unexpected gap |
| 3. Score finiteness | **Zero non-finite values that are not NaN**, in every loss column of every cell. Every NaN is accounted for |
| 4. Alignment canary | **Exact.** `target_index == origin + h` on all 645,151 rows; stored target reproduces the study's own series bit for bit; losses reproduce from independent closed forms |

---

## 1. QLIKE positivity

QLIKE is `r - log r - 1` with `r = proxy_var / forecast_var`. It needs both
strictly positive and diverges as either approaches zero.

### 1.1 Per asset

`proxy_min` is the smallest **strictly positive** realized target; `forecast_min`
is the smallest strictly positive variance forecast over all 13 models.

| asset | targets | proxy min (>0) | proxy = 0 | proxy < 0 | proxy NaN | forecast min | forecast <= 0 | forecast NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | 2,791 | 8.260e-06 | 0 | 0 | 0 | 2.596e-05 | 0 | 0 |
| CAC | 5,038 | **2.431e-10** | 0 | 0 | 28 | 8.289e-06 | 0 | 0 |
| DAX | 4,996 | 5.806e-08 | 0 | 0 | 0 | 9.058e-06 | 0 | 0 |
| DIA | 4,904 | 5.113e-07 | 0 | 0 | 0 | 6.584e-06 | 0 | 0 |
| ETH-USD | 2,791 | 1.306e-05 | 0 | 0 | 0 | 4.344e-05 | 0 | 0 |
| HSI | 4,828 | 1.179e-08 | **12** | 0 | 1 | 9.129e-06 | 0 | 0 |
| KOSPI | 4,837 | 2.070e-07 | 0 | 0 | 0 | 1.710e-05 | 0 | 0 |
| NDX | 4,942 | 1.572e-07 | 0 | 0 | 0 | 1.255e-05 | 0 | 0 |
| NKX | 4,795 | **6.716e-11** | **1** | 0 | 0 | 5.764e-06 | 0 | 168 |
| SPY | 4,904 | 1.124e-06 | 0 | 0 | 0 | 6.469e-06 | 0 | 0 |
| TWSE | 4,801 | **4.401e-11** | 0 | 0 | 80 | 6.491e-06 | 0 | 80 (see note) |

The proxy is a property of the asset, not of the model: the target column is
identical across each asset's 13 cells (`proxy_distinct_series = 1` for all
eleven, checked). NKX's 168 NaN forecasts are the `InsufficientHistoryError`
rows of §2 — 21 origins x 8 variance-fed configs. The TWSE figure in the last
column is NaN *targets*, not forecasts; no forecast is NaN there.

### 1.2 How zeros and floors are handled

**Realized targets are never floored.** `data/panel/build_targets` computes
`overnight_plus_range = (ln(O_t/C_{t-1}))^2 + RogersSatchell_t` (equities) or
5-minute realized variance (crypto) and writes what it gets, NaN included. No
clip, no epsilon.

**Forecasts have floors, and none of them binds.** `models/naive.py` floors
sigma at 1e-12 (variance 1e-24), `models/ewma.py` floors variance at 1e-24,
`models/tsfm_common.py` clips negative RV quantiles at zero and then *raises*
if the resulting mean is still non-positive. The smallest positive forecast
variance anywhere in the grid is **5.764e-06** (NKX / `autoarima`), nineteen
orders of magnitude above the naive/EWMA floors, and the smallest per model is:

| model | min forecast var | max forecast var | max `proxy/forecast` |
|---|---:|---:|---:|
| `autoarima` | 5.764e-06 | 5.672e-02 | 109.3 |
| `autoets` | 6.015e-06 | 5.536e-02 | 88.2 |
| `har` | 6.970e-06 | 5.653e-02 | 97.7 |
| `patchtst` | 7.824e-06 | 1.164e-02 | 185.8 |
| `chronos` | 8.237e-06 | 2.633e-02 | 119.9 |
| `moirai` | 8.492e-06 | 3.529e-02 | 76.1 |
| `ewma` | 8.776e-06 | 2.539e-02 | 83.1 |
| `timesfm` | 9.895e-06 | 4.796e-02 | 89.9 |
| `lgbm` | 9.917e-06 | 9.676e-03 | 147.0 |
| `garch11_t` | 1.297e-05 | 7.212e-02 | 64.0 |
| `gjr` | 1.563e-05 | 5.172e-02 | 93.8 |
| `garch11` | 1.612e-05 | 3.861e-02 | 64.2 |
| `naive` | 3.219e-05 | 3.778e-03 | 209.2 |

**Non-positive inputs produce NaN plus a named reason, never a floored score.**
`evaluate._score` writes `qlike = NaN` with `proxy_nonpositive` when
`proxy_var <= 0`, `proxy_nan` when it is NaN, and `forecast_var_nonpositive`
when the forecast is not finite and positive. The last of these never occurred
in this grid.

### 1.3 The thirteen zero-target days, and why they are not quiet days

Twelve on HSI and one on NKX. They are dropped (QLIKE NaN,
`proxy_nonpositive`), which is the correct handling, and `data/panel/diagnostics.csv`
records them independently as `zero_primary_days` — 12 for HSI, 2 for NKX (the
second is at trimmed position 203, inside the first fit window, so it is never
a target; it is the day that causes §2's `InsufficientHistoryError`).

They are **not** days with no price movement. Both terms of the primary target
vanish for a structural reason, on days with large close-to-close moves:

| asset | date | prev C | O | H | L | C | ln(C/C_-1) | target | Parkinson on the same bar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HSI | 2018-11-02 | 25416.00 | 25416.00 | 26486.35 | 25416.00 | 26486.35 | **+4.13%** | **0** | 6.137e-04 |
| HSI | 2023-08-25 | 18212.17 | 18212.17 | 18212.17 | 17956.38 | 17956.38 | **-1.41%** | **0** | 7.216e-05 |
| NKX | 2005-11-01 | 13606.50 | 13606.50 | 13867.86 | 13606.50 | 13867.86 | **+1.90%** | **0** | 1.306e-04 |

Rogers-Satchell is `ln(H/C)ln(H/O) + ln(L/C)ln(L/O)`, so it is **identically
zero on any monotone bar** — one where the open is the low and the close the
high, or the reverse — however large the move. The overnight term
`(ln(O_t/C_{t-1}))^2` is zero exactly when the open prints at the previous
close (a "stale open"). A monotone bar with a stale open therefore gives a
target of exactly 0.0 on a day the index moved several percent. HSI carries 490
stale-open days and 38 monotone bars in `diagnostics.csv`; the 12 zeros are
their intersection.

This is the documented D-016 revisit trigger reaching the panel, not a new
defect, and it is recorded here because the exact-zero days are the ones QLIKE
drops.

### 1.4 The five near-zero days, which are scored

The same geometry with the open *near* rather than *at* the previous close
leaves only the overnight term, and the target survives as a very small
positive number. Five such days are scored across three assets:

| asset | target_index | date | target | QLIKE range over the 13 models |
|---|---:|---|---:|---:|
| TWSE | 4,587 | 2023-09-14 | 4.401e-11 | 12.31 – 13.74 |
| NKX | 3,854 | 2020-10-01 | 6.716e-11 | 12.64 – 13.85 |
| TWSE | 3,676 | 2019-12-17 | 1.578e-10 | 10.72 – 12.02 |
| CAC | 2,432 | 2014-07-04 | 2.431e-10 | 10.54 – 11.98 |
| CAC | 2,403 | 2014-05-26 | 2.620e-09 | 8.61 – 9.67 |

The smallest ratio anywhere in the grid is `proxy/forecast = 3.54e-07`
(NKX/`naive`, 2020-10-01), giving QLIKE 13.85 — the largest such term. NKX
2020-10-01 is a bar whose close-to-close return is exactly 0.0 and whose four
prices span 19 index points.

**Size of the effect, stated as a number and nothing more.** Over the 65 rows
(5 days x 13 models), the maximum share of any cell's QLIKE *sum* taken by
these days is **1.01 %** (CAC/`gjr`), and the largest resulting shift in a
cell's QLIKE *mean* is **+0.98 %** (same cell). The days affect all 13 models
of an asset, since the target is shared.

For contrast, the largest single QLIKE term per asset comes from the opposite
direction — the proxy far above the forecast, `r` of 82 to 209 — not from the
near-zero targets.

## 2. Missing-row accounting

`scored` is the strict reading: rows whose `missing_reason` is empty. It is not
the same as "has a CRPS": a row that lost only QLIKE keeps its CRPS. All three
counts are given because they differ.

| asset | fed by | cells | origins | scored/cell | missing/cell | CRPS scored | QLIKE scored |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | return / variance | 5 / 8 | 2,791 | 2,791 | 0 | 2,791 | 2,791 |
| CAC | return / variance | 5 / 8 | 5,038 | 5,010 | 28 | 5,038 | 5,010 |
| DAX | return / variance | 5 / 8 | 4,996 | 4,996 | 0 | 4,996 | 4,996 |
| DIA | return / variance | 5 / 8 | 4,904 | 4,904 | 0 | 4,904 | 4,904 |
| ETH-USD | return / variance | 5 / 8 | 2,791 | 2,791 | 0 | 2,791 | 2,791 |
| HSI | return / variance | 5 / 8 | 4,828 | 4,815 | 13 | 4,828 | 4,815 |
| KOSPI | return / variance | 5 / 8 | 4,837 | 4,837 | 0 | 4,837 | 4,837 |
| NDX | return / variance | 5 / 8 | 4,942 | 4,942 | 0 | 4,942 | 4,942 |
| **NKX** | **return** | **5** | 4,795 | **4,794** | 1 | 4,795 | 4,794 |
| **NKX** | **variance** | **8** | 4,795 | **4,773** | 22 | 4,774 | 4,773 |
| SPY | return / variance | 5 / 8 | 4,904 | 4,904 | 0 | 4,904 | 4,904 |
| TWSE | return / variance | 5 / 8 | 4,801 | 4,721 | 80 | 4,801 | 4,721 |

**Totals: 645,151 rows, 643,397 scored (99.728 %), 1,754 missing (0.272 %).**
CRPS/log-score/pinball are scored on 644,983 rows; QLIKE on 643,397.

**The expectation is confirmed exactly.** NKX's five return-fed configs score
**4,794** and its eight variance-fed configs score **4,773** — a difference of
21, the first refit block, lost on the variance-fed configs only.

**No other asset has an unexpected gap.** Within each (asset, feed) group all
cells have byte-identical scored and missing counts — checked, `min == max` for
every one of the 22 groups. NKX is the only asset where the return-fed and
variance-fed groups differ at all.

### 2.1 Every missing row, by cause

Only three distinct `missing_reason` strings exist in the whole grid.

| kind | rows | where | per cell |
|---|---:|---|---:|
| `proxy_nan` | 1,417 | CAC (all 13 cells), TWSE (all 13), HSI (all 13) | 28 / 80 / 1 |
| `proxy_nonpositive` | 169 | HSI (all 13), NKX (all 13) | 12 / 1 |
| `fit_error/InsufficientHistoryError` | 168 | NKX, the 8 variance-fed cells only | 21 |

`1,417 + 169 + 168 = 1,754`. No row carries more than one kind; no
`target_nan`, `forecast_var_nonpositive`, `log_score_undefined`, `es_undefined`,
`predict_error`, `update_error` or `score_error` appears anywhere.

The `fit_error` rows carry their own explanation in full:

> `fit_error@499: InsufficientHistoryError: only 499 valid observations at or
> before origin 499, but the protocol asks for a 500-observation fit window
> (invalid-target policy 'compact'; 1 of the first 500 days are invalid target
> days). Fitting on a short window would report a window length the run did not
> use`

### 2.2 Reconciled against the panel diagnostics

Every count above is independently explained by `data/panel/diagnostics.csv`,
which was written by the data layer before any model ran:

| asset | store: NaN targets | store: zero targets | diagnostics `nan_overnight_plus_range` | diagnostics `zero_primary_days` |
|---|---:|---:|---:|---:|
| CAC | 28 | 0 | 28 (= its 28 `inconsistent_bars`) | 0 |
| TWSE | 80 | 0 | 80 (= its 80 `inconsistent_bars`) | 0 |
| HSI | 1 | 12 | 1 | 12 |
| NKX | 0 | 1 | 0 | **2** |
| SPY, DIA, BTC-USD, ETH-USD | 0 | 0 | **1** each | 0 |
| NDX, DAX, KOSPI | 0 | 0 | 0 | 0 |

The two apparent discrepancies both resolve to positions the store never
scores, and both were checked rather than assumed:

- **NKX 2 vs 1.** The zero-target days are at trimmed positions 203 and 3,854.
  Targets begin at position 500, so 203 is never a target — it sits *inside*
  the first fit window, and it is exactly the invalid day that makes origin 499
  one observation short under D-018 compaction. One data defect, two visible
  consequences, no third.
- **SPY / DIA / BTC-USD / ETH-USD, 1 vs 0.** Their archives begin at the panel
  window, so their first bar has no `C_{t-1}` and its overnight term is NaN.
  The driver's leading trim removes exactly that bar. Verified directly: after
  the trim, **no asset has a NaN or zero target at any position below 500**,
  and every NaN or zero at position >= 500 appears in the table above.

## 3. Score finiteness

Every loss column, every asset x model cell, 645,151 rows x 6 columns:

| column | values | NaN | +inf | -inf | non-finite that is not NaN |
|---|---:|---:|---:|---:|---:|
| `crps` | 645,151 | 168 | 0 | 0 | **0** |
| `log_score` | 645,151 | 168 | 0 | 0 | **0** |
| `qlike` | 645,151 | 1,754 | 0 | 0 | **0** |
| `pinball_0p01` | 645,151 | 168 | 0 | 0 | **0** |
| `pinball_0p025` | 645,151 | 168 | 0 | 0 | **0** |
| `pinball_0p05` | 645,151 | 168 | 0 | 0 | **0** |

**No infinity anywhere, in any column, in any cell.** There are no offending
rows to list.

NaN is separated from infinity deliberately. A NaN loss is the contract's own
way of saying "unscorable, and here is why", and every one of them is one of
§2's 1,754 rows: the 168 NaN in the return-side losses are exactly the
`InsufficientHistoryError` rows, and QLIKE's 1,754 are the full set. An
infinity would be a defect, because nothing in `evaluate._score` can produce
one — `qlike()` raises rather than returning `inf`, and the closed-form CRPS
and log score are finite for every finite input at a positive variance.

The forecast-description columns were checked too, at all three levels: 644,983
rows carry a finite `(var, es)` pair and 168 carry neither; **no row** has
`es >= 0`, `es > var`, or `var >= 0`. FZ0 is therefore computable wherever a
forecast exists (docs/P3_ANALYSIS_ASSUMPTIONS.md §2).

## 4. Alignment canary

The most expensive possible error here is an off-by-one between forecast and
realization. Two independent things are checked, because they catch different
failures — and a loss recomputed from a row's own stored target **cannot** catch
a misalignment, since a loss computed against the wrong day's realization is
still perfectly self-consistent.

### 4.1 Index arithmetic and the stored target, over the whole grid

| check | rows | result |
|---|---:|---|
| `target_index == origin_index + horizon` | 645,151 | **645,151 / 645,151** |
| `realized_return` equals the study's own return series at `target_index` | 645,151 | max abs error **0.000e+00** |
| `proxy_var` equals the study's own target series at `target_index` | 645,151 | max abs error **0.000e+00** |
| `target_index` outside the series | 645,151 | 0 |

The comparison series were rebuilt through the driver's own bridge
(`build_panel()` -> `log_returns(close)` -> first-valid-index trim), so the
known hazard is handled at its source: a results-frame `target_index` of `j` is
position `j + 1` of the untrimmed OHLC frame, and it is position `j` of the
trimmed series the backtest was run on. That arithmetic has already produced
one false leak report (docs/P3_GRID.md §6).

**Negative control.** The same comparison against the series shifted by one day
makes **all 645,151 rows differ**, max abs error 8.08e-01. The check can fail,
so its passing is evidence.

### 4.2 Losses recomputed from independent closed forms

Every loss was recomputed from the row's recovered predictive law and its
stored target, using closed forms written out in `volbench.analysis` from
Gneiting & Raftery (2007) eq. 21, Jordan, Krüger & Lerch (2019) App. A, and
docs/metrics_reference.md — never by calling `volbench.dist` or
`volbench.metrics`, which produced the numbers under test. The analysis module
is forbidden from importing either.

| recomputation | rows | one-sided NaN | max abs error | max rel error |
|---|---:|---:|---:|---:|
| QLIKE, all 13 configs | 645,151 | 0 | **0.000e+00** | 0 |
| CRPS, the 12 Gaussian configs | 595,524 | 0 | 8.33e-17 | 1.11e-15 |
| CRPS, `garch11_t` (Student-t, `nu` recovered) | 49,627 | 0 | 1.13e-15 | — |
| stored `var_0p01` vs the Normal quantile | 595,524 | — | **0.000e+00** | — |
| stored `var_0p025` vs the Normal quantile | 595,524 | — | 1.67e-16 | — |
| stored `var_0p05` vs the Normal quantile | 595,524 | — | 5.55e-17 | — |

QLIKE reproduces to the **bit** on every one of the 643,397 rows where it is
defined, and is NaN on exactly the same 1,754 where the store says NaN. CRPS
agrees to floating-point roundoff. A NaN on one side only would have counted as
a disagreement, not a match (`analysis._abs_error` returns infinity there);
there were none.

`garch11_t`'s law is recovered per scheduled fit from the ratio of two stored
tail quantiles and then required to reproduce the row's own `forecast_var`
before it is accepted. All 2,367 fits recover: 2,349 Student-t with `nu` in
[2.264, 50.000] (61 at D-032's upper bound of 50), and 18 Gaussian — exactly
the 18 that `fit_status` labels `fallback=ewma`. The recovered `nu` is constant
across every origin of all 2,349 blocks, which is independent confirmation from
the stored artifacts alone that `update` re-conditions without re-estimating.

### 4.3 One row per asset, worked by hand

Rows drawn at random (seed 20260828) from each asset's scored rows.

| asset | model | origin | target | date | law | `nu` | stored CRPS | recomputed | err | stored QLIKE | recomputed | err |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | `moirai` | 1164 | 1165 | 2020-10-26 | Normal | — | 0.00568413 | 0.00568413 | 0 | 0.00106905 | 0.00106905 | 0 |
| CAC | `naive` | 5326 | 5327 | 2025-10-23 | Normal | — | 0.00234816 | 0.00234816 | 0 | 0.602876 | 0.602876 | 0 |
| DAX | `har` | 2892 | 2893 | 2016-05-23 | Normal | — | 0.00462336 | 0.00462336 | 0 | 0.00409902 | 0.00409902 | 0 |
| DIA | `garch11_t` | 793 | 794 | 2008-04-24 | Student-t | 5.166 | 0.00403712 | 0.00403712 | 6.9e-18 | 0.0451295 | 0.0451295 | 0 |
| ETH-USD | `garch11_t` | 3026 | 3027 | 2025-12-01 | Student-t | 3.455 | 0.0501685 | 0.0501685 | 0 | 0.077596 | 0.077596 | 0 |
| HSI | `gjr` | 1141 | 1142 | 2009-08-24 | Normal | — | 0.00989911 | 0.00989911 | 0 | 0.0026235 | 0.0026235 | 0 |
| KOSPI | `patchtst` | 2508 | 2509 | 2015-02-11 | Normal | — | 0.00302586 | 0.00302586 | 0 | 0.0160258 | 0.0160258 | 0 |
| NDX | `autoets` | 4288 | 4289 | 2022-01-18 | Normal | — | 0.0190805 | 0.0190805 | 0 | 0.309774 | 0.309774 | 0 |
| NKX | `chronos` | 2354 | 2355 | 2014-08-08 | Normal | — | 0.0266802 | 0.0266802 | 0 | 1.83129 | 1.83129 | 0 |
| SPY | `patchtst` | 3228 | 3229 | 2017-12-26 | Normal | — | 0.00113024 | 0.00113024 | 0 | 0.613573 | 0.613573 | 0 |
| TWSE | `patchtst` | 2697 | 2698 | 2015-12-09 | Normal | — | 0.0136994 | 0.0136994 | 0 | 0.0391121 | 0.0391121 | 0 |

For all eleven: `target_index == origin_index + 1`, and both the stored
`realized_return` and the stored `proxy_var` match the study's own series at
that position exactly.

---

## Nothing mechanically wrong was found

No NaN where none should be. No infinity anywhere. No non-positive forecast. No
score that disagrees with its own inputs. No misalignment. No asset with an
unexplained gap. Every one of the 1,754 missing rows is named, counted and
reconciled against a diagnostics file written before any model ran.

Three facts are recorded because a later table could be read wrongly without
them, and none of them is a defect:

1. **13 target days are exactly zero** and lose their QLIKE. They are monotone
   bars with stale opens, not quiet days; the largest carries a 4.1 %
   close-to-close return (§1.3).
2. **5 further days are within 1e-8 of zero and are scored.** They contribute
   QLIKE terms of 8.6 to 13.9 and take up to 1.01 % of a cell's QLIKE sum
   (§1.4).
3. **"Scored" has three different values** — 643,397 strictly, 644,983 for the
   return-side losses, 643,397 for QLIKE — and a table quoting one of them as
   all three would misstate up to 1,586 rows (§2).
