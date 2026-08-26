# Primary grid run — 11 assets x 13 configs, on volbench 0.6.0

**Run health only.** This document reports what the run *did*, not what it
found. No score is read, no model is ranked, no hypothesis is addressed here;
the results review happens on the planning machine, off the stored fragments.

| | |
|---|---|
| Commit | `d964e9f791785290a1a407e8c7a5ab6b642091c1` (`main`) |
| Package version | 0.6.0 (`v0.6.0-determinism` is an ancestor) |
| Date | 2026-08-26 |
| Machine | i9-13900KF (32 threads), RTX 4090 24 GB, driver 535.309.01 |
| Store | `data/grid_primary/store/` — 143 parquet fragments + 143 JSON sidecars, 100 MB |
| Manifest | `data/grid_primary/manifest_primary.json` (64 KB) |
| Driver | `data/grid_primary/run_grid.py` (gitignored with the results) |

---

## 1. Protocol as run

Horizon 1, window 500 (D-019), step 1, `refit_every=21` with daily
re-conditioning (D-015), invalid-target policy at its D-018 default. One
protocol arm, labelled `headline`. Grid seed 20260825.

**Scoring target is each asset's own primary** (D-017 — a property of the
cell, never of the model): `overnight_plus_range` on the nine equity series
(D-016), `realized_variance` on BTC-USD and ETH-USD (D-004). The instruction
for this run named `overnight_plus_range` throughout; it was applied to the
equity series only, because `build_crypto_series` states in terms that on a
24/7 market that estimator's "overnight" term is a one-minute gap and "is not
the target anything is scored against here". Scoring crypto on it would have
contradicted D-004. Confirmed before the run.

**Thirteen configs, not twelve.** `docs/research_design.md` lists thirteen and
excludes TimeGPT from the headline, which leaves twelve; but the twelve it
leaves include GJR-GARCH, while the implemented-and-exercised set
(`benchmarks/toy.py` + `benchmarks/smoke_tsfm.py`) carries the Student-t GARCH
of D-014/D-032 in that slot instead. Both were run rather than choosing
between them. TimeGPT is out: an API model behind a key, excluded from the
headline by the design where access is unstable.

| lane | configs |
|---|---|
| cpu (9) | `naive` `ewma` `garch11` `garch11_t` `gjr` `har` `autoets` `autoarima` `lgbm` |
| gpu (4) | `chronos` `timesfm` `moirai` `patchtst` |

Lanes are declared per config and never inferred from a label (D-027). CPU
lane on `ProcessExecutor(workers=12, start_method='forkserver')`; GPU lane on
`ProcessExecutor(workers=1)`, one cell at a time on the one card.

## 2. Determinism pins, as they actually stood

Recorded on the manifest under `environment`:

```
blas_threads          1
thread_pin_explicit   true
kernel_signature      4c64e2eacc61926e
cpu_count             32
NPY_DISABLE_CPU_FEATURES  "X86_V4 AVX512_ICL AVX512_SPR"
OMP_NUM_THREADS           "1"
OPENBLAS_NUM_THREADS      "1"
BLAS                  scipy-openblas 0.3.31.188.0
```

`blas_threads = 1` is in every one of the 143 config hashes, checked
per fragment (§5).

Two facts about this machine, measured rather than assumed:

- **The D-032 thread pin binds here.** Unpinned, `thread_pin()` on this box
  resolves to **32**. Every GARCH number in this store is a property of the
  pinned value, and an unpinned re-run would miss the cache rather than be
  served these fragments — which is the mechanism D-032 exists to provide.
- **The D-026 kernel pin is a no-op here.** The i9-13900KF has no AVX-512
  (`AVX512F = False`); numpy reports `baseline=X86_V2, dispatch=[X86_V3]` and
  the kernel signature is **identical pinned and unpinned**. This grid was
  computed natively in the x86-v3 family the pin names. The pin is still set,
  because it is what makes that a fact rather than a coincidence of hardware.

The driver refuses to start unless both pins are in force: neither can be
repaired after numpy is imported, so a run that discovered the problem later
would already have been wrong.

## 3. Run health

```
cells attempted 143   computed 130   cached 13   failed 0
wall clock 69.6 min    peak RSS 1.01 GiB
rows 645,151   missing 1,754 (0.27%)   scored 643,397
```

The 13 cached cells are the SPY column, computed by the pre-flight into the
same store and resumed as cache hits — the resumability guarantee doing its
job on its first real occasion rather than as a test.

**No cell failed. No exception type was raised by any cell.** The `error`
field is null on all 143 manifest rows.

### 3.1 Wall-clock

Sum of per-cell wall-clock is 80.6 min against 69.6 min elapsed; the
difference is the CPU lane's twelvefold overlap.

| lane | cells | cell-seconds | elapsed |
|---|---:|---:|---:|
| cpu | 99 | 12.4 min | **82 s** (14:28:13 → 14:29:35) |
| gpu | 44 | 68.2 min | **68.2 min** (serialized by design) |

Per config, summed over its 11 assets:

| config | lane | wall (min) | | config | lane | wall (min) |
|---|---|---:|---|---|---|---:|
| `patchtst` | gpu | 27.12 | | `lgbm` | cpu | 0.70 |
| `timesfm` | gpu | 26.79 | | `autoets` | cpu | 0.49 |
| `chronos` | gpu | 8.55 | | `har` | cpu | 0.24 |
| `autoarima` | cpu | 7.07 | | `ewma` | cpu | 0.15 |
| `moirai` | gpu | 5.71 | | `naive` | cpu | 0.13 |
| `garch11_t` | cpu | 1.67 | | | | |
| `gjr` | cpu | 1.03 | | | | |
| `garch11` | cpu | 0.96 | | | | |

The GPU lane is 98% of the elapsed time and `patchtst` + `timesfm` are 79% of
the GPU lane. Slowest single cell: `CAC/patchtst`, 3.5 min.

Peak RSS 1.01 GiB across the parent and every worker it reaped — the same
figure the single-asset pre-flight reported, so the twelvefold CPU fan-out and
the eleven-asset panel cost essentially nothing in memory.

**Projection accuracy.** The pre-flight projected ~80 min total and ~72 min
remaining after the cached SPY column; the run took 69.6 min. The projection
scaled by 49,627 / 4,904 = **10.12x**, not 11x, because BTC-USD and ETH-USD
list in 2017 and contribute 2,791 origins each against an equity series' ~4,900.

### 3.2 Fallback and non-convergence

Denominator is scheduled fits that reported a status at all; only the three
GARCH-family configs do. **38 of 7,101 fits (0.54%)** ran the EWMA fallback,
and every one of them was a non-convergence — the two counts coincide exactly,
which is what D-032 says to expect, a fallback being what non-convergence
causes here.

**26 of the 33 GARCH-family cells are clean** (zero fallback, zero
non-convergence). The seven that are not:

| asset | config | fits | fallback | rate | non-conv |
|---|---|---:|---:|---:|---:|
| BTC-USD | `garch11_t` | 133 | 15 | 11.28% | 15 |
| HSI | `gjr` | 230 | 14 | 6.09% | 14 |
| DIA | `garch11` | 234 | 3 | 1.28% | 3 |
| DIA | `garch11_t` | 234 | 3 | 1.28% | 3 |
| DIA | `gjr` | 234 | 1 | 0.43% | 1 |
| ETH-USD | `gjr` | 133 | 1 | 0.75% | 1 |
| SPY | `garch11` | 234 | 1 | 0.43% | 1 |

Every other config reports `n_fits = 0` — it estimates nothing that can fail
to converge — and its fallback rate is therefore `nan`, not `0`, on the
manifest. "No fit fell back" and "no fit said" are different claims.

Reported, not interpreted: `BTC-USD/garch11_t` at 11.28% means that cell is
about one-ninth an EWMA cell, and that is now readable from the manifest
without opening a parquet, which is what D-032 shipped `fit_status` and
`n_fits_fallback` for. Whether it should change anything is a question for the
results review.

### 3.3 `missing_reason` rows

1,754 of 645,151 rows (0.27%) carry a `missing_reason`. Every one is
accounted for, and the counts are identical across the 13 configs of an asset
except where noted:

| asset | rows/cell | per cell | cause |
|---|---:|---|---|
| TWSE | 4,801 | `proxy_nan` x 80 | closes printed outside their own session range; range targets NaN'd (D-018) |
| CAC | 5,038 | `proxy_nan` x 28 | same |
| HSI | 4,828 | `proxy_nan` x 1, `proxy_nonpositive` x 12 | one NaN'd bar; 12 monotone-bar/stale-open days where the overnight term and Rogers-Satchell are both exactly zero |
| NKX | 4,795 | `proxy_nonpositive` x 1 (+ 21, see below) | one zero-target day |
| NDX DAX KOSPI SPY DIA BTC ETH | — | none | — |

These are D-018(a) rows: the day keeps its place on the calendar, the scores
are NaN, and the cause is named. They are scoring failures, not model
failures, and they match `data/panel/diagnostics.csv` exactly.

**NKX carries 21 extra rows, on the eight variance-fed configs only:**

```
NKX/har       fit_error/InsufficientHistoryError=21, proxy_nonpositive=1
NKX/naive     proxy_nonpositive=1
```

NKX has an invalid target day *inside* its first 500-observation window.
Compaction drops it, so origin 499 has 499 valid observations behind it where
the protocol asks for 500, and `InsufficientHistoryError` is raised rather
than a 499-observation window being silently fitted. A failed scheduled fit
fails every origin in its block and the cadence is never adjusted to work
around it, so the loss is exactly one refit block — **21 of 4,795 origins,
0.44%**, recovered at the next refit. The five return-fed configs are not
handed the compacted series and see only the one `proxy_nonpositive` row.

This is the behaviour D-018 specifies ("possible only at a series' start")
reaching the real panel for the first time. Nothing was changed for it.

## 4. Nothing was fixed silently

No cell failed, so no fix was needed and none was applied. The grid ran on
`d964e9f` exactly as committed; the only code written for this run is the
driver, which is gitignored with the results it produced.

## 5. Gate

| Gate | Result | How checked |
|---|---|---|
| Manifest complete | **PASS** | 143 rows = 11 assets x 13 configs, each pair present exactly once |
| No cell in an unknown state | **PASS** | every status in `{computed, cached}`; every non-failed cell carries a `config_hash`; no `failed` rows at all |
| Fragments readable | **PASS** | all 143 parquet fragments and JSON sidecars re-read; row counts match the manifest; each sidecar's `data.asset` matches its cell |
| Fragments hash-consistent | **PASS** | each stored config re-hashed with `results.config_hash` reproduces the filename it is stored under; 143 distinct hashes, no two cells sharing one |
| `blas_threads` in every hash | **PASS** | `environment.blas_threads == 1` in all 143 sidecars |
| Resumable | **PASS** | a full re-run reports `attempted 143, computed 0, cached 143, failed 0`; all 143 fragments byte-identical **and unrewritten** (mtimes unchanged) |
| Repo clean | **PASS** | only this document and the manifest are committed; `/data/` is gitignored |

## 6. Leakage audit of the new surface

The only new code is the driver's panel-to-`AssetData` bridge. Audited against
`.claude/skills/leakage-check`: **PASS, no findings.** The splitter monopoly
holds (the driver contains no `.iloc` window or date filter), the leading trim
is applied identically to returns, proxy and fit series and removes leading
rows only, `overnight_variance` is `(ln(O_t / C_{t-1}))^2` and strictly
backward-looking, and no cross-asset join exists — the eleven calendars never
meet.

The demanded canary was run against the driver's own bridge rather than a
stand-in: corrupt SPY's raw OHLC strictly after target row 560, rebuild
`PanelSeries` -> `asset_data` -> `run_backtest`, and require every row at or
before the cutoff to be bit-identical.

```
SPY: 5405 bars, cutoff index 560, 61 rows at or before it
  naive     future-corruption: identical   past-corruption: differs (canary alive)
  ewma      future-corruption: identical   past-corruption: differs (canary alive)
  garch11   future-corruption: identical   past-corruption: differs (canary alive)
  har       future-corruption: identical   past-corruption: differs (canary alive)
VERDICT: PASS
```

One fact worth recording for whoever reads the panel next, because the first
attempt at this canary got it wrong and reported a false leak: the driver
trims one leading bar, so a results-frame `target_index` of `j` is position
`j + 1` of the raw OHLC frame. The canary is sensitive enough that a one-row
error in that arithmetic fails it.

## 7. Drift flagged, not edited

`docs/research_design.md` still states "MODELS (13 configs)" with GJR in the
list and no Student-t GARCH, and "window 1000 obs" in the protocol line, which
D-019 superseded with 500. Both are mirrors here and were not edited. The
model-list question was settled for this run by running both (§1); the window
question is already settled by D-019 and only the mirror lags.

## 8. What this run does not say

No number in the store has been read. Model rankings, MCS membership, DM
tests, crisis sub-samples and economic value are all downstream of this and
are not addressed here. The store and this manifest are the input to that
review, not a summary of it.
