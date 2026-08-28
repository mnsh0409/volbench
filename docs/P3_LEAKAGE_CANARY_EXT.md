# P3 — the leakage canary, extended from four configs to thirteen

**What was asked.** `docs/P3_GRID.md` §6 ran the leakage canary over `naive`,
`ewma`, `garch11` and `har` only. The nine it did not cover include every model
where leakage is hardest to argue from source alone. The instruction was to
extend the same canary — through the driver's own bridge, not a stand-in — to at
least `patchtst`, `lgbm`, `chronos`, `autoarima` and `garch11_t`.

**What was done.** Those five, and then **all thirteen**, since covering the
remaining four (`gjr`, `autoets`, `timesfm`, `moirai`) cost one more run of the
same harness and leaves no config in the primary grid uncanaried.

**Reported, not interpreted.** This says what the canary did and what it
returned. It does not say anything about any model's forecasts.

| | |
|---|---|
| Driver | `src/volbench/benchmarks/leakage_canary.py` (committed — docs/P3_DRIVER_PROVENANCE.md §6) |
| Series | SPY, 5,405 windowed bars, 5,404 after the driver's leading trim |
| Cutoff | results-frame `target_index` **560** = windowed position 561 = **2007-05-21** |
| Rows compared | **61** — `target_index` in [500, 560] — the same 61 P3_GRID §6 compared |
| Corrupted | future leg **4,843** raw bars (2007-05-22 onwards); past leg **22** (2007-04-20 .. 2007-05-21) |
| Environment | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` |

---

## 1. The verdicts

```
SPY: 5405 bars, cutoff index 560, 61 rows at or before it
  naive      future-corruption: identical   past-corruption: differs (canary alive)
  ewma       future-corruption: identical   past-corruption: differs (canary alive)
  garch11    future-corruption: identical   past-corruption: differs (canary alive)
  garch11_t  future-corruption: identical   past-corruption: differs (canary alive)
  gjr        future-corruption: identical   past-corruption: differs (canary alive)
  har        future-corruption: identical   past-corruption: differs (canary alive)
  autoets    future-corruption: identical   past-corruption: differs (canary alive)
  autoarima  future-corruption: identical   past-corruption: differs (canary alive)
  lgbm       future-corruption: identical   past-corruption: differs (canary alive)
  chronos    future-corruption: identical   past-corruption: differs (canary alive)
  timesfm    future-corruption: identical   past-corruption: differs (canary alive)
  moirai     future-corruption: identical   past-corruption: differs (canary alive)
  patchtst   future-corruption: identical   past-corruption: differs (canary alive)
VERDICT: PASS
```

**No model reported "past-corruption: identical".** The canary is live on every
one of the thirteen, so every "identical" above is a test that could have
failed and did not.

Supporting counts, uniform across all thirteen unless noted:

| config | adapter | determinism (clean vs clean) | rows compared | past leg: columns moved | past leg: rows moved | after cutoff: rows moved |
|---|---|---|---:|---:|---:|---:|
| `naive` | `naive_rw_vol` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `ewma` | `ewma` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `garch11` | `garch(1,1)-normal` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `garch11_t` | `garch(1,1)-studentst` | bit-identical | 61 | **19** | 22 | 4,843 / 4,843 |
| `gjr` | `gjr_garch(1,1)-normal` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `har` | `har_rv-smearing` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `autoets` | `autoets_rv-smearing` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `autoarima` | `autoarima_rv-smearing` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `lgbm` | `lightgbm_rv-smearing` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `chronos` | `chronos_bolt_small` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `timesfm` | `timesfm_2_5_200m_pytorch` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `moirai` | `moirai_2_0_r_small` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |
| `patchtst` | `patchtst` | bit-identical | 61 | 18 | 22 | 4,843 / 4,843 |

`garch11_t`'s extra column is `fit_status`: corrupting its training window
changed its optimizer's outcome, so the past leg moves the diagnostic as well
as the forecast. It is the only config with a `fit_status` that is not constant
under a changed window, because it is one of the three that reports one at all
(docs/P3_INSTRUMENTATION_GAP.md).

**No model could not be canaried.** All thirteen ran through the bridge with no
structural obstacle; nothing was skipped.

## 2. What the harness does, and why each leg is needed

### 2.1 Three legs, not one

A canary that reports "identical" proves nothing unless it can report
"differs", and it cannot report either unless the run is deterministic to begin
with.

1. **Determinism** — the same clean inputs, twice, into **two different
   stores** so neither read is a cache hit. Bit-identity here is the null
   hypothesis of legs 2 and 3. This is the leg that matters for `patchtst`,
   which trains a network per origin with Adam, dropout and early stopping in
   the path.
2. **Future corruption** — raw OHLC strictly after the cutoff is corrupted;
   every row at or before it must be bit-identical to the clean run.
3. **Past corruption** — raw OHLC *at or before* the cutoff is corrupted; the
   same rows must now **differ**. A model reporting "past-corruption:
   identical" would mean the canary is inert there — a different and worse
   finding than a leak, because it invalidates leg 2's verdict.

### 2.2 It goes in at the raw CSV

The corruption is written into a copy of SPY's Stooq archive, so the entire
production path runs on it:

```
ingest_manual_csv -> repair_bars -> build_targets -> PanelSeries
  -> benchmarks.grid_primary.asset_data   (the driver's own bridge)
    -> runner.run_grid -> evaluate.run_backtest
```

Nothing is a stand-in for a stage, and the model configs are taken from
`grid_primary.model_configs()` rather than re-declared — a canary run against a
re-declaration would prove something about the re-declaration.

### 2.3 The bars stay valid, and the perturbation is not a rescaling

Each selected bar's four prices are perturbed independently by 5 % Gaussian
noise, and the high and low are then reset to the max and min of the four, so
the corrupted bar is still a valid OHLC bar and its targets are still
computable.

Scaling a bar by a single factor would have been useless: Rogers-Satchell and
Parkinson are functions of within-bar log *ratios* and are scale-free, so a
uniform rescale leaves the range term untouched and corrupts only the overnight
term. Measured on the resulting archive: 4,843 of 5,405 rows changed, median
|relative close change| **3.35 %**, max **23.2 %**.

### 2.4 The index arithmetic, stated once

The driver trims one leading bar (`log_returns` has no `C_{t-1}` on the first
one), so a results-frame `target_index` of `j` is position `j + 1` of the
windowed OHLC frame. The cutoff is converted **once**, in `run_canary`, and
selection into the CSV is then by **calendar date**, never by position — the
raw file, the windowed frame and the results frame have three different
origins, and a positional rule would be a fourth chance to make the off-by-one
this canary exists to detect. That arithmetic has already produced one false
leak report (P3_GRID §6).

Confirmed on the written archives: the last **unchanged** row of the future leg
is dated 2007-05-21 and the first changed row 2007-05-22; the past leg changed
exactly the 22 rows 2007-04-20 .. 2007-05-21.

### 2.5 What is compared

All **31** of the fragment's 32 columns, with NaN equal to NaN. Only
`config_hash` is excluded, because it is a digest of the whole input series and
therefore *must* differ between a clean and a corrupted run — that is the cache
refusing to serve one run's fragment for another (D-011), not a forecast
changing.

So the comparison covers not only the six losses but every description of the
forecast (`forecast_mean`, `forecast_var`, `var_*`, `es_*`, `hit_*`), the
target as the run saw it (`realized_return`, `proxy_var`), the protocol trace
(`fit_origin`, `conditioned_through`, `refit`, `fit_status`) and the
accounting (`missing_reason`). A leak that moved a variance forecast without
moving a loss, or that changed which origin a block rested on, would be caught.

## 3. Four facts that make the verdicts readable

### 3.1 The clean run reproduces the primary grid byte for byte

The full-length clean leg computes SPY at the study's own protocol, so its
config hash **equals** the primary grid's SPY hash for that config — and its
parquet fragment is byte-identical to the one in `data/grid_primary/store/`.
Checked for all thirteen: 13/13 on both the hash and the bytes.

This is worth more than it looks. It says the canary harness is not an
approximation of the grid: it recomputed exactly what the grid published, with
all of this phase's new code in the tree. It is also an incidental
reproducibility check of 13 of the grid's 143 cells.

### 3.2 The corruption actually reached the models

If a corrupted run produced identical output everywhere, "identical before the
cutoff" would be vacuous. Every row **after** the cutoff differs — 4,843 of
4,843, for all thirteen configs. The config hash moves too, on both corrupted
legs, which is the content digest doing its job: a corrupted run can never be
served a clean run's fragment.

### 3.3 The past leg moves exactly the rows it should

The past corruption changed raw bars at results positions 539–560. The rows
whose scores move are `target_index` **539 .. 560**, contiguous, 22 of the 61
compared — and nothing earlier. So the past leg is not "everything changed
because the whole series moved"; it is a bounded disturbance landing precisely
where it was put.

18 columns move (19 for `garch11_t`, which also moves `fit_status`: the
corrupted window changed its optimizer's outcome).

### 3.4 A fourth, independent leg: truncation

The short legs run on the series cut to 561 observations; the full legs run on
all 5,404. For rows with `target_index <= 560` the two agree **bit for bit**, for all
thirteen configs.
Deleting 4,843 future bars outright changes nothing about the forecasts for
earlier targets — the same claim as future corruption, reached by a different
mechanism.

## 4. Budget

Roughly ten minutes was the budget. Thirteen configs, five runs:

| leg | series length | cells | wall clock |
|---|---:|---:|---:|
| determinism A (clean) | 561 | 13 | 16.8 s |
| determinism B (clean, separate store) | 561 | 13 | 5.5 s |
| past corruption | 561 | 13 | 5.3 s |
| clean, full length | 5,404 | 13 | 425.6 s |
| future corruption, full length | 5,404 | 13 | 424.0 s |
| | | | **877 s = 14.6 min** |

The three cheap legs run on the series cut to 561 observations, which is all
the compared rows need — 61 origins instead of 4,904 — and §3.4 shows the cut
changes nothing about them. The two full-length legs are what make the future
corruption 4,843 bars rather than a token few, and they are 97 % of the cost.

The first pass covered only the five required configs and took 9.0 minutes; the
extension to all thirteen cost 5.6 more.

## 5. What this does and does not establish

**Does.** For every one of the 13 configs, on SPY, at the study's own protocol:
no information from raw bars after 2007-05-21 reaches any forecast for a target
at or before it, through any path the production code takes — the driver's
bridge, the panel's target construction, the compaction policy, the splitter,
the refit schedule, `update`, a TSFM's context window, PatchTST's per-origin
training loop, LightGBM's feature buffer, or AutoARIMA's model selection. And
the test that says so is demonstrably able to fail.

**Does not.** It is one asset and one cutoff. It cannot see a leak that is
invariant to the input (a hard-coded constant fitted offline), it says nothing
about the other ten assets except by the argument that they run identical code
on identically-shaped inputs, and it is not a statement about any forecast's
quality. `.claude/skills/leakage-check` items 8 (cross-asset calendar joins) and
10 (survivorship) are outside what a single-asset canary can reach; P3_GRID §6
addresses the first by observing that no cross-asset join exists — the eleven
calendars never meet.
